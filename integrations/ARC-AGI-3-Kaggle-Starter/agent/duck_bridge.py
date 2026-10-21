# -*- coding: utf-8 -*-
"""duck 接入路由的那道缝（只做接线，不做决定）。

为什么要单独一个文件
--------------------
`_choose_action_inner` 里已经有三道逐级降级的尝试：本地专用求解器 → shadow
→ IDDFS。三道路都空手而归时，说明这块板子躲开了我们能写出来的所有搜索，
剩下的唯一信息源就是"让模型看一眼图"。这就是 duck。

本文件是**缝**，不是大脑：判断 duck 什么时候可以接手、每一 tick 把真实帧喂给它、
再把它的一个动作交回给 harness。duck 自己的循环（`arc_adaptor/duck/deferred.py`）
一行不动。

为什么默认是 off 而不是 on
--------------------------
全量 25 局现在是 44.57 分，这个数字是在 duck 完全缺席的情况下得到的。一条
"悄悄把未知局改走模型"的线若默认打开，等于未经允许改动了那个数字 —— 更糟的
是，本机 8001 端口上跑着的任何 mock 网关都会让它"看起来能用"。所以默认关：

  LINGJING_DUCK=off（默认）   永不构造 duck，行为与今天逐字节一致
  LINGJING_DUCK=auto          端点能在 LINGJING_DUCK_PROBE 秒内建立连接才接线
  LINGJING_DUCK=on            强制接线（探测失败也接，用于本机已确认服务在跑）

auto 是"起完模型就不用管"的那一档；想跑对照实验就用它，想保住已知分数就用
默认的 off。两者之间没有第三态。

预算为什么卡在 chat 上
----------------------
duck 的一次 LLM 调用是同步阻塞的，跑在 `choose_action` 里。如果只在 tick 之间
检查墙钟，一次 120 秒的调用就能把整局拖爆。所以预算包在 chat 外层：
`GuardedChat` 在每次调用**之前**检查，超预算就 raise，duck 的 `_inner` 里本来
就有 `except Exception: return False`，于是生成器自己收尾，`poll()` 返回 None，
路由无感地退回原流程。这是唯一不需要改 duck 内部就能硬停的地方。
"""
from __future__ import annotations

import os
import socket
import sys
import time
from typing import Any, Optional
from urllib.parse import urlparse


# --------------------------------------------------------------------- imports


def _repo_root() -> Optional[str]:
    """向上找含 `arc_adaptor/duck/deferred.py` 的祖先目录。

    本地仓库是 `<root>/integrations/ARC-AGI-3-Kaggle-Starter/agent/`，三层。
    打包进 Kaggle notebook 时这个文件不随行，于是返回 None —— 这必须是一条
    安静的死路，而不是 ImportError。
    """
    cur = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        cur = os.path.dirname(cur)
        if not cur or cur in (os.sep, ""):
            break
        if os.path.isfile(os.path.join(cur, "arc_adaptor", "duck", "deferred.py")):
            return cur
    return None


def _load_backend():
    """(Chat, Endpoint, DuckDriver) 或 None。"""
    root = _repo_root()
    if root and root not in sys.path:
        sys.path.insert(0, root)
    try:
        from arc_adaptor.duck.chat import Chat, Endpoint
        from arc_adaptor.duck.deferred import DuckDriver

        return Chat, Endpoint, DuckDriver
    except Exception:  # noqa: BLE001 - 缺包/版本不对都只是"这次不接"
        return None


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return default if raw is None or not str(raw).strip() else str(raw).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env_str(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env_str(name, str(default)))
    except ValueError:
        return default


def _env_on(name: str, default: bool = True) -> bool:
    return _env_str(name, "on" if default else "off").lower() not in {
        "0", "off", "no", "false", "",
    }


def _probe(url: str, timeout: float) -> tuple[bool, str]:
    """只建 TCP 连接：不花 token，也不赌哪个端点实现了 /models。"""
    try:
        parsed = urlparse(url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except Exception as exc:  # noqa: BLE001
        return False, f"bad url {url!r}: {exc}"
    try:
        with socket.create_connection((host, port), timeout=max(0.05, timeout)):
            return True, f"{host}:{port} reachable"
    except OSError as exc:
        return False, f"{host}:{port} unreachable ({type(exc).__name__})"


# ----------------------------------------------------------------------- chat


class GuardedChat:
    """给 chat 加一道"到点就别再问了"的闸。

    `tokens` 与 `calls` 直接透传，因为 `DuckDriver` 会读它们决定何时压缩。
    没有 `tokens` 属性的模型（自研客户端）会走 getattr 的默认分支。
    """

    def __init__(self, chat: Any, *, deadline: float):
        self._chat = chat
        self._deadline = deadline

    def __call__(self, messages: list, tools: list | None = None) -> dict:
        left = self._deadline - time.time()
        if left <= 0:
            raise TimeoutError("duck budget spent")
        return self._chat(messages, tools)

    def tokens(self) -> int:
        fn = getattr(self._chat, "tokens", None)
        return int(fn()) if callable(fn) else 0

    @property
    def calls(self) -> int:
        return int(getattr(self._chat, "calls", 0))


# --------------------------------------------------------------------- bridge


class DuckBridge:
    """一局一个。`tick()` 每步调一次，返回 (engine_action_name, spec) 或 None。"""

    def __init__(self, game_id: str, *, chat: Any = None, log=print):
        self.game_id = str(game_id)
        self.log = log
        self.mode = _env_str("LINGJING_DUCK", "off").lower()
        self.enabled = False
        self.reason = ""
        self.actions = 0
        self.turns = 0
        self.calls = 0
        self.started_at = 0.0

        self._chat = chat
        self._Chat = self._Endpoint = self._Driver = None
        self._driver = None
        self._need_feed = False
        self._deadline = 0.0
        self._budget = _env_float("LINGJING_DUCK_BUDGET", 300.0)

        if self.mode in {"off", "0", "no", "false"}:
            self.reason = "LINGJING_DUCK=off"
            return

        backend = _load_backend()
        if backend is None:
            self.reason = "arc_adaptor.duck not importable"
            return
        self._Chat, self._Endpoint, self._Driver = backend

        if self._chat is None:
            endpoint = self._Endpoint()
            if self.mode not in {"on", "force", "1", "yes", "true"}:
                ok, detail = _probe(endpoint.url,
                                    _env_float("LINGJING_DUCK_PROBE", 0.6))
                if not ok:
                    self.reason = detail
                    return
                self.reason = detail
        self.enabled = True

    # ------------------------------------------------------------------ 启动

    def _sysprompt_ok(self) -> bool:
        """duck 的 system prompt 里含棋盘/动作词汇；缺 prompts 就没得谈。"""
        try:
            from arc_adaptor.duck import prompts  # noqa: F401
            return True
        except Exception:  # noqa: BLE001
            return False

    def _start(self, frame, levels: int) -> bool:
        if not self._sysprompt_ok():
            self.reason = "arc_adaptor.duck.prompts missing"
            return False
        try:
            base = self._chat if self._chat is not None else self._Chat(
                self._Endpoint(), think=_env_on("LINGJING_DUCK_THINK", False))
            self._deadline = time.time() + self._budget
            self._driver = self._Driver(
                GuardedChat(base, deadline=self._deadline),
                game_id=self.game_id,
                frame=frame,
                level=int(levels),
                max_steps=max(1, _env_int("LINGJING_DUCK_STEPS", 80)),
                max_turns=max(1, _env_int("LINGJING_DUCK_TURNS", 24)),
                level_transfer=_env_on("LINGJING_DUCK_TRANSFER", True),
                level_memory=_env_on("LINGJING_DUCK_MEMORY", True),
                # 局内 duck 通常几十步就结束，等不到压缩；"level" 才能让过关
                # 记忆立刻进 system prompt（"drain" 是上游的缓存安全档）。
                pin_policy=_env_str("LINGJING_DUCK_PIN", "level"),
                verbose=_env_on("LINGJING_DUCK_VERBOSE", False),
            )
            self.started_at = time.time()
            self.log(f"[duck-bridge] start gid={self.game_id} L={levels} "
                     f"budget={int(self._budget)}s steps<={self._driver.max_steps}")
            return True
        except Exception as exc:  # noqa: BLE001
            self.reason = f"start failed {type(exc).__name__}: {exc}"
            self.log(f"[duck-bridge] {self.reason}")
            return False

    # -------------------------------------------------------------------- tick

    def tick(self, frame, levels: int,
             request: Optional[int] = None) -> Optional[tuple[str, dict]]:
        """把这一帧喂进去，换一个动作出来。None 表示"这一步不归 duck"。

        握手是 poll → feed → poll（见 deferred.py 的说明）：`feed` 只暂存，
        下一次 `poll` 才把它交给生成器。
        """
        if not self.enabled:
            return None
        if self._driver is None:
            # R3 是在哪一关失败的，就只能接着那一关干；换关了这条请求作废。
            if request is None or int(request) != int(levels):
                return None
            if not self._start(frame, levels):
                self.enabled = False
                return None
        elif self._budget > 0 and time.time() >= self._deadline:
            self._abandon(f"budget {int(self._budget)}s spent")
            return None

        try:
            if self._need_feed:
                self._driver.feed(frame)
                self._need_feed = False
            nxt = self._driver.poll()
        except Exception as exc:  # noqa: BLE001 - 任何异常都只等于"退回原流程"
            self._abandon(f"tick raised {type(exc).__name__}: {exc}")
            return None

        if nxt is None:
            self._abandon("driver finished")
            return None

        self._need_feed = True
        self.actions += 1
        return nxt

    def _abandon(self, why: str) -> None:
        self.enabled = False
        self.reason = why
        stats = self.stats()
        self.log(f"[duck-bridge] stop gid={self.game_id} ({why}) {stats}")
        self._driver = None

    def stats(self) -> dict:
        driver = self._driver
        out = {
            "actions": self.actions,
            "seconds": round(time.time() - self.started_at, 1) if self.started_at else 0,
            "reason": self.reason,
        }
        if driver is not None:
            try:
                out.update(driver.stats())
            except Exception:  # noqa: BLE001
                pass
        return out
