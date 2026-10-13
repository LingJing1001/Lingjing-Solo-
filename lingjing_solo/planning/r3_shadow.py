"""R3 旁路探针（shadow）：跑真内核、记证据，绝不参与发动作。

为什么要这么一层
------------------
`R3Kernel` / `WorldView` / `registry` 落地之后一直只有接口 + 闸门 + 单测，
线上决策路径没人调它，于是《最小可运行闭环》§7.1.1 的 R3-A~R3-D 门槛在真实局上
永远是"未验证"。直接把它接成决策者会一次性动到打分路径；本模块先把风险拆开：
**同一条 tick 上同时记录"内核会选什么"和"统一路由会选什么"**，
先拿到 P1/P3/P4 到底会不会 fire 的证据，再谈生效。

三条硬规矩
----------
1. 默认关闭：`R3_SHADOW` 必须显式设为 `1`；游戏白名单 `R3_SHADOW_GAMES`
   （逗号分隔，默认只有 `dc22`）。Kaggle 线上不带这两个环境变量 → 一行都不执行。
2. 只读不写决策：`observe()` 恒返回 `None`，调用方不许拿它的 plan 当动作。
3. 任何异常都不许带崩打分：内部全捕获，累计 `MAX_ERRORS` 次后自检停用并打一行说明。

产物
----
- stdout：首 `PRINT_FIRST` tick 逐条打，之后每 `PRINT_EVERY` tick 打一条；
  每个剪枝契约**首次** fire 时单独打一行；旁路结论与统一路由结论不一致时打一行。
- JSONL：`<state_dir>/r3_shadow/<gid>_<runstamp>.jsonl`，每 tick 一条记录
  （`state_dir` 默认仓库 `state/`，可用 `R3_SHADOW_DIR` 覆盖）。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

from .backends import LearnedGraphBackend
from .r3_kernel import R3Kernel, SearchBudget
from .registry import (
    get_world_view,
    register_backend,
    register_world_view,
    resolve_planner,
)
from .world_view import action_edge_key

#: 白名单里可用的 profile：game_id 前缀 → WorldView 类名
_SUPPORTED: dict[str, str] = {"dc22": "DC22WorldView"}

PRINT_FIRST = 3
PRINT_EVERY = 25
MAX_ERRORS = 5

#: 旁路预算：墙钟压到 150ms，宁可在证据里承认"P4 超时"也不让记录器拖慢打分路径。
DEFAULT_BUDGET = SearchBudget(max_steps=400, max_nodes=2000, max_depth=40, time_ms=150)


def shadow_enabled() -> bool:
    return os.environ.get("R3_SHADOW", "0").strip() == "1"


def shadow_games() -> set[str]:
    raw = os.environ.get("R3_SHADOW_GAMES", "dc22")
    return {item.strip() for item in raw.split(",") if item.strip()}


def _state_dir() -> Path:
    """产物目录：`R3_SHADOW_DIR` 优先，其次仓库 `state/`，再退到 cwd。"""
    override = os.environ.get("R3_SHADOW_DIR", "").strip()
    if override:
        return Path(override)
    repo_state = Path(__file__).resolve().parents[2] / "state"
    return repo_state if repo_state.parent.exists() else Path.cwd() / "state"


def _action_name(action_input: Any) -> str:
    """`FrameData.action_input` 的官方回显 → abstract name。

    这是线上唯一能拿到"上一步发了什么"的正规来源（不读引擎私有对象）。
    """
    raw = getattr(action_input, "id", None)
    if raw is None:
        return ""
    name = getattr(raw, "name", None)
    if isinstance(name, str) and name.startswith("ACTION"):
        return name
    try:
        return f"ACTION{int(raw)}"
    except (TypeError, ValueError):
        return ""


def _action_edge_key(action_input: Any) -> str:
    """官方回显 → learned graph 的边键（点击动作带坐标）。

    `_action_name` 只能给出枚举名；`ACTION6/ACTION7` 的坐标在 `action_input.data`
    里，旧接线把它丢了 ⇒ 同状态不同坐标的点击互相覆盖。这里把坐标接回键上。
    """
    name = _action_name(action_input)
    data = getattr(action_input, "data", None)
    if not isinstance(data, dict):
        data = {}
    return action_edge_key(name, {"x": data.get("x"), "y": data.get("y")})


class R3ShadowProbe:
    """挂在决策路径旁边的记录器。

    由打分入口按 `R3_SHADOW` 建**一个进程级实例**、各 agent 实例共享；
    所有状态（内核、learned graph、上一帧 key、tick 计数、产物句柄）都按 gid 分表，
    所以多局跑测各自有各自的转移图和各自的 JSONL，互不污染。
    """

    def __init__(self, *, budget: SearchBudget | None = None) -> None:
        self.budget = budget or DEFAULT_BUDGET
        self._kernels: dict[str, R3Kernel] = {}
        self._prev_key: dict[str, str] = {}
        self._prev_level: dict[str, int] = {}
        self._ticks: dict[str, int] = {}
        self._runstamp = time.strftime("%Y%m%d_%H%M%S")
        self._contract_seen: set[str] = set()
        self._contract_hits: dict[str, int] = {}
        self._route_mismatches = 0
        self._errors = 0
        self._disabled = False
        self._files: dict[str, Any] = {}

    # ── 主入口 ─────────────────────────────────────────────

    def observe(
        self,
        frame: Any,
        legal_actions: Sequence[str],
        *,
        gid: str,
        levels: int,
    ) -> None:
        """记录这一 tick 的内核视角。**恒返回 None**，不参与决策。"""
        if self._disabled or gid not in shadow_games() or gid not in _SUPPORTED:
            return
        try:
            self._observe(list(legal_actions or []), frame, gid=gid, levels=levels)
        except Exception as exc:  # noqa: BLE001 — 旁路失败只能自我停用
            self._errors += 1
            if self._errors == 1 or self._errors == MAX_ERRORS:
                print(
                    f"[r3-shadow] error gid={gid} #{self._errors} "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
            if self._errors >= MAX_ERRORS:
                self._disabled = True
                print("[r3-shadow] disabled after repeated errors（打分路径不受影响）", flush=True)

    def _observe(self, legal: list[str], frame: Any, *, gid: str, levels: int) -> None:
        kernel = self._ensure_armed(gid)
        if kernel is None:
            return
        tick = self._ticks.get(gid, 0)
        self._ticks[gid] = tick + 1

        grids = kernel.wv.grids(frame)
        if not grids:
            return
        wv_key = str(kernel.wv.key(grids))

        # 学到的转移补边：(上一帧 key, 本帧官方回显的动作边键) → 本帧 key
        # 边键 = 动作名 + 点击坐标（`_action_edge_key`）；只用动作名会让同状态下
        # 不同坐标的点击塌成一条边、后写覆盖先写。
        # 换关/重开后状态空间不同，跨关不补边，免得把 P1 noop 判成假阳性。
        prev_key = self._prev_key.get(gid, "")
        prev_level = self._prev_level.get(gid, levels)
        action_input = getattr(frame, "action_input", None)
        echoed = _action_name(action_input)
        echoed_edge = _action_edge_key(action_input)
        edge_added = False
        if prev_key and echoed and levels == prev_level:
            kernel.backend.update(prev_key, echoed_edge, wv_key)
            edge_added = True
        self._prev_key[gid] = wv_key
        self._prev_level[gid] = levels

        plan = kernel.plan(frame, legal, wv_key, game_id=gid, level=levels)
        violations = [
            {
                "contract": getattr(getattr(v, "contract", None), "value", ""),
                "reason": getattr(v, "reason", ""),
                "action": getattr(v, "action", ""),
                "divergence": getattr(v, "divergence", ""),
            }
            for v in kernel.violations
        ]
        stats = dict(kernel.stats)
        for hit in violations:
            name = hit["contract"] or "unknown"
            self._contract_hits[name] = self._contract_hits.get(name, 0) + 1
            if name not in self._contract_seen:
                self._contract_seen.add(name)
                print(
                    f"[r3-shadow] gate-fired first {name} gid={gid} L={levels} "
                    f"action={hit['action']!r} reason={hit['reason']}",
                    flush=True,
                )

        # R3-D 交叉核对：registry 的统一路由必须给出同一份结论
        route_plan = resolve_planner(
            gid, frame, legal, wv_key, level=levels, budget=self.budget
        )
        route_same = _plan_agrees(plan, route_plan)
        if not route_same:
            self._route_mismatches += 1
            if self._route_mismatches == 1:
                print(
                    f"[r3-shadow] route-divergence gid={gid} L={levels} "
                    f"kernel={(plan or {}).get('validity')}, "
                    f"{(plan or {}).get('candidate_actions')!r} vs "
                    f"route={(route_plan or {}).get('validity')}, "
                    f"{(route_plan or {}).get('candidate_actions')!r}",
                    flush=True,
                )

        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "gid": gid,
            "level": levels,
            "tick": tick,
            "state_key": wv_key,
            "echoed_action": echoed,
            "edge_key": echoed_edge,
            "edge_added": edge_added,
            "graph_edges": getattr(kernel.backend, "edge_count", 0),
            "legal": legal,
            "plan": plan,
            "route_plan": route_plan,
            "route_agrees": route_same,
            "violations": violations,
            "stats": stats,
            "counters": _safe_counters(kernel.wv, grids),
        }
        self._write(gid, record)

        if tick < PRINT_FIRST or (tick - PRINT_FIRST) % PRINT_EVERY == 0:
            cand = (plan or {}).get("candidate_actions") or []
            print(
                f"[r3-shadow] gid={gid} L={levels} tick={tick} "
                f"validity={(plan or {}).get('validity')} "
                f"goal={(plan or {}).get('expected_goal')} "
                f"cands={cand[:6]} layer={(stats.get('layer_tried') or [''])[0]} "
                f"nodes={stats.get('nodes')} edges={record['graph_edges']} "
                f"viol={[h['contract'] for h in violations]}",
                flush=True,
            )

    # ── 注册与产物 ─────────────────────────────────────────

    def _ensure_armed(self, gid: str) -> R3Kernel | None:
        kernel = self._kernels.get(gid)
        if kernel is not None:
            return kernel
        if _SUPPORTED[gid] == "DC22WorldView":
            from .dc22_world_view import DC22WorldView

            register_world_view(gid, DC22WorldView())
        # 每局一张自己的 learned graph：状态键跨局不可比，共表会让 P1 判成假阳性
        graph = LearnedGraphBackend()
        register_backend(gid, graph)
        wv = get_world_view(gid)
        if wv is None:
            return None
        kernel = R3Kernel(wv, graph, budget=self.budget, planner_name="generic")
        self._kernels[gid] = kernel
        print(
            f"[r3-shadow] armed gid={gid} profile={_SUPPORTED[gid]} "
            f"budget=time{self.budget.time_ms}ms/nodes{self.budget.max_nodes} "
            f"（旁路：只记录，不改动作）",
            flush=True,
        )
        return kernel

    def _write(self, gid: str, record: dict[str, Any]) -> None:
        handle = self._files.get(gid)
        if handle is None:
            path = _state_dir() / "r3_shadow" / f"{gid}_{self._runstamp}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = open(path, "a", encoding="utf-8")  # noqa: SIM115 — 生命周期同 agent
            self._files[gid] = handle
            print(f"[r3-shadow] jsonl={path}", flush=True)
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        handle.flush()

    def summary(self) -> dict[str, Any]:
        """剪枝契约累计 fire 次数（跑测结束时打这一条）。"""
        return {
            "contract_hits": dict(self._contract_hits),
            "route_mismatches": self._route_mismatches,
            "graph_edges": {
                gid: getattr(kernel.backend, "edge_count", 0)
                for gid, kernel in self._kernels.items()
            },
            "errors": self._errors,
            "disabled": self._disabled,
        }

    def close(self) -> None:
        for handle in self._files.values():
            try:
                handle.close()
            except Exception:  # noqa: BLE001
                pass
        self._files.clear()


def _plan_agrees(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    if (a is None) != (b is None):
        return False
    if a is None or b is None:
        return True
    keys = ("validity", "expected_goal", "candidate_actions", "planner")
    return all(a.get(k) == b.get(k) for k in keys)


def _safe_counters(wv: Any, grids: Any) -> dict[str, int]:
    try:
        return dict(wv.counters(grids) or {})
    except Exception:  # noqa: BLE001
        return {}


__all__ = [
    "R3ShadowProbe",
    "shadow_enabled",
    "shadow_games",
    "DEFAULT_BUDGET",
]
