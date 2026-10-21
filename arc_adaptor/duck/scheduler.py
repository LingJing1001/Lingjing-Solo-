# -*- coding: utf-8 -*-
"""优先级调度器：多局轮转，共用一个动作预算。

为什么需要它
------------
单局串行时不存在"预算怎么分"的问题：一局跑到死为止。可 `deferred.py` 让一局
能停在 `yield` 上之后，局面变了 —— 同一个进程里可以同时挂着好几个局，每局都
处在"我刚交出一个动作，正在等结果"的状态。这时真正要回答的问题变成了：

    下一步该把预算花在哪一局上？

冠军的 27.89 分主要不是模型更聪明，而是这个问题的答案更好：死局不再独自吃光
全局预算，有产出的局能继续吃。这是本文件唯一要做的事。

判据：边际产出，不是累计产出
----------------------------
用累计过关数排序会让先跑起来的局永远领先（先手优势），所以优先级用

    rate = (过关数 + 0.5) / (已花步数 + 1)

0.5 和 1 是平滑项，作用只有一个：让没跑过的局（0/0）以 0.5 起步，比任何已经
花掉几十步却没过关的局都高。于是自然形成"先每条 lane 各看一眼，然后谁还有产出
谁继续"。这不是调参，是让除法在 0/0 处有定义且方向正确。

三条可复现性约束
----------------
1. 配额按**步数**切分，不按墙钟。用时间切分的话，同一份代码在慢机器上会得到
   不同的动作序列，横评就不可比了。`seconds` 只写进报告，从不参与决策。
2. 没有任何随机。pick 的 tie-break 固定为 lane 下标升序。
3. 每个决定都进 `trace`，可以落盘、可以回放。环境状态本身不可序列化，所以
   "复现"的含义是：同一批 game_id + 同一份 trace = 同一个动作序列。

lane 数量上限是调用方的事
------------------------
本文件不限制并发 lane 数，因为上限取决于引擎的显存/文件句柄，不是这里的逻辑。
`scripts/duck_multi.py` 默认按 CPU 核数的一半封顶（本地不超过 6）。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# 平滑项。见模块头：唯一作用是让 0/0 有定义，且新局天然优先。
PRIOR_GAIN = 0.5
PRIOR_STEPS = 1.0


@dataclass
class Lane:
    """一局游戏 + 它的 duck driver + 它当前分到的预算。"""

    index: int
    game_id: str
    env: Any
    driver: Any
    quota: int = 0
    steps: int = 0
    gained: int = 0
    level: int = 0
    deaths: int = 0
    wins: int = 0
    finished: bool = False
    reason: str = ""
    closed: bool = False
    history: list = field(default_factory=list)

    @property
    def rate(self) -> float:
        return (self.gained + PRIOR_GAIN) / (self.steps + PRIOR_STEPS)

    def priority(self) -> float:
        return -1.0 if self.finished else self.rate

    def exhausted(self) -> bool:
        return bool(self.quota) and self.steps >= self.quota

    def stats(self) -> dict:
        out = {
            "lane": self.index,
            "game": self.game_id,
            "steps": self.steps,
            "quota": self.quota,
            "levels": self.gained,
            "level_now": self.level,
            "deaths": self.deaths,
            "wins": self.wins,
            "rate": round(self.rate, 4),
            "finished": self.finished,
            "reason": self.reason,
        }
        driver_stats = getattr(self.driver, "stats", None)
        if callable(driver_stats):
            try:
                d = driver_stats()
                out["turns"] = d.get("turns")
                out["tokens"] = d.get("tokens")
                out["refusals"] = d.get("refusals")
            except Exception:  # noqa: BLE001
                pass
        return out


class PriorityScheduler:
    """轮转多个 lane，按边际产出分配动作预算。"""

    def __init__(self, lanes: list[Lane], *, total_steps: int,
                 log=print, trace_path: str | None = None):
        if not lanes:
            raise ValueError("scheduler needs at least one lane")
        self.lanes = list(lanes)
        for i, lane in enumerate(self.lanes):
            lane.index = i
        self.total_steps = max(1, int(total_steps))
        self.steps_total = 0
        self.rounds = 0
        self.ticks = 0
        self.trace: list[dict] = []
        self.log = log
        self.trace_path = trace_path
        self.started = time.time()
        self._assign_first_round()
        self._wire_reset_hooks()

    # ------------------------------------------------------------------ 初始化

    def _assign_first_round(self) -> None:
        """第一轮只分一半预算，人均一份。

        为什么不是全分：如果首轮就把预算均分干净，`_rebalance` 永远不会被调用，
        "按边际产出再分配"就成了一句空话 —— 每局都只拿到平均份额，调度器和
        简单的 round-robin 没有区别。留一半做竞争池，产出高的局才有机会多拿。

        这一半也不能太小：太小则每局连一次"试探性动作"都做不完，产出率的分母
        还没稳定就进入竞争。一半是这两者之间唯一好解释的取值。
        """
        share = max(1, self.total_steps // (2 * len(self.lanes)))
        for lane in self.lanes:
            lane.quota = min(share, self.total_steps)

    def _wire_reset_hooks(self) -> None:
        """让 driver 在"这一局结束"时向引擎要新棋盘，而不是继续看旧棋盘。

        没有这个钩子，`DuckDriver` 会在 game_over 之后沿用 `sandbox.current`，
        也就是那具"尸体"的盘面 —— 模型会对着一个已经不存在的局面推理到本局结束。
        """
        for lane in self.lanes:
            driver = lane.driver
            if driver is None:
                continue

            def hook(lane=lane):
                # `ArcEnv.reset()` gives a StepResult; the sandbox wants the raw
                # engine frame (`FrameDataRaw`), whose `.frame` is the layer stack.
                # Handing it the wrapper makes `grid_of` try to index the wrapper
                # itself and fail with "'FrameDataRaw' object is not subscriptable".
                fresh = lane.env.reset()
                lane.level = int(getattr(fresh, "level", lane.level))
                return getattr(fresh, "frame", fresh)

            try:
                driver.reset_hook = hook
            except Exception as exc:  # noqa: BLE001 - 老版本 driver 没有这个属性
                self.log(f"[sched] lane {lane.index} reset_hook 不可用: "
                         f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ 选择

    def alive(self) -> list[Lane]:
        return [lane for lane in self.lanes if not lane.finished]

    def pick(self) -> Lane | None:
        """返回下一个该跑的 lane；没有可跑的返回 None。

        排序键是 `(-priority, index)`：优先级降序，同分时下标小的先跑。这条
        tie-break 是硬性的，不然轮转会依赖 list 顺序的偶然性。
        """
        for _ in range(len(self.lanes) + 2):
            ready = [lane for lane in self.alive() if not lane.exhausted()]
            if not ready:
                if not self._rebalance():
                    return None
                continue
            ready.sort(key=lambda lane: (-lane.priority(), lane.index))
            return ready[0]
        return None

    def _rebalance(self) -> bool:
        """所有人都把配额花完了，但全局预算还有剩：按当前产出再切一轮。"""
        left = self.total_steps - self.steps_total
        alive = self.alive()
        if left <= 0 or not alive:
            return False
        weights = [lane.rate for lane in alive]
        total_w = sum(weights) or 1.0
        for lane, weight in zip(alive, weights):
            # 每局至少 1 步，否则低产出的局永远轮不到，也就永远无法反证
            # "它确实不行"——那等于把调度器变成一次性的初筛。
            add = max(1, int(round(left * weight / total_w)))
            lane.quota = lane.steps + add
        self.rounds += 1
        self.log(f"[sched] round {self.rounds}: 重新分配 {left} 步给 "
                 f"{len(alive)} 局 ({', '.join(f'{l.game_id}+{l.quota - l.steps}' for l in alive)})")
        return True

    # ------------------------------------------------------------------ 一步

    def tick(self, lane: Lane) -> dict | None:
        """让 lane 走一步：poll -> env.step -> feed。"""
        if lane.finished or self.steps_total >= self.total_steps:
            return None

        try:
            nxt = lane.driver.poll()
        except Exception as exc:  # noqa: BLE001
            self._retire(lane, f"poll raised {type(exc).__name__}: {exc}")
            return None
        if nxt is None:
            self._retire(lane, "driver finished")
            return None

        name, spec = nxt
        try:
            result = lane.env.step(name, spec)
        except Exception as exc:  # noqa: BLE001
            self._retire(lane, f"env.step raised {type(exc).__name__}: {exc}")
            return None

        lane.steps += 1
        self.steps_total += 1
        self.ticks += 1

        flags = dict(getattr(result, "flags", None) or {})
        before = lane.level
        lane.level = int(getattr(result, "level", lane.level))
        if lane.level > before:
            lane.gained += lane.level - before
            self.log(f"[sched] {lane.game_id} 过关 -> L{lane.level} "
                     f"(步数 {lane.steps}, 产出率 {lane.rate:.3f})")
        if flags.get("game_over"):
            lane.deaths += 1
        if flags.get("run_complete"):
            lane.wins += 1
            lane.finished = True
            lane.reason = "win"

        try:
            lane.driver.feed(result)
        except Exception as exc:  # noqa: BLE001 - 喂不进去这条 lane 就废了
            self._retire(lane, f"feed raised {type(exc).__name__}: {exc}")
            return None

        record = {
            "i": len(self.trace),
            "lane": lane.index,
            "game": lane.game_id,
            "action": name,
            "spec": spec,
            "level": lane.level,
            "gained": lane.gained,
            "flags": {k: bool(v) for k, v in flags.items()},
        }
        self.trace.append(record)
        lane.history.append(name)
        return record

    def _retire(self, lane: Lane, reason: str) -> None:
        lane.finished = True
        lane.reason = reason
        self.log(f"[sched] lane {lane.index} ({lane.game_id}) 退出: {reason}")

    # ------------------------------------------------------------------ 主循环

    def run(self, *, max_ticks: int | None = None) -> dict:
        while self.steps_total < self.total_steps:
            if max_ticks is not None and self.ticks >= max_ticks:
                break
            lane = self.pick()
            if lane is None:
                break
            self.tick(lane)
        self.close()
        report = self.report()
        if self.trace_path:
            self.dump(self.trace_path)
        return report

    def report(self) -> dict:
        return {
            "steps": self.steps_total,
            "budget": self.total_steps,
            "rounds": self.rounds,
            "ticks": self.ticks,
            "levels_total": sum(lane.gained for lane in self.lanes),
            "seconds": round(time.time() - self.started, 1),
            "lanes": [lane.stats() for lane in self.lanes],
        }

    def dump(self, path: str) -> str:
        """把决策序列写盘。这是本调度器唯一可复现的凭据。"""
        payload = {
            "budget": self.total_steps,
            "steps": self.steps_total,
            "rounds": self.rounds,
            "report": self.report(),
            "trace": self.trace,
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                          encoding="utf-8")
        return str(target)

    def close(self) -> None:
        for lane in self.lanes:
            if lane.closed:
                continue
            lane.closed = True
            closer = getattr(lane.env, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:  # noqa: BLE001 - best effort
                    pass


# ---------------------------------------------------------------------- 组装


def build_lane(index: int, game_id: str, chat, *, max_steps: int, max_turns: int,
               pin_policy: str = "level", level_memory: bool = True,
               level_transfer: bool = True, verbose: bool = False) -> Lane:
    """起一局：引擎 reset 出首帧，再用它构造 driver。

    `max_steps` 是**沙箱内**的步数上限，不是这局的预算 —— 预算在调度器手上。
    两者都给大一点没有坏处：调度器先耗尽配额，driver 不会先撞墙。
    """
    from .deferred import DuckDriver
    from .env import ArcEnv

    env = ArcEnv(game_id)
    fresh = env.reset()
    level = int(getattr(fresh, "level", 0))
    driver = DuckDriver(
        # Same layering rule as `reset_hook`: raw frame in, not the StepResult.
        chat, game_id=game_id, frame=getattr(fresh, "frame", fresh), level=level,
        max_steps=max_steps, max_turns=max_turns,
        level_transfer=level_transfer, level_memory=level_memory,
        pin_policy=pin_policy, verbose=verbose,
    )
    return Lane(index=index, game_id=game_id, env=env, driver=driver, level=level)
