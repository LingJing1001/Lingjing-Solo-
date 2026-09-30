"""轨迹收割器（边界侧）：上真机引擎回放/探针，产出统一 StepRecord 数据集。

本脚本属于引擎边界（tests/test_abstract_action_boundary.py 的 BOUNDARY_DIRS 规则：
lingjing_solo 包内只有 harness/ 能 import 引擎），所以引擎回放器放 scripts/ 而不是包内。
纯逻辑（计划 ast 提取、格式归一、解析器）在 lingjing_solo/transfer/miner.py。

用法：
  python scripts/harvest_trajectories.py            # 全来源收割 → state/affordance_dataset.jsonl
  python scripts/harvest_trajectories.py --out ...  --no-walk --seed 0

两阶段设计（bp35 上实测的教训）：探针直接对 game perform_action 且 generic_restore
无法保证救回引擎内部动作循环（'Action took too many frames' 卡死），因此
  pass 1 replay_levels  纯路线采集（探针零接触）；
  pass 2 probe_level_starts 每关全新 env 定位关首再探针，异常即弃该游戏。
随机游走不带探针，连续执行失败换新 env。
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from lingjing_solo.planning.search.generic_shadow import generic_restore, generic_snapshot
from lingjing_solo.transfer.trajectory import (
    StepRecord,
    background_color,
    classify_effect_kind,
    classify_outcome,
    click_context,
    connected_components,
    grid_digest,
    grid_of,
    keyboard_context,
    outcome_distribution,
    screen_delta,
    write_jsonl,
)
from lingjing_solo.transfer.miner import (
    DEFAULT_OUT,
    ENV_DIR,
    PLAN_VAR_FORMAT,
    extract_agent_plans,
    counter_free_digest,
    gid_to_full_id,
    load_route_plans,
    load_solution_plans,
    plan_steps,
)


# --------------------------------------------------------------- 引擎侧收割


class TrajectoryMiner:
    """在真机引擎上回放各来源动作序列并逐步捕获引擎真值 StepRecord。"""

    def __init__(self, seed: int = 0, sig_enabled: bool = True) -> None:
        from arc_agi import Arcade, OperationMode

        import logging

        self._log = logging.getLogger("trajectory_miner")
        self._log.setLevel(logging.WARNING)
        self.arc = Arcade(environments_dir=str(ENV_DIR), logger=self._log,
                          operation_mode=OperationMode.OFFLINE)
        self.rng = random.Random(seed)
        self.seed = seed
        self.sig_enabled = sig_enabled
        self.records: List[StepRecord] = []
        self._seen: set = set()
        self._env: Any = None
        self._game: Any = None

    # -- 基础设施 ----------------------------------------------------------

    def _make(self, full_id: str) -> Any:
        self._env = self.arc.make(full_id, seed=self.seed, save_recording=False)
        self._game = getattr(self._env, "_game", None)
        return self._env

    def _snap_state(self) -> Tuple[str, str]:
        """(hidden 摘要, 去计数器状态摘要)。"""
        hidden = ""
        sig = ""
        if self._game is not None:
            try:
                hs = self._game._get_hidden_state()
                if hs is not None:
                    hidden = grid_digest(np.asarray(hs).astype(np.int64) % 251)
            except Exception:
                pass
            if self.sig_enabled:
                try:
                    sig = counter_free_digest(generic_snapshot(self._game))
                except Exception:
                    pass
        return hidden, sig

    def _emit(self, rec: StepRecord) -> bool:
        key = (rec.is_probe, rec.game_id, rec.level_idx, rec.grid_hash_before,
               rec.action.get("id"), rec.action.get("x"), rec.action.get("y"))
        if key in self._seen:
            return False
        self._seen.add(key)
        self.records.append(rec)
        return True

    def _step_spec(self, spec: Dict[str, Optional[int]], base: Dict[str, Any],
                   step_idx: int, source: str, reliability: float,
                   is_probe: bool = False, base_sig: str = "",
                   capture: bool = True) -> bool:
        """执行一个动作 spec。返回是否真的执行了（去重不影响本返回值）。
        capture=False 用于探针定位（不产记录、不算上下文）。
        探针直接对 game perform（不走 wrapper），可能把引擎动作循环卡死且
        restore 救不回来——所以探针永远放在采集之后、每关新 env，见 probe_level_starts。"""
        env, game = self._env, self._game
        frame_before = self._frame
        grid_before = grid_of(frame_before.frame)
        hidden_b, sig_b = self._snap_state()
        levels_b = int(self._frame.levels_completed)
        state_b = self._frame.state.name
        aid = int(spec["aid"])
        x, y = spec.get("x"), spec.get("y")
        levels_b = int(frame_before.levels_completed)
        state_b = frame_before.state.name

        from arcengine import ActionInput, GameAction

        action_input = ActionInput(id=getattr(GameAction, f"ACTION{aid}"),
                                   data={"x": x, "y": y} if aid == 6 and x is not None else {})
        try:
            if is_probe:
                frame = game.perform_action(action_input, raw=True)
            else:
                frame = env.step(getattr(GameAction, f"ACTION{aid}"),
                                 data={"x": x, "y": y} if aid == 6 and x is not None else None)
                self._frame = frame
        except Exception:
            return False
        if frame is None:
            return False
        if not capture:
            if not is_probe:
                self._frame = frame
            return True
        grid_after = grid_of(frame.frame)
        hidden_a, sig_a = self._snap_state()
        levels_a = int(frame.levels_completed)
        state_a = frame.state.name

        delta = screen_delta(grid_before, grid_after)
        delta_n = len(delta)
        hidden_changed = bool(hidden_b) and hidden_b != hidden_a
        sig_changed = bool(sig_b) and sig_b != sig_a
        outcome = classify_outcome(
            levels_before=levels_b, levels_after=levels_a, state_after=state_a,
            delta_count=delta_n, hidden_changed=hidden_changed, sig_changed=sig_changed,
        )
        kind = classify_effect_kind(grid_before, grid_after)
        comps = connected_components(grid_before)
        if aid == 6 and x is not None:
            ctx = click_context(grid_before, int(x), int(y), comps)
        else:
            ctx = keyboard_context(grid_before)
        rec = StepRecord(
            game_id=base["game_id"], gid=base["gid"], level_idx=levels_b,
            step_idx=step_idx, action={"id": aid, "x": x, "y": y},
            source=source, reliability=reliability, outcome=outcome, effect_kind=kind,
            is_probe=is_probe, base_sig=base_sig,
            grid_hash_before=grid_digest(grid_before), grid_hash_after=grid_digest(grid_after),
            delta_count=delta_n, screen_delta=delta if outcome == "screen" else [],
            hidden_before=hidden_b, hidden_after=hidden_a,
            state_sig_before=sig_b, state_sig_after=sig_a,
            levels_before=levels_b, levels_after=levels_a, context=ctx,
        )
        return self._emit(rec)

    def _probe_at(self, base: Dict[str, Any], max_cands: int = 6,
                  add_random: int = 2, kbd_cands: int = 0) -> int:
        """反事实探针：快照 → 试候选动作 → 恢复。返回写入的探针记录数。"""
        if self._game is None:
            return 0
        from arcengine import GameAction

        grid = grid_of(self._frame.frame)
        base_sig = grid_digest(grid)
        avail = [a for a in (self._frame.available_actions or []) if int(a) != 7]
        snap = generic_snapshot(self._game)
        comps = connected_components(grid)
        cands = [(int(c["cx"]), int(c["cy"]))
                 for c in sorted(comps, key=lambda c: -int(c["area"]))[:max_cands]]
        h, w = grid.shape
        for _ in range(add_random):
            cands.append((self.rng.randrange(w), self.rng.randrange(h)))
        n_before = len(self.records)
        try:
            for x, y in cands:
                generic_restore(self._game, snap)
                self._step_spec({"aid": 6, "x": x, "y": y}, base,
                                step_idx=-1, source="probe", reliability=0.5,
                                is_probe=True, base_sig=base_sig)
            for aid in avail[:kbd_cands]:
                generic_restore(self._game, snap)
                self._step_spec({"aid": int(aid), "x": None, "y": None}, base,
                                step_idx=-1, source="probe", reliability=0.5,
                                is_probe=True, base_sig=base_sig)
        finally:
            generic_restore(self._game, snap)
        return len(self.records) - n_before

    def _terminal(self) -> bool:
        return self._frame is not None and self._frame.state.name in ("WIN", "GAME_OVER")

    # -- 来源 1/2/3：计划、routes、solution -------------------------------

    def _locate(self, full_id: str, gid: str, per_level: Dict[int, List],
                level: int, source: str, reliability: float) -> bool:
        """执行 levels < level 的计划把引擎推进到 level 关首（不采集）。"""
        for lv in sorted(per_level):
            if lv >= level:
                break
            base = {"game_id": full_id, "gid": gid, "level_idx": lv}
            for spec in per_level[lv]:
                if self._terminal() or int(self._frame.levels_completed) > lv:
                    break
                if not self._step_spec(spec, base, step_idx=-1, source=source,
                                       reliability=reliability, capture=False):
                    return False
        return not self._terminal() and int(self._frame.levels_completed) == level

    def replay_levels(self, full_id: str, gid: str, source: str, reliability: float,
                      per_level: Dict[int, List[Dict[str, Optional[int]]]],
                      step_cap: int = 2000) -> Dict[str, int]:
        """pass 1：逐关回放采集。纯路线、无探针——探针可能把引擎动作循环卡死，
        必须放在采集全部完成之后（见 probe_level_starts）。"""
        stats = {"steps": 0, "levels_ok": 0}
        env = self._make(full_id)
        if self._game is None:
            return stats
        for level in sorted(per_level):
            self._frame = env.reset()
            if self._frame is None or int(self._frame.levels_completed) > level:
                continue
            if not self._locate(full_id, gid, per_level, level, source, reliability):
                continue
            base = {"game_id": full_id, "gid": gid, "level_idx": level}
            for i, spec in enumerate(per_level[level]):
                if self._terminal() or stats["steps"] > step_cap:
                    break
                if self._step_spec(spec, base, step_idx=i, source=source,
                                   reliability=reliability):
                    stats["steps"] += 1
            if int(self._frame.levels_completed) > level:
                stats["levels_ok"] += 1
        return stats

    def probe_level_starts(self, full_id: str, gid: str,
                           per_level: Dict[int, List], max_cands: int = 6) -> int:
        """pass 2：每关一个全新 env，定位到关首后做反事实探针。

        探针直接对 game perform 且 restore 无法保证救回引擎内部动作循环
        （bp35 上实测卡死：'Action took too many frames'），所以：采集已在
        pass 1 落袋；这里每关换新 env；任何异常立即放弃该游戏剩余探针。"""
        n = 0
        for level in sorted(per_level):
            try:
                self._make(full_id)
                if self._game is None:
                    break
                self._frame = self._env.reset()
                if self._frame is None or not self._locate(
                        full_id, gid, per_level, level, "probe", 0.0):
                    continue
                n += self._probe_at({"game_id": full_id, "gid": gid, "level_idx": level},
                                    max_cands=max_cands, add_random=2, kbd_cands=2)
            except Exception:
                break
        return n

    # -- 来源 4：无计划游戏的种子随机游走 ---------------------------------

    def walk_game(self, full_id: str, gid: str, episodes: int = 2, steps: int = 60) -> Dict[str, int]:
        """无计划游戏的种子随机游走（组件中心点击 70% / 键盘 30%）。

        不带探针（会卡死引擎）；连续执行失败就换新 env——游走数据是 LOGO
        的"未见游戏"来源，被坏引擎状态污染还不如没有。
        """
        stats = {"steps": 0, "resets": 0}
        env = self._make(full_id)
        if self._game is None:
            return stats
        for ep in range(episodes):
            self._frame = env.reset()
            if self._frame is None:
                break
            stats["resets"] += 1
            fails = 0
            for t in range(steps):
                if self._terminal():
                    break
                frame = self._frame
                grid = grid_of(frame.frame)
                base = {"game_id": full_id, "gid": gid, "level_idx": int(frame.levels_completed)}
                avail = [int(a) for a in (frame.available_actions or [])
                         if int(a) not in (6, 7)]  # 6 要坐标、7 是 RESET，都不进键盘分支
                comps = connected_components(grid)
                spec: Dict[str, Optional[int]]
                if comps and self.rng.random() < 0.7:
                    c = sorted(comps, key=lambda c: -int(c["area"]))[
                        self.rng.randrange(min(4, len(comps)))]
                    spec = {"aid": 6, "x": int(c["cx"]), "y": int(c["cy"])}
                elif avail:
                    spec = {"aid": self.rng.choice(avail), "x": None, "y": None}
                else:
                    spec = {"aid": 6,
                            "x": self.rng.randrange(int(grid.shape[1])),
                            "y": self.rng.randrange(int(grid.shape[0]))}
                if self._step_spec(spec, base, step_idx=t, source="walk", reliability=0.6):
                    stats["steps"] += 1
                    fails = 0
                else:
                    fails += 1
                    if fails >= 5:
                        self._frame = env.reset()
                        stats["resets"] += 1
                        fails = 0
        return stats


def harvest_all(out_path: Path = DEFAULT_OUT, do_walk: bool = True,
                seed: int = 0) -> Dict[str, Any]:
    """收割全部来源 → 写统一 JSONL。返回统计摘要。"""
    t0 = time.time()
    miner = TrajectoryMiner(seed=seed)
    summary: Dict[str, Any] = {"sources": {}, "per_game": {}}

    def bump(source: str, **kw):
        s = summary["sources"].setdefault(source, {"steps": 0, "probes": 0})
        s["steps"] += kw.get("steps", 0)
        s["probes"] += kw.get("probes", 0)

    # 1) agent 计划（ast 提取）：pass 1 采集 + pass 2 关首探针
    plans = extract_agent_plans()
    for var, (fmt, gid) in PLAN_VAR_FORMAT.items():
        if var not in plans:
            continue
        full = gid_to_full_id(gid)
        if full is None:
            continue
        per_level = {lv: plan_steps(fmt, seq) for lv, seq in plans[var].items()}
        st = miner.replay_levels(full, gid, "agent-plan", 1.0, per_level)
        st["probes"] = miner.probe_level_starts(full, gid, per_level)
        bump("agent-plan", **st)
        summary["per_game"].setdefault(gid, {}).update(steps=st["steps"], probes=st["probes"])

    # 2) state 解（tr87 白赚；cd82/tn36 与 agent-plan 去重后基本只剩补充）
    for full, gid, per_level in load_solution_plans():
        st = miner.replay_levels(full, gid, "solution", 1.0, per_level)
        st["probes"] = miner.probe_level_starts(full, gid, per_level)
        bump("solution", **st)

    # 3) routes（与上面大量重叠，去重后只剩少数补充步；不重复探针）
    for full, gid, per_level in load_route_plans():
        st = miner.replay_levels(full, gid, "route", 1.0, per_level)
        bump("route", **st)

    # 4) 无计划游戏随机游走（LOGO 的"未见游戏"来源）
    if do_walk:
        planned = {g for g, _, _ in load_solution_plans()} | {
            gid_to_full_id(fmt_gid) for _, (_, fmt_gid) in PLAN_VAR_FORMAT.items()
            if gid_to_full_id(fmt_gid)
        }
        for gdir in sorted(ENV_DIR.iterdir()):
            gid = gdir.name
            full = gid_to_full_id(gid)
            if full is None or full in planned:
                continue
            try:
                st = miner.walk_game(full, gid)
            except Exception as exc:  # noqa: BLE001 — 游走是补充来源，单局崩了跳过
                summary.setdefault("walk_errors", {})[gid] = f"{type(exc).__name__}: {exc}"
                continue
            bump("walk", **st)
            summary["per_game"].setdefault(gid, {}).update(
                steps=st["steps"], probes=st.get("probes", 0))

    n = write_jsonl(miner.records, out_path)
    summary["total_records"] = n
    summary["elapsed_s"] = round(time.time() - t0, 1)
    from lingjing_solo.transfer.trajectory import outcome_distribution  # noqa: F811 顶部已导入，保持引用一致

    summary["outcome_dist"] = outcome_distribution(miner.records)
    return summary


if __name__ == "__main__":
    import argparse
    import pprint

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--no-walk", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    pprint.pprint(harvest_all(out_path=args.out, do_walk=not args.no_walk,
                              seed=args.seed))
