"""R3 剪枝实验：闭包搜索的分支扩展按 affordance 排序/剪枝是否省节点。

假设（2026-09-30 四策略跑分后的唯一活口）：affordance 排序器当独立策略 = 0/25，
但它可能 redeem 自己的位置是**搜索回路内部的候选排序/剪枝层**——R3 闭包搜索
在点击候选很多（每节点 ~14 分支）时按分数扩展，应比固定顺序更早命中关卡推进。

三臂（同一副骨架：快照/恢复、去计数器状态去重、深度/节点/时限预算全同）：
  control  组件中心按面积序全量(8) + 键盘升序            —— 现状等价
  ranked   同全量，但点击按 productive 分数降序           —— 只动顺序
  pruned   点击按分数剪到 top-4 + 键盘                    —— 顺序+剪枝

测试集：
  known   7 个有真值解的点击关（vc33 L0/L1、sb26/r11l/tn36/cd82/ft09 的 L0；
          计划长度 ≤14 在 max_depth 内，路径必然存在——找不到就是搜索的锅）
  all0    25 局的 L0（未知故事，大多数可能无 14 步内解）

指标：found / nodes（引擎 perform_action 次数）/ 时限内路径长。
判读：common-success 实例上 ranked/pruned 的 nodes 相对 control 的下降比。
"""
from __future__ import annotations

import heapq
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import logging

import numpy as np  # noqa: E402

from arcengine import ActionInput, GameAction  # noqa: E402

from lingjing_solo.planning.search.generic_shadow import (  # noqa: E402
    generic_restore, generic_snapshot,
)
from lingjing_solo.transfer.affordance import AffordanceRanker  # noqa: E402
from lingjing_solo.transfer.miner import (  # noqa: E402
    PLAN_VAR_FORMAT, counter_free_digest, extract_agent_plans,
    gid_to_full_id, plan_steps,
)
from lingjing_solo.transfer.trajectory import (  # noqa: E402
    click_context, connected_components, grid_of, read_jsonl,
)

sys.path.insert(0, str(ROOT / "scripts"))
from harvest_trajectories import TrajectoryMiner  # noqa: E402

DATASET = ROOT / "state" / "affordance_dataset.jsonl"
OUT_JSON = ROOT / "state" / "r3_prune_bench.json"
ARMS = ("control", "ranked", "pruned")
MAX_DEPTH = 14
MAX_NODES = 2000
T_LIMIT = 20.0
N_CLICK_CANDS = 8
PRUNE_K = 4
# 已知真值解实例：(gid, level)，计划长度 ≤ MAX_DEPTH
KNOWN = [("vc33", 0), ("vc33", 1), ("sb26", 0), ("r11l", 0),
         ("tn36", 0), ("cd82", 0), ("ft09", 0)]


def load_ranker() -> AffordanceRanker:
    return AffordanceRanker(epochs=15).fit(read_jsonl(DATASET))


def click_candidates(grid, n: int = N_CLICK_CANDS) -> List[Tuple[int, int, int]]:
    comps = connected_components(grid)
    comps.sort(key=lambda c: -int(c["area"]))
    return [(6, int(c["cx"]), int(c["cy"])) for c in comps[:n]]


def candidates_for(arm: str, grid, kb: List[int], ranker: Optional[AffordanceRanker]):
    clicks = click_candidates(grid)
    if arm in ("ranked", "pruned", "random4") and ranker is not None:
        clicks.sort(key=lambda c: -ranker.score(click_context(grid, c[1], c[2])))
    if arm == "pruned":
        clicks = clicks[:PRUNE_K]
    if arm == "random4":
        # 归因对照：同样剪 4 个，但按状态确定性随机选（归因"排序器 vs 剪枝本身"）
        import hashlib as _hl
        import random as _rnd
        rng = _rnd.Random(int(_hl.md5(np.ascontiguousarray(grid, np.int64).tobytes())
                              .hexdigest()[:8], 16))
        pool = clicks + [(6, c['cx'], c['cy']) for c in []]
        rng.shuffle(pool)
        clicks = pool[:PRUNE_K]
    specs = [(a, None, None) for a in sorted(kb)]
    return clicks + specs


def r3_click_search(miner: TrajectoryMiner, arm: str,
                    ranker: Optional[AffordanceRanker]) -> Dict[str, Any]:
    """在 miner 当前定位好的关首状态上做三臂之一的闭包搜索。

    搜索原语 = 「回起点 + 前向重放」（clone-replay 教义）：vc33 上实测，
    snapshot/restore 在跨分支后不满足 MDP（充能链被分支间点击打断，
    2026-09-30），而 restore(start) 后纯前向重放恒可靠。每节点展开 =
    restore 起点一次 → 重放该节点路径 → 逐候选试招。nodes 计所有
    perform_action（重放 + 试招），三臂口径一致。
    """
    game = miner._game
    t0 = time.time()
    out = {"arm": arm, "found": False, "nodes": 0, "path_len": 0, "time": 0.0,
           "reason": ""}
    if game is None:
        out["reason"] = "no game"
        return out
    start_level = int(game.level_index)
    start_snap = generic_snapshot(game)
    start_grid = grid_of(miner._frame.frame)
    avail = [int(a) for a in (miner._frame.available_actions or []) if int(a) not in (6, 7)]
    heap: List[Tuple] = [(0, 0, [], start_grid)]
    seq = 0
    seen = {counter_free_digest(start_snap)}
    nodes = 0

    def perform(spec) -> Optional[Any]:
        aid, x, y = spec
        data = {"x": x, "y": y} if aid == 6 and x is not None else {}
        try:
            return game.perform_action(
                ActionInput(id=getattr(GameAction, f"ACTION{aid}"), data=data), raw=True)
        except Exception:
            return None

    try:
        while heap:
            depth, _, path, grid = heapq.heappop(heap)
            if depth >= MAX_DEPTH:
                continue
            # 回起点 + 前向重放到该节点
            generic_restore(game, start_snap)
            replay_ok = True
            for spec in path:
                nodes += 1
                if perform(spec) is None:
                    replay_ok = False
                    break
            if not replay_ok:
                continue
            for spec in candidates_for(arm, grid, avail, ranker):
                if time.time() - t0 > T_LIMIT:
                    out["reason"] = "timeout"
                    return out
                nodes += 1
                raw = perform(spec)
                if raw is None:
                    continue
                child_grid = grid_of(raw.frame)
                if int(game.level_index) > start_level or str(game._state) == "GameState.WIN":
                    out.update(found=True, nodes=nodes, path_len=len(path) + 1,
                               time=round(time.time() - t0, 1), reason="goal")
                    return out
                if str(game._state) == "GameState.GAME_OVER":
                    continue
                child_snap = generic_snapshot(game)
                key = counter_free_digest(child_snap)
                if key in seen:
                    continue
                seen.add(key)
                seq += 1
                heapq.heappush(heap, (depth + 1, seq, path + [spec], child_grid))
            if nodes > MAX_NODES:
                out["reason"] = "node cap"
                return out
        out["reason"] = "exhausted"
        return out
    finally:
        generic_restore(game, start_snap)
        out["nodes"] = nodes
        out["time"] = round(time.time() - t0, 1)


def main() -> int:
    logging.getLogger("arc_agi").setLevel(logging.CRITICAL)
    ranker = load_ranker()
    plans = extract_agent_plans()
    miner = TrajectoryMiner(seed=0)

    def per_level_of(gid: str):
        for var, (fmt, g) in PLAN_VAR_FORMAT.items():
            if g == gid and var in plans:
                return {lv: plan_steps(fmt, seq) for lv, seq in plans[var].items()}
        return None

    instances: List[Tuple[str, int]] = []
    for gid, lv in KNOWN:
        if per_level_of(gid):
            instances.append((gid, lv))
    all_gids = sorted(p.name for p in
                      (ROOT / "environment_files").iterdir() if p.is_dir())
    for gid in all_gids:
        if (gid, 0) not in instances:
            instances.append((gid, 0))

    results = []
    for gid, level in instances:
        full = gid_to_full_id(gid)
        per_level = per_level_of(gid) or {}
        for arm in ARMS:
            miner._make(full)
            if miner._game is None:
                results.append({"gid": gid, "level": level, "arm": arm,
                                "found": False, "nodes": 0, "path_len": 0,
                                "time": 0.0, "reason": "no env"})
                continue
            miner._frame = miner._env.reset()
            if per_level and not miner._locate(full, gid, per_level, level,
                                               "bench", 0.0):
                results.append({"gid": gid, "level": level, "arm": arm,
                                "found": False, "nodes": 0, "path_len": 0,
                                "time": 0.0, "reason": "locate failed"})
                continue
            r = r3_click_search(miner, arm, ranker)
            r.update(gid=gid, level=level)
            results.append(r)
        last = [r for r in results if r["gid"] == gid and r["level"] == level]
        tag = " ".join(f"{r['arm'][:3]}={'Y' if r['found'] else 'n'}/{r['nodes']}"
                       for r in last)
        print(f"  {gid:6s} L{level}  {tag}", flush=True)

    OUT_JSON.write_text(json.dumps(results, indent=1), encoding="utf-8")

    # 汇总：known 实例（真值解存在）上三臂对比
    print(f"\n== R3 剪枝实验（max_depth={MAX_DEPTH}, nodes<={MAX_NODES}, "
          f"t<={T_LIMIT}s）==")
    for label, sel in (("known(7)", KNOWN), ("all-L0(25)", None)):
        rows = [r for r in results
                if (sel is None and r["level"] == 0)
                or (sel is not None and (r["gid"], r["level"]) in sel)]
        by_inst: Dict[Tuple, Dict[str, dict]] = {}
        for r in rows:
            by_inst.setdefault((r["gid"], r["level"]), {})[r["arm"]] = r
        found = {a: sum(1 for arms in by_inst.values()
                        if arms.get(a, {}).get("found")) for a in ARMS}
        common = [arms for arms in by_inst.values()
                  if all(arms.get(a, {}).get("found") for a in ARMS)]
        red_b = statistics.mean(
            arms["control"]["nodes"] / max(1, arms["ranked"]["nodes"])
            for arms in common) if common else 0.0
        red_c = statistics.mean(
            arms["control"]["nodes"] / max(1, arms["pruned"]["nodes"])
            for arms in common) if common else 0.0
        print(f"{label:10s} found: " + "  ".join(f"{a}={found[a]}" for a in ARMS)
              + f"  common={len(common)}"
              + (f"  nodes比 control/ranked={red_b:.2f} control/pruned={red_c:.2f}"
                 if common else ""))
    print(f"→ {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
