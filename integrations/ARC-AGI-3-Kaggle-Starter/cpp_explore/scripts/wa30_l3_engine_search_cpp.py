"""wa30 L3 beam（C++ 加速版）

相对 scripts/wa30_l3_engine_search.py 的改动：
- seen 字典 → C++ Visited（lj_hash 混 level/pos/blocks）
- beam 截断 → lj_beam_top_k
- 环境转移仍用 Python deepcopy + Arcade（下一步才抽象转移模型）

用法（在 Kaggle-Starter 根目录）：
  python cpp_explore/scripts/wa30_l3_engine_search_cpp.py
"""
from __future__ import annotations

import copy
import os
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parents[1]
sys.path[:0] = [str(REPO), str(ROOT), str(ROOT / "vendor" / "ARC-AGI-3-Agents"),
                str(ROOT / "cpp_explore" / "python")]
os.chdir(ROOT)

from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import ActionInput, GameAction  # noqa: E402
from agent.my_agent import _WA30_PLANS  # noqa: E402
from lingjing_explore import Visited, beam_top_k, hash_bytes, mix3, version  # noqa: E402


def in_zone_count(g) -> int:
    zone = set()
    for sp in g.current_level.get_sprites_by_tag("fsjjayjoeg"):
        zone |= {(sp.x + i, sp.y + j) for i in range(sp.width) for j in range(sp.height)}
    return sum(
        1
        for b in g.current_level.get_sprites_by_tag("geezpjgiyd")
        if (b.x, b.y) in zone
    )


def key_u64(g) -> int:
    """把 wa30 抽象状态压成 u64（C++ visited）。"""
    lv = g.current_level
    p = lv.get_sprites_by_tag("wbmdvjhthc")[0]
    blocks = sorted((b.x, b.y) for b in lv.get_sprites_by_tag("geezpjgiyd"))
    carry = len(getattr(g, "nsevyuople", {}) or {})
    blob = struct.pack("<iiii", g.level_index, p.x, p.y, int(p.rotation))
    blob += struct.pack("<i", carry)
    for x, y in blocks:
        blob += struct.pack("<hh", x, y)
    return mix3(g.level_index, carry, hash_bytes(blob))


def act(g, a):
    g.perform_action(ActionInput(id=GameAction.from_id(a), data={}), raw=True)


def main() -> None:
    print("lingjing_explore:", version())
    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(ROOT / "environment_files"),
    )
    env = arc.make("wa30")
    env.reset()
    g = env._game
    for lv in (0, 1, 2):
        for aid in _WA30_PLANS[lv]:
            act(g, aid)
    start_li = g.level_index
    n_blocks = len(g.current_level.get_sprites_by_tag("geezpjgiyd"))
    print(f"start L{start_li} blocks={n_blocks} in_zone={in_zone_count(g)}")

    g0 = copy.deepcopy(g)
    # beam 元素: (score, steps, g, path)  score 越大越好
    beam = [(0.0, 0, g0, [])]
    seen = Visited()
    seen.insert(key_u64(g0))
    t0 = time.time()
    best_iz = 0
    best_path = None
    acts = [1, 2, 3, 4, 5]
    limit_s = 300.0

    for depth in range(100):
        if time.time() - t0 > limit_s:
            print("timeout")
            break
        cand_scores = []
        cand_items = []  # (steps, g2, npath, iz)
        for _sc, steps, gg, path in beam:
            for a in acts:
                g2 = copy.deepcopy(gg)
                try:
                    act(g2, a)
                except Exception:
                    continue
                npath = path + [a]
                if g2.level_index > start_li or str(g2._state) == "GameState.WIN":
                    print("FOUND", npath)
                    p = ROOT / "scripts" / "_wa30_l3_plan.txt"
                    p.write_text(repr(npath), encoding="utf-8")
                    print("wrote", p)
                    return
                iz = in_zone_count(g2)
                if iz > best_iz:
                    best_iz = iz
                    best_path = npath
                    print(f"progress in_zone={iz}/{n_blocks} steps={len(npath)}")
                k = key_u64(g2)
                if not seen.insert(k):
                    continue
                carry = len(getattr(g2, "nsevyuople", {}) or {})
                # 分数：优先 in_zone，其次少步数，再次 carry
                score = iz * 1e6 - steps * 10 + carry
                cand_scores.append(float(score))
                cand_items.append((steps + 1, g2, npath, iz))

        if not cand_items:
            break
        top = beam_top_k(cand_scores, min(240, len(cand_scores)))
        beam = []
        for i in top:
            steps, g2, npath, iz = cand_items[i]
            beam.append((cand_scores[i], steps, g2, npath))
        if depth % 5 == 0:
            print(
                f"d={depth} beam={len(beam)} seen={len(seen)} "
                f"best_iz={best_iz} top={beam[0][0]:.0f} t={time.time()-t0:.1f}"
            )

    print("FAIL best_iz", best_iz, "/", n_blocks)
    if best_path:
        (ROOT / "scripts" / "_wa30_l3_partial.txt").write_text(
            repr(best_path), encoding="utf-8"
        )
        print("partial", best_path)


if __name__ == "__main__":
    main()
