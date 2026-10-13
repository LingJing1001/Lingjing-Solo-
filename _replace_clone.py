import re
from pathlib import Path

p = Path("integrations/ARC-AGI-3-Kaggle-Starter/agent/my_agent.py")
content = p.read_text(encoding="utf-8")

# New greedy function
new_func = '''def _wa30_plan_level_clone(g, hard_cap: int = 760, beam: int = 200):
    """克隆版 wa30 规划：贪心逐箱交付，每箱在真引擎副本上验证。

    策略：每次选最近未交付的箱，在克隆上试 4 个方向抓放，
    取最短可行宏，执行后进入下一箱。比 beam 轻很多。
    """
    import copy as _copy
    try:
        from arcengine import ActionInput as _AI, GameAction as _GA
    except Exception:
        return None

    lv = g.current_level
    zone = set()
    for sp in lv.get_sprites_by_tag("fsjjayjoeg"):
        zone |= {(sp.x + i, sp.y + j) for i in range(sp.width) for j in range(sp.height)}
    qth = {(sp.x, sp.y) for sp in lv.get_sprites_by_tag("bnzklblgdk")}
    player = lv.get_sprites_by_tag("wbmdvjhthc")[0]
    block_ids = {id(b) for b in lv.get_sprites_by_tag("geezpjgiyd")}
    static_solid = {(sp.x, sp.y) for sp in lv.get_sprites()
                    if sp.is_collidable and id(sp) not in block_ids
                    and id(sp) != id(player)}
    W, H = g.camera.width, g.camera.height
    budget = getattr(getattr(g, "kuncbnslnm", None), "dbdarsgrbj", None)
    cap = min(hard_cap, int(budget)) if budget else hard_cap

    def _exec_on_clone(g2, actions):
        try:
            for aid in actions:
                g2.perform_action(_AI(id=_GA.from_id(aid), data={}), raw=True)
            return True
        except Exception:
            return False

    def _read_state(g2):
        lv2 = g2.current_level
        pl = lv2.get_sprites_by_tag("wbmdvjhthc")
        blocks = tuple((b.x, b.y) for b in lv2.get_sprites_by_tag("geezpjgiyd"))
        p = (pl[0].x, pl[0].y) if pl else None
        rot = pl[0].rotation if pl else 0
        return p, rot, blocks

    blocks0 = tuple((b.x, b.y) for b in lv.get_sprites_by_tag("geezpjgiyd"))
   !     if all(b in zone for b in blocks0):
        return []

    g_work = _copy.deepcopy(g)
    full_seq: list[int] = []
    max_boxes = len(blocks0)

    for _box_iter in range(max_boxes * 2):
        p, rot, blocks = _read_state(g_work)
        if p is None:
            return None
        if all(b in zone for b in blocks):
            return full_seq
        if len(full_seq) > cap:
            return None

        solid = static_solid | set(blocks)
        blocked = (solid | qth) - {p}
        best = None

        undelivered = [(bi, blk) for bi, blk in enumerate(blocks) if blk not in zone]
        if not undelivered:
            return full_seq

        for bi, blk in undelivered:
            for d, (dx, dy) in _WA30_DIRV.items():
                stance = (blk[0] - dx, blk[1] - dy)
                w = _wa30_walk_bfs(p, rot, stance, _WA30_ROTD[d], blocked, W, H)
                if w is None:
                    continue
                off = (blk[0] - stance[0], blk[1] - stance[1])
                static_c = (solid | qth) - {p, blk}
                c = _wa30_carry_bfs(stance, off, static_c, zone, qth, W, H)
                if c is None:
                    continue
                macro = w + [5] + c + [5]
                if len(full_seq) + len(macro) > cap:
                    continue
                g_test = _copy.deepcopy(g_work)
                if not _exec_on_clone(g_test, macro):
                    continue
                _, _, test_blocks = _read_state(g_test)
                if test_blocks[bi] == blk:
                    continue
                if best is None or len(macro) < best[0]:
                    best = (len(macro), macro, g_test)

        if best is None:
            return None
        full_seq += best[1]
        g_work = best[2]

    return full_seq if all(b in zone for b in _read_state(g_work)[2]) else None
'''

# Replace the old function (from "def _wa30_plan_level_clone" to the line before "def _wa30_validate_plan")
pattern = r'def _wa30_plan_level_clone\(g.*?(?=\ndef _wa30_validate_plan)'
new_content = re.sub(pattern, new_func + '\n', content, flags=re.DOTALL)

if new_content != content:
    p.write_text(new_content, encoding="utf-8")
    print("replaced OK")
else:
    print("no match found")
