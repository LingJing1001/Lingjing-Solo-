"""wa30 L3: only drop on walls where cyjrduhzmz-equivalent path exists now."""
from __future__ import annotations

import logging
import os
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path[:0] = [str(REPO), str(ROOT), str(ROOT / "vendor" / "ARC-AGI-3-Agents")]
os.chdir(ROOT)
logging.disable(logging.CRITICAL)

from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import ActionInput, GameAction  # noqa: E402
from agent.my_agent import _WA30_PLANS  # noqa: E402

STEP = 4
DIRS = [(1, 0, -STEP), (2, 0, STEP), (3, -STEP, 0), (4, STEP, 0)]
STANDS = [(0, -STEP, 2), (0, STEP, 1), (-STEP, 0, 4), (STEP, 0, 3)]
# Exterior NPC stands relative to wall cell (outside the soft box)
EXTERIOR = {
    # bottom wall y=44 → stand below
    **{(x, 44): (x, 48) for x in range(20, 44, 4)},
    # top wall y=20 → stand above
    **{(x, 20): (x, 16) for x in range(20, 44, 4)},
    # left wall x=20 → stand left
    **{(20, y): (16, y) for y in range(20, 48, 4)},
    # right wall x=40 → stand right
    **{(40, y): (44, y) for y in range(20, 48, 4)},
}
GOOD_WALLS = set(EXTERIOR.keys())


def act(g, a: int) -> None:
    g.perform_action(ActionInput(id=GameAction.from_id(a), data={}), raw=True)


def steps_left(g) -> int:
    return int(getattr(g.kuncbnslnm, "current_steps", 0) or 0)


def free_iz(g) -> int:
    carried = set(g.zmqreragji)
    return sum(
        1
        for b in g.current_level.get_sprites_by_tag("geezpjgiyd")
        if (b.x, b.y) in g.wyzquhjerd and b not in carried
    )


def on_wall(g):
    carried = set(g.zmqreragji)
    return [
        (b.x, b.y)
        for b in g.current_level.get_sprites_by_tag("geezpjgiyd")
        if (b.x, b.y) in g.qthdiggudy and b not in carried
    ]


def npc_carry_count(g) -> int:
    return sum(
        1
        for h in g.nsevyuople
        if "kdweefinfi" in (getattr(h, "tags", []) or [])
    )


def accounted(g) -> int:
    return free_iz(g) + npc_carry_count(g) + len(on_wall(g))


def player_targets(g):
    carried = set(g.zmqreragji)
    out = []
    for b in g.current_level.get_sprites_by_tag("geezpjgiyd"):
        if b in carried:
            continue
        pos = (b.x, b.y)
        if pos in g.wyzquhjerd or pos in g.qthdiggudy:
            continue
        if not (20 < b.x < 40 and 20 < b.y < 44):
            continue
        out.append(pos)
    return out


def npc_deliver_ok(g, wall: tuple[int, int], extra_block_at=None) -> bool:
    """True if NPC at exterior stand carrying block on wall can reach zone."""
    stand = EXTERIOR.get(wall)
    if stand is None:
        return False
    hx, hy = stand
    bx, by = wall
    ox, oy = bx - hx, by - hy
    # occupancy snapshot
    occ = set(g.pkbufziase)
    # pretend block is on wall and holder on stand; remove real player etc. noise
    # add all current blocks except we'll place ours
    if extra_block_at:
        occ.add(extra_block_at)
    occ.add(wall)
    occ.add(stand)
    # soft walls blocked for holder
    qwalls = set(g.qthdiggudy)

    def can_move(chx, chy, nhx, nhy) -> bool:
        nbx, nby = nhx + ox, nhy + oy
        if (nhx, nhy) in qwalls:
            return False
        cur_b = (chx + ox, chy + oy)
        # holder dest
        if (nhx, nhy) in occ and (nhx, nhy) != cur_b and (nhx, nhy) != (chx, chy):
            # allow stepping onto current block cell (swapping trail)
            pass
        occ2 = set(occ)
        occ2.discard((chx, chy))
        occ2.discard(cur_b)
        if (nhx, nhy) in occ2 and (nhx, nhy) != cur_b:
            return False
        if (nbx, nby) in occ2 and (nbx, nby) != (chx, chy):
            return False
        if not (0 <= nhx < 64 and 0 <= nhy < 64 and 0 <= nbx < 64 and 0 <= nby < 64):
            return False
        return True

    q = deque([(hx, hy)])
    seen = {(hx, hy)}
    while q:
        chx, chy = q.popleft()
        cbx, cby = chx + ox, chy + oy
        if (cbx, cby) in g.wyzquhjerd:
            return True
        for dx, dy in ((-STEP, 0), (STEP, 0), (0, -STEP), (0, STEP)):
            nhx, nhy = chx + dx, chy + dy
            if (nhx, nhy) in seen:
                continue
            if not can_move(chx, chy, nhx, nhy):
                continue
            seen.add((nhx, nhy))
            q.append((nhx, nhy))
    return False


def viable_walls(g) -> list[tuple[int, int]]:
    out = []
    for w in GOOD_WALLS:
        # skip if currently occupied by a sitting block
        if w in on_wall(g):
            continue
        if npc_deliver_ok(g, w):
            out.append(w)
    # prefer bottom, then left, then top, then right; spread x
    def key(w):
        x, y = w
        side = 0 if y == 44 else (1 if x == 20 else (2 if y == 20 else 3))
        return (side, abs(x - 28) + abs(y - 32))
    out.sort(key=key)
    return out


def walk_bfs(g, goal, ignore=None):
    p = g.current_level.get_sprites_by_tag("wbmdvjhthc")[0]
    if p in g.nsevyuople:
        return None
    blocked = set(g.pkbufziase) | set(g.qthdiggudy)
    for b in g.current_level.get_sprites_by_tag("geezpjgiyd"):
        if ignore and (b.x, b.y) == ignore:
            continue
        if b not in g.zmqreragji:
            blocked.add((b.x, b.y))
    for s in g.current_level.get_sprites_by_tag("kdweefinfi"):
        blocked.add((s.x, s.y))
    blocked.discard((p.x, p.y))
    q = deque([((p.x, p.y), [])])
    seen = {(p.x, p.y)}
    while q:
        (x, y), path = q.popleft()
        if (x, y) == goal:
            return path
        for a, dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if (nx, ny) in seen or (nx, ny) in blocked:
                continue
            if not (0 <= nx < 64 and 0 <= ny < 64):
                continue
            seen.add((nx, ny))
            q.append(((nx, ny), path + [a]))
    return None


def _carry_ok(g, player, hx, hy, ox, oy, nhx, nhy, hx0, hy0) -> bool:
    nbx, nby = nhx + ox, nhy + oy
    if (nhx, nhy) in g.qthdiggudy:
        return False
    occ = set(g.pkbufziase)
    occ.discard((player.x, player.y))
    occ.discard((hx0 + ox, hy0 + oy))
    occ.discard((hx, hy))
    occ.discard((hx + ox, hy + oy))
    if (nhx, nhy) in occ and (nhx, nhy) != (hx + ox, hy + oy):
        return False
    if (nbx, nby) in occ and (nbx, nby) != (hx, hy):
        return False
    if not (0 <= nhx < 64 and 0 <= nhy < 64 and 0 <= nbx < 64 and 0 <= nby < 64):
        return False
    return True


def _carry_bfs_to(g, hx0, hy0, ox, oy, targets: set) -> list[int] | None:
    p = g.current_level.get_sprites_by_tag("wbmdvjhthc")[0]
    q = deque([((hx0, hy0), [])])
    seen = {(hx0, hy0)}
    best = None
    while q:
        (hx, hy), path = q.popleft()
        bx, by = hx + ox, hy + oy
        if (bx, by) in targets:
            cand = (len(path), path)
            if best is None or cand < best:
                best = cand
            if len(path) <= 6:
                return path
        if len(path) > 22:
            continue
        for a, dx, dy in DIRS:
            nhx, nhy = hx + dx, hy + dy
            if (nhx, nhy) in seen:
                continue
            if not _carry_ok(g, p, hx, hy, ox, oy, nhx, nhy, hx0, hy0):
                continue
            seen.add((nhx, nhy))
            q.append(((nhx, nhy), path + [a]))
    return None if best is None else best[1]


def push_to_viable(g, path, bx, by) -> tuple[bool, tuple | None]:
    targets = set(viable_walls(g))
    if not targets:
        print("  no viable walls!", flush=True)
        return False, None
    scored = []
    for sdx, sdy, face in STANDS:
        sx, sy = bx + sdx, by + sdy
        if (sx, sy) in g.qthdiggudy or not (0 <= sx < 64 and 0 <= sy < 64):
            continue
        w = walk_bfs(g, (sx, sy), ignore=(bx, by))
        if w is None:
            continue
        ox, oy = bx - sx, by - sy
        cpath = _carry_bfs_to(g, sx, sy, ox, oy, targets)
        if cpath is None:
            continue
        # verify final drop cell still viable (recompute after geometric path)
        # approximate final block pos
        hx, hy = sx, sy
        for a in cpath:
            dx = {1: 0, 2: 0, 3: -STEP, 4: STEP}[a]
            dy = {1: -STEP, 2: STEP, 3: 0, 4: 0}[a]
            hx, hy = hx + dx, hy + dy
        drop = (hx + ox, hy + oy)
        if drop not in targets:
            continue
        total = len(w) + 2 + len(cpath) + 1
        side = 0 if drop[1] == 44 else (1 if drop[0] == 20 else 2)
        scored.append((side, total, w, face, cpath, drop))
    if not scored:
        return False, None
    scored.sort()
    _, _, w, face, cpath, drop = scored[0]
    if steps_left(g) < len(w) + len(cpath) + 4:
        return False, None
    for a in w + [face, 5] + cpath:
        act(g, a)
        path.append(a)
        if g.level_index > 3:
            return True, None
    p = g.current_level.get_sprites_by_tag("wbmdvjhthc")[0]
    if p in g.nsevyuople:
        blk = g.nsevyuople[p]
        drop = (blk.x, blk.y)
        ok = drop in GOOD_WALLS and npc_deliver_ok(g, drop)
        print(f"  drop {drop} viable={ok} walls_left={len(viable_walls(g))}", flush=True)
        act(g, 5)
        path.append(5)
        return ok, drop
    return False, None


def tick(g, path, n: int = 1) -> None:
    for i in range(n):
        if steps_left(g) <= 1 or g.level_index > 3:
            return
        a = 3 if (len(path) + i) % 2 == 0 else 4
        act(g, a)
        path.append(a)


def wait_until(g, path, pred, max_n: int) -> bool:
    for _ in range(max_n):
        if pred():
            return True
        if steps_left(g) <= 1 or g.level_index > 3:
            return pred()
        tick(g, path, 1)
    return pred()


def main() -> None:
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
    path: list[int] = []
    order = [(24, 40), (36, 40), (32, 36), (24, 24), (36, 24), (32, 24)]
    print("start", steps_left(g), "viable", len(viable_walls(g)), flush=True)

    for r in range(24):
        if g.level_index > 3:
            break
        need = player_targets(g)
        walls = on_wall(g)
        acc = accounted(g)
        print(
            f"r{r} st={steps_left(g)} iz={free_iz(g)} wall={walls} "
            f"npcC={npc_carry_count(g)} acc={acc} need={need} "
            f"viable={len(viable_walls(g))}",
            flush=True,
        )
        if steps_left(g) <= 1:
            break
        if not need or acc >= 7:
            wait_until(
                g, path,
                lambda: free_iz(g) >= 7 and npc_carry_count(g) == 0 and not on_wall(g),
                55,
            )
            break
        if walls:
            # wait pick; if wall sits too long, maybe unviable — still wait
            wait_until(g, path, lambda: not on_wall(g), 8)
            continue
        # if any NPC stuck (carry but no progress), wait a bit for others then continue
        if npc_carry_count(g) >= 2:
            # let one finish before adding more pressure
            wait_until(g, path, lambda: npc_carry_count(g) <= 1, 12)

        if not viable_walls(g):
            # wait for zone to clear paths
            print("  waiting for viable walls...", flush=True)
            wait_until(g, path, lambda: bool(viable_walls(g)) or free_iz(g) >= 7, 15)
            if not viable_walls(g) and need:
                # last resort: wait all NPC idle
                wait_until(g, path, lambda: npc_carry_count(g) == 0, 20)
            continue

        target = next((t for t in order if t in need), need[0])
        ok, drop = push_to_viable(g, path, *target)
        print(f"  push {target} ok={ok} drop={drop} st={steps_left(g)}", flush=True)
        if ok and drop:
            wait_until(g, path, lambda: not on_wall(g), 6)
        elif not ok:
            order = order[1:] + [order[0]]

    wait_until(
        g, path,
        lambda: g.level_index > 3 or (
            free_iz(g) >= 7 and npc_carry_count(g) == 0 and not on_wall(g)
        ),
        55,
    )
    print(
        "final", g.level_index, "iz", free_iz(g), "st", steps_left(g),
        "len", len(path), "wall", on_wall(g), "need", player_targets(g),
        "npcC", npc_carry_count(g), flush=True,
    )
    if npc_carry_count(g):
        for h, b in g.nsevyuople.items():
            if "kdweefinfi" in (getattr(h, "tags", []) or []):
                print(
                    "STUCK", (h.x, h.y), "->", (b.x, b.y),
                    "engine", g.cyjrduhzmz(h) is not None,
                    "ours", npc_deliver_ok(g, (b.x, b.y)),
                    flush=True,
                )
    if g.level_index > 3:
        print("FOUND", path, flush=True)
        (ROOT / "scripts" / "_wa30_l3_plan.txt").write_text(repr(path), encoding="utf-8")
    else:
        (ROOT / "scripts" / "_wa30_l3_partial.txt").write_text(repr(path), encoding="utf-8")


if __name__ == "__main__":
    main()
