"""Connected-component segmentation of a single frame layer.

Ported verbatim from ``Tufalabs/duck-harness`` (MIT), module
``ARC3-Inference/inference/utils/segmentation.py``.  The upstream module is
deliberately self-contained -- standard library only, no project imports -- so
that its source can also be spliced into the Python-tool sandbox bootstrap
where project packages are not importable.  We keep it standard-library-only
here for the same reason: this file must stay importable from anywhere.

What it gives us that ``perception/`` did not have before:

* 4-connected flood-fill labelling of a frame layer,
* a **translation-invariant object hash** (colour + normalised cell shape), which
  is the piece that lets the same object be matched *across frames*,
* a containment tree (``children``) so a sprite inside a frame is distinguishable
  from a sprite standing on the background,
* an adjacency list between touching components.

``segment_frame`` is the only adapter we added on top, so callers holding an
ARC-AGI-3 observation frame (a list of 64x64 integer layers) do not have to
unpack it themselves.
"""

import hashlib

# ARC-AGI-3 colour symbols indexed by integer colour value.  Matches the
# character set used throughout the harness' perception helpers.
ARC_COLOR_CHARS = "0123456789abcdef"

_ORTH = ((-1, 0), (1, 0), (0, -1), (0, 1))
# clockwise Moore-neighbour offsets, starting at NW
_CW = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))
_CW_INDEX = {off: i for i, off in enumerate(_CW)}


def _trace_outer_contour(cells, start):
    """Moore-neighbour trace of a 4-connected component's outer perimeter, clockwise."""
    if len(cells) == 1:
        return [start]

    contour = [start]
    b = start
    prev = (start[0], start[1] - 1)  # W neighbour: outside the component since start is reading-order-min
    second = None
    for _ in range(8 * len(cells) + 16):
        idx = _CW_INDEX[(prev[0] - b[0], prev[1] - b[1])]
        nxt = None
        for k in range(1, 9):
            off = _CW[(idx + k) % 8]
            cand = (b[0] + off[0], b[1] + off[1])
            if cand in cells:
                nxt = cand
                back = _CW[(idx + k - 1) % 8]
                new_prev = (b[0] + back[0], b[1] + back[1])
                break
        if nxt is None:
            break
        if second is None:
            second = nxt
        elif b == start and nxt == second:  # Jacob's stopping criterion
            break
        contour.append(nxt)
        prev, b = new_prev, nxt

    if len(contour) > 1 and contour[-1] == contour[0]:
        contour.pop()
    return contour


def _corner_points(contour):
    """Reduce a traced contour loop to only the points where its direction changes."""
    if len(contour) <= 2:
        return list(contour)
    m = len(contour)
    corners = []
    for i in range(m):
        prev, cur, nxt = contour[i - 1], contour[i], contour[(i + 1) % m]
        d_in = (cur[0] - prev[0], cur[1] - prev[1])
        d_out = (nxt[0] - cur[0], nxt[1] - cur[1])
        if d_in != d_out:
            corners.append(cur)
    return corners


def _object_hash(cells, color):
    """Translation-invariant signature of an object: its color plus its cell shape,
    normalized so the top-left of its bounding box is the origin. Same shape + color
    => same hash regardless of position, so objects can be matched across frames."""
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm = sorted((r - min_r, c - min_c) for r, c in cells)
    payload = repr((color, norm)).encode()
    return hashlib.sha1(payload).hexdigest()[:16]


def _compute_enclosers_floodfill(components, comp_id, height, width):
    """Upstream implementation, kept for cross-checking.

    For each component b, flood-fill its complement inward from the grid border;
    any component whose cells are never reached is enclosed by b.  Correct but
    O(n * H * W): ~50 ms on a 64x64 frame with 14 components, which is too slow
    to call on every agent step.  ``_compute_enclosers_rag`` is the default.
    """
    n = len(components)
    enclosers = [set() for _ in range(n)]
    for b in range(n):
        reached = [[False] * width for _ in range(height)]
        stack = []
        for r in range(height):
            for c in (0, width - 1):
                if comp_id[r][c] != b and not reached[r][c]:
                    reached[r][c] = True
                    stack.append((r, c))
        for c in range(width):
            for r in (0, height - 1):
                if comp_id[r][c] != b and not reached[r][c]:
                    reached[r][c] = True
                    stack.append((r, c))
        while stack:
            r, c = stack.pop()
            for dr, dc in _ORTH:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and not reached[nr][nc] and comp_id[nr][nc] != b:
                    reached[nr][nc] = True
                    stack.append((nr, nc))
        for a in range(n):
            if a == b:
                continue
            ar, ac = components[a]["start"]
            if not reached[ar][ac]:
                enclosers[a].add(b)
    return enclosers


def _compute_enclosers_rag(components, comp_id, height, width, adj_pairs):
    """Same result as ``_compute_enclosers_floodfill``, but on the component graph.

    "a is enclosed by b" means: every path from a to the grid edge crosses b.
    Because components are 4-connected and touching components are recorded in
    ``adj_pairs``, that is exactly "a is unreachable from the border in the
    component adjacency graph with b deleted".  So we BFS once per b over the
    component graph -- O(n * (n + E)) instead of O(n * H * W).  On the same
    64x64 / 14-component frame this is ~0.5 ms instead of ~50 ms.

    Verified element-for-element against the flood-fill version on the full
    ``data/aop/episodes`` corpus (see state/_verify_segmentation.py).
    """
    n = len(components)
    rag = [[] for _ in range(n)]
    for a, b in adj_pairs:
        rag[a].append(b)
        rag[b].append(a)

    border = set()
    for r in range(height):
        border.add(comp_id[r][0])
        border.add(comp_id[r][width - 1])
    for c in range(width):
        border.add(comp_id[0][c])
        border.add(comp_id[height - 1][c])

    enclosers = [set() for _ in range(n)]
    for b in range(n):
        seen = set()
        stack = []
        for x in border:
            if x != b and x not in seen:
                seen.add(x)
                stack.append(x)
        while stack:
            x = stack.pop()
            for y in rag[x]:
                if y == b or y in seen:
                    continue
                seen.add(y)
                stack.append(y)
        for a in range(n):
            if a != b and a not in seen:
                enclosers[a].add(b)
    return enclosers


def segment_layer(layer, color_chars=ARC_COLOR_CHARS, containment="rag"):
    """Segment one frame layer into connected-component nodes.

    Pass a single layer (if the frame has multiple) and ``color_chars``, the ARC
    color-symbol mapping (indexed by integer color value -> single-char label). The
    layer is partitioned into 4-connected components of equal integer value via flood
    fill, and each component becomes a node. Nodes are ordered by their
    top-most-left-most cell -- unique within a 64x64 layer -- and that order is the
    node ``id``.

    Each node is a dict with:
      - ``id``: index in the top-left ordering.
      - ``color``: the component's ARC color character (looked up in ``color_chars``).
      - ``hash``: a translation-invariant signature of the object -- its color plus its
        cell shape normalized to a top-left origin -- so the same shape gets the same
        hash regardless of position (lets objects be matched across frames, or when
        several similar objects appear in one frame).
      - ``pixels``: number of cells in the component.
      - ``boundary``: the component's outer perimeter as an ordered, clockwise list of
        ``[row, col]`` corner points -- a Moore-neighbour trace reduced to only the
        vertices where the contour changes direction (enclosed holes are not traced).
      - ``children``: ids of components directly enclosed by this node. A is a child of
        B only if B is the innermost component that fully surrounds A (every path from A
        to the grid edge crosses B), which yields a clean nesting tree.

    Returns a dict with:
      - ``nodes``: list of the node dicts above, in id order.
      - ``adjacency_list``: sorted list of ``[i, j]`` id pairs for components that share
        a 4-connected edge (includes parent/child pairs, since they physically touch).

    ``containment`` selects the enclosure algorithm: ``"rag"`` (default, fast,
    result-identical) or ``"floodfill"`` (upstream, slow, kept for cross-checking).
    """
    height = len(layer)
    width = len(layer[0]) if height else 0

    # connected components, 4-connectivity. Reading-order scan => component ids are
    # already ordered by top-most-left-most cell (each layer is 64x64, so it is unique).
    comp_id = [[-1] * width for _ in range(height)]
    components = []  # each: {"value": int, "cells": set[(r, c)], "start": (r, c)}
    for sr in range(height):
        for sc in range(width):
            if comp_id[sr][sc] != -1:
                continue
            value = layer[sr][sc]
            cid = len(components)
            cells = set()
            stack = [(sr, sc)]
            comp_id[sr][sc] = cid
            while stack:
                r, c = stack.pop()
                cells.add((r, c))
                for dr, dc in _ORTH:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and comp_id[nr][nc] == -1 and layer[nr][nc] == value:
                        comp_id[nr][nc] = cid
                        stack.append((nr, nc))
            components.append({"value": int(value), "cells": cells, "start": (sr, sc)})

    n = len(components)

    # adjacency between components: any two components with 4-adjacent cells
    adj_pairs = set()
    for r in range(height):
        for c in range(width):
            cid = comp_id[r][c]
            if r + 1 < height and comp_id[r + 1][c] != cid:
                other = comp_id[r + 1][c]
                adj_pairs.add((min(cid, other), max(cid, other)))
            if c + 1 < width and comp_id[r][c + 1] != cid:
                other = comp_id[r][c + 1]
                adj_pairs.add((min(cid, other), max(cid, other)))
    adjacency_list = sorted([a, b] for a, b in adj_pairs)

    # containment: who encloses whom (see the two implementations above)
    if containment == "floodfill":
        enclosers = _compute_enclosers_floodfill(components, comp_id, height, width)
    else:
        enclosers = _compute_enclosers_rag(components, comp_id, height, width, adjacency_list)

    # parent = innermost encloser. enclosers are transitive, so along a nesting chain the
    # innermost component is the one that is itself most deeply enclosed.
    children = [[] for _ in range(n)]
    for a in range(n):
        if enclosers[a]:
            parent = max(enclosers[a], key=lambda e: (len(enclosers[e]), -e))
            children[parent].append(a)
    for child_list in children:
        child_list.sort()

    nodes = []
    for cid in range(n):
        comp = components[cid]
        color = color_chars[max(0, min(15, comp["value"]))]
        boundary = _corner_points(_trace_outer_contour(comp["cells"], comp["start"]))
        nodes.append(
            {
                "id": cid,
                "color": color,
                "hash": _object_hash(comp["cells"], color),
                "pixels": len(comp["cells"]),
                "boundary": [[r, c] for r, c in boundary],
                "children": children[cid],
            }
        )

    return {"nodes": nodes, "adjacency_list": adjacency_list}


def segment_frame(frame, color_chars=ARC_COLOR_CHARS):
    """Segment an ARC-AGI-3 observation frame.

    ``frame`` is either a single 2-D layer or a list of layers (as delivered in
    ``recording.jsonl``'s ``data.frame``).  Multi-layer frames are segmented per
    layer; the returned dict carries one ``segment_layer`` result per layer plus a
    flat ``nodes`` list whose node ``id`` is namespaced as ``layer * 10**6 + id``
    so ids stay unique across layers.
    """
    if not frame:
        return {"layers": [], "nodes": []}

    # frame[0][0] is an int  -> frame is a single 2-D layer
    # frame[0][0] is a list -> frame is a list of layers
    if isinstance(frame[0][0], list):
        layers = frame
    else:
        layers = [frame]

    per_layer = [segment_layer(layer, color_chars) for layer in layers]
    flat = []
    for li, res in enumerate(per_layer):
        for node in res["nodes"]:
            merged = dict(node)
            merged["layer"] = li
            merged["local_id"] = node["id"]
            merged["id"] = li * 10**6 + node["id"]
            flat.append(merged)
    return {"layers": per_layer, "nodes": flat}
