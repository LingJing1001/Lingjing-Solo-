"""Object-level frame diff core, shared by the sandbox runtime and the host.

This module must stay dependency-free (stdlib only, no imports): its source is
injected verbatim into the sandbox bootstrap template (see
python_tool_sandbox.py), which runs under ``python -I -S`` and cannot import
project modules. The host imports it normally for the auto frame-diff feature.
"""


def _node_topleft(node):
    points = node.get("boundary") or []
    if not points:
        return (0, 0)
    return (min(p[0] for p in points), min(p[1] for p in points))

def _hungarian(cost):
    # Minimum-cost assignment (Jonker-Volgonant style with potentials).
    # cost: n x m matrix with n <= m. Returns list row -> column.
    n = len(cost)
    m = len(cost[0]) if n else 0
    INF = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)      # p[j] = row assigned to column j (1-based; 0 = none)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = [-1] * n
    for j in range(1, m + 1):
        if p[j]:
            assignment[p[j] - 1] = j - 1
    return assignment

def _node_bbox(node):
    points = node.get("boundary") or []
    if not points:
        return (0, 0, 0, 0)
    rows = [p[0] for p in points]; cols = [p[1] for p in points]
    return (min(rows), min(cols), max(rows), max(cols))

def _node_center2(node):
    # bbox center in half-cell units (ints): (r0+r1, c0+c1)
    points = node.get("boundary") or []
    if not points:
        return (0, 0)
    rows = [p[0] for p in points]; cols = [p[1] for p in points]
    return (min(rows) + max(rows), min(cols) + max(cols))

def compute_frame_diff(grid_a, grid_b, nodes_a, nodes_b, max_group_match=None):
    # Object-level diff between two same-shaped grids with precomputed
    # segmentation nodes. Returns only what changed; unchanged objects are
    # omitted. When max_group_match is set, identical-hash groups larger than
    # it skip pairwise matching and report plain appeared/disappeared entries
    # (guards against O(n^3) assignment on e.g. hundreds of identical pixels).
    changed = set()
    for r, (row_a, row_b) in enumerate(zip(grid_a, grid_b)):
        if row_a == row_b:
            continue
        for c, (va, vb) in enumerate(zip(row_a, row_b)):
            if va != vb:
                changed.add((r, c))
    if not changed:
        return {"changed_cell_count": 0, "moved": [], "rotated": [],
                "appeared": [], "disappeared": [],
                "changed_color": [], "resized": []}

    def touched_nodes(nodes):
        touched = []
        for node in nodes:
            points = node.get("boundary") or []
            if not points:
                continue
            r0 = min(p[0] for p in points); r1 = max(p[0] for p in points)
            c0 = min(p[1] for p in points); c1 = max(p[1] for p in points)
            # dilate by 1 so objects that shrank/grew still register as
            # touched by the cells they vacated/claimed
            if any(r0 - 1 <= r <= r1 + 1 and c0 - 1 <= c <= c1 + 1 for (r, c) in changed):
                touched.append(node)
        return touched

    before_nodes = touched_nodes(nodes_a)
    after_nodes = touched_nodes(nodes_b)

    moved, appeared, disappeared = [], [], []
    rotated = []

    def _classify_match(node_a, node_b):
        rot_a = node_a.get("rotation")
        rot_b = node_b.get("rotation")
        symmetry = node_a.get("rotational_symmetry") or 1
        rotated_by = None
        if rot_a is not None and rot_b is not None:
            modulus = 180 if symmetry == 2 else 360
            delta = (rot_b - rot_a) % modulus
            if delta:
                rotated_by = delta
        pa, pb = _node_topleft(node_a), _node_topleft(node_b)
        base_entry = {"color": node_a.get("color"),
                      "pixels": node_a.get("pixels"),
                      "hash": node_a.get("hash")}
        if rotated_by is not None:
            ca, cb = _node_center2(node_a), _node_center2(node_b)
            in_place = (abs(ca[0] - cb[0]) <= 2 and abs(ca[1] - cb[1]) <= 2)
            base_entry["rotated_by"] = rotated_by
            if in_place:
                base_entry["at"] = list(pb)
                rotated.append(base_entry)
            else:
                base_entry["from"] = list(pa); base_entry["to"] = list(pb)
                moved.append(base_entry)
        elif pa != pb:
            base_entry["from"] = list(pa); base_entry["to"] = list(pb)
            moved.append(base_entry)
        # same hash, same position, same rotation: unchanged -> omit

    groups_before = {}
    for node in before_nodes:
        groups_before.setdefault(node.get("hash"), []).append(node)
    groups_after = {}
    for node in after_nodes:
        groups_after.setdefault(node.get("hash"), []).append(node)

    for key, group_a in groups_before.items():
        group_b = groups_after.pop(key, [])
        if (
            max_group_match is not None
            and max(len(group_a), len(group_b)) > max_group_match
        ):
            # too many identical objects to match affordably
            for node_a in group_a:
                disappeared.append({"color": node_a.get("color"),
                                    "pixels": node_a.get("pixels"),
                                    "hash": node_a.get("hash"),
                                    "shape_hash": node_a.get("shape_hash"),
                                    "at": list(_node_topleft(node_a)),
                                    "bbox": list(_node_bbox(node_a))})
            for node_b in group_b:
                appeared.append({"color": node_b.get("color"),
                                 "pixels": node_b.get("pixels"),
                                 "hash": node_b.get("hash"),
                                 "shape_hash": node_b.get("shape_hash"),
                                 "at": list(_node_topleft(node_b)),
                                 "bbox": list(_node_bbox(node_b))})
            continue
        if not group_b:
            for node_a in group_a:
                disappeared.append({"color": node_a.get("color"),
                                    "pixels": node_a.get("pixels"),
                                    "hash": node_a.get("hash"),
                                    "shape_hash": node_a.get("shape_hash"),
                                    "at": list(_node_topleft(node_a)),
                                    "bbox": list(_node_bbox(node_a))})
            continue
        # optimal assignment over bbox-center distances; rows = smaller side
        transposed = len(group_a) > len(group_b)
        rows, cols = (group_b, group_a) if transposed else (group_a, group_b)
        cost = []
        for row_node in rows:
            cr = _node_center2(row_node)
            cost.append([
                abs(cr[0] - _node_center2(col_node)[0])
                + abs(cr[1] - _node_center2(col_node)[1])
                for col_node in cols
            ])
        assignment = _hungarian(cost)
        matched_cols = set()
        for row_index, col_index in enumerate(assignment):
            if col_index < 0:
                continue
            matched_cols.add(col_index)
            if transposed:
                _classify_match(cols[col_index], rows[row_index])
            else:
                _classify_match(rows[row_index], cols[col_index])
        if transposed:
            for col_index, node_a in enumerate(cols):
                if col_index not in matched_cols:
                    disappeared.append({"color": node_a.get("color"),
                                        "pixels": node_a.get("pixels"),
                                        "hash": node_a.get("hash"),
                                    "shape_hash": node_a.get("shape_hash"),
                                        "at": list(_node_topleft(node_a)),
                                        "bbox": list(_node_bbox(node_a))})
        else:
            for col_index, node_b in enumerate(cols):
                if col_index not in matched_cols:
                    appeared.append({"color": node_b.get("color"),
                                     "pixels": node_b.get("pixels"),
                                     "hash": node_b.get("hash"),
                             "shape_hash": node_b.get("shape_hash"),
                                     "at": list(_node_topleft(node_b)),
                                     "bbox": list(_node_bbox(node_b))})
            continue
        # transposed: every after-node (a row) got matched; nothing appeared
    for group_b in groups_after.values():
        for node_b in group_b:
            appeared.append({"color": node_b.get("color"),
                             "pixels": node_b.get("pixels"),
                             "hash": node_b.get("hash"),
                             "shape_hash": node_b.get("shape_hash"),
                             "at": list(_node_topleft(node_b)),
                             "bbox": list(_node_bbox(node_b))})
    # Phase A - changed_color: EXACT pairs only. Same position, byte-
    # identical shape (color-independent pose signature), different
    # color. This asserts "these exact cells changed color" and nothing
    # more; fuzzy transformation claims are deliberately not made.
    changed_color = []
    used_gone, used_new = set(), set()
    for gi, gone in enumerate(disappeared):
        for ni, new in enumerate(appeared):
            if ni in used_new:
                continue
            if (gone["at"] == new["at"]
                    and gone.get("shape_hash")
                    and gone.get("shape_hash") == new.get("shape_hash")
                    and gone["color"] != new["color"]):
                changed_color.append({"at": gone["at"],
                                      "pixels": gone["pixels"],
                                      "before_color": gone["color"],
                                      "after_color": new["color"],
                                      "before_hash": gone.get("hash"),
                                      "after_hash": new.get("hash")})
                used_gone.add(gi); used_new.add(ni)
                break
    disappeared = [entry for index, entry in enumerate(disappeared) if index not in used_gone]
    appeared = [entry for index, entry in enumerate(appeared) if index not in used_new]

    # Phase B - resized: same-color pairs whose bboxes substantially
    # overlap (overlap area / smaller bbox area >= 0.5); greedy
    # one-to-one by descending overlap then size similarity.
    resized = []

    def _bbox_area(b):
        return max(0, b[2] - b[0] + 1) * max(0, b[3] - b[1] + 1)

    def _overlap_ratio(b1, b2):
        inter_r = min(b1[2], b2[2]) - max(b1[0], b2[0]) + 1
        inter_c = min(b1[3], b2[3]) - max(b1[1], b2[1]) + 1
        if inter_r <= 0 or inter_c <= 0:
            return 0.0
        smaller = min(_bbox_area(b1), _bbox_area(b2))
        return (inter_r * inter_c) / smaller if smaller else 0.0

    candidates = []
    for gi, gone in enumerate(disappeared):
        for ni, new in enumerate(appeared):
            if gone["color"] != new["color"]:
                continue
            bg = gone.get("bbox") or (0, 0, -1, -1)
            bn = new.get("bbox") or (0, 0, -1, -1)
            ratio = _overlap_ratio(bg, bn)
            if ratio >= 0.5:
                area_g, area_n = _bbox_area(bg), _bbox_area(bn)
                similarity = (min(area_g, area_n) / max(area_g, area_n)) if max(area_g, area_n) else 0.0
                candidates.append((ratio, similarity, gi, ni))
    candidates.sort(key=lambda item: (-item[0], -item[1]))
    used_gone, used_new = set(), set()
    for ratio, similarity, gi, ni in candidates:
        if gi in used_gone or ni in used_new:
            continue
        used_gone.add(gi); used_new.add(ni)
        gone, new = disappeared[gi], appeared[ni]
        resized.append({"at_before": gone["at"], "at_after": new["at"],
                        "color": gone["color"],
                        "before_pixels": gone["pixels"],
                        "after_pixels": new["pixels"],
                        "before_hash": gone.get("hash"),
                        "after_hash": new.get("hash")})
    disappeared = [entry for index, entry in enumerate(disappeared) if index not in used_gone]
    appeared = [entry for index, entry in enumerate(appeared) if index not in used_new]
    return {"changed_cell_count": len(changed), "moved": moved,
            "rotated": rotated,
            "appeared": appeared, "disappeared": disappeared,
            "changed_color": changed_color, "resized": resized}
