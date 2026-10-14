"""跨帧物体身份：用 segmentation 的平移不变 hash 认「同一个东西」。

``segmentation.segment_layer`` 给每个连通域一个**平移不变签名**（颜色 + 归一化到
左上原点的形状）。于是「形状没变、位置变了」就等于「同一个物体在移动」——这正是
agent 一直缺的那一环：它过去只能看「这一帧和上一帧差了几个像素」，答不出
**这一步把什么推向了哪**。

两种用法，按调用点是否持状态选：

* :func:`objects_of` / :func:`moving_nodes` —— **无状态**，比较给定的两帧。
  适合已经握着 ``prev`` 的调用点（例如 click 候选生成，它的签名本来就有 ``prev``）。
* :class:`ObjectTracker` —— **有状态**，逐帧 :meth:`~ObjectTracker.observe`，
  累积身份与位移历史，供主循环使用。

三条硬约束：

1. **形状变化不匹配**。生长 / 变形会产生新 hash，这是刻意的 —— 形状变了就不是
   同一个东西，硬凑只会制造假位移。实测（``data/aop/episodes`` 24 局）：刚体平移
   能稳定认出，逐帧填充类物体每帧换 hash，会被正确判为「新物体」。
2. **位置来自外轮廓**。``segment_layer`` 只导出 ``boundary``（外轮廓折点），所以
   bbox 由轮廓的极值点推出。对含孔洞的凹形，这仍是**包围盒**而非像素集，足够做
   位移估计；需要精确像素集时请直接用 ``PerceptionEncoder.segment``。
3. **异常全吞只计数**。观测永远不该让 agent 崩，也不该改变既有行为 —— 调用点拿到
   ``ok=False`` 就退回原逻辑，等价于本模块不存在。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["objects_of", "moving_nodes", "ObjectTracker"]


# --------------------------------------------------------------------------- #
# 基础：一帧 → 带 bbox / 中心 的物体清单
# --------------------------------------------------------------------------- #

def _as_layer(grid: Any) -> Optional[list]:
    """把 ndarray / 嵌套 list 归一成 ``layer[row][col]``，失败返回 None。"""
    if grid is None:
        return None
    try:
        if hasattr(grid, "tolist"):
            grid = grid.tolist()
        if not isinstance(grid, (list, tuple)) or not grid:
            return None
        first = grid[0]
        if not isinstance(first, (list, tuple)) or not first:
            return None
        # 单层 frame 的写法是 [[row0, row1, ...]]，多一层时取第一个 2-D 层
        if isinstance(first[0], (list, tuple)):
            grid = first
        return [list(row) for row in grid]
    except Exception:
        return None


def _bbox_from_boundary(boundary: Sequence) -> Optional[Tuple[int, int, int, int]]:
    if not boundary:
        return None
    try:
        rs = [int(p[0]) for p in boundary]
        cs = [int(p[1]) for p in boundary]
    except Exception:
        return None
    if not rs or not cs:
        return None
    return (min(rs), min(cs), max(rs), max(cs))


def _centre(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return ((r0 + r1) // 2, (c0 + c1) // 2)


def objects_of(grid: Any, min_npix: int = 0) -> List[Dict[str, Any]]:
    """一帧的物体清单：``{hash, color, npix, bbox, centre, children}``。

    ``min_npix`` 丢掉像素数不足的小连通域 —— 实测有些帧会有 800+ 个 1~3 像素的
    碎块，它们的 bbox 抖动会被误读成「位移」。默认 0（不过滤）保持纯粹，由调用方
    按场景决定阈值。

    失败返回 ``[]``（不抛）。``grid`` 可以是 2-D ndarray、list-of-list，
    或 ``[[layer]]`` 形式的单层 frame。
    """
    layer = _as_layer(grid)
    if layer is None:
        return []
    try:
        from .segmentation import segment_layer

        res = segment_layer(layer)
    except Exception:
        return []

    floor = int(min_npix)
    out: List[Dict[str, Any]] = []
    for node in (res.get("nodes") or []):
        bbox = _bbox_from_boundary(node.get("boundary"))
        if bbox is None:
            continue
        npix = int(node.get("pixels") or 0)
        if npix < floor:
            continue
        out.append({
            "hash": node.get("hash"),
            "color": node.get("color"),
            "npix": npix,
            "bbox": bbox,
            "centre": _centre(bbox),
            "children": list(node.get("children") or []),
        })
    return out


# --------------------------------------------------------------------------- #
# 无状态：比较两帧，找出「移动了的同一个物体」
# --------------------------------------------------------------------------- #

def _pair_by_hash(prev_objs: List[Dict], cur_objs: List[Dict]):
    """按 hash 配对；同 hash 多实例按左上角（读序）顺序 zip，保证确定性。"""
    prev_idx: Dict[Any, List[Dict]] = {}
    for o in prev_objs:
        prev_idx.setdefault(o["hash"], []).append(o)
    for lst in prev_idx.values():
        lst.sort(key=lambda o: (o["bbox"][0], o["bbox"][1]))

    cur_idx: Dict[Any, List[Dict]] = {}
    for o in cur_objs:
        cur_idx.setdefault(o["hash"], []).append(o)
    for lst in cur_idx.values():
        lst.sort(key=lambda o: (o["bbox"][0], o["bbox"][1]))

    pairs = []
    matched = new = lost = 0
    for h, cur_list in cur_idx.items():
        prev_list = prev_idx.get(h) or []
        if not prev_list:
            new += len(cur_list)
            continue
        for c, p in zip(cur_list, prev_list):
            pairs.append((p, c))
            matched += 1
        if len(cur_list) > len(prev_list):
            new += len(cur_list) - len(prev_list)
    for h, prev_list in prev_idx.items():
        cur_list = cur_idx.get(h) or []
        if len(prev_list) > len(cur_list):
            lost += len(prev_list) - len(cur_list)
    return pairs, matched, new, lost


def compare_frames(prev_grid: Any, cur_grid: Any,
                   min_shift: int = 1, limit: int = 16,
                   prev_objs: Optional[List[Dict[str, Any]]] = None,
                   cur_objs: Optional[List[Dict[str, Any]]] = None,
                   min_npix: int = 0) -> Dict[str, Any]:
    """比较两帧，返回位移报告。

    ``min_shift`` 是位移的 L1 阈值（``|dr| + |dc|`` 大于它才算移动），默认 1 ——
    轮廓给出的 bbox 可能有 ±0 抖动，阈值能滤掉噪声。``limit`` 截断 ``moved``。
    ``min_npix`` 见 :func:`objects_of`；碎块多的帧把它调高能显著降噪。

    已经算过物体清单的调用方（例如 :class:`ObjectTracker` 逐帧循环）可以把
    ``prev_objs`` / ``cur_objs`` 直接传进来，省掉重复分割 —— 这是本模块最贵的一步。

    返回 ``{ok, n_prev, n_cur, matched, new, lost, moved, dominant}``；
    ``moved`` 每项是 ``{hash, color, npix, bbox, delta, shift}``，``delta`` 为
    ``(dr, dc)``（行向下为正、列向右为正）。``dominant`` 是**按像素数加权的平均
    位移**；多个物体反向移动时它会相消甚至回到 ``(0, 0)``，那种情况下请看
    ``moved`` 里各自的 ``delta``。
    """
    report: Dict[str, Any] = {
        "ok": False, "n_prev": 0, "n_cur": 0, "matched": 0,
        "new": 0, "lost": 0, "moved": [], "dominant": None,
    }
    try:
        if prev_objs is None:
            prev_objs = objects_of(prev_grid, min_npix)
        if cur_objs is None:
            cur_objs = objects_of(cur_grid, min_npix)
        report["n_prev"] = len(prev_objs)
        report["n_cur"] = len(cur_objs)
        if not prev_objs or not cur_objs:
            report["ok"] = bool(cur_objs)
            return report

        pairs, matched, new, lost = _pair_by_hash(prev_objs, cur_objs)
        report["matched"] = matched
        report["new"] = new
        report["lost"] = lost

        moved = []
        dr_sum = dc_sum = w_sum = 0
        for p, c in pairs:
            dr = c["centre"][0] - p["centre"][0]
            dc = c["centre"][1] - p["centre"][1]
            if abs(dr) + abs(dc) > int(min_shift):
                moved.append({
                    "hash": c["hash"],
                    "color": c["color"],
                    "npix": c["npix"],
                    "bbox": c["bbox"],
                    "delta": (dr, dc),
                    "shift": abs(dr) + abs(dc),
                })
                # 位移按物体大小加权：大物体移动更能代表「场景被推向哪」
                w = max(1, int(c["npix"]))
                dr_sum += dr * w
                dc_sum += dc * w
                w_sum += w
        moved.sort(key=lambda m: (-m["npix"], m["hash"] or ""))
        report["moved"] = moved[:max(1, int(limit))]
        if moved and w_sum > 0:
            report["dominant"] = (int(round(dr_sum / w_sum)), int(round(dc_sum / w_sum)))
        report["ok"] = True
    except Exception:
        report["ok"] = False
    return report


def moving_nodes(prev_grid: Any, cur_grid: Any,
                 limit: int = 8, min_shift: int = 1,
                 min_npix: int = 0) -> List[Tuple[int, int]]:
    """移动物体的**当前位置中心**，按位移量从大到小。给不了就返回 ``[]``。

    这是喂给 click 候选生成器的那种形态：它要的是「点哪里」，所以给 ``(row, col)``。
    """
    rep = compare_frames(prev_grid, cur_grid, min_shift=min_shift,
                         limit=limit, min_npix=min_npix)
    if not rep.get("ok"):
        return []
    moved = sorted(rep.get("moved") or [], key=lambda m: -m["shift"])
    out = []
    for m in moved[:max(1, int(limit))]:
        c = _centre(m["bbox"])
        if c not in out:
            out.append(c)
    return out


# --------------------------------------------------------------------------- #
# 有状态：逐帧观察，累积身份与位移历史
# --------------------------------------------------------------------------- #

class ObjectTracker:
    """一个关卡内的跨帧物体身份跟踪器。

    ::

        tr = ObjectTracker()
        ...
        if level_changed:
            tr.reset()
        rep = tr.observe(grid)
        if rep["moved"]:
            # rep["dominant"] 就是「这一步整体把这些物体推向了 (dr, dc)」
    """

    def __init__(self, min_shift: int = 2, min_npix: int = 8, limit: int = 16):
        self.min_shift = int(min_shift)
        self.min_npix = int(min_npix)
        self.limit = int(limit)
        self.reset()

    def reset(self) -> None:
        """关卡切换 / RESET 时调用：忘掉上一帧。"""
        self._prev_objs: List[Dict[str, Any]] = []
        self._steps = 0
        self._moved_events = 0
        self._errors = 0
        self._last: Dict[str, Any] = {}

    # -- 只读状态 ------------------------------------------------------------
    @property
    def steps(self) -> int:
        return self._steps

    @property
    def moved_events(self) -> int:
        """累计「至少有一个物体移动」的帧数。"""
        return self._moved_events

    @property
    def errors(self) -> int:
        return self._errors

    @property
    def last(self) -> Dict[str, Any]:
        return dict(self._last)

    def observe(self, grid: Any) -> Dict[str, Any]:
        """折入一帧，返回与 :func:`compare_frames` 同构的报告。异常不外抛。

        内部只对当前帧分割一次（结果缓存给下一帧当 ``prev_objs``），所以逐帧
        调用的成本就是一次 ``segment_layer``。
        """
        layer = _as_layer(grid)
        if layer is None:
            self._errors += 1
            self._last = {"ok": False, "n_prev": 0, "n_cur": 0, "matched": 0,
                          "new": 0, "lost": 0, "moved": [], "dominant": None}
            return dict(self._last)

        cur_objs = objects_of(layer, self.min_npix)
        report = compare_frames(None, None,
                                min_shift=self.min_shift, limit=self.limit,
                                prev_objs=self._prev_objs, cur_objs=cur_objs,
                                min_npix=self.min_npix)
        if not report.get("ok") and cur_objs:
            self._errors += 1
        self._prev_objs = cur_objs
        self._steps += 1
        if report.get("moved"):
            self._moved_events += 1
        self._last = report
        return report

    def dominant(self) -> Optional[Tuple[int, int]]:
        """最近一帧的加权限位移，没有则 ``None``。"""
        return (self._last or {}).get("dominant")

    def summary(self) -> Dict[str, Any]:
        """给日志/诊断用的一句话级摘要（不含浮点，方便逐字节稳定）。"""
        rep = self._last or {}
        moved = rep.get("moved") or []
        return {
            "steps": self._steps,
            "n_cur": rep.get("n_cur", 0),
            "matched": rep.get("matched", 0),
            "moved": len(moved),
            "dominant": rep.get("dominant"),
            "moved_events": self._moved_events,
            "errors": self._errors,
        }
