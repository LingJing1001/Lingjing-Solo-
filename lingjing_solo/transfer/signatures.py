"""结构化状态签名：对象摘要 + 动作空间，替代 CEAX 玩具网格字符串匹配。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ObjectSig:
    """连通域对象的可哈希摘要（无坐标脚本，可跨布局比较颜色/尺寸）。"""

    color: int
    area: int
    bbox_w: int
    bbox_h: int

    def mechanism_key(self) -> Tuple[int, int]:
        """跨布局机制键：颜色 + 面积粗分桶（忽略绝对几何）。"""
        return (self.color, min(self.area // 16, 8))

    def color_key(self) -> int:
        return self.color


@dataclass(frozen=True)
class StateSignature:
    """可迁移状态签名：对象机制摘要 + 可行动作，不含绝对坐标。"""

    objects: Tuple[ObjectSig, ...]
    actions: Tuple[str, ...]
    level: int = 0
    grid_hash: str = ""

    def mechanism_fingerprint(self) -> Tuple:
        """同机制异几何：颜色集合 + 动作空间（面积仅作辅证，指纹忽略尺寸）。"""
        return (
            tuple(sorted({o.color for o in self.objects})),
            self.actions,
        )


def _bbox_of(pixels: Sequence[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    if not pixels:
        return (0, 0, 0, 0)
    ys = [p[0] for p in pixels]
    xs = [p[1] for p in pixels]
    return (min(ys), min(xs), max(ys), max(xs))


def objects_from_grid(grid: np.ndarray, max_objects: int = 24) -> List[ObjectSig]:
    """轻量连通域：按颜色分组估计对象（不依赖完整 encoder）。"""
    if grid is None:
        return []
    g = np.asarray(grid)
    if g.ndim == 3:
        g = g[0]
    out: List[ObjectSig] = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) != 0):
        ys, xs = np.where(g == color)
        if len(ys) == 0:
            continue
        y0, x0, y1, x1 = int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())
        out.append(
            ObjectSig(
                color=color,
                area=int(len(ys)),
                bbox_w=int(x1 - x0 + 1),
                bbox_h=int(y1 - y0 + 1),
            )
        )
        if len(out) >= max_objects:
            break
    return out


def objects_from_perception(objects: Optional[Iterable[Any]]) -> List[ObjectSig]:
    """从 PerceptionSnapshot.objects / GameObject 提取签名。"""
    if not objects:
        return []
    out: List[ObjectSig] = []
    for obj in objects:
        color = int(getattr(obj, "color", 0))
        pixels = getattr(obj, "pixels", None) or []
        bbox = getattr(obj, "bbox", None)
        if bbox and len(bbox) == 4:
            x0, y0, x1, y1 = bbox
            # bbox 约定可能是 (x0,y0,x1,y1) 或 (y0,x0,y1,x1)；用宽高绝对值
            bw = abs(int(x1) - int(x0)) + 1
            bh = abs(int(y1) - int(y0)) + 1
            area = len(pixels) if pixels else bw * bh
        elif pixels:
            y0, x0, y1, x1 = _bbox_of([(p[0], p[1]) for p in pixels])
            bw, bh = x1 - x0 + 1, y1 - y0 + 1
            area = len(pixels)
        else:
            continue
        out.append(ObjectSig(color=color, area=int(area), bbox_w=int(bw), bbox_h=int(bh)))
    return out


def build_signature(
    grid=None,
    *,
    objects=None,
    actions: Optional[Sequence[str]] = None,
    level: int = 0,
    grid_hash: str = "",
) -> StateSignature:
    objs = objects_from_perception(objects) if objects else objects_from_grid(grid)
    acts = tuple(sorted(str(a) for a in (actions or ())))
    return StateSignature(
        objects=tuple(sorted(objs, key=lambda o: (o.color, o.area, o.bbox_w, o.bbox_h))),
        actions=acts,
        level=int(level or 0),
        grid_hash=grid_hash or "",
    )
