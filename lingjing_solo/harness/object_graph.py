"""帧 → 对象图（供 LLM+REPL harness 消费）。

这个模块是 C-Local 旁路的**第一块**：把 64×64 的整数网格压成一张"物体清单"，
让模型看到的是**对象**而不是像素。它刻意对齐 Duck harness 的两条机制：

1. **平移不变身份** —— 每个对象带 ``hash`` = 颜色 + 归一化到左上原点的形状签名
   （复用 ``lingjing_solo.perception.segmentation`` 的实现）。同一物体移动后
   hash 不变、只有坐标变，模型不必重新认它。
2. **确定性渲染** —— :func:`render` 对同一个 :class:`Graph` 逐字节可复现：
   对象按 id 升序、字段定序、不含浮点、不含 dict 迭代序。

第 2 条是**前缀缓存能命中的前提**：任何"稳定段"只要有一个字节抖动，
PagedAttention 的块共享 / RadixAttention 的基数树就会整段 miss。
所以本模块把"渲染"从"提示词拼装"里独立出来，方便单独做字节稳定性测试。

原始数值网格**刻意不进** :func:`render`（对齐 Duck：模型看对象图，不看像素）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from ..core.frame_decode import grid_from_frame
from ..perception.segmentation import segment_frame

__all__ = ["Obj", "Graph", "build_graph", "render", "graph_signature"]


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Obj:
    """一个分割出的物体。

    ``bbox`` 是 ``(r0, c0, r1, c1)``（含端点）。``r`` / ``c`` 是 bbox 中心，
    也就是"这个物体大概在哪"——点击动作与相对方位判断都靠它。
    ``children`` 是**直接被本对象包住**的子对象 id（嵌套树，父是"最内层包住它的那个"）。
    """

    id: int
    color: str
    hash: str
    n: int
    bbox: Tuple[int, int, int, int]
    children: Tuple[int, ...] = ()
    layer: int = 0

    @property
    def r(self) -> int:
        return (self.bbox[0] + self.bbox[2]) // 2

    @property
    def c(self) -> int:
        return (self.bbox[1] + self.bbox[3]) // 2

    def as_dict(self) -> Dict[str, Any]:
        """给沙箱里脚本用的视图（纯 int/str/list，可安全 repr）。"""
        return {
            "id": self.id,
            "color": self.color,
            "hash": self.hash,
            "n": self.n,
            "r": self.r,
            "c": self.c,
            "bbox": list(self.bbox),
            "children": list(self.children),
            "layer": self.layer,
        }


@dataclass(frozen=True)
class Graph:
    """一帧的对象图。不可变，可安全跨回合缓存。"""

    shape: Tuple[int, int]
    objects: Tuple[Obj, ...]
    adjacency: Tuple[Tuple[int, int], ...] = ()

    @property
    def n_objects(self) -> int:
        return len(self.objects)

    def by_hash(self, h: str) -> Tuple[Obj, ...]:
        return tuple(o for o in self.objects if o.hash == h)

    def as_api(self) -> List[Dict[str, Any]]:
        """沙箱 API 里的 ``objects``。逐对象 :meth:`Obj.as_dict`。"""
        return [o.as_dict() for o in self.objects]


# --------------------------------------------------------------------------- #
# 构造
# --------------------------------------------------------------------------- #


def _normalize_frame(frame: Any) -> List[List[int]]:
    """把任意一帧统一成 ``list[list[int]]`` 的**单层**；取不到返回 ``[]``。

    委托给 :func:`lingjing_solo.core.frame_decode.grid_from_frame` —— 与
    ``arc_env.frame_to_grid`` / ``core.utils.extract_grid`` 共用**同一份**解码。

    这里原先只认 ndarray / 嵌套 list，**不认官方 ``FrameData``**，导致
    ``build_graph`` 必须由调用方先解码（那条 FrameData 接缝）。改走统一入口后，
    裸帧可以直接喂进来：``build_graph(latest_frame)`` / ``decide(latest_frame)``
    都能工作。仍然是纯 stdlib（``grid_from_frame`` 内部只用 ``.tolist()``）。
    """
    return grid_from_frame(frame) or []


def _bbox_from_boundary(boundary: Sequence[Sequence[int]]) -> Tuple[int, int, int, int]:
    """`segment_layer` 的 ``boundary`` 是外轮廓角点，其范围即物体 bbox。"""
    if not boundary:
        return (0, 0, 0, 0)
    rows = [int(p[0]) for p in boundary]
    cols = [int(p[1]) for p in boundary]
    return (min(rows), min(cols), max(rows), max(cols))


def build_graph(frame: Any) -> Graph:
    """把一帧（numpy 数组 / 嵌套 list）分割成 :class:`Graph`。

    只用 ``perception.segmentation`` 的纯 stdlib 实现，可在无 numpy 环境运行。
    """
    layer = _normalize_frame(frame)
    if not layer:
        return Graph(shape=(0, 0), objects=(), adjacency=())

    result = segment_frame(layer)
    shape = (len(layer), len(layer[0]) if layer else 0)

    objs: List[Obj] = []
    for node in result["nodes"]:
        bbox = _bbox_from_boundary(node.get("boundary") or [])
        objs.append(
            Obj(
                id=int(node["id"]),
                color=str(node["color"]),
                hash=str(node["hash"]),
                n=int(node["pixels"]),
                bbox=bbox,
                children=tuple(int(c) for c in (node.get("children") or ())),
                layer=int(node.get("layer", 0)),
            )
        )
    objs.sort(key=lambda o: o.id)  # segment_frame 已按 id 有序，这里再钉死一次

    # 邻接表挂在每个 layer 的结果上（segment_frame 的顶层只有 layers/nodes），
    # 这里按同样的 id 命名空间 <<10**6>> 汇总，保证与节点 id 对得上。
    adj_pairs: List[Tuple[int, int]] = []
    for li, layer_res in enumerate(result.get("layers") or ()):
        for a, b in layer_res.get("adjacency_list") or ():
            base = li * 10**6
            adj_pairs.append((base + int(a), base + int(b)))
    adj = tuple(sorted(adj_pairs))

    return Graph(shape=shape, objects=tuple(objs), adjacency=adj)


# --------------------------------------------------------------------------- #
# 渲染（确定性）
# --------------------------------------------------------------------------- #


def render(graph: Graph, max_objects: int = 0, max_adjacency: int = 0) -> str:
    """把对象图渲染成**逐字节稳定**的文本。

    ``max_objects=0`` 表示不截断。截断一律按 id 升序取前 N 个，并在末尾追加
    ``(+M more objects)`` —— 截断点也必须确定，否则前缀又会抖动。
    ``max_adjacency`` 同理（默认 0 = 全部；邻接表对大帧最占 token，可单独限流）。

    输出格式（字段固定、分隔符固定）::

        FRAME 64x64 objects=7 adjacency=3
        #0  c=0 n=4096 r=31 c=31 h=8f2a11de kids=[1,2]
        #1  c=5 n=48   r=31 c=30 h=77aa0011
        adj 0-1 1-2 0-3
    """
    if graph.shape == (0, 0) and graph.n_objects == 0:
        return "FRAME 0x0 objects=0 adjacency=0"

    h, w = graph.shape
    adjs = graph.adjacency if max_adjacency <= 0 else graph.adjacency[:max_adjacency]
    lines = [f"FRAME {h}x{w} objects={graph.n_objects} adjacency={len(adjs)}"]

    kept = graph.objects if max_objects <= 0 else graph.objects[:max_objects]
    for o in kept:
        kids = f" kids=[{','.join(str(k) for k in o.children)}]" if o.children else ""
        lines.append(
            f"#{o.id} c={o.color} n={o.n} r={o.r} c={o.c} h={o.hash} bbox={o.bbox[0]},{o.bbox[1]}-{o.bbox[2]},{o.bbox[3]}{kids}"
        )
    if max_objects > 0 and graph.n_objects > max_objects:
        lines.append(f"(+{graph.n_objects - max_objects} more objects)")

    if adjs:
        lines.append("adj " + " ".join(f"{a}-{b}" for a, b in adjs))

    return "\n".join(lines)


def graph_signature(graph: Graph) -> str:
    """渲染结果的 sha1 前 16 位 —— 用来在单测里断言"渲染是稳定的"。"""
    import hashlib

    return hashlib.sha1(render(graph).encode("utf-8")).hexdigest()[:16]
