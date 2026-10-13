"""帧 → 2D 整数网格：**唯一**的解码入口。

## 为什么要有这个模块

历史上"帧 → 网格"被写了**三份**，各自只认自己遇到的那一种输入：

====================  ==============================  ==========================
位置                  认得的输入                        不认的（→ 失败表现）
====================  ==============================  ==========================
``harness/arc_env``    FrameData(.frame) / ndarray /    ——
``frame_to_grid``      嵌套 list / 帧的列表
``harness/object_graph`` ndarray / 嵌套 list /          **FrameData**（接真帧即炸）
``_normalize_frame``    多层 list
``core/utils``         ndarray / Frame(dataclass) /     ——（返回 numpy）
``extract_grid``        FrameData / dict
====================  ==============================  ==========================

最要命的是 ``_normalize_frame`` 不认 ``FrameData`` —— 于是 ``build_graph`` /
``Harness.decide`` 只能在"调用方已经解过码"的前提下工作。谁要是把引擎直接给的
裸帧递进去，就会在 ``frame[0]`` 处炸。这就是那条 **FrameData 接缝**。

本模块把三份合并成一份，harness 旁路与提交链都委托过来，接缝随之消失。

## 约束

- **纯 stdlib**：不 ``import numpy``，只用 ``.tolist()`` 鸭子接口 ——
  这样 harness 侧可以在"没有 numpy 的沙箱替身"里照常跑。
- **确定性**：同样的输入 → 逐字节相同的输出（``int`` 化、无浮点、无 dict 迭代序）。
- **零依赖**：本模块不 import 本仓库任何其它模块，因此谁都能安全地 import 它，
  不会触发 core↔perception↔harness 的循环。

## 语义（与旧的三份对齐）

- 容器对象/字典按 ``frame`` → ``grid`` → ``array`` → ``observation`` 顺序脱壳；
- 网格 = 行是"不含子序列的序列"；
- "帧的列表" / "多层帧" 一律取**最后一层**（约定：渲染层在最上）；
- 取不到 → ``None``（调用方自行决定回退，本模块不抛异常）。
"""

from __future__ import annotations

from typing import Any, List, Optional

__all__ = ["grid_from_frame", "is_grid"]

# 递归深度上限：防御自引用容器（FrameData.frame 指着自己之类）造成的死循环。
_MAX_DEPTH = 12

# 脱壳属性顺序：官方 FrameData(.frame) → 包装 Frame(.grid) → 通用别名
_CONTAINER_ATTRS = ("frame", "grid", "array", "observation")


def _seq(x: Any) -> Any:
    """ndarray → list；其它原样返回。刻意不 import numpy（鸭子接口）。"""
    if hasattr(x, "tolist"):
        try:
            return x.tolist()
        except Exception:  # noqa: BLE001 - 拿不到就按原样继续，由调用方兜底
            return x
    return x


def _is_row(x: Any) -> bool:
    """一行：序列，且元素都不是序列。"""
    return isinstance(x, (list, tuple)) and all(not isinstance(v, (list, tuple)) for v in x)


def is_grid(x: Any) -> bool:
    """``x`` 是不是一张 2D 网格（行是标量序列）。空序列不算。"""
    return isinstance(x, (list, tuple)) and len(x) > 0 and all(_is_row(r) for r in x)


def _as_int_grid(rows: Any) -> List[List[int]]:
    return [[int(v) for v in _seq(r)] for r in rows]


def grid_from_frame(frame: Any, _depth: int = 0) -> Optional[List[List[int]]]:
    """把任意一帧压成 2D ``list[list[int]]``；取不到返回 ``None``。

    接受：``FrameData``（带 ``.frame``）、任意带上述属性的容器、``dict``、
    ndarray（经 ``.tolist()``）、嵌套 ``list``/``tuple``、"帧的列表" / "多层帧"。

    >>> grid_from_frame([[0, 1], [2, 3]])
    [[0, 1], [2, 3]]
    >>> # 多层帧 / 帧的列表 → 取最后一层
    >>> grid_from_frame([[[0, 1], [2, 3]], [[9, 8], [7, 6]]])
    [[9, 8], [7, 6]]
    >>> grid_from_frame(None) is None
    True
    """
    if frame is None or _depth > _MAX_DEPTH:
        return None

    # 1) 显式容器：FrameData(.frame) / 包装 Frame(.grid) / dict 的同名字段。
    for attr in _CONTAINER_ATTRS:
        inner = getattr(frame, attr, None)
        if inner is None and isinstance(frame, dict):
            inner = frame.get(attr)
        if inner is None or inner is frame:
            continue
        got = grid_from_frame(inner, _depth + 1)
        if got is not None:
            return got

    # 2) 裸序列
    seq = _seq(frame)
    if not isinstance(seq, (list, tuple)) or len(seq) == 0:
        return None

    if is_grid(seq):
        return _as_int_grid(seq)

    # 3) 帧栈 / 多层帧 → 取最后一层（与 arc_env.frame_to_grid 旧行为一致）
    last = _seq(seq[-1])
    if is_grid(last):
        return _as_int_grid(last)
    return grid_from_frame(last, _depth + 1)
