"""PORTER：冠军的 ``inference/utils/grid_utils.py`` 未随补丁公开。

``animation.py`` 只用到其中两个符号，这里按可验证的语义补上：字符表取自
冠军 lexicon 源码里的一行注释——``DEFAULT_COLOR_CHARS = "WwgGcBMPRbSYOrNp"
# the Duck harness's ARC_COLOR_CHARS, index = colour value``。因此 16 个字符
# 与色号 0–15 一一对应，这个映射不是猜的。

``format_grid_ascii`` 的原文没拿到，但它只被 ``build_animation_view`` 用来
把一帧渲染成提示里的字符画，语义唯一：逐行拼接、换行分隔。若日后拿到原文，
只应替换本文件，不应改动 ``animation.py``。
"""

from __future__ import annotations

# index = colour value (0–15)。出处见模块 docstring。
ARC_COLOR_CHARS = "WwgGcBMPRbSYOrNp"

__all__ = ["ARC_COLOR_CHARS", "format_grid_ascii"]


def _char(value: int) -> str:
    return ARC_COLOR_CHARS[max(0, min(len(ARC_COLOR_CHARS) - 1, int(value)))]


def format_grid_ascii(grid) -> str:
    """Grid of colour values -> one character per cell, one line per row."""
    return "\n".join("".join(_char(v) for v in row) for row in grid)
