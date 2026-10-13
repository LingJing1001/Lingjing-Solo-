"""动作契约：harness 内部唯一的动作表示。

刻意与 Duck harness 对齐：**六个具名动作，没有 Undo**。
- ``UP`` / ``DOWN`` / ``LEFT`` / ``RIGHT``  —— 方向键
- ``SPACE``                                —— 中性/交互键
- ``CLICK``（带 row/col）                   —— 点格子

不把 Undo 交给模型是有意的：Undo 会让"探索"变成免费回退，
模型于是不再需要对自己的动作负责，历史里也留不下可归因的因果。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

__all__ = ["Action", "ACTION_NAMES", "normalize_action_name", "make_action", "CLICK"]

ACTION_NAMES = ("UP", "DOWN", "LEFT", "RIGHT", "SPACE", "CLICK")

# 常见别名 -> 规范名（模型偶尔会写这些）
_ALIASES = {
    "MOUSE": "CLICK",
    "CLICK_AT": "CLICK",
    "TAP": "CLICK",
    "SELECT": "CLICK",
    "ENTER": "SPACE",
    "SPACE_KEY": "SPACE",
    "UP_KEY": "UP",
    "DOWN_KEY": "DOWN",
    "LEFT_KEY": "LEFT",
    "RIGHT_KEY": "RIGHT",
    "UNDO": "SPACE",  # 明确不支持 Undo：降级成 SPACE，不静默失败
    "RESET": "SPACE",
    "NOOP": "SPACE",
    "WAIT": "SPACE",
}

CLICK = "CLICK"


def normalize_action_name(name: Any) -> str:
    """把模型给的任意写法归一成 :data:`ACTION_NAMES` 之一。

    ``None`` 或无法识别 -> 抛 ``ValueError``（由调用方决定是记为失败还是降级）。
    """
    if name is None:
        raise ValueError("action name is None")
    s = str(name).strip().upper().replace("-", "_").replace(" ", "_")
    if s in ACTION_NAMES:
        return s
    if s in _ALIASES:
        return _ALIASES[s]
    raise ValueError(f"unknown action: {name!r}")


@dataclass(frozen=True)
class Action:
    """一个动作。``CLICK`` 必须有 ``row`` / ``col``；其余动作忽略坐标。"""

    kind: str
    row: Optional[int] = None
    col: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", normalize_action_name(self.kind))
        if self.kind == CLICK:
            if self.row is None or self.col is None:
                raise ValueError("CLICK requires row and col")
            object.__setattr__(self, "row", int(self.row))
            object.__setattr__(self, "col", int(self.col))
        else:
            # 非点击动作把坐标抹平，避免"带了但被忽略"的坐标偷偷影响去重/日志
            object.__setattr__(self, "row", None)
            object.__setattr__(self, "col", None)

    @property
    def needs_coords(self) -> bool:
        return self.kind == CLICK

    def as_dict(self) -> dict:
        d = {"kind": self.kind}
        if self.kind == CLICK:
            d["row"] = self.row
            d["col"] = self.col
        return d

    def __str__(self) -> str:  # 日志友好
        if self.kind == CLICK:
            return f"CLICK({self.row},{self.col})"
        return self.kind


def make_action(name: Any, row: Any = None, col: Any = None) -> Action:
    """沙箱里 ``action()`` 的实现。宽松接受数字字符串（模型偶尔写 ``"3"``）。"""
    def _int_or_none(v: Any) -> Optional[int]:
        if v is None:
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    return Action(normalize_action_name(name), _int_or_none(row), _int_or_none(col))
