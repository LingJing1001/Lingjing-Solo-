"""动作命名：兼容 UP/DOWN… 与官方 ACTION1… 两套别名。

官方约定（reasoning_agent / docs）：
    ACTION1 = MOVE_UP
    ACTION2 = MOVE_DOWN
    ACTION3 = MOVE_LEFT
    ACTION4 = MOVE_RIGHT
    ACTION5 = 交互键（SPACE 语义）
    ACTION6 = 坐标点击
    ACTION7 = UNDO
    RESET   = 开局 / 关卡重置
"""
from __future__ import annotations

# 别名 → 官方规范名
ALIAS_TO_CANONICAL = {
    "UP": "ACTION1",
    "DOWN": "ACTION2",
    "LEFT": "ACTION3",
    "RIGHT": "ACTION4",
    "SPACE": "ACTION5",
    "CLICK": "ACTION6",
    "UNDO": "ACTION7",
    "RESET": "RESET",
    "ACTION1": "ACTION1",
    "ACTION2": "ACTION2",
    "ACTION3": "ACTION3",
    "ACTION4": "ACTION4",
    "ACTION5": "ACTION5",
    "ACTION6": "ACTION6",
    "ACTION7": "ACTION7",
}

CANONICAL_TO_ALIAS = {
    "ACTION1": "UP",
    "ACTION2": "DOWN",
    "ACTION3": "LEFT",
    "ACTION4": "RIGHT",
    "ACTION5": "SPACE",
    "ACTION6": "CLICK",
    "ACTION7": "UNDO",
    "RESET": "RESET",
}

# 动作 id（available_actions 整数）→ 规范名
ID_TO_CANONICAL = {
    0: "RESET",
    1: "ACTION1",
    2: "ACTION2",
    3: "ACTION3",
    4: "ACTION4",
    5: "ACTION5",
    6: "ACTION6",
    7: "ACTION7",
}

DEFAULT_SIMPLE = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"]


def canonicalize(action: str) -> str:
    if action is None:
        return ""
    key = str(action).strip().upper()
    return ALIAS_TO_CANONICAL.get(key, key)


def to_alias(action: str) -> str:
    c = canonicalize(action)
    return CANONICAL_TO_ALIAS.get(c, c)


def from_available_ids(ids) -> list[str]:
    """把 FrameData.available_actions 的整数 id 转成规范动作名（不含 RESET）。"""
    out = []
    for i in ids or []:
        try:
            n = int(i)
        except (TypeError, ValueError):
            continue
        name = ID_TO_CANONICAL.get(n)
        if name and name != "RESET":
            out.append(name)
    return out


def normalize_valid(valid_actions) -> list[str]:
    """去重并规范为官方名；过滤空值。"""
    seen = set()
    out = []
    for a in valid_actions or []:
        c = canonicalize(a)
        if c and c not in seen and c != "RESET":
            seen.add(c)
            out.append(c)
    return out
