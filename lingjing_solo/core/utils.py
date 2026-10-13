"""通用工具函数：状态哈希、帧解析、坐标变换、日志。"""
from __future__ import annotations

import hashlib
import numpy as np
from .actions import from_available_ids, canonicalize
from .frame_decode import grid_from_frame


def hash_grid(grid: np.ndarray, levels: int = None) -> str:
    """对网格做确定性哈希；可选混入 levels_completed（关卡切分状态空间）。"""
    if grid is None:
        return ""
    h = hashlib.md5()
    if levels is not None:
        h.update(str(int(levels)).encode())
    arr = np.asarray(grid)
    h.update(arr.tobytes())
    return h.hexdigest()[:12]


def bbox_of(pixels):
    if not pixels:
        return (0, 0, 0, 0)
    xs = [p[0] for p in pixels]
    ys = [p[1] for p in pixels]
    return (min(xs), min(ys), max(xs), max(ys))


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def delta_region(grid_prev, grid_curr, pad=1):
    """计算两帧差异区域：返回变化像素坐标列表。"""
    if grid_prev is None or grid_curr is None:
        return []
    diff = np.argwhere(np.asarray(grid_prev) != np.asarray(grid_curr))
    return [tuple(p) for p in diff.tolist()]


def _layers_of(frame) -> list:
    """兼容 FrameData.frame = list[list[list[int]]] 或多层。

    保留（不再被 :func:`extract_grid` 使用）：解码已统一到
    :func:`lingjing_solo.core.frame_decode.grid_from_frame`，
    这里留作外部脚本的兼容入口。
    """
    raw = getattr(frame, "frame", None)
    if raw is None and isinstance(frame, dict):
        raw = frame.get("frame")
    if raw is None:
        return []
    if not raw:
        return []
    first = raw[0]
    if isinstance(first, np.ndarray):
        return [first] if first.ndim == 2 else [first[i] for i in range(len(first))]
    # 单层: list[list[int]]；多层: list[list[list[int]]]
    if isinstance(first, (list, tuple)) and first and isinstance(first[0], list):
        return raw
    return [raw]


def extract_grid(frame) -> "np.ndarray | None":
    """从多种帧输入提取顶层网格为 numpy 数组。

    慢路径（FrameData / dict-with-frame / 嵌套 list）统一委托给
    :func:`lingjing_solo.core.frame_decode.grid_from_frame` —— 与 harness 旁路
    （``arc_env.frame_to_grid`` / ``object_graph.build_graph``）共用**同一份**解码，
    这样"提交链认得的帧"和"harness 认得的帧"不会再有差异。

    快路径（已是 ndarray / 已是 ``.grid`` ndarray）原样返回，避免多一次拷贝与
    dtype 抖动 —— 上层 ``hash_grid`` 是按字节哈希的，dtype 必须稳定。
    """
    if frame is None:
        return None
    if isinstance(frame, np.ndarray):
        return frame
    # 已包装的 Frame dataclass
    if hasattr(frame, "grid") and isinstance(getattr(frame, "grid"), np.ndarray):
        return frame.grid
    for attr in ("array", "observation"):
        if hasattr(frame, attr):
            val = getattr(frame, attr)
            if val is not None:
                return np.asarray(val)
    if isinstance(frame, dict):
        for k in ("grid", "array", "observation"):
            if frame.get(k) is not None:
                return np.asarray(frame[k])
    # 唯一解码入口（FrameData(.frame) / 帧栈 / 多层 list / 裸嵌套 list）
    grid = grid_from_frame(frame)
    if grid is not None:
        return np.asarray(grid, dtype=np.int16)
    return None


def extract_state(frame) -> str:
    """提取环境状态字符串：WIN / GAME_OVER / NOT_PLAYED / NOT_FINISHED。"""
    if frame is None:
        return "NOT_FINISHED"
    state = getattr(frame, "state", None)
    if state is None and isinstance(frame, dict):
        state = frame.get("state")
    if state is None:
        return "NOT_FINISHED"
    # Enum → value / name
    val = getattr(state, "value", None) or getattr(state, "name", None) or state
    return str(val).upper().replace("GAMESTATE.", "")


def extract_levels(frame) -> int:
    if frame is None:
        return 0
    lv = getattr(frame, "levels_completed", None)
    if lv is None and isinstance(frame, dict):
        lv = frame.get("levels_completed", 0)
    try:
        return int(lv or 0)
    except (TypeError, ValueError):
        return 0


def extract_available_actions(frame, fallback=None) -> list[str]:
    """从 FrameData.available_actions 或包装 Frame 提取合法动作列表。"""
    if frame is not None:
        aa = getattr(frame, "available_actions", None)
        if aa is None and isinstance(frame, dict):
            aa = frame.get("available_actions")
        if aa:
            # 已是字符串列表
            if isinstance(aa[0], str):
                return [canonicalize(a) for a in aa if canonicalize(a) != "RESET"]
            return from_available_ids(aa)
    return list(fallback or [])


def is_win_state(frame_or_state) -> bool:
    if isinstance(frame_or_state, str):
        return frame_or_state.upper() == "WIN"
    return extract_state(frame_or_state) == "WIN"


def needs_reset(frame_or_state) -> bool:
    s = frame_or_state if isinstance(frame_or_state, str) else extract_state(frame_or_state)
    return s in ("NOT_PLAYED", "GAME_OVER")


class Logger:
    """轻量结构化日志，避免 print 污染评测。"""
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.records = []

    def log(self, layer, msg, **kw):
        line = {"layer": layer, "msg": msg, **kw}
        self.records.append(line)
        if self.enabled:
            print(f"[{layer}] {msg}")

    def tail(self, n=20):
        return self.records[-n:]
