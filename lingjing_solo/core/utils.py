"""通用工具函数：状态哈希、坐标变换、日志。"""
import hashlib
import numpy as np


def hash_grid(grid: np.ndarray) -> str:
    """对网格做确定性哈希，用于循环检测与转移表主键。

    注：直接用 np.ndarray.tobytes() 保证相同状态产生相同哈希。
    """
    if grid is None:
        return ""
    return hashlib.md5(grid.tobytes()).hexdigest()[:12]


def bbox_of(pixels):
    if not pixels:
        return (0, 0, 0, 0)
    xs = [p[0] for p in pixels]
    ys = [p[1] for p in pixels]
    return (min(xs), min(ys), max(xs), max(ys))


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def delta_region(grid_prev, grid_curr, pad=1):
    """计算两帧差异区域（"泡壁"对应物）：返回变化像素坐标列表。"""
    if grid_prev is None or grid_curr is None:
        return []
    diff = np.argwhere(grid_prev != grid_curr)
    return [tuple(p) for p in diff.tolist()]


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
