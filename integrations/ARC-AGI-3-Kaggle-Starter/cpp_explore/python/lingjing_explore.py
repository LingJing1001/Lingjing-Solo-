"""Python 绑定：ctypes 加载 lingjing_explore.dll

用于 ARC-AGI-3 未知环境冲关：
- 快速状态哈希
- C++ visited
- beam top-k
- 紧凑 BFS 调度（路径存 C++，Python 只负责 env.step）
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

_DLL = None
_DIR = Path(__file__).resolve().parents[1] / "bin"


def _load():
    global _DLL
    if _DLL is not None:
        return _DLL
    names = ["lingjing_explore.dll", "liblingjing_explore.dll", "lingjing_explore.so"]
    candidates = [_DIR / n for n in names]
    # 也允许同目录 / PATH
    candidates += [Path(n) for n in names]
    last_err = None
    for p in candidates:
        try:
            if p.exists() or p.name in names:
                _DLL = ctypes.CDLL(str(p if p.exists() else p.name))
                break
        except OSError as e:
            last_err = e
    if _DLL is None:
        raise OSError(
            f"找不到 lingjing_explore.dll，请先运行 cpp_explore/build_msvc.bat\n"
            f"期望路径: {_DIR / 'lingjing_explore.dll'}\n最后错误: {last_err}"
        )

    d = _DLL
    d.lj_explore_version.restype = ctypes.c_char_p

    d.lj_hash_bytes.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
    d.lj_hash_bytes.restype = ctypes.c_uint64

    d.lj_hash_grid_i8.argtypes = [ctypes.POINTER(ctypes.c_int8), ctypes.c_int, ctypes.c_int]
    d.lj_hash_grid_i8.restype = ctypes.c_uint64

    d.lj_hash_mix2.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
    d.lj_hash_mix2.restype = ctypes.c_uint64
    d.lj_hash_mix3.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
    d.lj_hash_mix3.restype = ctypes.c_uint64

    d.lj_visited_create.restype = ctypes.c_void_p
    d.lj_visited_destroy.argtypes = [ctypes.c_void_p]
    d.lj_visited_insert.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    d.lj_visited_insert.restype = ctypes.c_int
    d.lj_visited_contains.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    d.lj_visited_contains.restype = ctypes.c_int
    d.lj_visited_size.argtypes = [ctypes.c_void_p]
    d.lj_visited_size.restype = ctypes.c_size_t
    d.lj_visited_clear.argtypes = [ctypes.c_void_p]

    d.lj_beam_top_k.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]

    d.lj_bfs_create.argtypes = [ctypes.c_int, ctypes.c_int]
    d.lj_bfs_create.restype = ctypes.c_void_p
    d.lj_bfs_destroy.argtypes = [ctypes.c_void_p]
    d.lj_bfs_push_root.argtypes = [ctypes.c_void_p]
    d.lj_bfs_push_root.restype = ctypes.c_int
    d.lj_bfs_pop.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    d.lj_bfs_pop.restype = ctypes.c_int
    d.lj_bfs_path.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int),
        ctypes.c_int, ctypes.POINTER(ctypes.c_int),
    ]
    d.lj_bfs_path.restype = ctypes.c_int
    d.lj_bfs_try_child.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_int),
    ]
    d.lj_bfs_try_child.restype = ctypes.c_int
    d.lj_bfs_mark_seen.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    d.lj_bfs_queue_size.argtypes = [ctypes.c_void_p]
    d.lj_bfs_queue_size.restype = ctypes.c_size_t
    d.lj_bfs_seen_size.argtypes = [ctypes.c_void_p]
    d.lj_bfs_seen_size.restype = ctypes.c_size_t
    return _DLL


def version() -> str:
    return _load().lj_explore_version().decode("utf-8", errors="replace")


def hash_bytes(data: Union[bytes, bytearray, memoryview]) -> int:
    d = _load()
    buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
    return int(d.lj_hash_bytes(buf, len(data)))


def hash_grid_i8(flat: Sequence[int], h: int, w: int) -> int:
    d = _load()
    arr = (ctypes.c_int8 * (h * w))(*[int(x) for x in flat])
    return int(d.lj_hash_grid_i8(arr, h, w))


def hash_numpy_grid(grid) -> int:
    """numpy 数组 → u64（按 int8 bytes）。"""
    import numpy as np
    a = np.asarray(grid).astype(np.int8, copy=False)
    return hash_bytes(a.tobytes())


def mix2(a: int, b: int) -> int:
    return int(_load().lj_hash_mix2(a, b))


def mix3(a: int, b: int, c: int) -> int:
    return int(_load().lj_hash_mix3(a, b, c))


def encode_action(a) -> int:
    """Python 动作 → int 编码。a 为 int 或 (6,x,y)。"""
    if isinstance(a, tuple):
        act, x, y = a
        return 0x6000000 | ((int(x) & 0xFF) << 8) | (int(y) & 0xFF)
    return int(a)


def decode_action(code: int):
    if (code & 0x6000000) == 0x6000000:
        x = (code >> 8) & 0xFF
        y = code & 0xFF
        return (6, x, y)
    return int(code)


class Visited:
    def __init__(self):
        self._h = _load().lj_visited_create()

    def __del__(self):
        try:
            if getattr(self, "_h", None):
                _load().lj_visited_destroy(self._h)
        except Exception:
            pass

    def insert(self, key: int) -> bool:
        return bool(_load().lj_visited_insert(self._h, key))

    def contains(self, key: int) -> bool:
        return bool(_load().lj_visited_contains(self._h, key))

    def __len__(self) -> int:
        return int(_load().lj_visited_size(self._h))

    def clear(self):
        _load().lj_visited_clear(self._h)


def beam_top_k(scores: Sequence[float], k: int) -> List[int]:
    d = _load()
    n = len(scores)
    if n == 0 or k <= 0:
        return []
    k = min(k, n)
    arr = (ctypes.c_double * n)(*[float(s) for s in scores])
    out = (ctypes.c_int * k)()
    d.lj_beam_top_k(arr, n, k, out)
    return [int(out[i]) for i in range(k)]


class CompactBFS:
    """C++ 管 queue/seen/path；Python 负责对每个弹出节点做 env 展开。"""

    def __init__(self, max_nodes: int = 20000, max_depth: int = 40):
        self._h = _load().lj_bfs_create(max_nodes, max_depth)

    def __del__(self):
        try:
            if getattr(self, "_h", None):
                _load().lj_bfs_destroy(self._h)
        except Exception:
            pass

    def push_root(self) -> int:
        return int(_load().lj_bfs_push_root(self._h))

    def pop(self) -> Optional[Tuple[int, int]]:
        nid = ctypes.c_int()
        depth = ctypes.c_int()
        ok = _load().lj_bfs_pop(self._h, ctypes.byref(nid), ctypes.byref(depth))
        if not ok:
            return None
        return int(nid.value), int(depth.value)

    def path(self, node_id: int) -> List:
        buf = (ctypes.c_int * 64)()
        n = ctypes.c_int()
        ok = _load().lj_bfs_path(self._h, node_id, buf, 64, ctypes.byref(n))
        if not ok:
            return []
        return [decode_action(int(buf[i])) for i in range(int(n.value))]

    def try_child(self, parent_id: int, action, child_hash: int) -> Optional[int]:
        out = ctypes.c_int()
        ok = _load().lj_bfs_try_child(
            self._h, parent_id, encode_action(action), child_hash, ctypes.byref(out)
        )
        return int(out.value) if ok else None

    def mark_seen(self, key: int) -> None:
        _load().lj_bfs_mark_seen(self._h, int(key))

    @property
    def queue_size(self) -> int:
        return int(_load().lj_bfs_queue_size(self._h))

    @property
    def seen_size(self) -> int:
        return int(_load().lj_bfs_seen_size(self._h))


def state_key_u64(level: int, score: int, frame_bytes: bytes, sprite_sig_bytes: bytes = b"") -> int:
    """对应 my_agent._key 的压缩版：全部混成一个 u64，供 visited 使用。"""
    g = hash_bytes(frame_bytes)
    s = hash_bytes(sprite_sig_bytes) if sprite_sig_bytes else 0
    return mix3(level & 0xFFFFFFFF, score & 0xFFFFFFFF, mix2(g, s))
