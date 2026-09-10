"""通用搜索算法模块。

Sources:
    - C:\\newtask-pi\\ar25_solver\\solver.py  (A*, macro search)
    - C:\\newtask-pi\\arc_agi_solver\\solver\\bfs_solver.py
    - C:\\newtask-pi\\arc_agi_solver\\solver\\dfs_solver.py
"""
from .astar import astar_search, AStarResult
from .bfs import bfs_search, BFSResult
from .dfs import dfs_search, DFSResult

__all__ = [
    "astar_search", "AStarResult",
    "bfs_search", "BFSResult",
    "dfs_search", "DFSResult",
]
