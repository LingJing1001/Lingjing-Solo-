"""通用搜索算法模块。

Sources:
    - C:\\newtask-pi\\ar25_solver\\solver.py  (A*, macro search)
    - C:\\newtask-pi\\arc_agi_solver\\solver\\bfs_solver.py
    - C:\\newtask-pi\\arc_agi_solver\\solver\\dfs_solver.py

Also re-exports SearchEngine / LightweightPlanner from the sibling
``search.py`` module, because the ``search/`` package shadows that file.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from .astar import astar_search, AStarResult
from .bfs import bfs_search, BFSResult
from .dfs import dfs_search, DFSResult


def _load_legacy_engine():
    path = Path(__file__).resolve().parent.parent / "search.py"
    spec = importlib.util.spec_from_file_location(
        "lingjing_solo.planning._legacy_search_engine",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load SearchEngine from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SearchEngine, mod.LightweightPlanner


SearchEngine, LightweightPlanner = _load_legacy_engine()

__all__ = [
    "astar_search", "AStarResult",
    "bfs_search", "BFSResult",
    "dfs_search", "DFSResult",
    "SearchEngine", "LightweightPlanner",
]
