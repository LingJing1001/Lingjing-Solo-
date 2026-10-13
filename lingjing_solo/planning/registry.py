"""R3 planner/profile 注册表（§3.2.1 / §7.1.1 R3-D 统一路由）。

registry 的末端接入 planner，而不是只接罐头路线。resolve_planner 按
T0 → T1 → T2 → T3 顺序尝试，返回第一个非 none 的 plan。
"""
from __future__ import annotations

from typing import Any, Sequence

from .world_view import WorldView, assert_world_view_invariant, WorldViewInvariantError
from .backends import SearchBackend, LearnedGraphBackend
from .r3_kernel import R3Kernel, SearchBudget


# ── 注册表 ────────────────────────────────────────────────────

_WORLD_VIEWS: dict[str, WorldView] = {}
_BACKENDS: dict[str, SearchBackend] = {}


def register_world_view(game_id: str, wv: WorldView, *, sample_frame: Any = None) -> None:
    """注册一个游戏的 WorldView。

    注册时调用 assert_world_view_invariant 做 P2 硬约束断言。
    """
    if not game_id:
        raise ValueError("game_id 不能为空")
    assert_world_view_invariant(wv, sample_frame=sample_frame)
    _WORLD_VIEWS[game_id] = wv


def get_world_view(game_id: str) -> WorldView | None:
    """获取游戏的 WorldView，未注册返回 None。"""
    return _WORLD_VIEWS.get(game_id)


def register_backend(game_id: str, backend: SearchBackend) -> None:
    """注册一个游戏的搜索后端。"""
    if not game_id:
        raise ValueError("game_id 不能为空")
    if not isinstance(backend, SearchBackend):
        raise TypeError(f"backend 必须是 SearchBackend 子类，收到 {type(backend).__name__}")
    _BACKENDS[game_id] = backend


def get_backend(game_id: str) -> SearchBackend:
    """获取游戏的搜索后端，未注册返回 LearnedGraphBackend 默认。"""
    backend = _BACKENDS.get(game_id)
    if backend is None:
        return LearnedGraphBackend()
    return backend


def clear_registry() -> None:
    """清空注册表（测试用）。"""
    _WORLD_VIEWS.clear()
    _BACKENDS.clear()


# ── 统一路由 ────────────────────────────────────────────────

def resolve_planner(
    game_id: str,
    frame: Any,
    legal_actions: Sequence[str],
    state_hash: str,
    *,
    level: int = 0,
    budget: SearchBudget | None = None,
    script_bank_fn: Any = None,
) -> dict[str, Any] | None:
    """统一路由入口（R3-D）：按 T0→T1→T2→T3 顺序尝试。

    Returns:
        plan dict，或 None（游戏未注册 WorldView 时）。
    """
    wv = get_world_view(game_id)
    if wv is None:
        return None
    backend = get_backend(game_id)
    kernel = R3Kernel(
        wv,
        backend,
        budget=budget,
        script_bank_fn=script_bank_fn,
    )
    return kernel.plan(
        frame,
        legal_actions,
        state_hash,
        game_id=game_id,
        level=level,
    )


__all__ = [
    "register_world_view",
    "get_world_view",
    "register_backend",
    "get_backend",
    "clear_registry",
    "resolve_planner",
]
