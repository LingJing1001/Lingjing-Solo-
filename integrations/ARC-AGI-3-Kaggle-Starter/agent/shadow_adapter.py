# -*- coding: utf-8 -*-
"""C:\newtask-pi 专用求解器适配器

把 shadow 模块的 solve_level(t_limit) 包装成 solve(game, t_limit) 接口。
shadow 模块内部维护全局 _game，调用 solve_level 前需用 env_ref._game 替换。
"""
import sys
import os
import time
import threading
from typing import Optional, List, Tuple, Any

# C:\newtask-pi 在 sys.path 中
_NEWTASK_PI = "C:/newtask-pi"

# 模块名 → shadow 模块
_SHADOW_MODULES = {
    "g50t": "arc_shadow_g50t",
    "ka59": "arc_shadow_ka59",
    "dc22": "arc_shadow_dc22",
    "lf52": "arc_shadow_lf52",
    "re86": "arc_shadow_re86",
    "m0r0": "arc_shadow_m0r0",
    "ar25": "arc_shadow_ar25",
    "bp35": "arc_shadow_bp35",
    "cd82": "arc_shadow_cd82",
    "cn04": "arc_shadow_cn04",
    "ft09": "arc_shadow_ft09",
    "lp85": "arc_shadow_lp85",
    "ls20": "arc_shadow_ls20",
    "r11l": "arc_shadow_r11l",
    "s5i5": "arc_shadow_s5i5",
    "sb26": "arc_shadow_sb26",
    "sc25": "arc_shadow_sc25",
    "sk48": "arc_shadow_sk48",
    "sp80": "arc_shadow_sp80",
    "su15": "arc_shadow_su15",
    "tn36": "arc_shadow_tn36",
    "tr87": "arc_shadow_tr87",
    "tu93": "arc_shadow_tu93",
    "vc33": "arc_shadow_vc33",
    "wa30": "arc_shadow_wa30",
}

# 缓存已加载的模块
_modules = {}
_locks = {}


def _load_shadow(gid: str):
    """加载 shadow 模块"""
    if gid in _modules:
        return _modules[gid]
    mod_name = _SHADOW_MODULES.get(gid)
    if not mod_name:
        return None
    if _NEWTASK_PI not in sys.path:
        sys.path.insert(0, _NEWTASK_PI)
    try:
        mod = __import__(mod_name)
        _modules[gid] = mod
        _locks[gid] = threading.RLock()
        return mod
    except Exception as e:
        print(f"[shadow_adapter] 加载 {mod_name} 失败: {e}")
        return None


def solve(gid: str, game, t_limit: float = 60.0, **kwargs) -> Optional[List]:
    """求解指定游戏的当前关。

    Args:
        gid: 游戏ID前缀 (如 "g50t", "ka59")
        game: 引擎 game 对象（替换 shadow 模块的全局 _game）
        t_limit: 超时秒数
        **kwargs: 传给 solve_level 的额外参数

    Returns:
        动作序列: int (1-5) 或 tuple (6, x, y) 或 dict {"a":6,"x":x,"y":y}
        已通关: []
        失败: None
    """
    mod = _load_shadow(gid)
    if not mod:
        return None

    lock = _locks.get(gid)
    if lock:
        lock.acquire()
    try:
        # 替换 shadow 模块的全局 _game
        if hasattr(mod, '_game'):
            mod._game = game
        if hasattr(mod, '_env'):
            # 有些 shadow 模块有 _env，需要重置
            try:
                mod._env = game._env if hasattr(game, '_env') else None
            except:
                pass

        # 调用 solve_level 或 solve_level_enhanced
        solve_fn = getattr(mod, 'solve_level_enhanced', None)
        if not solve_fn:
            solve_fn = getattr(mod, 'solve_level', None)
        if not solve_fn:
            return None

        # 调用
        result = solve_fn(t_limit=t_limit, **kwargs)
        return result
    except Exception as e:
        print(f"[shadow_adapter] {gid} solve 失败: {e}")
        return None
    finally:
        if lock:
            lock.release()


def solve_g50t(game, t_limit=120.0, **kwargs):
    return solve("g50t", game, t_limit, **kwargs)


def solve_ka59(game, t_limit=180.0, **kwargs):
    return solve("ka59", game, t_limit, **kwargs)


def solve_dc22(game, t_limit=120.0, **kwargs):
    return solve("dc22", game, t_limit, **kwargs)


def solve_lf52(game, t_limit=120.0, **kwargs):
    return solve("lf52", game, t_limit, **kwargs)


def solve_re86(game, t_limit=120.0, **kwargs):
    return solve("re86", game, t_limit, **kwargs)
