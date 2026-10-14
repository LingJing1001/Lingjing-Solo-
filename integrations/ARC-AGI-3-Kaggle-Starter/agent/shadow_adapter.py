# -*- coding: utf-8 -*-
"""专用求解器（shadow solver）适配器：方言归一 + 目录解析。

职责边界
--------
各 `arc_shadow_*` 模块是**别人机器上的专用解法工厂**，返回的动作方言互不一致：
dc22 用 `("m", 3)` / `("c", 13, 30)`，sp80 用 `("p", 5)`（放水宏），另一些用 int、
`(6, x, y)`、`{"a":6,"x":..,"y":..}`。
`my_agent.py` 原先假定"int 就是 int、二元组就是点击"，于是 dc22 的 `("m", 3)`
被当成点击坐标（后面 `int('m')` 炸），`("c", x, y)` 直接撞 `int(tuple)` 的 TypeError，
整段 R3 尝试连同 IDDFS 兜底一起被外层 except 吞掉（2026-10-11 `_bench_main_root.log:130`）。

所以本模块只做一件事：**在边界上把方言归一成一种规范格式**，
让决策层不必知道有几个方言：

    规范格式 = int（键位动作 id）  或  (6, x, y)（点击）

方言不认识时整条解法判为不可用（返回 None），调用方回落到通用搜索——
绝不允许"半条解法被执行"，那是把乱点击当进度。

目录解析
--------
求解器模块所在目录**不再写死任何成员本机的绝对路径**（团队规范 §3.2：
"不能把 Windows 路径、私有绝对路径或某个成员本地 checkout 写入公共跑测入口"）。
只认环境变量 `SHADOW_SOLVER_DIR`；没配就等于"本机没有专用求解器"，
`available()`/`solve()` 安静返回不可用，走通用搜索。
"""
import os
import sys
import threading
from pathlib import Path
from typing import Any, List, Optional

#: 专用求解器目录的环境变量名（唯一入口，无默认私有路径）
SOLVER_DIR_ENV = "SHADOW_SOLVER_DIR"

# 游戏前缀 → shadow 模块名
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

# 已加载模块与并发锁
_modules: dict = {}
_locks: dict = {}
#: 只打一次的状态日志
_state_logged: set = set()


#: 历史别名，保留给外部引用。**solver_dir 已不再依赖它**：写死的 ``parents[3]``
#: 只在本地 checkout 的目录形状下成立（见 :func:`_shadow_dir_candidates`）。
_DEFAULT_SHADOW_DIR = Path(__file__).resolve().parents[3] / "lingjing_solo" / "planning"


def _shadow_dir_candidates() -> list[Path]:
    """默认候选目录，按优先级排列。

    上线时本文件被拷到 ``ARC-AGI-3-Agents/agents/templates/``，而 ``lingjing_solo``
    是 ``agents/`` 的**兄弟**目录 —— 包落在 ``parents[2]`` 而不是 ``parents[3]``。
    旧写法在 Kaggle 上解析到 ``/kaggle/working``（不存在），于是专用求解器整体
    静默失效；本地却因为恰好深一层而一直正常（2026-10-14 定位）。
    """
    here = Path(__file__).resolve()
    candidates: list[Path] = []
    for up in (2, 3):
        parents = here.parents
        if len(parents) > up:
            candidates.append(parents[up] / "lingjing_solo" / "planning")
    # 求解器与适配器同目录铺放（dataset 直接铺平的形态）
    candidates.append(here.parent)
    return candidates


def solver_dir() -> Optional[Path]:
    """解析专用求解器目录。

    优先使用环境变量 ``SHADOW_SOLVER_DIR``（比赛里由 ``.env`` 显式写入）；
    未配置时依次尝试 :func:`_shadow_dir_candidates`；
    都不存在时返回 None（= 不可用，不报错）。
    """
    raw = os.environ.get(SOLVER_DIR_ENV, "").strip().strip('"').strip("'")
    if raw:
        try:
            path = Path(raw).expanduser()
            if path.is_dir():
                return path
        except (OSError, RuntimeError):
            pass
    for candidate in _shadow_dir_candidates():
        if candidate.is_dir():
            return candidate
    return None


def _log_state(gid: str, reason: str) -> None:
    key = f"{gid}:{reason}"
    if key in _state_logged:
        return
    _state_logged.add(key)
    print(f"[shadow_adapter] {gid}: {reason}", flush=True)


def _load_shadow(gid: str):
    """按配置的目录加载 shadow 模块；目录没配就判定不可用。"""
    if gid in _modules:
        return _modules[gid]
    mod_name = _SHADOW_MODULES.get(gid)
    if not mod_name:
        return None
    root = solver_dir()
    if root is None:
        _log_state(gid, f"{SOLVER_DIR_ENV} 未配置或目录不存在 → 无专用求解器，回落通用搜索")
        return None
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        mod = __import__(mod_name)
        _modules[gid] = mod
        _locks[gid] = threading.RLock()
        return mod
    except Exception as exc:  # noqa: BLE001 — 第三方本地模块，加载失败是常态不是异常
        _log_state(gid, f"加载 {mod_name} 失败: {type(exc).__name__}: {exc}")
        return None


# ── 方言归一 ─────────────────────────────────────────────────

def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_action(item: Any) -> Optional[Any]:
    """一个动作 → 规范格式 `int` 或 `(6, x, y)`；认不出就 None（整条解法作废）。"""
    if isinstance(item, bool):
        return None
    if isinstance(item, (int, float)):
        value = _as_int(item)
        return value if value is not None and 1 <= value <= 7 else None
    if isinstance(item, dict):
        act = _as_int(item.get("a", item.get("action", item.get("id"))))
        if act is None:
            return None
        if act == 6:
            x = _as_int(item.get("x"))
            y = _as_int(item.get("y"))
            return (6, x, y) if x is not None and y is not None else None
        return act if 1 <= act <= 7 else None
    if isinstance(item, (tuple, list)):
        if not item:
            return None
        head = item[0]
        # 方言 A：("m"/"k"/"key", 动作号) —— 键位宏
        if isinstance(head, str) and head.strip().lower() in ("m", "k", "key", "move") and len(item) == 2:
            return normalize_action(item[1])
        # 方言 B：("c"/"click", x, y) —— 点击
        if isinstance(head, str) and head.strip().lower() in ("c", "click") and len(item) == 3:
            x = _as_int(item[1])
            y = _as_int(item[2])
            return (6, x, y) if x is not None and y is not None else None
        # 方言 C：("i", 动作号) —— 键盘动作（arc_shadow_cn04 格式）
        if isinstance(head, str) and head.strip().lower() == "i" and len(item) == 2:
            val = _as_int(item[1])
            return val if val is not None and 1 <= val <= 7 else None
        # 方言 D：("r",) —— 旋转（arc_shadow_cn04 格式）
        if isinstance(head, str) and head.strip().lower() == "r":
            return 5  # ACTION5 = rotate
        # 方言 E：("p", n) —— "放水/pour" 宏（arc_shadow_sp80 格式）
        # sp80 的 solve_level 以 ``list(path) + [("p", 5)]`` 收尾（见
        # ``lingjing_solo/planning/arc_shadow_sp80.py`` 的 return 处，引擎侧就是 ACTION5）。
        # 缺这一分支时整条已算出的解法会被判成"方言不认识"而作废 →
        # 2026-10-13 实测日志：``sp80: 解法方言不认识（首元素 tuple=('m', 4)，共 4 步）``。
        if isinstance(head, str) and head.strip().lower() in ("p", "pour"):
            if len(item) == 1:
                return 5
            if len(item) == 2:
                val = _as_int(item[1])
                return val if val is not None and 1 <= val <= 7 else None
            return None
        if isinstance(head, str):
            # 其余字符串方言一律不猜
            return None
        nums = [_as_int(v) for v in item]
        if any(v is None for v in nums):
            return None
        if len(nums) == 1:
            return nums[0] if 1 <= nums[0] <= 7 else None
        if len(nums) == 2:
            # 历史约定：二元纯数字 = 点击坐标
            return (6, nums[0], nums[1])
        if len(nums) == 3:
            # 规范格式里三元只允许 (6, x, y)；带别的头编号一律不猜
            return (6, nums[1], nums[2]) if nums[0] == 6 else None
        return None
    return None


def normalize_path(path: Any) -> Optional[List[Any]]:
    """整条解法归一；任一元素不可识别 → 整条判为不可用（返回 None）。

    `[]` 是合法值（求解器说"这关已经过了"），原样透传给调用方处理。
    """
    if path is None or isinstance(path, (str, bytes, dict)) or not hasattr(path, "__iter__"):
        return None
    out: List[Any] = []
    for item in path:
        norm = normalize_action(item)
        if norm is None:
            return None
        out.append(norm)
    return out


# ── 对外接口 ─────────────────────────────────────────────────

def available(gid: str) -> bool:
    """这一局是否有可用的专用求解器。"""
    return gid in _SHADOW_MODULES and _load_shadow(gid) is not None


def solve(gid: str, game, t_limit: float = 60.0, env=None, **kwargs) -> Optional[List]:
    """求解指定游戏的当前关，返回**规范格式**动作序列。

    Args:
        gid: 游戏 ID 前缀（如 "g50t"、"dc22"）
        game: 引擎 game 对象（替换 shadow 模块的全局 _game；仅离线可用）
        t_limit: 超时秒数
        env: 引擎 env wrapper（替换 shadow 模块的全局 _env；仅离线可用）
        **kwargs: 传给 solve_level 的额外参数

    Returns:
        list：元素为 int 或 (6, x, y)；已通关为 []；
        不可用 / 失败 / 方言不认识 → None（调用方回落通用搜索）。
    """
    mod = _load_shadow(gid)
    if not mod:
        return None

    lock = _locks.get(gid)
    if lock:
        lock.acquire()
    try:
        # 替换 shadow 模块的全局 _game
        if hasattr(mod, "_game"):
            mod._game = game
        if hasattr(mod, "_env"):
            # 优先使用显式传入的 env wrapper；fallback 到 game._env（兼容旧调用方）
            try:
                if env is not None:
                    mod._env = env
                elif hasattr(game, "_env"):
                    mod._env = game._env
                else:
                    mod._env = None
            except Exception:  # noqa: BLE001
                pass

        solve_fn = getattr(mod, "solve_level_enhanced", None) or getattr(mod, "solve_level", None)
        if not solve_fn:
            _log_state(gid, "模块里没有 solve_level / solve_level_enhanced")
            return None

        raw = solve_fn(t_limit=t_limit, **kwargs)
        norm = normalize_path(raw)
        if raw and norm is None:
            _log_state(
                gid,
                f"解法方言不认识（首元素 {type(raw[0]).__name__}={raw[0]!r}，共 {len(raw)} 步）"
                f"→ 整条作废，回落通用搜索",
            )
        return norm
    except Exception as exc:  # noqa: BLE001
        _log_state(gid, f"solve 失败: {type(exc).__name__}: {exc}")
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