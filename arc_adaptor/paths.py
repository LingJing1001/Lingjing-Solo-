"""公共跑测入口的路径解析：不允许把私有 checkout 绝对路径写进脚本。

约定（见 docs/最小可运行闭环_团队统一集成与跑测验证方法.md §3.2、§4.3）：
    环境变量显式指定 > 由本文件位置推导的项目内相对路径 > 同级官方 ARC checkout。

解析不到游戏环境时 **fail-closed 抛错**，绝不静默回退到某个成员的本地目录：
以前 runner 里写死 ``F:/pro/...``，换一台机器（WSL 成员是
``/srv/agent-platform/projects/...``）就直接 ImportError，而且失败点离真正原因很远。

用法：
    from paths import add_to_sys_path, environments_dir, state_dir
    add_to_sys_path()
    arcade = Arcade(environments_dir=environments_dir("ls20"), ...)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# 本文件位于 <repo>/arc_adaptor/paths.py
ADAPTOR_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = ADAPTOR_DIR.parent

# 官方 ARC checkout 的默认推导：与 Lingjing 仓库同级、同名。
# 成员各自把两个仓库 clone 在同一父目录下即可，无需改代码。
DEFAULT_ARC_HOME_NAME = "ARC-AGI-3-Agents"


def _clean(raw: str) -> Path | None:
    value = raw.strip()
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def _candidates(name: str) -> list[Path]:
    """按优先级给出候选根目录；环境变量在最前。"""
    out: list[Path] = []
    for var in ("LINGJING_ENVIRONMENTS_DIR", "ARC_ENVIRONMENTS_DIR"):
        env = os.environ.get(var, "")
        path = _clean(env)
        if path is not None:
            out.append(path)
            break
    if name:
        # LINGJING_ARC_HOME 指向官方 checkout 根；其 environment_files/ 才是环境目录
        arc_home = _clean(os.environ.get("LINGJING_ARC_HOME", ""))
        if arc_home is not None:
            out.append(arc_home / "environment_files")
    out.append(PROJECT_ROOT / "environment_files")
    out.append(PROJECT_ROOT.parent / DEFAULT_ARC_HOME_NAME / "environment_files")
    return out


def add_to_sys_path() -> None:
    """把项目根与 arc_adaptor 挂进 sys.path（替代写死的 sys.path.insert）。"""
    for path in (str(ADAPTOR_DIR), str(PROJECT_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)


def environments_dir(game_prefix: str = "") -> str:
    """返回能真正加载到该游戏的 environments_dir。

    ``game_prefix`` 用游戏的短 id（如 ``ls20``/``ar25``）；留空则只要求目录存在。
    注意：``environment_files/<prefix>/...`` 之外还要有实际游戏包，所以这里校验
    的是「目录下存在以该前缀开头的子目录」，避免拿到一个空目录后在 Arcade 内部
    才炸出难以定位的报错。
    """
    prefix = game_prefix.lower()
    tried: list[str] = []
    for candidate in _candidates(prefix):
        tried.append(str(candidate))
        if not candidate.is_dir():
            continue
        if not prefix:
            return str(candidate)
        has_game = any(
            child.is_dir() and child.name.lower().startswith(prefix)
            for child in candidate.iterdir()
        )
        if has_game:
            return str(candidate)
    raise LookupError(
        f"找不到包含游戏 '{game_prefix}' 的 environment_files。已尝试:\n  "
        + "\n  ".join(tried)
        + "\n请设 LINGJING_ENVIRONMENTS_DIR=<环境目录> 或 "
          "LINGJING_ARC_HOME=<官方 ARC-AGI-3-Agents checkout 根>"
    )


def state_dir() -> Path:
    """跑测产物目录：默认 <repo>/state，可用 LINGJING_STATE_DIR 覆盖。"""
    path = _clean(os.environ.get("LINGJING_STATE_DIR", "")) or (PROJECT_ROOT / "state")
    path.mkdir(parents=True, exist_ok=True)
    return path


def git_output(*args: str, cwd: Path | None = None) -> str:
    """取 git 元数据；拿不到返回 'unknown' 而不是抛错。

    evidence manifest 要求 branch/commit（§5.1），但缺 git 时不该让整次跑测白跑，
    所以这里降级为显式 'unknown'，报告里一眼能看出证据不完整。
    """
    import subprocess

    try:
        return subprocess.run(
            ["git", *args], cwd=str(cwd or PROJECT_ROOT), check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"
