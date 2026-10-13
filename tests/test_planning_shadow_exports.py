"""`lingjing_solo.planning` 的 shadow 符号导出测试（#13）。

**缺陷（2026-10-12 由跑分日志坐实）**：打分入口取符号的写法是
`from lingjing_solo.planning import shadow_enabled, shadow_games`
（实跑副本 `score_loop/mirror_aopoff/.../agent/my_agent.py:344`），
再 `from lingjing_solo.planning import R3ShadowProbe`（`:349`）。
包 `__init__.py` 没有这三个名字 ⇒ `ImportError: cannot import name
'shadow_enabled' from 'lingjing_solo.planning'`，被调用方的兜底 except 吞成
`r3-shadow unavailable` ⇒ 旁路探针在打分路径上恒 off，#9 那条证据链修得再对
也一行都不执行。

实测证据：`score_loop/logs/round_10{2,8}_*.log` 第 6 行都是那句 ImportError
（同一份 `lingjing_solo/planning/__init__.py`）。

本文件覆盖：
    1. 打分入口那两行 import 原样可用（含 `R3ShadowProbe` 构造）
    2. 默认口径：不带环境变量 = 关；白名单默认只有 dc22
    3. 惰性：`import lingjing_solo.planning` 不把 `r3_shadow` 拖进 sys.modules
       （包的开销与失败面维持原样）
    4. `__all__` / `dir()` 认账；未知属性仍抛 AttributeError
    5. 白名单外的局：`observe()` 空转、不建内核、不落产物

只读被测代码，不碰打分入口、不写任何产物。
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

SHADOW_EXPORTS = ("shadow_enabled", "shadow_games", "R3ShadowProbe")


# ── 1. 打分入口的取符号写法 ─────────────────────────────────────

def test_live_import_form_resolves():
    """逐字复刻 my_agent.py:344 的那一行。"""
    from lingjing_solo.planning import shadow_enabled, shadow_games

    assert callable(shadow_enabled)
    assert isinstance(shadow_games(), set)


def test_probe_class_is_exported_and_constructible():
    """复刻 :349：拿到类并实例化，构造本身必须零副作用。"""
    from lingjing_solo.planning import R3ShadowProbe

    probe = R3ShadowProbe()
    assert probe._disabled is False
    assert probe._files == {}          # 没开任何产物句柄


def test_star_import_exposes_shadow_symbols():
    namespace: dict = {}
    exec("from lingjing_solo.planning import *", namespace)  # noqa: S102
    for name in SHADOW_EXPORTS:
        assert name in namespace, f"`{name}` 不在星号导入结果里（__all__ 没认账）"


def test_dir_lists_shadow_symbols():
    import lingjing_solo.planning as pkg

    for name in SHADOW_EXPORTS:
        assert name in dir(pkg), f"dir() 未列出 {name}"


def test_unknown_attribute_still_raises():
    import lingjing_solo.planning as pkg

    with pytest.raises(AttributeError):
        pkg.this_symbol_does_not_exist


# ── 2. 默认口径：关掉、白名单只有 dc22 ──────────────────────────

def test_off_by_default(monkeypatch):
    from lingjing_solo.planning import shadow_enabled

    monkeypatch.delenv("R3_SHADOW", raising=False)
    assert shadow_enabled() is False, "默认必须是关：Kaggle 线上不带这个变量"

    monkeypatch.setenv("R3_SHADOW", "0")
    assert shadow_enabled() is False

    monkeypatch.setenv("R3_SHADOW", "1")
    assert shadow_enabled() is True


def test_whitelist_defaults_to_dc22_only(monkeypatch):
    from lingjing_solo.planning import shadow_games

    monkeypatch.delenv("R3_SHADOW_GAMES", raising=False)
    assert shadow_games() == {"dc22"}

    monkeypatch.setenv("R3_SHADOW_GAMES", "dc22, cn04 ,,")
    assert shadow_games() == {"dc22", "cn04"}, "空白项要滤掉、项要 strip"


# ── 3. 惰性：取符号之前不加载 r3_shadow ─────────────────────────

def _run_clean(code: str) -> str:
    """在干净解释器里跑，避免同进程别的测试已经把模块导进去了。"""
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env.pop("R3_SHADOW", None)
    env.pop("R3_SHADOW_GAMES", None)
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True, text=True, timeout=180, cwd=str(REPO_ROOT), env=env,
    )
    assert proc.returncode == 0, proc.stderr[-400:]
    return proc.stdout.strip()


def test_package_import_does_not_load_r3_shadow():
    out = _run_clean("""
        import sys
        import lingjing_solo.planning
        print("loaded" if "lingjing_solo.planning.r3_shadow" in sys.modules else "lazy")
    """)
    assert out == "lazy", "包导入就把 r3_shadow 拖进来了：#13 应该走惰性导出"


def test_accessing_symbol_loads_it_on_demand():
    out = _run_clean("""
        import sys
        import lingjing_solo.planning as pkg
        before = "lingjing_solo.planning.r3_shadow" in sys.modules
        fn = pkg.shadow_enabled
        after = "lingjing_solo.planning.r3_shadow" in sys.modules
        print(f"{before}|{after}|{fn.__name__}")
    """)
    assert out == "False|True|shadow_enabled"


# ── 5. 白名单外的局：observe 空转，不建内核、不落产物 ───────────

class _Frame:
    def __init__(self):
        self.grid = [[0, 0]]
        self.state = "NOT_FINISHED"
        self.levels = 0
        self.action_input = None


def test_observe_is_noop_for_unsupported_games(tmp_path, monkeypatch):
    from lingjing_solo.planning import R3ShadowProbe, shadow_games

    out_dir = tmp_path / "r3_shadow"
    monkeypatch.setenv("R3_SHADOW", "1")
    monkeypatch.setenv("R3_SHADOW_DIR", str(out_dir))
    monkeypatch.setenv("R3_SHADOW_GAMES", "not_a_real_game")

    probe = R3ShadowProbe()
    assert "dc22" not in shadow_games()
    assert probe.observe(_Frame(), ["ACTION1"], gid="dc22", levels=0) is None
    assert probe._kernels == {}, "白名单外的局不许建内核"
    assert probe._files == {}, "白名单外的局不许开产物文件"
    assert not out_dir.exists() or list(out_dir.iterdir()) == []


def test_observe_never_returns_an_action(monkeypatch):
    """硬规矩 2：旁路恒返回 None，调用方拿不到动作。"""
    from lingjing_solo.planning import R3ShadowProbe

    monkeypatch.delenv("R3_SHADOW", raising=False)   # 未开启 ⇒ 直接空转
    probe = R3ShadowProbe()
    assert probe.observe(_Frame(), ["ACTION1"], gid="dc22", levels=0) is None
