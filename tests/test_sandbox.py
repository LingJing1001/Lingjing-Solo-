"""受限 REPL 沙箱：白名单、超时、步数上限、动作捕获。"""

from __future__ import annotations

import pytest

from lingjing_solo.harness.actions import Action, make_action, normalize_action_name
from lingjing_solo.harness.sandbox import (
    ALLOWED_IMPORTS,
    SANDBOX_BUILTINS,
    Sandbox,
    SandboxContext,
)


def run(code, **ctx):
    return Sandbox().run(code, SandboxContext(**ctx))


# --------------------------------------------------------------------------- #
# 动作契约
# --------------------------------------------------------------------------- #


def test_normalize_action_names_and_aliases():
    assert normalize_action_name("up") == "UP"
    assert normalize_action_name(" Down ") == "DOWN"
    assert normalize_action_name("mouse") == "CLICK"
    assert normalize_action_name("undo") == "SPACE"  # 不提供 Undo，降级不静默
    with pytest.raises(ValueError):
        normalize_action_name("teleport")
    with pytest.raises(ValueError):
        normalize_action_name(None)


def test_action_requires_coords_for_click():
    with pytest.raises(ValueError):
        Action("CLICK")
    a = Action("CLICK", 3, 4)
    assert (a.row, a.col) == (3, 4)
    assert str(a) == "CLICK(3,4)"


def test_non_click_action_drops_coords():
    a = Action("UP", 1, 2)
    assert a.row is None and a.col is None


def test_make_action_accepts_numeric_strings():
    a = make_action("click", "7", "8")
    assert (a.row, a.col) == (7, 8)


# --------------------------------------------------------------------------- #
# 正常路径
# --------------------------------------------------------------------------- #


def test_simple_action():
    r = run("action('UP')")
    assert r.ok and r.action.kind == "UP"


def test_last_action_wins():
    r = run("action('UP')\naction('LEFT')")
    assert r.ok and r.action.kind == "LEFT"


def test_log_capture():
    r = run("log('hello')\nlog(42)\naction('SPACE')")
    assert r.logs == ["hello", "42"]


def test_memory_is_shared_and_persisted():
    mem = {"count": 0}
    r = run("memory['count'] += 1\naction('SPACE')", memory=mem)
    assert r.ok and mem["count"] == 1
    assert r.memory is mem


def test_objects_and_history_visible():
    r = run(
        "log(len(objects))\nlog(len(history))\naction('SPACE')",
        objects=[{"id": 0}], history=[{"turn": 0}],
    )
    assert r.logs == ["1", "1"]


def test_allowed_import_works():
    r = run("import math\nlog(math.gcd(12, 18))\naction('SPACE')")
    assert r.ok and r.logs == ["6"]


def test_import_from_allowed_module():
    r = run("from collections import Counter\nlog(Counter([1,1,2]).most_common(1)[0][0])\naction('SPACE')")
    assert r.ok and r.logs == ["1"]


# --------------------------------------------------------------------------- #
# 拦截
# --------------------------------------------------------------------------- #


def test_blocked_import_os():
    r = run("import os\naction('SPACE')")
    assert not r.ok and r.error_kind == "runtime" and "not allowed" in r.error


def test_blocked_import_socket():
    r = run("import socket\naction('SPACE')")
    assert not r.ok and "not allowed" in r.error


def test_blocked_import_subprocess():
    r = run("import subprocess\naction('SPACE')")
    assert not r.ok


def test_open_is_not_available():
    r = run("open('/etc/passwd')\naction('SPACE')")
    assert not r.ok and r.error_kind == "runtime"
    assert "open" in r.error


def test_eval_and_exec_not_available():
    for expr in ("eval('1+1')", "exec('x=1')", "compile('1','<s>','eval')"):
        r = run(f"{expr}\naction('SPACE')")
        assert not r.ok, expr


def test_dunder_import_not_exposed():
    assert "__import__" not in SANDBOX_BUILTINS


def test_no_allowed_dangerous_module_in_whitelist():
    for bad in ("os", "sys", "socket", "subprocess", "pathlib", "shutil",
                "pickle", "ctypes", "threading", "multiprocessing", "importlib"):
        assert bad not in ALLOWED_IMPORTS


# --------------------------------------------------------------------------- #
# 失败归档
# --------------------------------------------------------------------------- #


def test_syntax_error():
    r = run("action('UP'")
    assert not r.ok and r.error_kind == "syntax"


def test_runtime_error():
    r = run("1/0\naction('SPACE')")
    assert not r.ok and r.error_kind == "runtime" and "ZeroDivisionError" in r.error


def test_click_without_coords_is_runtime_error():
    r = run("action('CLICK')")
    assert not r.ok and "CLICK" in r.error


def test_unknown_action_name_is_runtime_error():
    r = run("action('TELEPORT')")
    assert not r.ok and "unknown action" in r.error


def test_no_action_call():
    r = run("x = 1 + 1")
    assert not r.ok and r.error_kind == "no_action"


def test_user_except_exception_cannot_swallow_timeout():
    """超时/步数上限用 BaseException 子类抛出，`except Exception` 吞不掉。"""
    r = Sandbox(timeout_s=0.05).run(
        "try:\n"
        "    while True:\n"
        "        pass\n"
        "except Exception:\n"
        "    pass\n"
        "action('SPACE')\n"
    )
    assert not r.ok and r.error_kind == "timeout"


def test_step_limit():
    r = Sandbox(timeout_s=30.0, max_steps=5_000).run("while True:\n    pass\n")
    assert not r.ok and r.error_kind == "step_limit"


def test_too_long_code():
    r = Sandbox(max_code_chars=50).run("x=1\n" * 100)
    assert not r.ok and "too long" in r.error


def test_none_code():
    r = Sandbox().run(None)
    assert not r.ok and r.error_kind == "no_action"


# --------------------------------------------------------------------------- #
# 状态隔离
# --------------------------------------------------------------------------- #


def test_previous_run_variable_not_visible():
    sb = Sandbox()
    sb.run("leak = 123\naction('SPACE')")
    r = sb.run("action('SPACE')\nlog('ok')")
    assert r.ok and r.logs == ["ok"]


def test_global_name_leak_produces_nameerror():
    sb = Sandbox()
    sb.run("leak = 123\naction('SPACE')")
    r = sb.run("log(leak)\naction('SPACE')")
    assert not r.ok and "NameError" in r.error


def test_memory_carries_across_runs():
    sb = Sandbox()
    mem = {}
    sb.run("memory['n'] = memory.get('n', 0) + 1\naction('SPACE')", SandboxContext(memory=mem))
    sb.run("memory['n'] = memory.get('n', 0) + 1\naction('SPACE')", SandboxContext(memory=mem))
    assert mem["n"] == 2
