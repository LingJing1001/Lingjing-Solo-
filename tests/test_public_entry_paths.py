"""公共跑测入口不得含机器相关绝对路径（团队规范 §3.2 兼容要求第三条的机械闸门）。

为什么写成测试而不是靠 review：这条规则此前被破过（runner 里写死某个成员的盘符路径，
换机器直接 ImportError，失败点离原因很远），而它恰好是最容易在赶工时被顺手复制的那种
字符串。`paths.py` 已经把解析统一成"环境变量 > 由 __file__ 推导 > 同级官方 checkout"，
所以正常情况下这个测试永远应该是绿的；它红了就说明有人又开始硬编码。

只扫跑测入口（`arc_adaptor/**/*.py|sh` 与 `lingjing_solo/**/*.py`），不扫 docs：文档里
举例说明"哪里不能写"必然要出现路径字样，把它算成违规只会逼人把规则注释掉。
"""
from __future__ import annotations

import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

SCAN_GLOBS = ("arc_adaptor/**/*.py", "arc_adaptor/**/*.sh", "lingjing_solo/**/*.py")
SKIP_PARTS = {"__pycache__", ".venv", "build", "dist"}

#: 三类"只在某台机器上成立"的写法。`/srv/agent-platform` 这类团队共享挂载点不算，
#: 所以模式故意收窄到盘符 / 成员家目录 / WSL 挂载。
FORBIDDEN: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("Windows 盘符绝对路径", re.compile(r"[A-Za-z]:[\\/]{1,2}(?:pro|Users|home|srv)\b", re.I)),
    ("POSIX 家目录", re.compile(r"/home/[^/\s\"']+/")),
    ("WSL 挂载路径", re.compile(r"/mnt/wsl", re.I)),
    ("仓库备份目录", re.compile(r"Lingjing-Solo[-_]backup", re.I)),
)


def _scan_targets() -> list[pathlib.Path]:
    seen: set[pathlib.Path] = set()
    for pattern in SCAN_GLOBS:
        for path in REPO_ROOT.glob(pattern):
            if SKIP_PARTS & set(path.parts) or path.is_dir():
                continue
            seen.add(path)
    return sorted(seen)


def test_scan_targets_is_not_empty():
    """目标集为空就等于闸门形同虚设——先证明它真的扫到了东西。"""
    assert len(_scan_targets()) > 20


def test_public_run_entries_have_no_machine_specific_paths():
    offenders = []
    for path in _scan_targets():
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_no, line in enumerate(text.splitlines(), 1):
            for label, pattern in FORBIDDEN:
                if pattern.search(line):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{line_no} [{label}] {line.strip()[:90]}")
    assert not offenders, "公共跑测入口含机器相关路径（改走 arc_adaptor/paths.py 解析）:\n" + "\n".join(offenders)


def test_runners_resolve_paths_through_paths_module():
    """禁硬编码的同时，还要正面确认入口确实走 paths.py：只查 `sys.path.insert(<字面量>)`。"""
    offenders = []
    literal = re.compile(r"""sys\.path\.insert\(\d+,\s*["']""")
    for path in REPO_ROOT.glob("arc_adaptor/**/*.py"):
        if SKIP_PARTS & set(path.parts):
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if literal.search(line):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{line_no}")
    assert not offenders, "不要用字面量往 sys.path 里塞目录，改用 paths.add_to_sys_path():\n" + "\n".join(offenders)
