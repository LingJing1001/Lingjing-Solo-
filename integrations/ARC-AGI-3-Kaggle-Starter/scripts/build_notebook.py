"""Splice the current `agent/my_agent.py` into `notebooks/submission.ipynb`.

The notebook follows the exact pattern used by Kaggle's official sample
("ARC3 Sample Submission - Stochastic Goose"):

  Cell 1: install the `arc-agi` wheel from the offline competition dataset.
  Cell 2: write `my_agent.py` to /kaggle/working/ — its body is THIS file.
  Cell 3: if running inside the Kaggle competition rerun, wait for the
          gateway sidecar, copy the framework into /kaggle/working/, register
          MyAgent, and run `python main.py --agent myagent`.
  Cell 4: otherwise (during commit / save-and-run-all), write a dummy
          submission.parquet so Kaggle accepts the commit.

You don't normally need to call this directly — `make submit` runs it for you.
"""
from __future__ import annotations

import ast
import base64
import io
import json
import os
import sys
import tokenize
from pathlib import Path
from textwrap import dedent

# Kaggle's SaveKernel rejects any kernel whose source is >= 1 MB, and it measures
# the JSON-normalised (ASCII-escaped) form, where each CJK character costs 6
# bytes. The agent + lingjing_solo sources carry a lot of Chinese comments, so
# the notebook used to be rejected at ~1.06 MB. Comments are the only fat we can
# cut without touching behaviour, and every cut is AST-verified below.
_KERNEL_SOURCE_LIMIT = 1_048_576


def _strip_comments(src: str) -> str:
    """Drop ``#`` comments, keeping one output line per input line.

    Truncates each commented line at the comment's column instead of using
    ``tokenize.untokenize``, so line numbers in the shipped source stay identical
    to the local files (``file:line`` references keep meaning the same thing).
    Returns ``src`` unchanged on any tokenize error.
    """
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
        return src
    lines = src.splitlines(keepends=True)
    for tok in reversed(toks):
        if tok.type != tokenize.COMMENT:
            continue
        row, col = tok.start          # 1-based row, 0-based column
        if row > len(lines):
            continue
        head = lines[row - 1][:col].rstrip()
        nl = "\n" if lines[row - 1].endswith("\n") else ""
        lines[row - 1] = head + nl
    return "".join(lines)


def _ast_equal(a: str, b: str) -> bool:
    """True when two sources parse to identical ASTs (comments invisible there)."""
    try:
        return ast.dump(ast.parse(a)) == ast.dump(ast.parse(b))
    except SyntaxError:
        return False


def _docstring_spans(tree: ast.AST) -> list[tuple[int, int, int, int, int]]:
    """所有 docstring 的位置：``(lineno, end_lineno, col, end_col, body_stmt_count)``。

    只收**独占一行的** docstring（前面没有代码）；``def f(): "doc"`` 这种
    同行写法整行删掉会连函数定义一起毁掉，所以跳过不压。
    """
    out: list[tuple[int, int, int, int, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if not (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            continue
        value = first.value
        if not (value.lineno and value.end_lineno and value.end_col_offset is not None):
            continue
        out.append((value.lineno, value.end_lineno, value.col_offset, value.end_col_offset, len(body)))
    return out


def _blank_docstrings(src: str) -> str:
    """把 docstring 的字符清掉，但**一行都不删**（行号必须保持）。

    行号之所以必须保持：agent 里到处是 ``file:line`` 形式的引用与断言，
    而比赛日志只给行号。省字符不省行 —— 空行里补 ``pass`` 只发生在
    "docstring 就是整个函数体"的情况（否则清空会变成 IndentationError）。
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src
    lines = src.splitlines(keepends=True)
    targets = _docstring_spans(tree)
    for lo, hi, col, end_col, body_len in targets:
        if lo < 1 or lo > len(lines) or hi > len(lines):
            continue
        head = lines[lo - 1][:col]
        if head.strip():                       # 同行还有代码：跳过，别毁掉定义
            continue
        tail = lines[hi - 1][end_col:]
        if hi == lo and tail.strip():
            continue
        if body_len == 1:
            lines[lo - 1] = head + "pass\n"
        else:
            lines[lo - 1] = "\n" if lines[lo - 1].endswith("\n") else ""
        for i in range(lo, hi):
            lines[i] = "\n" if lines[i].endswith("\n") else ""
    return "".join(lines)


def _ast_dump_sans_docstrings(src: str) -> str:
    """把 docstring 从 AST 里摘掉再 dump —— 用来证明压缩只动了 docstring。"""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if not (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            continue
        if len(body) == 1:
            body[0] = ast.Pass()              # 与 _blank_docstrings 的 pass 对应
        else:
            del body[0]
    return ast.dump(tree)


# Filled while splicing: bytes saved by comment/docstring stripping, and file count.
_STRIP_STATS = {"bytes": 0, "files": 0}


def _embed_source(rel: str, source: str) -> str:
    """压掉一个内嵌模块的注释与 docstring，任一处语义变化即拒绝发货。

    为什么要压到这一步：Kaggle 的 1 MiB 上限是**按 JSON-ASCII 转义后**算的，
    一个中文字符要 6 字节。而收录集合改成"按真实闭包推导"之后（原来手写的
    5 个 roots 漏了 13 个模块），payload 直接顶穿上限 —— 靠继续删模块腾空间
    就是回到"静默降级"的老路，所以只能从注释和 docstring 里省。
    """
    compact = _blank_docstrings(_strip_comments(source))
    if compact == source:
        return source
    try:
        same_semantics = _ast_dump_sans_docstrings(source) == _ast_dump_sans_docstrings(compact)
    except SyntaxError:
        same_semantics = False
    if not same_semantics:
        raise SystemExit(f"[build_notebook] ABORT: 压缩改变了 {rel} 的语义")
    if len(compact.splitlines()) != len(source.splitlines()):
        raise SystemExit(f"[build_notebook] ABORT: 压缩移动了 {rel} 的行号")
    _STRIP_STATS["bytes"] += len(source) - len(compact)
    _STRIP_STATS["files"] += 1
    return compact


# ─────────────────────────────────────────────────────────────────────────────
# CHANGE THIS ONE LINE TO PICK YOUR KAGGLE ACCELERATOR
# Options:
#   "cpu"      — no GPU. Good for the random starter or any non-ML agent.
#   "t4"       — Nvidia T4 ×2 (default; matches Kaggle's sample submission).
#   "p100"     — Nvidia P100 (single big-memory GPU).
#   "rtx6000"  — Nvidia RTX 6000 (g4-standard-48). ARC-AGI-3 exclusive,
#                burns GPU quota faster — use only when you're confident.
# ─────────────────────────────────────────────────────────────────────────────
ACCELERATOR = "cpu"

# Internal mapping; don't edit unless Kaggle adds new options.
_ACCELERATORS = {
    "cpu":     {"name": "none",            "gpu": False},
    "t4":      {"name": "nvidiaTeslaT4",   "gpu": True},
    "p100":    {"name": "nvidiaTeslaP100", "gpu": True},
    "rtx6000": {"name": "nvidiaRtx6000",   "gpu": True},
}

ROOT = Path(__file__).resolve().parents[1]
AGENT_SRC = ROOT / "agent" / "my_agent.py"
LINGJING_SRC = Path(
    os.environ.get("LINGJING_SRC", str(ROOT.parents[1] / "lingjing_solo"))
).expanduser().resolve()
NOTEBOOK_PATH = ROOT / "notebooks" / "submission.ipynb"

# --------------------------------------------------------------------------- #
# 打包完整性
#
# 比赛 `enable_internet=false`，任何"运行期才发现缺文件"都是不可修的。而 agent
# 把可选能力都写成 `try: import X / except: 降级`，所以**缺文件不会崩**——
# 只会让提交悄悄少跑一半能力（本地验证过的分和线上分对不上，就是这样来的）。
#
# 历史上收录集合是手写的 5 个 roots，随代码增长失准，实测漏了：
#   lingjing_solo.engine_solver / closed_book.{ar25,tu93}_solver /
#   neural.aop_controller / perception.stability / exploration.ft09_solver /
#   planning.search.generic_shadow，以及 lingjing_solo 之外的整个 arc_adaptor。
# 现在改为**由 agent 的真实导入闭包推导**，并用同一套闭包做收尾校验。
# 审计与校验的实现在 lingjing_solo/harness/offline.py（纯 stdlib，可独立加载）。
# --------------------------------------------------------------------------- #

#: `lingjing_solo/` 之外、但 agent 会 import 的包的**入口模块**。
#: 只取其闭包（`arc_adaptor.click_heatmap` 闭包 = 1 个文件 8.7 KB），
#: 不整包搬（整包 45 文件 463 KB，会把 notebook 顶穿 1 MiB 上限）。
EXTRA_ROOTS: tuple[str, ...] = ("arc_adaptor.click_heatmap",)

#: ``run_cell`` 里**字面**处理的外部包（复制到 /kaggle/working + 导入校验）。
#: 只作为一致性锚点，由 `_verify_packaging` 校验，防止两处清单漂移。
_RUN_CELL_EXTRA_PKGS: tuple[str, ...] = ("arc_adaptor",)

#: 需要随包到达的数据资产：仓库内相对路径 -> `/tmp/lingjing_solo/` 下的落点。
#: 只放小文件；二进制大件（如 AOP 检查点 288 KB）走 `dataset_sources`，
#: base64 内联会让 notebook 直接越过 Kaggle 的 1 MiB 源码上限。
ASSET_FILES: dict[str, str] = {
    "state/click_heatmap_v3.npz": "assets/click_heatmap_v3.npz",
}

#: 内联资产的总 base64 预算（字节）。超了就 ABORT，逼作者改走 dataset_sources。
_ASSET_BUDGET = 128 * 1024

METADATA_PATH = ROOT / "notebooks" / "kernel-metadata.json"


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {"trusted": True},
        "outputs": [],
        "execution_count": None,
        "source": source,
    }


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def _load_offline_module():
    """按文件路径加载 `harness/offline.py`，**绕开** ``lingjing_solo/__init__.py``。

    直接 ``import lingjing_solo.harness.offline`` 会先把包 ``__init__`` 跑一遍
    （它 import core/agent/engine，需要 numpy），打包脚本就多了一个依赖。
    offline.py 本身是纯 stdlib，所以按路径直接加载最干净。
    """
    import importlib.util

    path = LINGJING_SRC / "harness" / "offline.py"
    if not path.is_file():
        raise SystemExit(
            f"[build_notebook] ABORT: 找不到 {path}；打包完整性校验无法进行。"
        )
    spec = importlib.util.spec_from_file_location("_lj_offline", path)
    module = importlib.util.module_from_spec(spec)
    # 必须先登记进 sys.modules：offline.py 用了 `from __future__ import annotations`
    # + `@dataclass`，dataclasses 解析注解时要去 sys.modules 取模块 __dict__，
    # 没登记就 AttributeError: 'NoneType' object has no attribute '__dict__'。
    sys.modules["_lj_offline"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _agent_solo_roots() -> set[str]:
    """从 ``agent/my_agent.py`` 推导需要收录的 ``lingjing_solo.*`` 模块。"""
    tree = ast.parse(AGENT_SRC.read_text(encoding="utf-8-sig"), filename=str(AGENT_SRC))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return {
        n for n in names if n == "lingjing_solo" or n.startswith("lingjing_solo.")
    } | {"lingjing_solo"}


def _collect_extra_files() -> list[tuple[str, str, str]]:
    """收录 `lingjing_solo/` 之外的外部包闭包。

    返回 ``[(包名, 包内相对路径, 源码), ...]``。闭包用 offline 模块的导入遍历，
    这样"外部包缺不缺失"和"内部模块缺不缺失"用同一套判定。
    """
    offline = _load_offline_module()
    repo_root = LINGJING_SRC.parent
    out: list[tuple[str, str, str]] = []
    for root in EXTRA_ROOTS:
        pkg = root.split(".")[0]
        pkg_dir = repo_root / pkg
        if not pkg_dir.is_dir():
            raise SystemExit(f"[build_notebook] ABORT: 外部包 {pkg} 不存在于 {repo_root}")
        entry = repo_root.joinpath(*root.split(".")).with_suffix(".py")
        if not entry.is_file():
            raise SystemExit(f"[build_notebook] ABORT: 外部包入口 {root} 找不到文件")
        # 复用同一套导入遍历，但只保留该包内部的模块
        local, _, _ = offline.import_closure(entry, repo_root)
        wanted = {entry}
        for name, _ref in local.items():
            if not name.startswith(pkg + "."):
                continue
            f = repo_root.joinpath(*name.split("."))
            for cand in (f.with_suffix(".py"), f / "__init__.py"):
                if cand.is_file():
                    wanted.add(cand)
        init = pkg_dir / "__init__.py"
        if init.is_file():
            wanted.add(init)
        for f in sorted(wanted):
            rel = f.relative_to(pkg_dir).as_posix()
            out.append((pkg, rel, _embed_source(rel, f.read_text(encoding="utf-8-sig"))))
    return out


def _collect_assets() -> list[tuple[str, bytes]]:
    """收集要内联的小资产，返回 ``[(落点相对路径, 原始字节)]``。"""
    out: list[tuple[str, bytes]] = []
    total = 0
    for src_rel, dest_rel in ASSET_FILES.items():
        src = LINGJING_SRC.parent / src_rel
        if not src.is_file():
            raise SystemExit(f"[build_notebook] ABORT: 资产 {src_rel} 不存在于 {src}")
        data = src.read_bytes()
        total += len(base64.b64encode(data))
        out.append((dest_rel, data))
    if total > _ASSET_BUDGET:
        raise SystemExit(
            f"[build_notebook] ABORT: 内联资产 base64 合计 {total} B > 预算 {_ASSET_BUDGET} B。"
            " 大件请改走 kernel-metadata.json 的 dataset_sources。"
        )
    return out


def _verify_packaging(files: list[tuple[str, str]], extra: list[tuple[str, str, str]]) -> None:
    """收尾校验：用真实导入闭包比对收录集合，缺了就 ABORT。

    这是"本地验证过、提交却静默降级"那次事故的**可执行防线**。
    """
    offline = _load_offline_module()
    local, external, _ = offline.import_closure(AGENT_SRC, LINGJING_SRC.parent)
    have = {rel for rel, _ in files}
    missing: list[str] = []
    for name in sorted(local):
        parts = name.split(".")[1:]
        if not parts:
            continue
        rel = "/".join(parts) + ".py"
        pkg_init = "/".join(parts) + "/__init__.py"
        if rel not in have and pkg_init not in have:
            missing.append(name)
    have_ext = {pkg for pkg, _, _ in extra}
    missing_ext = sorted({r.split(".")[0] for r in EXTRA_ROOTS} - have_ext)

    # run_cell 里的"复制到 /kaggle/working + 导入校验"片段是字面量；改了
    # EXTRA_ROOTS 就必须同步改它，否则包写出来了却没被复制进可导入位置。
    wanted_pkgs = {r.split(".")[0] for r in EXTRA_ROOTS}
    if wanted_pkgs != set(_RUN_CELL_EXTRA_PKGS):
        raise SystemExit(
            "[build_notebook] ABORT: EXTRA_ROOTS 与 run_cell 内联的包清单不一致。"
            f" EXTRA_ROOTS={sorted(wanted_pkgs)}，run_cell={sorted(_RUN_CELL_EXTRA_PKGS)}。"
            " 请同步更新 run_cell 的复制/校验片段与 _RUN_CELL_EXTRA_PKGS。"
        )

    if missing or missing_ext:
        detail = []
        if missing:
            detail.append(f"  lingjing_solo 模块缺 {len(missing)} 个：" + ", ".join(missing[:8]))
        if missing_ext:
            detail.append(f"  外部包缺：{', '.join(missing_ext)}")
        raise SystemExit(
            "[build_notebook] ABORT: 提交包会缺文件（比赛里不崩，只是静默降级）\n"
            + "\n".join(detail)
            + "\n  修法：把入口补进 EXTRA_ROOTS，或让 _collect_lingjing_files 能解析该模块。"
        )


def _collect_lingjing_files() -> list[tuple[str, str]]:
    """收录 agent 真正需要的 ``lingjing_solo`` 模块（含惰性 / 受保护导入）。

    收录集合由 ``agent/my_agent.py`` 的导入闭包推导，**不再手写 roots** ——
    手写清单会随代码增长失准，而比赛里缺文件的后果是"不崩、只是能力消失"，
    最难发现。收尾由 :func:`_verify_packaging` 用同一套闭包再校一遍。
    """
    if not LINGJING_SRC.is_dir():
        raise SystemExit(
            f"Could not find {LINGJING_SRC}. "
            "Copy/vendoring lingjing_solo into the starter root is required."
        )
    module_paths: dict[str, Path] = {"lingjing_solo": LINGJING_SRC / "__init__.py"}
    for path in LINGJING_SRC.rglob("*.py"):
        rel = path.relative_to(LINGJING_SRC)
        parts = list(rel.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        module_paths["lingjing_solo." + ".".join(parts)] = path

    def resolve_relative(current: str, name: str, level: int) -> str:
        current_path = module_paths[current]
        package = current if current_path.name == "__init__.py" else current.rsplit(".", 1)[0]
        base = package.split(".")[: -(level - 1)] if level > 1 else package.split(".")
        return ".".join(base + (name.split(".") if name else []))

    roots = _agent_solo_roots()
    queue = list(roots)
    seen: set[str] = set()
    while queue:
        module = queue.pop()
        if module in seen:
            continue
        path = module_paths.get(module)
        if path is None or not path.is_file():
            continue
        seen.add(module)
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                target = (
                    resolve_relative(module, node.module or "", node.level)
                    if node.level
                    else (node.module or "")
                )
                targets = [target]
                if node.module is None and node.level:
                    targets.extend(f"{target}.{alias.name}" for alias in node.names)
            else:
                continue
            queue.extend(
                candidate
                for candidate in targets
                if candidate == "lingjing_solo" or candidate.startswith("lingjing_solo.")
            )

    # 祖先包必须一并收录：`import lingjing_solo.a.b.c` 会先把 `lingjing_solo.a`
    # 和 `lingjing_solo.a.b` 的 `__init__.py` 跑一遍。只收叶子文件的话，
    # payload 里就是一个"没有 __init__.py 的中间目录"，导入时直接
    # FileNotFoundError —— 这正是 engine_solver → planning.search.generic_shadow
    # 那条链在探针里挂掉的原因。
    for module in list(seen):
        parts = module.split(".")
        for i in range(2, len(parts) + 1):
            ancestor = ".".join(parts[:i])
            if ancestor in module_paths:
                seen.add(ancestor)

    package_shims = {
        "__init__.py": '"""Kaggle SmartRouter minimal package surface."""\n__version__ = "0.5.0"\n',
        "transfer/__init__.py": '"""Kaggle SmartRouter transfer surface."""\nfrom .ceax_controller import CeaxController\n__all__ = ["CeaxController"]\n',
        "v14/__init__.py": '"""Kaggle SmartRouter V14 transition surface."""\nfrom .arc_transition import V14ArcTransition, TransitionDiagnostics\n__all__ = ["V14ArcTransition", "TransitionDiagnostics"]\n',
        # `planning/__init__.py` 真身会连带拉起 advisor / ls20_solver /
        # data.verified_solutions / simulator —— 闭包里没人用它们（`agent.py`
        # 才用，而 `agent.py` 不在闭包内），整套搬进来只是白占 1 MiB 配额。
        "planning/__init__.py": '"""Kaggle SmartRouter planning surface (shim)."""\n',
        # `search/__init__.py` 真身会 import astar/bfs/dfs，并在导入期用**文件路径**
        # 去加载被目录遮蔽的兄弟文件 `planning/search.py`。闭包里唯一的需求是
        # `planning.search.generic_shadow`（引擎求解器用），而那个文件只依赖
        # stdlib + numpy，所以这里换成空替身，省掉一整条脆弱链。
        "planning/search/__init__.py": '"""Kaggle SmartRouter search package surface (shim)."""\n',
        # `harness/__init__.py` 真身第一行就是 `from .kaggle_adapter import ...`，
        # 而 kaggle_adapter 又 import `..agent` → planning/discrete_nav/ls20_solver
        # → 整条 agent 栈。闭包里唯一需要的是 `harness.engine_action_adapter`
        # （engine_solver 的模块级导入），而它只要子模块导入就够 —— 所以换成
        # 空替身，避免把用不到的整条链塞进 1 MiB 配额。
        "harness/__init__.py": (
            '"""Kaggle SmartRouter harness surface (shim).\n\n'
            "只支持**子模块**导入，例如 ``from lingjing_solo.harness."
            "engine_action_adapter import ...``；\n"
            "``from lingjing_solo.harness import X`` 这类根级 re-export 在提交包里不可用。\n"
            '"""\n'
        ),
    }
    out: list[tuple[str, str]] = []
    for module in sorted(seen | {"lingjing_solo"}):
        path = module_paths[module]
        rel = path.relative_to(LINGJING_SRC).as_posix()
        source = package_shims.get(rel, path.read_text(encoding="utf-8-sig"))
        out.append((rel, _embed_source(rel, source)))
    if not any(r.endswith(".py") for r, _ in out):
        raise SystemExit(f"No .py files found under {LINGJING_SRC}")
    return out


def _lingjing_write_cell(files: list[tuple[str, str]]) -> dict:
    """One notebook cell that materializes the lingjing_solo package under /tmp."""
    parts = [
        "import os\n",
        "from pathlib import Path\n",
        "_LJ_ROOT = Path('/tmp/lingjing_solo')\n",
        "_LJ_ROOT.mkdir(parents=True, exist_ok=True)\n",
        "_LJ_FILES = {\n",
    ]
    for rel, text in files:
        parts.append(f"    {rel!r}: {text!r},\n")
    parts.append("}\n")
    parts.append(
        "for _rel, _src in _LJ_FILES.items():\n"
        "    _p = _LJ_ROOT / _rel\n"
        "    _p.parent.mkdir(parents=True, exist_ok=True)\n"
        "    _p.write_text(_src, encoding='utf-8')\n"
        "print(f'[lingjing_solo] wrote {len(_LJ_FILES)} files -> {_LJ_ROOT}')\n"
    )
    return code_cell("".join(parts))


def _extra_write_cell(extra: list[tuple[str, str, str]]) -> dict:
    """把 `lingjing_solo/` 之外的外部包写到 ``/tmp/lj_ext/<pkg>/...``。

    刻意**不**塞进 ``/tmp/lingjing_solo``：那是包目录，把 `arc_adaptor` 放进去
    就成了 `lingjing_solo.arc_adaptor`，与 agent 的 ``from arc_adaptor import ...``
    对不上。外部包必须是独立的顶层目录，再由 run 单元复制到工作目录。
    """
    parts = [
        "import os\n",
        "from pathlib import Path\n",
        "_EXT_ROOT = Path('/tmp/lj_ext')\n",
        "_EXT_ROOT.mkdir(parents=True, exist_ok=True)\n",
        "_EXT_FILES = {\n",
    ]
    for pkg, rel, text in extra:
        parts.append(f"    {pkg + '/' + rel!r}: {text!r},\n")
    parts.append("}\n")
    parts.append(
        "for _rel, _src in _EXT_FILES.items():\n"
        "    _p = _EXT_ROOT / _rel\n"
        "    _p.parent.mkdir(parents=True, exist_ok=True)\n"
        "    _p.write_text(_src, encoding='utf-8')\n"
        "print(f'[lj_ext] wrote {len(_EXT_FILES)} files -> {_EXT_ROOT}')\n"
    )
    return code_cell("".join(parts))


def _assets_write_cell(assets: list[tuple[str, bytes]]) -> dict:
    """把小资产以 base64 内联到 ``/tmp/lingjing_solo/<落点>``。

    比赛里没有网络、也没有 pip 通道，权重只能随包到达。小文件内联最省事；
    大文件会因为 Kaggle 的 1 MiB 源码上限被 :func:`_collect_assets` 拦下。
    """
    payload = "".join(
        f"    {dest!r}: {base64.b64encode(data).decode('ascii')!r},\n"
        for dest, data in assets
    )
    return code_cell(
        "import base64\n"
        "from pathlib import Path\n"
        "_LJ_ROOT = Path('/tmp/lingjing_solo')\n"
        "_LJ_ASSETS = {\n" + payload + "}\n"
        "for _rel, _b64 in _LJ_ASSETS.items():\n"
        "    _p = _LJ_ROOT / _rel\n"
        "    _p.parent.mkdir(parents=True, exist_ok=True)\n"
        "    _p.write_bytes(base64.b64decode(_b64))\n"
        "print(f'[lingjing_solo] wrote {len(_LJ_ASSETS)} assets')\n"
    )


def build() -> dict:
    if not AGENT_SRC.exists():
        raise SystemExit(f"Could not find {AGENT_SRC}")
    agent_body = _embed_source("agent/my_agent.py", AGENT_SRC.read_text(encoding="utf-8"))
    lj_files = _collect_lingjing_files()
    extra_files = _collect_extra_files()
    assets = _collect_assets()
    # 收尾校验：闭包要的都得在包里。缺一个就 ABORT，别让比赛现场才发现。
    _verify_packaging(lj_files, extra_files)
    print(
        f"[build_notebook] payload: lingjing_solo {len(lj_files)} files, "
        f"external {len(extra_files)} files ({', '.join(sorted({p for p, _, _ in extra_files}))}), "
        f"assets {len(assets)}"
    )

    # The competition-data mount is not always ready when the first cell runs:
    # the 2026-10-12 Phase A log showed pip rejecting the wheels dir with
    # "Location '...' is ignored: it is either a non-existing path", while the
    # teammate's byte-identical cell installed fine. Wait for the mount, then
    # pip -- and print [TRACE] evidence, because Phase A is the only run whose
    # log we can actually read back from Kaggle.
    install_cell = code_cell(
        dedent(
            """\
            import os, time
            _WHEELS = '/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels'
            _waited = 0
            while _waited < 60 and not os.path.isdir(_WHEELS):
                time.sleep(2); _waited += 2
            if os.path.isdir(_WHEELS):
                print(f'[TRACE] wheels dir OK after {_waited}s, files={len(os.listdir(_WHEELS))}', flush=True)
                os.system(f\"pip install --no-index --find-links {_WHEELS} arc-agi python-dotenv numpy\")
            else:
                print('[TRACE] wheels dir MISSING after 60s -> skip offline pip, rely on image packages', flush=True)
            try:
                import arc_agi
                print(f'[TRACE] import arc_agi OK file={arc_agi.__file__}', flush=True)
            except Exception as _e:
                print(f'[TRACE] import arc_agi FAIL {type(_e).__name__}: {_e}', flush=True)
            """
        )
    )

    # We write the agent to /tmp/ (not /kaggle/working/) so it does NOT appear
    # as a notebook output. Otherwise the "Submit to Competition" UI would
    # offer it as a candidate submission file alongside submission.parquet,
    # and an unlucky default selection rejects the submission.
    write_agent_cell = code_cell(
        "%%writefile /tmp/my_agent.py\n" + agent_body
    )
    write_lingjing_cell = _lingjing_write_cell(lj_files)
    write_extra_cell = _extra_write_cell(extra_files)
    write_assets_cell = _assets_write_cell(assets)

    # Phase B's stdout is not retrievable from Kaggle at all (verified 2026-10-12
    # on three routes: `competitions episodes` returns "No episodes found" even
    # for submissions that scored, the kernel "Logs" tab only ever shows the
    # Save-and-run-all run, and the submissions row opens nothing). Phase A's log
    # IS readable, so Phase A is where the execution evidence has to be printed.
    # Everything is wrapped so a probe failure can never break the commit.
    self_check_cell = code_cell(
        dedent(
            """\
            import os, sys
            if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
                def _t(msg):
                    print(f'[TRACE] {msg}', flush=True)
                _CFGS = '/kaggle/input/competitions/arc-prize-2026-arc-agi-3'
                for _p in (f'{_CFGS}/environment_files', '/tmp/lingjing_solo/environment_files'):
                    try:
                        _t(f'dir {_p} exists={os.path.isdir(_p)}'
                           + (f' games={len(os.listdir(_p))}' if os.path.isdir(_p) else ''))
                    except Exception as _e:
                        _t(f'dir {_p} probe FAIL {_e!r}')
                sys.path.insert(0, '/tmp')
                try:
                    import my_agent as _ma
                    _t(f"BUILD_TAG={getattr(_ma, 'BUILD_TAG', '?')}")
                    for _nm in ('_LP85_PLANS', '_SU15_PLANS'):
                        _d = getattr(_ma, _nm, {})
                        _t(f'{_nm} levels={len(_d)} clicks={sum(len(v) for v in _d.values())}')
                except Exception as _e:
                    _t(f'import my_agent FAIL {type(_e).__name__}: {_e}')
                for _m in ('torch', 'scipy', 'sklearn', 'pandas', 'numpy'):
                    try:
                        __import__(_m)
                        _t(f'dep {_m} OK')
                    except Exception as _e:
                        _t(f'dep {_m} MISSING ({_e})')
                _t('phaseA self-check done')
            """
        )
    )

    run_cell_source = dedent(
        """\
        import os

        if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            # Wait for the gateway sidecar to be ready.
            !curl --fail --retry 999 --retry-all-errors --retry-delay 5 \\
                  --retry-max-time 600 http://gateway:8001/api/games

            # Copy the framework into a writable location.
            !cp -r /kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents \\
                   /kaggle/working/ARC-AGI-3-Agents

            # Drop our agent + lingjing_solo package into the framework tree.
            !cp /tmp/my_agent.py \\
                /kaggle/working/ARC-AGI-3-Agents/agents/templates/my_agent.py
            !rm -rf /kaggle/working/ARC-AGI-3-Agents/lingjing_solo
            !cp -r /tmp/lingjing_solo \\
                /kaggle/working/ARC-AGI-3-Agents/lingjing_solo

            # External packages (arc_adaptor) must be a TOP-LEVEL dir next to the
            # framework so `from arc_adaptor import ...` resolves; my_agent's
            # _ensure_adaptor_on_path() probes parents[3] == /kaggle/working.
            !rm -rf /kaggle/working/arc_adaptor
            !cp -r /tmp/lj_ext/arc_adaptor /kaggle/working/arc_adaptor
            !python -c "import sys; sys.path.insert(0,'/kaggle/working'); import arc_adaptor.click_heatmap as m; print('EXT_CHECK arc_adaptor.click_heatmap OK', m.__file__)"

            # Register MyAgent FIRST (slim __init__) — fix0 order: rewrite before import.
            with open('/kaggle/working/ARC-AGI-3-Agents/agents/__init__.py', 'w') as f:
                f.write(\"\"\"from typing import Type
        from dotenv import load_dotenv
        from .agent import Agent, Playback
        from .swarm import Swarm
        from .templates.random_agent import Random
        from .templates.my_agent import MyAgent

        load_dotenv()

        AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
            'random': Random,
            'myagent': MyAgent,
        }
        \"\"\")

            # Fail loud if wrong agent landed (SmartRouter fingerprint).
            !python -c "import sys; sys.path.insert(0,'/kaggle/working/ARC-AGI-3-Agents'); from agents.templates.my_agent import BUILD_TAG, MyAgent; assert BUILD_TAG.startswith('smart-router'), BUILD_TAG; open('/kaggle/working/BUILD_TAG.txt','w').write(BUILD_TAG); print('SMART_CHECK', BUILD_TAG, 'MAX_ACTIONS', MyAgent.MAX_ACTIONS)"

            # Point the framework at the gateway sidecar.
            with open('/kaggle/working/ARC-AGI-3-Agents/.env', 'w') as f:
                f.write(\"\"\"SCHEME=http
        HOST=gateway
        PORT=8001
        ARC_API_KEY=test-key-123
        ARC_BASE_URL=http://gateway:8001/
        OPERATION_MODE=online
        ENVIRONMENTS_DIR=
        RECORDINGS_DIR=/kaggle/working/server_recording
        AETHER_LLM=0
        LINGJING_SCRIPT_DEBUG=0
        \"\"\")

            # Run it. The gateway records every action and emits submission.parquet.
            !cd /kaggle/working/ARC-AGI-3-Agents && \\
                MPLBACKEND=agg \\
                PYTHONPATH=/kaggle/working/ARC-AGI-3-Agents:${PYTHONPATH} \\
                python main.py --agent myagent
            !python -c "print('MAIN_EXIT', 0)"
        """
    )
    run_cell = code_cell(run_cell_source)

    dummy_submission_cell = code_cell(
        dedent(
            """\
            import os
            if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
                # Save-and-run-all (commit) mode: emit a dummy submission so the
                # commit succeeds. The real submission.parquet is produced by the
                # gateway during competition rerun.
                import pandas as pd
                submission = pd.DataFrame(
                    data=[['1_0', '1', True, 1]],
                    columns=['row_id', 'game_id', 'end_of_game', 'score'])
                submission.to_parquet('/kaggle/working/submission.parquet', index=False)
                submission.head()
            """
        )
    )

    if ACCELERATOR not in _ACCELERATORS:
        raise SystemExit(
            f"Unknown ACCELERATOR={ACCELERATOR!r}. Pick one of: "
            f"{sorted(_ACCELERATORS)}"
        )
    accel = _ACCELERATORS[ACCELERATOR]

    notebook = {
        "metadata": {
            "kernelspec": {
                "language": "python",
                "display_name": "Python 3",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "mimetype": "text/x-python",
                "file_extension": ".py",
                "pygments_lexer": "ipython3",
            },
            "kaggle": {
                "accelerator": accel["name"],
                "isInternetEnabled": False,
                "isGpuEnabled": accel["gpu"],
                "language": "python",
                "sourceType": "notebook",
            },
        },
        "nbformat_minor": 4,
        "nbformat": 4,
        "cells": [
            markdown_cell(
                "# ARC Prize 2026 — CEAX Unknown v3.6 (no plugins)\n\n"
                "Submission agent = **SmartRouter v1** (`smart-router-v1+inline-ls20/ar25/ft09+ceax`). "
                "No PluginRegistry / ScriptBank / canned ls20. "
                "Local focus8@300 aggregate ≈**2.26** (7/8 non-zero, 8 levels). "
                "Phase A validates; Phase B competition rerun scores."
            ),
            install_cell,
            write_lingjing_cell,
            write_assets_cell,
            write_extra_cell,
            write_agent_cell,
            self_check_cell,
            run_cell,
            dummy_submission_cell,
        ],
    }
    return notebook


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    notebook = build()
    text = json.dumps(notebook, indent=1, ensure_ascii=False)
    # Kaggle measures the JSON-normalised form with non-ASCII escaped to \uXXXX,
    # which costs 6 bytes per CJK character -- that is the number the 1 MB cap
    # is checked against, not the file size on disk.
    escaped = len(json.dumps(notebook).encode("utf-8"))
    raw = len(text.encode("utf-8"))
    print(f"[build_notebook] comments+docstrings stripped: {_STRIP_STATS['bytes']} chars "
          f"across {_STRIP_STATS['files']} files")
    print(f"[build_notebook] size: on-disk {raw} B, as-measured-by-Kaggle {escaped} B "
          f"(limit {_KERNEL_SOURCE_LIMIT}, {"OK" if escaped < _KERNEL_SOURCE_LIMIT else "OVER"})")
    if escaped >= _KERNEL_SOURCE_LIMIT:
        raise SystemExit(
            f"[build_notebook] ABORT: kernel source would be rejected "
            f"({escaped} >= {_KERNEL_SOURCE_LIMIT} bytes as measured by Kaggle). "
            f"Trim embedded sources before pushing."
        )
    NOTEBOOK_PATH.write_text(text, encoding="utf-8")
    print(f"[build_notebook] Wrote {NOTEBOOK_PATH.relative_to(ROOT)}  "
          f"(accelerator: {ACCELERATOR})")

    # Keep notebooks/kernel-metadata.json in sync so the user never has to
    # edit it just to flip CPU ↔ GPU.
    if METADATA_PATH.exists():
        meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        wanted = _ACCELERATORS[ACCELERATOR]["gpu"]
        if meta.get("enable_gpu") != wanted:
            meta["enable_gpu"] = wanted
            METADATA_PATH.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            print(f"[build_notebook] Synced enable_gpu={wanted} in "
                  f"{METADATA_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
