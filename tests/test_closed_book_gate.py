"""Reverse gate — the closed-book containment of every baked answer.

The M1 remediation (2026-10-13) established that the repo's *canned* score
(52.60) is ~96% hardcoded action tables and only 1.2% real closed-book
capability.  The ordinary CI (pytest + route replay) only checks that the
agent DOES pass levels — it happily rewards a bigger answer table.  That is
exactly the failure mode this file reverses.

This gate is deliberately **static** (it reads ``my_agent.py`` as text/AST and
never imports or runs the engine), so it runs in the plain ``pytest`` CI job
where ``arcengine`` is not installed.

It enforces three invariants:

1. **Switch present** — ``LINGJING_CLOSED_BOOK`` exists and both levels
   (``_CLOSED_BOOK`` / ``_CLOSED_BOOK_STRICT``) are wired as module flags.
2. **No unregistered baked table** — every module-level ``_*_PLANS`` /
   ``_*_LEVEL_ACTIONS`` constant must be on the frozen allowlist.  Smuggling a
   new canned answer in without touching this list is a CI failure.
3. **Every baked lookup is gated** — inside the decision function
   ``_choose_action_inner``, each reference to a baked table (and to the
   ``ls20`` / ``ar25`` inline scripts) must be enclosed by ``if not
   _CLOSED_BOOK:``.  The EngineSolver entry must be gated by ``not
   _CLOSED_BOOK`` and the R3 search by ``not _CLOSED_BOOK_STRICT``.

If this gate is red, a baked answer has escaped the closed-book switch:
either move it under the guard, or (if it is genuinely frame-derived)
rename it so it is no longer matched as a canned table.
"""
from __future__ import annotations

import ast
import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_DEFAULT_AGENT = REPO / "integrations" / "ARC-AGI-3-Kaggle-Starter" / "agent" / "my_agent.py"
# Optional override so the same gate can be pointed at a candidate agent file
# (used by the mutation test that proves the gate actually bites).
AGENT = Path(os.environ.get("LINGJING_GATE_AGENT", _DEFAULT_AGENT))

# Frozen allowlist of baked answer tables.  Adding an entry here is a
# deliberate acknowledgement that the repo ships another canned answer;
# removing one without deleting the table (or vice versa) is a hard failure.
EXPECTED_CANNED_TABLES = {
    "_VC33_PLANS",
    "_WA30_PLANS",
    "_SB26_PLANS",
    "_SU15_PLANS",
    "_R11L_PLANS",
    "_CD82_PLANS",
    "_TN36_PLANS",
    "_TR87_PLANS",
    "_SC25_PLANS",
    "_BP35_PLANS",
    "_TR87_LEVEL_ACTIONS",
    "_LS20_LEVEL_ACTIONS",
    "_AR25_LEVEL_ACTIONS",
    "_TU93_LEVEL_ACTIONS",
}

# Any node whose source text matches one of these is a baked lookup that must
# sit under the closed-book guard when it appears in the decision function.
CANNED_PATTERNS = (
    re.compile(r"^_[A-Z0-9_]+_(?:PLANS|LEVEL_ACTIONS)$"),
    re.compile(r"^self\.(?:ls20|ar25)$"),
)

GUARD_RE = re.compile(r"not\s+_CLOSED_BOOK\b")
GUARD_STRICT_RE = re.compile(r"not\s+_CLOSED_BOOK_STRICT\b")

DECISION_FUNC = "_choose_action_inner"


def _module_ast() -> ast.Module:
    if not AGENT.exists():
        pytest.fail(f"agent source not found: {AGENT}")
    src = AGENT.read_text(encoding="utf-8")
    return ast.parse(src)


def _find_func(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node  # type: ignore[return-value]
    pytest.fail(f"function {name!r} not found in {AGENT.name}")


def _module_level_names(tree: ast.Module) -> list[tuple[str, int]]:
    names: list[tuple[str, int]] = []
    for node in tree.body:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for tgt in targets:
            if isinstance(tgt, ast.Name):
                names.append((tgt.id, node.lineno))
    return names


def _is_canned(text: str) -> bool:
    return any(p.match(text) for p in CANNED_PATTERNS)


class _GuardTracker(ast.NodeVisitor):
    """Collect canned references and the unparsed tests of enclosing ``if``s."""

    def __init__(self) -> None:
        self.stack: list[str] = []
        self.hits: list[tuple[str, list[str], int]] = []
        self.ifs: list[tuple[str, int]] = []

    def visit_If(self, node: ast.If) -> None:
        test = ast.unparse(node.test)
        self.ifs.append((test, node.lineno))
        self.stack.append(test)
        self.generic_visit(node)
        self.stack.pop()

    def _record(self, text: str, lineno: int) -> None:
        if _is_canned(text):
            self.hits.append((text, list(self.stack), lineno))

    def visit_Name(self, node: ast.Name) -> None:
        self._record(node.id, node.lineno)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self._record(ast.unparse(node), node.lineno)


def test_closed_book_switch_exists():
    """The two-level switch must be present and wired as module flags."""
    tree = _module_ast()
    src = AGENT.read_text(encoding="utf-8")

    assert "LINGJING_CLOSED_BOOK" in src, "env switch LINGJING_CLOSED_BOOK is gone"

    flags = {name for name, _ in _module_level_names(tree)}
    assert "_CLOSED_BOOK" in flags, "_CLOSED_BOOK module flag missing"
    assert "_CLOSED_BOOK_STRICT" in flags, "_CLOSED_BOOK_STRICT module flag missing"

    # strict implies closed-book; assert the wiring is still there.
    assert re.search(r"if\s+_CLOSED_BOOK_STRICT\s*:\s*\n\s+_CLOSED_BOOK\s*=\s*True", src), (
        "strict no longer implies closed-book (if _CLOSED_BOOK_STRICT: _CLOSED_BOOK = True)"
    )


def test_no_unregistered_canned_tables():
    """Every baked table must be on the frozen allowlist."""
    tree = _module_ast()
    found = {name for name, _ in _module_level_names(tree) if _is_canned(name)}

    unregistered = sorted(found - EXPECTED_CANNED_TABLES)
    missing = sorted(EXPECTED_CANNED_TABLES - found)

    msg = []
    if unregistered:
        msg.append(
            "New baked answer table(s) detected: "
            + ", ".join(unregistered)
            + ".\nIf this is a genuine hardcoded answer, add it to EXPECTED_CANNED_TABLES "
            "AND make sure every lookup is behind `if not _CLOSED_BOOK:` "
            "in _choose_action_inner."
        )
    if missing:
        msg.append(
            "Allowlisted table(s) no longer present: "
            + ", ".join(missing)
            + ".\nDelete them from EXPECTED_CANNED_TABLES if the canned answer was removed."
        )
    assert not msg, "\n".join(msg)


def test_baked_lookups_are_gated_in_decision_path():
    """Inside _choose_action_inner, every baked lookup must be under the guard."""
    tree = _module_ast()
    func = _find_func(tree, DECISION_FUNC)

    tracker = _GuardTracker()
    tracker.visit(func)

    ungated = [
        (name, lineno)
        for name, guards, lineno in tracker.hits
        if not any(GUARD_RE.search(g) for g in guards)
    ]

    assert tracker.hits, (
        f"expected baked lookups inside {DECISION_FUNC}; found none — did the gate patterns rot?"
    )
    assert not ungated, (
        "Baked lookup escaped the closed-book guard in "
        f"{DECISION_FUNC}:\n"
        + "\n".join(f"  line {ln}: {nm}" for nm, ln in ungated)
        + "\nWrap it in `if not _CLOSED_BOOK:` (or the strict variant)."
    )

    # The inline guard block must actually be the enclosing scope for the bulk
    # of them; a single stray reference slipping through is what we caught above.
    gated = [h for h in tracker.hits if any(GUARD_RE.search(g) for g in h[1])]
    assert len(gated) >= 8, f"only {len(gated)} baked lookups under guard; expected >= 8"


def test_engine_solver_entry_is_gated():
    """The EngineSolver branch must be disabled under closed-book."""
    tree = _module_ast()
    func = _find_func(tree, DECISION_FUNC)

    tracker = _GuardTracker()
    tracker.visit(func)

    solver_ifs = [(t, ln) for t, ln in tracker.ifs if "_ENGINE_SOLVER_GAMES" in t]
    assert solver_ifs, "no `_ENGINE_SOLVER_GAMES` branch found in decision path"
    ungated = [(t, ln) for t, ln in solver_ifs if not GUARD_RE.search(t)]
    assert not ungated, (
        "EngineSolver branch is not gated by `not _CLOSED_BOOK`:\n"
        + "\n".join(f"  line {ln}: {t}" for t, ln in ungated)
    )


def test_r3_search_is_gated_under_strict():
    """The R3 shadow/IDDFS branch must be disabled under strict closed-book."""
    tree = _module_ast()
    func = _find_func(tree, DECISION_FUNC)

    tracker = _GuardTracker()
    tracker.visit(func)

    # The R3 *entry* guard is the only `if` that consults the skip set; inner
    # conditions (e.g. `hasattr(self, '_r3_path') and self._r3_path`) are its
    # children and need no guard of their own.
    r3_ifs = [(t, ln) for t, ln in tracker.ifs if "_SKIP_R3" in t]
    assert r3_ifs, "no R3 entry branch found in decision path"
    ungated = [(t, ln) for t, ln in r3_ifs if not GUARD_STRICT_RE.search(t)]
    assert not ungated, (
        "R3 search is not gated by `not _CLOSED_BOOK_STRICT`:\n"
        + "\n".join(f"  line {ln}: {t}" for t, ln in ungated)
    )


def test_tu93_closed_book_branch_is_wired():
    """The frame-only tu93 solver must be reachable from the decision path.

    Regression guard: the branch, its lazy import and its fall-through shape are
    exactly what turns the tu93 frame model into score.  If someone deletes or
    ungates it, the closed-book number silently drops back to the AR25-only
    baseline -- catch that here instead of in the score.
    """
    tree = _module_ast()
    func = _find_func(tree, DECISION_FUNC)

    branches = [
        node
        for node in ast.walk(func)
        if isinstance(node, ast.If)
        and re.search(r"_CLOSED_BOOK\b", ast.unparse(node.test))
        and re.search(r"gid\s*==\s*['\"]tu93['\"]", ast.unparse(node.test))
    ]
    assert len(branches) == 1, (
        f"expected exactly one `_CLOSED_BOOK and gid == 'tu93'` branch in "
        f"{DECISION_FUNC}; found {len(branches)}"
    )

    body = ast.unparse(branches[0])
    assert "closed_book.tu93_solver" in body, "tu93 branch no longer imports the frame solver"
    assert "_tu93_frame_solver" in body, "tu93 branch lost its lazy solver handle"
    assert "next_action" in body, "tu93 branch no longer asks the solver for an action"
    assert "except Exception" in body, "tu93 branch must swallow errors and fall through"


def test_entry_shortcut_is_gated():
    """The choose_action entry shortcut must also respect closed-book."""
    tree = _module_ast()
    func = _find_func(tree, "choose_action")

    tracker = _GuardTracker()
    tracker.visit(func)

    solver_ifs = [(t, ln) for t, ln in tracker.ifs if "_ENGINE_SOLVER_GAMES" in t]
    assert solver_ifs, "no `_ENGINE_SOLVER_GAMES` entry shortcut found in choose_action"
    ungated = [(t, ln) for t, ln in solver_ifs if not GUARD_RE.search(t)]
    assert not ungated, (
        "choose_action entry shortcut bypasses the closed-book guard:\n"
        + "\n".join(f"  line {ln}: {t}" for t, ln in ungated)
    )


# --- the closed-book capability modules must stay honest ----------------------

CLOSED_BOOK_DIR = Path(
    os.environ.get("LINGJING_GATE_CLOSED_BOOK_DIR", REPO / "lingjing_solo" / "closed_book")
)

# Substrings that would mean the closed-book path is peeking at the engine
# (source file, baked solution tables, obfuscated sprite tags).
FORBIDDEN_IN_CLOSED_BOOK = (
    "environment_files",
    "KNOWN_SOLUTIONS",
    "arc_shadow",
    "ar25.py",
    "reference/",
)
# e.g. "0001sruqbuvukh" — the engine's obfuscated sprite tags
OBFUSCATED_TAG_RE = re.compile(r"['\"][0-9]{4}[a-z]{10}['\"]")


def _strip_docstrings(src: str) -> str:
    """Normalise to AST-unparsed code with all docstrings removed."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body:
            first = body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                node.body = body[1:]
    return ast.unparse(tree)


def test_closed_book_modules_never_read_the_engine():
    """The frame-only modules must not reference the engine or baked answers.

    Docstrings are allowed to *mention* the prohibition, so they are stripped
    before scanning.
    """
    assert CLOSED_BOOK_DIR.is_dir(), f"missing package: {CLOSED_BOOK_DIR}"
    files = sorted(CLOSED_BOOK_DIR.glob("*.py"))
    assert files, "expected at least one closed-book module"

    problems: list[str] = []
    for path in files:
        code = _strip_docstrings(path.read_text(encoding="utf-8"))
        for bad in FORBIDDEN_IN_CLOSED_BOOK:
            if bad in code:
                problems.append(f"{path.name}: contains {bad!r}")
        for m in OBFUSCATED_TAG_RE.finditer(code):
            problems.append(f"{path.name}: obfuscated engine tag {m.group(0)}")

    assert not problems, (
        "closed-book modules must stay frame-only:\n  " + "\n  ".join(problems)
    )


# Names that would betray a baked answer hidden inside a closed-book module.
BAKED_NAME_RE = re.compile(r"(LEVEL|PLAN|SOLUTION|SCRIPT|SEQUENCE|ANSWER)", re.IGNORECASE)
# A list/tuple/set literal this long, of bare ints, is an action sequence.
_BAKED_SEQ_MIN = 15


def _long_int_literals(tree: ast.Module) -> list[tuple[int, int]]:
    """(lineno, length) for every >=15-element literal of bare ints."""
    out: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            continue
        if len(node.elts) < _BAKED_SEQ_MIN:
            continue
        if all(isinstance(e, ast.Constant) and isinstance(e.value, int) for e in node.elts):
            out.append((node.lineno, len(node.elts)))
    return out


def test_no_baked_sequence_in_closed_book_modules():
    """The frame-only modules must not smuggle a per-level action table.

    Two smells are rejected: a module-level constant whose *name* suggests a
    baked plan (LEVEL/PLAN/SOLUTION/...), and any literal of >=15 bare ints
    (an action sequence).  Solving a level in closed book means deriving the
    moves from the frame, never pasting them.
    """
    assert CLOSED_BOOK_DIR.is_dir(), f"missing package: {CLOSED_BOOK_DIR}"
    files = sorted(CLOSED_BOOK_DIR.glob("*.py"))
    assert files, "expected at least one closed-book module"

    problems: list[str] = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            targets: list[ast.expr] = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            for tgt in targets:
                if isinstance(tgt, ast.Name) and BAKED_NAME_RE.search(tgt.id):
                    problems.append(
                        f"{path.name}:{node.lineno}: baked-looking constant {tgt.id!r}"
                    )
        for lineno, n in _long_int_literals(tree):
            problems.append(
                f"{path.name}:{lineno}: literal of {n} bare ints (looks like an action sequence)"
            )

    assert not problems, (
        "closed-book modules must derive moves, not bake them:\n  "
        + "\n  ".join(problems)
    )

