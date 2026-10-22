"""Merge upstream ``agent/my_agent.py`` into ours without losing either side.

Why this script exists
----------------------
``origin/main`` and our local ``main`` diverged from a common ancestor
(``3b18403``). A plain ``git merge`` produced a file with **ten duplicated
top-level definitions** because our ``my_agent.py`` had been produced by
pasting the whole-qwen LLM chain *after* an already-merged body -- Python then
silently ran the *last* definition of each name, so the file was 4608 lines
long but only ~3287 lines of behaviour.

Rather than hand-merge 12 conflict hunks, this script performs a **symbol-level
merge**: upstream is the base, and every symbol is explicitly resolved by a
decision table below. Every entry is justified by a three-way comparison
against the merge base (``base`` vs ``upstream`` vs ``ours``).

Decision table
--------------
``TAKE_OURS``    -- ours changed it, upstream either left it alone or its
                    change is a strict subset of ours.
``TAKE_UPSTREAM``-- upstream changed it and the change is a genuine
                    improvement that ours does not have (or ours only differs
                    by a stale docstring).
``ADD_OURS``     -- ours-only symbols (the whole-qwen chain) inserted after a
                    stable anchor so the file stays readable.
``UNION``        -- constants where both sides added a member and the union is
                    strictly better than either.

Run
---
    python scripts/merge_my_agent_upstream.py \
        --upstream origin/main --base 3b18403 \
        --ours-ref HEAD --out agent/my_agent.py

``--ours-ref`` reads our side straight out of git, so the script is re-runnable
and reproducible after the merge commit exists.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
DEFAULT_TARGET = ROOT / "agent" / "my_agent.py"

# --------------------------------------------------------------------------
# Decision table
# --------------------------------------------------------------------------

# Symbols where our version is the one to ship.
TAKE_OURS: set[tuple] = {
    # ours: adds the whole-qwen call budget (cap 800 vs 30 when the
    # whole-episode mode owns the game). Upstream has no _llm_whole_on().
    ("func", "_llm_call_budget"),
    # ours: whole-qwen state (dead actions, stuck counters, episode mind),
    # the SELFLEARN_LLM_MAXCHARS knob and the cn04/dc22/m0r0/sp80 plan routes.
    ("meth", "MyAgent", "__init__"),
    # ours: captures _engine_reason and adds the terminal LLM override.
    ("meth", "MyAgent", "choose_action"),
    # ours: the whole-episode (千问整局) path + 4 calibrated plan routes.
    # Upstream's version is a strict subset of ours (see the diff in the
    # merge report).
    ("meth", "MyAgent", "_choose_action_inner"),
    # ours: C++ CompactBFS acceleration. Upstream never touched this function
    # (upstream == base), so there is nothing to lose.
    ("meth", "MyAgent", "_bfs_search"),
    # ours: C++ Visited acceleration + the cosine-dedup lifetime fix that
    # upstream also carries (_n_seen[0] = 0 per depth iteration).
    ("meth", "MyAgent", "_iddfs_search"),
    # ours: the whole-qwen batch call carries history/briefing, and the legacy
    # planner it delegates to accepts them too. Upstream's pair takes neither,
    # so taking upstream here breaks the whole-episode path with
    # "TypeError: _try_llm_batch() got an unexpected keyword argument 'history'".
    ("meth", "MyAgent", "_try_llm_batch"),
    ("meth", "MyAgent", "_legacy_plan_specs"),
    # ours: inserts the lexicon parent dir at the FRONT of sys.path, so the
    # lexicon we ship is the one that gets imported.
    ("func", "_lp_load"),
    # ours: a strict superset of upstream's capability tags.
    ("assign", "BUILD_TAG"),
    # ours: upstream's protected set + cn04/dc22/m0r0/sp80 (all four have a
    # calibrated WIN plan, so the plan route must win over engine/LLM).
    ("assign", "_PLAN_PROTECTED_GAMES"),
}

# Symbols where upstream's version wins.
TAKE_UPSTREAM: set[tuple] = {
    ("func", "_action_label"),        # upstream docstring
    ("func", "_bootstrap_paths"),     # upstream also puts duck_bridge dir on sys.path
    ("func", "_llm_preceax_on"),      # upstream docstring
    ("func", "_llm_spec_to_action"),  # upstream docstring
    ("func", "_lp_action_spec"),      # upstream docstring
    ("meth", "MyAgent", "_duck_tick"),  # upstream adds LINGJING_DUCK_FORCE_GIDS
    # Docstring-only differences: upstream's are better written.
    ("meth", "MyAgent", "_lp_observe"),
    ("meth", "MyAgent", "_clear_llm_queue"),
    ("meth", "MyAgent", "_pop_llm_queue"),
    ("meth", "MyAgent", "_lp_plan_specs"),
}

# ours-only symbols, inserted after a stable anchor.
ADD_OURS: dict[tuple, list[tuple]] = {
    ("func", "_llm_call_budget"): [
        ("func", "_llm_stall_threshold"),
        ("func", "_llm_whole_on"),
        ("func", "_load_llm_policy_plan"),
        ("func", "_llm_action_sig"),
        ("func", "_llm_unused_action"),
        ("func", "_try_llm_action"),
        ("func", "_try_import_lingjing_explore"),
    ],
    ("assign", "_BP35_PLANS"): [
        ("assign", "_M0R0_PLANS"),
        ("assign", "_SP80_PLANS"),
        ("assign", "_CN04_PLANS"),
        ("assign", "_DC22_PLANS"),
    ],
}

# Constants merged as the union of both sides.
UNION_CONSTS: dict[tuple, list[str]] = {
    # upstream added "sp80" to the engine-solver whitelist; ours removed it
    # because sp80 has a calibrated WIN plan. Union keeps both: the plan route
    # returns first, the engine solver is the fallback for other levels.
    ("assign", "_ENGINE_SOLVER_GAMES"): ["sp80"],
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def git_show(repo: Path, ref: str, path: str) -> str:
    r = subprocess.run(
        ["git", "-C", str(repo), "show", f"{ref}:{path}"],
        capture_output=True,
    )
    if r.returncode != 0:
        raise SystemExit(f"git show {ref}:{path} failed: {r.stderr.decode('utf-8', 'replace')}")
    return r.stdout.decode("utf-8", errors="replace")


def index(text: str) -> dict[tuple, tuple[int, int, str]]:
    """Map symbol -> (start_line0, end_line_exclusive, source_text).

    Duplicated names resolve to the *last* definition, matching Python's
    runtime semantics.
    """
    lines = text.splitlines()
    tree = ast.parse(text)
    out: dict[tuple, tuple[int, int, str]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[("func", node.name)] = (
                node.lineno - 1,
                node.end_lineno,
                "\n".join(lines[node.lineno - 1:node.end_lineno]),
            )
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out[("meth", node.name, sub.name)] = (
                        sub.lineno - 1,
                        sub.end_lineno,
                        "\n".join(lines[sub.lineno - 1:sub.end_lineno]),
                    )
        elif isinstance(node, ast.Assign):
            for tg in node.targets:
                if isinstance(tg, ast.Name):
                    out[("assign", tg.id)] = (
                        node.lineno - 1,
                        node.end_lineno,
                        "\n".join(lines[node.lineno - 1:node.end_lineno]),
                    )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out[("assign", node.target.id)] = (
                node.lineno - 1,
                node.end_lineno,
                "\n".join(lines[node.lineno - 1:node.end_lineno]),
            )
    return out


def merge(upstream: str, ours: str) -> str:
    up_lines = upstream.splitlines()
    up_idx = index(upstream)
    our_idx = index(ours)

    # (start, end, replacement_text) -- end is exclusive
    edits: list[tuple[int, int, str]] = []
    inserts: list[tuple[int, str]] = []  # (insert_after_line0, text)

    for key in TAKE_OURS:
        if key not in up_idx:
            raise SystemExit(f"upstream is missing {key}")
        if key not in our_idx:
            raise SystemExit(f"ours is missing {key}")
        s, e, _ = up_idx[key]
        edits.append((s, e, our_idx[key][2]))

    for key in TAKE_UPSTREAM:
        if key not in up_idx:
            raise SystemExit(f"upstream is missing {key}")

    for key, members in ADD_OURS.items():
        if key not in up_idx:
            raise SystemExit(f"upstream anchor missing {key}")
        chunk = []
        for m in members:
            if m in up_idx:
                print(f"  note: upstream already has {m}, skipping")
                continue
            if m not in our_idx:
                raise SystemExit(f"ours is missing {m}")
            chunk.append(our_idx[m][2])
        if chunk:
            anchor_end = up_idx[key][1]
            inserts.append((anchor_end - 1, "\n\n" + "\n\n\n".join(chunk) + "\n"))

    for key, extra in UNION_CONSTS.items():
        if key not in up_idx:
            raise SystemExit(f"upstream is missing {key}")
        s, e, text = up_idx[key]
        new_text = text
        for member in extra:
            if f'"{member}"' in new_text:
                continue
            # insert right after the opening brace of the frozenset literal
            brace = new_text.index("{")
            new_text = new_text[: brace + 1] + f' "{member}",' + new_text[brace + 1:]
        edits.append((s, e, new_text))

    # apply edits + inserts, bottom-up so line numbers stay valid
    ops = [(s, e, t) for s, e, t in edits]
    for after, text in inserts:
        ops.append((after + 1, after + 1, text))

    ops.sort(key=lambda o: (o[0], o[1]))
    out = list(up_lines)
    for s, e, text in reversed(ops):
        out[s:e] = text.splitlines()

    return "\n".join(out) + "\n"


def verify(text: str) -> None:
    tree = ast.parse(text)
    seen: dict[str, int] = {}
    dup = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            seen[node.name] = seen.get(node.name, 0) + 1
            if seen[node.name] > 1:
                dup.append(node.name)
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            ms: dict[str, int] = {}
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    ms[sub.name] = ms.get(sub.name, 0) + 1
                    if ms[sub.name] > 1:
                        dup.append(f"{node.name}.{sub.name}")
    if dup:
        raise SystemExit(f"merged file still has duplicate definitions: {dup}")
    print(f"  verify: AST OK, {len(tree.body)} top-level nodes, 0 duplicates, "
          f"{len(text.splitlines())} lines")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default="origin/main")
    ap.add_argument("--base", default="3b18403")
    ap.add_argument("--ours-ref", default="HEAD")
    ap.add_argument("--path", default="integrations/ARC-AGI-3-Kaggle-Starter/agent/my_agent.py")
    ap.add_argument("--out", default=str(DEFAULT_TARGET))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    upstream = git_show(REPO, args.upstream, args.path)
    ours = git_show(REPO, args.ours_ref, args.path)

    print(f"upstream {args.upstream}: {len(upstream.splitlines())} lines")
    print(f"ours     {args.ours_ref}: {len(ours.splitlines())} lines")
    print(f"base     {args.base}: {len(git_show(REPO, args.base, args.path).splitlines())} lines")

    merged = merge(upstream, ours)
    verify(merged)

    if args.dry_run:
        print("dry-run: nothing written")
        return 0
    out = Path(args.out)
    out.write_text(merged, encoding="utf-8", newline="\n")
    print(f"  wrote {out} ({len(merged.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
