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
import json
import os
from pathlib import Path
from textwrap import dedent

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


def _collect_lingjing_files() -> list[tuple[str, str]]:
    """Return the static import closure needed by the Kaggle SmartRouter.

    Shipping every local module made the notebook exceed Kaggle's 1 MiB
    kernel-source limit and unnecessarily included tests/benchmarks. The
    roots below are the imports used by ``agent/my_agent.py``; relative and
    absolute ``lingjing_solo`` imports are followed transitively.
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

    roots = {
        "lingjing_solo",
        "lingjing_solo.core",
        "lingjing_solo.perception",
        "lingjing_solo.transfer.ceax_controller",
        "lingjing_solo.arc_transition",
    }
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

    package_shims = {
        "__init__.py": '"""Kaggle SmartRouter minimal package surface."""\n__version__ = "0.5.0"\n',
        "transfer/__init__.py": '"""Kaggle SmartRouter transfer surface."""\nfrom .ceax_controller import CeaxController\n__all__ = ["CeaxController"]\n',
        "v14/__init__.py": '"""Kaggle SmartRouter V14 transition surface."""\nfrom .arc_transition import V14ArcTransition, TransitionDiagnostics\n__all__ = ["V14ArcTransition", "TransitionDiagnostics"]\n',
    }
    out: list[tuple[str, str]] = []
    for module in sorted(seen | {"lingjing_solo"}):
        path = module_paths[module]
        rel = path.relative_to(LINGJING_SRC).as_posix()
        source = package_shims.get(rel, path.read_text(encoding="utf-8-sig"))
        out.append((rel, source))
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


def build() -> dict:
    if not AGENT_SRC.exists():
        raise SystemExit(f"Could not find {AGENT_SRC}")
    agent_body = AGENT_SRC.read_text(encoding="utf-8")
    lj_files = _collect_lingjing_files()

    install_cell = code_cell(
        "!pip install --no-index --find-links \\\n"
        "    /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels \\\n"
        "    arc-agi python-dotenv numpy"
    )

    # We write the agent to /tmp/ (not /kaggle/working/) so it does NOT appear
    # as a notebook output. Otherwise the "Submit to Competition" UI would
    # offer it as a candidate submission file alongside submission.parquet,
    # and an unlucky default selection rejects the submission.
    write_agent_cell = code_cell(
        "%%writefile /tmp/my_agent.py\n" + agent_body
    )
    write_lingjing_cell = _lingjing_write_cell(lj_files)

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
            write_agent_cell,
            run_cell,
            dummy_submission_cell,
        ],
    }
    return notebook


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(build(), indent=1), encoding="utf-8")
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
