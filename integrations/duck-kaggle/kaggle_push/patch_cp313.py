"""Rebuild lingjing-duck-harness.ipynb from the upstream share notebook.

Current Kaggle image needs:
- machine_shape NvidiaRtxPro6000
- cp313 wheelhouse (bilgisarmal/vllm031-cp313-wheelhouse)
- pip --find-links -> dataset/wheels/ when that subdir exists
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent  # integrations/duck-kaggle or integrations/duck-harness
# Prefer vendored Tufa clone next to this overlay folder.
_VENDOR = ROOT.parent / "duck-harness"
SRC_NB = (
    (_VENDOR / "taaf-duck-harness-kaggle-share.ipynb")
    if (_VENDOR / "taaf-duck-harness-kaggle-share.ipynb").is_file()
    else (ROOT / "taaf-duck-harness-kaggle-share.ipynb")
)
NB_PATH = OUT / "lingjing-duck-harness.ipynb"
META_PATH = OUT / "kernel-metadata.json"

NEW_WH = "bilgisarmal/vllm031-cp313-wheelhouse"
NEW_OWNER, NEW_SLUG = NEW_WH.split("/", 1)
OLD_WH = "driessmit1/arc3-vllm-h100-wheelhouse-v3"

FIND_LINKS_OLD = "'--find-links',\n        str(WHEELHOUSE),"
FIND_LINKS_NEW = (
    "'--find-links',\n"
    "        str((WHEELHOUSE / 'wheels') if (WHEELHOUSE / 'wheels').is_dir() else WHEELHOUSE),"
)


def _as_source(src: str) -> list[str]:
    lines = src.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    return lines


def main() -> None:
    if not SRC_NB.is_file():
        raise SystemExit(f"missing upstream notebook: {SRC_NB}")
    shutil.copy2(SRC_NB, NB_PATH)
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

    nb.setdefault("metadata", {})["kaggle"] = {
        "accelerator": "nvidiaRtx6000",
        "isInternetEnabled": False,
        "isGpuEnabled": True,
        "language": "python",
        "sourceType": "notebook",
    }

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "DATASET_SOURCES" in src and "taaf-kaggle-bundle.json" in src:
            src = re.sub(
                r"DATASET_SOURCES\s*=\s*\[[^\]]*\]",
                "DATASET_SOURCES = ["
                '"jeroencottaar/taaf-kaggle-source-share", '
                f'"{NEW_WH}", '
                '"driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"]',
                src,
                count=1,
            )
            cell["source"] = _as_source(src)

    # Notebook cell code that rewrites setup_commands.json before execution.
    # Use chr(10) so this inject itself has no raw newlines inside Python strings.
    inject = f'''
# --- lingjing patch: rewrite setup_commands for cp313 wheelhouse ---
_setup_path = BUNDLE_DIR / "setup_commands.json"
_cmds = json.loads(_setup_path.read_text(encoding="utf-8"))
_patched_cmds = []
_nl = chr(10)
_find_old = "'--find-links'," + _nl + "        str(WHEELHOUSE),"
_find_new = (
    "'--find-links'," + _nl
    + "        str((WHEELHOUSE / 'wheels') if (WHEELHOUSE / 'wheels').is_dir() else WHEELHOUSE),"
)
for _cmd in _cmds:
    if not isinstance(_cmd, str):
        _patched_cmds.append(_cmd)
        continue
    _cmd = _cmd.replace("WHEELHOUSE_OWNER = 'driessmit1'", "WHEELHOUSE_OWNER = '{NEW_OWNER}'")
    _cmd = _cmd.replace(
        "WHEELHOUSE_SLUG = 'arc3-vllm-h100-wheelhouse-v3'",
        "WHEELHOUSE_SLUG = '{NEW_SLUG}'",
    )
    if _find_old not in _cmd:
        raise RuntimeError("find-links pattern missing in setup command")
    _cmd = _cmd.replace(_find_old, _find_new)
    _patched_cmds.append(_cmd)
_cmds_file = WORKING_DIR / "_setup_commands.lingjing.json"
try:
    _setup_path.write_text(json.dumps(_patched_cmds, indent=2), encoding="utf-8")
    _cmds_file = _setup_path
except OSError:
    _cmds_file.write_text(json.dumps(_patched_cmds, indent=2), encoding="utf-8")
print("taaf.kaggle: lingjing patched setup ->", _cmds_file, "n=", len(_patched_cmds), flush=True)
assert "wheels" in _patched_cmds[0], "wheels retarget missing from patched setup"
env = _command_env()
for command in json.loads(_cmds_file.read_text(encoding="utf-8")):
'''

    pattern = re.compile(
        r"env\s*=\s*_command_env\(\)\s*"
        r"for command in json\.loads\(\(BUNDLE_DIR / [\"']setup_commands\.json[\"']\)"
        r"\.read_text\(\)\):",
        re.MULTILINE,
    )

    setup_patched = False
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "setup_commands.json" not in src or "_command_env" not in src:
            continue
        if not pattern.search(src):
            raise SystemExit("setup loop marker not found in fresh share notebook")
        src = pattern.sub(inject.rstrip() + "\n", src, count=1)
        cell["source"] = _as_source(src)
        setup_patched = True
        break
    if not setup_patched:
        raise SystemExit("setup cell not found")

    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")

    meta = {
        "id": "liaojiaxuan/lingjing-duck-harness-v1",
        "title": "lingjing-duck-harness-v1",
        "code_file": "lingjing-duck-harness.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "machine_shape": "NvidiaRtxPro6000",
        "keywords": [],
        "dataset_sources": [
            "jeroencottaar/taaf-kaggle-source-share",
            NEW_WH,
            "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot",
        ],
        "kernel_sources": [],
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "model_sources": [],
    }
    META_PATH.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    blob = "".join("".join(c.get("source", [])) for c in nb["cells"])
    assert NEW_WH in blob
    assert "lingjing patched setup" in blob
    assert FIND_LINKS_NEW.split("\n")[0] in blob or "wheels') if (WHEELHOUSE" in blob
    print("OK rebuilt", NB_PATH)
    print("datasets", meta["dataset_sources"])


if __name__ == "__main__":
    main()
