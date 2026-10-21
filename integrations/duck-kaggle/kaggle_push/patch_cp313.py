"""Rebuild lingjing-duck-harness.ipynb for Lingjing Franzen+Qwen3.8 path.

- RTX Pro 6000 + cp313 wheelhouse
- Qwen3.8-27B-FP8 weights
- Franzen overlay dataset (patched inference modules)
- vLLM: enforce-eager + cutlass + max-num-seqs 8 + max-model-len 32k
- solver.concurrency=8 (matched to admit limit)
- MULTIMODAL_UPSCALE=10
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent  # duck-kaggle
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
MODEL_DS = "driessmit1/qwen3-8-27b-fp8-hf-017b9c7a"
MODEL_OWNER, MODEL_SLUG = MODEL_DS.split("/", 1)
SERVED_NAME = "Qwen3.8-27B-FP8"
OVERLAY_DS = "liaojiaxuan/lingjing-duck-franzen-overlay"
SOURCE_DS = "jeroencottaar/taaf-kaggle-source-share"

FIND_LINKS_NEW_HINT = "wheels') if (WHEELHOUSE"


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

    datasets_literal = (
        f'["{SOURCE_DS}", "{NEW_WH}", "{MODEL_DS}", "{OVERLAY_DS}"]'
    )

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "DATASET_SOURCES" in src and "taaf-kaggle-bundle.json" in src:
            src = re.sub(
                r"DATASET_SOURCES\s*=\s*\[[^\]]*\]",
                f"DATASET_SOURCES = {datasets_literal}",
                src,
                count=1,
            )
            cell["source"] = _as_source(src)

    inject = f'''
# --- lingjing patch: cp313 wheelhouse + Qwen3.8 + Franzen vLLM flags ---
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
    # Prefer explicit Qwen3.8 snapshot when present in the setup script.
    _cmd = _cmd.replace("MODEL_OWNER = 'driessmit1'", "MODEL_OWNER = '{MODEL_OWNER}'")
    _cmd = _cmd.replace(
        "MODEL_SLUG = 'vrfai-qwen3-6-27b-fp8-hf-snapshot'",
        "MODEL_SLUG = '{MODEL_SLUG}'",
    )
    _cmd = _cmd.replace(
        "MODEL_SLUG = 'qwen3-6-27b-fp8-hf-snapshot'",
        "MODEL_SLUG = '{MODEL_SLUG}'",
    )
    _cmd = _cmd.replace(
        "SERVED_MODEL_NAME = 'vrfai/Qwen3.6-27B-FP8'",
        "SERVED_MODEL_NAME = '{SERVED_NAME}'",
    )
    _cmd = _cmd.replace(
        "SERVED_MODEL_NAME = 'Qwen3.6-27B-FP8'",
        "SERVED_MODEL_NAME = '{SERVED_NAME}'",
    )
    # Franzen: larger board images.
    _cmd = _cmd.replace("MULTIMODAL_UPSCALE', '4'", "MULTIMODAL_UPSCALE', '10'")
    _cmd = _cmd.replace('MULTIMODAL_UPSCALE", "4"', 'MULTIMODAL_UPSCALE", "10"')
    _cmd = _cmd.replace("MULTIMODAL_UPSCALE = '4'", "MULTIMODAL_UPSCALE = '10'")
    # Upstream writes the setup_env dict with a colon: 'MULTIMODAL_UPSCALE': '4'.
    _cmd = _cmd.replace("MULTIMODAL_UPSCALE': '4'", "MULTIMODAL_UPSCALE': '10'")
    if _find_old not in _cmd:
        raise RuntimeError("find-links pattern missing in setup command")
    _cmd = _cmd.replace(_find_old, _find_new)
    _vllm_marker = "'--max-model-len'," + _nl + "        str(VLLM_MAX_MODEL_LEN),"
    _vllm_extra = (
        _vllm_marker + _nl
        + "        '--enforce-eager'," + _nl
        + "        '--linear-backend'," + _nl
        + "        'cutlass'," + _nl
        + "        '--max-num-seqs'," + _nl
        + "        '8',"
    )
    if _vllm_marker not in _cmd:
        raise RuntimeError("vLLM max-model-len marker missing in setup command")
    if "'--enforce-eager'" not in _cmd:
        _cmd = _cmd.replace(_vllm_marker, _vllm_extra, 1)
    # Ensure admit limit is 8 even if upstream already shipped enforce-eager.
    for _old_seqs in ("'16'", "'4'", "'28'", "'32'"):
        _seq_pat = "'--max-num-seqs'," + _nl + "        " + _old_seqs
        if _seq_pat in _cmd:
            _cmd = _cmd.replace(
                _seq_pat,
                "'--max-num-seqs'," + _nl + "        '8'",
                1,
            )
            break
    else:
        if "'--max-num-seqs'," not in _cmd:
            _cmd = _cmd.replace(
                _vllm_marker,
                _vllm_marker + _nl + "        '--max-num-seqs'," + _nl + "        '8'",
                1,
            )
    # 64k + 8 concurrent multimodal jobs OOMs on a single RTX Pro 6000 FP8;
    # Franzen analyzer tier that fits is 32k (same as SmartRouter ctx32768).
    _cmd = _cmd.replace(
        "VLLM_MAX_MODEL_LEN = 65536",
        "VLLM_MAX_MODEL_LEN = 32768",
    )
    if "min(int(VLLM_MAX_MODEL_LEN), 32768)" not in _cmd:
        _cmd = _cmd.replace(
            "str(VLLM_MAX_MODEL_LEN)",
            "str(min(int(VLLM_MAX_MODEL_LEN), 32768))",
            1,
        )
    # Qwen 27B FP8 on Kaggle NFS often needs >15 min to prefetch (~33GB).
    # Stock wait_for_vllm_server default is 900s — same failure mode as SmartRouter v14.
    _cmd = _cmd.replace(
        "timeout_seconds: int = 900",
        "timeout_seconds: int = 2400",
    )
    _cmd = _cmd.replace(
        "wait_for_vllm_server()",
        "wait_for_vllm_server(timeout_seconds=2400)",
    )
    # Prefer Triton GDN, but only if this vLLM build actually accepts the flag:
    # an unrecognized argument aborts the api_server before it binds the port,
    # which would waste the whole run. Probe --help once (~15s) and fall back to
    # the SmartRouter-proven config (no flag) when unsupported.
    # NOTE: the replaced print() line keeps its own 4-space indent, so continuation
    # lines below must start at 8 spaces (12 inside the nested if).
    if "'--gdn-prefill-backend'" not in _cmd:
        _cmd = _cmd.replace(
            "print('Starting vLLM OpenAI server:', ' '.join(cmd), flush=True)",
            (
                "if '--gdn-prefill-backend' not in cmd:" + _nl
                + "        _hp = subprocess.run([sys.executable, '-m'," + _nl
                + "            'vllm.entrypoints.openai.api_server', '--help']," + _nl
                + "            capture_output=True, text=True, env=vllm_env())" + _nl
                + "        if '--gdn-prefill-backend' in (_hp.stdout or '') + (_hp.stderr or ''):" + _nl
                + "            cmd.extend(['--gdn-prefill-backend', 'triton'])" + _nl
                + "            print('LINGJING_GDN_TRITON_OK', flush=True)" + _nl
                + "        else:" + _nl
                + "            print('LINGJING_GDN_TRITON_UNSUPPORTED', flush=True)" + _nl
                + "    print('Starting vLLM OpenAI server:', ' '.join(cmd), flush=True)"
            ),
            1,
        )
    _cmd = _cmd.replace(
        "os.environ.get('VLLM_USE_FLASHINFER_SAMPLER', '1')",
        "'0'",
    )
    _cmd = _cmd.replace(
        "'VLLM_USE_FLASHINFER_SAMPLER': '1'",
        "'VLLM_USE_FLASHINFER_SAMPLER': '0'",
    )
    # Upstream never sets the sampler env at all; inject it into vllm_env() and the
    # persisted setup_env so the SM12x FlashInfer top-p/top-k fallback stays silent
    # (Duck 10-13 log: "FlashInfer top-p/top-k sampling unavailable ... falling back").
    _cmd = _cmd.replace(
        "'VLLM_NO_USAGE_STATS': '1',",
        "'VLLM_NO_USAGE_STATS': '1'," + _nl + "            'VLLM_USE_FLASHINFER_SAMPLER': '0',",
    )
    # Thinking traces dominate decode time on a single GPU. Cap output too.
    _cmd = _cmd.replace(
        "'LOCAL_ANALYZER_ENABLE_THINKING': 'true'",
        "'LOCAL_ANALYZER_ENABLE_THINKING': 'false'",
    )
    _cmd = _cmd.replace(
        "'LOCAL_ANALYZER_ENABLE_THINKING': '1'",
        "'LOCAL_ANALYZER_ENABLE_THINKING': '0'",
    )
    # Avoid braces here — this block is an f-string.
    _cmd = _cmd.replace('"preserve_thinking": true', '"preserve_thinking": false')
    _cmd = _cmd.replace("'preserve_thinking': true", "'preserve_thinking': false")
    _cmd = _cmd.replace(
        "'LOCAL_ANALYZER_MAX_OUTPUT': '0'",
        "'LOCAL_ANALYZER_MAX_OUTPUT': '1536'",
    )
    _cmd = _cmd.replace(
        "'LOCAL_ANALYZER_TOOL_STEPS': '0'",
        "'LOCAL_ANALYZER_TOOL_STEPS': '3'",
    )
    _cmd = _cmd.replace(
        "'LOCAL_ANALYZER_YIELD_SECONDS': '60'",
        "'LOCAL_ANALYZER_YIELD_SECONDS': '20'",
    )
    # Admit limit already pinned to 8 above (matches solver.concurrency).
    # Torch 2.10 needs libcudart.so.13 from nvidia-cuda-runtime; pip --target often
    # leaves those .so files off the dynamic linker path. Extract + LD_LIBRARY_PATH.
    # NOTE: this whole inject block is an f-string — literal braces must be doubled.
    _vllm_env_marker = (
        "def vllm_env() -> dict[str, str]:" + _nl
        + "    env = os.environ.copy()" + _nl
        + "    existing = env.get('PYTHONPATH', '')" + _nl
        + "    env['PYTHONPATH'] = str(SITE_PACKAGES) if not existing else f'{{SITE_PACKAGES}}{{os.pathsep}}{{existing}}'" + _nl
        + "    env.update("
    )
    _vllm_env_hook = (
        "def vllm_env() -> dict[str, str]:" + _nl
        + "    env = os.environ.copy()" + _nl
        + "    existing = env.get('PYTHONPATH', '')" + _nl
        + "    env['PYTHONPATH'] = str(SITE_PACKAGES) if not existing else f'{{SITE_PACKAGES}}{{os.pathsep}}{{existing}}'" + _nl
        + "    _lib_dirs: list[str] = []" + _nl
        + "    _seen: set[str] = set()" + _nl
        + "    def _add_lib(_p: Path) -> None:" + _nl
        + "        _s = str(_p)" + _nl
        + "        if _s and _s not in _seen and Path(_s).is_dir():" + _nl
        + "            _seen.add(_s)" + _nl
        + "            _lib_dirs.append(_s)" + _nl
        + "    # torch/lib first — we copy libcudart here so the linker always finds it" + _nl
        + "    _add_lib(SITE_PACKAGES / 'torch' / 'lib')" + _nl
        + "    _add_lib(SITE_PACKAGES / '_lingjing_cuda_libs')" + _nl
        + "    _add_lib(SITE_PACKAGES / 'nvidia' / 'cu13' / 'lib')" + _nl
        + "    _add_lib(SITE_PACKAGES / 'nvidia' / 'cuda_runtime' / 'lib')" + _nl
        + "    for _hit in SITE_PACKAGES.rglob('libcudart.so*'):" + _nl
        + "        _add_lib(_hit.parent)" + _nl
        + "    _nvidia = SITE_PACKAGES / 'nvidia'" + _nl
        + "    if _nvidia.is_dir():" + _nl
        + "        for _p in _nvidia.rglob('*'):" + _nl
        + "            if _p.is_dir() and _p.name in ('lib', 'lib64'):" + _nl
        + "                _add_lib(_p)" + _nl
        + "    for _sys in (Path('/usr/local/nvidia/lib64'), Path('/usr/local/cuda/lib64')):" + _nl
        + "        _add_lib(_sys)" + _nl
        + "    _old_ld = env.get('LD_LIBRARY_PATH', '')" + _nl
        + "    env['LD_LIBRARY_PATH'] = os.pathsep.join([*_lib_dirs, *([_old_ld] if _old_ld else [])])" + _nl
        + "    print('LINGJING_LD_LIBRARY_PATH_V2', env['LD_LIBRARY_PATH'][:1200], flush=True)" + _nl
        + "    env.update("
    )
    # Force V2 even if an older LD hook was somehow present.
    if "LINGJING_LD_LIBRARY_PATH_V2" not in _cmd:
        if "LINGJING_LD_LIBRARY_PATH" in _cmd and _vllm_env_marker not in _cmd:
            # Already patched once; leave as-is and rely on torch/lib cudart copy.
            pass
        elif _vllm_env_marker not in _cmd:
            raise RuntimeError("vllm_env marker missing in setup command")
        else:
            _cmd = _cmd.replace(_vllm_env_marker, _vllm_env_hook, 1)
    # Persist LD_LIBRARY_PATH into taaf setup env for the solver process.
    _setup_env_marker = (
        "setup_env = {{" + _nl
        + "    'USE_TF': '0',"
    )
    _setup_env_hook = (
        "setup_env = {{" + _nl
        + "    'LD_LIBRARY_PATH': vllm_env().get('LD_LIBRARY_PATH', '')," + _nl
        + "    'USE_TF': '0',"
    )
    if "'LD_LIBRARY_PATH': vllm_env()" not in _cmd:
        if _setup_env_marker not in _cmd:
            raise RuntimeError("setup_env marker missing in setup command")
        _cmd = _cmd.replace(_setup_env_marker, _setup_env_hook, 1)
    # Ensure cudart + cutlass before launching vLLM.
    _start_marker = (
        "def start_vllm_server() -> None:" + _nl
        + "    install_vllm_wheelhouse()"
    )
    _start_hook = (
        "def start_vllm_server() -> None:" + _nl
        + "    install_vllm_wheelhouse()" + _nl
        + "    import shutil as _shutil" + _nl
        + "    import zipfile as _zipfile" + _nl
        + "    _wh = (WHEELHOUSE / 'wheels') if (WHEELHOUSE / 'wheels').is_dir() else WHEELHOUSE" + _nl
        + "    _pkgs = []" + _nl
        + "    for _whl in sorted(_wh.glob('nvidia*.whl')):" + _nl
        + "        _stem = _whl.name.split('-')[0].replace('_', '-')" + _nl
        + "        if _stem not in _pkgs:" + _nl
        + "            _pkgs.append(_stem)" + _nl
        + "    print('LINGJING_NVIDIA_PKGS', _pkgs, flush=True)" + _nl
        + "    if _pkgs:" + _nl
        + "        subprocess.run([" + _nl
        + "            sys.executable, '-m', 'pip', 'install', '--no-index'," + _nl
        + "            '--find-links', str(_wh), '--target', str(SITE_PACKAGES)," + _nl
        + "            '--upgrade', '--ignore-installed', '--only-binary', ':all:'," + _nl
        + "            '--no-compile', '--disable-pip-version-check', '--no-warn-conflicts'," + _nl
        + "            *_pkgs," + _nl
        + "        ], check=False)" + _nl
        + "    print('CUTLASS_DSL_INSTALL_DONE', flush=True)" + _nl
        + "    # Put every NVIDIA .so next to torch so the dynamic linker finds cupti/cudart/cublas." + _nl
        + "    _torch_lib = SITE_PACKAGES / 'torch' / 'lib'" + _nl
        + "    _torch_lib.mkdir(parents=True, exist_ok=True)" + _nl
        + "    _nlink = 0" + _nl
        + "    _nvidia_root = SITE_PACKAGES / 'nvidia'" + _nl
        + "    if _nvidia_root.is_dir():" + _nl
        + "        for _so in _nvidia_root.rglob('*'):" + _nl
        + "            if not _so.is_file() or '.so' not in _so.name:" + _nl
        + "                continue" + _nl
        + "            _dest = _torch_lib / _so.name" + _nl
        + "            if _dest.exists() or _dest.is_symlink():" + _nl
        + "                continue" + _nl
        + "            try:" + _nl
        + "                _dest.symlink_to(_so)" + _nl
        + "            except OSError:" + _nl
        + "                _shutil.copy2(_so, _dest)" + _nl
        + "            _nlink += 1" + _nl
        + "    if not any(_torch_lib.glob('libcupti.so*')):" + _nl
        + "        for _whl in sorted(_wh.glob('nvidia_cuda_cupti*.whl')):" + _nl
        + "            print('LINGJING_EXTRACT_CUPTI', _whl, flush=True)" + _nl
        + "            with _zipfile.ZipFile(_whl) as _zf:" + _nl
        + "                for _name in _zf.namelist():" + _nl
        + "                    _base = _name.split('/')[-1]" + _nl
        + "                    if _base.startswith('libcupti'):" + _nl
        + "                        (_torch_lib / _base).write_bytes(_zf.read(_name))" + _nl
        + "                        print('LINGJING_CUPTI_WROTE', _base, flush=True)" + _nl
        + "    print('LINGJING_SO_LINK', _nlink, 'cupti', any(_torch_lib.glob('libcupti.so*')), 'cudart', any(_torch_lib.glob('libcudart.so*')), flush=True)" + _nl
        + "    if not any(_torch_lib.glob('libcudart.so*')) or not any(_torch_lib.glob('libcupti.so*')):" + _nl
        + "        raise RuntimeError('LINGJING_CUDA_LIBS_MISSING torch/lib lacks cudart or cupti')" + _nl
        + "    _probe = subprocess.run(" + _nl
        + "        [sys.executable, '-c', 'import torch; print(torch.__version__, torch.version.cuda)']," + _nl
        + "        env=vllm_env(), text=True, capture_output=True," + _nl
        + "    )" + _nl
        + "    print('LINGJING_TORCH_PROBE', _probe.returncode, _probe.stdout.strip(), (_probe.stderr or '')[-500:], flush=True)" + _nl
        + "    if _probe.returncode != 0:" + _nl
        + "        raise RuntimeError('LINGJING_TORCH_IMPORT_FAIL ' + (_probe.stderr or _probe.stdout))"
    )
    if "LINGJING_SO_LINK" not in _cmd:
        if _start_marker not in _cmd:
            raise RuntimeError("start_vllm_server/install marker missing")
        _cmd = _cmd.replace(_start_marker, _start_hook, 1)
    _patched_cmds.append(_cmd)
_cmds_file = WORKING_DIR / "_setup_commands.lingjing.json"
try:
    _setup_path.write_text(json.dumps(_patched_cmds, indent=2), encoding="utf-8")
    _cmds_file = _setup_path
except OSError:
    _cmds_file.write_text(json.dumps(_patched_cmds, indent=2), encoding="utf-8")
print("taaf.kaggle: lingjing patched setup ->", _cmds_file, "n=", len(_patched_cmds), flush=True)
assert "wheels" in _patched_cmds[0], "wheels retarget missing from patched setup"

# Apply Franzen overlay (patched inference modules) ahead of the stock bundle.
import shutil as _shutil
import zipfile as _zipfile
_overlay_root = None
for _cand in (
    Path("/kaggle/input/lingjing-duck-franzen-overlay"),
    Path("/kaggle/input/datasets/liaojiaxuan/lingjing-duck-franzen-overlay"),
):
    if _cand.is_dir():
        _overlay_root = _cand
        break
if _overlay_root is None:
    for _p in Path("/kaggle/input").rglob("LINGJING_OVERLAY.txt"):
        _overlay_root = _p.parent
        break
if _overlay_root is None:
    raise SystemExit("LINGJING_OVERLAY_MISSING attach liaojiaxuan/lingjing-duck-franzen-overlay")
_src_inference = _overlay_root / "inference"
if not _src_inference.is_dir():
    _zip_path = _overlay_root / "inference.zip"
    if not _zip_path.is_file():
        raise SystemExit("LINGJING_OVERLAY_NO_INFERENCE path=%s" % _overlay_root)
    _unzip = WORKING_DIR / "_lingjing_overlay_unzip"
    if _unzip.exists():
        _shutil.rmtree(_unzip)
    _unzip.mkdir(parents=True)
    with _zipfile.ZipFile(_zip_path) as _zf:
        _zf.extractall(_unzip)
    _src_inference = _unzip / "inference"
    if not _src_inference.is_dir():
        # zip may contain files at root of archive
        _hits = list(_unzip.rglob("action_names.py"))
        if not _hits:
            raise SystemExit("LINGJING_OVERLAY_BAD_ZIP")
        _src_inference = _hits[0].parents[1]  # .../inference/agent/file -> inference
# Merge patched modules onto a full copy of stock inference.
# Putting the partial overlay alone on sys.path hides inference.utils.
_stock_inference = None
for _entry in source_entries:
    _cand_inf = Path(_entry) / "inference"
    if (_cand_inf / "utils").is_dir() and (_cand_inf / "agent").is_dir():
        _stock_inference = _cand_inf
        break
if _stock_inference is None:
    raise SystemExit("LINGJING_STOCK_INFERENCE_MISSING")
_overlay_dst = WORKING_DIR / "lingjing_franzen_inference"
if _overlay_dst.exists():
    _shutil.rmtree(_overlay_dst)
_shutil.copytree(_stock_inference, _overlay_dst / "inference")
_copied = 0
for _f in _src_inference.rglob("*"):
    if not _f.is_file():
        continue
    _rel = _f.relative_to(_src_inference)
    _dest = _overlay_dst / "inference" / _rel
    _dest.parent.mkdir(parents=True, exist_ok=True)
    _shutil.copy2(_f, _dest)
    _copied += 1
# Heal nested triple-quote docstrings that break outer raw/setup templates.
_qq = chr(34) * 3
_sq = chr(39) * 3
_heal = [
    (
        _overlay_dst / "inference" / "agent" / "python_tool_sandbox.py",
        _qq + "Compact object-level diff (Franzen). Defaults to previous" + chr(8594) + "current." + _qq,
        _sq + "Compact object-level diff (Franzen). Defaults to previous->current." + _sq,
    ),
    (
        _overlay_dst / "inference" / "agent" / "python_tool_sandbox.py",
        _qq + "Compact object-level diff (Franzen). Defaults to previous->current." + _qq,
        _sq + "Compact object-level diff (Franzen). Defaults to previous->current." + _sq,
    ),
    (
        _overlay_dst / "inference" / "framework" / "kaggle.py",
        _qq + "FlashInfer GDN does `import cutlass`; wheelhouse ships nvidia-cutlass-dsl*." + _qq,
        _sq + "FlashInfer GDN does import cutlass; wheelhouse ships nvidia-cutlass-dsl*." + _sq,
    ),
]
_healed = 0
for _path, _old, _new in _heal:
    if not _path.is_file():
        continue
    _txt = _path.read_text(encoding="utf-8")
    if _old in _txt:
        _path.write_text(_txt.replace(_old, _new), encoding="utf-8")
        _healed += 1
print("taaf.kaggle: overlay docstring heals", _healed, flush=True)
_sys_parent = str(_overlay_dst)
if _sys_parent not in sys.path:
    sys.path.insert(0, _sys_parent)
if pth_path.is_file():
    _prev = pth_path.read_text(encoding="utf-8")
    if _sys_parent not in _prev.splitlines():
        pth_path.write_text(_sys_parent + chr(10) + _prev, encoding="utf-8")
(_overlay_dst / "LINGJING_OVERLAY_APPLIED.txt").write_text(
    ("overlay=%s" + chr(10) + "stock=%s" + chr(10) + "copied=%s" + chr(10) + "served={SERVED_NAME}" + chr(10))
    % (_overlay_root, _stock_inference, _copied),
    encoding="utf-8",
)
print("taaf.kaggle: Franzen overlay merged", _overlay_dst, "files", _copied, flush=True)
os.environ["LINGJING_DISABLE_WORLD_MODEL"] = "1"
os.environ["MULTIMODAL_UPSCALE"] = "10"

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

    # Concurrency 8 matched to vLLM --max-num-seqs 8. deploy_target.pkl often
    # ships ~28 which starves a single GPU; 4 was too timid (9h wall, low cover).
    # Fail-fast analyzer timeout keeps one stuck game from blocking the pool.
    custom_patched = False
    custom_src = (
        "# lingjing: single-GPU score budget (8-way + 32k ctx, thinking off).\n"
        "import os as _lj_os\n"
        "_lj_os.environ['LOCAL_ANALYZER_ENABLE_THINKING'] = '0'\n"
        "_lj_os.environ['LOCAL_ANALYZER_MAX_OUTPUT'] = '1536'\n"
        "_lj_os.environ['LOCAL_ANALYZER_TOOL_STEPS'] = '3'\n"
        "_lj_os.environ['LOCAL_ANALYZER_YIELD_SECONDS'] = '20'\n"
        "_lj_os.environ['LOCAL_ANALYZER_CONTEXT_WINDOW'] = '32768'\n"
        "bm.solver.concurrency = 8\n"
        "bm.solver.max_runtime_s_per_game = 5400\n"
        "bm.solver.analyzer_timeout = 240\n"
        "import inference.agent.tool_agent as _lj_ta\n"
        "_lj_ta._LOCAL_ANALYZER_ENABLE_THINKING = False\n"
        "_lj_ta._LOCAL_ANALYZER_MAX_OUTPUT = 1536\n"
        "_lj_ta._LOCAL_ANALYZER_TOOL_STEPS = 3\n"
        "_lj_ta._LOCAL_ANALYZER_YIELD_SECONDS = 20.0\n"
        "print('LINGJING_BUDGET', bm.solver.concurrency, bm.solver.max_runtime_s_per_game,"
        " bm.solver.analyzer_timeout, 'think', _lj_ta._LOCAL_ANALYZER_ENABLE_THINKING,"
        " 'ctx', _lj_os.environ.get('LOCAL_ANALYZER_CONTEXT_WINDOW'), flush=True)\n"
    )
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "bm.games" not in src or "before the run starts" not in src:
            continue
        if "LINGJING_BUDGET" in src:
            custom_patched = True
            break
        cell["source"] = _as_source(custom_src + src)
        custom_patched = True
        break
    if not custom_patched:
        raise SystemExit("customization cell not found")

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
        "dataset_sources": [SOURCE_DS, NEW_WH, MODEL_DS, OVERLAY_DS],
        "kernel_sources": [],
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "model_sources": [],
    }
    META_PATH.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    blob = "".join("".join(c.get("source", [])) for c in nb["cells"])
    assert NEW_WH in blob
    assert MODEL_DS in blob
    assert OVERLAY_DS in blob
    assert "lingjing patched setup" in blob
    assert "Franzen overlay" in blob
    assert "LINGJING_BUDGET" in blob
    assert FIND_LINKS_NEW_HINT in blob
    assert '"preserve_thinking": true' in blob  # replace source present in inject
    assert '"preserve_thinking": false' in blob
    assert "''{\"preserve_thinking\"" not in blob
    print("OK rebuilt", NB_PATH)
    print("datasets", meta["dataset_sources"])


if __name__ == "__main__":
    main()
