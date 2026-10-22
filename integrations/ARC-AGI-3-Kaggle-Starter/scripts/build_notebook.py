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
import json
import os
import zlib
from pathlib import Path
from textwrap import dedent

# ─────────────────────────────────────────────────────────────────────────────
# Accelerator: env LINGJING_KAGGLE_ACCEL overrides (cpu|t4|p100|rtx6000).
# Default rtx6000 for hybrid LLM path; set LINGJING_KAGGLE_ACCEL=cpu for floors-only.
# ─────────────────────────────────────────────────────────────────────────────
ACCELERATOR = str(os.environ.get("LINGJING_KAGGLE_ACCEL", "rtx6000")).strip().lower()

# SELFLEARN_LLM is the authoritative master switch of the agent:
#   1 = 大模型 planner（lexicon 结构化视图 + 核内千问）
#   0 = 原策略（engine solver + 校准计划 + CEAX 兜底，纯算法，不起 vLLM）
# Build two notebooks for an A/B by flipping this one variable:
#   LINGJING_SELFLEARN_LLM=1 python scripts/build_notebook.py   # 大模型版
#   LINGJING_SELFLEARN_LLM=0 python scripts/build_notebook.py   # 原策略版
SELFLEARN_LLM = str(os.environ.get("LINGJING_SELFLEARN_LLM", "1")).strip()
if SELFLEARN_LLM not in ("0", "1"):
    raise SystemExit("LINGJING_SELFLEARN_LLM must be '0' or '1'")

# LINGJING_BENCH=1 adds an offline self-benchmark cell: it plays the
# competition's public environment files (25 games) inside the kernel, so a
# plain "Save & Run All" measures the agent instead of doing nothing.
# Off by default -- the submission notebook stays lean.
BENCH = str(os.environ.get("LINGJING_BENCH", "0")).strip() == "1"
# Action cap for the offline self-benchmark (per game). 400 matches
# scripts/benchmark_all_games.py locally; a Kaggle LLM run cannot afford 400
# steps on all 25 games, so the A/B builds use a smaller cap for BOTH arms.
BENCH_MAX_STEPS = int(os.environ.get("LINGJING_BENCH_MAX_STEPS", "400") or 400)


def _strip_comment_only_lines(source: str) -> str:
    """Reduce notebook payload without changing executable Python statements."""
    lines = [line for line in source.splitlines() if not line.lstrip().startswith("#")]
    return "\n".join(lines) + ("\n" if source.endswith("\n") else "")

# Internal mapping; don't edit unless Kaggle adds new options.
# `name` → notebook metadata.kaggle.accelerator (legacy)
# `machine_shape` → kernel-metadata.json field honored by Kaggle CLI push
_ACCELERATORS = {
    "cpu":     {"name": "none",                 "gpu": False, "machine_shape": ""},
    "t4":      {"name": "nvidiaTeslaT4",        "gpu": True,  "machine_shape": "NvidiaTeslaT4"},
    "p100":    {"name": "nvidiaTeslaP100",      "gpu": True,  "machine_shape": "NvidiaTeslaP100"},
    "rtx6000": {"name": "nvidiaRtx6000",        "gpu": True,  "machine_shape": "NvidiaRtxPro6000"},
}

# Duck-compatible offline datasets (Qwen 3.6 27B FP8 + vLLM wheelhouse).
_LLM_DATASET_SOURCES = [
    # Kaggle notebook image is CPython 3.13 — need cp313 wheels (H100 house is cp312).
    "bilgisarmal/vllm031-cp313-wheelhouse",
    "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot",
]

ROOT = Path(__file__).resolve().parents[1]
AGENT_SRC = ROOT / "agent" / "my_agent.py"
LLM_POLICY_SRC = ROOT / "agent" / "llm_policy.py"
EPISODE_SRC = ROOT / "agent" / "llm_episode.py"
LINGJING_SRC = Path(
    os.environ.get("LINGJING_SRC", str(ROOT.parents[1] / "lingjing_solo"))
).expanduser().resolve()
# Repo-root arc_adaptor/lexicon (Franzen structured view). Starter is
# integrations/ARC-AGI-3-Kaggle-Starter → parents[1] is 灵境战队AGI课题探索.
LEXICON_SRC = Path(
    os.environ.get(
        "LINGJING_LEXICON_SRC",
        str(ROOT.parents[1] / "arc_adaptor" / "lexicon"),
    )
).expanduser().resolve()
NOTEBOOK_PATH = Path(
    os.environ.get(
        "LINGJING_NOTEBOOK_OUT",
        str(ROOT / "notebooks" / "submission.ipynb"),
    )
).expanduser().resolve()
METADATA_PATH = ROOT / "notebooks" / "kernel-metadata.json"
# 32k is the Duck/Franzen analyzer tier on RTX Pro 6000 96GB FP8: enough for
# objects+NATIVE+RULES+PREDICTED without the KV blow-up of 64k under concurrency.
DEFAULT_VLLM_MAX_LEN = "32768"
DEFAULT_VLLM_MAX_SEQS = "8"
DEFAULT_VLLM_MEM = "0.90"


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
        source = _strip_comment_only_lines(source)
        out.append((rel, source))
    if not any(r.endswith(".py") for r, _ in out):
        raise SystemExit(f"No .py files found under {LINGJING_SRC}")
    return out


def _packed_files_cell(
    files: list[tuple[str, str]],
    *,
    root: str,
    label: str,
) -> dict:
    """Write files from a zlib blob so the kernel stays under Kaggle's 1MB cap."""
    raw = json.dumps(dict(files), ensure_ascii=False).encode("utf-8")
    blob = base64.b64encode(zlib.compress(raw, 9)).decode("ascii")
    source = (
        "import base64, json, zlib\n"
        "from pathlib import Path\n"
        f"_ROOT = Path({root!r})\n"
        "_ROOT.mkdir(parents=True, exist_ok=True)\n"
        f"_FILES = json.loads(zlib.decompress(base64.b64decode({blob!r})).decode('utf-8'))\n"
        "for _rel, _src in _FILES.items():\n"
        "    _p = _ROOT / _rel\n"
        "    _p.parent.mkdir(parents=True, exist_ok=True)\n"
        "    _p.write_text(_src, encoding='utf-8')\n"
        f"print({label!r}, len(_FILES), _ROOT)\n"
    )
    return code_cell(source)


def _lingjing_write_cell(files: list[tuple[str, str]]) -> dict:
    """One notebook cell that materializes the lingjing_solo package under /tmp."""
    return _packed_files_cell(files, root="/tmp/lingjing_solo", label="[lingjing_solo] wrote")


def _collect_lexicon_files() -> list[tuple[str, str]]:
    """Ship arc_adaptor/lexicon into the notebook (structured Franzen view)."""
    if not LEXICON_SRC.is_dir():
        raise SystemExit(
            f"Could not find lexicon at {LEXICON_SRC}. "
            "Expected repo arc_adaptor/lexicon from feat/llm-structured-view."
        )
    out: list[tuple[str, str]] = [
        ("__init__.py", '"""ARC-AGI-3 adaptor surface (lexicon / structured view)."""\n'),
    ]
    for path in sorted(LEXICON_SRC.glob("*.py")):
        out.append((f"lexicon/{path.name}", path.read_text(encoding="utf-8")))
    if len(out) < 2:
        raise SystemExit(f"No lexicon .py files under {LEXICON_SRC}")
    return out


def _lexicon_write_cell(files: list[tuple[str, str]]) -> dict:
    return _packed_files_cell(files, root="/tmp/arc_adaptor", label="[arc_adaptor] wrote")


def build() -> dict:
    if not AGENT_SRC.exists():
        raise SystemExit(f"Could not find {AGENT_SRC}")
    # Merge resolution (v27 upstream-merge round):
    #   ours  -> ship agent/llm_policy.py + agent/llm_episode.py into the
    #            notebook (the whole-qwen LLM chain needs both).
    #   theirs-> strip comment-only lines from the agent body (keeps the
    #            notebook under Kaggle's 1 MiB kernel-source limit).
    # Both are needed, so take the union.
    if not LLM_POLICY_SRC.exists():
        raise SystemExit(f"Could not find {LLM_POLICY_SRC}")
    agent_body = _strip_comment_only_lines(AGENT_SRC.read_text(encoding="utf-8"))
    llm_policy_body = LLM_POLICY_SRC.read_text(encoding="utf-8")
    episode_body = EPISODE_SRC.read_text(encoding="utf-8")
    lj_files = _collect_lingjing_files()
    ax_files = _collect_lexicon_files()
    use_gpu = bool(_ACCELERATORS.get(ACCELERATOR, {}).get("gpu"))
    # GPU builds enable local vLLM by default; CPU stays fail-closed off.
    # SELFLEARN_LLM=0 forces the pure-algorithm route: no model server at all.
    llm_env_default = ("on" if use_gpu else "off") if SELFLEARN_LLM == "1" else "off"
    selflearn = SELFLEARN_LLM
    vllm_max_len = os.environ.get("LINGJING_VLLM_MAX_LEN", DEFAULT_VLLM_MAX_LEN)
    vllm_max_seqs = os.environ.get("LINGJING_VLLM_MAX_SEQS", DEFAULT_VLLM_MAX_SEQS)
    vllm_mem = os.environ.get("LINGJING_VLLM_MEM", DEFAULT_VLLM_MEM)

    install_cell = code_cell(
        "import subprocess, sys\n"
        "subprocess.run([\n"
        "    sys.executable, '-m', 'pip', 'install', '--no-index',\n"
        "    '--find-links', '/kaggle/input/arc-prize-2026-arc-agi-3/arc_agi_3_wheels',\n"
        "    'arc-agi', 'python-dotenv', 'numpy'\n"
        "], check=True)"
    )

    # We write the agent to /tmp/ (not /kaggle/working/) so it does NOT appear
    # as a notebook output. Otherwise the "Submit to Competition" UI would
    # offer it as a candidate submission file alongside submission.parquet,
    # and an unlucky default selection rejects the submission.
    write_agent_cell = code_cell(
        "%%writefile /tmp/my_agent.py\n" + agent_body
    )
    write_llm_policy_cell = code_cell(
        "%%writefile /tmp/llm_policy.py\n" + llm_policy_body
    )
    write_episode_cell = code_cell(
        "%%writefile /tmp/llm_episode.py\n" + episode_body
    )
    write_lingjing_cell = _lingjing_write_cell(lj_files)
    write_lexicon_cell = _lexicon_write_cell(ax_files)

    vllm_cell_source = dedent(
        f"""\
        import os, glob, subprocess, time, urllib.request
        from pathlib import Path

        # Singleton vLLM for SmartRouter LLM policy.
        # GPU submissions keep LINGJING_LLM=on. A missing wheel, a CUDA mismatch,
        # or a server that never opens /v1/models fails the cell instead of
        # silently scoring with the LLM switched off.
        # SELFLEARN_LLM is the master switch (1 = 大模型, 0 = 原策略/纯算法).
        os.environ['SELFLEARN_LLM'] = '{selflearn}'
        os.environ['LINGJING_LLM'] = '{llm_env_default}'
        os.environ.setdefault('LINGJING_LLM_URL', 'http://127.0.0.1:1234/v1')
        os.environ.setdefault('LINGJING_LLM_MODEL', 'vrfai/Qwen3.6-27B-FP8')
        # Planner view is text (objects/NATIVE/RULES). Vision stays off so the
        # 32k window is spent on the structured board, not a PNG.
        os.environ['LINGJING_LLM_VISION'] = 'off'
        os.environ.setdefault('LINGJING_LLM_PRECEAX', 'on')
        os.environ['LINGJING_LLM_WHOLE'] = 'on'
        os.environ['LINGJING_LLM_VIEW'] = 'planner'
        os.environ['LINGJING_LLM_BATCH'] = '4'
        os.environ['LINGJING_LLM_EVERY'] = '1'
        os.environ['LINGJING_LLM_NATIVE'] = '1'
        os.environ['SELFLEARN_LLM_MAXCHARS'] = '12000'
        os.environ['SELFLEARN_LLM_THINK'] = '0'
        os.environ['SELFLEARN_LLM_MAXTOK'] = '512'
        os.environ['LINGJING_LLM_MAX_CALLS'] = '800'
        os.environ['LINGJING_LLM_STRIDE'] = '1'
        os.environ.setdefault('LINGJING_VLLM_MAX_LEN', '{vllm_max_len}')
        os.environ.setdefault('LINGJING_VLLM_MAX_SEQS', '{vllm_max_seqs}')
        os.environ.setdefault('LINGJING_VLLM_MEM', '{vllm_mem}')
        os.environ['LINGJING_LEXICON_DIR'] = '/tmp/arc_adaptor'
        _require_llm = os.environ['LINGJING_LLM'].lower() in ('1', 'on', 'true', 'yes')

        def _find_input(slug_part: str):
            root = Path('/kaggle/input')
            if not root.is_dir():
                return None
            for p in root.rglob('*'):
                if p.is_dir() and slug_part in p.name:
                    return p
            return None

        _wheel = (
            _find_input('vllm031') or _find_input('arc3-vllm')
            or _find_input('wheelhouse')
        )
        _model = _find_input('qwen3-6-27b') or _find_input('qwen') or _find_input('vrfai')
        _llm_ready = False
        if _require_llm and not (_wheel and _model):
            raise SystemExit(
                'VLLM_INPUT_MISSING wheel=%s model=%s' % (_wheel, _model)
            )
        if _require_llm:
            _links = Path(_wheel) / 'wheels' if (Path(_wheel) / 'wheels').is_dir() else Path(_wheel)
            req = Path(_wheel) / 'requirements.lock'
            if not req.is_file():
                req = Path(_wheel) / 'requirements.txt'
            if req.is_file():
                subprocess.check_call([
                    'pip', 'install', '--no-index', '--find-links', str(_links),
                    '-r', str(req),
                ])
            # The Kaggle image already has torchaudio==2.11.0+cu128. Pip treats
            # that as the same version as the wheelhouse's 2.11.0+cu130 and
            # skips it, then vLLM dies on the CUDA 13.0 vs 12.8 check.
            subprocess.check_call([
                'pip', 'install', '--no-index', '--find-links', str(_links),
                '--force-reinstall', '--no-deps',
                'torchaudio==2.11.0', 'torchvision==0.28.0', 'torchcodec==0.17.0',
            ])
            # Lockfile replaces numpy/pydantic/dotenv. Put the competition
            # agent stack back on top without downgrading the vLLM torch build.
            _comp = '/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels'
            subprocess.check_call([
                'pip', 'install', '--no-index', '--find-links', _comp,
                'arc-agi', 'python-dotenv',
            ])
            _probe = subprocess.run(
                [
                    'python', '-c',
                    'import torch, torchaudio, dotenv, arcengine, vllm; '
                    'print(\"ENV_OK\", torch.__version__, torch.version.cuda, torchaudio.__version__, arcengine.__name__)',
                ],
                capture_output=True, text=True,
            )
            print(_probe.stdout, _probe.stderr, flush=True)
            if _probe.returncode != 0:
                raise SystemExit('ENV_PROBE_FAIL')
            logf = open('/kaggle/working/vllm_server.log', 'w')
            # RTX Pro 6000 is SM 12.0. FlashInfer's default FP8 GEMM tries to
            # JIT sm_120 and dies when nvcc is older than CUDA 12.9. Cutlass
            # is the backend that actually boots this FP8 checkpoint there.
            os.environ['VLLM_USE_FLASHINFER_SAMPLER'] = '0'
            cmd = [
                'python', '-m', 'vllm.entrypoints.openai.api_server',
                '--model', str(_model),
                '--served-model-name', os.environ['LINGJING_LLM_MODEL'],
                '--host', '127.0.0.1', '--port', '1234',
                '--max-model-len', os.environ.get('LINGJING_VLLM_MAX_LEN', '{vllm_max_len}'),
                '--tensor-parallel-size', '1',
                '--gpu-memory-utilization',
                os.environ.get('LINGJING_VLLM_MEM', '{vllm_mem}'),
                '--linear-backend', 'cutlass',
                # Qwen3.6 is a GDN/Mamba hybrid. On the 96GB RTX Pro 6000 the
                # default max_num_seqs=1024 asks for more Mamba blocks than the
                # leftover memory can hold (v10 died at 999). SmartRouter sends
                # one request at a time; 8 sequences leaves KV headroom for 32k.
                # Eager skips the CUDA-graph capture that enforces that limit.
                '--max-num-seqs', os.environ.get('LINGJING_VLLM_MAX_SEQS', '{vllm_max_seqs}'),
                '--enforce-eager',
            ]
            subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
            # 33GB FP8 weights on Kaggle NFS took longer than 15 minutes
            # (v14 was killed with VLLM_NOT_READY while prefetch was still running).
            for _i in range(480):
                try:
                    urllib.request.urlopen(
                        'http://127.0.0.1:1234/v1/models', timeout=2
                    )
                    _llm_ready = True
                    break
                except Exception:
                    if _i % 12 == 0:
                        print('VLLM_WAIT', _i * 5, 's', flush=True)
                    time.sleep(5)
            print('VLLM_READY', _llm_ready, 'model', _model, 'wheel', _wheel, flush=True)
            if not _llm_ready:
                try:
                    print(Path('/kaggle/working/vllm_server.log').read_text(encoding='utf-8', errors='replace')[-4000:], flush=True)
                except Exception:
                    pass
                raise SystemExit('VLLM_NOT_READY')
            import json as _json
            _probe_body = _json.dumps({{
                'model': os.environ['LINGJING_LLM_MODEL'],
                'messages': [{{'role': 'user', 'content': 'Reply with the single word READY'}}],
                'max_tokens': 8,
                'temperature': 0,
                'chat_template_kwargs': {{'enable_thinking': False}},
            }}).encode()
            _probe_req = urllib.request.Request(
                os.environ['LINGJING_LLM_URL'].rstrip('/') + '/chat/completions',
                data=_probe_body,
                headers={{'Content-Type': 'application/json'}},
                method='POST',
            )
            try:
                with urllib.request.urlopen(_probe_req, timeout=120) as _probe_resp:
                    _probe_status = getattr(_probe_resp, 'status', 200)
                    _probe_resp.read(200)
                if int(_probe_status) != 200:
                    raise SystemExit('VLLM_PROBE_FAIL status=%s' % _probe_status)
            except SystemExit:
                raise
            except Exception as _probe_exc:
                raise SystemExit('VLLM_PROBE_FAIL %s' % _probe_exc)
            print('VLLM_PROBE_OK', os.environ['LINGJING_LLM_MODEL'], flush=True)
            _fp = {{
                'vllm_ready': True,
                'probe': 'ok',
                'model': os.environ['LINGJING_LLM_MODEL'],
                'enable_internet': False,
                'required_tag': 'llm-whole-qwen',
                'selflearn_llm': os.environ.get('SELFLEARN_LLM', '0'),
                'llm_whole': os.environ['LINGJING_LLM_WHOLE'],
                'llm_view': os.environ.get('LINGJING_LLM_VIEW', 'planner'),
                'max_model_len': os.environ.get('LINGJING_VLLM_MAX_LEN', '{vllm_max_len}'),
                'max_calls': os.environ['LINGJING_LLM_MAX_CALLS'],
            }}
            Path('/kaggle/working/PHASE_B_FINGERPRINT.json').write_text(
                _json.dumps(_fp), encoding='utf-8'
            )
            print('PHASE_B_FINGERPRINT', _json.dumps(_fp), flush=True)
        else:
            print('VLLM_SKIP selflearn_llm={selflearn} llm={llm_env_default}', flush=True)
        open('/kaggle/working/LINGJING_LLM.txt', 'w').write(os.environ['LINGJING_LLM'])
        open('/kaggle/working/SELFLEARN_LLM.txt', 'w').write(os.environ['SELFLEARN_LLM'])
        """
    )
    vllm_cell = code_cell(vllm_cell_source)

    run_cell_source = dedent(
        f"""\
        import os

        if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            # Wait for the gateway sidecar to be ready.
            !curl --fail --retry 999 --retry-all-errors --retry-delay 5 \\
                  --retry-max-time 600 http://gateway:8001/api/games

            # Copy the framework into a writable location.
            !cp -r /kaggle/input/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents \\
                   /kaggle/working/ARC-AGI-3-Agents

            # Drop our agent + llm_policy + lingjing_solo into the framework tree.
            !cp /tmp/my_agent.py \\
                /kaggle/working/ARC-AGI-3-Agents/agents/templates/my_agent.py
            !cp /tmp/llm_policy.py \\
                /kaggle/working/ARC-AGI-3-Agents/agents/templates/llm_policy.py
            !cp /tmp/llm_episode.py \\
                /kaggle/working/ARC-AGI-3-Agents/agents/templates/llm_episode.py
            !rm -rf /kaggle/working/ARC-AGI-3-Agents/lingjing_solo
            !cp -r /tmp/lingjing_solo \\
                /kaggle/working/ARC-AGI-3-Agents/lingjing_solo
            !rm -rf /kaggle/working/ARC-AGI-3-Agents/arc_adaptor
            !cp -r /tmp/arc_adaptor \\
                /kaggle/working/ARC-AGI-3-Agents/arc_adaptor

            # Register MyAgent FIRST (slim __init__) — fix0 order: rewrite before import.
            with open('/kaggle/working/ARC-AGI-3-Agents/agents/__init__.py', 'w') as f:
                f.write(\"\"\"from typing import Type
        from dotenv import load_dotenv
        from .agent import Agent, Playback
        from .swarm import Swarm
        from .templates.random_agent import Random
        from .templates.my_agent import MyAgent

        load_dotenv()

        AVAILABLE_AGENTS: dict[str, Type[Agent]] = {{
            'random': Random,
            'myagent': MyAgent,
        }}
        \"\"\")

            # Fail loud if wrong agent landed (SmartRouter fingerprint).
            !python -c "import sys; sys.path.insert(0,'/kaggle/working/ARC-AGI-3-Agents'); sys.path.insert(0,'/kaggle/working/ARC-AGI-3-Agents/arc_adaptor'); from agents.templates.my_agent import BUILD_TAG, MyAgent; assert BUILD_TAG.startswith('smart-router'), BUILD_TAG; assert 'llm-structured-view' in BUILD_TAG, BUILD_TAG; open('/kaggle/working/BUILD_TAG.txt','w').write(BUILD_TAG); print('SMART_CHECK', BUILD_TAG, 'MAX_ACTIONS', MyAgent.MAX_ACTIONS)"

            _llm_flag = os.environ.get('LINGJING_LLM', '{llm_env_default}')
            _selflearn_flag = os.environ.get('SELFLEARN_LLM', '{selflearn}')
            if _selflearn_flag != '{selflearn}':
                raise SystemExit('SELFLEARN_LLM_DRIFT notebook=%s runtime=%s' % ('{selflearn}', _selflearn_flag))
            if _llm_flag.lower() in ('1', 'on', 'true', 'yes'):
                import json as _json
                from pathlib import Path as _Path
                _fp_path = _Path('/kaggle/working/PHASE_B_FINGERPRINT.json')
                if not _fp_path.is_file():
                    raise SystemExit('PHASE_B_FINGERPRINT_MISSING')
                _fp = _json.loads(_fp_path.read_text(encoding='utf-8'))
                if (
                    _fp.get('vllm_ready') is not True
                    or _fp.get('probe') != 'ok'
                    or _fp.get('enable_internet') is not False
                    or _fp.get('required_tag') != 'llm-whole-qwen'
                ):
                    raise SystemExit('PHASE_B_FINGERPRINT_MISMATCH %s' % _fp)
            _tag = open('/kaggle/working/BUILD_TAG.txt', encoding='utf-8').read()
            if (
                'llm-whole-qwen' not in _tag
                or 'llm-structured-view' not in _tag
                or not _tag.startswith('smart-router')
            ):
                raise SystemExit('BUILD_TAG_MISMATCH %s' % _tag)
            print('PHASE_B_OK', _tag, 'LLM', _llm_flag, 'SELFLEARN_LLM', _selflearn_flag, flush=True)

            # Point the framework at the gateway sidecar.
            with open('/kaggle/working/ARC-AGI-3-Agents/.env', 'w') as f:
                f.write(f\"\"\"SCHEME=http
        HOST=gateway
        PORT=8001
        ARC_API_KEY=test-key-123
        ARC_BASE_URL=http://gateway:8001/
        OPERATION_MODE=online
        ENVIRONMENTS_DIR=
        RECORDINGS_DIR=/kaggle/working/server_recording
        AETHER_LLM=0
        LINGJING_SCRIPT_DEBUG=0
        SELFLEARN_LLM={selflearn}
        LINGJING_LLM={{_llm_flag}}
        LINGJING_LLM_URL=http://127.0.0.1:1234/v1
        LINGJING_LLM_MODEL=vrfai/Qwen3.6-27B-FP8
        LINGJING_LLM_VISION=off
        LINGJING_LLM_PRECEAX=on
        LINGJING_LLM_WHOLE=on
        LINGJING_LLM_VIEW=planner
        LINGJING_LLM_BATCH=4
        LINGJING_LLM_EVERY=1
        LINGJING_LLM_NATIVE=1
        SELFLEARN_LLM_MAXCHARS=12000
        SELFLEARN_LLM_THINK=0
        SELFLEARN_LLM_MAXTOK=512
        LINGJING_LEXICON_DIR=/kaggle/working/ARC-AGI-3-Agents/arc_adaptor
        LINGJING_LLM_STALL=80
        LINGJING_LLM_STRIDE=1
        LINGJING_LLM_MAX_CALLS=800
        \"\"\")

            # Run it. The gateway records every action and emits submission.parquet.
            !cd /kaggle/working/ARC-AGI-3-Agents && \\
                MPLBACKEND=agg \\
                PYTHONPATH=/kaggle/working/ARC-AGI-3-Agents/arc_adaptor:/kaggle/working/ARC-AGI-3-Agents:${{PYTHONPATH}} \\
                python main.py --agent myagent
            !python -c "print('MAIN_EXIT', 0)"
        """
    )
    run_cell = code_cell(run_cell_source)

    # ---------------------------------------------------------------------
    # Offline self-benchmark (A/B builds only).
    #
    # The competition rerun path needs the Kaggle gateway, which only exists
    # during a real submission -- so a normal kernel run measures nothing.
    # This cell instead plays the competition's *public* environment files
    # offline, which is the same 25-game set scripts/benchmark_all_games.py
    # uses locally, and writes the scorecard next to the logs.
    # ---------------------------------------------------------------------
    bench_cell = code_cell(
        dedent(
            """
            # ── Offline self-benchmark over the public games ────────────────
            import importlib.util
            import json
            import os
            import shutil
            import sys
            import time
            from pathlib import Path

            _WORK = Path("/kaggle/working")
            _FW = _WORK / "ARC-AGI-3-Agents"
            if not _FW.is_dir():
                shutil.copytree(
                    "/kaggle/input/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents", _FW
                )
            # The slim registry below does `from .templates.my_agent import MyAgent`,
            # so the agent has to actually EXIST inside the framework tree. The
            # notebook only wrote it to /tmp/my_agent.py, so the first version of
            # this cell died with
            #   ModuleNotFoundError: No module named 'agents.templates.my_agent'
            # Drop it (and its two siblings) into agents/templates/ first --
            # exactly what the submission cell does with `cp`.
            _TPL = _FW / "agents" / "templates"
            _TPL.mkdir(parents=True, exist_ok=True)
            for _src, _dst in (("/tmp/my_agent.py", "my_agent.py"),
                               ("/tmp/llm_policy.py", "llm_policy.py"),
                               ("/tmp/llm_episode.py", "llm_episode.py")):
                _s = Path(_src)
                if _s.is_file():
                    shutil.copy2(_s, _TPL / _dst)
            # The stock agents/__init__.py imports every template, including the
            # langgraph ones -- and the Kaggle image's langgraph has no
            # `langgraph.store.sqlite`, so `import agents` explodes. Rewrite it to
            # the slim registry the submission cell uses (same fix, same reason).
            with open(_FW / "agents" / "__init__.py", "w") as _f:
                _f.write('''from typing import Type
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
            ''')
            # The agent imports `agents.agent`, `lingjing_solo`, `arc_adaptor`
            # and `llm_episode`; cells 2-6 already wrote the last three to /tmp.
            for _p in (str(_FW), "/tmp", "/tmp/arc_adaptor"):
                if _p not in sys.path:
                    sys.path.insert(0, _p)
            os.environ.setdefault("LINGJING_SRC", "/tmp")
            os.environ.setdefault("LINGJING_LEXICON_DIR", "/tmp/arc_adaptor")
            os.environ.setdefault("MPLBACKEND", "Agg")

            import arc_agi

            # Where the competition's public environment files live. The layout
            # under /kaggle/input differs between the competition bundle and the
            # `competitions/` mirror, so probe instead of guessing -- guessing
            # gave us "BENCH_GAMES 0 []" (an empty benchmark that looks like a
            # zero-score agent) on the first push.
            _ENV_CANDS = [
                Path("/kaggle/input/arc-prize-2026-arc-agi-3/environment_files"),
                Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files"),
                Path("/kaggle/input/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents/environment_files"),
                Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents/environment_files"),
                Path("/kaggle/input/arc-prize-2026-arc-agi-3"),
            ]
            _ENV_DIR = ""
            for _c in _ENV_CANDS:
                try:
                    if _c.is_dir() and any(p.is_dir() for p in _c.iterdir()):
                        _ENV_DIR = str(_c)
                        break
                except OSError:
                    continue
            print("BENCH_ENV_DIR", _ENV_DIR or "<none>",
                  "existing=", [str(c) for c in _ENV_CANDS if c.is_dir()],
                  flush=True)
            _arc = arc_agi.Arcade(
                operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=_ENV_DIR
            )
            _ids = [e.game_id.split("-")[0] for e in _arc.get_environments()]
            print("BENCH_GAMES", len(_ids), _ids, flush=True)

            _spec = importlib.util.spec_from_file_location("bench_agent", "/tmp/my_agent.py")
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            _MyAgent = _mod.MyAgent
            _MyAgent.MAX_ACTIONS = min(_MyAgent.MAX_ACTIONS, __BENCH_MAX_STEPS__)
            print("BENCH_AGENT", _mod.BUILD_TAG, "max_actions", _MyAgent.MAX_ACTIONS, flush=True)

            def _pick_run(scorecard, game_id):
                if not scorecard or not scorecard.environments:
                    return None
                for _env in scorecard.environments:
                    if (_env.id or "").split("-")[0] != game_id:
                        continue
                    if not _env.runs:
                        return None
                    return max(_env.runs, key=lambda r: r.score or 0).model_dump()
                return None

            _rows = []
            for _i, _gid in enumerate(_ids, 1):
                _t0 = time.time()
                _row = {
                    "game_id": _gid, "levels_completed": 0, "actions": 0,
                    "score": 0.0, "state": "ERROR", "error": None,
                }
                try:
                    _env = _arc.make(_gid, seed=0)
                    if _env is None:
                        _row["state"] = "SKIP"
                        _row["error"] = "env_create_failed"
                    else:
                        _agent = _MyAgent(
                            card_id="bench", game_id=_gid, agent_name=f"bench.{_gid}",
                            ROOT_URL="http://localhost", record=False,
                            arc_env=_env, tags=["benchmark"],
                        )
                        _agent.main()
                        _final = _agent.frames[-1]
                        _row["levels_completed"] = int(_final.levels_completed or 0)
                        _row["actions"] = int(_agent.action_counter)
                        _row["state"] = getattr(_final.state, "name", str(_final.state))
                        _run = _pick_run(_arc.get_scorecard(), _gid)
                        if _run:
                            _row["score"] = float(_run.get("score") or 0)
                            _row["level_scores"] = _run.get("level_scores") or []
                except Exception as _exc:
                    _row["error"] = f"{type(_exc).__name__}: {_exc}"
                _row["elapsed_sec"] = round(time.time() - _t0, 2)
                _rows.append(_row)
                print(
                    f"[BENCH {_i}/{len(_ids)}] {_gid:8} L={_row['levels_completed']} "
                    f"score={_row['score']:.2f} act={_row['actions']} {_row['state']}",
                    flush=True,
                )

            _sc = _arc.get_scorecard()
            _summary = {
                "selflearn_llm": os.environ.get("SELFLEARN_LLM", "0"),
                "build_tag": _mod.BUILD_TAG,
                "aggregate_score_official": float(_sc.score) if _sc is not None else None,
                "total_levels_completed": sum(r["levels_completed"] for r in _rows),
                "total_actions": sum(r["actions"] for r in _rows),
                "zero_games": [r["game_id"] for r in _rows if r["levels_completed"] == 0],
                "games": _rows,
            }
            _out = Path("/kaggle/working/BENCH_SELFLEARN___SELFLEARN__.json")
            _out.write_text(json.dumps(_summary, ensure_ascii=False, indent=2), encoding="utf-8")
            print("BENCH_SUMMARY", json.dumps({
                "selflearn_llm": _summary["selflearn_llm"],
                "aggregate_score_official": _summary["aggregate_score_official"],
                "total_levels_completed": _summary["total_levels_completed"],
                "total_actions": _summary["total_actions"],
                "zero_games": _summary["zero_games"],
            }, ensure_ascii=False), flush=True)
            print("BENCH_WROTE", _out, flush=True)
            """
        ).replace("__SELFLEARN__", selflearn).replace(
            "__BENCH_MAX_STEPS__", str(BENCH_MAX_STEPS)
        )
    )

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

    cells = [
        markdown_cell(
            "# ARC Prize 2026 — SmartRouter + optional local LLM\n\n"
            "Submission agent = **SmartRouter** (`BUILD_TAG` starts with `smart-router`). "
            "Plan floors stay on verified plans. Every other game, including hidden "
            "ids, is played for the whole episode by in-kernel Qwen (`llm-whole-qwen` "
            "+ `llm-structured-view` batch planner, vLLM context 32768). "
            "Phase B refuses to start the games unless vLLM probes OK and "
            "`enable_internet` is false. "
            "GPU default: RTX 6000 + vllm031-cp313 + qwen3.6-27b-fp8. "
            "CPU fallback: `LINGJING_KAGGLE_ACCEL=cpu` → LLM off."
        ),
        install_cell,
        write_lingjing_cell,
        write_lexicon_cell,
        write_agent_cell,
        write_llm_policy_cell,
        write_episode_cell,
        vllm_cell,
        run_cell,
    ]
    if BENCH:
        cells.append(bench_cell)
    cells.append(dummy_submission_cell)
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
        "cells": cells,
    }
    return notebook


def main() -> None:
    if ACCELERATOR not in _ACCELERATORS:
        raise SystemExit(
            f"Unknown ACCELERATOR={ACCELERATOR!r}. Pick one of: "
            f"{sorted(_ACCELERATORS)}"
        )
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(
        json.dumps(build(), ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"[build_notebook] Wrote {NOTEBOOK_PATH.name}  "
          f"(accelerator: {ACCELERATOR}, SELFLEARN_LLM: {SELFLEARN_LLM})")

    # Keep notebooks/kernel-metadata.json in sync so the user never has to
    # edit it just to flip CPU ↔ GPU / attach LLM datasets.
    if METADATA_PATH.exists():
        meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        wanted = _ACCELERATORS[ACCELERATOR]["gpu"]
        shape = _ACCELERATORS[ACCELERATOR].get("machine_shape", "")
        dirty = False
        if meta.get("enable_internet") is not False:
            meta["enable_internet"] = False
            dirty = True
        if meta.get("enable_gpu") != wanted:
            meta["enable_gpu"] = wanted
            dirty = True
        if meta.get("machine_shape", "") != shape:
            meta["machine_shape"] = shape
            dirty = True
        # Drop obsolete free-form accelerator key if present.
        if "accelerator" in meta:
            meta.pop("accelerator", None)
            dirty = True
        if wanted and SELFLEARN_LLM == "1":
            if meta.get("dataset_sources") != _LLM_DATASET_SOURCES:
                meta["dataset_sources"] = list(_LLM_DATASET_SOURCES)
                dirty = True
        else:
            # CPU floors-only, or SELFLEARN_LLM=0 (pure algorithm, no vLLM):
            # drop heavy model datasets to keep commits light.
            if meta.get("dataset_sources"):
                meta["dataset_sources"] = []
                dirty = True
        if dirty:
            METADATA_PATH.write_text(
                json.dumps(meta, indent=2) + "\n", encoding="utf-8"
            )
            print(
                f"[build_notebook] Synced enable_gpu={wanted} "
                f"machine_shape={shape!r} "
                f"dataset_sources={meta.get('dataset_sources')} in "
                f"{METADATA_PATH.relative_to(ROOT)}"
            )


if __name__ == "__main__":
    main()
