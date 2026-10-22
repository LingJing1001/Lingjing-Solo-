"""Point `LINGJING_LLM_URL` at a server that is actually running -- and prove it.

The champion never typed a key: it starts SGLang on 127.0.0.1 and sets
`OPENAI_BASE_URL` to it.  Reproducing that by hand has three sharp edges:

  1. the port and the `--served-model-name` must match what goes into `.env`;
  2. the served id is often *not* what you think (SGLang echoes the HF path);
  3. `.env` is invisible to the agent unless it is written before launch.

This script closes all three: probe -> start (or attach) -> wait for `/v1/models`
-> read the real model id -> write `.env` -> send one real prompt.

    python scripts/llm_up.py probe                      # what is already running here?
    python scripts/llm_up.py up --backend auto          # start the best available server
    python scripts/llm_up.py attach --url http://127.0.0.1:8000/v1
    python scripts/llm_up.py check                      # re-ping whatever .env points at

Nothing here needs an API key for a local server: vLLM / SGLang / llama.cpp ignore
the bearer token.  `--key` exists only for hosted endpoints.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
LOG_DIR = ROOT / "logs"

# Ports worth trying, with the backend that usually owns them.
COMMON = [
    (8001, "sglang (champion's port)"),
    (8000, "vllm"),
    (30000, "sglang default"),
    (11434, "ollama"),
    (8080, "llama.cpp / llama-server"),
    (1234, "lm-studio"),
    (5000, "local proxy"),
]


# --------------------------------------------------------------------------- io

def _opener(url: str):
    """Bypass HTTP_PROXY only for loopback and private LAN addresses.

    A system proxy turns `http://127.0.0.1:8001` into a 502 Bad Gateway, which looks
    exactly like "the server never started" -- that wasted an hour once.  Private
    ranges (a GPU box on the LAN) have the same problem.  Public hosts must keep
    using the proxy, otherwise `attach` to a hosted endpoint breaks on proxy-only
    networks.
    """
    host = urllib.parse.urlparse(url).hostname or ""
    local = host in ("localhost", "0.0.0.0", "::1")
    if not local:
        try:
            ip = ipaddress.ip_address(host)
            local = ip.is_loopback or ip.is_private
        except ValueError:
            local = False
    if local:
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


def _get(url: str, timeout: float = 2.0) -> tuple[int, str]:
    req = urllib.request.Request(url, method="GET")
    try:
        with _opener(url).open(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        return 0, f"{type(e).__name__}: {e}"


def _post(url: str, body: dict, key: str, timeout: float) -> tuple[int, str]:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"},
        method="POST")
    try:
        with _opener(url).open(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        return 0, f"{type(e).__name__}: {e}"


def models_at(base_url: str) -> list[str]:
    """Model ids the server advertises, or [] when it is not speaking OpenAI."""
    code, text = _get(base_url.rstrip("/") + "/models", timeout=3)
    if code != 200:
        return []
    try:
        data = json.loads(text)
    except Exception:  # noqa: BLE001
        return []
    out = []
    for item in data.get("data") or []:
        mid = item.get("id") or item.get("model")
        if mid:
            out.append(str(mid))
    return out


def port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.4)
        return s.connect_ex((host, int(port))) == 0


# ----------------------------------------------------------------- hardware

def hardware() -> dict:
    info = {"gpu": None, "vram_gb": 0.0, "ram_gb": 0.0, "cpus": os.cpu_count() or 1,
            "platform": platform.system()}
    exe = shutil.which("nvidia-smi")
    if exe:
        try:
            out = subprocess.run(
                [exe, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=8).stdout.strip().splitlines()
            if out:
                name, mem = out[0].rsplit(",", 1)
                info["gpu"] = name.strip()
                # "2048 MiB" / "24564 MiB"
                num = float("".join(ch for ch in mem if ch.isdigit() or ch == ".") or 0)
                info["vram_gb"] = round(num / 1024.0, 1)
        except Exception:  # noqa: BLE001
            pass
    try:
        if platform.system() == "Windows":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MEMORYSTATUSEX()
            m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            info["ram_gb"] = round(m.ullTotalPhys / 2 ** 30, 1)
        else:
            with open("/proc/meminfo") as f:
                kb = int(f.readline().split()[1])
            info["ram_gb"] = round(kb / 2 ** 20, 1)
    except Exception:  # noqa: BLE001
        pass
    return info


def recommend(hw: dict) -> list[str]:
    """Which checkpoint this box can actually serve, biggest first."""
    v, r = hw["vram_gb"], hw["ram_gb"]
    tips: list[str] = []
    if v >= 40:
        tips.append("Qwen/Qwen3-14B  (bf16, ctx 32768+)  <- closest to the champion")
        tips.append("Qwen/Qwen3-8B   (bf16, ctx 131072)")
    elif v >= 22:
        tips.append("Qwen/Qwen3-8B   (bf16 or AWQ, ctx 32768)")
        tips.append("Qwen/Qwen3-4B   (bf16, ctx 32768)")
    elif v >= 10:
        tips.append("Qwen/Qwen3-4B   (AWQ/GPTQ, ctx 32768)")
        tips.append("Qwen/Qwen3-1.7B (bf16)")
    elif v >= 6:
        tips.append("Qwen/Qwen3-1.7B (AWQ) or Qwen3-4B with --cpu-offload")
    elif v >= 3:
        tips.append("Qwen/Qwen3-0.6B (bf16) -- usable for plumbing, weak at planning")
    else:
        tips.append("no model fits this GPU; see the notes below")
    return tips


# ------------------------------------------------------------------- env file

ENV_KEYS = ["SELFLEARN_LLM", "SELFLEARN_LLM_BACKEND", "SELFLEARN_LLM_THINK",
            "LINGJING_LLM_URL", "LINGJING_LLM_MODEL", "LINGJING_LLM_KEY",
            "LINGJING_LLM_TIMEOUT"]


def write_env(values: dict) -> Path:
    """Rewrite only the keys we own, keep every comment the user wrote."""
    lines = ENV_PATH.read_text(encoding="utf-8").splitlines() if ENV_PATH.is_file() else []
    pending = dict(values)
    out: list[str] = []
    for line in lines:
        raw = line.strip()
        if raw and not raw.startswith("#") and "=" in raw:
            k = raw.split("=", 1)[0].strip()
            if k in pending:
                out.append(f"{k}={pending.pop(k)}")
                continue
        out.append(line)
    if pending:
        if out and out[-1].strip():
            out.append("")
        out.append("# --- written by scripts/llm_up.py ---")
        for k in ENV_KEYS:
            if k in pending:
                out.append(f"{k}={pending.pop(k)}")
    # keep a stable, readable tail
    while out and not out[-1].strip():
        out.pop()
    out.append("")
    ENV_PATH.write_text("\n".join(out), encoding="utf-8")
    return ENV_PATH


def read_env() -> dict:
    vals: dict[str, str] = {}
    if not ENV_PATH.is_file():
        return vals
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        k, v = raw.split("=", 1)
        vals[k.strip()] = v.strip().strip('"').strip("'")
    return vals


# ----------------------------------------------------------------- commands

def cmd_probe(args) -> int:
    hw = hardware()
    print("== hardware ==")
    print(f"  gpu   : {hw['gpu'] or 'none'}"
          + (f"  ({hw['vram_gb']:.1f} GB VRAM)" if hw["gpu"] else ""))
    print(f"  ram   : {hw['ram_gb']:.1f} GB")
    print(f"  cpus  : {hw['cpus']}")
    print(f"  os    : {hw['platform']}")

    print("\n== libraries ==")
    for mod in ("vllm", "sglang", "transformers", "torch"):
        try:
            m = __import__(mod)
            print(f"  {mod:13s} {getattr(m, '__version__', '?')}")
        except Exception:  # noqa: BLE001
            print(f"  {mod:13s} MISSING")
    for exe in ("ollama", "llama-server", "docker"):
        print(f"  {exe:13s} {shutil.which(exe) or 'MISSING'}")

    print("\n== already listening ==")
    found = []
    for port, who in COMMON:
        if not port_open("127.0.0.1", port):
            continue
        base = f"http://127.0.0.1:{port}/v1"
        ids = models_at(base)
        if ids:
            print(f"  127.0.0.1:{port:<6d} {who}")
            for i in ids:
                print(f"      model id: {i}")
            found.append((base, ids[0]))
        else:
            code, _ = _get(f"http://127.0.0.1:{port}/health", timeout=2)
            print(f"  127.0.0.1:{port:<6d} open but no /v1/models "
                  f"({who}, health={code or 'n/a'})")
    if not found:
        print("  nothing OpenAI-compatible is running")

    print("\n== this box can serve ==")
    for tip in recommend(hw):
        print(f"  - {tip}")
    if not hw["gpu"] or hw["vram_gb"] < 6:
        print("\n  Caveat: under ~6 GB VRAM (or CPU-only) a local model will be too slow "
              "and too weak\n  to make ARC decisions. Two honest options: run the server "
              "on a machine that has a GPU,\n  or attach to a hosted endpoint. "
              "`llm_up.py attach --url ...` works for both.")

    if found:
        base, model = found[0]
        print(f"\nNext: python scripts/llm_up.py attach --url {base} --model {model}")
    return 0


def _wait_ready(base_url: str, proc, timeout: float, log_path: Path) -> list[str]:
    t0 = time.time()
    last = 0.0
    while time.time() - t0 < timeout:
        if proc is not None and proc.poll() not in (None, 0):
            print(f"\nserver exited early (code {proc.returncode}); tail of {log_path}:")
            try:
                tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
                print("\n".join(tail))
            except Exception:  # noqa: BLE001
                pass
            return []
        ids = models_at(base_url)
        if ids:
            return ids
        if time.time() - last > 10:
            last = time.time()
            print(f"  ... waiting for {base_url}/models  ({time.time() - t0:.0f}s)")
        time.sleep(2.0)
    return []


def _start(backend: str, model: str | None, port: int, ctx: int, gpu_frac: float):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"llm_{backend}_{port}.log"
    log = open(log_path, "w", encoding="utf-8")
    py = sys.executable

    if backend == "vllm":
        if not model:
            raise SystemExit("--model is required for vllm")
        cmd = [py, "-m", "vllm.entrypoints.openai.api_server",
               "--model", model, "--served-model-name", model,
               "--host", "127.0.0.1", "--port", str(port),
               "--max-model-len", str(ctx),
               "--gpu-memory-utilization", str(gpu_frac)]
    elif backend == "sglang":
        if not model:
            raise SystemExit("--model is required for sglang")
        cmd = [py, "-m", "sglang.launch_server",
               "--model-path", model, "--served-model-name", model,
               "--host", "127.0.0.1", "--port", str(port),
               "--context-length", str(ctx)]
    elif backend == "ollama":
        print("  `ollama serve` then `ollama pull <model>`; starting server in background")
        cmd = ["ollama", "serve"]
        if model:
            subprocess.run(["ollama", "pull", model], check=False)
    elif backend == "llamacpp":
        if not model:
            raise SystemExit("--model must be a .gguf path for llamacpp")
        cmd = ["llama-server", "-m", model, "--host", "127.0.0.1",
               "--port", str(port), "-c", str(ctx)]
    else:
        raise SystemExit(f"unknown backend {backend!r}")

    print(f"  $ {' '.join(cmd)}")
    print(f"  log -> {log_path}")
    flags = 0
    if os.name == "nt":
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | \
                getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT),
                            creationflags=flags)
    return proc, log_path


def _pick_backend(explicit: str) -> str:
    if explicit != "auto":
        return explicit
    for name, probe in (("sglang", "sglang"), ("vllm", "vllm")):
        try:
            __import__(probe)
            return name
        except Exception:  # noqa: BLE001
            continue
    if shutil.which("ollama"):
        return "ollama"
    if shutil.which("llama-server"):
        return "llamacpp"
    raise SystemExit(
        "no local serving stack found (vllm / sglang / ollama / llama-server).\n"
        "Either install one, or attach to an endpoint that already runs:\n"
        "  python scripts/llm_up.py attach --url http://HOST:PORT/v1 --model MODEL")


def cmd_up(args) -> int:
    hw = hardware()
    backend = _pick_backend(args.backend)
    port = args.port or {"vllm": 8000, "sglang": 8001, "ollama": 11434,
                         "llamacpp": 8080}.get(backend, 8000)
    ctx = args.ctx or (32768 if hw["vram_gb"] >= 10 else 8192)
    gpu_frac = 0.92 if hw["vram_gb"] >= 24 else 0.95
    base = f"http://127.0.0.1:{port}/v1"

    if models_at(base):
        print(f"already serving at {base}: {models_at(base)}")
        return _finish(base, models_at(base)[0], args)

    print(f"== starting {backend} on 127.0.0.1:{port} (ctx={ctx}) ==")
    proc, log_path = _start(backend, args.model, port, ctx, gpu_frac)
    print(f"  waiting up to {args.wait:.0f}s for it to answer /v1/models ...")
    ids = _wait_ready(base, proc, args.wait, log_path)
    if not ids:
        print("\nFAIL: server never became ready. Read the log above for the real cause "
              "(usually: model not downloaded, not enough VRAM, or a wrong model id).")
        return 1
    print(f"  ready: {ids}")
    return _finish(base, ids[0], args)


def cmd_attach(args) -> int:
    base = args.url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    ids = models_at(base)
    if not ids:
        print(f"FAIL: {base}/models did not answer with a model list.")
        print("  - server not started?  check the port and that it is bound to 0.0.0.0/127.0.0.1")
        print("  - proxied host?        this script already bypasses HTTP_PROXY for localhost")
        return 1
    model = args.model or ids[0]
    if args.model and args.model not in ids:
        print(f"WARN: {args.model!r} is not in the served list {ids} -- requests will 404.")
        print(f"      using {ids[0]!r} instead.")
        model = ids[0]
    print(f"attached: {base}  model={model}")
    return _finish(base, model, args)


def _finish(base: str, model: str, args) -> int:
    think = 1 if args.think else 0
    values = {
        "SELFLEARN_LLM": "1",
        "SELFLEARN_LLM_BACKEND": "openai",
        "SELFLEARN_LLM_THINK": str(think),
        "LINGJING_LLM_URL": base,
        "LINGJING_LLM_MODEL": model,
        "LINGJING_LLM_KEY": args.key or "EMPTY",
        "LINGJING_LLM_TIMEOUT": str(int(args.timeout)),
    }
    path = write_env(values)
    print(f"\n== wrote {path} ==")
    for k in ENV_KEYS:
        print(f"  {k}={values[k]}")

    ok = _ping(base, model, values["LINGJING_LLM_KEY"], think, args.timeout)
    if not ok:
        return 1
    print("\nNext:")
    print("  python scripts/play_local.py --agent agent/self_learner.py "
          "--game ka59 --max-steps 45")
    return 0


def _ping(base: str, model: str, key: str, think: int, timeout: float) -> bool:
    body = {
        "model": model,
        "messages": [{"role": "system", "content": 'Reply with exactly: {"ok":true}'},
                     {"role": "user", "content": "ping"}],
        "temperature": 0.0,
        "max_tokens": 512 if think else 32,
    }
    if think:
        body["chat_template_kwargs"] = {"enable_thinking": True}
    print(f"\n== ping {base}/chat/completions (think={think}) ==")
    t0 = time.time()
    code, text = _post(base.rstrip("/") + "/chat/completions", body, key, timeout + 60)
    dt = time.time() - t0
    if code != 200:
        print(f"FAIL HTTP {code} after {dt:.1f}s")
        print(text[:600])
        if code == 404:
            print("Hint: model id mismatch -- it must equal what /v1/models returned.")
        elif code == 400:
            print("Hint: some servers reject chat_template_kwargs; retry with --no-think.")
        return False
    try:
        msg = json.loads(text)["choices"][0]["message"]
        content = (msg.get("content") or "").strip()
        reasoning = (msg.get("reasoning_content") or "").strip()
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: unparsable reply ({e}): {text[:300]}")
        return False
    print(f"OK  200  {dt:.2f}s")
    if content:
        print(f"  content  : {content[:160]!r}")
    if reasoning:
        print(f"  reasoning: {reasoning[:120]!r}  ({len(reasoning)} chars)")
    if think and not content and reasoning:
        print("  WARN: thinking used the whole budget -- raise SELFLEARN_LLM_MAXTOK.")
    return True


def cmd_check(args) -> int:
    env = read_env()
    base = args.url or env.get("LINGJING_LLM_URL") or ""
    model = args.model or env.get("LINGJING_LLM_MODEL") or ""
    if not base:
        print("FAIL: no LINGJING_LLM_URL in .env -- run `llm_up.py probe` first.")
        return 2
    think = args.think if args.think is not None else (
        str(env.get("SELFLEARN_LLM_THINK", "0")).lower() in ("1", "true", "yes", "on"))
    print(f"base  = {base}")
    print(f"model = {model or '(unset)'}")
    ids = models_at(base)
    print(f"served= {ids or 'n/a'}")
    if not model and ids:
        model = ids[0]
    if not model:
        print("FAIL: no model id. Pass --model.")
        return 2
    ok = _ping(base, model, env.get("LINGJING_LLM_KEY", "EMPTY"), 1 if think else 0,
               args.timeout)
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("probe", help="what runs here, and what this box could serve")
    sp.set_defaults(fn=cmd_probe)

    su = sub.add_parser("up", help="start a local server, then write .env and ping it")
    su.add_argument("--backend", default="auto",
                    choices=["auto", "vllm", "sglang", "ollama", "llamacpp"])
    su.add_argument("--model", default=None,
                    help="HF id (Qwen/Qwen3-8B-Instruct) or a .gguf path")
    su.add_argument("--port", type=int, default=None)
    su.add_argument("--ctx", type=int, default=None)
    su.add_argument("--wait", type=float, default=600.0)
    su.add_argument("--think", action="store_true", default=True)
    su.add_argument("--no-think", dest="think", action="store_false")
    su.add_argument("--key", default=None)
    su.add_argument("--timeout", type=float, default=180.0)
    su.set_defaults(fn=cmd_up)

    sa = sub.add_parser("attach", help="use a server that already runs")
    sa.add_argument("--url", required=True)
    sa.add_argument("--model", default=None)
    sa.add_argument("--think", action="store_true", default=True)
    sa.add_argument("--no-think", dest="think", action="store_false")
    sa.add_argument("--key", default=None)
    sa.add_argument("--timeout", type=float, default=180.0)
    sa.set_defaults(fn=cmd_attach)

    sc = sub.add_parser("check", help="re-ping what .env points at")
    sc.add_argument("--url", default=None)
    sc.add_argument("--model", default=None)
    sc.add_argument("--think", default=None, type=lambda s: s.lower() in ("1", "true", "yes"))
    sc.add_argument("--timeout", type=float, default=180.0)
    sc.set_defaults(fn=cmd_check)

    args = p.parse_args()
    if not hasattr(args, "think"):
        args.think = True
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
