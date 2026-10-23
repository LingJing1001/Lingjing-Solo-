"""生成 Duck v28 kernel（以 v27 notebook 为输入的后处理器）。

为什么要有 v28
--------------
v27 把「四处自伤旋钮」回归了官方，但**没有动覆盖率**。算一下就知道问题在哪：

    9 h 窗口 32400 s − 启动 ~1834 s = 30566 s 可跑
    v27 : concurrency 8  × 每局 5400 s  ⇒ 5.6 波 ⇒ 约 45 / 110 局（41%）
    官方: concurrency 28 × 每局 7920 s  ⇒ 4 波 ⇒ 110 / 110 局（100%）

**覆盖率差 2.4 倍 —— 这才是我们（0.744/局）远低于官方（1.6002/局）的真正原因**，
不是旋钮、也不是模型能力。官方自己的注释写得很清楚：
「110 局 ÷ 28 并发 = 4 波；4 波 × 7920 s = 31680 s ≈ 32400 s（9 h 窗口）」——
官方把参数**按铺满 9 小时反推**出来的。

v28 的四项改动
--------------
1. **覆盖率**：solver 配置向官方靠（concurrency 16 / max_runtime 7920 / max-num-seqs 16）。
   28 并发配 32k ctx 有请求饥饿风险（官方是 64k ctx + 默认 max-num-seqs），取 16 做折中：
   30566 s ÷ (0.6×7920 + 0.4×2700) ≈ 5.2 波 ⇒ 约 84 / 110 局（vs v27 的 45）。
2. **提前放弃开关真正生效**：开关的代码在 `inference/framework/solver.py`，它随 overlay
   数据集分发，而数据集上传走 `www.googleapis.com`（本机不通）⇒ overlay 卡在 v5。
   **v28 直接在 notebook 里覆盖那个文件**（远端 v5 的 solver.py 是本地 v27 版的**严格子集**：
   0 删除行、4 hunk、69 行纯新增，已用 difflib 核对过 ⇒ 覆盖绝对安全）。
3. **短 bench 模式**（只在**非竞赛 rerun** 时生效）：把每局上限压到 900 s，
   推一次 kernel 从 8.3 h 降到 ~1 h ⇒ **A/B 队列终于跑得动了**（之前一天只能测一个变体）。
   竞赛 rerun 里仍是官方 7920 s / 110 局，不受影响。
4. **路径探测**：wheels 与 environment_files 都改成「多候选 + glob 兜底 + 响亮失败」
   （§8.5 同类问题在 SmartRouter 线上刚犯过一次，不能在 Duck 线上再犯一次）。

用法:
    python patch_v28.py           # 生成 push_v28/lingjing-duck-harness-v28.ipynb
    python patch_v28.py --check   # 只校验生成的 notebook 含全部改动
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DUCK = HERE.parents[1]                     # integrations/duck-kaggle
REPO = HERE.parents[2]                     # 灵境战队AGI课题探索
V27_NB = HERE / "push_v27" / "lingjing-duck-harness-v27.ipynb"
OUT_DIR = HERE / "push_v28"
OUT_NB = OUT_DIR / "lingjing-duck-harness-v28.ipynb"
OUT_META = OUT_DIR / "kernel-metadata.json"
SOLVER_SRC = REPO / "integrations" / "duck-harness" / "ARC3-Inference" / "inference" / "framework" / "solver.py"

SLUG = "liaojiaxuan/lingjing-duck-harness-v28"
TITLE = "lingjing-duck-harness-v28"

# --- v28 的求解参数（向官方靠，但给 VRAM 留余量） ---
SOLVER_CONCURRENCY = 16
MAX_RUNTIME_S_PER_GAME = 7920        # 官方值
ANALYZER_TIMEOUT_S = 900             # 官方值
NO_PROGRESS_GIVEUP_S = 2700
VLLM_MAX_NUM_SEQS = 16
LOCAL_ANALYZER_CONTEXT_WINDOW = 32768
MULTIMODAL_UPSCALE = "4"

# --- 短 bench（仅非竞赛 rerun 生效） ---
SHORT_BENCH_RUNTIME_S = 900         # 每局上限
SHORT_BENCH_GAMES = 25

WHEELS_CANDIDATES = [
    "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels",
    "/kaggle/input/arc-prize-2026-arc-agi-3/arc_agi_3_wheels",
]


def _as_source(src: str) -> list[str]:
    lines = src.splitlines(keepends=True)
    return lines


def _replace_once(src: str, old: str, new: str, what: str) -> str:
    if src.count(old) != 1:
        raise SystemExit(f"[v28] 锚点命中 {src.count(old)} 次（应为 1）：{what}")
    return src.replace(old, new, 1)


# --------------------------------------------------------------------------
# 1. cell 4：wheels 路径 -> 多候选探测
# --------------------------------------------------------------------------
def patch_wheels(src: str) -> str:
    old = '''        "--find-links",
        "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels",
        "arc-agi",'''
    new = '''        "--find-links",
        str(_lj_wheels_dir()),
        "arc-agi",'''
    src = _replace_once(src, old, new, "cell4 wheels 路径")

    helper = '''
# lingjing v28: the competition dataset can be mounted under /kaggle/input/<slug>
# or /kaggle/input/competitions/<slug> depending on the kernel.  Never hardcode
# one prefix -- probe the known candidates, fall back to a glob, and fail loudly.
def _lj_wheels_dir():
    import pathlib as _pl

    candidates = [
        _pl.Path(p) for p in (
            "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels",
            "/kaggle/input/arc-prize-2026-arc-agi-3/arc_agi_3_wheels",
        )
    ]
    for c in candidates:
        if c.is_dir():
            print("taaf.kaggle: wheels", c, flush=True)
            return c
    for c in sorted(_pl.Path("/kaggle/input").rglob("arc_agi_3_wheels")):
        if c.is_dir():
            print("taaf.kaggle: wheels (glob)", c, flush=True)
            return c
    raise SystemExit("WHEELS_NOT_FOUND candidates=%s" % [str(c) for c in candidates])

'''
    # helper 放在 cell 顶部（import 之后即可用）
    src = src.replace("\nsubprocess.check_call(", helper + "\nsubprocess.check_call(", 1)
    return src


# --------------------------------------------------------------------------
# 2. cell 8：--max-num-seqs 抬到 16 + 覆盖 overlay 的 solver.py
# --------------------------------------------------------------------------
def patch_max_num_seqs(src: str) -> str:
    """把 vLLM 的 admit limit (--max-num-seqs) 从 8 抬到 16。

    cell 8 里一共有 **三处** 把 admit limit 写死成 8（v27 的本意是「钉到 8 以匹配
    solver.concurrency = 8」）：

      A. `_vllm_extra` 注入分支 —— 上游 setup 命令里 **既没有 --max-num-seqs 也没有
         --enforce-eager**（已对 setup_commands.json 实测：两个都是 0 次），
         所以运行时真正决定 admit limit 的就是这一处，v27 实测跑出来的就是 8。
      B. for 循环的替换值      —— 上游自带 --max-num-seqs 时走这里。
      C. else fallback         —— 上游完全没提 max-num-seqs 时走这里。

    只改 B/C 而漏掉 A，运行时仍然是 8 ⇒ solver 并发提到 16 之后请求会在 vLLM
    的 admission gate 上排队，覆盖率提升 **完全落空**。这就是三处必须同改的原因。

    改法用行级识别而不是精确字符串锚点：v27 的原始写法把同一段字面量拆成了
    几种不同的引号/缩进组合，靠手写锚点极易只命中一处（本次就踩了：
    fallback 锚点多了一个前导 `"`，命中 0 次）。
    """
    lines = src.split("\n")
    hits = 0
    for i, line in enumerate(lines):
        if "'--max-num-seqs'" not in line:
            continue
        # 写死的 '8' 可能在本行（A/B/C 里有两处同行、一处下一行）
        for j in range(i, min(i + 3, len(lines))):
            if "'8'" in lines[j] and "_old_seqs" not in lines[j]:
                lines[j] = lines[j].replace("'8'", f"'{VLLM_MAX_NUM_SEQS}'", 1)
                hits += 1
                break
    if hits != 3:
        raise SystemExit(
            f"[v28] max-num-seqs 写死值命中 {hits} 处（应为 3：A 注入 / B for 循环 / C fallback）"
        )
    src = "\n".join(lines)
    return _inject_seq_guard(src)


# 运行期兜底：静态替换之外，再加一道正则归一 + 断言。
# 覆盖率是 v28 唯一的杠杆，admit limit 被静默改回 8 就等于白跑 9 小时，
# 所以这里宁可响亮地炸掉，也不能带着 8 静悄悄跑完。
_SEQ_GUARD = r'''

# --- lingjing v28: --max-num-seqs 运行期归一 + 断言 ---
# 上游 setup 命令不带 --max-num-seqs（实测 0 次），admit limit 完全由 cell 8 里
# 注入的那几处决定。静态替换已经把三处全改成 16，这里再兜一层：无论上游命令
# 怎么漂移，最终传给 vLLM 的 admit limit 必须是 16。
# 断言放在 cell 8（setup 阶段），炸掉只浪费 setup 时间，不会白烧 9 小时 benchmark。
import re as _re

_v28_seq_re = _re.compile(r"(--max-num-seqs',\s*)'(\d+)'")
for _i8, _c8 in enumerate(_patched_cmds):
    if isinstance(_c8, str) and "--max-num-seqs" in _c8:
        _patched_cmds[_i8] = _v28_seq_re.sub(r"\g<1>'16'", _c8)
_v28_vals = set()
for _c8 in _patched_cmds:
    if isinstance(_c8, str):
        _v28_vals.update(_re.findall(r"--max-num-seqs',\s*'(\d+)'", _c8))
print("LINGJING_MAX_NUM_SEQS", sorted(_v28_vals), flush=True)
assert _v28_vals == {"16"}, "LINGJING_MAX_NUM_SEQS_BAD %s" % sorted(_v28_vals)
'''


def _inject_seq_guard(src: str) -> str:
    """把 _SEQ_GUARD 插到 _patched_cmds 建好、命令写盘执行之前。"""
    if "LINGJING_MAX_NUM_SEQS" in src:
        return src
    lines = src.split("\n")
    idx = [i for i, ln in enumerate(lines) if "_setup_commands.lingjing.json" in ln]
    if len(idx) != 1:
        raise SystemExit(
            f"[v28] _setup_commands.lingjing.json 锚点命中 {len(idx)} 次（应为 1）"
        )
    at = idx[0]
    new_lines = lines[:at] + _SEQ_GUARD.strip("\n").split("\n") + [""] + lines[at:]
    return "\n".join(new_lines)


def patch_solver_overlay(src: str, solver_b64: str, solver_sha: str) -> str:
    anchor = 'print("taaf.kaggle: Franzen overlay merged", _overlay_dst, "files", _copied, flush=True)'
    inject = anchor + '''

# --- lingjing v28: activate the no-progress give-up without publishing a new
# overlay dataset version (the Kaggle dataset upload path goes through
# www.googleapis.com, which is unreachable from this network, so the overlay on
# Kaggle is still the pre-v27 revision that lacks the feature).
#
# The remote overlay's solver.py is a STRICT SUBSET of the local v27 one
# (0 deleted lines, 4 hunks, 69 added lines -- verified with difflib), so
# replacing it wholesale is safe.  The bytes below are that file verbatim.
try:
    import base64 as _lj_b64
    import hashlib as _lj_hashlib

    _lj_solver = _overlay_dst / "inference" / "framework" / "solver.py"
    _lj_blob = _lj_b64.b64decode("%s")
    _lj_sha = "%s"
    if _lj_solver.is_file() and _lj_solver.read_bytes() == _lj_blob:
        print("taaf.kaggle: solver.py already v28", flush=True)
    else:
        _lj_solver.write_bytes(_lj_blob)
    _lj_got = _lj_hashlib.sha256(_lj_solver.read_bytes()).hexdigest()
    if _lj_got != _lj_sha:
        raise RuntimeError(f"LINGJING_SOLVER_SHA_MISMATCH {_lj_got} != {_lj_sha}")
    print("taaf.kaggle: solver.py patched to v28", _lj_solver,
          "sha", _lj_got[:12], flush=True)
except Exception as _lj_exc:
    # 刻意 **不** 抛出：overlay 覆盖失败不该让整轮 9 小时白跑。
    # 退化后果是 give-up 开关失效（覆盖��� ≈ 52/110 而非 84/110），仍然好过 v27 的 45。
    print("taaf.kaggle: SOLVER_PATCH_FAILED", repr(_lj_exc), flush=True)
    print("taaf.kaggle: !!! v28 give-up switch NOT active, coverage degrades to ~52/110", flush=True)
''' % (solver_b64, solver_sha)
    src = _replace_once(src, anchor, inject, "cell8 overlay 合并锚点")
    return src


# --------------------------------------------------------------------------
# 3. cell 12：solver 配置向官方靠
# --------------------------------------------------------------------------
def patch_solver_config(src: str) -> str:
    src = _replace_once(
        src,
        "bm.solver.concurrency = 8",
        f"bm.solver.concurrency = {SOLVER_CONCURRENCY}",
        "cell12 concurrency",
    )
    src = _replace_once(
        src,
        "bm.solver.max_runtime_s_per_game = 5400",
        f"bm.solver.max_runtime_s_per_game = {MAX_RUNTIME_S_PER_GAME}",
        "cell12 max_runtime_s_per_game",
    )
    # giveup 守卫：v28 已通过覆盖 solver.py 真正拿到该属性，守卫应命中而不是打印 UNAVAILABLE
    return src


# --------------------------------------------------------------------------
# 4. cell 14：environment_files 路径探测 + 短 bench
# --------------------------------------------------------------------------
def patch_env_files_and_short_bench(src: str) -> str:
    old = '''    competition_env_files = str(Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels").parent / "environment_files")
    bm.games = _offline_games(competition_env_files)'''
    new = '''    # lingjing v28: probe the wheelhouse's sibling environment_files instead of
    # hardcoding one mount prefix (the competition dataset path prefix varies).
    _lj_env_dirs = []
    for _lj_w in (
        Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels"),
        Path("/kaggle/input/arc-prize-2026-arc-agi-3/arc_agi_3_wheels"),
    ):
        if _lj_w.is_dir():
            _lj_env_dirs.append(_lj_w.parent / "environment_files")
    if not _lj_env_dirs:
        for _lj_w in sorted(Path("/kaggle/input").rglob("arc_agi_3_wheels")):
            _lj_env_dirs.append(_lj_w.parent / "environment_files")
    _lj_env_dirs = [d for d in _lj_env_dirs if d.is_dir()]
    if not _lj_env_dirs:
        raise RuntimeError("ENVIRONMENT_FILES_NOT_FOUND under /kaggle/input")
    competition_env_files = str(_lj_env_dirs[0])
    print("taaf.kaggle: environment_files", competition_env_files, flush=True)
    bm.games = _offline_games(competition_env_files)
    # lingjing v28: SHORT BENCH -- only outside a real competition rerun.
    # Pushing the kernel used to run the full 25-game bench (5400 s/game, 8-way
    # => ~8.3 h of GPU), which no GPU quota could afford and which made A/B
    # impossible (one variant per day).  Shrink it so a push costs ~1 h.
    _lj_short_s = int(os.environ.get("LINGJING_SHORT_BENCH_RUNTIME_S", "%d") or %d)
    _lj_short_n = int(os.environ.get("LINGJING_SHORT_BENCH_GAMES", "%d") or %d)
    if _lj_short_s > 0:
        bm.solver.max_runtime_s_per_game = min(bm.solver.max_runtime_s_per_game, _lj_short_s)
    if 0 < _lj_short_n < len(bm.games):
        bm.games = bm.games[:_lj_short_n]
    print("taaf.kaggle: SHORT_BENCH", "per_game_s", bm.solver.max_runtime_s_per_game,
          "games", len(bm.games), flush=True)''' % (
        SHORT_BENCH_RUNTIME_S, SHORT_BENCH_RUNTIME_S,
        SHORT_BENCH_GAMES, SHORT_BENCH_GAMES,
    )
    src = _replace_once(src, old, new, "cell14 environment_files + 短 bench")
    return src


def build() -> dict:
    nb = json.loads(V27_NB.read_text(encoding="utf-8"))

    solver_bytes = SOLVER_SRC.read_bytes()
    solver_b64 = base64.b64encode(solver_bytes).decode("ascii")
    solver_sha = hashlib.sha256(solver_bytes).hexdigest()

    touched = []
    for idx, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        orig = src
        if "pip" in src and "arc_agi_3_wheels" in src and "--find-links" in src:
            src = patch_wheels(src)
            touched.append(f"cell{idx}: wheels 探测")
        if "taaf.kaggle: Franzen overlay merged" in src:
            src = patch_max_num_seqs(src)
            src = patch_solver_overlay(src, solver_b64, solver_sha)
            touched.append(f"cell{idx}: max-num-seqs + solver.py 覆盖")
        if "LINGJING_BUDGET" in src:
            src = patch_solver_config(src)
            touched.append(f"cell{idx}: solver 配置回归官方")
        if "competition_env_files" in src:
            src = patch_env_files_and_short_bench(src)
            touched.append(f"cell{idx}: environment_files 探测 + 短 bench")
        if src != orig:
            cell["source"] = _as_source(src)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")

    meta = {
        "id": SLUG,
        "title": TITLE,
        "code_file": OUT_NB.name,
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
            "bilgisarmal/vllm031-cp313-wheelhouse",
            "driessmit1/qwen3-8-27b-fp8-hf-017b9c7a",
            "liaojiaxuan/lingjing-duck-franzen-overlay",
        ],
        "kernel_sources": [],
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "model_sources": [],
    }
    OUT_META.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return {"touched": touched, "solver_sha": solver_sha,
            "solver_b64_len": len(solver_b64), "nb": str(OUT_NB)}


def check() -> int:
    nb = json.loads(OUT_NB.read_text(encoding="utf-8"))
    src = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
    must = [
        ("wheels 探测", "WHEELS_NOT_FOUND"),
        ("environment_files 探测", "ENVIRONMENT_FILES_NOT_FOUND"),
        ("短 bench", "SHORT_BENCH"),
        ("solver.py 覆盖", "solver.py patched to v28"),
        ("并发 16", f"bm.solver.concurrency = {SOLVER_CONCURRENCY}"),
        ("每局 7920", f"bm.solver.max_runtime_s_per_game = {MAX_RUNTIME_S_PER_GAME}"),
        ("max-num-seqs 16", f"'--max-num-seqs',\" + _nl + \"        '{VLLM_MAX_NUM_SEQS}'\""),
        ("max-num-seqs 注入分支 16", f"+ \"        '{VLLM_MAX_NUM_SEQS}',\""),
        ("max-num-seqs 运行期守卫", "LINGJING_MAX_NUM_SEQS"),
        ("max-num-seqs 守卫断言", "LINGJING_MAX_NUM_SEQS_BAD"),
    ]
    bad = []
    for label, needle in must:
        if needle not in src:
            bad.append(f"缺: {label} ({needle[:60]})")
    # 不能出现旧值
    for label, needle in (
        ("旧并发 8", "bm.solver.concurrency = 8"),
        ("旧每局 5400", "bm.solver.max_runtime_s_per_game = 5400"),
        ("旧 max-num-seqs 8", "'--max-num-seqs',\" + _nl + \"        '8'\""),
        ("旧 max-num-seqs 注入分支 8", "+ \"        '8',\""),
    ):
        if needle in src:
            bad.append(f"仍有旧值: {label}")
    for b in bad:
        print("[FAIL]", b)
    if bad:
        return 1
    print("[ok] v28 notebook 全部改动到位")
    print("     nb size:", OUT_NB.stat().st_size)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        return check()
    info = build()
    for t in info["touched"]:
        print("  改:", t)
    print("solver sha:", info["solver_sha"], "b64 len:", info["solver_b64_len"])
    print("written:", info["nb"])
    return check()


if __name__ == "__main__":
    raise SystemExit(main())
