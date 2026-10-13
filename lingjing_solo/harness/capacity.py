"""C-Local（本地大模型 serving）容量与吞吐估算。

纯 stdlib、无 GPU 依赖。存在的理由很实际：**本仓库的开发机没有 GPU**
（venv 里是 ``torch-2.13.0+cpu``，也没有 vllm / sglang），所以「把开源模型塞进
Kaggle 的 RTX 6000（96GB）之后，9 小时预算里能跑完多少决策步」这个问题**没法在本机实测**，
只能按第一性原理先算清楚，再把实测放到真机上做。

三类输入
--------
``ModelSpec``     模型体型、KV 结构、量化位宽
``HardwareSpec``  显存 / 带宽 / 峰值算力 / 各类利用率
``Workload``      局数、每局决策步数、上下文长度、前缀缓存命中率、时间预算

估计模型（四条公式，全部写在代码里可复核）
------------------------------------------
1. **权重显存**：``params * bits/8 * overhead``
2. **每 token KV**：``2 * layers * kv_heads * head_dim * kv_bits/8``
   （MLA 结构改用 ``layers * mla_kv_dim``）
3. **单轮延迟**（一轮 = 并发 batch 的一次完整「prefill + decode」）：
   - prefill 是**算力受限**：``2N * 未命中前缀 token / (峰值 * MFU)``
     —— 前缀缓存命中部分不重算，这正是 RadixAttention / Prefix Cache 的全部价值
   - decode 是**带宽受限**：每生成一个 token 要读一遍权重 + 全部活跃序列的 KV，
     ``t = (W + B*ctx*kv_per_tok) / (带宽 * MBU)``
4. **总耗时**：``ceil(总调用数 / 并发) * 单轮延迟``

判定的关键不是「快不快」，而是**绑定约束是显存还是时间** —— 这两者的优化方向完全相反。

CLI
---
    python -m lingjing_solo.harness.capacity estimate
    python -m lingjing_solo.harness.capacity estimate --model qwen3-27b-fp8 --prefix-hit 0.9
    python -m lingjing_solo.harness.capacity sweep --axis model
    python -m lingjing_solo.harness.capacity sweep --axis prefix_hit
    python -m lingjing_solo.harness.capacity sweep --axis ctx
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

GIB = 1024 ** 3
SEC_PER_HOUR = 3600.0
SEC_PER_MIN = 60.0


# --------------------------------------------------------------------------- #
# 输入规格
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ModelSpec:
    """模型体型与 KV 结构。

    ``weight_overhead`` 覆盖量化 scale / embedding 未量化部分带来的额外占用。
    ``mla_kv_dim`` 非 None 时改用 MLA 的紧凑 KV 表示（层 × 潜在维），忽略 kv_heads。
    """

    name: str
    params_b: float
    layers: int
    kv_heads: int
    head_dim: int
    weight_bits: float = 8.0
    kv_bits: float = 8.0
    mla_kv_dim: Optional[int] = None
    weight_overhead: float = 1.06

    @property
    def weight_bytes(self) -> float:
        return self.params_b * 1e9 * (self.weight_bits / 8.0) * self.weight_overhead

    @property
    def kv_bytes_per_token(self) -> float:
        if self.mla_kv_dim is not None:
            elems = float(self.layers) * float(self.mla_kv_dim)
        else:
            elems = 2.0 * float(self.layers) * float(self.kv_heads) * float(self.head_dim)
        return elems * (self.kv_bits / 8.0)


@dataclass(frozen=True)
class HardwareSpec:
    """显卡规格。``fp8_tflops`` 是**假定值**，不是实测 —— 换真机后应替换为 profile 结果。"""

    name: str
    vram_gb: float
    bw_gbs: float
    fp8_tflops: float
    usable_frac: float = 0.94      # 驱动 / CUDA context 之外的可用比例
    reserve_gb: float = 6.0        # 激活、CUDA graph、通信缓冲
    mfu_prefill: float = 0.35      # prefill 算力利用率
    mbu_decode: float = 0.75       # decode 带宽利用率

    @property
    def vram_bytes(self) -> float:
        return self.vram_gb * GIB

    @property
    def bw_bytes_s(self) -> float:
        return self.bw_gbs * 1e9

    @property
    def flops_s(self) -> float:
        return self.fp8_tflops * 1e12


@dataclass(frozen=True)
class Workload:
    """工作负载。``concurrency=0`` 表示「显存允许多少路就用多少路」。

    ``shared_prefix_tokens`` 是**所有并发序列共享的公共前缀长度**
    （system prompt + 对象图模板 + 已驱逐后仍保留的历史头）。
    PagedAttention 的块共享 / RadixAttention 的基数树只把它**存一份**，
    因此它同时抬高并发上限、压低单轮 KV 读取量 —— 这是前缀缓存真正值钱的地方
    （而不是省那点 prefill）。
    """

    games: int = 16
    decisions_per_game: int = 400
    calls_per_decision: float = 1.0
    ctx_tokens: int = 12_000
    output_tokens: int = 400
    prefix_hit: float = 0.85
    shared_prefix_tokens: int = 0
    budget_hours: float = 9.0
    concurrency: int = 0

    @property
    def total_calls(self) -> float:
        return self.games * self.decisions_per_game * self.calls_per_decision


# --------------------------------------------------------------------------- #
# 预置规格
# --------------------------------------------------------------------------- #

HARDWARE: Dict[str, HardwareSpec] = {
    # Kaggle 为本竞赛提供的机器：GCP g4-standard-48 / 单卡 RTX PRO 6000。
    # fp8_tflops 取保守偏中值；decode 是带宽受限，该值对结论不敏感。
    "rtx6000-96g": HardwareSpec("RTX PRO 6000 (g4-standard-48)", 96.0, 1792.0, 1000.0),
    "b200": HardwareSpec("B200", 180.0, 8000.0, 4500.0),
    "t4x2": HardwareSpec("2x T4", 32.0, 640.0, 130.0, reserve_gb=4.0),
    "l4": HardwareSpec("L4", 24.0, 300.0, 120.0, reserve_gb=4.0),
    "cpu-only": HardwareSpec("CPU only（本机现状）", 0.0, 0.0, 0.0),
}

MODELS: Dict[str, ModelSpec] = {
    "qwen3-4b-fp8": ModelSpec("Qwen3-4B FP8", 4.0, 36, 8, 128),
    "qwen3-8b-fp8": ModelSpec("Qwen3-8B FP8", 8.0, 36, 8, 128),
    "qwen3-27b-fp8": ModelSpec("Qwen3-27B FP8", 27.0, 64, 8, 128),
    "qwen3-27b-int4": ModelSpec("Qwen3-27B W4", 27.0, 64, 8, 128, weight_bits=4.0),
    "qwen3-27b-int4kv4": ModelSpec("Qwen3-27B W4 KV4", 27.0, 64, 8, 128,
                                   weight_bits=4.0, kv_bits=4.0),
    "llama-70b-fp8": ModelSpec("Llama-70B FP8", 70.0, 80, 8, 128),
    "mla-30b-fp8": ModelSpec("MLA-30B FP8", 30.0, 61, 0, 0, mla_kv_dim=576),
}

DEFAULT_MODEL = "qwen3-27b-fp8"
DEFAULT_HW = "rtx6000-96g"


# --------------------------------------------------------------------------- #
# 核心估计
# --------------------------------------------------------------------------- #

def estimate(model: ModelSpec, hw: HardwareSpec, wl: Workload) -> Dict[str, Any]:
    """返回一个可直接打印/序列化的估计结果。"""
    w_bytes = model.weight_bytes
    kv_budget = hw.vram_bytes * hw.usable_frac - w_bytes - hw.reserve_gb * GIB
    kv_budget = max(kv_budget, 0.0)
    kv_per_tok = model.kv_bytes_per_token

    # 显存决定的 KV 容量（token 槽位）
    token_slots = kv_budget / kv_per_tok if kv_per_tok > 0 else float("inf")

    # 显存决定的并发上限。公共前缀只存一份，所以
    #   cap = (槽位 - 共享前缀) / 每路私有部分
    ctx = wl.ctx_tokens
    shared = max(0, min(int(wl.shared_prefix_tokens), max(ctx - 1, 0)))
    private = max(ctx - shared, 1)
    if math.isinf(token_slots):
        conc_vram_cap = 10 ** 6
    else:
        conc_vram_cap = int((token_slots - shared) // private) if token_slots > shared else 0

    # 并发 = **在飞的 LLM 调用数**，不是局数。
    # 上限由显存决定；当并发 > 局数时，意味着单局同时有多个调用在飞
    # （多 pass / 多分支 / 多 seed），这正是 Duck "20 passes" 的做法。
    conc_req = wl.concurrency if wl.concurrency > 0 else max(conc_vram_cap, 1)
    conc = max(1, int(conc_req))

    # 常驻 KV token 数（共享前缀只算一次）
    resident_tokens = shared + conc * private
    need_bytes = resident_tokens * kv_per_tok
    vram_ok = need_bytes <= kv_budget + 1e-6
    fits_in_slots = token_slots >= shared  # 能否装下共享前缀本身
    # prefill：算力受限，只重算未被前缀缓存命中的部分
    prefill_tok = ctx * (1.0 - wl.prefix_hit)
    t_prefill = (2.0 * model.params_b * 1e9 * prefill_tok) / (hw.flops_s * hw.mfu_prefill) \
        if hw.flops_s > 0 else float("inf")

    # decode：带宽受限，每 token 读一遍权重 + **全部常驻 KV**
    bytes_per_token_step = w_bytes + resident_tokens * kv_per_tok
    t_tok = bytes_per_token_step / (hw.bw_bytes_s * hw.mbu_decode) if hw.bw_bytes_s > 0 else float("inf")
    t_decode = wl.output_tokens * t_tok

    t_iter = t_prefill + t_decode
    iters = math.ceil(wl.total_calls / conc) if conc > 0 else float("inf")
    t_total = iters * t_iter

    budget_s = wl.budget_hours * SEC_PER_HOUR
    if not vram_ok:
        verdict, binding = "FAIL", "vram"
    elif hw.bw_bytes_s <= 0 or hw.flops_s <= 0:
        verdict, binding = "FAIL", "no-compute"
    elif t_total > budget_s:
        verdict, binding = "TIGHT", "time" if t_total <= budget_s * 1.5 else "time-hard"
    else:
        verdict, binding = "OK", "time" if t_total > budget_s * 0.5 else "none"

    # 预算内能完成的决策步数（按同一并发）
    decisions_capacity = (budget_s / t_iter) * conc if t_iter > 0 else float("inf")
    decisions_needed = wl.total_calls

    return {
        "model": model.name,
        "hardware": hw.name,
        "weights_gib": w_bytes / GIB,
        "kv_per_token_kib": kv_per_tok / 1024.0,
        "kv_budget_gib": kv_budget / GIB,
        "token_slots": token_slots,
        "shared_prefix_tokens": shared,
        "private_tokens": private,
        "resident_kv_tokens": resident_tokens,
        "fits_in_slots": fits_in_slots,
        "concurrency": conc,
        "concurrency_vram_cap": conc_vram_cap,
        "conc_exceeds_games": conc > wl.games,
        "games_in_flight": min(conc, wl.games),
        "calls_per_game_in_flight": conc / float(wl.games) if wl.games else 0.0,
        "vram_needed_gib": need_bytes / GIB,
        "vram_ok": vram_ok,
        "prefill_tokens": prefill_tok,
        "t_prefill_ms": t_prefill * 1000.0,
        "t_decode_ms": t_decode * 1000.0,
        "t_iter_s": t_iter,
        "iterations": iters,
        "t_total_s": t_total,
        "t_total_min": t_total / SEC_PER_MIN,
        "budget_h": wl.budget_hours,
        "decisions_needed": decisions_needed,
        "decisions_capacity": decisions_capacity,
        "throughput_dec_s": (conc / t_iter) if t_iter > 0 else 0.0,
        "verdict": verdict,
        "binding": binding,
        "prefill_share": (t_prefill / t_iter) if t_iter > 0 else 0.0,
    }


def max_ctx_for(model: ModelSpec, hw: HardwareSpec, concurrency: int) -> float:
    """给定并发下，显存允许的最大上下文长度。"""
    kv_budget = hw.vram_bytes * hw.usable_frac - model.weight_bytes - hw.reserve_gb * GIB
    if kv_budget <= 0 or model.kv_bytes_per_token <= 0 or concurrency <= 0:
        return 0.0
    return kv_budget / (model.kv_bytes_per_token * concurrency)


# --------------------------------------------------------------------------- #
# 展示
# --------------------------------------------------------------------------- #

def _fmt(x: Any, nd: int = 2) -> str:
    if isinstance(x, float):
        if math.isinf(x):
            return "inf"
        return f"{x:,.{nd}f}"
    return str(x)


def format_estimate(e: Dict[str, Any]) -> str:
    L: List[str] = []
    L.append(f"模型   : {e['model']}")
    L.append(f"硬件   : {e['hardware']}")
    L.append("")
    L.append("[显存]")
    L.append(f"  权重占用          : {_fmt(e['weights_gib'])} GiB")
    L.append(f"  每 token KV       : {_fmt(e['kv_per_token_kib'])} KiB")
    L.append(f"  KV 可用预算       : {_fmt(e['kv_budget_gib'])} GiB  → token 槽位 {_fmt(e['token_slots'],0)}")
    if e["shared_prefix_tokens"]:
        L.append(f"  共享前缀 / 私有    : {_fmt(e['shared_prefix_tokens'],0)} / {_fmt(e['private_tokens'],0)} token"
                 f"  → 常驻 {_fmt(e['resident_kv_tokens'],0)} token（公共部分只存一份）")
    L.append(f"  当前并发所需 KV   : {_fmt(e['vram_needed_gib'])} GiB  "
             f"{'✓ 放得下' if e['vram_ok'] else '✗ 放不下'}")
    L.append(f"  并发              : {e['concurrency']}  (显存上限 {e['concurrency_vram_cap']})")
    if e.get("conc_exceeds_games"):
        L.append(f"    ⚠ 并发 > 局数({e['games_in_flight']})，等价于单局 "
                 f"{e['calls_per_game_in_flight']:.1f} 个调用在飞（多 pass / 多分支）")
    L.append("")
    L.append("[延迟 / 一轮 = 一次并发 batch 的 prefill + decode]")
    L.append(f"  未命中前缀 token  : {_fmt(e['prefill_tokens'],0)}")
    L.append(f"  prefill           : {_fmt(e['t_prefill_ms'],1)} ms   (占比 {100*e['prefill_share']:.1f}%)")
    L.append(f"  decode            : {_fmt(e['t_decode_ms'],1)} ms")
    L.append(f"  单轮合计          : {_fmt(e['t_iter_s'],2)} s")
    L.append("")
    L.append("[总账]")
    L.append(f"  调用数 / 决策步   : {_fmt(e['decisions_needed'],0)}  → {_fmt(e['iterations'],0)} 轮")
    L.append(f"  总耗时            : {_fmt(e['t_total_min'],1)} min  (预算 {e['budget_h']} h = "
             f"{_fmt(e['budget_h']*60,0)} min)")
    L.append(f"  预算内可完成步数  : {_fmt(e['decisions_capacity'],0)}  ")
    L.append(f"  决策吞吐          : {_fmt(e['throughput_dec_s'],2)} 步/秒")
    L.append("")
    L.append(f"判定 : {e['verdict']}    绑定约束 : {e['binding']}")
    return "\n".join(L)


def _row(e: Dict[str, Any], label: str) -> str:
    return (f"{label:22} {_fmt(e['weights_gib'],1):>8} {_fmt(e['kv_per_token_kib'],0):>7} "
            f"{e['concurrency']:>4} {_fmt(e['t_iter_s'],1):>8} {_fmt(e['t_total_min'],0):>9} "
            f"{_fmt(e['decisions_capacity'],0):>12}  {e['verdict']:<6} {e['binding']}")


def _header() -> str:
    return (f"{'scenario':22} {'W(GiB)':>8} {'KV/tok':>7} {'conc':>4} {'T/iter':>8} "
            f"{'T(min)':>9} {'cap(dec)':>12}  {'verdict':<6} binding")


def sweep(axis: str, model_key: str, hw_key: str, wl: Workload) -> str:
    hw = HARDWARE[hw_key]
    rows: List[str] = [_header()]

    if axis == "model":
        for key, m in MODELS.items():
            rows.append(_row(estimate(m, hw, wl), key))
    elif axis == "prefix_hit":
        m = MODELS[model_key]
        for p in (0.0, 0.5, 0.7, 0.85, 0.95, 0.99):
            rows.append(_row(estimate(m, hw, replace(wl, prefix_hit=p)), f"prefix_hit={p}"))
    elif axis == "shared_prefix":
        m = MODELS[model_key]
        for s in (0, 1_000, 2_000, 4_000, 6_000, 9_000):
            rows.append(_row(estimate(m, hw, replace(wl, shared_prefix_tokens=s)), f"shared={s}"))
    elif axis == "ctx":
        m = MODELS[model_key]
        for c in (2_000, 4_000, 8_000, 12_000, 24_000, 48_000):
            rows.append(_row(estimate(m, hw, replace(wl, ctx_tokens=c)), f"ctx={c}"))
    elif axis == "concurrency":
        m = MODELS[model_key]
        for c in (1, 2, 4, 8, 16, 32):
            rows.append(_row(estimate(m, hw, replace(wl, concurrency=c)), f"conc={c}"))
    elif axis == "output":
        m = MODELS[model_key]
        for o in (64, 128, 256, 400, 800, 1600):
            rows.append(_row(estimate(m, hw, replace(wl, output_tokens=o)), f"out_tok={o}"))
    elif axis == "decisions":
        m = MODELS[model_key]
        for d in (100, 200, 400, 800):
            rows.append(_row(estimate(m, hw, replace(wl, decisions_per_game=d)), f"dec/game={d}"))
    elif axis == "hardware":
        m = MODELS[model_key]
        for key, h in HARDWARE.items():
            rows.append(_row(estimate(m, h, wl), key))
    else:
        raise SystemExit(f"未知 axis: {axis}")
    return "\n".join(rows)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _build_wl(a: argparse.Namespace) -> Workload:
    return Workload(
        games=a.games,
        decisions_per_game=a.decisions,
        calls_per_decision=a.calls_per_decision,
        ctx_tokens=a.ctx,
        output_tokens=a.output,
        prefix_hit=a.prefix_hit,
        shared_prefix_tokens=a.shared_prefix,
        budget_hours=a.budget,
        concurrency=a.concurrency,
    )


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="capacity",
        description="C-Local 容量与吞吐估算（纯 stdlib，无需 GPU）",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS))
        sp.add_argument("--hw", default=DEFAULT_HW, choices=sorted(HARDWARE))
        sp.add_argument("--games", type=int, default=16)
        sp.add_argument("--decisions", type=int, default=400, help="每局决策步数")
        sp.add_argument("--calls-per-decision", type=float, default=1.0, help="每步的 LLM 调用数（多 pass 时 >1）")
        sp.add_argument("--ctx", type=int, default=12_000, help="上下文 token 数")
        sp.add_argument("--output", type=int, default=400, help="每次调用生成 token 数")
        sp.add_argument("--prefix-hit", type=float, default=0.85, help="前缀缓存命中率 0~1（只影响 prefill）")
        sp.add_argument("--shared-prefix", type=int, default=0,
                        help="并发序列共享的公共前缀 token 数（只存一份，抬高并发上限）")
        sp.add_argument("--budget", type=float, default=9.0, help="时间预算（小时）")
        sp.add_argument("--concurrency", type=int, default=0, help="0 = 显存允许多少用多少")
        sp.add_argument("--json", action="store_true")

    pe = sub.add_parser("estimate", help="单个场景的完整拆解")
    add_common(pe)

    ps = sub.add_parser("sweep", help="沿单一维度扫描")
    add_common(ps)
    ps.add_argument("--axis", required=True,
                    choices=["model", "prefix_hit", "shared_prefix", "ctx", "concurrency",
                             "output", "decisions", "hardware"])

    sub.add_parser("presets", help="列出预置的模型与硬件键名")

    a = p.parse_args(argv)

    if a.cmd == "presets":
        print("模型:")
        for k, m in MODELS.items():
            print(f"  {k:20} {m.name:18} {m.params_b:>5}B  KV {m.kv_bytes_per_token/1024:>6.1f} KiB/tok")
        print("硬件:")
        for k, h in HARDWARE.items():
            print(f"  {k:20} {h.name:28} {h.vram_gb:>5}GB  {h.bw_gbs:>6}GB/s")
        return 0

    wl = _build_wl(a)
    if a.cmd == "estimate":
        e = estimate(MODELS[a.model], HARDWARE[a.hw], wl)
        if a.json:
            print(json.dumps(e, ensure_ascii=False, indent=2))
        else:
            print(format_estimate(e))
    else:
        print(sweep(a.axis, a.model, a.hw, wl))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
