"""``lingjing_solo.harness.capacity`` 的单元测试。

纯 stdlib，不依赖 GPU / numpy。覆盖：两条显存公式、单调性、并发语义、
可行性判定、以及 CLI 冒烟。
"""

from __future__ import annotations

import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingjing_solo.harness import capacity as C  # noqa: E402


Q27 = C.MODELS["qwen3-27b-fp8"]
RTX = C.HARDWARE["rtx6000-96g"]
WL = C.Workload()


# --------------------------------------------------------------------------- #
# 公式
# --------------------------------------------------------------------------- #

def test_weight_bytes_fp8_with_overhead():
    # 27e9 参数 × 1 byte × 1.06 overhead
    assert Q27.weight_bytes == pytest.approx(27e9 * 1.06)
    assert Q27.weight_bytes / C.GIB == pytest.approx(26.65, abs=0.01)


def test_weight_bytes_w4_is_half_of_fp8():
    w4 = C.MODELS["qwen3-27b-int4"]
    assert w4.weight_bytes == pytest.approx(Q27.weight_bytes / 2.0)


def test_kv_bytes_per_token_gqa():
    # 2(K,V) × 64 层 × 8 个 KV head × 128 维 × 1 byte
    assert Q27.kv_bytes_per_token == pytest.approx(2 * 64 * 8 * 128)
    assert Q27.kv_bytes_per_token / 1024 == pytest.approx(128.0)


def test_kv_bytes_per_token_mla_uses_latent_dim():
    mla = C.MODELS["mla-30b-fp8"]
    assert mla.kv_bytes_per_token == pytest.approx(61 * 576)
    # MLA 的紧凑表示应当显著小于同层数 GQA
    assert mla.kv_bytes_per_token < C.MODELS["llama-70b-fp8"].kv_bytes_per_token


def test_kv_quantization_halves_kv():
    kv4 = C.MODELS["qwen3-27b-int4kv4"]
    assert kv4.kv_bytes_per_token == pytest.approx(Q27.kv_bytes_per_token / 2.0)


# --------------------------------------------------------------------------- #
# 单调性
# --------------------------------------------------------------------------- #

def test_prefix_hit_never_hurts():
    """前缀命中率越高，prefill 越省，总耗时不该变长。"""
    prev = math.inf
    for p in (0.0, 0.25, 0.5, 0.75, 0.9, 0.99):
        e = C.estimate(Q27, RTX, C.Workload(prefix_hit=p))
        assert e["t_total_s"] <= prev + 1e-9, f"prefix_hit={p} 反而更慢"
        prev = e["t_total_s"]


def test_prefix_hit_zero_maximizes_prefill():
    e0 = C.estimate(Q27, RTX, C.Workload(prefix_hit=0.0))
    e9 = C.estimate(Q27, RTX, C.Workload(prefix_hit=0.9))
    assert e0["t_prefill_ms"] > e9["t_prefill_ms"]
    assert e0["prefill_tokens"] == pytest.approx(12_000)


def test_output_tokens_linear_in_decode():
    a = C.estimate(Q27, RTX, C.Workload(output_tokens=200))
    b = C.estimate(Q27, RTX, C.Workload(output_tokens=400))
    assert b["t_decode_ms"] == pytest.approx(2 * a["t_decode_ms"], rel=1e-6)


def test_ctx_monotonic_on_decode_at_fixed_concurrency():
    """固定并发时，ctx 越长 decode 越慢（每 token 要读更多 KV）。"""
    small = C.estimate(Q27, RTX, C.Workload(ctx_tokens=4_000, concurrency=16))
    big = C.estimate(Q27, RTX, C.Workload(ctx_tokens=16_000, concurrency=16))
    assert big["t_decode_ms"] > small["t_decode_ms"]


def test_auto_concurrency_pins_total_kv_tokens():
    """★ 反直觉但正确：自动并发下，ctx 变大只是把 batch 变小，
    常驻 KV token 量被显存钉住 —— 于是单轮成本恒定，而**总耗时 ∝ ctx**
    （因为轮数 = 调用数 / 并发，并发 ∝ 1/ctx）。"""
    small = C.estimate(Q27, RTX, C.Workload(ctx_tokens=4_000))
    big = C.estimate(Q27, RTX, C.Workload(ctx_tokens=16_000))
    assert big["resident_kv_tokens"] == pytest.approx(small["resident_kv_tokens"], rel=0.02)
    assert big["concurrency"] < small["concurrency"]
    assert big["t_iter_s"] == pytest.approx(small["t_iter_s"], rel=0.05)
    # 总耗时的放大倍数应当≈并发缩小的倍数
    assert big["t_total_s"] / small["t_total_s"] == pytest.approx(
        small["concurrency"] / big["concurrency"], rel=0.05)


# --------------------------------------------------------------------------- #
# 共享前缀（RadixAttention / 块共享）
# --------------------------------------------------------------------------- #

def test_shared_prefix_raises_concurrency_cap():
    base = C.estimate(Q27, RTX, C.Workload())
    sh = C.estimate(Q27, RTX, C.Workload(shared_prefix_tokens=6_000))
    assert sh["concurrency"] > base["concurrency"]
    assert sh["concurrency_vram_cap"] > base["concurrency_vram_cap"]


def test_shared_prefix_shortens_total_time_without_changing_iter():
    """共享前缀的价值在**并发上限**，不在省 prefill —— 单轮成本基本不变。"""
    base = C.estimate(Q27, RTX, C.Workload())
    sh = C.estimate(Q27, RTX, C.Workload(shared_prefix_tokens=6_000))
    assert sh["t_iter_s"] == pytest.approx(base["t_iter_s"], rel=0.05)
    assert sh["t_total_s"] < base["t_total_s"] * 0.7


def test_resident_tokens_formula():
    e = C.estimate(Q27, RTX, C.Workload(shared_prefix_tokens=2_000))
    assert e["resident_kv_tokens"] == pytest.approx(
        e["shared_prefix_tokens"] + e["concurrency"] * e["private_tokens"])


def test_shared_prefix_clamped_below_ctx():
    e = C.estimate(Q27, RTX, C.Workload(ctx_tokens=1_000, shared_prefix_tokens=999_999))
    assert e["shared_prefix_tokens"] <= 999
    assert e["private_tokens"] >= 1
    assert math.isfinite(e["t_iter_s"])


def test_shared_prefix_never_worse():
    prev = math.inf
    for s in (0, 2_000, 4_000, 6_000, 9_000):
        e = C.estimate(Q27, RTX, C.Workload(shared_prefix_tokens=s))
        assert e["t_total_s"] <= prev + 1e-9
        prev = e["t_total_s"]


def test_bigger_model_is_slower():
    order = ["qwen3-4b-fp8", "qwen3-8b-fp8", "qwen3-27b-fp8"]
    times = [C.estimate(C.MODELS[k], RTX, WL)["t_total_s"] for k in order]
    assert times == sorted(times), times


# --------------------------------------------------------------------------- #
# 并发语义
# --------------------------------------------------------------------------- #

def test_auto_concurrency_equals_vram_cap():
    e = C.estimate(Q27, RTX, WL)
    assert e["concurrency"] == e["concurrency_vram_cap"]
    assert e["concurrency"] > 0


def test_explicit_concurrency_respected():
    e = C.estimate(Q27, RTX, C.Workload(concurrency=8))
    assert e["concurrency"] == 8


def test_concurrency_clamped_to_at_least_one():
    e = C.estimate(Q27, RTX, C.Workload(concurrency=-3, ctx_tokens=10 ** 7))
    assert e["concurrency"] == 1


def test_concurrency_above_games_is_flagged():
    e = C.estimate(Q27, RTX, C.Workload(concurrency=32))
    assert e["conc_exceeds_games"] is True
    assert e["games_in_flight"] == 16
    assert e["calls_per_game_in_flight"] == pytest.approx(2.0)


def test_more_concurrency_raises_throughput_until_vram():
    """并发放大摊薄权重读取，吞吐应先升；超过显存上限则被容量约束拦下。"""
    t8 = C.estimate(Q27, RTX, C.Workload(concurrency=8))["throughput_dec_s"]
    t16 = C.estimate(Q27, RTX, C.Workload(concurrency=16))["throughput_dec_s"]
    assert t16 > t8


# --------------------------------------------------------------------------- #
# 可行性判定
# --------------------------------------------------------------------------- #

def test_huge_ctx_fails_on_vram():
    e = C.estimate(Q27, RTX, C.Workload(ctx_tokens=2_000_000))
    assert e["vram_ok"] is False
    assert e["verdict"] == "FAIL"
    assert e["binding"] == "vram"


def test_70b_does_not_fit_on_l4():
    e = C.estimate(C.MODELS["llama-70b-fp8"], C.HARDWARE["l4"], WL)
    assert e["vram_ok"] is False
    assert e["verdict"] == "FAIL"


def test_tiny_budget_is_time_bound():
    e = C.estimate(Q27, RTX, C.Workload(budget_hours=0.2))
    assert e["verdict"] == "TIGHT"
    assert e["binding"] == "time-hard"


def test_baseline_fits_comfortably():
    e = C.estimate(Q27, RTX, WL)
    assert e["verdict"] == "OK"
    assert e["t_total_s"] <= 9 * C.SEC_PER_HOUR
    assert e["decisions_capacity"] > e["decisions_needed"]


def test_vram_ok_implies_kv_fits():
    e = C.estimate(Q27, RTX, WL)
    assert e["vram_needed_gib"] <= e["kv_budget_gib"] + 1e-6


def test_cpu_only_hardware_fails():
    e = C.estimate(Q27, C.HARDWARE["cpu-only"], WL)
    assert e["verdict"] == "FAIL"


def test_decisions_capacity_scales_with_budget():
    a = C.estimate(Q27, RTX, C.Workload(budget_hours=3.0))
    b = C.estimate(Q27, RTX, C.Workload(budget_hours=6.0))
    assert b["decisions_capacity"] == pytest.approx(2 * a["decisions_capacity"], rel=1e-6)


# --------------------------------------------------------------------------- #
# 工具函数 / CLI
# --------------------------------------------------------------------------- #

def test_max_ctx_for_is_consistent_with_estimate():
    e = C.estimate(Q27, RTX, C.Workload(concurrency=16))
    cap = C.max_ctx_for(Q27, RTX, 16)
    assert e["vram_ok"] is True
    # 手算：KV 预算 57.59 GiB / (128 KiB × 16)
    assert cap == pytest.approx(57.59 * C.GIB / (128 * 1024 * 16), rel=0.02)
    assert cap > 12_000, "16 路并发下 12k 上下文应当放得下且有余量"
    assert C.max_ctx_for(Q27, RTX, 0) == 0.0


def test_estimate_is_json_serializable():
    s = json.dumps(C.estimate(Q27, RTX, WL))
    assert "verdict" in json.loads(s)


def test_format_estimate_mentions_verdict():
    txt = C.format_estimate(C.estimate(Q27, RTX, WL))
    assert "判定" in txt and "OK" in txt


@pytest.mark.parametrize("axis", ["model", "prefix_hit", "shared_prefix", "ctx", "concurrency",
                                  "output", "decisions", "hardware"])
def test_sweep_smoke(axis):
    txt = C.sweep(axis, C.DEFAULT_MODEL, C.DEFAULT_HW, WL)
    assert "verdict" in txt
    assert len(txt.strip().splitlines()) >= 3


def test_cli_presets_and_estimate(capsys):
    assert C.main(["presets"]) == 0
    assert C.main(["estimate", "--json"]) == 0
    out = capsys.readouterr().out
    assert "verdict" in out


def test_cli_unknown_axis_exits():
    with pytest.raises(SystemExit):
        C.main(["sweep", "--axis", "nope"])
