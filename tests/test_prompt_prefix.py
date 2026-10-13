"""提示词分段：**共享前缀必须字节不变**，这是前缀缓存能命中的前提。"""

from __future__ import annotations

import json

from lingjing_solo.harness.object_graph import build_graph
from lingjing_solo.harness.prompt import (
    SYSTEM_PREFIX,
    build_prompt,
    estimate_tokens,
    prefix_signature,
    prompt_stats,
)


def _grid(n=6, cells=()):
    g = [[0] * n for _ in range(n)]
    for r, c, v in cells:
        g[r][c] = v
    return g


A = build_graph(_grid(6, [(1, 1, 5), (5, 5, 3)]))
B = build_graph(_grid(9, [(2, 2, 7), (2, 3, 7), (8, 0, 9)]))


# --------------------------------------------------------------------------- #
# ★ 核心不变量：前缀跨帧 / 跨回合 / 跨局都不变
# --------------------------------------------------------------------------- #


def test_prefix_is_invariant_across_graphs_turns_and_memory():
    p1 = build_prompt(A, turn=0).prefix
    p2 = build_prompt(
        B,
        history=[{"turn": 0, "action": "DOWN", "moved": True, "changed": True}],
        last_result="moved",
        memory={"plan": ["a", "b"], "n": 3},
        turn=999,
    ).prefix
    assert p1 == p2 == SYSTEM_PREFIX


def test_prefix_signature_stable_across_calls():
    assert prefix_signature() == prefix_signature()
    assert len(prefix_signature()) == 16


def test_text_starts_with_system_prefix():
    parts = build_prompt(A, turn=3)
    assert parts.text.startswith(SYSTEM_PREFIX)
    assert parts.text == parts.prefix + parts.suffix


def test_shared_prefix_tokens_matches_prefix():
    st = prompt_stats()
    assert st["shared_prefix_tokens"] == estimate_tokens(SYSTEM_PREFIX)
    assert st["prefix_chars"] == len(SYSTEM_PREFIX)
    assert st["shared_prefix_tokens"] > 500, "前缀太短就吃不到缓存收益，需要够大"


def test_prefix_contains_no_volatile_markers():
    """前缀里不允许出现回合号/历史/记忆这类会变的东西（示例块里的 FRAME 字面量除外）。"""
    for bad in ("CURRENT TURN", "RECENT HISTORY", "YOUR MEMORY", "=== NOW"):
        assert bad not in SYSTEM_PREFIX


# --------------------------------------------------------------------------- #
# 后缀：该变的要变
# --------------------------------------------------------------------------- #


def test_suffix_changes_with_graph():
    assert build_prompt(A, turn=0).suffix != build_prompt(B, turn=0).suffix


def test_suffix_changes_with_turn():
    assert build_prompt(A, turn=0).suffix != build_prompt(A, turn=1).suffix


def test_suffix_contains_object_lines():
    s = build_prompt(A, turn=0).suffix
    assert "FRAME 6x6" in s
    for o in A.objects:
        assert f"#{o.id} c={o.color}" in s


def test_suffix_history_tail_omits_old_steps():
    hist = [
        {"turn": i, "action": "UP", "moved": True, "changed": False} for i in range(30)
    ]
    s = build_prompt(A, history=hist, turn=30, history_tail=5).suffix
    assert "25 earlier steps omitted" in s
    assert "t=29" in s
    assert "t=1 " not in s and "t=1\n" not in s


def test_suffix_memory_sorted_and_truncated():
    mem = {"z": 1, "a": "x" * 400}
    s = build_prompt(A, memory=mem, turn=0).suffix
    assert "a = " in s and "z = 1" in s
    assert s.index("a = ") < s.index("z = ")  # 键排序 -> 稳定
    assert "..." in s  # 超长被截断


def test_suffix_empty_history_and_memory_are_marked():
    s = build_prompt(A, turn=0).suffix
    assert "(no steps yet)" in s
    assert "(empty)" in s


def test_prompt_stats_grows_with_suffix():
    st = prompt_stats(graph=B, turn=0)
    assert st["suffix_tokens"] > 0
    assert st["total_tokens"] == st["prefix_tokens"] + st["suffix_tokens"]


def test_estimate_tokens_monotonic_and_sane():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 100) > estimate_tokens("a" * 40)
    assert estimate_tokens(SYSTEM_PREFIX) == (len(SYSTEM_PREFIX) + 3) // 4


# --------------------------------------------------------------------------- #
# 给容量模型的接口
# --------------------------------------------------------------------------- #


def test_prompt_stats_is_json_safe():
    json.dumps(prompt_stats(graph=A, turn=2))


def test_prompt_stats_feeds_capacity_workload():
    """真实共享前缀 token 数能直接填进容量模型。"""
    from lingjing_solo.harness.capacity import (
        HARDWARE,
        MODELS,
        Workload,
        estimate,
    )

    st = prompt_stats(graph=A, turn=0)
    wl = Workload(shared_prefix_tokens=st["shared_prefix_tokens"], ctx_tokens=12_000)
    e = estimate(MODELS["qwen3-27b-fp8"], HARDWARE["rtx6000-96g"], wl)
    assert e["shared_prefix_tokens"] == st["shared_prefix_tokens"]
    assert e["concurrency"] >= 1
