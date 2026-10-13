"""回合循环：代码块解析、失败容忍、环境只在成功时推进、提示词开销计量。"""

from __future__ import annotations

from lingjing_solo.harness.actions import Action
from lingjing_solo.harness.loop import (
    EpisodeResult,
    FakeEnv,
    FakeLLM,
    Harness,
    extract_code,
)
from lingjing_solo.harness.prompt import SYSTEM_PREFIX, estimate_tokens
from lingjing_solo.harness.sandbox import Sandbox


# --------------------------------------------------------------------------- #
# 代码块解析
# --------------------------------------------------------------------------- #


def test_extract_code_python_fence():
    assert extract_code("```python\naction('UP')\n```") == "action('UP')"


def test_extract_code_bare_fence():
    assert extract_code("前缀\n```\naction('UP')\n```\n后缀") == "action('UP')"


def test_extract_code_takes_first_block_only():
    txt = "```python\naction('UP')\n```\n```python\naction('DOWN')\n```"
    assert extract_code(txt) == "action('UP')"


def test_extract_code_falls_back_to_raw_when_action_present():
    assert extract_code("action('SPACE')") == "action('SPACE')"


def test_extract_code_none_when_no_code():
    assert extract_code("I will move up now.") is None
    assert extract_code("") is None
    assert extract_code(None) is None


def test_extract_code_ignores_empty_fence():
    assert extract_code("```python\n\n```") is None


# --------------------------------------------------------------------------- #
# 假环境
# --------------------------------------------------------------------------- #


def test_fake_env_step_semantics():
    env = FakeEnv(size=5, agent=(1, 1), goal=(4, 4))
    env.reset()
    out = env.step(Action("DOWN"))
    assert env.agent == (2, 1) and not out.done and out.reward == 0.0
    out = env.step(Action("SPACE"))
    assert env.agent == (2, 1) and not out.done  # SPACE 不动
    out = env.step(Action("CLICK", 0, 0))
    assert env.agent == (2, 1)  # CLICK 在玩具环境里也不动


def test_fake_env_clamps_at_border():
    env = FakeEnv(size=4, agent=(0, 0), goal=(3, 3))
    env.reset()
    env.step(Action("UP"))
    assert env.agent == (0, 0)


def test_fake_env_goal_reached():
    env = FakeEnv(size=4, agent=(0, 0), goal=(0, 1))
    env.reset()
    out = env.step(Action("RIGHT"))
    assert out.done and out.reward == 1.0


# --------------------------------------------------------------------------- #
# 端到端
# --------------------------------------------------------------------------- #


def test_harness_solves_toy_env_with_fake_llm():
    env = FakeEnv(size=8, agent=(1, 1), goal=(6, 6))
    res = Harness(FakeLLM()).run_episode(env, max_turns=60)
    assert res.solved
    assert res.turns == 10  # Chebyshev 距离 5 + 5
    assert res.failed_steps == 0
    assert res.actions[0] == "DOWN" and res.actions[-1] == "RIGHT"


def test_prefix_tokens_constant_and_matches_prompt_module():
    env = FakeEnv()
    res = Harness(FakeLLM()).run_episode(env, max_turns=60)
    assert res.prefix_tokens == estimate_tokens(SYSTEM_PREFIX)


def test_suffix_tokens_grow_with_history():
    env = FakeEnv()
    res = Harness(FakeLLM()).run_episode(env, max_turns=60)
    toks = [s.suffix_tokens for s in res.steps]
    assert toks[-1] >= toks[0]


def test_llm_receives_prompt_starting_with_prefix():
    llm = FakeLLM()
    Harness(llm).run_episode(FakeEnv(), max_turns=3)
    assert llm.calls
    assert all(c.startswith(SYSTEM_PREFIX) for c in llm.calls)


def test_moved_and_changed_flags():
    env = FakeEnv(size=6, agent=(1, 1), goal=(5, 5))
    res = Harness(FakeLLM()).run_episode(env, max_turns=20)
    assert res.steps[0].moved and res.steps[0].changed
    assert all(s.moved for s in res.steps)


# --------------------------------------------------------------------------- #
# 失败路径
# --------------------------------------------------------------------------- #


def test_env_not_advanced_on_failed_step():
    """模型没产出动作时，环境必须原地不动（否则"失败"就等于随机乱走）。"""
    llm = FakeLLM(script=["no code here", "```python\naction('DOWN')\n```", "```python\naction('DOWN')\n```"])
    env = FakeEnv(size=5, agent=(0, 0), goal=(4, 4))
    res = Harness(llm).run_episode(env, max_turns=3)
    assert res.failed_steps == 1
    assert res.steps[0].ok is False
    assert res.steps[0].action is None
    assert res.actions == ["DOWN", "DOWN"]
    assert env.agent == (2, 0)  # 只推进了两次


def test_syntax_error_step_is_recorded_not_raised():
    llm = FakeLLM(script=["```python\naction('UP'\n```", "```python\naction('DOWN')\n```"])
    res = Harness(llm).run_episode(FakeEnv(size=5, agent=(0, 0), goal=(4, 4)), max_turns=2)
    assert res.steps[0].error_kind == "syntax"
    assert res.steps[1].ok


def test_timeout_step_is_recorded():
    llm = FakeLLM(script=["```python\nwhile True:\n    pass\n```"])
    res = Harness(llm, Sandbox(timeout_s=0.05)).run_episode(FakeEnv(), max_turns=1)
    assert res.failed_steps == 1
    assert res.steps[0].error_kind == "timeout"


def test_running_out_of_turns_marks_unsolved():
    env = FakeEnv(size=8, agent=(1, 1), goal=(6, 6))
    res = Harness(FakeLLM()).run_episode(env, max_turns=3)
    assert not res.solved
    assert res.turns == 3


def test_episode_result_defaults():
    r = EpisodeResult()
    assert r.actions == [] and not r.solved and r.turns == 0
