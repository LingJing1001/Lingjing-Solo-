"""把 C-Local harness 接到**官方** ARC-AGI-3 harness 上。

官方契约定得很死（``vendor/ARC-AGI-3-Agents/agents/agent.py``）：

    class Agent(ABC):
        def is_done(self, frames, latest_frame) -> bool
        def choose_action(self, frames, latest_frame) -> GameAction

    Agent.main():
        while not is_done(...) and action_counter <= MAX_ACTIONS:
            action = choose_action(frames, latest_frame)
            take_action(action)          # 引擎推进；agent 只负责出动作

所以有**两条**接入路径，本模块分别提供：

======================  ==========================================  ====================
路径                    谁推进环境                                    入口
======================  ==========================================  ====================
离线自跑（本地评测）     我们（``Harness.run_episode``）                :class:`OfficialEnv`
官方提交（Kaggle）       官方引擎（``Agent.main`` 里的 take_action）      ``HarnessAgent``
======================  ==========================================  ====================

``HarnessAgent`` **不做成模块级类**：它必须继承官方 ``Agent``，
而官方 ``agents`` 包只在 starter 的 ``vendor/`` 里、不是通用依赖。
所以按需构造（:func:`harness_agent_class`），并用 PEP 562 的模块 ``__getattr__``
让 ``arc_env.HarnessAgent`` 这种写法照常能用。这样 ``import lingjing_solo.harness``
在没装 arcengine 的环境里依然成立 —— 单测就是这么跑的。

纯 stdlib 可导入；``arcengine`` / ``agents`` / ``arc_agi`` 全部延迟到调用时。
（注意 ``import lingjing_solo.harness`` 仍需 numpy —— 那是 ``lingjing_solo/__init__``
拉起的 ``core``，与本模块无关；本模块自身只依赖 stdlib 与 harness 内部模块。）
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core.frame_decode import grid_from_frame
from .actions import Action
from .contract import RunJournal
from .loop import (
    Decision,
    EpisodeState,
    Harness,
    LLMPolicy,
    StepOutcome,
    shape_signature,
    position_signature,
)
from .sandbox import Sandbox

__all__ = [
    "CANONICAL_TO_GAME_ACTION",
    "GAME_ACTION_TO_CANONICAL",
    "OfficialEnv",
    "canonical_action_names",
    "frame_to_grid",
    "harness_agent_class",
    "load_game_action",
    "make_arcade",
    "make_harness_agent",
    "resolve_game_action",
]

# --------------------------------------------------------------------------- #
# 动作命名：harness 规范名 ↔ 官方 ACTIONn
# --------------------------------------------------------------------------- #
# 与 lingjing_solo/core/actions.py 一致。这里**刻意复制一份**而不 import：
# core 会拉起 numpy/引擎那一串依赖，而本模块要能在纯 stdlib 下被 import。
CANONICAL_TO_GAME_ACTION: Dict[str, str] = {
    "UP": "ACTION1",     # 官方：ACTION1 = MOVE_UP
    "DOWN": "ACTION2",
    "LEFT": "ACTION3",
    "RIGHT": "ACTION4",
    "SPACE": "ACTION5",  # 交互键
    "CLICK": "ACTION6",  # 坐标点击
}
GAME_ACTION_TO_CANONICAL: Dict[str, str] = {v: k for k, v in CANONICAL_TO_GAME_ACTION.items()}

# FrameData.available_actions 里的整数 id → 官方名（含 RESET，过滤时再剔）
_ID_TO_NAME: Dict[int, str] = {
    0: "RESET", 1: "ACTION1", 2: "ACTION2", 3: "ACTION3",
    4: "ACTION4", 5: "ACTION5", 6: "ACTION6", 7: "ACTION7",
}

# 兜底顺序：优先方向键，再交互键，最后点击
_FALLBACK_ORDER = ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6")


def load_game_action():
    """延迟取官方 ``GameAction`` 枚举。没装 arcengine 时给出可读的报错。"""
    try:
        from arcengine import GameAction
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "arcengine is not importable — the official-harness bridge "
            "(OfficialEnv / HarnessAgent) is unavailable in this environment"
        ) from exc
    return GameAction


def canonical_action_names(available_actions: Any) -> List[str]:
    """``available_actions``（整数 id 或字符串）→ 官方名列表，**剔除 RESET**。

    剔除 RESET 是有意的：reset 会重置关卡，把它当普通动作交给模型等于送它
    一个免费的"重开"，历史里也留不下因果（同 ``actions.py`` 不给 Undo 的理由）。
    """
    out: List[str] = []
    for a in available_actions or []:
        name: Optional[str] = None
        if isinstance(a, str):
            candidate = a.strip().upper()
            # 字符串也必须是已知的官方名；"??" 之类的噪声不该混进可用集
            name = candidate if candidate in _ID_TO_NAME.values() else None
        else:
            try:
                name = _ID_TO_NAME.get(int(a))
            except (TypeError, ValueError):
                name = None
        if name and name != "RESET" and name not in out:
            out.append(name)
    return out


def _first_allowed(allowed: Sequence[str]) -> str:
    for cand in _FALLBACK_ORDER:
        if cand in allowed:
            return cand
    return _FALLBACK_ORDER[0]


def resolve_game_action(action: Action, available: Any = None) -> Any:
    """:class:`Action` → 官方 ``GameAction``。

    ``available`` 给定时，若目标动作不在可用集里，**降级**到第一个可用动作
    （而不是抛异常）—— 模型不该因为引擎这一帧关掉了某个键就整步作废。
    """
    GameAction = load_game_action()
    name = CANONICAL_TO_GAME_ACTION.get(action.kind)
    if name is None:
        raise ValueError(f"no official mapping for action {action.kind!r}")

    if available:
        allowed = canonical_action_names(available)
        if allowed and name not in allowed:
            name = _first_allowed(allowed)

    ga = GameAction[name]
    if name == "ACTION6":
        # 官方点击用 (x, y)；我们的 Action 用 (row, col) —— 这里是唯一的换算点
        ga.set_data({"x": int(action.col), "y": int(action.row)})
    return ga


# --------------------------------------------------------------------------- #
# 帧 → 网格
# --------------------------------------------------------------------------- #


def frame_to_grid(frame: Any) -> Optional[List[List[int]]]:
    """把官方帧（或帧列表、或裸 ndarray）压成 2D ``list[list[int]]``。

    委托给 :func:`lingjing_solo.core.frame_decode.grid_from_frame` —— 与
    ``object_graph.build_graph`` / ``core.utils.extract_grid`` 共用**同一份**解码
    （原先是三份各写各的，见该模块 docstring）。真实结构
    ``FrameData.frame = [np.ndarray(64,64)]``、"多个历史帧的列表"、裸网格
    都走同一套规则（帧栈 / 多层 → 取最后一层）。取不到返回 ``None``。
    """
    return grid_from_frame(frame)


# --------------------------------------------------------------------------- #
# 离线路径：官方 EnvironmentWrapper → harness Env
# --------------------------------------------------------------------------- #


def make_arcade(mode: str = "auto"):
    """打开官方 ``Arcade``。``auto`` = 先试 NORMAL，失败退 OFFLINE。

    注意 ``NORMAL`` 会去拉 API 上的环境清单（**需要网络**）；
    评测环境无网络时应直接 ``mode="offline"``，走 ``environment_files/`` 本地源码。
    """
    from arc_agi import Arcade, OperationMode

    if mode == "offline":
        return Arcade(operation_mode=OperationMode.OFFLINE)
    if mode == "normal":
        return Arcade(operation_mode=OperationMode.NORMAL)
    try:
        return Arcade(operation_mode=OperationMode.NORMAL)
    except Exception:  # noqa: BLE001
        return Arcade(operation_mode=OperationMode.OFFLINE)


def _action_data(ga: Any) -> Optional[Dict[str, Any]]:
    ad = getattr(ga, "action_data", None)
    if ad is None:
        return None
    if hasattr(ad, "model_dump"):
        return ad.model_dump()
    return dict(ad) if isinstance(ad, dict) else None


class OfficialEnv:
    """官方 ``EnvironmentWrapper`` → harness :class:`Env`。

    用于**本地离线跑真游戏**（``arc_agi.Arcade(OFFLINE)``）：
    ``Harness.run_episode(OfficialEnv(wrapper))`` 就能拿真帧走完对象图 →
    提示词 → LLM → 沙箱 → 动作这条链，并把每步证据写进 journal。
    """

    def __init__(self, wrapper: Any, game_id: str = "", reasoning_tag: str = "c-local") -> None:
        self.wrapper = wrapper
        self.game_id = game_id
        self.reasoning_tag = reasoning_tag
        self._raw: Any = None
        self._shape: Tuple[int, int] = (64, 64)
        self._levels: int = 0

    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape

    @property
    def available_actions(self) -> List[str]:
        return canonical_action_names(getattr(self._raw, "available_actions", None))

    def reset(self) -> Optional[List[List[int]]]:
        raw = self.wrapper.reset()
        if raw is None:
            raise RuntimeError("official environment reset() returned None")
        self._absorb(raw)
        self._levels = int(getattr(raw, "levels_completed", 0) or 0)
        return frame_to_grid(self._raw)

    def step(self, action: Action) -> StepOutcome:
        ga = resolve_game_action(action, available=getattr(self._raw, "available_actions", None))
        raw = self.wrapper.step(
            ga,
            data=_action_data(ga),
            reasoning={"text": f"{self.reasoning_tag}:{action}"},
        )
        if raw is None:
            raise RuntimeError("official environment step() returned None")
        self._absorb(raw)

        levels = int(getattr(raw, "levels_completed", 0) or 0)
        advanced = levels > self._levels
        self._levels = levels
        state = str(getattr(getattr(raw, "state", None), "name", ""))
        win_levels = int(getattr(raw, "win_levels", 0) or 0)

        return StepOutcome(
            frame=frame_to_grid(self._raw),
            # 官方没有逐步 reward；唯一有意义的即时信号是"过关了"
            reward=1.0 if advanced else 0.0,
            done=bool(advanced or state in ("WIN", "GAME_OVER") or (win_levels and levels >= win_levels)),
            info={
                "state": state,
                "levels_completed": levels,
                "win_levels": win_levels,
                "available_actions": list(getattr(raw, "available_actions", []) or []),
                "game_id": getattr(raw, "game_id", self.game_id),
            },
        )

    # -- 内部 -------------------------------------------------------------- #

    def _absorb(self, raw: Any) -> None:
        self._raw = raw
        try:
            import numpy as np

            arr = np.array(getattr(raw, "frame", None))
            if arr.ndim >= 3:
                self._shape = (int(arr.shape[-2]), int(arr.shape[-1]))
            elif arr.ndim == 2:
                self._shape = (int(arr.shape[0]), int(arr.shape[1]))
        except Exception:  # noqa: BLE001 - 形状只是提示，拿不到就保持默认
            pass


# --------------------------------------------------------------------------- #
# 提交路径：官方 Agent 子类（按需构造）
# --------------------------------------------------------------------------- #

_HARNESS_AGENT_CLS: Any = None


def harness_agent_class():
    """构造（并缓存）继承官方 ``Agent`` 的 ``HarnessAgent`` 类。"""
    global _HARNESS_AGENT_CLS
    if _HARNESS_AGENT_CLS is not None:
        return _HARNESS_AGENT_CLS

    from agents.agent import Agent  # 官方 vendor 包，只在 starter 环境里可见

    from .loop import FakeLLM

    class HarnessAgent(Agent):
        """把 C-Local harness 当作官方 Agent 的"大脑"。

        与 :class:`Harness.run_episode` 的分工：**环境由官方 ``main()`` 推进**，
        本类只在每次回调里做一次 ``decide()``，并维护同一套
        ``EpisodeState``（history / memory / 重置策略），使两条路径的
        提示词与证据结构完全一致 —— 否则本地验证过的提示词到了提交环境会失效。
        """

        MAX_ACTIONS = 400
        reasoning_tag = "c-local"

        def __init__(
            self,
            *args: Any,
            harness: Optional[Harness] = None,
            llm: Optional[LLMPolicy] = None,
            journal: Optional[RunJournal] = None,
            sandbox: Optional[Sandbox] = None,
            reset_policy: Optional[str] = None,
            max_objects: int = 0,
            max_adjacency: int = 0,
            history_tail: int = 12,
            fallback_action: str = "ACTION1",
            **kwargs: Any,
        ) -> None:
            super().__init__(*args, **kwargs)
            self.harness = harness or Harness(
                llm if llm is not None else FakeLLM(),
                sandbox=sandbox,
                max_objects=max_objects,
                max_adjacency=max_adjacency,
                history_tail=history_tail,
                journal=journal,
                reset_policy=reset_policy,
            )
            # 注意：官方 ``Agent.state`` 是个 property（读 frames[-1].state），
            # 不能占用这个名字，所以跨步状态叫 ep_state。
            self.ep_state = EpisodeState(reset_policy=self.harness.reset_policy)
            self.decisions: List[Decision] = []
            self.fallback_action = fallback_action
            self._prev_graph = None
            self.last_grid: Optional[List[List[int]]] = None

        # -- 官方接口 1 ---------------------------------------------------- #

        def is_done(self, frames: Any, latest_frame: Any) -> bool:
            """只看引擎给的终局状态；MAX_ACTIONS 由基类循环兜住。"""
            state = str(getattr(getattr(latest_frame, "state", None), "name", ""))
            return state in ("WIN", "GAME_OVER")

        # -- 官方接口 2 ---------------------------------------------------- #

        def choose_action(self, frames: Any, latest_frame: Any) -> Any:
            grid = frame_to_grid(latest_frame)
            self.last_grid = grid
            turn = int(getattr(self, "action_counter", 0) or 0)

            dec = self.harness.decide(
                grid, turn=turn, history=self.ep_state.history, memory=self.ep_state.memory
            )
            self.decisions.append(dec)

            if not dec.ok or dec.action is None:
                err = dec.result.error if dec.result is not None else "no decision"
                self.ep_state.note_failure(turn, err, dec.graph.n_objects)
                return self._fallback(latest_frame)

            # 比较"上一次决策时的对象图"与"现在这一帧的对象图" ——
            # 中间夹着的正是上一步动作，所以这就是它的因果（官方路径没有
            # 我们自己的 env.step 返回值可用）。
            prev = self._prev_graph
            moved = changed = bool(False)
            if prev is not None:
                moved = position_signature(dec.graph) != position_signature(prev)
                changed = shape_signature(dec.graph) != shape_signature(prev)
            self.ep_state.note_success(
                turn,
                str(dec.action),
                moved,
                changed,
                prev.n_objects if prev is not None else dec.graph.n_objects,
                dec.graph.n_objects,
            )
            self._prev_graph = dec.graph

            return resolve_game_action(dec.action, available=getattr(latest_frame, "available_actions", None))

        # -- 证据 ---------------------------------------------------------- #

        @property
        def records(self) -> List[Any]:
            return [d.record for d in self.decisions if d.record is not None]

        def _fallback(self, latest_frame: Any) -> Any:
            GameAction = load_game_action()
            allowed = canonical_action_names(getattr(latest_frame, "available_actions", None))
            name = next((c for c in _FALLBACK_ORDER if c in allowed), self.fallback_action)
            return GameAction[name]

    _HARNESS_AGENT_CLS = HarnessAgent
    return HarnessAgent


def make_harness_agent(llm: Optional[LLMPolicy] = None, harness: Optional[Harness] = None, **kwargs: Any):
    """构造 ``HarnessAgent`` 实例（参数直接透传）。"""
    return harness_agent_class()(harness=harness, llm=llm, **kwargs)


def __getattr__(name: str) -> Any:
    """PEP 562：``arc_env.HarnessAgent`` 按需构造，避免 import 期拉官方包。"""
    if name == "HarnessAgent":
        return harness_agent_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
