"""回合循环：观察 → 拼提示词 → LLM → 解析代码 → 沙箱执行 → 落动作。

这是 C-Local 旁路的**第四块**，把前三块（对象图 / 提示词 / 沙箱）串成一个
可跑的闭环。刻意的设计点：

- **LLM 可插拔**：:class:`LLMPolicy` 只有一个方法 ``complete(prompt, ctx)``。
  真模型、本地 vLLM、假模型都是同一个接口 —— 所以整条链路**能在纯 CPU 上先跑通**，
  不需要 GPU 就能验证"提示词有没有把信息喂全、沙箱能不能把代码跑出动作"。
- **动作、状态变更、提示词开销全部留痕**：每步都记 ``moved`` / ``changed``
  与 ``prefix_tokens`` / ``suffix_tokens``，前者用于复盘因果，后者直接喂容量模型。
- **失败不致命**：语法错、超时、没调 ``action()`` 都记为一步，不抛异常。
  真实 LLM 会经常写错，harness 必须能扛住。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from .actions import Action
from .contract import (
    ExecutionRecord,
    RunJournal,
    SandboxSpec,
    apply_reset_policy,
    memory_digest,
    take_snapshot,
)
from .object_graph import Graph, build_graph, graph_signature
from .prompt import build_prompt, estimate_tokens
from .sandbox import Sandbox, SandboxContext, SandboxResult

__all__ = [
    "StepOutcome",
    "Env",
    "FakeEnv",
    "LLMContext",
    "LLMPolicy",
    "FakeLLM",
    "EpisodeState",
    "Decision",
    "Step",
    "EpisodeResult",
    "Harness",
    "extract_code",
    "position_signature",
    "shape_signature",
]


# --------------------------------------------------------------------------- #
# 环境
# --------------------------------------------------------------------------- #


@dataclass
class StepOutcome:
    frame: Any
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)


class Env(Protocol):
    """harness 对环境的全部要求。官方 Kaggle harness 的 ``choose_action`` 契约
    可以很薄地适配成这个 Protocol（见 ``kaggle_adapter``）。"""

    def reset(self) -> Any: ...
    def step(self, action: Action) -> StepOutcome: ...
    @property
    def shape(self) -> Tuple[int, int]: ...


class FakeEnv:
    """纯 CPU 玩具环境：网格上一颗"棋子"走向"目标"。

    刻意做得极简，但它有真实游戏的关键性质：**动作的因果可见**
    （移动会让棋子对象的 (r,c) 变），所以"对象图 → 代码 → 动作"这条链路
    能被端到端验证，而不只是单测各自的零件。
    """

    BG, AGENT, GOAL = 0, 5, 3

    def __init__(self, size: int = 8, agent=(1, 1), goal=(6, 6), max_turns: int = 60) -> None:
        self.n = size
        self._agent0 = tuple(agent)
        self._goal = tuple(goal)
        self.agent = self._agent0
        self.max_turns = max_turns
        self.turns = 0

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.n, self.n)

    def reset(self) -> List[List[int]]:
        self.agent = self._agent0
        self.turns = 0
        return self._frame()

    def _frame(self) -> List[List[int]]:
        g = [[self.BG] * self.n for _ in range(self.n)]
        gr, gc = self._goal
        g[gr][gc] = self.GOAL
        ar, ac = self.agent
        g[ar][ac] = self.AGENT
        return g

    def step(self, action: Action) -> StepOutcome:
        self.turns += 1
        ar, ac = self.agent
        if action.kind == "UP":
            ar = max(0, ar - 1)
        elif action.kind == "DOWN":
            ar = min(self.n - 1, ar + 1)
        elif action.kind == "LEFT":
            ac = max(0, ac - 1)
        elif action.kind == "RIGHT":
            ac = min(self.n - 1, ac + 1)
        # SPACE / CLICK 在玩具环境里不动棋子（用来验证"动作被吃下但不改状态"）
        self.agent = (ar, ac)
        done = self.agent == self._goal
        return StepOutcome(
            frame=self._frame(),
            reward=1.0 if done else 0.0,
            done=done or self.turns >= self.max_turns,
            info={"turns": self.turns},
        )


# --------------------------------------------------------------------------- #
# LLM
# --------------------------------------------------------------------------- #


@dataclass
class LLMContext:
    """给 policy 的旁路上下文。

    ⚠️ 真实部署里 policy 只应读 ``prompt``（否则等于偷看内部状态）。
    ``graph`` 是给 :class:`FakeLLM` 这种测试替身用的便捷入口。
    """

    graph: Graph
    history: List[Dict[str, Any]]
    memory: Dict[str, Any]
    turn: int = 0
    frame: Any = None


class LLMPolicy(Protocol):
    def complete(self, prompt: str, ctx: LLMContext) -> str: ...


class FakeLLM:
    """确定性"模型"：不看提示词，直接从对象图算出朝目标走一步的代码。

    它存在的意义是让整条链路可测 —— 提示词、沙箱、解析、循环都能在
    **无 GPU、无网络**的条件下验证。要换成真模型，只需换掉这个类。
    """

    def __init__(self, agent_color: str = "5", goal_color: str = "3", script: Optional[Sequence[str]] = None):
        self.agent_color = agent_color
        self.goal_color = goal_color
        self._script = list(script) if script is not None else None
        self.calls: List[str] = []

    def complete(self, prompt: str, ctx: LLMContext) -> str:
        self.calls.append(prompt)
        if self._script is not None:
            idx = min(len(self.calls) - 1, len(self._script) - 1)
            return self._script[idx]

        return (
            "```python\n"
            f"agent = [o for o in objects if o['color'] == '{self.agent_color}']\n"
            f"goal  = [o for o in objects if o['color'] == '{self.goal_color}']\n"
            "if not (agent and goal):\n"
            "    action('SPACE')\n"
            "else:\n"
            "    a, g = agent[0], goal[0]\n"
            "    dr, dc = g['r'] - a['r'], g['c'] - a['c']\n"
            "    if dr > 0:\n"
            "        action('DOWN')\n"
            "    elif dr < 0:\n"
            "        action('UP')\n"
            "    elif dc > 0:\n"
            "        action('RIGHT')\n"
            "    elif dc < 0:\n"
            "        action('LEFT')\n"
            "    else:\n"
            "        action('SPACE')\n"
            # 必须在 else 分支内：真帧上常常找不到目标色，(a, g) 就不存在。
            # 这一行原先顶格写在 if 之外，只在 FakeEnv 那颗"必有 5 和 3"的
            # 玩具网格上侥幸没暴露 —— 接官方真帧立刻 NameError。
            "    memory['last'] = (a['r'], a['c'])\n"
            "```"
        )


# --------------------------------------------------------------------------- #
# 代码块解析
# --------------------------------------------------------------------------- #

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> Optional[str]:
    """从模型回复里取第一个代码块。

    容错顺序：```python 块 → 任意 ``` 块 → 整段文本（若含 ``action(``）。
    取不到返回 ``None``（调用方记为一步失败）。
    """
    if not text:
        return None
    m = _FENCE.search(text)
    if m:
        code = m.group(1).strip("\n")
        if code.strip():
            return code
    if "action(" in text:
        return text.strip()
    return None


# --------------------------------------------------------------------------- #
# 一步 / 一局
# --------------------------------------------------------------------------- #


@dataclass
class Step:
    turn: int
    action: Optional[str]
    ok: bool
    moved: bool = False
    changed: bool = False
    error: Optional[str] = None
    error_kind: Optional[str] = None
    logs: List[str] = field(default_factory=list)
    n_objects: int = 0
    suffix_tokens: int = 0
    reward: float = 0.0
    # 执行证据（来自 ExecutionRecord）：调度器只看这两个字段即可判定重试
    exit_code: Optional[int] = None
    termination_reason: Optional[str] = None


@dataclass
class EpisodeResult:
    steps: List[Step] = field(default_factory=list)
    solved: bool = False
    turns: int = 0
    prefix_tokens: int = 0
    total_suffix_tokens: int = 0
    total_tokens: int = 0
    failed_steps: int = 0
    # 本局的执行证据（每步一条），与 harness.journal 共享同一批对象
    records: List[ExecutionRecord] = field(default_factory=list)

    @property
    def actions(self) -> List[str]:
        return [s.action for s in self.steps if s.action]

    @property
    def exit_codes(self) -> List[int]:
        return [s.exit_code for s in self.steps if s.exit_code is not None]


def _no_code_record(memory: Dict[str, Any], turn: int, spec: SandboxSpec) -> ExecutionRecord:
    """模型回复里没有代码块 —— 连执行都没发生，但按教程"日志与回放"要求
    仍要留一条证据（否则失败步会从统计里凭空消失）。"""
    return ExecutionRecord(
        tool="extract_code",
        args={"reply_has_code": False},
        schema_version=spec.schema_version,
        observation="no_action: no code block in model reply",
        exit_code=126,
        termination_reason="no_action",
        state_diff={"added": [], "removed": [], "changed": []},
        memory_digest=memory_digest(memory),
        error_kind="no_action",
        ok=False,
        frame_id=turn,
        spec_fingerprint=spec.fingerprint(),
    )


@dataclass
class EpisodeState:
    """跨步状态：``history`` + ``memory`` + 检查点 + 重置策略。

    抽出来是因为它有两个消费方，语义必须一致：
    - ``Harness.run_episode``（我们自己推进环境，离线跑）
    - ``arc_env.HarnessAgent``（**官方引擎**推进环境，每次只回调一帧）

    ``memory`` 是**同一个 dict 对象**贯穿整局（沙箱原地改它即持久化），
    所以 ``decide(memory=state.memory)`` 不需要回写。
    """

    history: List[Dict[str, Any]] = field(default_factory=list)
    memory: Dict[str, Any] = field(default_factory=dict)
    reset_policy: str = "persistent"
    snapshot: Optional[Dict[str, Any]] = None

    def note_failure(self, turn: int, error: Optional[str], n_objects: int) -> None:
        self.history.append(
            {
                "turn": turn,
                "action": None,
                "moved": False,
                "changed": False,
                "objects_before": n_objects,
                "objects_after": n_objects,
                "error": error,
            }
        )
        self._reset(False)

    def note_success(
        self,
        turn: int,
        action: str,
        moved: bool,
        changed: bool,
        n_before: int,
        n_after: int,
    ) -> None:
        self.history.append(
            {
                "turn": turn,
                "action": action,
                "moved": moved,
                "changed": changed,
                "objects_before": n_before,
                "objects_after": n_after,
            }
        )
        self._reset(True)

    def _reset(self, ok: bool) -> None:
        """教程的"重置策略"字段：决定**下一步**看到什么 memory。

        - fresh      ：每步清空（每步都是新环境）
        - snapshot   ：成功步记为检查点，失败步回滚到检查点
        - persistent ：原地累积（与 Duck 一致）
        """
        if self.reset_policy == "fresh":
            self.memory.clear()
        elif self.reset_policy == "snapshot":
            if ok:
                self.snapshot = take_snapshot(self.memory)
            else:
                apply_reset_policy(self.memory, "snapshot", self.snapshot)


@dataclass
class Decision:
    """一次"观察 → 动作"的完整产物。**不碰环境**。

    官方 Kaggle harness 的 ``choose_action(frames, latest_frame)`` 每次只给一帧，
    与"跑一整局"是两件事。这个类型就是把两者共用的那一半抽出来：
    ``Harness.decide()`` 负责它，``Harness.run_episode()`` 与
    ``arc_env.HarnessAgent`` 都调它。
    """

    graph: Graph
    code: Optional[str] = None
    action: Optional[Action] = None
    result: Optional[SandboxResult] = None
    record: Optional[ExecutionRecord] = None
    prefix_tokens: int = 0
    suffix_tokens: int = 0

    @property
    def ok(self) -> bool:
        return bool(self.result is not None and self.result.ok and self.action is not None)


def position_signature(graph: Graph) -> List[Tuple[str, int, int]]:
    """对象位置指纹（含平移不变 hash）。用于"这一步有没有动"。"""
    return sorted((o.hash, o.r, o.c) for o in graph.objects)


def shape_signature(graph: Graph) -> List[Tuple[str, str, int]]:
    """对象形状/颜色/像素数指纹。用于"这一步有没有改变结构"。"""
    return sorted((o.color, o.hash, o.n) for o in graph.objects)


# 旧名保留：外部（含 arc_env）走公开名，内部沿用短名以免噪音
_positions = position_signature
_shape_sig = shape_signature


class Harness:
    """把一个 :class:`Env` 和一条 :class:`LLMPolicy` 接起来跑。

    ``journal`` 传入后，每步的 :class:`ExecutionRecord` 会追加进去；
    这是"证据先于结论"的落点：先攒证据，再用 :meth:`RunJournal.diff_against`
    比较两次运行，而不是直接比总分。
    """

    def __init__(
        self,
        llm: LLMPolicy,
        sandbox: Optional[Sandbox] = None,
        max_objects: int = 0,
        max_adjacency: int = 0,
        history_tail: int = 12,
        journal: Optional[RunJournal] = None,
        reset_policy: Optional[str] = None,
    ) -> None:
        self.llm = llm
        self.sandbox = sandbox or Sandbox()
        self.max_objects = max_objects
        self.max_adjacency = max_adjacency
        self.history_tail = history_tail
        self.journal = journal
        # 默认跟随沙盒合同（合同是唯一事实来源）
        self.reset_policy = reset_policy or self.sandbox.spec.reset_policy

    # -- 单步决策（不碰环境）----------------------------------------------- #

    def decide(
        self,
        frame: Any,
        turn: int = 0,
        history: Optional[List[Dict[str, Any]]] = None,
        memory: Optional[Dict[str, Any]] = None,
        frame_id: Optional[int] = None,
    ) -> Decision:
        """观察一帧 → 出一动作。**不改环境、不改 history/memory 的语义**。

        ``memory`` 若是 None 则本次调用只是一次无状态试探（跨帧状态得由调用方持有）。
        """
        history = history if history is not None else []
        memory = memory if memory is not None else {}
        graph = build_graph(frame)

        parts = build_prompt(
            graph,
            history=history,
            memory=memory,
            turn=turn,
            max_objects=self.max_objects,
            max_adjacency=self.max_adjacency,
            history_tail=self.history_tail,
        )
        ctx = LLMContext(graph=graph, history=history, memory=memory, turn=turn, frame=frame)
        reply = self.llm.complete(parts.text, ctx)
        code = extract_code(reply)

        if code is None:
            result = SandboxResult(ok=False, error="no code block", error_kind="no_action")
            record = _no_code_record(memory, turn, self.sandbox.spec)
        else:
            result, record = self.sandbox.execute(
                code,
                SandboxContext(
                    objects=graph.as_api(),
                    history=history,
                    memory=memory,
                    frame_id=turn if frame_id is None else frame_id,
                ),
            )
        if self.journal is not None:
            self.journal.append(record)

        return Decision(
            graph=graph,
            code=code,
            action=result.action,
            result=result,
            record=record,
            prefix_tokens=estimate_tokens(parts.prefix),
            suffix_tokens=estimate_tokens(parts.suffix),
        )

    # -- 一整局 ------------------------------------------------------------ #

    def run_episode(self, env: Env, max_turns: int = 60, verbose: bool = False) -> EpisodeResult:
        frame = env.reset()
        state = EpisodeState(reset_policy=self.reset_policy)
        res = EpisodeResult()
        episode_records: List[ExecutionRecord] = []

        for turn in range(max_turns):
            dec = self.decide(frame, turn=turn, history=state.history, memory=state.memory)
            sb: SandboxResult = dec.result or SandboxResult(
                ok=False, error="no decision", error_kind="no_action"
            )
            rec: ExecutionRecord = dec.record or _no_code_record(
                state.memory, turn, self.sandbox.spec
            )
            episode_records.append(rec)

            step = Step(
                turn=turn,
                action=(str(dec.action) if dec.action else None),
                ok=sb.ok,
                error=sb.error,
                error_kind=sb.error_kind,
                logs=list(sb.logs),
                n_objects=dec.graph.n_objects,
                suffix_tokens=dec.suffix_tokens,
                exit_code=rec.exit_code,
                termination_reason=rec.termination_reason,
            )
            res.prefix_tokens = dec.prefix_tokens
            res.total_suffix_tokens += step.suffix_tokens

            if not dec.ok or dec.action is None:
                res.failed_steps += 1
                res.steps.append(step)
                state.note_failure(turn, sb.error, dec.graph.n_objects)
                if verbose:
                    print(f"[t={turn}] FAIL {sb.error_kind}: {sb.error}")
                # 失败一步也要推进环境吗？不 —— 失败不产生动作，环境不动。
                continue

            graph = dec.graph
            before_pos, before_sig = _positions(graph), _shape_sig(graph)
            out = env.step(dec.action)
            frame = out.frame
            new_graph = build_graph(frame)
            moved = _positions(new_graph) != before_pos
            changed = _shape_sig(new_graph) != before_sig

            step.moved, step.changed, step.reward = moved, changed, out.reward
            res.steps.append(step)
            state.note_success(
                turn, str(dec.action), moved, changed, graph.n_objects, new_graph.n_objects
            )
            if verbose:
                print(f"[t={turn}] {dec.action} moved={moved} changed={changed} r={out.reward}")

            graph = new_graph
            res.turns = turn + 1
            if out.done:
                res.solved = out.reward > 0
                break
        else:
            res.turns = max_turns

        res.records = episode_records
        res.total_tokens = res.prefix_tokens * max(1, len(res.steps)) + res.total_suffix_tokens
        return res


# --------------------------------------------------------------------------- #
# CLI 演示
# --------------------------------------------------------------------------- #


def _demo() -> int:
    env = FakeEnv()
    llm = FakeLLM()
    h = Harness(llm)
    res = h.run_episode(env, max_turns=60, verbose=True)
    print("")
    print(f"solved={res.solved}  turns={res.turns}  failed_steps={res.failed_steps}")
    print(f"prefix_tokens={res.prefix_tokens}  suffix_tokens(total)={res.total_suffix_tokens}")
    return 0 if res.solved else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_demo())
