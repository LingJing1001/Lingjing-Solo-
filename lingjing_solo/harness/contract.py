"""沙盒执行契约与执行证据（Agent Task2 的四层表 + 可复现性五字段表）。

来源：``llm-algo-leetcode`` 智能体基础设施专题
``topic_discussion/agent/02_agent_sandbox_environment.md``。该页把"模型提出动作"
与"动作是否获准执行、执行后留下什么证据"分成四层：

===============  ================================  ==============================
层                负责什么                          要保存的证据
===============  ================================  ==============================
工具接口          schema、参数类型与返回格式         tool name / args / schema ver
执行隔离          运行命令、访问文件与网络           image / 依赖锁定 / 资源配额
资源控制          timeout、CPU/GPU、内存、并发       预算 / 起止时间 / 终止原因
状态与日志        记录可回放结果                    observation / exit code / diff
===============  ================================  ==============================

以及"哪些条件决定环境能否复现"五字段：镜像与依赖 / 权限边界 / 资源预算 /
重置策略 / 日志与回放。

本模块只做**契约与留痕**，不做真正的隔离（隔离仍是 :mod:`sandbox` 的白名单 +
超时）。设计目标是把这些字段钉成不可省略的结构，使"调度策略或任务质量的差异"
可以归因于候选行为而不是环境漂移。

纯 stdlib。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "EXIT_CODES",
    "TERMINATION_REASONS",
    "RESET_POLICIES",
    "ToolSpec",
    "SandboxSpec",
    "ExecutionRecord",
    "RunJournal",
    "state_diff",
    "memory_digest",
    "apply_reset_policy",
    "ACTION_TOOL",
    "LOG_TOOL",
    "DEFAULT_TOOLS",
    "DEFAULT_SPEC",
]

# --------------------------------------------------------------------------- #
# exit_code / 终止原因 约定
# --------------------------------------------------------------------------- #
# 刻意贴近 shell 惯例：0 = 成功；124 = 超时（与 coreutils ``timeout`` 一致）；
# 126 = 不可执行（含"没产生动作"与"工具未授权"两种，用 termination_reason 细分）；
# 127 = 找不到命令（这里借来表示"代码本身不可接受"，如超过体积上限）。
# 这样上层调度器可以用同一套判据处理"代码跑失败"与"进程起不来"。
EXIT_CODES: Dict[str, int] = {
    "ok": 0,
    "runtime_error": 1,
    "syntax_error": 2,
    "timeout": 124,
    "step_limit": 125,
    "no_action": 126,
    "tool_not_allowed": 126,
    "code_too_long": 127,
}

TERMINATION_REASONS: Tuple[str, ...] = tuple(EXIT_CODES)

# memory 跨步行为的三种重置策略（教程"重置策略"字段）。
#   fresh      —— 每任务新环境：每步 memory 清空，不跨步污染
#   snapshot   —— 快照恢复：保留一份检查点，失败时回滚到检查点
#   persistent —— 持久状态：memory 原地累积（我们的默认，与 Duck 一致）
RESET_POLICIES: Tuple[str, ...] = ("fresh", "snapshot", "persistent")


# --------------------------------------------------------------------------- #
# 工具接口层
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ToolSpec:
    """一个可被模型调用的工具的接口契约。

    ``args`` 用 ``((参数名, 类型名), ...)`` 而非 dict，是为了让
    :meth:`signature` 在字段顺序上稳定（dict 顺序在旧解释器里不可靠）。
    """

    name: str
    args: Tuple[Tuple[str, str], ...] = ()
    returns: str = "None"
    schema_version: str = "1"

    def signature(self) -> str:
        """``action(name:str, row:int|None, col:int|None) -> Action`` 形式的一行式签名。"""
        inner = ", ".join(f"{n}:{t}" for n, t in self.args)
        return f"{self.name}({inner}) -> {self.returns}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "args": [[n, t] for n, t in self.args],
            "returns": self.returns,
            "schema_version": self.schema_version,
        }


ACTION_TOOL = ToolSpec(
    name="action",
    args=(("name", "str"), ("row", "int|None"), ("col", "int|None")),
    returns="Action",
)
LOG_TOOL = ToolSpec(
    name="log",
    args=(("msg", "Any"),),
    returns="None",
)
DEFAULT_TOOLS: Tuple[ToolSpec, ...] = (ACTION_TOOL, LOG_TOOL)


# --------------------------------------------------------------------------- #
# 执行隔离层 + 资源控制层
# --------------------------------------------------------------------------- #


@dataclass
class SandboxSpec:
    """一份可复现的沙盒合同。``fingerprint()`` 即该环境的"任务身份"。"""

    # 工具接口
    schema_version: str = "1"
    tools: Sequence[ToolSpec] = DEFAULT_TOOLS

    # 执行隔离
    image: str = "python:3.13-slim+sandbox-whitelist"
    deps_lock: Mapping[str, str] = field(default_factory=dict)
    network: bool = False
    filesystem: str = "none"  # none | readonly | workspace

    # 资源控制
    timeout_s: float = 30.0
    max_steps: int = 2_000_000
    max_code_chars: int = 40_000
    cpu_limit: Optional[float] = None
    mem_limit_mb: Optional[int] = None
    max_concurrency: int = 1

    # 重置策略与验收
    reset_policy: str = "persistent"
    success_criteria: Tuple[str, ...] = ("produced exactly one action()",)
    log_fields: Tuple[str, ...] = (
        "tool", "args", "schema_version", "observation", "exit_code",
        "termination_reason", "started_at", "ended_at", "duration_s",
        "steps", "state_diff", "memory_digest",
    )

    def __post_init__(self) -> None:
        if self.reset_policy not in RESET_POLICIES:
            raise ValueError(
                f"reset_policy must be one of {RESET_POLICIES}, got {self.reset_policy!r}"
            )
        if self.max_code_chars <= 0:
            raise ValueError("max_code_chars must be positive")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")

    # -- 身份 -------------------------------------------------------------- #

    def fingerprint(self) -> str:
        """环境身份哈希：只取**影响行为**的字段，不含 timeout 这类预算。

        两个 fingerprint 相同的沙盒，其"同一输入 → 同一环境语义"的假设才成立。
        """
        payload = {
            "schema_version": self.schema_version,
            "tools": [t.to_dict() for t in self.tools],
            "image": self.image,
            "deps_lock": dict(sorted(self.deps_lock.items())),
            "network": self.network,
            "filesystem": self.filesystem,
            "reset_policy": self.reset_policy,
            "success_criteria": list(self.success_criteria),
            "log_fields": list(self.log_fields),
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def budget_fingerprint(self) -> str:
        """资源预算指纹：预算变了但断言"环境变了"是错的，故与身份分开。"""
        payload = {
            "timeout_s": self.timeout_s,
            "max_steps": self.max_steps,
            "max_code_chars": self.max_code_chars,
            "cpu_limit": self.cpu_limit,
            "mem_limit_mb": self.mem_limit_mb,
            "max_concurrency": self.max_concurrency,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    # -- 可复现性自检 ------------------------------------------------------ #

    def gaps(self) -> List[str]:
        """对照教程"哪些条件决定环境能否复现"五字段，列出仍然悬空的项。

        返回空列表 = 该合同已足以支撑"差异归因于候选行为"的结论。
        """
        out: List[str] = []
        if not self.image:
            out.append("镜像与依赖: image 未声明 → 同一任务无法保证重新运行")
        if not self.deps_lock:
            out.append("镜像与依赖: deps_lock 为空 → 包版本漂移无法复现")
        if not self.tools:
            out.append("工具接口: tools 为空 → 允许调用什么工具未固定")
        elif not all(t.schema_version for t in self.tools):
            out.append("工具接口: 存在无 schema_version 的工具")
        if self.reset_policy not in RESET_POLICIES:
            out.append(f"重置策略: {self.reset_policy!r} 非法")
        if not self.success_criteria:
            out.append("成功条件: success_criteria 为空 → 无法判定一步是否成功")
        if not self.log_fields:
            out.append("日志与回放: log_fields 为空 → 失败步骤无法定位")
        missing = {"exit_code", "termination_reason", "state_diff"} - set(self.log_fields)
        if missing:
            out.append(f"日志与回放: log_fields 缺少 {sorted(missing)}")
        if self.network:
            out.append("权限边界: network=True → 工具获得了非必要权限")
        if self.filesystem == "workspace":
            out.append("权限边界: filesystem=workspace 为可写 → 上一步可能污染下一步")
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tools": [t.to_dict() for t in self.tools],
            "image": self.image,
            "deps_lock": dict(self.deps_lock),
            "network": self.network,
            "filesystem": self.filesystem,
            "timeout_s": self.timeout_s,
            "max_steps": self.max_steps,
            "max_code_chars": self.max_code_chars,
            "cpu_limit": self.cpu_limit,
            "mem_limit_mb": self.mem_limit_mb,
            "max_concurrency": self.max_concurrency,
            "reset_policy": self.reset_policy,
            "success_criteria": list(self.success_criteria),
            "log_fields": list(self.log_fields),
            "fingerprint": self.fingerprint(),
            "budget_fingerprint": self.budget_fingerprint(),
        }


DEFAULT_SPEC = SandboxSpec()


# --------------------------------------------------------------------------- #
# 状态与日志层：执行记录
# --------------------------------------------------------------------------- #


def state_diff(before: Mapping[str, Any], after: Mapping[str, Any]) -> Dict[str, List[str]]:
    """memory 的浅层差异。只记**键名**，不记值 —— 值是模型自己写的，
    进日志会泄漏上下文也给不了可比性；键名变化足以定位"这一步动了什么状态"。"""
    b, a = set(before), set(after)
    changed = sorted(k for k in (b & a) if before[k] != after[k])
    return {
        "added": sorted(a - b),
        "removed": sorted(b - a),
        "changed": changed,
    }


def memory_digest(memory: Mapping[str, Any]) -> str:
    """memory 的稳定摘要，用于跨机/跨次比对"状态是否一致"。"""
    try:
        blob = json.dumps(memory, sort_keys=True, ensure_ascii=False, default=repr)
    except Exception:  # noqa: BLE001 - 模型可能塞进不可序列化对象
        blob = repr(sorted((str(k), repr(v)) for k, v in memory.items()))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass
class ExecutionRecord:
    """一次工具执行留下的全部证据（教程四层表的"要保存的证据"列）。"""

    tool: str
    args: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1"
    observation: Optional[str] = None
    exit_code: int = 0
    termination_reason: str = "ok"
    started_at: float = 0.0
    ended_at: float = 0.0
    duration_s: float = 0.0
    steps: int = 0
    state_diff: Dict[str, List[str]] = field(default_factory=dict)
    memory_digest: str = ""
    error_kind: Optional[str] = None
    ok: bool = True
    frame_id: int = 0
    spec_fingerprint: str = ""

    # 允许模型自报一个动作字符串（action 工具的返回）。留在记录里但不进
    # prompt，避免把"上一轮的答案"喂回去。
    action: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool,
            "args": dict(self.args),
            "schema_version": self.schema_version,
            "observation": self.observation,
            "exit_code": self.exit_code,
            "termination_reason": self.termination_reason,
            "started_at": round(self.started_at, 6),
            "ended_at": round(self.ended_at, 6),
            "duration_s": round(self.duration_s, 6),
            "steps": self.steps,
            "state_diff": dict(self.state_diff),
            "memory_digest": self.memory_digest,
            "error_kind": self.error_kind,
            "ok": self.ok,
            "frame_id": self.frame_id,
            "spec_fingerprint": self.spec_fingerprint,
            "action": self.action,
        }

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)

    # -- 可比性：忽略时间 -------------------------------------------------- #

    def comparable(self) -> Dict[str, Any]:
        """剥掉时间与耗时后的"行为等价类"，用于断言可复现性。"""
        d = self.to_dict()
        for k in ("started_at", "ended_at", "duration_s"):
            d.pop(k, None)
        return d


# --------------------------------------------------------------------------- #
# 日志与回放层
# --------------------------------------------------------------------------- #


class RunJournal:
    """append-only 的执行日志。回放即"重新读一遍记录并比对"。"""

    def __init__(self, spec: Optional[SandboxSpec] = None) -> None:
        self.spec = spec or DEFAULT_SPEC
        self._records: List[ExecutionRecord] = []

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        return iter(self._records)

    @property
    def records(self) -> List[ExecutionRecord]:
        return list(self._records)

    def append(self, rec: ExecutionRecord) -> ExecutionRecord:
        self._records.append(rec)
        return rec

    def extend(self, recs: Sequence[ExecutionRecord]) -> None:
        self._records.extend(recs)

    # -- 统计 -------------------------------------------------------------- #

    def failure_reasons(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for r in self._records:
            if not r.ok:
                out[r.termination_reason] = out.get(r.termination_reason, 0) + 1
        return dict(sorted(out.items()))

    def success_rate(self) -> float:
        if not self._records:
            return 0.0
        return sum(1 for r in self._records if r.ok) / len(self._records)

    @property
    def total_steps(self) -> int:
        return sum(r.steps for r in self._records)

    def to_jsonl(self) -> str:
        return "\n".join(r.to_jsonl() for r in self._records)

    # -- 回放 -------------------------------------------------------------- #

    def replay(self) -> List[ExecutionRecord]:
        """回放 = 把日志再读一遍。返回**副本**，防止回放方污染原始证据。"""
        out: List[ExecutionRecord] = []
        for r in self._records:
            out.append(ExecutionRecord(**{**r.to_dict(), "args": dict(r.args),
                                          "state_diff": {k: list(v) for k, v in r.state_diff.items()}}))
        return out

    def diff_against(self, other: "RunJournal") -> Dict[str, Any]:
        """两次同契约运行的差异。教程的判据：只有当镜像/依赖/预算/重置/日志
        字段全部一致时，差异才可归因于候选行为 —— 所以先比指纹，再比行为。
        """
        same_env = self.spec.fingerprint() == other.spec.fingerprint()
        same_budget = self.spec.budget_fingerprint() == other.spec.budget_fingerprint()
        a, b = self.records, other.records
        n = min(len(a), len(b))
        mismatches = [
            i for i in range(n) if a[i].comparable() != b[i].comparable()
        ]
        return {
            "same_env": same_env,
            "same_budget": same_budget,
            "attributable_to_policy": bool(same_env),
            "len_a": len(a),
            "len_b": len(b),
            "first_mismatch": mismatches[0] if mismatches else None,
            "n_mismatch": len(mismatches),
        }


# --------------------------------------------------------------------------- #
# 重置策略
# --------------------------------------------------------------------------- #


def apply_reset_policy(
    memory: Dict[str, Any],
    policy: str,
    snapshot: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """按策略决定**下一步**看到的 memory。

    - ``fresh``：清空 —— 每步都是新环境
    - ``snapshot``：回到快照（无快照则清空）
    - ``persistent``：原样返回（原地累积）

    ``snapshot`` 下若 ``memory`` 已回到快照内容，视为"本步未越界"。
    """
    if policy == "persistent":
        return memory
    if policy == "fresh":
        memory.clear()
        return memory
    if policy == "snapshot":
        memory.clear()
        if snapshot:
            memory.update(snapshot)
        return memory
    raise ValueError(f"unknown reset_policy {policy!r}")


def take_snapshot(memory: Mapping[str, Any]) -> Dict[str, Any]:
    """浅拷贝快照。memory 的值约定为可 JSON 的原语，浅拷贝即足够。"""
    return dict(memory)
