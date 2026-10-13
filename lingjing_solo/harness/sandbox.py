"""受限 Python REPL：模型写代码在这里跑。

纯 stdlib，三条约束：
1. **builtins 白名单** —— 不是黑名单。``open`` / ``eval`` / ``exec`` / ``compile`` /
   ``__import__``（原生）全部不在表里。
2. **import 白名单** —— 用一个自定义 ``__import__`` 包住，只放数学/集合/文本类模块。
   没有 ``os`` / ``sys`` / ``socket`` / ``subprocess`` / ``pathlib`` / ``pickle`` /
   ``ctypes`` / ``threading`` / ``multiprocessing``。
3. **超时 + 步数上限** —— 用 ``sys.settrace`` 在每条 line 事件检查墙钟与步数，
   超限抛一个 ``BaseException`` 子类（这样用户代码里的 ``except Exception`` 吞不掉它）。

⚠️ 这是**骨架级**隔离，不是安全边界。``type`` / ``object`` 仍在白名单里（模型常用），
因此 ``().__class__.__base__.__subclasses__()`` 这类逃逸在原理上可行。
真要上生产（跑不可信代码），必须换成子进程 + 容器/seccomp/nsjail，
本模块只负责把**接口契约**钉死，让上层逻辑可以先在纯 CPU 上跑通。

**执行证据**（教程 Task2 的四层表）在 :mod:`contract` 里定义：
``run_record()`` 产出带 ``exit_code`` / ``termination_reason`` /
``started_at`` / ``ended_at`` / ``state_diff`` 的 :class:`ExecutionRecord`，
``run()`` 保持旧的 :class:`SandboxResult` 返回以兼容既有调用方。两者共用
同一次执行，不重复跑代码。
"""

from __future__ import annotations

import builtins as _bi
import sys
import time
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

from .actions import Action, make_action
from .contract import (
    DEFAULT_SPEC,
    EXIT_CODES,
    ExecutionRecord,
    SandboxSpec,
    memory_digest,
    state_diff,
)

__all__ = [
    "ALLOWED_IMPORTS",
    "SANDBOX_BUILTINS",
    "SandboxContext",
    "SandboxResult",
    "Sandbox",
]

ALLOWED_IMPORTS = frozenset(
    {
        "math",
        "cmath",
        "collections",
        "itertools",
        "functools",
        "heapq",
        "bisect",
        "statistics",
        "re",
        "json",
        "copy",
        "operator",
        "string",
        "random",
        "typing",
        "dataclasses",
        "enum",
        "fractions",
        "decimal",
        "array",
    }
)

# builtins 白名单。刻意不含 open/eval/exec/compile/input/globals/locals/vars/
# dir/getattr/setattr/delattr/breakpoint/help/exit/quit/memoryview/super。
_BUILTIN_NAMES = (
    "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes", "callable",
    "chr", "dict", "divmod", "enumerate", "filter", "float", "format", "frozenset",
    "hash", "hex", "int", "isinstance", "issubclass", "iter", "len", "list", "map",
    "max", "min", "next", "object", "oct", "ord", "pow", "print", "range", "repr",
    "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple", "type", "zip",
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
    "ZeroDivisionError", "ArithmeticError", "LookupError", "OverflowError",
    "RuntimeError", "StopIteration", "NotImplementedError", "AssertionError",
    "AttributeError", "NameError", "FloatingPointError",
    "True", "False", "None", "Ellipsis", "NotImplemented",
)

SANDBOX_BUILTINS: Dict[str, Any] = {
    name: getattr(_bi, name) for name in _BUILTIN_NAMES if hasattr(_bi, name)
}


def _install_import_guard() -> Any:
    """自定义 ``__import__``：只放行白名单模块，且**限制顶级包**。"""

    real_import = _bi.__import__

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        if level != 0:
            raise ImportError("relative imports are not allowed in the sandbox")
        top = name.split(".")[0]
        if top not in ALLOWED_IMPORTS:
            raise ImportError(f"import of {name!r} is not allowed in the sandbox")
        return real_import(name, globals, locals, fromlist, level)

    return _guarded_import


class _Stop(BaseException):
    """沙箱中断信号。继承 ``BaseException`` 以免被用户代码 ``except Exception`` 吞掉。"""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class SandboxContext:
    """一次执行的全部输入。``memory`` 是**同一个 dict 对象**，原地修改即持久化。"""

    objects: List[Dict[str, Any]] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)
    memory: Dict[str, Any] = field(default_factory=dict)
    frame_id: int = 0


@dataclass
class SandboxResult:
    ok: bool
    action: Optional[Action] = None
    logs: List[str] = field(default_factory=list)
    memory: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    error_kind: Optional[str] = None  # "syntax" | "runtime" | "timeout" | "step_limit" | "no_action"
    duration_s: float = 0.0
    steps: int = 0

    @property
    def produced_action(self) -> bool:
        return self.action is not None


# error_kind（内部） → termination_reason（对外证据）。刻意不合并：前者是
# "怎么失败的"，后者是"这次执行以什么名义结束"，调度器只看后者。
_KIND_TO_REASON: Dict[str, str] = {
    "syntax": "syntax_error",
    "runtime": "runtime_error",
    "timeout": "timeout",
    "step_limit": "step_limit",
    "no_action": "no_action",
    "code_too_long": "code_too_long",
    "tool_not_allowed": "tool_not_allowed",
}


class Sandbox:
    """一次一调（stateless）的代码执行器。

    两个返回口：
    - :meth:`run` → :class:`SandboxResult`（旧接口，只有结论）
    - :meth:`run_record` → :class:`ExecutionRecord`（教程 Task2 要求的完整证据）

    两者共用 :meth:`_run_core`，**代码只跑一次**。
    """

    def __init__(
        self,
        timeout_s: float = 30.0,
        max_steps: int = 2_000_000,
        max_code_chars: int = 40_000,
        spec: Optional[SandboxSpec] = None,
    ) -> None:
        # 传了 spec 就以 spec 为准（显式合同覆盖零散预算参数），
        # 这是刻意的：预算必须来自一份可哈希的合同，而不是散落的 kwargs。
        if spec is None:
            spec = replace(
                DEFAULT_SPEC,
                timeout_s=timeout_s,
                max_steps=max_steps,
                max_code_chars=max_code_chars,
            )
        self.spec = spec
        self.timeout_s = spec.timeout_s
        self.max_steps = spec.max_steps
        self.max_code_chars = spec.max_code_chars

    @property
    def tool_names(self) -> Tuple[str, ...]:
        return tuple(t.name for t in self.spec.tools)

    # -- 内部 -------------------------------------------------------------- #

    def _make_tracer(self, deadline: float, counter: List[int]):
        def _local(frame, event, arg):
            if event == "line":
                counter[0] += 1
                if counter[0] > self.max_steps:
                    raise _Stop("step_limit")
                if time.monotonic() >= deadline:
                    raise _Stop("timeout")
            return _local

        def _global(frame, event, arg):
            return _local

        return _global

    # -- 公开接口 ---------------------------------------------------------- #

    def run(self, code: str, ctx: Optional[SandboxContext] = None) -> SandboxResult:
        """执行代码，只返回结论（旧接口）。"""
        return self.execute(code, ctx)[0]

    def run_record(
        self,
        code: str,
        ctx: Optional[SandboxContext] = None,
        spec: Optional[SandboxSpec] = None,
    ) -> ExecutionRecord:
        """执行代码，返回**完整执行证据**（教程 Task2 四层表）。"""
        return self.execute(code, ctx, spec=spec)[1]

    # -- 执行核心（run / run_record / harness 共用，代码只跑一次）----------- #

    def execute(
        self,
        code: str,
        ctx: Optional[SandboxContext] = None,
        spec: Optional[SandboxSpec] = None,
    ) -> Tuple[SandboxResult, ExecutionRecord]:
        spec = spec or self.spec
        ctx = ctx or SandboxContext()
        t0 = time.monotonic()
        mem_before = dict(ctx.memory)
        code = "" if code is None else code

        def _record(res: SandboxResult, kind: Optional[str], err: Optional[str]):
            reason = "ok" if res.ok else _KIND_TO_REASON.get(kind or "", "runtime_error")
            if reason == "ok":
                observation = f"action={res.action}"
                exit_code = EXIT_CODES["ok"]
            else:
                observation = f"{reason}: {err}" if err else reason
                exit_code = EXIT_CODES.get(reason, EXIT_CODES["runtime_error"])
            rec = ExecutionRecord(
                tool="python_exec",
                args={"code_chars": len(code)},
                schema_version=spec.schema_version,
                observation=observation,
                exit_code=exit_code,
                termination_reason=reason,
                started_at=t0,
                ended_at=time.monotonic(),
                duration_s=res.duration_s,
                steps=res.steps,
                state_diff=state_diff(mem_before, ctx.memory),
                memory_digest=memory_digest(ctx.memory),
                error_kind=kind,
                ok=res.ok,
                frame_id=ctx.frame_id,
                spec_fingerprint=spec.fingerprint(),
                action=(str(res.action) if res.action else None),
            )
            return res, rec

        # 资源预算：代码体积（超限直接判死，不编译）
        if len(code) > spec.max_code_chars:
            return _record(
                SandboxResult(
                    ok=False,
                    error=f"code too long ({len(code)} > {spec.max_code_chars})",
                    error_kind="runtime",
                    duration_s=time.monotonic() - t0,
                ),
                "code_too_long",
                f"code too long ({len(code)} > {spec.max_code_chars})",
            )

        # 工具契约：spec 未授权 action 工具时，本步按"工具未授权"归档
        if "action" not in {t.name for t in spec.tools}:
            msg = "tool 'action' is not allowed by the sandbox spec"
            return _record(
                SandboxResult(ok=False, error=msg, error_kind="runtime",
                              duration_s=time.monotonic() - t0),
                "tool_not_allowed",
                msg,
            )

        logs: List[str] = []
        chosen: List[Action] = []

        def _action(name: Any, row: Any = None, col: Any = None) -> Action:
            a = make_action(name, row, col)
            chosen.append(a)  # 取最后一次调用
            return a

        def _log(msg: Any) -> None:
            logs.append(str(msg))

        g: Dict[str, Any] = {
            "__builtins__": dict(SANDBOX_BUILTINS, __import__=_install_import_guard()),
            "__name__": "__sandbox__",
            "objects": ctx.objects,
            "history": ctx.history,
            "memory": ctx.memory,
            "frame_id": ctx.frame_id,
            "action": _action,
            "log": _log,
        }

        try:
            compiled = compile(code, "<repl>", "exec")
        except SyntaxError as e:
            msg = f"{type(e).__name__}: {e}"
            return _record(
                SandboxResult(ok=False, logs=logs, memory=ctx.memory, error=msg,
                              error_kind="syntax", duration_s=time.monotonic() - t0),
                "syntax",
                msg,
            )

        counter = [0]
        deadline = time.monotonic() + spec.timeout_s
        tracer = self._make_tracer(deadline, counter)
        prev_trace = sys.gettrace()  # 可能被覆盖率工具占用：用完要还回去
        sys.settrace(tracer)
        try:
            exec(compiled, g)  # noqa: S102 - 这是设计意图（受限沙箱）
            err = None
            kind = None
        except _Stop as e:
            err = f"{e.reason} after {counter[0]} steps"
            kind = e.reason if e.reason in ("timeout", "step_limit") else "runtime"
        except BaseException as e:  # noqa: BLE001 - 沙箱要吞掉一切并归档
            err = f"{type(e).__name__}: {e}"
            kind = "runtime"
        finally:
            sys.settrace(prev_trace)

        dur = time.monotonic() - t0
        action = chosen[-1] if chosen else None

        if err is not None:
            return _record(
                SandboxResult(ok=False, action=action, logs=logs, memory=ctx.memory,
                              error=err, error_kind=kind, duration_s=dur, steps=counter[0]),
                kind, err,
            )
        if action is None:
            msg = "no action() call"
            return _record(
                SandboxResult(ok=False, action=None, logs=logs, memory=ctx.memory,
                              error=msg, error_kind="no_action",
                              duration_s=dur, steps=counter[0]),
                "no_action", msg,
            )
        return _record(
            SandboxResult(ok=True, action=action, logs=logs, memory=ctx.memory,
                          error=None, error_kind=None, duration_s=dur, steps=counter[0]),
            None, None,
        )
