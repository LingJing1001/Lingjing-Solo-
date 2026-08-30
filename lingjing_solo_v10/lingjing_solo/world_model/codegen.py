"""
codegen.py — LLM 动力学代码生成（v0.8 核心，突破表达力瓶颈）

v0.7 的 Layer B（规则→代码）只能处理「单对象独立响应」。
v0.8：Layer A 用 LLM 把「关系规则集」翻译成可执行的 step(state, action) 函数。

设计：
    - 输入：当前关系规则集（已晋升的）+ 最近 N 条转移证据（few-shot 示例）
    - LLM 输出：一段 Python（含 step(state, action) -> new_state）
    - 安全：AST 白名单 + 命名白名单（允许字面量构造 dict/list/tuple、内置 len/isinstance）
    - 降级：LLM 不可用 / 编译失败 → 回退到 Layer B（关系规则解释执行）

安全模型说明：
    我们允许的安全调用：dict() / list() / tuple() / set() 以及 len / isinstance 等。
    我们禁止的：属性调用（o.method）、任意名字的调用、import、全局变量读写。
    dict(...) 这类「同名内置构造调用」通过 _is_safe_call 放行。
"""
import ast
import re
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Any, Tuple

from .relations import RelationalRule, RelationGraph, Relation
from .induction import RelationalInducer


# ---------- AST 白名单 ----------

_ALLOWED_NODES = (
    ast.Module, ast.FunctionDef, ast.Return, ast.Assign, ast.AugAssign,
    ast.For, ast.If, ast.Compare, ast.BoolOp, ast.BinOp, ast.UnaryOp,
    ast.Name, ast.Constant, ast.Subscript, ast.Slice, ast.List, ast.Tuple,
    ast.Dict, ast.Set, ast.Expr, ast.Pass, ast.Break, ast.Continue,
    ast.While, ast.IfExp, ast.Assert,
    ast.arguments, ast.arg, ast.keyword, ast.Starred,
    ast.Store, ast.Load, ast.Del, ast.Index,
    ast.DictComp, ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.comprehension,
    ast.Try, ast.ExceptHandler, ast.Raise, ast.With, ast.withitem,
    ast.Lambda, ast.AnnAssign,
    ast.Call, ast.Attribute, ast.Import, ast.ImportFrom,  # 显式在遍历中单独处理
    ast.Is, ast.IsNot,  # BoolOp(or_/and_) 的比较操作符，visit 时会单独走到
    ast.And, ast.Or,
    ast.In, ast.NotIn,  # Compare 操作符（"action not in (...)"）
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,  # 比较操作符
)


# 可在「加载」位置安全出现的名字（builtin + 方向常量）
_SAFE_LOAD_NAMES = {
    "True", "False", "None",
    "len", "range", "enumerate", "zip", "abs", "min", "max",
    "sum", "all", "any", "sorted", "reversed",
    "int", "float", "str", "bool",
    "list", "dict", "set", "tuple",
    "isinstance", "ValueError", "Exception",
    "UP", "DOWN", "LEFT", "RIGHT", "GRID_W", "GRID_H",
}


# 可被「调用」的安全函数名（含容器构造）
_SAFE_CALL_NAMES = {
    "len", "range", "enumerate", "zip", "abs", "min", "max", "sum",
    "all", "any", "sorted", "reversed", "isinstance",
    "list", "dict", "set", "tuple", "int", "float", "str", "bool",
}


# 禁止以任何形式出现的标识符（危险模块/函数）
_FORBIDDEN_NAMES = {
    "os", "sys", "subprocess", "shutil", "pathlib", "socket", "threading",
    "multiprocessing", "ctypes", "cffi", "json", "pickle", "marshal",
    "__import__", "globals", "locals", "vars", "compile",
    "open", "exec", "eval", "getattr", "setattr", "delattr",
    "input", "exit", "quit", "breakpoint", "system", "popen",
}


class UnsafeCodeError(ValueError):
    pass


def _safe_compile(source: str) -> str:
    """
    安全编译：AST 白名单 + 调用白名单 + 命名黑名单。
    允许的安全调用：dict() / list() / tuple() / set()（字面量与调用均可）、
                   len / isinstance 等白名单内置。
    禁止：属性调用（o.method）、未知函数调用、import、危险名字。
    """
    tree = ast.parse(source)
    _op_types = (
        ast.operator, ast.boolop, ast.cmpop, ast.unaryop,
    )
    for node in ast.walk(tree):
        # 1. 节点类型白名单（Call/Attribute/Import 也在其中，但下面单独判死）
        #    兜底：操作符类型标记（Add/Sub/Mult/And/Or/In/Eq/...）只是枚举值，
        #    不含可执行逻辑，且由 BinOp/BoolOp/Compare 等已白名单的父节点承载，
        #    一律放行——避免逐个枚举 operator 子类的打地鼠问题。
        if not isinstance(node, _ALLOWED_NODES):
            if isinstance(node, _op_types):
                continue
            raise UnsafeCodeError(f"disallowed AST node: {type(node).__name__}")

        # 2. 显式禁止 import
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise UnsafeCodeError("import not allowed")

        # 3. 属性访问：仅放行白名单「安全属性读取」
        #    （当前白名单为空 → 任何 .xxx 都被拒，含 .get / .keys）
        if isinstance(node, ast.Attribute):
            full = _attribute_chain(node)
            if full not in _SAFE_ATTRIBUTES:
                raise UnsafeCodeError(f"disallowed attribute access: {full}")

        # 4. 调用：仅允许白名单函数名（禁止 o.method() 与未知函数）
        if isinstance(node, ast.Call):
            func = node.func
            if not isinstance(func, ast.Name):
                raise UnsafeCodeError("only direct builtin calls allowed")
            if func.id not in _SAFE_CALL_NAMES:
                raise UnsafeCodeError("disallowed call: " + func.id)

        # 5. 名称：黑名单优先。
        #    设计说明（v0.9）：变量「引用」本身没有副作用，真正危险的是
        #    「调用未知函数」和「属性访问」，这两者在下面的 Call / Attribute
        #    分支里已经单独拦截。因此这里只做黑名单检查，不再要求加载名
        #    属于白名单——否则合法的局部变量（tx/ty/o/oid/_dirs 等）会被
        #    误杀，导致 Layer B 降级代码与 FakeLLM 代码永远编译失败。
        if isinstance(node, ast.Name):
            if node.id in _FORBIDDEN_NAMES:
                raise UnsafeCodeError("disallowed name: " + node.id)
            # 允许一切非黑名单名字（含普通变量与下划线局部变量）。
            # 危险行为由 Call / Attribute 分支兜底拦截。
    return source


# 允许的属性访问白名单（链形式，如 "dict.get"）。当前为空，保持严格。
# 若未来需要 dict.get / list.append，在此添加即可。
_SAFE_ATTRIBUTES: set = set()


def _attribute_chain(node: ast.Attribute) -> str:
    """把 a.b.c 形式的属性访问还原成 'a.b.c'（仅用于白名单比对）。"""
    parts = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    return ".".join(reversed(parts))


# ---------- 状态表示契约 ----------

@dataclass
class GameState:
    """模拟器的状态契约（LLM 生成代码的操作对象）。"""
    objects: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    avatar_id: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {"objects": dict(self.objects), "avatar_id": self.avatar_id, "extras": dict(self.extras)}

    @classmethod
    def from_dict(cls, d: Dict) -> "GameState":
        return cls(objects=d.get("objects", {}), avatar_id=d.get("avatar_id"),
                   extras=d.get("extras", {}))


# ---------- LLM 提示词构造 ----------

_TEMPLATE = """# ARC-AGI-3 动力学代码生成
# 任务：写一个 Python 函数 step(state, action) -> new_state
#   - state: dict, 含 objects={{id:{{"x","y","role",...}}}}, avatar_id, extras
#   - action: str, 取值 "up"/"down"/"left"/"right" 或 "click:x,y"
# 约束：
#   - 只修改副本，不修改原 state（用 dict(state) / dict(obj) 拷贝）
#   - 用 objects 的 role 判断：avatar / box / wall / goal / target
#   - 允许使用的函数仅限：dict/list/tuple/len/isinstance 及算术比较
#   - 禁止 import、属性调用、全局变量
#   - 返回同结构的 dict
# 已归纳的关系规则（供参考，可增删）：
{rules_text}
# 真实转移观测（few-shot 示例）：
{traces_text}
def step(state, action):
"""


def _format_rules(rules: List[RelationalRule]) -> str:
    if not rules:
        return "    # （暂无已晋升规则）"
    return "\n".join(f"    # {r.rule_id}: {r.effect}" for r in rules)


def _format_traces(traces: List[Dict]) -> str:
    if not traces:
        return "    # （暂无观测示例）"
    lines = []
    for t in traces[:6]:
        action = t.get("action", "?") if isinstance(t, dict) else str(t)
        delta = t.get("delta", "") if isinstance(t, dict) else ""
        lines.append(f"    # {action} -> {delta}")
    return "\n".join(lines)


# ---------- 代码生成器 ----------

def _deepcopy_state(state: Dict) -> Dict:
    """
    深拷贝模拟器状态（v1.0 根因修复）。

    背景：生成的 step() 内部通常只做「外层浅拷贝」
        s = dict(state); s['objects'] = dict(state['objects'])
    而每个 obj（{'x':..,'y':..,'role':..}）仍是同一引用。
    一旦 step 里对 obj 做 o['x'] = ... 原地修改，就会污染调用方的 state，
    导致 BFS 缓存/搜索树状态互相污染（表现为方向依赖的隐蔽错位）。

    由于 LLM 生成的代码不可控，最稳妥的做法是在**唯一入口**处保证：
    传给 step_fn 的 state 与调用方的 state 完全独立。

    这里手工拷贝而非 copy.deepcopy，是为了：
      (a) 可控、可审计（安全边界内）；
      (b) 避免引入 copy 模块的运行时依赖风险。
    """
    if not isinstance(state, dict):
        return state
    out: Dict = {}
    for k, v in state.items():
        if k == "objects" and isinstance(v, dict):
            # 每个 obj dict 独立拷贝
            out[k] = {oid: dict(obj) for oid, obj in v.items()}
        elif k == "extras" and isinstance(v, dict):
            out[k] = dict(v)
        else:
            out[k] = v
    return out


@dataclass
class DynamicsProgram:
    """一个已编译的可执行动力学程序。"""
    source: str
    step_fn: Callable = field(repr=False)
    provenance: str = "llm"  # "llm" | "rules" | "fallback"

    def simulate(self, state: Dict, action: str) -> Optional[Dict]:
        try:
            # v1.0 根因修复：深拷输入，隔离生成的 step() 可能的原地修改。
            return self.step_fn(_deepcopy_state(state), action)
        except Exception:
            return None


class CodeGenerator:
    """
    LLM 动力学代码生成器。
        prog = generator.generate(rules, traces)   # 有 LLM → Layer A
        # 失败 / 无 LLM → 自动降级 Layer B（规则解释执行）
    """

    def __init__(self, llm_client=None, rules_inducer: Optional[RelationalInducer] = None):
        self.llm_client = llm_client
        self.inducer = rules_inducer

    def generate(self, rules: List[RelationalRule], traces: List[Dict]) -> DynamicsProgram:
        if self.llm_client is not None:
            try:
                source = self._llm_generate(rules, traces)
                step_fn = self._compile(source)
                return DynamicsProgram(source=source, step_fn=step_fn, provenance="llm")
            except Exception:
                pass  # 降级 Layer B
        return self._fallback(rules)

    # ---------- Layer A：LLM ----------

    def _llm_generate(self, rules: List[RelationalRule], traces: List[Dict]) -> str:
        prompt = _TEMPLATE.format(
            rules_text=_format_rules(rules),
            traces_text=_format_traces(traces),
        )
        raw = self.llm_client.generate(prompt)
        match = re.search(r"def step\s*\(.*?\):.*", raw, re.DOTALL)
        if not match:
            raise UnsafeCodeError("LLM output has no step() function")
        return match.group(0)

    # 运行时真正可用的安全 builtin（编译白名单 ∩ 运行时沙箱）。
    # 只放「纯函数/容器构造」，绝不放 open/exec/eval/__import__/os 等。
    _RUNTIME_SAFE = {
        "dict": dict, "list": list, "tuple": tuple, "set": set,
        "len": len, "isinstance": isinstance,
        "range": range, "enumerate": enumerate, "zip": zip,
        "min": min, "max": max, "sum": sum, "abs": abs,
        "all": all, "any": any,
        "True": True, "False": False, "None": None,
    }

    def _compile(self, source: str) -> Callable:
        _safe_compile(source)
        # 运行时命名空间：白名单 builtin + 方向常量。__builtins__ 保持空，
        # 防止代码通过 builtins 反射拿到 open/exec 等危险函数。
        namespace: Dict[str, Any] = {"__builtins__": {}}
        namespace.update(self._RUNTIME_SAFE)
        namespace.update({"UP": "up", "DOWN": "down", "LEFT": "left", "RIGHT": "right"})
        exec(compile(source, "<dynamics>", "exec"), namespace)
        if "step" not in namespace:
            raise UnsafeCodeError("step function not defined")
        return namespace["step"]

    # ---------- Layer B：规则解释执行（降级） ----------

    def _fallback(self, rules: List[RelationalRule]) -> DynamicsProgram:
        """
        无 LLM 时：把关系规则翻译成朴素 step（push + move）。
        约束：零属性调用（不用 .get/.keys/.items），仅下标 + 白名单内置
              dict/list/tuple + len/isinstance，满足 _safe_compile 白名单。
        """
        source = (
            "def step(state, action):\n"
            "    s = dict(state)\n"
            "    s['objects'] = dict(state['objects'])\n"
            "    s['extras'] = dict(state['extras'])\n"
            "    objs = s['objects']\n"
            "    aid = s['avatar_id']\n"
            "    if aid is None:\n"
            "        return s\n"
            "    if action not in ('up', 'down', 'left', 'right'):\n"
            "        return s\n"
            "    _dirs = {'up': (0, -1), 'down': (0, 1), 'left': (-1, 0), 'right': (1, 0)}\n"
            "    dx, dy = _dirs[action]\n"
            "    av = objs[aid]\n"
            "    tx = av['x'] + dx\n"
            "    ty = av['y'] + dy\n"
            "    for oid in list(objs):\n"
            "        if oid == aid:\n"
            "            continue\n"
            "        o = objs[oid]\n"
            "        role = o['role']\n"
            "        if role == 'box' or role == 'pushable':\n"
            "            if o['x'] == tx and o['y'] == ty:\n"
            "                o['x'] = o['x'] + dx\n"
            "                o['y'] = o['y'] + dy\n"
            "                break\n"
            "    av['x'] = tx\n"
            "    av['y'] = ty\n"
            "    return s\n"
        )
        return DynamicsProgram(source=source, step_fn=self._compile(source), provenance="rules")


# ---------- 测试桩：假 LLM ----------

class FakeLLM:
    """测试用：返回一个固定的合法 step 函数（含 push 语义）。"""
    def generate(self, prompt: str) -> str:
        return (
            "def step(state, action):\n"
            "    s = dict(state)\n"
            "    s['objects'] = dict(state['objects'])\n"
            "    s['extras'] = dict(state['extras'])\n"
            "    objs = s['objects']\n"
            "    aid = s['avatar_id']\n"
            "    if aid is None:\n"
            "        return s\n"
            "    if action not in ('up', 'down', 'left', 'right'):\n"
            "        return s\n"
            "    _dirs = {'up': (0, -1), 'down': (0, 1), 'left': (-1, 0), 'right': (1, 0)}\n"
            "    dx, dy = _dirs[action]\n"
            "    av = objs[aid]\n"
            "    tx = av['x'] + dx\n"
            "    ty = av['y'] + dy\n"
            "    for oid in list(objs):\n"
            "        if oid == aid:\n"
            "            continue\n"
            "        o = objs[oid]\n"
            "        role = o['role']\n"
            "        if role == 'box' or role == 'pushable':\n"
            "            if o['x'] == tx and o['y'] == ty:\n"
            "                o['x'] = o['x'] + dx\n"
            "                o['y'] = o['y'] + dy\n"
            "                break\n"
            "    av['x'] = tx\n"
            "    av['y'] = ty\n"
            "    return s\n"
        )
