"""debugB.py — 二分法：最简 push 代码在受限空间测试"""
import ast

_RUNTIME_SAFE = {
    "dict": dict, "list": list, "tuple": tuple, "set": set,
    "len": len, "isinstance": isinstance,
    "range": range, "enumerate": enumerate, "zip": zip,
    "min": min, "max": max, "sum": sum, "abs": abs, "all": all, "any": any,
    "True": True, "False": False, "None": None,
}

def safe_compile(src):
    """复刻 v0.9 的 _safe_compile（变量名仅黑名单）"""
    tree = ast.parse(src)
    op_types = (ast.operator, ast.boolop, ast.cmpop, ast.unaryop)
    allowed = (
        ast.Module, ast.FunctionDef, ast.Return, ast.Assign, ast.AugAssign,
        ast.For, ast.If, ast.Compare, ast.BoolOp, ast.BinOp, ast.UnaryOp,
        ast.Name, ast.Constant, ast.Subscript, ast.Slice, ast.List, ast.Tuple,
        ast.Dict, ast.Set, ast.Expr, ast.Pass, ast.Break, ast.Continue,
        ast.While, ast.IfExp, ast.Assert,
        ast.arguments, ast.arg, ast.keyword, ast.Starred,
        ast.Store, ast.Load, ast.Del, ast.Index,
        ast.DictComp, ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.comprehension,
        ast.Try, ast.ExceptHandler, ast.Raise, ast.With, ast.withitem,
        ast.Lambda, ast.AnnAssign, ast.Call, ast.Attribute, ast.Import, ast.ImportFrom,
        ast.And, ast.Or, ast.In, ast.NotIn, ast.Is, ast.IsNot,
        ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    )
    forbidden = {"os", "sys", "open", "exec", "eval", "getattr", "__import__",
                 "subprocess", "shutil", "pathlib", "socket"}
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            if isinstance(node, op_types):
                continue
            raise ValueError(f"node: {type(node).__name__}")
        if isinstance(node, ast.Call):
            f = node.func
            if not isinstance(f, ast.Name):
                raise ValueError(f"call not Name: {type(f).__name__}")
            if f.id not in {*_RUNTIME_SAFE, "int","float","str","bool"}:
                raise ValueError(f"call: {f.id}")
        if isinstance(node, ast.Name):
            if node.id in forbidden:
                raise ValueError(f"name: {node.id}")

# ---- 二分测试 1：完全无循环，只做正方向移动 ----
v1 = (
    "def step(state, action):\n"
    "    s = dict(state)\n"
    "    s['objects'] = dict(state['objects'])\n"
    "    objs = s['objects']\n"
    "    av = objs['a']\n"
    "    av['x'] = av['x'] + 1\n"   # 正方向 +1
    "    return s\n"
)
# ---- 二分测试 2：带 for 循环 + break ----
v2 = (
    "def step(state, action):\n"
    "    s = dict(state)\n"
    "    s['objects'] = dict(state['objects'])\n"
    "    objs = s['objects']\n"
    "    av = objs['a']\n"
    "    for oid in list(objs):\n"
    "        if oid == 'a':\n"
    "            continue\n"
    "        o = objs[oid]\n"
    "        if o['role'] == 'box':\n"
    "            o['x'] = o['x'] + 1\n"
    "            break\n"
    "    av['x'] = av['x'] + 1\n"   # 这行应在 break 后仍执行
    "    return s\n"
)

def run(src, tag):
    safe_compile(src)
    ns = {"__builtins__": {}}
    ns.update(_RUNTIME_SAFE)
    exec(compile(src, "<x>", "exec"), ns)
    step = ns["step"]
    state = {"objects": {"a": {"x": 1, "y": 1, "role": "avatar"},
                        "b": {"x": 2, "y": 1, "role": "box"}}, "avatar_id": "a"}
    print(f"{tag}: {step(dict(state), 'right')['objects']}")

run(v1, "v1(无循环)")
run(v2, "v2(有循环+break)")
