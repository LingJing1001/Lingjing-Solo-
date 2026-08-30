"""debug7.py — 把 FakeLLM 源码放进受限命名空间 exec，看 right 行为"""
import ast

src = (
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

# 复刻真实 _compile 命名空间
ns = {"__builtins__": {}}
ns.update({
    "dict": dict, "list": list, "tuple": tuple, "set": set,
    "len": len, "isinstance": isinstance,
    "range": range, "enumerate": enumerate, "zip": zip,
    "min": min, "max": max, "sum": sum, "abs": abs, "all": all, "any": any,
    "True": True, "False": False, "None": None,
})
ns.update({"UP": "up", "DOWN": "down", "LEFT": "left", "RIGHT": "right"})

# 先做 AST 检查（复刻 _safe_compile 逻辑）
tree = ast.parse(src)
allowed = {
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
}
op_types = (ast.operator, ast.boolop, ast.cmpop, ast.unaryop)
forbidden = {"os", "sys", "open", "exec", "eval", "getattr", "__import__"}
for node in ast.walk(tree):
    if not isinstance(node, allowed):
        if isinstance(node, op_types):
            continue
        print(f"  拒绝节点: {type(node).__name__}")
        break
    if isinstance(node, ast.Call):
        f = node.func
        if not isinstance(f, ast.Name):
            print(f"  拒绝调用: 非Name {type(f).__name__}")
            break
        if f.id not in {"len","range","enumerate","zip","abs","min","max","sum",
                       "all","any","isinstance","list","dict","set","tuple",
                       "int","float","str","bool"}:
            print(f"  拒绝调用名: {f.id}")
            break
    if isinstance(node, ast.Name):
        if node.id in forbidden:
            print(f"  拒绝名字: {node.id}")
            break
else:
    print("AST 检查通过")

exec(compile(src, "<d>", "exec"), ns)
step = ns["step"]

state = {
    "objects": {
        "a": {"x": 1, "y": 1, "role": "avatar"},
        "b": {"x": 2, "y": 1, "role": "box"},
    },
    "avatar_id": "a",
    "extras": {},
}
print("right →", step(dict(state), "right")['objects'])
