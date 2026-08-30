"""debug8.py — 直接修改 codegen 让 step 打印调试"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lingjing_solo.world_model.codegen import FakeLLM, CodeGenerator, _safe_compile, _compile_from_source
from lingjing_solo.world_model.induction import RelationalInducer

# 取 FakeLLM 生成的源码，注入打印
src = FakeLLM().generate("x")
# 在第 2 行后插入调试打印
lines = src.split("\n")
# 找到 av['x'] = tx 之前，加入 print
instrumented = (
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
    "    print('DBG: av_was', av, 'tx', tx, 'objs_now', objs)\n"  # 调试
    "    av['x'] = tx\n"
    "    av['y'] = ty\n"
    "    print('DBG: after av assign, objs', objs)\n"
    "    return s\n"
)

_RUNTIME_SAFE = {
    "dict": dict, "list": list, "tuple": tuple, "set": set,
    "len": len, "isinstance": isinstance, "print": print,
    "range": range, "enumerate": enumerate, "zip": zip,
    "min": min, "max": max, "sum": sum, "abs": abs, "all": all, "any": any,
    "True": True, "False": False, "None": None,
}

# 绕过安全编译（直接用裸源码测试逻辑）
ns = {"__builtins__": {}}
ns.update(_RUNTIME_SAFE)
ns.update({"UP": "up", "DOWN": "down", "LEFT": "left", "RIGHT": "right"})
try:
    _safe_compile(instrumented)
    print("安全编译通过")
except Exception as e:
    print("安全编译拒绝（print 不在白名单）:", e)
    # print 不在白名单，改用 Exception 也无法打印。改用纯计算记录
    instrumented = instrumented.replace("print(", "none_print(")
    ns["none_print"] = lambda *a, **k: None
    _safe_compile(instrumented)
    print("替换后编译通过")

# 不用 print，改为写全局变量记录
instrumented2 = instrumented.replace(
    "print('DBG: av_was', av, 'tx', tx, 'objs_now', objs)\n",
    "TRACE.append(('before_av_assign', dict(av), tx, {k:dict(v) for k,v in objs.items()}))\n"
).replace(
    "print('DBG: after av assign, objs', objs)\n",
    "TRACE.append(('after_av_assign', {k:dict(v) for k,v in objs.items()}))\n"
)

ns2 = dict(ns)
ns2["TRACE"] = []
_safe_compile(instrumented2)
exec(compile(instrumented2, "<d>", "exec"), ns2)
step = ns2["step"]

state = {
    "objects": {
        "a": {"x": 1, "y": 1, "role": "avatar"},
        "b": {"x": 2, "y": 1, "role": "box"},
    },
    "avatar_id": "a",
    "extras": {},
}
res = step(dict(state), "right")
print("TRACE:", ns2["TRACE"])
print("result:", res['objects'])
