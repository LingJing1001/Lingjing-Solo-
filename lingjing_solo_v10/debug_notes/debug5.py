"""debug5.py — 用真实 _compile 的命名空间复现"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lingjing_solo.world_model.codegen import FakeLLM, CodeGenerator, _safe_compile
from lingjing_solo.world_model.induction import RelationalInducer

src = FakeLLM().generate("dummy prompt")

# 复刻 _compile 的命名空间
_RUNTIME_SAFE = {
    "dict": dict, "list": list, "tuple": tuple, "set": set,
    "len": len, "isinstance": isinstance,
    "range": range, "enumerate": enumerate, "zip": zip,
    "min": min, "max": max, "sum": sum, "abs": abs,
    "all": all, "any": any,
    "True": True, "False": False, "None": None,
}
ns = {"__builtins__": {}}
ns.update(_RUNTIME_SAFE)
ns.update({"UP": "up", "DOWN": "down", "LEFT": "left", "RIGHT": "right"})

print("编译中...")
try:
    _safe_compile(src)
    exec(compile(src, "<d>", "exec"), ns)
    step = ns["step"]
except Exception as e:
    print("编译/执行错误:", type(e).__name__, e)
    raise SystemExit

state = {
    "objects": {
        "a": {"x": 1, "y": 1, "role": "avatar"},
        "b": {"x": 2, "y": 1, "role": "box"},
    },
    "avatar_id": "a",
    "extras": {},
}

for act in ["up", "down", "left", "right"]:
    try:
        s = step(dict(state), act)
        print(f"  {act}: {s['objects'] if s else None}")
    except Exception as e:
        print(f"  {act}: 异常 {type(e).__name__}: {e}")
