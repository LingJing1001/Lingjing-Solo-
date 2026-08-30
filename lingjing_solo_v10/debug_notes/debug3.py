"""debug3.py — 直接调用编译后的 step 函数看 push 行为"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lingjing_solo.world_model.codegen import FakeLLM, CodeGenerator
from lingjing_solo.world_model.induction import RelationalInducer
from lingjing_solo.world_model.symbols import SymbolTable
from lingjing_solo.world_model.program import WorldModelProgram, WMPEvidence

gen = CodeGenerator(llm_client=FakeLLM(), rules_inducer=RelationalInducer())
prog = gen.generate(rules=[], traces=[])
print("provenance:", prog.provenance)
step = prog.step_fn

state = {
    "objects": {
        "a": {"x": 1, "y": 1, "role": "avatar"},
        "b": {"x": 2, "y": 1, "role": "box"},
    },
    "avatar_id": "a",
    "extras": {"grid_w": 8, "grid_h": 6},
}

for act in ["up", "down", "left", "right"]:
    s = step(dict(state), act)
    print(f"  {act}: {s['objects'] if s else None}")
