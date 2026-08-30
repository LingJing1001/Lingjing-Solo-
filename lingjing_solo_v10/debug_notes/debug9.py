"""debug9.py — 直接调 step_fn 看 right 行为"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lingjing_solo.world_model.codegen import FakeLLM, CodeGenerator
from lingjing_solo.world_model.induction import RelationalInducer

gen = CodeGenerator(llm_client=FakeLLM(), rules_inducer=RelationalInducer())
prog = gen.generate(rules=[], traces=[])
step_fn = prog.step_fn

state = {
    "objects": {
        "a": {"x": 1, "y": 1, "role": "avatar"},
        "b": {"x": 2, "y": 1, "role": "box"},
    },
    "avatar_id": "a",
    "extras": {},
}

print("up    :", step_fn(dict(state), "up")["objects"])
print("down  :", step_fn(dict(state), "down")["objects"])
print("left  :", step_fn(dict(state), "left")["objects"])
print("right :", step_fn(dict(state), "right")["objects"])
print("expected right: a=(2,1), b=(3,1)")
