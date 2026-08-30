"""debugA.py — 检查 step_fn 闭包变量"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lingjing_solo.world_model.codegen import FakeLLM, CodeGenerator
from lingjing_solo.world_model.induction import RelationalInducer

gen = CodeGenerator(llm_client=FakeLLM(), rules_inducer=RelationalInducer())
prog = gen.generate(rules=[], traces=[])

# 反编译查看 step 源码（code object 无法直接看，改用 inspect）
import inspect
print("step_fn 源码（通过 inspect）：")
try:
    print(inspect.getsource(prog.step_fn))
except Exception as e:
    print("无法获取源码:", e)

# 关键：检查闭包捕获的全局变量
print("\nstep_fn.__code__.co_varnames:", prog.step_fn.__code__.co_varnames)
print("step_fn.__code__.co_names:", prog.step_fn.__code__.co_names)
print("step_fn.__globals__ 含 UP?:", "UP" in prog.step_fn.__globals__)
print("step_fn.__globals__['UP']:", prog.step_fn.__globals__.get("UP"))
