"""
V14.1 接口自检验证：
用一个"故意写错签名"的坏场景，确认 Scenario.validate() / Engine 能 fail-fast 拦住它。

这是"平台化底座"的关键能力：第三方插件签名写错时，构造即报错，
而不是疏散跑到一半才崩。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.scenario import Scenario, ScenarioInterfaceError
from core.field import Field, FieldConfig
from agents import Agent
from engine import Engine, EngineConfig
import numpy as np


class BadScenario(Scenario):
    """故意只写 2 参数 observation（违反引擎调用约定）的坏场景。"""

    def __init__(self):
        super().__init__({})
        self.shape = (8, 8, 8)
        self.exits = [(7, 4, 4)]

    def setup(self, field):
        pass

    def create_agents(self, field):
        return []

    def step(self, field, agents, dt):
        pass

    def observation(self, field, agent):  # 故意：缺 all_agents 参数
        return {"gradient": np.zeros(3), "local_phi": 0.0, "neighbors": []}


class GoodScenario(Scenario):
    """正确签名（all_agents 作为可选第 3 参数）。"""

    def __init__(self):
        super().__init__({})
        self.shape = (8, 8, 8)
        self.exits = [(7, 4, 4)]

    def setup(self, field):
        pass

    def create_agents(self, field):
        return []

    def step(self, field, agents, dt):
        pass

    def observation(self, field, agent, all_agents=None):
        return {"gradient": np.zeros(3), "local_phi": 0.0, "neighbors": []}


def test_bad_scenario_rejected_by_validate():
    """BadScenario.validate() 必须抛 ScenarioInterfaceError。"""
    bad = BadScenario()
    try:
        bad.validate()
    except ScenarioInterfaceError as e:
        print(f"  ✅ validate() 拦住坏场景: {type(e).__name__}: {e}")
        return
    raise AssertionError("BadScenario 应被 validate() 拒绝，但没有")


def test_bad_scenario_rejected_by_engine_constructor():
    """把坏场景传给 Engine，构造时必须立即报错（fail-fast）。"""
    bad = BadScenario()
    cfg = EngineConfig(field=FieldConfig(shape=(8, 8, 8)))
    try:
        Engine(cfg, bad)
    except ScenarioInterfaceError as e:
        print(f"  ✅ Engine 构造拦住坏场景: {type(e).__name__}")
        return
    raise AssertionError("Engine 构造应因坏场景而失败")


def test_good_scenario_accepted():
    """正确签名的场景能被 Engine 正常构造。"""
    good = GoodScenario()
    good.validate()
    print("  ✅ GoodScenario.validate() 通过")
    cfg = EngineConfig(field=FieldConfig(shape=(8, 8, 8)))
    eng = Engine(cfg, good)
    assert eng is not None
    print("  ✅ GoodScenario 可被 Engine 正常构造")


if __name__ == "__main__":
    tests = [
        ("坏场景被 validate 拦住", test_bad_scenario_rejected_by_validate),
        ("坏场景被 Engine 构造拦住", test_bad_scenario_rejected_by_engine_constructor),
        ("正确场景被接受", test_good_scenario_accepted),
    ]
    passed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ✅ {name}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {name}: {type(e).__name__}: {e}")
    print(f"\n接口自检测试: {passed}/{len(tests)} 通过")
    sys.exit(0 if passed == len(tests) else 1)
