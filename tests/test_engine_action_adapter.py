from __future__ import annotations

import sys
import types

from lingjing_solo.harness.engine_action_adapter import action_input_for_id


def test_action_input_for_id_uses_named_action(monkeypatch):
    class FakeGameAction:
        ACTION1 = "enum:ACTION1"

    class FakeActionInput:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    module = types.ModuleType("arcengine")
    setattr(module, "GameAction", FakeGameAction)
    setattr(module, "ActionInput", FakeActionInput)
    monkeypatch.setitem(sys.modules, "arcengine", module)

    result = action_input_for_id(1)

    assert result.id == "enum:ACTION1"
    assert result.data == {}
    assert result.reasoning is None
