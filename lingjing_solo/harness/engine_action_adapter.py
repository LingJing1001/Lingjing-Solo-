"""ARC engine action conversion boundary.

Planner and search code use integer/abstract action identifiers.  Only this
harness-facing adapter knows how to construct ``arcengine`` action objects.
"""


def ensure_engine_available() -> None:
    """Raise ``ImportError`` when the optional ARC engine is unavailable."""
    from arcengine import ActionInput, GameAction  # noqa: F401


def action_input_for_id(action_id: int):
    """Convert an abstract numeric action id into an ARC engine input."""
    from arcengine import ActionInput, GameAction

    action = getattr(GameAction, f"ACTION{action_id}", None)
    if action is None:
        action = GameAction.from_id(action_id)
    return ActionInput(id=action, data={}, reasoning=None)
