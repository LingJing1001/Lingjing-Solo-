"""AR25 game-specific action and settled-frame contract."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AR25Action:
    name: str
    action_id: int
    settled_required: bool = True


@dataclass(frozen=True)
class AR25Profile:
    profile_id: str = "ar25-v1"
    reset_name: str = "RESET"
    actions: tuple[AR25Action, ...] = tuple(
        AR25Action(f"ACTION{i}", i) for i in range(1, 6)
    )

    @property
    def legal_action_names(self) -> list[str]:
        return [action.name for action in self.actions]

    def action_payload(self, action_id: int) -> dict[str, Any]:
        for action in self.actions:
            if action.action_id == action_id:
                return {"name": action.name, "payload": {"id": action.action_id}}
        raise ValueError(f"unsupported AR25 action id: {action_id}")

    def validate_tick(self, tick: dict[str, Any]) -> None:
        action = tick.get("requested_action") or {}
        if not isinstance(action, dict):
            raise ValueError("AR25 requested_action must be an object")
        name = action.get("name")
        if not isinstance(name, str):
            raise ValueError("AR25 action name must be a string")
        if name == self.reset_name:
            if tick.get("settled_frame") is not True:
                raise ValueError("AR25 RESET must have a settled frame")
            return
        by_name = {item.name: item for item in self.actions}
        rule = by_name.get(name)
        if rule is None:
            raise ValueError(f"AR25 action is not in profile: {name!r}")
        payload = action.get("payload")
        if not isinstance(payload, dict) or payload.get("id") != rule.action_id:
            raise ValueError(f"AR25 payload id does not match profile for {name}")
        if rule.settled_required and tick.get("settled_frame") is not True:
            raise ValueError(f"AR25 {name} requires a settled frame")


AR25_PROFILE = AR25Profile()
