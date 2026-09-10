"""跨游戏机制族：只迁移族标签技能，失败则隔离降信。"""
from __future__ import annotations

from typing import Dict, Optional

from .rule_transfer import CROSS_GAME_FAMILIES
from .skill_library import SkillLibrary


# 粗粒度游戏族标签（可按 game_id 前缀扩展）
GAME_FAMILY_HINTS = {
    "ls20": "switch_gate",
    "ar25": "navigation",
    "vc33": "click_interact",
    "sp80": "click_interact",
}


def infer_game_family(game_id: Optional[str]) -> str:
    if not game_id:
        return "generic"
    gid = str(game_id).lower()
    for prefix, family in GAME_FAMILY_HINTS.items():
        if gid.startswith(prefix):
            return family
    return "generic"


class FamilyTransfer:
    """管理跨 game 技能导入与污染隔离。"""

    def __init__(self, library: SkillLibrary):
        self.library = library
        self._quarantine: Dict[str, float] = {}

    def import_for_game(self, skills: dict, game_id: Optional[str] = None):
        family = infer_game_family(game_id)
        filtered = {}
        for name, value in (skills or {}).items():
            if not isinstance(value, dict):
                continue
            fam = value.get("family", "generic")
            if fam not in CROSS_GAME_FAMILIES:
                continue
            if name in self._quarantine and self._quarantine[name] < 0.5:
                continue
            # 同族优先；异族仍允许机制级技能
            payload = dict(value)
            if fam == family:
                payload["confidence"] = min(0.99, float(payload.get("confidence", 0.75)) + 0.05)
            filtered[name] = payload
        self.library.import_many(filtered, cross_game=True)

    def note_failure(self, skill_name: str):
        self.library.decay(skill_name, amount=0.2)
        self._quarantine[skill_name] = self._quarantine.get(skill_name, 1.0) * 0.5

    def note_success(self, skill_name: str):
        if skill_name in self._quarantine:
            del self._quarantine[skill_name]
