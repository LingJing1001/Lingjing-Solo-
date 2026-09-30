"""置信门控技能库：关卡切换可持久；跨游戏仅导入机制族。"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

from .rule_transfer import CROSS_GAME_FAMILIES


class SkillLibrary:
    def __init__(self, min_confidence: float = 0.75):
        self.min_confidence = min_confidence
        self.skills: Dict[str, dict] = {}
        self.hits: int = 0
        self.queries: int = 0

    def add(self, name: str, representation, confidence: float = 1.0, **extra):
        conf = float(confidence)
        if conf < self.min_confidence:
            return
        if isinstance(representation, dict):
            payload = dict(representation)
            payload.setdefault("confidence", conf)
        else:
            payload = {
                "representation": representation,
                "confidence": conf,
            }
        payload.update(extra)
        payload["confidence"] = max(conf, float(payload.get("confidence", conf)))
        # 同名取更高置信
        old = self.skills.get(name)
        if old and float(old.get("confidence", 0)) > payload["confidence"]:
            return
        self.skills[name] = payload

    def export(self) -> Dict[str, dict]:
        return {k: dict(v) for k, v in self.skills.items()}

    def import_many(self, skills: Optional[dict], *, cross_game: bool = False):
        if not skills:
            return
        for name, value in skills.items():
            if isinstance(value, dict):
                family = value.get("family", "generic")
                if cross_game and family not in CROSS_GAME_FAMILIES:
                    continue
                if cross_game and not value.get("cross_game", family in CROSS_GAME_FAMILIES):
                    continue
                # 禁止坐标脚本污染
                rep = str(value.get("representation", ""))
                if cross_game and any(tok in rep.lower() for tok in ("@", "coord", "x=", "y=")):
                    continue
                self.add(
                    name,
                    value.get("representation", value),
                    value.get("confidence", 1.0),
                    kind=value.get("kind"),
                    family=family,
                    cross_game=value.get("cross_game", family in CROSS_GAME_FAMILIES),
                    composed=value.get("composed", False),
                )
            else:
                if cross_game:
                    continue
                self.add(name, value, 1.0)

    def has(self, name: str) -> bool:
        return name in self.skills

    def by_family(self, family: str) -> Dict[str, dict]:
        return {k: v for k, v in self.skills.items() if v.get("family") == family}

    def prioritize(self, state=None) -> Optional[str]:
        """返回当前应优先验证的技能 id（区分实验先验）。"""
        self.queries += 1
        # 优先验证开关门 / 点击，再移动
        order = (
            "compose:toggle_on_contact->unblocks_when_toggled",
            "toggle_on_contact",
            "click_effect",
            "compose:click_effect->may_progress_level",
            "blocked_transition",
            "movement",
            "level_progress",
        )
        for name in order:
            if name in self.skills and float(self.skills[name].get("confidence", 0)) >= self.min_confidence:
                self.hits += 1
                return name
        # 任意高置信技能
        ranked = sorted(
            self.skills.items(),
            key=lambda kv: float(kv[1].get("confidence", 0)),
            reverse=True,
        )
        if ranked and float(ranked[0][1].get("confidence", 0)) >= self.min_confidence:
            self.hits += 1
            return ranked[0][0]
        return None

    @property
    def transfer_hit_rate(self) -> float:
        if self.queries <= 0:
            return 0.0
        return self.hits / self.queries

    def decay(self, name: str, amount: float = 0.15):
        """迁移失败时降信并可能剔除。"""
        if name not in self.skills:
            return
        conf = float(self.skills[name].get("confidence", 0)) - amount
        if conf < self.min_confidence:
            del self.skills[name]
        else:
            self.skills[name]["confidence"] = conf

    def clear(self):
        self.skills.clear()
        self.hits = 0
        self.queries = 0
