"""L6 CEAX Transfer 回归：结构化抽取、跨布局迁移、组合规则、跨游戏族。"""
from __future__ import annotations

import unittest

import numpy as np

from lingjing_solo.transfer import (
    CompetingHypotheses,
    HypothesisKind,
    RuleComposer,
    RulePrimitive,
    RuleTransfer,
    SkillLibrary,
    TransferLayer,
    build_signature,
)
from lingjing_solo.transfer.families import FamilyTransfer


def _grid_layout_a():
    g = np.zeros((16, 16), dtype=np.int32)
    g[2:4, 2:4] = 3  # agent-like
    g[8:10, 8:10] = 5  # trigger-like
    g[12:14, 4:10] = 9  # gate-like bar
    return g


def _grid_layout_b():
    """同机制异几何。"""
    g = np.zeros((16, 16), dtype=np.int32)
    g[1:3, 10:12] = 3
    g[6:8, 1:3] = 5
    g[10:12, 6:14] = 9
    return g


class TestSignatures(unittest.TestCase):
    def test_mechanism_fingerprint_stable_across_geometry(self):
        a = build_signature(_grid_layout_a(), actions=["ACTION1", "ACTION2"])
        b = build_signature(_grid_layout_b(), actions=["ACTION1", "ACTION2"])
        self.assertEqual(a.mechanism_fingerprint(), b.mechanism_fingerprint())
        self.assertNotEqual(a.objects[0].bbox_w + a.objects[0].bbox_h, 0)


class TestStructuredExtract(unittest.TestCase):
    def test_no_name_substring_matching(self):
        hyps = CompetingHypotheses()
        hyps.support(HypothesisKind.TOGGLE_ON_CONTACT, amount=0.5)
        hyps.support(HypothesisKind.BLOCKED_TRANSITION, amount=0.5)
        skills = RuleTransfer().extract(hyps.ranked())
        self.assertIn("toggle_on_contact", skills)
        self.assertIn("blocked_transition", skills)
        # 组合宏技能
        self.assertTrue(any(k.startswith("compose:") for k in skills))

    def test_composer_macro(self):
        prims = [
            RulePrimitive(HypothesisKind.TOGGLE_ON_CONTACT, "toggle_on_contact", "gate", 0.9),
            RulePrimitive(HypothesisKind.BLOCKED_TRANSITION, "blocked", "no_move", 0.85),
        ]
        out = RuleComposer().compose(prims)
        self.assertTrue(any(p.effect == "unblocks_when_toggled" for p in out))


class TestTransferAtoB(unittest.TestCase):
    def test_skill_import_reduces_cold_start(self):
        layer_a = TransferLayer()
        # 模拟 Task A 学到开关门
        for _ in range(8):
            layer_a.hyps.support(HypothesisKind.TOGGLE_ON_CONTACT, amount=0.12)
            layer_a.hyps.support(HypothesisKind.BLOCKED_TRANSITION, amount=0.12)
            layer_a.hyps.support(HypothesisKind.MOVEMENT, amount=0.10)
        layer_a.on_episode_end()
        self.assertGreater(len(layer_a.skills.skills), 0)

        layer_b = TransferLayer()
        layer_b.reset_episode(
            keep_skills=False,
            import_cross_game=layer_a.skills.export(),
        )
        self.assertGreater(len(layer_b.skills.skills), 0)
        # 导入后 prioritize 应命中
        skill = layer_b.skills.prioritize()
        self.assertIsNotNone(skill)
        suggestion = layer_b.suggest_action(
            ["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
            explore_scored=[("ACTION1", 0.1), ("ACTION2", 0.1), ("ACTION3", 0.1), ("ACTION4", 0.1)],
        )
        self.assertIsNotNone(suggestion)
        self.assertTrue(suggestion[1].startswith("transfer_"))


class TestLevelPersist(unittest.TestCase):
    def test_skills_survive_level_up(self):
        layer = TransferLayer()
        layer.hyps.support(HypothesisKind.CLICK_EFFECT, amount=0.5)
        layer.on_level_up()
        self.assertIn("click_effect", layer.skills.skills)
        n = len(layer.skills.skills)
        layer.on_level_up()
        self.assertEqual(len(layer.skills.skills), n)


class TestCrossGameFamily(unittest.TestCase):
    def test_blocks_coordinate_scripts(self):
        lib = SkillLibrary()
        fam = FamilyTransfer(lib)
        fam.import_for_game(
            {
                "bad": {
                    "representation": "go to coord x=12 y=34",
                    "confidence": 0.99,
                    "family": "navigation",
                    "cross_game": True,
                },
                "good": {
                    "representation": "IF move action THEN spatial change",
                    "confidence": 0.9,
                    "family": "navigation",
                    "cross_game": True,
                },
            },
            game_id="ls20",
        )
        self.assertNotIn("bad", lib.skills)
        self.assertIn("good", lib.skills)

    def test_quarantine_on_failure(self):
        lib = SkillLibrary()
        fam = FamilyTransfer(lib)
        lib.add("toggle_on_contact", "IF contact THEN toggle", 0.9, family="switch_gate", cross_game=True)
        fam.note_failure("toggle_on_contact")
        # 多次失败可剔除
        fam.note_failure("toggle_on_contact")
        fam.note_failure("toggle_on_contact")
        fam.note_failure("toggle_on_contact")
        self.assertTrue(
            "toggle_on_contact" not in lib.skills
            or float(lib.skills.get("toggle_on_contact", {}).get("confidence", 0)) < 0.75
        )


class TestR5Context(unittest.TestCase):
    def test_r5_lists_skills(self):
        layer = TransferLayer()
        layer.skills.add(
            "movement",
            "IF move THEN change",
            0.9,
            family="navigation",
            cross_game=True,
        )
        text = layer.r5_skill_context()
        self.assertIn("movement", text)


if __name__ == "__main__":
    unittest.main()
