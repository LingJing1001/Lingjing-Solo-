"""L6 CEAX Transfer 回归（课题探索包）。"""
from __future__ import annotations

import unittest

import numpy as np

from lingjing_solo.transfer import (
    CompetingHypotheses,
    HypothesisKind,
    RuleTransfer,
    TransferLayer,
    build_signature,
)


class TestTransferCore(unittest.TestCase):
    def test_same_mechanism_different_geometry(self):
        a = np.zeros((12, 12), dtype=np.int32)
        b = np.zeros((12, 12), dtype=np.int32)
        a[1:3, 1:3] = 4
        a[5:7, 5:7] = 7
        b[8:10, 2:4] = 4
        b[1:3, 8:10] = 7
        sa = build_signature(a, actions=["UP", "DOWN"])
        sb = build_signature(b, actions=["UP", "DOWN"])
        self.assertEqual(sa.mechanism_fingerprint(), sb.mechanism_fingerprint())

    def test_extract_by_kind_not_name(self):
        hyps = CompetingHypotheses()
        hyps.support(HypothesisKind.MOVEMENT, amount=0.4)
        hyps.support(HypothesisKind.LEVEL_PROGRESS, amount=0.4)
        skills = RuleTransfer().extract(hyps.ranked())
        self.assertIn("movement", skills)
        self.assertIn("level_progress", skills)

    def test_a_to_b_import(self):
        a = TransferLayer()
        for _ in range(6):
            a.hyps.support(HypothesisKind.TOGGLE_ON_CONTACT, amount=0.15)
        a.on_episode_end()
        b = TransferLayer()
        b.import_skills = None  # noqa: ensure attribute unused
        b.reset_episode(keep_skills=False, import_cross_game=a.skills.export())
        self.assertGreater(len(b.skills.skills), 0)
        self.assertIsNotNone(b.skills.prioritize())


if __name__ == "__main__":
    unittest.main()
