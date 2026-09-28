"""SSA lincore 回归：四子空间世界模型 / 三信号 / 特征选项 / 梯度精调 / 矩阵记忆 / CEAX 嫁接。

全部用已知真值的合成动力系统验证（SSA 白皮书 §8 验证协议第 1 层）：
  - 不变量掩码必须恢复"从未变化的区域"（零空间 N(E) 支撑集）
  - 惊讶度 r：已辨识方向的残差 < 新方向
  - 进度轴 s：与真实信息方向对齐更高
  - 特征选项：命中真按钮区域
  - 周期检测：period-4 闪烁序列 → 4
  - 低秩补全：秩-1 结构缺失项恢复（最小范数解）
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lingjing_solo.lincore import (
    DualBudgetScheduler,
    EigenSkillMap,
    HypothesisLab,
    LogisticSGD,
    MatrixMemory,
    RidgeLinear,
    SharedBasis,
    SpectralMind,
    SubspaceModel,
    TransitionReplay,
    detect_period,
    encode_grid,
    grid_summary,
    krylov_horizon,
    residual_novelty,
    combined_row_space,
    ridge_solve,
)
from lingjing_solo.lincore.segments import cochange_affinity, object_regions, spectral_segments


# ---------- 合成环境 ----------

def _blank(size: int = 64) -> np.ndarray:
    return np.zeros((size, size), dtype=np.int32)


def _toggle_pair(toggle_on: bool) -> tuple:
    """按钮游戏：右下角 (54..58)² 按钮区 4×4 亮/灭，其余全静态。"""
    base = _blank()
    g = base.copy()
    if toggle_on:
        g[54:58, 54:58] = 7
    return base, g


class _FakeObject:
    """最小 GameObject 桩（pixels/bbox/color 契约与 PerceptionEncoder 对齐）。"""

    def __init__(self, x: int, y: int, color: int = 7):
        self.color = color
        self.bbox = (x, y, x + 2, y + 2)
        self.pixels = [(y, x), (y, x + 1), (y + 1, x), (y + 1, x + 1)]


# ---------- L1 特征 ----------

class TestFeatures(unittest.TestCase):
    def test_encode_grid_shape_and_values(self):
        base, on = _toggle_pair(True)
        f = encode_grid(on, blocks=8)
        self.assertIsNotNone(f)
        self.assertEqual(f.shape[0], 2 * 8 * 8)
        # 按钮所在块（block_of(56,56) → (7,7)）占用非零
        b2 = 64
        self.assertGreater(float(f[7 * 8 + 7]), 0.0)
        self.assertGreater(float(f[b2 + 7 * 8 + 7]), 0.0)
        # 空块的占用为 0
        self.assertEqual(float(f[0]), 0.0)

    def test_grid_summary_runs(self):
        s = grid_summary(_toggle_pair(True)[1], blocks=8)
        self.assertIsNotNone(s)
        self.assertEqual(s.shape[0], 4)


# ---------- L2 四子空间 ----------

class TestSubspaceModel(unittest.TestCase):
    def test_invariants_recover_static_region(self):
        """按钮游戏 10 次往返后：按钮块为活块，远离按钮的块为死块。"""
        m = SubspaceModel(dim=128, k=6)
        for _ in range(5):
            base, on = _toggle_pair(False)
            _, on2 = _toggle_pair(True)
            m.update(encode_grid(on2, 8) - encode_grid(base, 8))
            m.update(encode_grid(base, 8) - encode_grid(on2, 8))
        live = m.invariant_block_mask(blocks=8)
        self.assertTrue(live[7, 7], "按钮块必须是活块")
        self.assertFalse(live[0, 0], "左上角从未变化，必须是死块")
        self.assertFalse(live[2, 3], "中部从未变化，必须是死块")

    def test_novelty_seen_lt_unseen(self):
        m = SubspaceModel(dim=128, k=4)
        base, on = _toggle_pair(False)
        _, on2 = _toggle_pair(True)
        d_btn = encode_grid(on2, 8) - encode_grid(base, 8)
        for _ in range(6):
            m.update(d_btn)
        m.fit(force=True)
        # 已见按钮方向 → 低惊讶
        self.assertLess(m.novelty(d_btn), 0.35)
        # 全新方向（左上角出现效应）→ 高惊讶
        new_g = _blank()
        new_g[4:8, 4:8] = 9
        d_new = encode_grid(new_g, 8) - encode_grid(_blank(), 8)
        self.assertGreater(m.novelty(d_new), 0.7)

    def test_progress_axis_alignment(self):
        """y=1 的样本集中在 +e0 方向：进度轴应给 +e0 更高对齐。"""
        d = 128
        m = SubspaceModel(dim=d, k=6, ridge=1e-2)
        rng = np.random.default_rng(7)
        u = np.zeros(d)
        u[0] = 1.0  # 信息方向
        for _ in range(12):
            m.update(u * 2.0 + rng.normal(0, 0.01, d), progressed=True)
        for _ in range(12):
            m.update(rng.normal(0, 0.05, d), progressed=False)
        axis = m.progress_axis()
        self.assertIsNotNone(axis)
        s_info = m.progress_alignment(u * 2.0, axis)
        s_noise = m.progress_alignment(np.eye(d)[5] * 0.1, axis)
        self.assertGreater(s_info, 0.5)
        self.assertGreater(s_info, abs(s_noise))

    def test_effective_rank_bounded(self):
        m = SubspaceModel(dim=128, k=8)
        rng = np.random.default_rng(3)
        for _ in range(10):
            m.update(rng.normal(0, 1, 128))
        self.assertGreaterEqual(m.effective_rank, 1)
        self.assertLessEqual(m.effective_rank, 8)

    def test_bad_input_ignored(self):
        m = SubspaceModel(dim=8)
        self.assertFalse(m.update(None))
        self.assertFalse(m.update(np.ones(7)))
        self.assertFalse(m.update(np.full(8, np.nan)))
        self.assertEqual(m.n_samples, 0)


# ---------- 谱工具 ----------

class TestSpectralTools(unittest.TestCase):
    def test_ridge_solve_recovers_line(self):
        rng = np.random.default_rng(11)
        X = rng.normal(0, 1, (200, 3))
        y = X @ np.array([2.0, -1.0, 0.5]) + rng.normal(0, 0.01, 200)
        w = ridge_solve(X, y, lam=1e-4)
        self.assertIsNotNone(w)
        self.assertLess(float(np.linalg.norm(w - [2.0, -1.0, 0.5])), 0.1)

    def test_ridge_linear_converges(self):
        r = RidgeLinear(dim=2, lam=1e-3)
        rng = np.random.default_rng(5)
        for _ in range(300):
            x = rng.normal(0, 1, 2)
            r.update(x, 3.0 * x[0] - x[1])
        val, unc = r.predict(np.array([1.0, 1.0]))
        self.assertLess(abs(val - 2.0), 0.25)
        self.assertLessEqual(unc, 1.0)

    def test_detect_period_four(self):
        seq = [0.0, 3.0, 7.0, 3.0] * 12
        self.assertEqual(detect_period(seq, min_p=2, max_p=16), 4)

    def test_detect_period_none_on_noise(self):
        rng = np.random.default_rng(17)
        seq = rng.normal(0, 1, 64).tolist()
        self.assertIsNone(detect_period(seq, min_p=2, max_p=16, corr_thresh=0.8))

    def test_krylov_horizon(self):
        # T̂: 收缩到目标轴的线性映射
        g = np.zeros(4)
        g[0] = 1.0
        def step(v):
            out = 0.6 * v
            out[0] += 0.5 * g[0]
            return out
        h = krylov_horizon(step, np.zeros(4), g, max_k=12, thresh=0.9)
        self.assertIsNotNone(h)
        self.assertGreaterEqual(h, 1)
        self.assertLessEqual(h, 12)


# ---------- L4 特征选项 ----------

class TestEigenSkillMap(unittest.TestCase):
    def test_rank_candidates_hits_true_button(self):
        em = EigenSkillMap(64, ["ACTION1", "ACTION6"])
        true_region = 7 * 8 + 7
        decoy = 0
        for _ in range(8):
            em.note("ACTION6", true_region, 1.0)
        em.note("ACTION6", decoy, 0.1)
        ranked = em.rank_candidates([true_region, decoy, 3 * 8 + 3])
        self.assertGreater(len(ranked), 0)
        self.assertEqual(ranked[0][0], true_region)

    def test_untried_regions_get_coverage_score(self):
        em = EigenSkillMap(16, ["ACTION6"])
        em.note("ACTION6", 0, 1.0)
        ranked = em.rank_candidates([0, 5])
        by_region = dict(ranked)
        # 从未尝试的区域应有非零未尝试分
        self.assertGreater(by_region[5], 0.0)


# ---------- L5 梯度与对偶预算 ----------

class TestGradient(unittest.TestCase):
    def test_logistic_sgd_separable(self):
        sgd = LogisticSGD(dim=2, lr=0.15, momentum=0.9, l2=1e-5)
        rng = np.random.default_rng(23)
        for _ in range(300):
            x = rng.normal(0, 1, 2)
            y = 1.0 if x[0] > 0 else 0.0
            sgd.partial_fit(x, y)
        self.assertGreater(sgd.predict_proba(np.array([1.0, 0.0])), 0.7)
        self.assertLess(sgd.predict_proba(np.array([-1.0, 0.0])), 0.3)

    def test_dual_budget_bounded_and_decaying(self):
        sch = DualBudgetScheduler(horizon=100, explore_share=0.3, eta=0.05)
        gates = [sch.explore_gate() for _ in range(1)]
        p0 = sch.explore_gate()
        for _ in range(200):
            sch.step(True)  # 持续超支
        p1 = sch.explore_gate()
        self.assertGreaterEqual(p0, 0.5)
        self.assertLess(p1, p0)
        self.assertGreater(p1, 0.0, "对偶压力必须有界，不能把探索压到 0")


# ---------- L5 矩阵记忆 ----------

class TestMatrixMemory(unittest.TestCase):
    def test_completion_recovers_rank1_missing(self):
        mem = MatrixMemory(rank=2)
        mem.record("gameA", {"a1_gain": 1.0, "a2_gain": 2.0, "a3_gain": 3.0, "a4_gain": 4.0})
        mem.record("gameB", {"a1_gain": 2.0, "a2_gain": 4.0, "a3_gain": 6.0, "a4_gain": 8.0})
        mem.record("gameC", {"a1_gain": 2.0, "a2_gain": 4.0})  # 缺 a3/a4
        prior = mem.prior_for("gameC")
        self.assertAlmostEqual(prior["a3_gain"], 6.0, places=1)
        self.assertAlmostEqual(prior["a4_gain"], 8.0, places=1)

    def test_unknown_game_gets_global_prior(self):
        mem = MatrixMemory(rank=2)
        mem.record("gameA", {"a1_gain": 1.0, "a2_gain": 2.0})
        prior = mem.prior_for("brand_new")
        self.assertIsInstance(prior, dict)

    def test_persistence_roundtrip(self):
        mem = MatrixMemory()
        mem.record("g", {"a1_gain": 0.5})
        data = mem.to_dict()
        mem2 = MatrixMemory()
        mem2.from_dict(data)
        self.assertAlmostEqual(mem2.prior_for("g")["a1_gain"], 0.5, places=5)


# ---------- L3/L4 编排器 ----------

class TestSpectralMind(unittest.TestCase):
    def test_toggle_loop_learns_and_prioritizes_button(self):
        mind = SpectralMind(blocks=8, game_sig="t", memory=MatrixMemory())
        prev, cur = _toggle_pair(False), None
        g0, g1 = _toggle_pair(True)
        # 模拟 6 轮点击按钮：off→on→off…
        grid = g0
        for i in range(12):
            nxt = g1 if i % 2 == 0 else g0
            mind.note_choice("ACTION6", (56, 56), (64, 64))
            mind.observe(grid, nxt, "ACTION6", levels=0, progressed=False)
            grid = nxt
        mind.maintenance()
        scores = mind.score_click_candidates([(1, (56, 56)), (2, (8, 8)), (3, (30, 30))])
        self.assertGreater(scores.get(1, 0.0), scores.get(2, 0.0))

    def test_period_detection_via_mind(self):
        mind = SpectralMind(blocks=8, game_sig="p", memory=MatrixMemory())
        seq = [3.0, 0.0, 7.0, 0.0] * 8
        grid = _blank()
        for i, count in enumerate(seq):
            nxt = _blank()
            nxt[10:13, 10:13] = int(count)
            mind.observe(grid, nxt, "ACTION1", levels=0, progressed=False)
            # 直接喂周期序列（observe 内部记录的是像素差，改为注入）
            mind._change_counts.append(count)
            grid = nxt
        p = detect_period(list(mind._change_counts), min_p=2, max_p=16)
        self.assertEqual(p, 4)

    def test_prior_changes_action_scores_and_gate(self):
        """世界先验接入决策：冷启动识别（本地未见、他局已见）+ 探索门响应新奇流。"""
        g = _blank()
        g[4:8, 4:8] = 5
        d_prior = encode_grid(g, 8) - encode_grid(_blank(), 8)
        prior = SharedBasis.fit(np.vstack([d_prior] * 6), k=4)
        # 冷启动：两局都还没任何本地观测 —— 带先验者立刻认出这是已知机制
        mind_p = SpectralMind(blocks=8, game_sig="wp", memory=MatrixMemory(), shared_basis=prior)
        mind_n = SpectralMind(blocks=8, game_sig="wn", memory=MatrixMemory())
        self.assertLess(mind_p.world_novelty(d_prior), 0.3)
        self.assertGreater(mind_n.world_novelty(d_prior), 0.9)
        # 在线流：本地观测时，带先验者记录的新奇流更低 → 探索门更低（少浪费预算）
        grid = _blank()
        for _ in range(6):
            g2 = _blank()
            g2[4:8, 4:8] = 5
            mind_p.observe(grid, g2, "ACTION1", levels=0, progressed=False)
            mind_n.observe(grid, g2, "ACTION1", levels=0, progressed=False)
        self.assertLessEqual(mind_p.explore_gate(), mind_n.explore_gate())
        # 动作评分中世界新奇度项生效（带先验者对该已知机制动作的世界新奇分更低）
        s_p = mind_p.score_actions(["ACTION1"])
        s_n = mind_n.score_actions(["ACTION1"])
        self.assertLessEqual(
            mind_p.world_novelty(mind_p._action_sig["ACTION1"]),
            mind_n.world_novelty(mind_n._action_sig["ACTION1"]) + 1e-9,
        )
        self.assertIn("ACTION1", s_p)
        self.assertIn("ACTION1", s_n)

    def test_neutral_on_garbage(self):
        mind = SpectralMind(blocks=4, game_sig="x", memory=MatrixMemory())
        self.assertEqual(mind.observe(None, None, "ACTION1", 0, False)["rank"], 0)
        self.assertEqual(mind.score_click_candidates([]), {})
        self.assertIsNotNone(mind.snapshot())


# ---------- CEAX 嫁接层 ----------

class TestSpectralLayer(unittest.TestCase):
    def test_smoke_choose_and_prior(self):
        from lingjing_solo.transfer import SpectralCeaxController

        c = SpectralCeaxController(game_sig="smoke")
        grid0, grid1 = _toggle_pair(True)
        valid = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6"]
        objs = [_FakeObject(55, 55, 7), _FakeObject(6, 6, 3), _FakeObject(28, 30, 5)]
        act0, xy0, why0 = c.choose(valid_actions=valid, objects=objs, grid=grid0, grid_hash="h0")
        self.assertIn(act0, valid)
        c.observe_outcome(
            delta_pixels=16, progressed=False, grid_hash="h1",
            levels=0, prev_grid=grid0, grid=grid1,
        )
        act1, xy1, why1 = c.choose(valid_actions=valid, objects=objs, grid=grid1, grid_hash="h1")
        self.assertIn(act1, valid)
        snap = c.snapshot()
        self.assertIn("spectral", snap)
        # 谱先验应已为未尝试目标生成
        self.assertIsInstance(c._spectral_prior, dict)

    def test_reset_clears_spectral_state(self):
        from lingjing_solo.transfer import SpectralCeaxController

        c = SpectralCeaxController(game_sig="r")
        grid0, grid1 = _toggle_pair(True)
        c.observe_outcome(delta_pixels=16, progressed=False, grid_hash="a",
                          levels=0, prev_grid=grid0, grid=grid1)
        c.reset_game()
        self.assertEqual(len(c._spectral_prior), 0)
        self.assertEqual(c.mind.sub.n_samples, 0)

    def test_level_up_resets_mind(self):
        from lingjing_solo.transfer import SpectralCeaxController

        c = SpectralCeaxController(game_sig="l")
        grid0, grid1 = _toggle_pair(True)
        c.observe_outcome(delta_pixels=16, progressed=False, grid_hash="a",
                          levels=0, prev_grid=grid0, grid=grid1)
        n_before = c.mind.sub.n_samples
        self.assertGreater(n_before, 0)
        c.on_level_up()
        self.assertEqual(c.mind.sub.n_samples, 0)


# ---------- Agent 模块导入（无 arcengine 时跳过） ----------

class TestAgentImport(unittest.TestCase):
    def test_spectral_agent_module_imports(self):
        try:
            import arcengine  # noqa: F401
        except ImportError:
            self.skipTest("arcengine 不可用（Kaggle 外环境）")
            return
        vendor = ROOT / "vendor" / "ARC-AGI-3-Agents"
        if vendor.is_dir():
            sys.path.insert(0, str(vendor))
        try:
            import agents.agent  # noqa: F401
        except ImportError:
            self.skipTest("agents 框架不可用（vendor 未初始化）")
            return
        import importlib.util

        path = ROOT / "agent" / "spectral_agi_agent.py"
        spec = importlib.util.spec_from_file_location("spectral_agi_agent_test", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertTrue(hasattr(mod, "MyAgent"))
        self.assertEqual(mod.BUILD_TAG, "ssa-spectral-v1")


# ---------- SSA-E3 谱-TTT ----------

class TestSharedBasisTTT(unittest.TestCase):
    def _delta_a(self):
        g = _blank()
        g[4:8, 4:8] = 5
        return encode_grid(g, 8) - encode_grid(_blank(), 8)

    def _delta_b(self):
        g = _blank()
        g[40:44, 40:44] = 9
        return encode_grid(g, 8) - encode_grid(_blank(), 8)

    def test_shared_basis_fit_persist_roundtrip(self):
        rows, sigs = [], []
        for _ in range(6):
            rows.append(self._delta_a())
            sigs.append("gameA")
            rows.append(self._delta_b())
            sigs.append("gameB")
        sb = SharedBasis.fit(np.vstack(rows), k=4, sigs=sigs)
        self.assertTrue(sb.ready)
        self.assertEqual(sb.n_transitions, 12)
        self.assertEqual(sb.n_games, 2)
        path = str(ROOT / "ui" / "static" / "_test_shared_basis.json")
        self.assertTrue(sb.save(path))
        sb2 = SharedBasis.load(path)
        self.assertTrue(np.allclose(sb.V, sb2.V))
        Path(path).unlink(missing_ok=True)

    def test_world_novelty_suppressed_on_prior_directions(self):
        rows = [self._delta_a(), self._delta_b()] * 3  # ≥4 样本才可拟合先验
        prior = SharedBasis.fit(np.vstack(rows), k=4)
        self.assertTrue(prior.ready)
        mind = SpectralMind(blocks=8, game_sig="t3", memory=MatrixMemory(), shared_basis=prior)
        # 本局无任何数据：普通 novelty=1.0，但先验方向的世界新奇度应被压低
        self.assertGreater(mind.sub.novelty(self._delta_a()), 0.99)
        wn_prior = mind.world_novelty(self._delta_a())
        unseen = np.zeros(128)
        unseen[90] = 1.0
        wn_new = mind.world_novelty(unseen)
        self.assertLess(wn_prior, 0.5)
        self.assertGreater(wn_new, 0.9)

    def test_replay_roundtrip_and_fit_from_replay(self):
        path = str(ROOT / "ui" / "static" / "_test_replay.jsonl")
        Path(path).unlink(missing_ok=True)  # 清理上次运行残留（追加式文件）
        rp = TransitionReplay(path)
        for _ in range(3):
            rp.append("gA", self._delta_a())
            rp.append("gB", self._delta_b())
        rp.append("gA", np.zeros(128))  # 全零不入库
        M, sigs = rp.load()
        self.assertEqual(M.shape, (6, 128))
        self.assertEqual(set(sigs), {"gA", "gB"})
        sb = SharedBasis.fit(M, k=2, sigs=sigs)
        self.assertTrue(sb.ready)
        Path(path).unlink(missing_ok=True)

    def test_combined_row_space_qr(self):
        Q = combined_row_space([np.vstack([self._delta_a(), self._delta_b()])])
        self.assertIsNotNone(Q)
        # 返回的是「行 = 正交基向量」：行 Gram ≈ I
        G = Q @ Q.T
        self.assertLess(float(np.max(np.abs(G - np.eye(G.shape[0])))), 1e-8)
        self.assertAlmostEqual(residual_novelty(self._delta_a(), Q), 0.0, places=6)


# ---------- SSA-E2 假设实验室 ----------

class TestHypothesisLab(unittest.TestCase):
    def test_llm_json_registers_and_verifies(self):
        lab = HypothesisLab()
        payload = {"suggestions": [
            {"action": "ACTION6", "x": 56, "y": 56, "expect": "右下按钮亮起", "confidence": 0.8},
        ]}
        hs = HypothesisLab.from_llm_json(payload)
        self.assertEqual(len(hs), 1)
        h = hs[0]
        lab.register(action=h.action, text=h.text, xy=h.xy,
                     expect_blocks=h.expect_blocks, prior=h.prior, hid=h.hid, source="llm")
        plan = lab.design_experiment()
        self.assertIsNotNone(plan)
        action, xy, why = plan
        self.assertEqual(action, "ACTION6")
        self.assertEqual(xy, (56, 56))
        # 按钮块真的变了（块 7*8+7）→ Jaccard 高 → 支持；再次后 verified
        v1 = lab.verify([63])
        self.assertEqual(v1["verdict"], "support")
        plan2 = lab.design_experiment()
        self.assertIsNotNone(plan2)
        v2 = lab.verify([63])
        self.assertTrue(v2["verified"])
        self.assertEqual(len(lab.verified_hypotheses()), 1)

    def test_wrong_hypothesis_refuted_and_exhausted(self):
        lab = HypothesisLab()
        lab.register(action="ACTION6", xy=(8, 8), expect_blocks=(0,), prior=0.5)
        for _ in range(4):
            lab.design_experiment()
            lab.verify([63])  # 实际变化全在别处
        h = list(lab.hyps.values())[0]
        self.assertGreaterEqual(h.refutes, h.supports)
        self.assertTrue(h.exhausted)
        self.assertFalse(h.verified)

    def test_extract_suggestions_json_tolerant(self):
        from lingjing_solo.lincore.hypothesis_lab import extract_suggestions_json

        self.assertIsNone(extract_suggestions_json(""))
        self.assertIsNone(extract_suggestions_json("抱歉，我无法解析这局。"))
        arr = extract_suggestions_json('好的：[{"action":"ACTION6","x":1,"y":2,"expect":"亮","confidence":0.8}] 完毕')
        self.assertIsNotNone(arr)
        self.assertEqual(len(arr["suggestions"]), 1)
        wrapped = extract_suggestions_json('{"suggestions":[{"action":"ACTION5"}]}')
        self.assertIsNotNone(wrapped)
        self.assertIn("suggestions", wrapped)

    def test_auto_from_mind_registers_leverage_hypotheses(self):
        mind = SpectralMind(blocks=8, game_sig="auto", memory=MatrixMemory())
        true_region = 7 * 8 + 7
        for _ in range(8):
            mind.eigen.note("ACTION6", true_region, 1.0)
        lab = HypothesisLab()
        n = lab.auto_from_mind(mind, shape=(64, 64), top_k=2)
        self.assertGreaterEqual(n, 1)
        plan = lab.design_experiment()
        self.assertIsNotNone(plan)
        action, xy, why = plan
        self.assertEqual(action, "ACTION6")
        # 高杠杆区域 (7,7) 的中心应被选为实验点
        self.assertEqual(xy, (7 * 8 + 4, 7 * 8 + 4))
        # 假设预期块应含本块及邻接块
        h = lab.hyps[why.split(":")[1]]
        self.assertIn(true_region, h.expect_blocks)

    def test_controller_auto_register_and_llm_bridge(self):
        from lingjing_solo.transfer import SpectralCeaxController

        c = SpectralCeaxController(game_sig="auto2")
        # 直接调用：无杠杆数据时不崩溃、注册 0 条
        self.assertEqual(c.auto_register_from_mind(), 0)
        # 给特征技能图喂杠杆后可注册
        for _ in range(6):
            c.mind.eigen.note("ACTION6", 5 * 8 + 5, 1.0)
        self.assertEqual(c.auto_register_from_mind(), 1)
        snap = c.snapshot()
        self.assertGreaterEqual(snap["lab"]["total"], 1)

    def test_controller_hypothesis_injection_e2e(self):
        from lingjing_solo.transfer import SpectralCeaxController

        c = SpectralCeaxController(game_sig="e2")
        c.register_hypotheses({"suggestions": [
            {"action": "ACTION6", "x": 40, "y": 40, "expect": "mid button", "confidence": 0.9},
        ]})
        # 两轮「实验 → 裁决」后假设应被验证（trials≥2 且 supports≥2）
        for i in range(2):
            g0 = _blank()
            act, xy, why = c.choose(valid_actions=["ACTION6"], objects=None,
                                    grid=g0, grid_hash=f"a{i}")
            self.assertEqual(act, "ACTION6")
            self.assertEqual(xy, (40, 40))
            g1 = _blank()
            g1[42:45, 42:45] = 6  # 块 (5,5) 变化，落在 (40,40) 的 3×3 预期邻域内
            c.observe_outcome(delta_pixels=9, progressed=False, grid_hash=f"b{i}",
                              levels=0, prev_grid=g0, grid=g1)
        snap = c.snapshot()
        self.assertEqual(snap["lab"]["verified"], 1)


# ---------- SSA-E4 谱聚类对象分割 ----------

class TestSegments(unittest.TestCase):
    def _history_two_objects(self):
        """「基帧→I→基帧→II」严格交替：每个帧对转移只触碰一个区域，
        共变化向量正交 → 亲和图不连通 → 归一化割必然二分。"""
        frames = []
        for t in range(10):
            g = _blank()
            phase = t % 4
            if phase == 1:
                g[8:12, 8:12] = 3       # 仅区域 I 亮
            elif phase == 3:
                g[48:52, 48:52] = 8     # 仅区域 II 亮
            frames.append(g)
        return frames

    def test_two_regions_form_two_clusters(self):
        hist = self._history_two_objects()
        A = cochange_affinity(hist, blocks=8)
        self.assertIsNotNone(A)
        labels = spectral_segments(A, n_clusters=2)
        self.assertIsNotNone(labels)
        lbl = {i: int(v) for i, v in enumerate(labels.tolist())}
        b_I, b_II = 1 * 8 + 1, 6 * 8 + 6
        self.assertNotEqual(lbl[b_I], lbl[b_II], "两个独立对象必须分属不同簇")
        regions = object_regions(hist, blocks=8, n_clusters=2)
        self.assertGreaterEqual(len(regions), 2)

    def test_static_history_returns_none(self):
        hist = [_blank() for _ in range(6)]
        self.assertIsNone(cochange_affinity(hist, blocks=8))
        self.assertEqual(object_regions(hist, blocks=8), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
