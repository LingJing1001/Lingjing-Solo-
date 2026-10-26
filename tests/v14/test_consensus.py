"""
V15.0 契约测试 —— 多 Agent 共识层（协议 6，R1-R5）。

  C20 写入顺序无关（100 次随机顺序提交，结果逐位一致）
  C21 版本号单调递增（每 tick +1，与 Agent 数无关）
  C22 C 值只读（c_value 不写场，不影响守恒）
"""
import numpy as np

from lingjing_solo.core.field import Field, FieldConfig
from lingjing_solo.core.consensus import ConsensusLayer
from lingjing_solo.core.writeback import AgentWriteback


def _make_wb(shape, values):
    """构造一个 AgentWriteback：values = {index: delta}。"""
    wb = AgentWriteback.empty(shape)
    for idx, v in values.items():
        wb.add(idx, v)
    return wb


def test_c20_write_order_independence():
    """C20：多 Agent 写入顺序无关（加法交换律）。

    3 个 Agent 各提交不同位置的源项，随机打乱提交顺序 100 次，
    最终场状态必须逐位一致。
    """
    cfg = FieldConfig(shape=(8, 8, 8), h=1.0 / 7, D=0.10, scheme="implicit")

    # 固定三个 Agent 的写回内容
    wb_a = _make_wb((8, 8, 8), {(2, 2, 2): 1.0})
    wb_b = _make_wb((8, 8, 8), {(4, 4, 4): 2.0})
    wb_c = _make_wb((8, 8, 8), {(6, 6, 6): 0.5})

    # 参考顺序：A → B → C
    ref_field = Field(cfg)
    consensus = ConsensusLayer(ref_field)
    consensus.begin_tick()
    consensus.submit("A", wb_a)
    consensus.submit("B", wb_b)
    consensus.submit("C", wb_c)
    J = consensus.aggregate(J_env=None)
    ref_field.apply_sources_and_evolve(J, dt=0.01)
    ref_phi = ref_field.phi.copy()

    # 随机打乱 100 次
    rng = np.random.default_rng(123)
    wbs = [("A", wb_a), ("B", wb_b), ("C", wb_c)]
    for _ in range(100):
        f = Field(cfg)
        c = ConsensusLayer(f)
        c.begin_tick()
        shuffled = rng.permutation(len(wbs))
        for i in shuffled:
            aid, wb = wbs[i]
            c.submit(aid, wb)
        J = c.aggregate(J_env=None)
        f.apply_sources_and_evolve(J, dt=0.01)
        assert np.allclose(f.phi, ref_phi, atol=1e-12), \
            f"C20 写入顺序相关: 与参考顺序 max_err={np.max(np.abs(f.phi - ref_phi)):.2e}"


def test_c21_version_monotonic():
    """C21：版本号每 tick +1，与 Agent 数量无关。"""
    cfg = FieldConfig(shape=(8, 8, 8), h=1.0 / 7, D=0.10, scheme="implicit")
    f = Field(cfg)
    c = ConsensusLayer(f)

    assert c.version == 0, f"初始版本号应为 0，实际 {c.version}"

    # 3 个 tick，每 tick 2 个 Agent
    for tick in range(3):
        c.begin_tick()
        c.submit("A", _make_wb((8, 8, 8), {(2, 2, 2): 1.0}))
        c.submit("B", _make_wb((8, 8, 8), {(4, 4, 4): 1.0}))
        J = c.aggregate(J_env=None)
        f.apply_sources_and_evolve(J, dt=0.01)
        rec = c.commit()
        assert c.version == tick + 1, \
            f"C21 版本号错: tick={tick} version={c.version} 预期={tick+1}"
        assert rec.version == c.version


def test_c22_c_value_readonly():
    """C22：C 值是只读监视器，不写场、不影响守恒。"""
    cfg = FieldConfig(shape=(8, 8, 8), h=1.0 / 7, D=0.10, scheme="implicit")
    f = Field(cfg)
    c = ConsensusLayer(f)

    S0 = f.total()

    # 正常提交（无 C 值）
    c.begin_tick()
    c.submit("A", _make_wb((8, 8, 8), {(2, 2, 2): 1.0}))
    J = c.aggregate(J_env=None)
    f.apply_sources_and_evolve(J, dt=0.01)
    c.commit(c_value=0.5)  # C 值传入但不写场

    S1 = f.total()
    delta = S1 - S0
    expected = 0.01 * 1.0  # dt * ΣJ

    # C 值不改变守恒：ΔΣφ 仍只由源项决定
    assert abs(delta - expected) < 1e-8, \
        f"C22 C值影响守恒: Δ={delta:.6f}, 预期={expected:.6f}"

    # records 里 c_value 被记录了
    assert c.records[-1].c_value == 0.5, \
        f"C22 c_value 未记录: {c.records[-1].c_value}"
