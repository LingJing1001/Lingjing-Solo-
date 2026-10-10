# 灵境引擎 V14.2 变更记录 — 「契约底座」

> 版本主题：**从"靠注释约束"升级为"靠类型系统与运行时校验约束"**。
> 不堆新功能，只做让"别人能基于它开发"成为可能的三件事：
> ① Agent 写场契约结构化  ② 场景配置 dataclass 化  ③ 彻底消除 pytest 硬依赖。

---

## 一、动机：为什么 V14.2 不做 CUDA、不做新场景

V14.1 已经证明（3/3 套件通过）：守恒机器精度、疏散 12/12 撤离、接口自检 fail-fast。
但审计暴露了**结构性盲区**——这些是"底座级"隐患，比再写一个场景更值得优先修：

| # | 隐患 | 后果 | V14.2 处置 |
|---|------|------|-----------|
| 1 | `Agent.apply_action` 返回 `Optional[ndarray]`，靠文档说"返回 J" | 子类忘了写场也不报错，守恒恒等式被悄悄破坏 | → `AgentWriteback` 结构化对象 + 钩子 `_emit_source` |
| 2 | `tests/test_v14.py` 顶层 `import pytest` | `unittest discover` 直接 ERROR；无 pytest 环境无法回归 | → pytest 改为可选，双模式运行 |
| 3 | 场景配置是裸 `dict`，缺字段到 `step()` 才 KeyError | 第三方插件调试困难 | → `ScenarioConfig` dataclass + `__post_init__` 校验 |
| 4 | 测试断言落后于契约（`test_perception_changes_after_write`） | 测的是 V14.0 "原地 inject" 语义，与 V14.2 矛盾 | → 重写为正确的因果链测试 |

**原则**：底座的可信度不来自"测试能跑"，而来自"错误写法必然被抓到"。

---

## 二、核心改动

### 2.1 `core/writeback.py`（新增）

```python
@dataclass
class AgentWriteback:
    source: np.ndarray   # 与场同 shape 的源项密度
    shape: tuple         # 显式声明 shape（防 (M,) 混入 (N³,)）

    @classmethod
    def empty(shape): ...        # 显式空写回（非静默 None）
    def add(index, value): ...  # 子类写场标准接口（含边界裁剪）

def validate_writeback(wb, field_shape) -> ndarray:
    """Engine 聚合前调用：None→空；类型错→TypeError；shape 错→ValueError(fail-fast)"""
```

**关键区分**（语义清晰化）：
- "忘记写场" = **策略**（空写回，守恒仍成立，允许）
- "写错 shape" = **错误**（立即报错，禁止）

### 2.2 `Agent` 基类重写

- `apply_action` 现在返回 `AgentWriteback`（不再 `Optional[ndarray]`）
- 新增 `_emit_source(wb)` 钩子：子类追加源项用 `wb.add((x,y,z), value)`
- **零方向 / 冻结 Agent → 显式空写回**（C7 契约）
- 新增 `bind_field(shape)`：Agent 知晓场 shape，本地即可校验

### 2.3 `Engine.step` 聚合逻辑

```python
for agent in agents:
    ...
    wb = agent.apply_action(direction, field)   # V14.2: AgentWriteback
    write_set.append(wb)
...
J_total = zeros_like(phi)
for wb in write_set:
    J_total += validate_writeback(wb, field.phi.shape)  # shape 校验点
field.apply_sources_and_evolve(J_total, dt)
```

### 2.4 `tests/test_v14.py` — pytest 可选化

```python
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
```

- 类 → 模块级函数（pytest / unittest / 裸跑三者通吃）
- 自运行入口 `if __name__ == "__main__"` 覆盖 10 项
- **验证**：`python -m unittest discover -s tests` 不再报 ImportError

### 2.5 `ScenarioConfig` dataclass（C9）

`EvacuationConfig` 示范：`__post_init__` 中
1. **先做维度校验** `assert len(shape)==3`
2. 再填默认值（exits / spawn_region）

---

## 三、本轮发现并修正的真实缺陷（2 个，均在测试中暴露）

### 缺陷 A｜`test_perception_changes_after_write` 测的是废弃语义

**症状**：`apply_action` 后 `phi[8,8,8]` 仍为 0。

**根因诊断**：V14.2 的核心设计就是 **Agent 不直接写场**（`apply_action` 只返回 wb），
写场必须经过 `apply_sources_and_evolve(wb.source, dt)`。原测试期望"apply 后 phi 立即变"，
测的是 V14.0 的"原地 inject"，与 V14.2 契约**直接矛盾**。

**修正**：重写为三段式正确因果链——
`Agent 写回 → 源项经演化注入（ΔΣφ=dt·ΣJ）→ 纯扩散一帧 → 局部 phi 传播变化`
**两段断言均通过**（注入即刻变化 + 扩散后继续变化）。

### 缺陷 B｜`ScenarioConfig.__post_init__` 校验顺序错误

**症状**：`shape=(16,16)`（2维）→ `IndexError: tuple index out of range`，
**不是**预期的 `AssertionError`。

**根因**：`self.exits = [(shape[0]-1, shape[1]//2, shape[2]//2)]` 用到了 `shape[2]`，
在维度校验**之前**执行，非法输入触发 IndexError 掩盖了校验。

**修正**：把 `assert len(shape)==3` **移到 `__post_init__` 最前**。
**验证**：`EvacuationConfig(shape=(16,16))` 现在正确抛出 `AssertionError`。

> **这两处修正的意义**：它们证明"测试能跑"不等于"测试正确"。
> V14.2 的契约守护价值就在于——错误的测试写法现在也会**自我暴露**。

---

## 四、验证结果

```
regress.py（4 套件 / 29 项断言）:

  ✅ 契约测试   test_contracts.py   9/9  (C1-C9)
  ✅ 端到端     test_e2e.py         7/7  (E1-E7)
  ✅ 接口自检   test_scenario_contract.py  3/3
  ✅ 历史覆盖   test_v14.py        10/10
                                  ─────
                                  总计 4/4 套件，29/29 通过
```

新增/强化的关键契约：

| 契约 | 内容 | 验证 |
|------|------|------|
| C7 | `apply_action → AgentWriteback`；非空写回总量 = speed；零方向 = 空写回 | ✅ |
| C8 | shape 不匹配 → `ValueError`（fail-fast）；None → 标准化空 | ✅ |
| C9 | `ScenarioConfig` 必填项构造期暴露；向后兼容 dict | ✅ |
| 因果链 | `wb.source → apply_sources_and_evolve → 感知变化` | ✅（含恒等式精确校验）|
| 守恒 | `ΔΣφ = dt·(J_env + J_agent)` 机器精度 | ✅ |

---

## 五、诚实边界（V14.2 明确披露）

1. **平台化 = 做对，不是做大**：本轮**没有**引入插件市场、没有多 Scenario 并行。
   做的是"让接口契约清晰到不可能误用"——这是任何生态的地基，必须先于生态。
2. **CUDA / 大规模尚未做**：面积律在 N≤30 小规模下加速比受 CSR 常数开销影响；
   `kernels/lingjing_stencil.cu`（V11 遗留）仍待接入。本轮**有意不做**，
   因为"契约正确性"比"规模"更前置。
3. **L4 意识层未实现**：按 IFC 文档 §7.1.1，L0-L3 已足够工程落地。
4. **C7 语义选择**："零方向 = 空写回"是**设计决策**，已明确记录。
   若未来需要"即使原地也留痕迹"，改 `_emit_source` 调用位置即可。

---

## 六、V14.2 → V14.3 候选（按优先级）

1. **多 Scenario 并行验证**：写一个非疏散场景（如 `AttractionScenario`），
   证明"换场景不改 Engine 内核"（V14.1 接口自检已铺垫，此乃真正验收）
2. **`ScenarioConfig` 纳入正式基类**：把 `EvacuationConfig` 的模式沉淀为
   `core/scenario.py` 中的 `ScenarioConfig` 抽象 dataclass
3. **CUDA 面积律终审**：编译 `.cu`，N≥128 实测斜率 ≈3 vs ≈0
4. **可观测性**：把 `TickRecord` 扩展为结构化指标流（守恒偏差实时监控）

---

*灵境引擎 · IFC 信息场范式工程 v1.0 · V14.2「契约底座」*
