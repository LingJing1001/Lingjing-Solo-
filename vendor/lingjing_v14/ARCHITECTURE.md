# 灵境引擎 V14 架构（IFC 信息场范式 · 工程 v1.0）

> "不是做更大的引擎，而是做更对的底座——让错误写法必然被抓到。"

---

## 一、分层原则（IFC L0–L3 映射）

```
┌─────────────────────────────────────────────────────────────┐
│  L5  应用层（场景插件）                                       │
│       EvacuationScenario / (未来) WarehouseScenario ...       │
│       只定义：场初始化 · Agent 生成 · 感知构造 · 环境源项       │
├─────────────────────────────────────────────────────────────┤
│  ★ 契约层（V14.2 核心）  ← 本层是"底座"的真正边界             │
│       Scenario 接口 + validate()      ← 场景插件自检          │
│       AgentWriteback + validate_writeback ← Agent 写场契约   │
│       ScenarioConfig (dataclass)       ← 配置构造期校验       │
├─────────────────────────────────────────────────────────────┤
│  L1–L2  引擎内核                                             │
│       Engine（因果闭合主循环）                                 │
│         感知 → 决策 → apply_action(返回 wb) → 即时撤离         │
│         → scenario.step(环境源项) → 聚合 J → 演化（唯一写入点）│
│       Field（守恒双通道：inject_sources / apply_sources_and_evolve）│
│       Laplacian（体积 7点 O(N³) / 泡壁 CSR O(R²)）            │
├─────────────────────────────────────────────────────────────┤
│  L0  离散基元层                                              │
│       phi (ndarray) · FieldState · TickRecord · 持久化/重放    │
└─────────────────────────────────────────────────────────────┘
```

**关键**：内核（L0–L2）**不感知**具体场景；应用（L5）**不直写**场。
两者仅靠"契约层"解耦——这就是平台化的正确粒度：内核不做大，接口不误用。

---

## 二、写场契约（V14.2 结构化，取代裸 ndarray）

**演进**：
```
V14.0  apply_action → Optional[ndarray]    # 靠注释，忘了也不报错
V14.2  apply_action → AgentWriteback       # 类型系统 + 运行时校验
```

**单 tick 因果链**（守恒恒等式 ΔΣφ = dt·ΣJ 的保证路径）：

```
Agent.decide() ──方向──▶ Agent.apply_action()
                              │  ① 位移（连续坐标 + 边界反射）
                              │  ② _emit_source(wb) 钩子
                              │     EvacuationAgent: wb.add(pos, speed)
                              ▼
                         AgentWriteback(source, shape)
                              │
Engine.step() ──聚合──▶ validate_writeback(wb, field.phi.shape)
                              │   • None    → 空（向前兼容）
                              │   • 类型错  → TypeError
                              │   • shape错 → ValueError (fail-fast)
                              ▼
                         J_total (ndarray, 同 shape)
                              │
                              ▼
                   Field.apply_sources_and_evolve(J_total, dt)
                              │
                              ▼
                    ΔΣφ = dt · (J_env + ΣJ_agent)   ← 守恒恒等式
```

**两种"空"的语义区分**（设计要点）：
| 情形 | 表现 | 含义 |
|------|------|------|
| 冻结 / 零方向 Agent | `AgentWriteback.empty` | **策略**：允许，守恒仍成立 |
| shape 不匹配 | `validate_writeback` 抛错 | **错误**：禁止，立即暴露 |

---

## 三、平台化的最小接口（V14.2 已实现）

第三方场景插件只需满足：

```python
class MyScenario(Scenario):
    def setup(self, field): ...
    def create_agents(self, field) -> list[Agent]: ...
    def step(self, field, agents, dt) -> ndarray | None: ...
    def observation(self, field, agent, all_agents=None) -> dict: ...
    # validate() 自动校验：observation 签名 + exits/shape 属性
```

Agent 插件只需：
- 继承 `Agent`，实现 `perceive` / `decide`
- （如需写场）重写 `_emit_source(self, wb)`，调用 `wb.add(...)`

**自检机制**（fail-fast 示例）：
```python
# 错误签名 → Engine 构造即报错，而非跑到一半崩
class BadScenario(Scenario):
    def observation(self, field, agent):  # 缺 all_agents → 报错
        ...
# ScenarioInterfaceError: observation 签名不满足契约...
```

---

## 四、验证体系（4 层 / 29 项）

| 层 | 文件 | 项数 | 覆盖 |
|----|------|------|------|
| 物理契约 | test_contracts.py | 9 | C1–C9（守恒 / 面积律 / **写场契约** / **shape fail-fast** / **config dataclass**）|
| 端到端集成 | test_e2e.py | 7 | E1–E7（守恒 / 撤离 / 质心 / 拥挤 / 软分离 / 确定性 / 边界）|
| 接口自检 | test_scenario_contract.py | 3 | 场景插件 fail-fast |
| 历史覆盖 | test_v14.py | 10 | 兼容 pytest + 裸跑（pytest 可选）|

入口：`python regress.py`（退出码供 CI）

---

## 五、诚实边界

1. **平台化 ≠ 做生态**：当前是"接口稳定、可被第三方正确实现"，尚未有第二方场景。
   V14.3 的**真正验收**是"不修改 Engine，跑通第二个场景"。
2. **规模**：N≤30，CSR 常数开销占比大；CUDA（`kernels/lingjing_stencil.cu`）待接入。
3. **L4 意识层**：未实现（IFC 文档 §7.1.1：L0–L3 已可工程落地）。

---

*V14.2「契约底座」· 2026*
