# ARC-AGI-3 竞赛详解与灵境-Solo 项目评估报告

---

## 一、什么是 ARC-AGI-3？

这是一个**交互式推理基准测试**（Interactive Reasoning Benchmark），而非传统的静态数据集问题。你的 Agent 必须在一个个隐藏的游戏中：

- 接收**帧**（Frames）— 包含 64×64 网格的 JSON 对象，网格中 0–15 的整数值表示不同的颜色/状态
- 通过执行**动作**与游戏环境交互
- 每个游戏包含多个难度递增的关卡
- 每个游戏可处于三种状态之一：`NOT_FINISHED`（未完成）、`WIN`（胜利）、或 `GAME_OVER`（游戏结束）

与 ARC-AGI-2 的关键区别在于：**动作是隐藏的、需要探索发现的**。Agent 必须通过探索来理解每个动作的含义，这使它成为一个真正的**探索 + 推理**问题。

---

## 二、如何提交并获得测试分数

### 第一步：获取 API 密钥

访问 [arcprize.org/platform](https://arcprize.org/platform) 注册并获得 `ARC_API_KEY`。这将使你可以在发布后访问公开游戏。

### 第二步：搭建开发环境

```bash
git clone https://github.com/arcprize/ARC-AGI-3-Agents.git
cd ARC-AGI-3-Agents
cp .env.example .env
# 编辑 .env：设置 ARC_API_KEY="你的密钥"
```

### 第三步：实现你的 Agent

你需要创建一个继承自 `Agent`（抽象基类）的类，并实现以下两个方法：

```python
from agents import Agent
from arcengine import FrameData, GameAction

class MyAgent(Agent):
    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        # 当游戏结束时返回 True
        return latest_frame.state == "WIN" or latest_frame.state == "GAME_OVER"

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        # 根据历史记录决定下一个动作
        # latest_frame.available_actions 中列出了当前可用的动作
        # latest_frame.state 显示当前游戏状态
        return GameAction.ACTION1  # 示例
```

Agent 基类提供的接口：
- `frames` 列表 — 完整的帧历史
- `latest_frame` — 当前状态（包含 `.state`、`.levels_completed`、`.available_actions`、`.grid`）
- `action_counter` — 当前步骤计数（最大 80 步，防止无限循环）
- `take_action(action)` — 向环境发送动作，返回下一帧 FrameData

### 第四步：本地运行（获取测试分数）

```bash
# 安装依赖：uv sync
uv run main.py --agent=myagent --game=ls20
```

这将在 ls20 游戏上运行你的 Agent，生成一个**分数卡**（Scorecard），其中包含每个关卡的分数。运行结束后会显示分数卡的在线 URL。

### 第五步：在 Kaggle 上提交

在 [Kaggle 竞赛页面](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3)上：
- "Code" 标签页提供 Notebook 支持
- 只要 Agent 在任何游戏中执行了至少一个动作，就会为所有游戏自动生成提交文件
- 也可以通过 [Google 表单](https://forms.gle/wMLZrEFGDh33DhzV9) 提交

---

## 三、评分机制

### 每关评分：0% – 100%

- **100%** 代表：
  1. Agent **解决了**该关卡（达到 WIN 状态）
  2. Agent 使用的**动作数与人类基准一致**

- 分数**上限为 100%** — 即使用更少的步数解决了关卡，也不会超过 100%

### RHAE（Relative Human Efficiency，相对人类效率）

评分会惩罚：
- **未解决**：该关卡 0% 分
- **比人类多用步数**：分数降至 100% 以下（多余步数有平方惩罚）
- **比人类少用步数**：上限仍为 100%，不超额

### 最终总分

**最终分数**是所测试所有关卡所有游戏每关分数的**平均值**。

### 排行榜

- 公共排行榜使用约 **50% 的测试数据** — 最终排名基于剩下的 50%
- 当前最高分：
  | 排名 | 队伍 | 分数 | 条目数 |
  |------|------|------|--------|
  | 1 | cstl | 5.99 | 37 |
  | 2 | Lord Han Solo | 4.99 | 45 |
  | 3 | Tufa Labs | 4.67 | 120 |
  | 4 | Tong Hui Kang | 3.88 | 55 |
  | 5 | rfbr | 3.37 | 13 |

> **注意**：当前最高分仅为 ~6%，说明这个基准测试难度极高，不是简单的规则匹配就能解决。

---

## 四、上榜最低要求

- **必须**在环境中设置 `ARC_API_KEY`
- **必须**在 `Agent` 类上实现 `is_done()` 和 `choose_action()`
- **无需最低分数** — 任何在任意游戏中至少执行一个动作的提交都会生成提交文件并出现在排行榜上
- 最基础的门槛是一个随机 Agent（随机选择动作），它几乎得 0 分，但仍会注册上榜

---

## 五、灵境-Solo 项目与 ARC-AGI-3 的对应关系

你的项目架构实际上与 ARC-AGI-3 的需求相当吻合：

| ARC-AGI-3 需求 | 灵境-Solo 匹配度 | 说明 |
|---|---|---|
| 需要探索的隐藏动作 | ✅ 匹配 | Layer 2 探索引擎（信息增益评分） |
| 多步推理 | ✅ 匹配 | Layer 3 规划 + Layer 1 世界模型（转移表） |
| 状态记忆 / 追踪 | ✅ 匹配 | Layer 1 Φ 场（已访问状态集合、循环检测） |
| 效率（RHAE） | ✅ 匹配 | LLM 预算控制、防循环、步数经济 |
| WIN 检测 | ❌ 未实现 | `detect_win()` 目前是一个占位函数（始终返回 False） |

---

## 六、项目当前存在的问题与改进方向

### 严重问题

### 1. 动作词汇表不匹配

Agent 的动作词汇（`UP`、`DOWN`、`LEFT`、`RIGHT`、`SPACE`）是硬编码的，不会匹配 ARC-AGI-3 实际使用的 `ACTION1` 到 `ACTION7` 接口。

**需要：**
- 映射到 `arcengine` 库中的 `GameAction` 对象
- 使用 `latest_frame.available_actions` 来确定每一步哪些动作是合法的
- 处理基于坐标的动作（`ACTION6` 需要 `(x, y)` 坐标）
- 实现真正的 `WIN` 检测 — `latest_frame.state == "WIN"` 才是正确判据

### 2. `LightweightPlanner.search()` 完全未实现

目前 BFS 搜索是一个硬编码的 **no-op**（始终返回 None）。这意味着 Agent 在每一步都会退化到探索评分（本质上是随机选择）。

**需要实现：**
- 基于转移表的 BFS 搜索，在 `(state_hash, action) → next_state` 图上工作
- 利用目标假设评估叶子节点的贴近度
- 这是目前最大的功能缺失

### 3. `WorldModelField.detect_win()` 始终返回 False

这是一个占位实现，WIN 检测需要根据具体环境注入更精细的判定时。

### 4. 网络依赖问题

项目依赖 `torch` 时才能使用神经 CNN 编码。评测期无网络、无 GPU，必须确保 fallback 特征系统完全可用。

### 5. `notebook_template.py` 中的类继承错误

```python
class MyAgent(MyAgent):  # ← 这是从自身继承，会报错
```

应该继承 `MyAgent` 从 `kaggle_adapter.py` 导出的适配类。

---

## 七、改进优先级建议

### 优先级最高（P0 — 必须做）

1. **实现真正的 `WIN` 检测**
   - 最简单的实现：检查 `state == "WIN"` 或 `levels_completed` 是否增加
   - 这需要接入 ARC-AGI-3 环境才能验证

2. **实现 BFS 规划搜索**
   - 在 `transition_table` 上实现有限深度的 BFS
   - 使用 `(state_hash, action) → next_state` 的转移图
   - 这将为 Agent 提供真正的规划能力

3. **实现探索动作评分的差异化**
   - 当前所有动作在信息不足时得分相同（都是 1.0）
   - 需要实现 action-specific 的探索评分（例如，模拟动作效果）

4. **适配 ARC-AGI-3 动作空间**
   - 使用 `available_actions` 从 `latest_frame` 动态获取合法动作
   - 处理 `ACTION6` 的坐标输入
   - 处理 `ACTION7`（额外简单动作）

### 高优先级（P1 — 强烈建议）

5. **实现具体的 WIN 启发式规则**
   - 对于 LS20 等具体游戏，可能需要实现针对性的 WIN 检测逻辑
   - 例如：颜色计数、位置匹配、目标形状识别

6. **接入实际的 ARC-AGI-3 工具运行测试**
   - 将 Agent 作为 `Agent` 的子类，运行 `main.py --agent=... --game=ls20`
   - 收集真实的分数卡数据

7. **增强 LLM 反思的质量**
   - 当前 LLM 预算为 8 次/关卡，评测期为 0
   - 反思信号应包含更多上下文（如具体游戏特征、已归纳规则等）

### 中优先级（P2 — 锦上添花）

8. **添加具体的游戏规则归纳**
   - 目前规则归纳仅从高频转移中提炼（过于简单）
   - 可以加入模式匹配、因果推理等策略

9. **实现多关卡迁移学习**
   - 跨关卡复用已学习的规则和特征
   - 这符合 ARC-AGI 对"快速学习"的核心要求

---

## 八、具体实施建议

### 短期目标（1-2 周）

1. 在 DGX 上搭建完整的 ARC-AGI-3 开发环境
2. 将 `LingjingSoloAgent` 改写成 `Agent` 的子类
3. 实现最小可用的 `choose_action`（随机合法动作）
4. 在 ls20 游戏上运行并获取初始分数

### 中期目标（1 个月）

1. 实现 BFS 搜索，使 Agent 能够进行有限深度的规划
2. 实现具体的 WIN 检测逻辑
3. 在 5-10 个公开游戏上测试并优化

### 长期目标（竞赛期间）

1. 实现多模态感知（如果评测支持视觉输入）
2. 实现 LLM 驱动的深层推理（如果本地环境允许）
3. 持续优化探索效率和步数经济

---

## 九、项目当前状态总结

| 组件 | 状态 | 完整性 |
|---|---|---|
| 架构设计 | ✅ 已完成 | 五层框架清晰 |
| 感知层 (L0) | ✅ 已完成 | 帧差分 + 连通域分割 + 特征编码 |
| 世界模型 (L1) | ✅ 基本完成 | 转移表 + 规则假设 + 循环检测 |
| 探索引擎 (L2) | ⚠️ 部分完成 | 框架存在但评分不够差异化 |
| 规划执行 (L3) | ❌ 未实现 | BFS 搜索是 no-op |
| 反思触发 (L4) | ✅ 基本完成 | 三类信号检测功能完整 |
| Kaggle 适配 | ⚠️ 需修复 | 类继承错误，动作空间不匹配 |
| 测试覆盖 | ✅ 完整 | 7 个测试全部通过 |
| WIN 检测 | ❌ 未实现 | 占位函数 |

**总体评估**：框架架构优秀，但核心功能（规划搜索、WIN 检测）缺失，且未与 ARC-AGI-3 实际 API 对接。下一步应该优先实现规划搜索和适配实际的游戏 API。
