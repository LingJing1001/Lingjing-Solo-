# Lingjing EtherealRealm-Solo · 封闭无 LLM 模式

## 方案评审结论：**方向正确，需 ARC-AGI-3 语境修正**

你的双模式 + 符号 advisor 替换 LLM 的设计 **完全适合 Kaggle**（`enable_internet: false`、无 API、纯 CPU）。

### ✅ 正确的部分

| 你的设计 | 评价 |
|---------|------|
| `USE_LLM_ADVISOR` 开关 | ✅ 已实现为 `use_llm_advisor` |
| advisor 双模式、文件名不变 | ✅ |
| reflector 触发 + 猜想耗尽终止 | ✅ |
| RHA 剪枝 + Top-N 上限 | ✅ |
| 6 文件分工不变 | ✅ |
| 封闭 = 无 GPU 大模型 | ✅ 符合 Kaggle 约束 |

### ⚠️ 必须修正：ARC-AGI-3 ≠ 静态 ARC 网格题

文档里「输入输出样例对、旋转/翻转/平移整张网格」适用于 **ARC-1/2**。

**ARC-AGI-3 是交互游戏：**

| 静态 ARC | ARC-AGI-3 |
|---------|-----------|
| 给定 train input→output | 无训练样例，只有逐步 play |
| 一次性输出网格 | 200 步内选 ACTION1–6 |
| 变换算子作用于整图 | **动作效应规则** + 点击坐标 + 关卡状态机 |

封闭模式符号扩增的对象应是：

- 转移链组合（`ACTION1;ACTION4 @ state`）
- 动作序列宏（高 effect_ema 动作排列）
- 未尝试动作 / 点击坐标扰动
- 从 `recent_transitions` 差分挖掘（非 IO 样例对）

### 已实现（v0.4.2）

```python
# config.py
use_llm_advisor: bool = False          # Kaggle 默认封闭
symbolic_advisor_max_rounds: int = 6
symbolic_max_hypotheses: int = 12
```

`advisor.py` → `SymbolicHypothesisExpander`：转移差分、动作组合、宏序列、剪枝。

### 能力取舍（与你文档一致）

- **优势**：零 LLM、可内网部署、CPU 即可
- **短板**：无 LLM 抽象能力；ls20 旋转/形状状态机仍需专用逻辑
- **开放版**：`use_llm_advisor=True` + `inject_llm(fn)` 恢复 LLM 分支

### 与「谷歌必须跑大模型」对比

**不准确。** ARC-AGI-3 比赛不强制 LLM；榜首方案各异。灵境封闭版的优势是 **可部署性**，不是「比赛禁止神经网络」。

## 配置

```python
SoloConfig(use_llm_advisor=False)   # Kaggle / 封闭（默认）
SoloConfig(use_llm_advisor=True, llm_calls_per_game=8)  # 开放联网
```
