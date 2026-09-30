# Lingjing-Solo · 单 Agent 世界模型场架构

> 把「灵境引擎」从**多 Agent 协同**重构为**单 Agent 世界模型**，对齐 **ARC-AGI-3** 的隐藏、回合制、零指令、步数敏感设定。

## 重构映射（多 Agent → 单 Agent）

| 原灵境引擎组件 | Lingjing-Solo 化身 | 说明 |
|---|---|---|
| 统一信息场 Φ | `WorldModelField` (Layer 1) | 从「多 Agent 共享介质」→「单 Agent 内部世界模型」 |
| 泡壁局部高精度 | ROI 差分感知 (Layer 0) | 64→128 维特征，避免每步全量 LLM |
| 版本化因果链 | 转移表 `(s,a)→s'` + 状态哈希 | 同 `(s,a)` 矛盾后继 → 规则降级 |
| 多体共识 / 消息总线 | **丢弃** | 单 Agent 无消息 |
| 面积律降算力 | **转义**为「决策算力节制」 | LLM 调用预算 + 轻量规划 |
| ConsciousnessMonitor | ReflectionTrigger (Layer 4) | 循环 / 冲突 / 预算告急 → LLM 反思 |

## 目录结构

```
lingjing_solo/
├── core/             # 配置、类型、工具（无业务依赖）
│   ├── config.py     # SoloConfig：所有超参集中
│   ├── types.py      # Frame / RuleHypothesis / Transition / FieldSnapshot ...
│   └── utils.py      # hash_grid / 连通域 / 日志
├── perception/       # Layer 0：帧差分 + 分割 + CNN/直方图编码
├── world_model/      # Layer 1：Φ 场（转移表、规则置信度、循环检测）
├── exploration/      # Layer 2：信息增益评分、规则归纳、目标推断
├── planning/         # Layer 3：轻量 BFS + LLM 战略顾问（预算节制）
├── reflection/       # Layer 4：三类反思信号 + Φ 摘要打包
├── harness/          # Kaggle 烟囱口适配（is_done / choose_action）
└── agent.py          # 五层编排 + reset + WIN 检测 + 硬步数上限
```

## 决策优先级（每步 choose_action）

```
1. WIN 检测 → 结束
2. 反思触发？  → LLM 战略重估（受每关硬预算节制）
3. 短程规划可解？ → 轻量 BFS/A*
4. 否则 → 探索引擎信息增益评分（贪心）
```

## RHAE 效率对策

- **动作效率预估**：`ExplorationEngine.score_actions` 用「单位步数信息增益」对冲平方惩罚
- **循环检测**：`visited_set` + `is_loop()` 直接避免重复访问
- **LLM 预算**：默认每关 8 次，评测期归零走纯轻量路线

## 快速开始

```bash
python test_solo.py        # 跑通五层闭环测试
python notebook_template.py # 本地自检（模拟 Kaggle 评测）
```

提交 Kaggle：把 `lingjing_solo/` + `notebook_template.py` 上传，在官方 Starter 的 `agent/my_agent.py` 里：

```python
from notebook_template import MyAgent
```

## 待补全的占位（明确标注）

- `WorldModelField.detect_win`：WIN 判定需按具体环境注入（官方提供 `is_win` 反馈时可闭环）
- `LightweightPlanner.search`：有限深度 BFS 骨架已就位，转移图搜索待填充
- `LingjingCNN`：带 torch 时启用神经编码，否则自动降级直方图特征
- LLM 调用函数：评测期无网络，默认不注入；本地调试用 `LingjingSoloAgent.with_llm(fn)`

## 关键设计抉择

1. **LLM 退居战略顾问** —— 纯 LLM 在 ARC-AGI-3 上 <1%，故仅在反思触发时调用
2. **剥离 IFC 宇宙学假说** —— 本赛道是工程基准，假说与工程事实分章呈现
3. **零样本迁移优先** —— 框架不记忆具体关卡规则，只学「如何探索未知规则」的元策略
