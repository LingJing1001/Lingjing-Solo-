# Lingjing-Solo ARC-AGI-3 Adapter 架构

## 1. 文档状态

- 当前架构分支：`feature/arc-strategy-registry`
- 重构提交：`2276016`
- 当前已验证游戏：`LS20`
- 下一目标：R11；在 LS20 新架构验证完成后再开始
- 本次范围：重组 ARC boundary adapter，不改动 Lingjing 核心 Agent

## 2. 架构决策

不为每个 ARC 游戏创建一个相互独立的完整 solver。项目使用一个稳定的 Agent 核心，在运行时根据 `game_id` 解析对应策略：

```text
ARC FrameData / GameAction
          │
          ▼
LingjingSolo ARC Adapter
          │
          ▼
GameStrategyRegistry
       ┌──┴──┐
       ▼     ▼
 LS20Strategy  GenericStrategy
       │             │
       ▼             ▼
已验证路线       Lingjing 通用探索/规划核心
```

只有在通用探索无法提供稳定路线时，才为特定游戏增加专用策略。路线本身属于可版本化的数据产物，不等于新建一个 solver 类。

## 3. 项目结构

```text
arc_adaptor/
├── agents/
│   ├── __init__.py
│   ├── templates/
│   │   └── lingjing_solo_agent.py   # 薄 ARC boundary adapter
│   └── strategies/
│       ├── __init__.py
│       ├── base.py                  # strategy protocol
│       ├── generic.py               # 未知游戏的默认策略
│       ├── ls20.py                  # 已验证的 LS20 策略
│       └── registry.py              # game_id → strategy
├── tests/
│   ├── test_lingjing_solo_agent.py
│   └── test_action_recording.py
├── tools/
│   └── ls20_single_action_probe.py
├── patches/
│   └── arc-agent-recording.patch
├── MANIFEST.md
└── sync_to_arc.sh
```

长期路线数据计划使用以下结构：

```text
routes/
├── ls20/
│   ├── level_0.json
│   └── ...
└── <game-id>/
    └── README.md
```

当前 LS20 route 为兼容既有验证结果，仍由 Lingjing planning implementation 提供；待本 adapter 结构稳定后，再迁移为独立、可版本化的 JSON route artifact。

同步脚本不会覆盖 ARC checkout 原生的 `agents/__init__.py`，只同步 adapter、strategies、测试和工具。ARC 侧仍需确认 `LingjingSolo` 已注册，避免删除 ARC 原有 exports。

## 4. 组件职责

### 4.1 `LingjingSolo`

只负责 ARC 生命周期和边界转换：

- 将 `FrameData.frame` 转换为 numpy grid；
- 规范化 legal actions；
- 调用 Lingjing 核心的 `observe()`；
- 处理 RESET 与终止状态；
- 通过 registry 解析策略；
- 将策略返回的 action name 转换为 `GameAction`；
- 对非法或未知 action 执行安全 fallback。

adapter 不应包含 LS20 或其他游戏的 route 分支。

### 4.2 `GameStrategy`

策略边界接收当前 frame、grid、合法 action 名称和已完成关卡数，返回抽象 action name 或 `None`。

生命周期接口：

```python
strategy.reset(frame)
strategy.choose_action(
    frames,
    frame,
    grid,
    legal_names,
    levels_completed,
)
```

### 4.3 `GenericStrategy`

未知游戏的默认策略：

- 委托 Lingjing 通用核心进行探索和规划；
- 不假设特定游戏的动作序列；
- 如果核心返回非法或未知 action，由 adapter 选择第一个合法 action；
- 为新游戏提供初始探测、记录和 profile 建立入口。

### 4.4 `LS20Strategy`

只拥有 LS20 专用行为：

- LS20 默认 Level 计划；
- `LINGJING_LS20_PLAN` 显式覆盖；
- LS20 solver reset 与关卡重新播种；
- 根据已完成关卡选择已验证路线动作。

其他游戏不得依赖 `LS20Strategy`。

### 4.5 `GameStrategyRegistry`

registry 是唯一的游戏 ID 到专用策略映射位置：

```python
if game_id.startswith("ls20"):
    return LS20Strategy(...)
return GenericStrategy(...)
```

未来新增游戏时，应优先增加 profile 或 route artifact；只有通用策略无法表达时，才添加新的专用策略。不要把新的 `if/else` 写回 `LingjingSolo.choose_action()`。

## 5. 新游戏接入流程

每个新游戏按以下顺序推进：

1. **Generic 模式**：重置游戏，探测合法动作，记录 frame、action 和 state 变化。
2. **Game profile**：记录动作类别、reset 行为、坐标需求和关卡数量。
3. **Route artifact**：发现稳定路线后，以可重放数据文件保存。
4. **专用策略**：只有当状态行为或路线选择无法由通用 planner 加数据表达时才增加。
5. **重放验证**：通过同一个 adapter 比较终止状态、动作合法性和关卡进度。

该流程避免“一游戏一 solver”的碎片化，同时保留处理复杂游戏的扩展能力。

## 6. LS20 验收门槛

新架构只有在以下条件全部满足时才视为通过：

- registry 将 `ls20-9607627b` 解析为 `LS20Strategy`；
- LS20 reset 能加载预期的 Level 0 计划；
- `LINGJING_LS20_PLAN` 显式覆盖仍然有效；
- 每个输出 action 都合法，或能安全 fallback；
- terminal state 仍能正确识别；
- 官方 ARC harness 完成 LS20 跑测并返回可观察结果；
- 应用可选 recording patch 后，recording 测试仍然通过。

R11 和其他新游戏不属于本次 LS20 验收范围。

## 7. 失败与安全行为

- legal action 列表为空：返回 RESET；
- 未知 game ID：使用 `GenericStrategy`；
- 策略返回未知 action：使用第一个合法 action；
- 复杂 action 缺少坐标：使用现有 neutral coordinate payload；
- recording patch 已应用：不要重复 apply，先检查目标 diff；
- bundle 不保存外部凭据、API key 或 secret。

## 8. 同步到 ARC checkout

在 ARC-AGI-3-Agents checkout 中运行：

```bash
bash ../Lingjing-Solo-/arc_adaptor/sync_to_arc.sh .
```

脚本会同步：

- `agents/templates/lingjing_solo_agent.py`；
- `agents/strategies/`；
- adapter tests；
- 调试工具。

脚本不会覆盖 ARC 原生 `agents/__init__.py`。同步后需确认 ARC 侧注册了 `LingjingSolo` 和 `lingjingsolo`。

## 9. 验证命令与实际证据

```bash
uv run pytest -q tests/unit/test_lingjing_solo_agent.py
uv run pytest -q \
  tests/unit/test_lingjing_solo_agent.py \
  tests/unit/test_action_recording.py
uv run python main.py \
  -a lingjingsolo \
  -g ls20-9607627b \
  -t restructure,ls20-reverify
```

最新真实 LS20 Scorecard：

```text
Scorecard: 904b4c76-f8bf-44fa-a218-b4cd665060f6
score: 100.0
state: WIN
levels_completed: 7
actions: 309
resets: 0
completed: true
```

Scorecard 地址：

https://arcprize.org/scorecards/904b4c76-f8bf-44fa-a218-b4cd665060f6

单元测试和官方线上跑测都必须有实际输出；仅有 `exit 0` 或生成 Scorecard 不能代替关卡通过证据。
