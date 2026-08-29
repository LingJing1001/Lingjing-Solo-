# Lingjing-Solo 项目计划与状态

> 最后更新：2026-08-29T10:54:03-07:00
> 状态权威：本文档用于跟踪实现里程碑及其验证证据。
> 范围：`Lingjing-Solo-` 核心 package 及其 ARC-AGI-3 adaptor 集成。

## 状态标记

- `[x]` 已完成，并有可执行证据支持。
- `[ ]` 未完成，或尚未完成验证。
- `P0` 阻塞项 / 最高优先级。
- `P1` 重要后续工作。
- `P2` 后续优化。

## 当前状态摘要

- Package、ARC adaptor、状态注入、基础 planner 和 LS20 安全执行框架已完成，并有本地测试证据。
- 另一个 thread 已完成 ARC recording 可观测性修复和四个真实 LS20 单动作 probe；现在能区分 Agent 实际发送的动作与服务端回传的 RESET 字段。
- `exploration/action_diff.py` 已完成单动作差分、fail-closed 汇总，以及从 ARC JSONL recording 读取有序多动作差分；Level 1 的 ACTION1–4 方向和胜利机制已由官方 source 与远程 E2E 共同验证。
- 最新真实结果已达到 `1/7`、`3.571428571428571`；Level 1 得分 `115.0`，15 actions 完成。提交链路和 Level 1 R4 策略已闭环，Level 2–7 尚未解决。
- 当前主要阻塞是 Level 2–7 的目标参数、开关路线和动态重规划；不是 Level 1、recording 可观测性、安装或运行链路故障。
- 当前 action 顺序：Level 2–7 source 规则反推 → 每关离线 BFS → 真实 E2E → 动态平台在线重规划。

## 已完成的基础工作

- [x] 创建可安装的 `lingjing_solo` Python package。
  - 证据：`pyproject.toml`；`uv build` 成功生成 `dist/lingjing_solo-0.1.0-py3-none-any.whl` 和 source archive。
- [x] 添加 package 安装和使用文档。
  - 证据：根目录 `README.md` 已说明 editable 安装、wheel 安装、ARC adaptor 配置、测试和可选 CNN 依赖。
- [x] 将项目许可证从 MIT 改为 MIT-0。
  - 证据：`LICENSE` 以 `MIT No Attribution` 开头；`pyproject.toml` 声明 `license = "MIT-0"` 并包含 `LICENSE`。
- [x] 更新 ARC adaptor，使其导入已安装的 package。
  - 证据：`../ARC-AGI-3-Agents/agents/templates/lingjing_solo_agent.py:14` 使用 `from lingjing_solo import LingjingSoloAgent`。
- [x] 在 ARC agent registry 中注册 adaptor。
  - 证据：`../ARC-AGI-3-Agents/agents/__init__.py` 注册了 `lingjingsolo` template。
- [x] 添加 ARC 帧转换和动作规范化单元测试。
  - 证据：`../ARC-AGI-3-Agents/tests/unit/test_lingjing_solo_agent.py`；此前验证结果：`5 passed`。
- [x] 验证干净环境中的 package import。
  - 证据：干净虚拟环境返回 `clean_import=LingjingSoloAgent`。
- [x] 验证 Lingjing-Solo package 测试套件。
  - 证据：`uv run pytest -q` 返回 `21 passed, 1 warning`；Hermes 环境的默认 pytest 插件加载会触发既有 Hydra/dataclass 兼容错误，使用 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 后通过。
- [x] 验证已检查的 package 入口 lint。
  - 证据：`uv run ruff check lingjing_solo/__init__.py --select E,F --ignore I001` 返回 `All checks passed!`。
- [x] 使用 ARC adaptor 完成端到端运行验证。
  - 证据：2026-08-27 ARC 运行退出码为 0，生成 scorecard `36d36853-2560-43b6-9b66-7f90992c1c0b`，执行 81 个动作；得分为 0.0，7 关完成 0 关，说明集成可运行但尚未解决游戏。
- [x] 将 ARC `GameState` 和 `levels_completed` 注入 package agent（兼容没有 `observe()` 的旧 fake agent）。
  - 证据：`lingjing_solo/agent.py` 新增 `observe()`；ARC adaptor 调用 `getattr(..., "observe")`；adaptor 单元测试 `5 passed`。
- [x] 让 package agent 在权威状态为 `WIN` / `GAME_OVER` 时终止，并在关卡数增加时重置内部模型。
  - 证据：`test_agent_honors_authoritative_terminal_state` 通过；package 测试 `10 passed`。
- [x] 实现基于已观测转移的安全短程 planner，并将合法动作集合传入 planner。
  - 证据：`test_planner_returns_unvisited_known_transition` 通过；`planner.py` 已不再无条件返回 `None`。
- [x] 为重复动作增加按当前状态衰减的信息增益评分。
  - 证据：`test_exploration_penalizes_repeated_action_for_current_state` 通过。

## 当前里程碑概览

- **基础集成：** 已完成。Package、ARC adaptor、状态注入、终止状态和基础 planner 均有测试证据。
- **LS20 solver：** Level 1 语义闭环已完成；已具备对象提取、动作合法性过滤、waypoint 路线、动态障碍安全接口，以及经过远程 ARC 验证的 15 步路线。
- **动态障碍：** 本地近邻阻挡 fixture 已通过；真实 adaptor 已能识别 color-1 玩家候选并传递坐标，但尚未证明跨帧/跨关卡鲁棒性。
- **真实结果：** scorecard `9aae4d01-d506-4f84-ae8c-cd72000cc28c` 为 `1/7`、`3.571428571428571`、Level 1 `115.0` 分、15 actions；总运行 81 actions 后进入 `GAME_OVER`。
- **当前结论：** 工程集成和 Level 1 机制已验证；剩余工作是把同样的 source 规则反推/路线规划方法扩展到 Level 2–7。
- **下一里程碑：** 为 Level 2–7 提取玩家/目标/开关参数，完成每关低于对应 baseline 的合法路线；之后推进全关动态重规划。

## 当前主要阻塞点与改进方向

### 阻塞点 1：Level 1 的真实动作语义尚未闭环

- 已知官方动作集合是 `[1, 2, 3, 4]`；历史 recording 的服务端 `action_input.id` 全为 `0`（RESET），不能作为 Agent 出站动作证据。
- 已完成：新增 `tools/ls20_single_action_probe.py`，逐个 reset 后只执行一个动作；真实 probe 显示 ACTION1/3/4 改变下方对象 52 格，ACTION2 改变 2 格，四者均未移动 color-1 marker 或触发换关。
- 已完成：ARC Agent recording 增加 `requested_action` 字段，区分 Agent 实际发送动作与服务端回传 `action_input`。
- 已完成：新增 `lingjing_solo/exploration/action_diff.py`，提供单动作前后帧差分、color-1 玩家质心位移、状态/关卡变化记录，以及按动作聚合的一致性/置信度汇总；多通道歧义帧 fail-closed。
  - 证据：`test_action_diff_records_single_action_evidence`、`test_action_diff_summary_is_fail_closed_on_inconsistent_motion`、`test_action_diff_rejects_ambiguous_multichannel_frame` 通过。
- 已完成：使用带 `requested_action` 的真实多步 recording 验证动作序列可追踪；暂不把变化区域直接解释为四方向移动。
  - 证据：`../ARC-AGI-3-Agents/recordings/ls20-9607627b.lingjingsolo.80.8f2c2270-bdb7-4bd6-85a8-af27c2a3d155.recording.jsonl`，`81` 帧、`ACTION1–4` 均出现，全部 `NOT_FINISHED`、`levels_completed=0`。
- Level 1 已完成；下一步转入 Level 2–7 的玩家、目标、开关参数和路线反推。

**当前 action item（P0）**

- [x] 让 ARC recording 保存 Agent 实际请求动作 `requested_action.name/id`。
- [x] 对 ACTION1–4 执行 reset 后单动作 probe，并保存前后帧差异。
- [x] 提供可复用的单动作差分和按动作聚合 API。
- [x] 采集带 `requested_action` 的有效多步 recording。
  - 证据：scorecard `21661d45-af0e-4e56-b385-734b9574f23e`；`81` actions，recording `...8f2c2270...recording.jsonl`；分析 API 返回 `80` 个有序 delta，覆盖四种动作。
- [x] 提供可复用的 ARC JSONL 多步 recording 分析 API。
  - 证据：`lingjing_solo/exploration/action_diff.py:140-190` 的 `analyze_recording()`；支持混合 `1xHxW` 与显式 `frame_channel=5` 的多通道帧，缺少 `requested_action` 时显式失败。
- [x] 分离玩家、目标、平台/交互对象的变化，并确认 Level 1 的胜利触发边界；已由官方 source 与远程 scorecard 验证。
- [x] 仅在重复实验一致后固化 Level 1 的 ACTION1–4 方向/语义映射；`ACTION1=up`、`ACTION2=down`、`ACTION3=left`、`ACTION4=right`。

### 本轮实验结论（2026-08-28）

- [x] `ACTION1×7` 隔离实验：前 5 次使中部对象沿纵向移动约 `5 px/次`，随后达到边界；与顶部对象接触/重叠后仍为 `NOT_FINISHED`、`levels_completed=0`。
  - 证据：scorecard `340333d0-c0c2-4724-a45d-ea2b814e91e7`；recording `../ARC-AGI-3-Agents/recordings/ls20-9607627b.lingjingsolo.80.85746bc7-dd36-423d-bbe1-afbdacda2456.recording.jsonl`；`81` actions、`0/7`、`0.0`。
- [x] `ACTION3×3` 隔离实验：中部对象横向移动候选未触发 Level 1。
  - 证据：scorecard `0e5d2c29-20dc-4b84-9aab-17cc8c06def6`；recording `../ARC-AGI-3-Agents/recordings/ls20-9607627b.lingjingsolo.80.6a1a13b6-c20c-4082-9f80-9ad19e66337a.recording.jsonl`；`81` actions、`0/7`、`0.0`。
- [x] color-1 玩家候选在 `ACTION1×7` recording 的全部观测中保持 2 像素、质心 `(x=20.5,y=32.5)`；因此中部移动对象不能标记为玩家。
- [x] `ACTION1×5 + ACTION3×3` 组合实验：纵向到 `y≈20` 后，ACTION3 未产生预期横向位移，仍为 `NOT_FINISHED`。
  - 证据：scorecard `673b40c3-4b70-477f-9e82-134109e206ee`；recording `../ARC-AGI-3-Agents/recordings/ls20-9607627b.lingjingsolo.80.097a3230-a323-4b33-8281-1eda95dc75ea.recording.jsonl`；`81` actions、`0/7`、`0.0`。
- [x] 位置条件横向动作对照：初始位 ACTION3 有效；中位（ACTION1×2）无效；高位（ACTION1×4）重新有效。
  - 证据：scorecard `8ed6b678-10a7-4038-98a2-2f22d03dac8e`、`b8dd08ca-027c-4720-b851-192c8606df43`、`d58bc004-8889-462d-b2f2-bedacd4bfa99`；三组均 `0/7`、`0.0`、`NOT_FINISHED`。
- [ ] 下一实验：围绕 color-12 对象的可用动作边界建立最小状态转移表，并单独验证颜色/形状变化是否与动作状态相关；仍保持默认策略和方向映射不变。
- [x] 精确 80 步 R4 组合矩阵已运行：`proc_326207fe21ae`；checkpoint `/tmp/lingjing_r4_exact80_20260828/checkpoint.jsonl`；heartbeat `/tmp/lingjing_r4_exact80_20260828/heartbeat.log`。
- [x] 精确 80 步矩阵已完成：6/6 候选均 `0.0`、`0/7`、`NOT_FINISHED`；离线事件分析显示 `down_hold_then_left` 使 color-12 与动态 color-9 最小质心距离约 `6.92`，但未触发胜利；color-1 在实验中保持静止。
- [x] 接触后交互动作对照已完成：以 `ACTION2×6 + ACTION3×3` 接近 color-9 后分别追加 ACTION1/2/3/4；4/4 候选均 `rc=0`、`0.0`、`0/7`、`NOT_FINISHED`。逐帧分析中 color-1 与 color-0 的 bbox 全程不变，color-12 仅发生已知位移或保持静止，未观察到颜色/形状/关卡状态变化。证据：checkpoint `/tmp/lingjing_r4_contact_20260828/checkpoint.jsonl`、heartbeat `/tmp/lingjing_r4_contact_20260828/heartbeat.log`、recordings `e788c503`/`65558a32`/`18e5c8d0`/`87371917`。
- [x] 接触 recording 隐状态扫描已完成：4/4 recording 的 `available_actions` 全程为 `[1,2,3,4]`，未发现隐藏第五动作；四案都在第 41 个 transition 附近出现 color-8 bbox 从 `(56,61,63,62)` 变为 `(56,61,60,62)`，且 color-11 移动平台持续变化。该同步事件应纳入状态模型，不能再把全部动态归因于 color-9/12。
- [x] 容错帧扫描确认对象关系：主要 recording 帧为 `(1,64,64)`，应取 `frame[0]`；ASCII 差分显示 color-8 是底部固定结构，color-11 是横向移动平台，其移动会覆盖/暴露 color-8。第 41 个 transition 是平台相位/遮挡变化，不是隐藏动作或胜利事件。
- [ ] 下一实验：停止重复“接近后追加方向键”矩阵；从首帧对象布局和连续帧差分中识别非位移交互机制或隐藏目标条件，再构造最小验证序列。
- [x] 已启动受控路线搜索：围绕 `color-12≈(34,45)` 到静态 `color-0≈(21,32)` 测试先横后纵、先纵后横、交替及平台相位错开序列；后台 checkpoint `/tmp/lingjing_r4_route_search_20260828/checkpoint.jsonl`，heartbeat `/tmp/lingjing_r4_route_search_20260828/heartbeat.log`，进程 `proc_339d0621bd5e`。
- [x] 受控路线搜索已完成：6/6 候选均 `rc=0` 但 `score=0.0`、`levels_completed=0`、`NOT_FINISHED`。逐帧证据显示 `horizontal_then_vertical` 第 5 步已将 color-12 推至 `(x≈19–23,y≈30–31)`，与静态 color-0 `(21–22,31–32)` 空间重叠，仍未触发胜利；因此 color-0 不是“接触即胜”的充分条件，停止继续围绕 color-0 做单纯曼哈顿路线搜索。
- [x] 官方 Arcade 单动作 probe 已复核动作语义：`ACTION2` 单步只改变底部 color-11 平台（变化 bbox 行 `(61,62)`），`color-1` 始终固定于 `(32,20),(33,21)`；`ACTION1/3/4` 会改变 color-9 与 color-12 区域，但四次均 `NOT_FINISHED`、`levels 0→0`。因此不能把 ACTION1–4 统一解释为直接移动玩家，ACTION2 应作为平台相位/等待候选单独建模。证据：scorecard `b57f7d81-3916-4edf-85f5-ee492e5bb6bb`、`48ec6399-42dc-4a08-81a5-122a330be04f`、`fea81086-cfff-457f-a9ab-d7c1f0d43bbf`、`eae38efe-d7f4-4979-92b5-32a3b784b581`；命令 `uv run python tools/ls20_single_action_probe.py ACTION1|ACTION2|ACTION3|ACTION4`。
- [x] 平台等待窗口实验已完成：等待 `ACTION2` 10/20/30/40 步后执行横纵移动，4/4 均 `total_actions=81`、`total_levels_completed=0`、`NOT_FINISHED`。对应 scorecard：`c30ef3e1-7067-4a5c-a30b-41e4cb1be18c`、`110a62f2-a663-47c1-b9e7-2e5c6ca9e65b`、`65e06846-8581-4fa5-8657-78d2d097fca7`、`1f2405df-a6f1-45dd-9a83-7ab0d7dda489`。结论：平台相位会改变观测，但“等待平台后把 color-12 推向 color-0”仍不是已验证的胜利机制。
- [x] 后台 R4 实验矩阵（16 个单动作/短组合候选）运行中：使用独立 scorecard、recording、checkpoint 和 heartbeat；只有真实 score 非零或 `levels_completed>0` 才提升为可执行路线。
- [x] 根据 recording 逐帧 bbox 修正 LS20 waypoint 默认动作映射：`ACTION1=up`、`ACTION2=down`、`ACTION3=left`、`ACTION4=right`；同步更新碰撞预测和单元测试。
- [x] 官方 source 规则反推并远程验证 Level 1：起点 `(34,45)`，旋转开关 `(19,30)`，目标 `(34,10)`；要求 `GoalColor=9`、`GoalRotation=0`、shape `5`。已固化 `LS20Solver.level1_verified_route()`，路线 `ACTION3×3 → ACTION1×6 → ACTION4×3 → ACTION1×3`。证据：离线 Arcade `levels_completed=1`；远程 scorecard `9aae4d01-d506-4f84-ae8c-cd72000cc28c`，Level 1 `score=115.0`、`level_actions=15`、总分 `3.571428571428571`。
- [x] 已提取 Level 2–7 官方 source 的目标参数、起点、目标坐标及交互开关坐标；Level 2 起点 `(29,40)`、目标 `(14,40)`、旋转开关 `(49,45)`、目标 rotation `270`。初版 Level 2 几何路线已在“先完成 Level 1、再执行 Level 2”的离线流程中验证为 `levels_completed=1`，尚未完成 Level 2，不能提交为远程路线。

### 阻塞点 2：目标识别只有颜色候选，没有胜利机制

- 首帧可观察到 color-1 玩家候选约 `(32,21)`、color-0 目标候选，但尚未证明“接触目标”“站上平台”或“改变形状/颜色”哪个条件触发 Level 1 完成。
- 改进方向：从录制帧中寻找 `levels_completed`、`WIN`、目标颜色/形状变化的前后边界；把触发条件写成确定性 fixture，而不是硬编码颜色即胜利。

### 阻塞点 3：planner 有路线生成能力，但没有可执行的起点到目标路线

- `plan_waypoints()` 能生成 Manhattan 路线，然而真实场景尚未确认可通行地标、碰撞规则和动作持续步数；因此默认路径落入 fallback，重复 `ACTION1..4`，最终 `0/7`。
- 改进方向：先只完成 Level 1；用 baseline `22` 反推最短路线候选，逐段验证“动作后玩家坐标是否按预期变化”，失败立即停止并保存 recording。

### 阻塞点 4：动态平台安全层缺少真实玩家轨迹验证

- 本地 fixture 已验证“远处移动不失效、阻挡下一步才失效、移开后可继续”，但真实 E2E 尚未展示路线消费或在线重规划。
- 改进方向：先完成 Level 1 静态合法路线，再把玩家轨迹传入动态层；只在下一动作碰撞预测为真时清除路线，并增加平台进入/离开路线的连续帧 fixture。

### 不应优先做的工作

- 暂不扩展 Level 2-7、LLM 规则归纳或大规模 scorecard 重试。
- 暂不把 color-0/color-1 规则推广为所有 ARC 游戏的通用语义。
- 暂不把本地 fixture 通过等同于真实得分提升。


### P0.1 恢复或重建 LS20 solver

- [x] 新增 `lingjing_solo/planning/ls20_solver.py`。
  - 证据：新增安全路线执行层；完整 package 测试通过。
- [x] 定义结构化 `LS20State`，包含玩家位置、目标位置、形状、颜色、旋转、地标和动态障碍。
  - 证据：`ls20_solver.py` 的 `LS20State` dataclass。
- [ ] 从观测网格中识别 LS20 玩家方块和目标区域。
- [ ] 识别旋转台、调色台和形状台。
- [x] 生成前往下一个必要地标的路线。
  - 证据：`LS20Solver.plan_waypoints()` 按 5 像素步长生成逐动作 Manhattan 路线；`test_ls20_solver_plans_aligned_waypoints` 通过。注意：当前方向映射仍属于实现假设/待真实多步轨迹确认，不能视为游戏语义已验证。
- [x] 每次只生成一个动作，避免执行完整的过期路线。
  - 证据：`LS20Solver.next_action()` 每次最多消费一个动作，并过滤非法动作。
- [x] 为路线失效和合法动作过滤添加单元测试。
  - 证据：`test_ls20_solver_discards_stale_route_on_observation_change` 通过。
- [x] 建立不依赖颜色硬编码的连通对象提取和移动跟踪基础。
  - 证据：`planning/ls20_perception.py`；`test_ls20_perception_extracts_objects_and_tracks_motion` 通过。

**验收标准：** 在确定性的 LS20 fixture 上，solver 能为 Level 1 旋转任务生成有效动作序列，并且不会输出不在合法动作集合中的动作。

### P0.2 将 LS20 solver 接入 Agent

- [x] 增加明确的 LS20 游戏识别，或注入式游戏策略选择器。
- [x] 识别为 LS20 时，在通用探索流程之前调用 LS20 solver。
- [x] 对未知游戏保留通用 fallback。
- [x] 添加回归测试，证明 LS20 路径会被选择，通用路径仍然可用。

**验收标准：** `LingjingSoloAgent.choose_action()` 在 LS20 中返回 solver 提供的合法下一步动作；solver 无计划时能够安全 fallback。

### P0.3 增加动态障碍跟踪和在线重规划

- [x] 对比连续帧，跟踪动态平台位置。
- [x] 区分静态地标和移动障碍（当前为保守的运动对象集合，尚未完成 LS20 语义分类）。
- [x] 每收到一帧新数据后重新验证下一步动作。
- [x] 障碍变化或阻挡路线时，废弃旧路线并保留重规划入口。
- [x] 添加以下本地 fixture：平台远离计划路径、平台进入计划路径、平台移开；真实录制闭环仍未验证。

**验收标准：** 当路线被动态障碍改变后，solver 会在下一次危险移动前清除旧路线并生成替代路线，不会继续盲目执行旧计划。

### P0.4 使用 ARC 权威终止状态和关卡状态

- [x] 将 `GameState` 和 `levels_completed` 通过 package 边界传入，不只依赖网格推断。
- [x] package 层 `is_done()` 在权威状态为 `WIN` / `GAME_OVER` 时停止。
- [x] `levels_completed` 增加时识别关卡切换并清理当前模型。
- [x] 添加 WIN、GAME_OVER 路径和状态兼容性测试。
- [x] 替换 `lingjing_solo/world_model/field.py:153-155` 的占位行为，改为有文档说明的 callback contract。
  - 证据：`test_field_win_detector_callback` 通过；无 callback 时 fail-closed。

**验收标准：** Agent 在权威状态为 WIN/GAME_OVER 时立即终止，并在新关卡开始时重置当前关卡计划。

## P1 — LS20 全关卡能力

- [ ] 解析 HUD 或等价状态信息，包括形状、颜色、旋转和剩余步数预算。
- [ ] 支持 Level 3 的颜色加旋转目标。
- [ ] 支持 Level 4 的形状切换。
- [ ] 支持 Level 5 的调色台行为。
- [ ] 支持 Level 6 的有序多目标任务。
- [ ] 支持 Level 7 的迷雾 / 局部观测规划。
- [ ] 建立七关回归矩阵，记录每关完成情况和步数。
- [ ] 每个主要 solver 里程碑后运行一次新的真实 ARC `ls20` scorecard。
  - 已记录的最新候选结果：scorecard `c30d6954-4a4c-46bb-a815-39151aa598ec`；候选 `ACTION4×7` 仍为 `0/7`、`0.0`。该结果不能证明方向语义已确认。

**验收标准：** Level 1 至少保持实验基线水平，Level 2 不再因为过期静态路线而必然失败。

## P2 — 通用框架改进

- [x] 在 `lingjing_solo/planning/planner.py` 中实现基于已观测转移图的安全 `LightweightPlanner.search()`。
- [ ] 为通用 planner 添加目标距离和转移置信度评分。
- [x] 将固定探索分数替换为按动作区分的信息增益。
- [ ] 基于预测状态哈希和最近状态添加循环惩罚。
- [ ] 将目标推断和规则归纳从简单的高频转移统计扩展为更强的机制。
- [ ] 只有在状态隔离和 reset 语义验证完成后，才实现跨关卡规则迁移。
- [ ] 修复并测试 Kaggle notebook template 的继承路径。
- [ ] 添加可选 LLM 上下文，包含地标、规则、动态障碍和当前预算。

## Package 发布与团队协作

- [x] 保持 ARC adaptor 简洁，并将核心推理保留在 package 中。
- [x] 为团队开发者记录本地 editable 安装方式。
- [x] 将 package/solver 变更 push 到团队约定的远程仓库分支。
  - 证据：本轮提交后 `origin/explore_plan` 已更新；push 后工作树干净（2026-08-29）。
- [ ] 将 ARC adaptor 固定到有版本号的 package release，或有文档说明的共享 Git 引用。
- [ ] 验证干净 checkout 可以安装两个仓库，不依赖开发者本机 sibling directory。
- [ ] 不提交 API key、`.env` 文件、生成的 scorecard 或 build cache。

## 每个实现里程碑的验证清单

- [ ] 在实现前先写出验收标准。
- [ ] 在适用的情况下覆盖以下场景：正常路径、空输入、非法输入、边界限制、已有状态、幂等性、部分失败、重启接管、回滚、权限/安全和集成。
- [ ] 通过针对性测试。
- [ ] 通过相关完整测试套件。
- [ ] lint 和空白检查通过；或单独记录继承而来的失败。
- [ ] 当变更影响 adaptor 或运行时行为时，完成真实 ARC 集成运行。
- [ ] 使用 `git status` 和 `git diff` 检查变更文件范围。
- [ ] 在本文档中记录未验证项目和已知限制。

## 已知限制 / 基线失败

- `LightweightPlanner.search()` 目前只使用已观测转移，不能推断完全未知动作的后继状态。
- `WorldModelField.detect_win()` 默认 fail-closed；调用方需要注入可靠的 detector，或使用 ARC 权威状态。
- 当前通用配置仍使用抽象动作名称（`UP`、`DOWN`、`LEFT`、`RIGHT`、`SPACE`）；LS20 必须使用明确的动作语义边界，不能假设这些名称直接对应隐藏 ARC 动作。
- 实验报告显示 Level 2+ 会受到动态平台影响而失败；实现后仍需新的可复现实验 fixture 和真实 scorecard。
- 仓库级 lint 尚未完全通过；在区分基线问题和新增问题之前，不应将 lint gate 标记为完成。

## 跨 thread 已完成工作同步

以下内容在另一个 thread 中完成，并已纳入本状态文档；它们属于 ARC-AGI-3 adaptor/实验侧，不是本仓库的独立猜测：

- [x] 修复 recording 可观测性：`../ARC-AGI-3-Agents/agents/agent.py` 保存并写出 Agent 实际请求的 `requested_action`，与服务端回传的 `action_input` 分开。
- [x] 新增真实单动作探针：`../ARC-AGI-3-Agents/tools/ls20_single_action_probe.py`，流程为 reset → 单个 ACTION → 前后帧差分。
- [x] 完成 ACTION1–4 的真实单步 probe：ACTION1/3/4 各改变下方对象 52 格，ACTION2 改变 2 格；四次均未移动 color-1 marker，均为 `NOT_FINISHED`、`levels_completed=0`。
- [x] 增加 recording 回归测试：`../ARC-AGI-3-Agents/tests/unit/test_action_recording.py`。
- [x] 远程协作分支已同步：本轮 `origin/explore_plan` 已更新并完成 push。

**跨 thread 结果的边界：** 这些 probe 证明了动作请求已可追踪、且单步动作会改变场景对象；没有证明 ACTION1–4 的最终方向、目标交互规则或 Level 1 胜利条件。因此后续 action item 仍从有效多步 recording 开始，不能直接把单步变化区域写成方向映射。

- 2026-08-27 Level 1 语义识别：首帧确认 color-1 为玩家候选（质心约 `(32,21)`），color-0 为目标候选；adaptor 新增小型 color-1 marker 识别并传入动态安全层。

## 证据记录

- 2026-08-27 ARC `ls20` E2E（动态近邻策略）：scorecard `73158b5e-a934-44d0-a665-8a3234ea42ad`；`81` actions；`0/7`；`0.0`。动作日志仍为 ACTION1..4 fallback，原因是当时 adaptor 尚未提供 player 坐标；后续已接入 color-1 玩家候选识别。

- 2026-08-27 ARC `ls20` 受控路线实验：scorecard `56781c2a-893b-4b03-93b6-792e8bc05b62`，显式路线仅首动作被消费，随后因动态平台帧触发路线失效而回退 ACTION1..4 循环，得分 `0.0`。
- 2026-08-28 多步 recording probe：scorecard `21661d45-af0e-4e56-b385-734b9574f23e`；`81` actions、`0/7`、`0.0`；`requested_action` 覆盖 ACTION1–4；分析后得到 `80` 个 delta，玩家位移 `0`、换关触发 `0`。其中第 43 条帧为 `(6,64,64)` 混合通道，使用显式 `frame_channel=5` 读取；该兼容路径已通过全量测试。
- 2026-08-28 ACTION1×7 / ACTION3×3 隔离实验：scorecard `340333d0-c0c2-4724-a45d-ea2b814e91e7` 与 `0e5d2c29-20dc-4b84-9aab-17cc8c06def6` 均为 `0/7`、`0.0`、`NOT_FINISHED`；排除“单纯纵向接触顶部对象”及“单纯横向接近 color-1 marker”为 Level 1 触发条件。
- 2026-08-28 ACTION1×5 + ACTION3×3 组合实验：scorecard `673b40c3-4b70-477f-9e82-134109e206ee`；对象到 `y≈20` 后横向动作无效；`0/7`、`0.0`、`NOT_FINISHED`；支持建立位置条件状态转移表。
- 2026-08-28 位置条件横向动作对照：scorecards `8ed6b678-10a7-4038-98a2-2f22d03dac8e`、`b8dd08ca-027c-4720-b851-192c8606df43`、`d58bc004-8889-462d-b2f2-bedacd4bfa99`；ACTION3 在初始/高位有效、中位无效；三组均 `0/7`、`0.0`、`NOT_FINISHED`。
- 2026-08-28 R4 direction-map validation：修正 `lingjing_solo/planning/ls20_solver.py` 的 waypoint 与碰撞方向映射；`uv run pytest -q` 为 `20 passed`。真实 probe scorecard `bc46d725-2f27-4875-b016-092ab2b3316d` 仍为 `0.0`、`0/7`、`NOT_FINISHED`；因此已修正动作方向，但“到达顶部位置即胜利”被再次否定，尚未形成可得分路线。
- 2026-08-28 R4 短序列矩阵：16/16 候选完成，全部 `rc=0` 但均 `0/7`、`0.0`、`NOT_FINISHED`；checkpoint `/tmp/lingjing_r4_matrix_20260828/checkpoint.jsonl`，heartbeat `/tmp/lingjing_r4_matrix_20260828/heartbeat.log`；未发现可提升为默认路线的候选。

| 日期 | 命令 / 文件 | 结果 |
|---|---|---|
| 2026-08-27 | `uv build` | wheel 和 source archive 构建成功。 |
| 2026-08-27 | `uv run pytest -q` | `7 passed, 1 warning`。 |
| 2026-08-27 | 干净虚拟环境 import | 返回 `clean_import=LingjingSoloAgent`。 |
| 2026-08-27 | ARC adaptor 单元测试 | `5 passed`。 |
| 2026-08-27 | ARC adaptor lint / diff 检查 | `All checks passed!`；空白检查通过。 |
| 2026-08-27 | package 新增状态/planner/exploration 测试 | `10 passed, 1 warning`；使用 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 绕过环境 Hydra 插件错误。 |
| 2026-08-27 | ARC adaptor 回归测试 | `5 passed`；lint / diff 检查 `All checks passed!`。 |
| 2026-08-27 | ARC `ls20` E2E | 退出码 0；scorecard `36d36853-2560-43b6-9b66-7f90992c1c0b`；执行 81 个动作；`0/7` 关卡完成；得分 `0.0`；recording 已生成。 |
| 2026-08-27 | LS20 安全执行层 / WIN callback 测试 | `12 passed, 1 warning`；新增路线失效、合法动作过滤和 callback contract 验证。 |
| 2026-08-27 | adaptor 动态帧接入修复 | 初始空 frame 防护；ARC adaptor `6 passed`；package `15 passed`；ruff 通过。 |
| 2026-08-27 | ARC `ls20` E2E（修复后） | scorecard `54304e35-32a0-40d6-acf5-67b1cc898eec`；执行 81 个动作；`0/7` 关卡完成；得分 `0.0`；无运行时异常。 |
| 2026-08-28 | 跨 thread recording/动作证据修复 | ARC targeted tests `1 passed`；Lingjing-Solo tests `15 passed`；单动作 probe 四次退出码 0；尚未得到通关分数。 |
| 2026-08-28 | 多步 recording 与分析 API | scorecard `21661d45-af0e-4e56-b385-734b9574f23e`；`81` actions、`0/7`、`0.0`；`analyze_recording(frame_channel=5)` 返回 `80` 个 delta，四种动作均有记录；全量 package `20 passed, 1 warning`，相关 lint 通过。 |
| 2026-08-29 | 当前分支回归验证 | `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q` 与 `uv run pytest -q` 均返回 `21 passed, 1 warning`；`git diff --check` 通过。 |
| 2026-08-28 | Level 2 官方引擎状态搜索 | 完成 Level 1 后从 Level 2 初始状态以 `ACTION1–4` 做深度 ≤21 的真实引擎 BFS；访问 `85` 个状态，未找到目标坐标 `(14,40)` 且 rotation `270` 的解。Level 2 尚未提交远程 ARC。 |
| 2026-08-28 | R4 Level 2 预算诊断 | 官方 source 显示 `StepCounter=42`、`StepsDecrement=None`（运行时默认 decrement `2`），即约 `21` 个有效动作；起点→目标几何下界 `17` 步，起点→旋转开关 `17` 步，且需三次旋转触发，当前简单开关路线超预算。 |
| 2026-08-28 | Level 2 无限预算精确 BFS | 修正 SDK 计数器字段 `current_steps` 后，官方引擎找到 Level 2 最短真实路线 `14111114424222222121211111113333332232222`，长度 `41`；路线满足 rotation `270` 并到达 `(14,40)`。但正常配置默认每步 decrement `2`，仅允许约 `21` 步，确认关卡数据/运行时预算矛盾；未提交远程 ARC。 |
