# Lingjing-Solo 项目计划与状态

> 最后更新：2026-08-27T23:52:04-07:00
> 状态权威：本文档用于跟踪实现里程碑及其验证证据。
> 范围：`Lingjing-Solo-` 核心 package 及其 ARC-AGI-3 adaptor 集成。

## 状态标记

- `[x]` 已完成，并有可执行证据支持。
- `[ ]` 未完成，或尚未完成验证。
- `P0` 阻塞项 / 最高优先级。
- `P1` 重要后续工作。
- `P2` 后续优化。

## 当前状态摘要

- Package、ARC adaptor、状态注入和基础 planner 已完成并有本地测试证据。
- LS20 solver 已有对象提取、合法动作过滤、waypoint 和动态安全框架，但 Level 1 仍未形成可执行闭环。
- 最新真实结果：scorecard `73158b5e-a934-44d0-a665-8a3234ea42ad`，`81` actions，`0/7`，`0.0`。
- 当前主要阻塞是从 recording 归纳真实动作语义、目标触发条件和从 `(32,21)` 出发的合法路线；不是安装或运行链路故障。
- 改进顺序：单动作差分 → Level 1 触发 fixture → 22-action baseline 路线 → 真实 E2E → 动态平台在线重规划。

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
  - 证据：`uv run pytest -q` 返回 `7 passed, 1 warning`。
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
- **LS20 solver：** 框架已完成，但 Level 1 语义闭环未完成。已具备对象提取、动作合法性过滤、waypoint 路线和动态障碍安全接口。
- **动态障碍：** 本地近邻阻挡 fixture 已通过；真实 adaptor 已能识别 color-1 玩家候选并传递坐标，但尚未证明跨帧/跨关卡鲁棒性。
- **真实结果：** 最新 scorecard `73158b5e-a934-44d0-a665-8a3234ea42ad` 为 `0/7`、`0.0`、`81` actions；提交链路可运行，算法尚未解决 Level 1。
- **当前结论：** 目前不是工程集成阻塞，而是 LS20 的游戏机制和动作路线尚未从 recording 中归纳出来。
- **下一里程碑：** 完成 Level 1 的玩家/目标识别、动作映射、胜利触发和低于 baseline `22` actions 的合法路线；之后再推进全关动态重规划。

## 当前主要阻塞点与改进方向

### 阻塞点 1：Level 1 的真实动作语义尚未闭环

- 已知官方动作集合是 `[1, 2, 3, 4]`，但 recording 中玩家未产生可验证位移，尚不能仅凭动作编号确认四方向语义和触发时机。
- 改进方向：建立单动作差分分析；每次只改变一个动作，记录 color-1 marker、color-0 目标、平台和关卡状态变化，形成 `ACTIONn -> 位移/触发` 表。

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
  - 证据：`LS20Solver.plan_waypoints()` 按 5 像素步长生成逐动作 Manhattan 路线；`test_ls20_solver_plans_aligned_waypoints` 通过；默认方向映射已根据真实 recording 校正为 `ACTION1=down, ACTION2=left, ACTION3=right, ACTION4=up`。
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
  - 最新 scorecard：`c30d6954-4a4c-46bb-a815-39151aa598ec`；候选 `ACTION4×7` 仍为 `0/7`、`0.0`。

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
- [ ] 将 package 发布或 push 到团队约定的远程仓库。
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

- 2026-08-27 Level 1 语义识别：首帧确认 color-1 为玩家候选（质心约 `(32,21)`），color-0 为目标候选；adaptor 新增小型 color-1 marker 识别并传入动态安全层。

## 证据记录

- 2026-08-27 ARC `ls20` E2E（动态近邻策略）：scorecard `73158b5e-a934-44d0-a665-8a3234ea42ad`；`81` actions；`0/7`；`0.0`。动作日志仍为 ACTION1..4 fallback，原因是当时 adaptor 尚未提供 player 坐标；后续已接入 color-1 玩家候选识别。

- 2026-08-27 ARC `ls20` 受控路线实验：scorecard `56781c2a-893b-4b03-93b6-792e8bc05b62`，显式路线仅首动作被消费，随后因动态平台帧触发路线失效而回退 ACTION1..4 循环，得分 `0.0`。

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
