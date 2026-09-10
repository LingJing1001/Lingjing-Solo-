# ARC-AGI-3 R11L 改动计划与状态

> 最后更新：2026-09-02T23:06:00-07:00
> 状态权威：本文档跟踪 R11L 接入、探测、路线建立和验证证据；实现状态以实际测试/recording/Scorecard 为准。
> 目标游戏：`r11l-495a7899`
> 当前阶段：**P0 已完成；P1.1/P1.3/P1.6 已完成；P1.2/P1.4/P1.5/P1.7 被 Level 0 源码/交互条件矛盾阻塞**

## 1. 目标与边界

R11L 是下一款验证游戏：`click-only`、6 个关卡，建议文档记录的 baseline actions 为 `[22, 33, 51, 26, 52, 49]`。这些数量只用于估计动作规模，不是路线或难度证明。

本轮目标不是把 LS20 方案复制到 R11L，而是沿当前 strategy-registry 架构完成：

1. 使用 Generic 模式完成 reset、单动作 probe 和 recording；
2. 从真实 frame/action/state 差分建立 R11L game profile；
3. 先得到 Level 1 的可重放最小路线，再逐关扩展；
4. 只有 Generic planner + route artifact 无法表达时，才新增 `R11LStrategy`；
5. 用同一个 `LingjingSolo` adapter 完成离线回放、集成 smoke test 和新的在线 Scorecard。

明确不在本轮范围内：

- 不修改 `LS20Strategy`、`LINGJING_LS20_PLAN` 或 LS20 canned route；
- 不把 click 坐标、动作顺序或目标语义凭空写入 production code；
- 不把生成 Scorecard 或 `exit 0` 当作通关证据；
- 不提交 `ARC_API_KEY`、`.env`、recording 或生成的 Scorecard 文件；
- 不改动 Lingjing 核心 Agent，除非 R11L 实验明确证明边界契约缺失。

## 2. 已知基线与当前状态

| 项目 | 当前状态 | 证据 |
|---|---|---|
| Adapter 架构 | 已存在 strategy registry；R11L 已接入独立策略 | `arc_adaptor/agents/strategies/registry.py:10-25` |
| LS20 专用策略 | 已隔离，R11L 不依赖 | `arc_adaptor/agents/strategies/ls20.py`；`docs/ARC-AGI3-adapter-architecture.md:115-124` |
| R11L registry 路由 | 已路由到 `R11LStrategy` | `arc_adaptor/tests/test_lingjing_solo_agent.py:32-35` |
| R11L API/游戏元数据 | 已由 `Arcade.get_environments()` 返回；title=`R11L`、tag=`click`、6 levels、baseline actions `[22,33,51,26,52,49]` | P0.1 命令输出 |
| R11L 单动作 probe | 已执行；服务端回显请求坐标，但旧策略误把 colour 6 UI 像素当作目标；本地源码确认 `sys_click` 未选中 sprite 为 colour 3 | `environment_files/r11l/495a7899/r11l.py:1511-1518`；probe 输出 |
| R11L game profile | 已建立初始 profile | `docs/ARC-AGI3-R11L-profile.md` |
| R11L route artifact | 未创建 | 坐标/transition 语义尚未证明 |
| R11L 专用策略 | 已创建最小观测驱动 profile，不宣称 solver | `arc_adaptor/agents/strategies/r11l.py` |
| R11L 全关在线验证 | 未执行 | P1.2/P1.4 尚未通过 |

## 3. 实施计划与 action items

### P0 — 建立可重复的探测入口

- [x] **P0.1 确认运行 checkout 与依赖边界**
  - 确认 ARC checkout、Lingjing editable install、`uv sync` 和 `main.py --help`。
  - 确认 `LINGJING_LS20_PLAN` 未设置；R11L 不使用 LS20 plan。
  - 确认 `ARC_API_KEY` 只存在于运行环境，不写入日志/文档。
  - 完成条件：命令输出和环境检查结果写入证据日志，敏感值只记录“已配置/未配置”。已完成：ARC checkout=`/srv/agent-platform/projects/ARC-AGI-3-Agents`，`uv`/Python 3.12.3/venv 可用，R11L 元数据返回 25 environments。

- [~] **P0.2 新增通用/可参数化 R11L 单动作 probe**
  - 优先复用 `arc_adaptor/tools/ls20_single_action_probe.py` 的 recording、checkpoint、heartbeat 和错误处理模式；不要复制 LS20 语义命名。
  - 建议新增：`arc_adaptor/tools/r11l_single_action_probe.py`，或将现有工具重构为显式 `--game-id` 参数后由 R11L 调用。
  - 每个候选 click：`RESET → 单一合法 click → 保存前后 frame、requested_action、available_actions、state、levels_completed`。
  - 坐标必须来自当次 frame/合法 action 契约；禁止使用猜测坐标作为固定路线。
  - 完成条件：至少每种合法 click action 有一条可解析 recording；异常路径保留日志并返回非零退出码。已验证 `save_recording=True` 会创建 JSONL；最近成功 recording：`recordings/00bfff15-7027-4ac1-b0a0-4f3b3e0b8bdd/r11l-495a7899-4c939b96-eb07-4551-b907-ba720d5cd2ee.jsonl`。probe 现在额外输出 `server_action_input`，以区分请求坐标与服务端回显。

- [x] **P0.3 增加 recording/action-diff 回归覆盖**
  - 检查 `requested_action` 与服务端 `action_input` 的区分仍然成立。
  - 检查 click 坐标、frame shape、嵌套 payload、空合法动作和终止状态。
  - 建议新增/调整：`arc_adaptor/tests/test_r11l_probe.py`、`arc_adaptor/tests/test_action_recording.py`。
  - 完成条件：fixture 测试先 RED，再实现后 GREEN；相关 adapter 测试与 recording 测试全部通过。当前 targeted pytest 为 13 passed。

### P1 — 建立 R11L profile 和 Level 1 路线

- [x] **P1.1 建立 R11L game profile**
  - 建议文件：`arc_adaptor/routes/r11l/README.md` 或 `docs/ARC-AGI3-R11L-profile.md`。
  - 记录：frame shape/通道、合法 click action 形式、坐标系、reset 行为、点击后的变化区域、state 变化、level 计数、game-over 条件、动作预算。
  - 每个结论附 recording 路径、命令、退出码和 frame/action 序号。已建立 `docs/ARC-AGI3-R11L-profile.md`；坐标语义结论保留为未验证。

- [~] **P1.2 从首关 recording 推导最小路线**
  - route schema 已设计，但暂不写入 `level_0.json`：旧 probe 的 frame diff 统计包含 leading channel，且旧策略选择 colour 6 UI 像素；本地源码已确认 display 坐标为 64x64，`ACTION6` 先选择 colour-3 `sys_click` sprite，再提交移动坐标。
  - 下一步采用闭环 route learner：每步保存 `before_hash → requested_action(data.x/data.y) → server_action_input → after_hash/diff → state/level`，只固化重复重放一致且导致 level transition 的 action。
  - 当前完成条件未通过：修正后的策略尚未在在线 runner 产生首关 transition，禁止生成伪 route。
  - 新增源码级 blocker 证据：Level 0 的 `roefwu-pumlzd` 初始正色集合为 `{6,15}`，目标 `flkdtg-pumlzd` 为 `{15}`；本地官方 engine 枚举 0–63 display 坐标、检查 3,144 个无障碍候选，未找到同时满足 `collides_with` 与 `ldzvchvkvp` 的 destination。该条件下 Level 0 无法完成，需 ARC 服务端/环境版本修复或确认 source 数据是否损坏。

- [x] **P1.3 选择 Generic、route artifact 或 R11LStrategy**
  - 默认实现：GenericStrategy + 版本化 route artifact。
  - 仅当路线选择依赖跨帧隐状态、动态目标、特殊 reset 或 Generic 契约无法表达时，才新增 `arc_adaptor/agents/strategies/r11l.py`。
  - 若新增策略，必须由 `GameStrategyRegistry.resolve()` 统一注册；禁止在 `LingjingSolo.choose_action()` 增加 R11L `if/else`。
  - 完成条件：新增架构决策有测试证明，LS20 registry 和未知游戏 fallback 不回归。已选择最小 `R11LStrategy`；adapter 支持带坐标 action request；13 项 targeted tests 通过。

### P1 — 扩展全关并验证

- [ ] **P1.4 逐关建立 route artifact**
  - 依次建立 `routes/r11l/level_0.json` 至 `level_5.json`；每关重新确认合法动作集合和 level transition。
  - 保留失败 recording，不覆盖成功样本；记录动作数与 baseline 的差异。
  - 出现 route 不稳定时，暂停固化，回到 profile/状态模型，不增加盲重试。

- [ ] **P1.5 离线 replay gate**
  - 对完整 6-level route 做从 Level 0 开始的连续前缀回放，不允许从后续关卡新建状态冒充前缀验证。
  - 验证：动作合法性、reset 次数、每关推进、终止状态、动作预算和 recording 可追踪性。
  - 完成条件：离线环境真实观察到 `levels_completed=6` 或明确记录离线环境不可用及替代证据边界。

- [x] **P1.6 ARC harness 集成 smoke test**
  - 同步命令：`bash ../Lingjing-Solo-/arc_adaptor/sync_to_arc.sh <ARC_DIR>`。
  - 确认 ARC 原生 `agents/__init__.py` 未被覆盖，且 `LingjingSolo/lingjingsolo` 注册仍存在。已恢复 checkout 缺失的 `agents/structs.py`（来自 ARC 历史提交 `4111109`），collection blocker 已解除；修正策略后定向 suite 为 14 passed；完整 suite 为 94 passed / 10 failed，失败为既有 Agent/Swarm API 漂移，不是 R11L 定向测试失败。

- [ ] **P1.7 新建 R11L 在线 Scorecard**
  - 仅在 P0/P1 定向测试、probe 和离线/recording gate 通过后执行。
  - 记录：Scorecard ID/URL、exit code、score、actions、resets、levels_completed、win/game-over state、recording 路径和 git commit。
  - 接受条件：6/6 levels、正分、最终 `WIN`；否则保留为实验结果，不标记 R11L 完成。

### P2 — 清理与长期迁移

- [ ] 将稳定 R11L route 纳入 `arc_adaptor/MANIFEST.md`，明确 artifact 版本和来源 recording。
- [ ] 把通用 probe/action-diff 能力抽象为多游戏参数，避免 `ls20_single_action_probe.py` 与 `r11l_single_action_probe.py` 长期分叉。
- [ ] 更新 `docs/ARC-AGI3-adapter-architecture.md`，仅在 R11L 真实验证完成后增加 R11L 作为已验证 route 示例。
- [ ] 更新根状态文档 `docs/STATUS.md`，同步 R11L 的真实证据和未验证项。

## 4. 验收门槛

### Gate A：边界与探测

- [ ] R11L game ID 正确，API/runner 可访问。
- [ ] reset 可重复，至少一个合法 click 可执行。
- [ ] 每条 recording 可读取 frame、requested action、state 和 level counter。
- [ ] click 坐标/动作不会泄漏非法 action；空合法动作安全返回 `RESET`。

### Gate B：路线与策略

- [ ] Level 1 route 两次重复回放一致。
- [ ] route artifact 可独立读取，不依赖 LS20 环境变量或 LS20 类。
- [ ] registry 测试证明 R11L 路由和未知游戏 fallback 均符合预期。
- [ ] 任何 R11L 专用策略都有 RED/GREEN 回归测试和“为何 Generic 不足”的证据。

### Gate C：真实集成

- [ ] ARC checkout 同步后定向测试通过。
- [ ] 可选 recording patch 应用前后均有明确状态；重复 apply 会安全失败或被检测阻止。
- [ ] 新建 Scorecard 记录 `6/6`、正分和 `WIN`；仅有 `exit 0` 不足以通过。

## 5. 11 类边界/故障检查矩阵

| 类别 | 当前状态 | R11L 必须验证的内容 |
|---|---|---|
| happy_path | [~] 部分验证 | reset、合法 click 已验证；Level 1→6、最终 WIN 未验证 |
| empty_input | [x] 部分验证 | 空合法动作已有 adapter 回归；空 frame/缺 payload 未完整验证 |
| invalid_input | [x] 部分验证 | 越界坐标由 `ComplexAction` 拒绝；非法 action、错误 game_id、损坏 recording 未完整验证 |
| boundary_limit | [x] 部分验证 | 0–63 坐标约束和 64×64 frame 已验证；完整路线预算未验证 |
| existing_state | [ ] 未验证 | 非零 `levels_completed`、中途已有 recording、已有 route artifact |
| idempotency | [x] 部分验证 | sync 脚本重复运行通过；重复 reset/probe 语义未完整验证 |
| partial_failure | [x] 部分验证 | click None/异常返回非零；网络/Scorecard 中断恢复未验证 |
| restart_adoption | [ ] 未验证 | runner/adaptor 重启后能从 reset/持久 route 安全接管；不复用过期 frame |
| rollback | [x] 部分验证 | LS20/未知游戏 targeted tests 通过；R11L 回退演练未执行 |
| security_permissions | [x] 部分验证 | key 未写入代码/文档；完整 git secret scan 未执行 |
| integration | [x] 部分验证 | package、ARC adapter、registry、在线 harness 单动作联通；全关未验证 |

## 6. 预期改动范围

本轮实际修改/同步路径如下；Level route artifact 尚未创建：

- `arc_adaptor/tools/r11l_single_action_probe.py`（候选）
- `arc_adaptor/tests/test_r11l_probe.py`（候选）
- `arc_adaptor/tests/test_action_recording.py`（必要时调整）
- `arc_adaptor/agents/strategies/r11l.py`
- `arc_adaptor/agents/strategies/registry.py`
- `arc_adaptor/agents/strategies/__init__.py`
- `arc_adaptor/agents/templates/lingjing_solo_agent.py`
- `arc_adaptor/tools/r11l_single_action_probe.py`
- `arc_adaptor/tests/test_r11l_probe.py`
- `arc_adaptor/tests/test_lingjing_solo_agent.py`
- `arc_adaptor/sync_to_arc.sh`
- `arc_adaptor/MANIFEST.md`
- `docs/ARC-AGI3-adapter-architecture.md`
- `docs/STATUS.md`

## 7. 证据日志

| 时间 | 操作 | 结果 |
|---|---|---|
| 2026-09-02 | 读取用户提供的 R11L 验证建议 | 确认推荐顺序为 R11L；建议执行单动作 probe → Level 1 route → 全关验证。 |
| 2026-09-02 | 读取当前 adapter architecture | 确认 registry、GenericStrategy、LS20Strategy 边界及同步约束。 |
| 2026-09-02 | `git status --short --untracked-files=all` | 当前工作树在写入本文档前无输出；既有分支为 `feature/arc-strategy-registry`。 |
| 2026-09-02 | `GET https://arcprize.org/api/games/r11l-495a7899` | HTTP 404；该路径不是本次可用 API 证据，不能据此否定用户源文档中的 API 记录；后续须用实际 ARC runner/API 路径重试。 |
| 2026-09-02 | `Arcade.get_environments()` / reset | 返回 25 environments；R11L 元数据为 click、6 levels；reset 返回合法 ACTION6、64×64 frame、NOT_FINISHED。 |
| 2026-09-02 | `tools/r11l_single_action_probe.py --x 32 --y 32` | 成功返回 after frame；`changed_cells=1`、bbox=row=0,col=0、level=0；Scorecard=`97f65ab1-b16c-4e61-aca1-baa364beeb4f`。 |
| 2026-09-02 | targeted pytest | 13 passed；覆盖 adapter、registry、R11L marker coordinate、probe evidence。 |
| 2026-09-02 | full pytest | 2 collection errors：ARC checkout 缺少 `agents.structs`，与本轮 R11L 文件无关；详见最终报告。 |

## 8. 当前结论与下一检查点

**结论：** P0 已完成；P1.1/P1.3 已完成。R11L 已有最小观测驱动 profile，但尚无可验证 route；不能宣称通关。

**下一检查点：** 获得第一批 R11L recording 后，补填 profile、动作差分和 Level 1 route 的证据；只有 Gate A 通过，才进入 P1.2。

**未验证项：** click 坐标服务端语义、Level 1 route、离线 replay、六关 route、完整 ARC harness、在线 6/6 Scorecard。

**重启要求：** 不需要 DGX 服务重启；ARC checkout 已通过 sync 更新，若 runner 进程已在运行，需要按其自身 session 重启以加载新 adapter。
