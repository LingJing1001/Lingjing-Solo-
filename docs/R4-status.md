# R4 热点检测状态

日期：2026-09-17
分支：`feature/r4-hotspot-detection`

## 本轮完成

- 保持观察契约、候选结构、离线热点检测、短时序噪声识别和有界探测规划彼此隔离。
- 保持反馈归因只接受可审计的状态、关卡或分数证据；视觉变化本身不会被当作成功。
- 新增 `probe_dry_run.py`，在任何真实动作调用前完成：
  - 探测动作名称和模式校验；
  - 坐标边界校验；
  - 运行标识、帧标识、候选标识和 evidence refs 保留；
  - 明确写入 `executed: false` 和 `status: dry_run`；
  - 以 JSONL 保存可复盘的探测请求。
- 新增 `probe_gate.py`，把 R4 探测能力放在 Field/Learner 之外：
  - feature flag 默认关闭并 fail-closed；
  - 启用时仅允许 click-family observation 继续；
  - keyboard observation 始终拒绝 click probe；
  - 输入 `ProbePlan` 保持不可变；
  - 回退只返回空 action plan，不调用外部 harness。
- 完成 Field/Learner 边界的最小契约：`HypothesisContext` 经 observation 传入 planner；probe 保留 `hypothesis_space_type`，并在有上下文时标记 `phi_interactive_hotspot`。
- 增加专项测试，覆盖正常 dry-run、JSONL 输出、越界坐标、动作/模式不一致、未知动作、feature flag 回退、keyboard 隔离、计划幂等性和 hypothesis/Φ 边界。

## 后续实施（2026-09-17）

本轮按 P0 继续完成了离线探索决策契约的第一步，未接入真实 harness：

- 在 `lingjing_solo/exploration/explorer.py:20,51-92` 增加 `last_score_details`，记录信息增益、反循环惩罚、目标奖励、总分、输入顺序和选择原因。
- 在 `lingjing_solo/exploration/explorer.py:124-133` 修正 `step_probe()` 预算语义：`probe_max_steps=N` 时恰好允许 N 次调用，耗尽后关闭 probing。
- 在 `lingjing_solo/core/config.py:43` 增加可选 `goal_score_bonus`，默认值为 `0.0`，保持原有默认评分行为；当已知后继命中目标状态时，可按目标置信度加分。
- 新增 `tests/test_r4_explorer.py`，覆盖预算边界、空动作、稳定 tie-break、目标后继奖励和评分审计字段。

本轮证据：

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/test_r4_explorer.py tests/test_probe_planner.py tests/test_probe_gate.py tests/test_probe_dry_run.py`：18 passed，exit 0。
- `ruff check lingjing_solo/core/config.py lingjing_solo/exploration/explorer.py tests/test_r4_explorer.py`：通过，exit 0。
- `git diff --check`：通过，exit 0。
- 全量 pytest：67 passed，exit 0；R5 reflection 基线已在后续 commit 修复。

本轮新增完成（2026-09-17）：

- `ExplorationEngine.infer_goal()` 不再是占位接口：接受带 `description/state_hash/confidence/kind` 的权威 WIN/level 反馈，并写入 `GoalHypothesis`；无 callback 时使用 Field 已记录的 WIN hash。
- `info_gain()` 统一使用 `WorldModelField.current_hash()`，避免 level-aware hash 与 transition index 不一致；在 novelty 衰减之外增加后继状态熵项，有限权重奖励不确定动作。
- `action_diff.analyze_recording()` 遇到 `state=RESET` 或 `requested_action=RESET` 时切断前后帧，reset 后重新建立 baseline；支持嵌套 action payload `{name/id}`。
- `action_diff._normalize_grid()` 支持 ARC recording 的 `N×H×W` 动画帧栈：默认取最后一层作为动作完成后的 settled/current observation，仍支持 `frame_channel` 显式选择。
- 新增 LS20 64×64 与 AR25 8×8 FrameData/action schema replay 测试，覆盖动作写入、关卡推进和 WIN 反馈。

- 真实 LS20 recording：`/srv/agent-platform/projects/ARC-AGI-3-Agents/recordings/ls20-9607627b.lingjingsolo.800.bbc5baae-3c02-44f1-9a96-70159143d4b2.recording.jsonl`。
- recording 解析：309 records、309 条 `requested_action`、frame shapes 为 `(1,64,64)/(2,64,64)/(6,64,64)/(17,64,64)`、state `NOT_FINISHED=308/WIN=1`、levels `0→7`。
- R4 action diff：308 deltas、308 个带动作 transition、最高 level 7。
- R4 Field replay：308 transitions、`field_levels=7`、`field_env_state=WIN`、`field_win_hash_count=1`、`field_version=308`，VERDICT PASS。
- 执行命令：`PYTHONPATH=/srv/agent-platform/projects/Lingjing-Solo- python3 /tmp/verify_ls20_r4.py`，exit 0。

本轮回归证据：

- R4 + recording boundary + goal/explorer 定向测试：`12 passed`，exit 0。
- 全量 pytest：`70 passed`，exit 0。
- `ruff check lingjing_solo tests`：`All checks passed!`。
- `git diff --check`：通过，exit 0。

本轮未完成：真实 ft09 L3、AR25 原始 recording、重启接管、真实部署回滚和真实收益对比；LS20 真实 recording replay 已通过，但不替代未知游戏的真实探索证据。

`L1`：真实 LS20 309-recording replay 闭环通过（Field 308 transitions，最终 `levels=7/state=WIN`）；尚未宣称真实 ft09 L3。

## 验收标准

- 所有计划探测在执行前都能转换为带有坐标、动作、模式和证据引用的 dry-run 请求。
- 非法动作、动作/模式不一致和越界坐标 fail-closed。
- dry-run artifact 明确表示未执行，不产生真实环境成功证据。
- 既有 detector、planner、feedback 和 evidence 测试不回归。

## 边界检查

- `happy_path`：通过 `tests/test_probe_dry_run.py` 的正常请求和 JSONL 测试。
- `empty_input`：通过空计划只写运行头的测试。
- `invalid_input`：通过非法动作、模式和坐标测试。
- `boundary_limit`：复用规划器的最多三个探测限制；既有测试通过。
- `existing_state`：只追加 JSONL，不覆盖既有 artifact；feature flag 不修改输入计划；hypothesis context 只读传递。
- `idempotency`：相同计划生成稳定请求字段；通过专项测试。
- `partial_failure`：验证阶段失败时不调用外部 harness；通过 fail-closed 异常测试。
- `restart_adoption`：未验证。
- `rollback`：feature flag 关闭时返回空 action plan；通过专项测试；真实部署回滚未验证。
- `security_permissions`：未验证真实部署目录权限；测试只写入临时目录。
- `integration`：真实 LS20 recording 已通过 `action_diff → WorldModelField` replay；AR25 仅有 schema replay，真实 harness、AR25/ft09 recording 和真实 action payload 仍未接入。

## 阻塞与未验证项

- DGX 官方 checkout 已发现 69 个 LS20 recording；本轮使用其中最新的 309-recording 文件完成真实 replay。
- 当前 `/srv/agent-platform/projects/ARC-AGI-3-Agents/recordings/` 未发现 AR25 或 ft09 原始 recording；这不表示官方环境没有对应游戏。
- 当前缺少的是 AR25/ft09 的真实 recording、坐标约定和动作 payload 样本；没有伪造这些真实环境证据。
- 全量 pytest 本轮已通过 70 tests；历史 `click_sweep` / `Ar25Config` 问题未作为本轮变更范围。

## 下一步

1. 对 AR25 获取并重放真实 recording，冻结其多层 frame / action payload 语义；
2. 接入真实 recording 后冻结坐标、action payload 和 scorecard schema；
3. 在真实环境仅开放 observe-only，再评估 click 动作；
4. 补齐 hypothesis space / Φ 证据的真实 Field/Learner 消费端，并验证 ft09/r11l 共用候选 schema；
5. 补充重启接管、回滚和部署集成验证。

## 本轮验证记录

- `PYTHONPATH=/srv/agent-platform/projects/Lingjing-Solo- python3 /tmp/verify_ls20_r4.py`：真实 LS20 recording replay PASS，exit 0。
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/test_r4_recording_boundaries.py tests/test_r4_explorer.py tests/test_r4_goal_inference.py`：11 passed，exit 0。
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q --disable-warnings`：69 passed，exit 0。
- `ruff check lingjing_solo tests`：通过，exit 0。
- `git diff --check`：通过，exit 0。
