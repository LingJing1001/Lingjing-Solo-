# R4 热点检测状态

日期：2026-09-10
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

## 当前证据等级

`L1`：离线与 synthetic fixture 闭环通过；尚未宣称真实 ft09 L3。

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
- `integration`：真实 harness、recording、action payload 尚未接入；本轮只验证 Field/Learner context 的离线传递。

## 阻塞与未验证项

- 已通过 SSH `wsl` 重新读取并更新 Windows 计划文档；当前可见路径为 `/mnt/c/JWang/2026/Projects/Lingjing/learning/`。
- WSL 侧搜索 `/mnt/c/JWang/2026/Projects/Lingjing` 和 `/mnt/c/newtask-pi` 后，未发现已生成的 ft09 本地 recording、带 frame data 的 JSONL recording 或动作 payload 样本；这不表示 ARC-AGI-3 没有 ft09 游戏。
- 当前缺少的是从官方 ARC-AGI-3 环境实际运行 ft09 后生成的 recording、坐标约定和动作 payload schema；没有伪造真实环境证据。
- 全量 pytest 仍受仓库既有导入/测试基线影响，未修改 `click_sweep` 或 `Ar25Config` 问题。

## 下一步

1. 通过 SSH `wsl` 继续检查官方 checkout 或真实 recording 入口；
2. 接入真实 recording 后冻结坐标、action payload 和 scorecard schema；
3. 在真实环境仅开放 observe-only，再评估 click 动作；
4. 补齐 hypothesis space / Φ 证据的真实 Field/Learner 消费端，并验证 ft09/r11l 共用候选 schema；
5. 基线导入问题修复后运行全量测试，并补充重启接管、回滚和集成验证。

## 本轮验证记录

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 /tmp/run_r4_m2_pytest.py`：35 passed，exit 0。
- `ruff check ...`：通过，exit 0。
- `python3 -m py_compile ...`：通过，exit 0。
- `git diff --check`：通过，exit 0。
