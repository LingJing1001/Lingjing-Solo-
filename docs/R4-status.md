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
- 将 dry-run 适配层导出为公开探索 API；它不调用 harness、不执行点击、不执行键盘动作。
- 增加专项测试，覆盖正常 dry-run、JSONL 输出、越界坐标、动作/模式不一致和未知动作。

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
- `existing_state`：只追加 JSONL，不覆盖既有 artifact；通过 writer 行为检查。
- `idempotency`：相同计划生成稳定请求字段；通过专项测试。
- `partial_failure`：验证阶段失败时不调用外部 harness；通过 fail-closed 异常测试。
- `restart_adoption`：未验证。
- `rollback`：未验证；本轮没有真实动作副作用。
- `security_permissions`：未验证真实部署目录权限；测试只写入临时目录。
- `integration`：真实 harness、recording、action payload 尚未接入。

## 阻塞与未验证项

- 当前环境不存在 C 盘映射路径 `/mnt/c/JWang/2026/Projects/Lingjing/learning/`，因此本轮无法直接重读或更新 Windows 计划文档。
- 尚未找到可读取的真实 ft09 recording、坐标约定和动作 payload schema；没有伪造真实环境证据。
- 全量 pytest 仍受仓库既有导入/测试基线影响，未修改 `click_sweep` 或 `Ar25Config` 问题。

## 下一步

1. 在可访问真实计划文档的环境中同步本轮 dry-run 适配层状态。
2. 用真实 recording 或官方 harness 冻结帧引用、坐标系和动作 payload schema。
3. 将 dry-run 请求接到真实动作调用的前置校验点，并保留执行前 artifact。
4. 在真实环境只开放明确 feature flag，先执行 observe-only，再评估 click 动作。
5. 基线导入问题修复后运行全量测试，并补充重启接管、回滚和集成验证。

## 本轮验证记录

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 /tmp/run_r4_m2_pytest.py`：27 passed，exit 0。
- `ruff check lingjing_solo/exploration lingjing_solo/perception/observation.py tests/test_hotspot_detector.py tests/test_probe_planner.py tests/test_progress_signal.py tests/test_temporal_noise_and_evidence.py tests/test_probe_dry_run.py`：通过，exit 0。
- `python3 -m py_compile lingjing_solo/exploration/*.py lingjing_solo/perception/observation.py tests/test_probe_dry_run.py`：通过，exit 0。
- `git diff --check`：通过，exit 0。
