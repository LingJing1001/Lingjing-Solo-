# R4 热点检测状态

日期：2026-09-10
分支：`feature/r4-hotspot-detection`

## 本轮完成

- 读取并核对 `R4-evaluation-and-plan.md` 与 `R4热点检测并行线实施计划.md`。
- 按 P0-B/P0-C/P0-E 建立离线契约：
  - `NormalizedObservation`：统一 2-D 帧、RGB/灰度转换、坐标尺寸和 evidence ref。
  - `HotspotCandidate` / `HotspotFeatures` / `HotspotScore` / `HotspotDetectionResult`。
  - `R4HotspotDetector`：4-连通域、非背景过滤、时序 delta 候选、稳定排序、去重和 `max_candidates <= 20`。
- 增加 fixture 风格的单元测试，覆盖正常路径、空/无上一帧、非法输入、键盘模式、预算边界、稳定 ID 和负证据边界。
- 按 R4-M2 增加 `ProbePlanner`、反馈归因、tabu 上下文和 synthetic probe evidence recorder。
- synthetic loop 已串通 `detector → planner → feedback → tabu/evidence`，不调用真实环境。
- P0-D 增加短时序窗口、周期变化识别和 `dynamic_noise` 降权；周期动画不会稳定排在交互候选前面。
- P1-B 增加 before/after frame refs、requested action、反馈 outcome 和 JSONL evidence writer。
- P1-B 增加 2-D frame delta ratio 计算，并写入 probe outcome artifact。

## 当前证据等级

`L1`：离线/synthetic fixture 闭环通过；尚未宣称真实 ft09 L3。

本轮验证：

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 /tmp/run_r4_m2_pytest.py`：24 passed，exit 0；
- `ruff check ...`：通过，exit 0；
- `python3 -m py_compile ...`：通过，exit 0；
- `git diff --check`：通过，exit 0；
- 全量 pytest：exit 2，因既有导入/测试基线问题收集失败。

## 阻塞与未验证项

- 当前仓库既有导入链仍缺少 `lingjing_solo.exploration.click_sweep`，导致标准 pytest 在收集阶段失败；本轮未修改该既有问题。
- 尚未找到并读取真实 ft09 recording/schema，因此没有伪造真实帧或动作证据。
- 真实环境闭环和动作 payload 适配尚未开始；JSONL evidence 已支持离线写出，但尚未接入真实 runner。

## 下一动作

1. 将 JSONL evidence writer 接入真实动作调用前的 dry-run 适配层。
2. 接入一条真实 recording，冻结坐标和动作 payload schema。
3. 让独立维护者修复现有导入/测试基线，再运行全量测试。
