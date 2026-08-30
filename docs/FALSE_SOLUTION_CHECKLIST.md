# 假解防护清单（Script / Oracle 必勾）

上线任何离线脚本或搜索解之前：

1. **权威计数**：以 Arcade / harness 的 `levels_completed`（或 `GameState.WIN`）为准，不以「看起来格子对了」为准。
2. **state_key 完整性**：BFS / 影子模拟必须包含会改变转移的隐状态（ar25：选中对象、镜像轴、撤销栈相关；ls20：旋转角、颜色、步数预算、补给）。
3. **每关隔离**：前缀必须 `env.reset()` 后完整重放；禁止在脏环境上续搜。
4. **关卡边界**：通关后若需「任意一步」才 `next_level`，脚本要包含切换步，否则会卡关。
5. **回放校验**：`python scripts/verify_scripts.py <game>` 必须 `ok=true`，且 `levels_passed` 与脚本关数一致。
6. **线上一致性**：Kaggle notebook 必须打进 `planning/data/*.json`（`build_notebook.py` 已支持）；本地通、线上不通先查是否漏打包。
7. **密钥**：禁止把 API key 写进脚本/报告；只用 `ARC_API_KEY` / `KAGGLE_API_TOKEN` 环境变量。
8. **文档矛盾**：报告文首与 checklist 冲突时，以 `verify_scripts` / scorecard 证据表为准，并回写 `CURRENT_STATUS.md`。
