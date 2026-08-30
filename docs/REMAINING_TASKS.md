# 剩余人工任务（工程已打底，需继续挖分）

工程侧「完善」已完成：ScriptBank、双入口接线、校验、提交打包、规范文档、WMP 可选桥。

下列项 **依赖解序列 / 逆向**，无法仅靠框架自动完成：

| ID | 任务 | 验收 | 建议人天 |
|----|------|------|----------|
| R1 | 向 AR25 报告作者要 `arc_solutions.json` 或 L2–L8 ACTION 串 | `verify_scripts.py ar25` → levels_passed≥6 | 0.5（有文件）/ 3–5（自搜） |
| R2 | ls20 L3+ 离线挖脚本 | ls20 levels≥3 | 2–4 |
| R3 | 再选 1～2 个游戏做 L1 脚本 | 全游戏摸底 aggregate 上升 | 每个 3–7 |
| R4 | 全 25 游戏重摸底（`benchmark_all_games.py`） | 更新 `CURRENT_STATUS.md` | 0.5 |
| R5 | 正式 `submit.ps1`（需 Kaggle token） | 线上分数 ≈ 本地 | 0.5 |

命令提示见 `CURRENT_STATUS.md` / `docs/SYNC_AND_SUBMIT.md`。
