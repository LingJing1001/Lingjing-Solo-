# 同步与提交规范

## 目录角色

| 路径 | 角色 |
|------|------|
| `ARC-AGI-3-Kaggle-Starter/` | **Kaggle 提交权威**；`my_agent.py` + vendored `lingjing_solo` |
| `灵境战队AGI课题探索/` | GitHub 课题探索 / Φ-Field / adaptor；`lingjing_team_agent` 优先 import 这里 |
| `lingjing_solo/`（根下 sibling） | 历史副本；勿当权威，改完请与 Starter 对齐 |

## 改脚本后的同步

```powershell
$src = "ARC-AGI-3-Kaggle-Starter\lingjing_solo\planning"
$dst = "灵境战队AGI课题探索\lingjing_solo\planning"
Copy-Item "$src\script_bank.py" "$dst\script_bank.py" -Force
Copy-Item "$src\data\*.json" "$dst\data\" -Force
```

## GitHub 拉取节奏（课题探索）

1. 备份本地报告 / 未提交桥接改动  
2. `git fetch origin` → `git reset --hard origin/main`（或 merge）  
3. 重新拷入 ScriptBank / 桥接改动  
4. 跑 `verify_scripts.py` + 抽样 `benchmark_all_games.py`  
5. 更新根目录 `CURRENT_STATUS.md`

## 提交前检查

1. `python scripts/verify_scripts.py`  
2. 至少抽测：`play_local` / benchmark 对 `ls20`、`ar25`  
3. `.\scripts\submit.ps1`（内含 verify + `build_notebook`）  
4. 确认 notebook 含 `planning/data/*_scripts.json`

## 环境变量

- `KAGGLE_API_TOKEN` / `ARC_API_KEY` — 密钥  
- `LINGJING_SCRIPT_PLAN` — 临时整局覆盖脚本  
- `LINGJING_LS20_PLAN` — ls20 扁平计划覆盖  
- `LINGJING_USE_WMP=1` — 打开团队 agent 的 WMP bridge  
