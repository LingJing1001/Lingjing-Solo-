# 当前权威状态（唯一分数/能力真相源）

> 更新日期：2026-08-30  
> 以本文件为准；其它报告若矛盾，先更新本页。

## 分数基线

| 入口 | Aggregate（约） | 已验证关数 | 备注 |
|------|-----------------|------------|------|
| `agent/my_agent.py`（Starter + ScriptBank） | **≈0.540**（全 25 游戏，2026-08-30） | **合计 3**：ls20=2，ar25=1 | **Kaggle 提交默认入口**；详见 `ARC-AGI-3-Kaggle-Starter/Starter整合全游戏摸底报告.md` |
| `agent/lingjing_team_agent.py` | 同脚本能力量级 | 同左 | 课题探索桥接 |
| AR25 报告作者 | 声称 8/8 | 我们仅复现 **L1** | 缺 L2–L8；有限 BFS 曾报 L2 假阳性（完整重放仍 L=1），已按清单拒绝入库 |

## 已落地能力

- [x] **ScriptBank**：`lingjing_solo/planning/script_bank.py` + `data/<game>_scripts.json`
- [x] **ls20 脚本** L1+L2（Starter / 课题探索 / sibling 已同步）
- [x] **ar25 脚本** L1 = `ACTION3×5 + ACTION2×10`（本地 Arcade 复现 `levels_completed=1`）
- [x] **my_agent / lingjing_team_agent / arc_adaptor** 优先走 ScriptBank
- [x] **offline_level_bfs.py**：多游戏、seed-prefix、可选 ACTION6 点击网格、merge
- [x] **verify_scripts.py**：上线前校验「脚本必须抬高 levels_completed」
- [x] **build_notebook.py**：打包 `planning/data/*.json` 进 Kaggle notebook
- [x] **submit.ps1**：提交前跑 verify；密钥只走环境变量
- [x] **WMP bridge**（可选）：`LINGJING_USE_WMP=1` → `agent_wmp_bridge.py`
- [x] **假解防护清单**：见 `docs/FALSE_SOLUTION_CHECKLIST.md`
- [x] **同步规范**：见 `docs/SYNC_AND_SUBMIT.md`

## 仍缺（需继续挖分）

| 项 | 状态 | 下一步 |
|----|------|--------|
| ar25 L2–L8 | 缺序列 | `offline_level_bfs.py ar25 25 --seed-prefix ACTION3x5,ACTION2x10 --merge`；或向报告作者要 `arc_solutions.json` |
| ls20 L3+ | 缺 | 同工具加深 / 专用启发式 |
| 其它 23 游戏 | 0 | 逆向 + 脚本固化 |
| WMP 真接入出分 | 可选开 | 符号表仍粗，勿指望立刻涨分 |

## 一键命令

```powershell
cd ARC-AGI-3-Kaggle-Starter
.\.venv\Scripts\python.exe scripts\verify_scripts.py
.\.venv\Scripts\python.exe scripts\benchmark_all_games.py --agent agent/my_agent.py --max-steps 400
# 提交（会 verify + build notebook）
.\scripts\submit.ps1 -Message "ScriptBank ls20+ar25-L1"
```

## Commit / 路径注意

- 权威脚本 JSON：`ARC-AGI-3-Kaggle-Starter/lingjing_solo/planning/data/`
- 课题探索镜像：`灵境战队AGI课题探索/lingjing_solo/planning/data/`（改完请互拷）
- 团队 agent import 优先课题探索包；ScriptBank 两边都要有
