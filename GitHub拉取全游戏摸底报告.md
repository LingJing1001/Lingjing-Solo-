# GitHub 最新拉取 + 全游戏摸底报告

- **日期**：2026-08-30  
- **仓库**：`LingJing1001/Lingjing-Solo-` → `灵境战队AGI课题探索`  
- **同步提交**：`818dd28`（`docs: bundle ARC Lingjing adaptor for reproduction`）

---

## 1. 拉取与整合

| 动作 | 结果 |
|------|------|
| `git fetch` / `reset --hard origin/main` | 成功，HEAD=`818dd28` |
| 远程新增 | `LS20Solver`、`level1_verified_route`、`arc_adaptor`、action-diff 等 |
| Starter 桥接 | 更新 `ARC-AGI-3-Kaggle-Starter/agent/lingjing_team_agent.py`：接入官方 adaptor 同款 LS20 路线（L1 验证 + L2 bootstrap），非 ls20 仍走 Φ-Field |

本地 v10 嵌入文件已从 package 清掉（随 hard reset）；`lingjing_solo_v10/` 与测试报告仍保留在目录旁。

---

## 2. 全游戏摸底（25×400 步）

| 指标 | 结果 |
|------|------|
| Aggregate | **0.429** |
| 总通关数 | **2** |
| 有分游戏 | **ls20 仅**（L=2，局分 **10.71**） |
| 其余 24 局 | L=0 / score=0 |

输出：`ARC-AGI-3-Kaggle-Starter/ui/static/games_benchmark_team.json`  
标签：`team-818dd28-ls20`

---

## 3. 对照

| 版本 | Aggregate | 关数 |
|------|-----------|------|
| 拉取前课题探索 | 0.00 | 0 |
| **本次 818dd28 + 桥接** | **0.43** | **2（ls20）** |
| Starter（脚本，此前） | ~0.14～更高（视版本） | ls20 曾到 2 |

---

## 4. 结论

- GitHub 最新 LS20 验证路线经桥接后，全游戏 **能过 ls20 两关**，总分从 0 提到约 **0.43**。  
- 其它 24 个环境仍无专用求解，分数为 0。  
- 下一步若要冲更高 aggregate：继续挖其它游戏短解，或把 Starter 的多关脚本/补给策略合进课题探索。
