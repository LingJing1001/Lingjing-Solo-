# Starter 整合全游戏摸底报告

- **日期**：2026-08-30  
- **入口**：`ARC-AGI-3-Kaggle-Starter/agent/my_agent.py`  
- **能力**：ScriptBank（ls20 L1–L2 + ar25 L1）→ Lingjing Solo（ls20 solver + Φ-field）  
- **步数上限**：400 / 局  
- **原始 JSON**：`ui/static/games_benchmark_scriptbank.json`

## 总分

| 指标 | 数值 |
|------|------|
| **Aggregate（官方 scorecard 均值）** | **≈ 0.540** |
| 有分游戏 | ls20（≈10.71）、ar25（≈2.78） |
| **总过关数** | **3**（ls20×2 + ar25×1） |
| 通关局数 WIN | 0 |
| 零关游戏 | 23 / 25 |
| 耗时 | ≈ 8 分钟 |

## 有分明细

| game | levels | score | actions | state |
|------|--------|-------|---------|-------|
| **ls20** | **2** | **10.71** | 263 | NOT_FINISHED |
| **ar25** | **1** | **2.78** | 217 | NOT_FINISHED |
| 其余 23 | 0 | 0 | ~200–401 | GAME_OVER / NOT_FINISHED |

## 整合说明

已在 Starter 内打通并作为默认提交路径：

1. `lingjing_solo/planning/script_bank.py` + `data/ls20_scripts.json` / `ar25_scripts.json`  
2. `agent/my_agent.py` 优先 ScriptBank，再回退 Solo  
3. `scripts/verify_scripts.py` 校验通过后再摸底  
4. `build_notebook.py` 会打包脚本 JSON 进 Kaggle notebook  

相对此前仅 ls20≈2 关、无 ar25 的摸底，aggregate 从约 **0.43** 提到约 **0.54**（+ar25 L1）。

## 结论

- **能稳定出分的关**：ls20 前 2 关 + ar25 第 1 关，共 **3 关**。  
- 其余游戏无脚本、通用策略未过关。继续涨分依赖 ar25 L2+ / ls20 L3+ / 新游戏脚本。  
