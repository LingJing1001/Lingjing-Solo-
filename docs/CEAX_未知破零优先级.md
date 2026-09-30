# CEAX 未知能力破零优先级

> 原则：**不用插件 / canned / ScriptBank**。以 CEAX 为底盘。  
> Agent: `ceax-primary-unknown-v3.6`

## 摸底（focus8 @300）

| 版本 | 非零 | 关数 | Aggregate(mean) | 备注 |
|------|------|------|-----------------|------|
| v3.3 | 5/8 | 6 | ≈1.96 | 基线 |
| v3.4 | 5/8 | 6 | ≈2.04 | 易局稳 |
| v3.5 | 6/8 | 7 | ≈2.05 | s5i5 L1 |
| **v3.6** | **7/8** | **8** | **≈2.26** | **+su15 L1**；仅 bp35 仍 0 |

| game | v3.6 |
|------|------|
| lp85/vc33/r11l/sb26 | L1 |
| tu93 | L2 |
| s5i5 | L1（滑条对齐） |
| **su15** | **L1（牵引 settle）** |
| bp35 | 0（平台节点+镜头，待攻） |

产物：`ui/static/bench_ceax_v36b_focus8.json`

## v3.6 通用增强（无局名硬编码）

1. **Attract/pull**：仅 `ACTION6+ACTION7` 纯点击局；稀有小球 → 目标盒短绳牵引  
2. **门控**：hybrid/键盘局禁用 pull（防误触发掉分）  
3. **Hybrid mutate**：周期点中屏可切换节点后 L/R 走位（bp35 方向，尚未破零）

## 入口

```powershell
.\.venv\Scripts\python.exe scripts\bench_ceax_priority.py --max-steps 400
agent/ceax_unknown_agent.py   # BUILD_TAG=ceax-primary-unknown-v3.6
```

## 下一刀：bp35

- 机制：L/R 平台 + 点击 `Q` 类节点清障/拉人；镜头跟随，屏外节点点不到  
- 需要：可见节点优先点击 → 走位穿洞上行 → 抵达稀有终点 gem  
