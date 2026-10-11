# feature/aop-readiness-v2 合并、跑测与更新纪要（2026-10-12）

> 分支：`feature/aop-readiness-v2`  
> tip（撰写时）：`f294fe8`  
> 关联：`docs/AOP就绪v2整合评估_20261011.md`、`docs/双系统交榜提分升级_20261011.md`  
> 结论：**合并线稳住后，全量 25@800（AOP=off）scorecard 42.80 / 10 WIN**；相对旧基线 35.38 / 7 WIN 约 **+7.4 / +3 WIN**。涨分来自 plan@800 与 lp85 engine，**不是 AOP 接管**。

---

## 1. 合并情况

### 1.1 合并线（时间序）

| 提交 | 说明 |
|------|------|
| `70e1ae5` | 线上 tip：AOP counterfactual 训练/排序 |
| `67113e8` | cherry-pick 双系统路径接缝 + readiness 评估文档 |
| `db8060b` | 记录 dual-path remmerge |
| `f41f4f3` | **plan-protect**：校准 plan 防 engine/AOP 抢占 |
| `9387800` | perc 点击 + R3 时限 + 全量 25@42.8 文档 |
| `f294fe8` | wa30 GBK 日志修复 + IDDFS 空键盘兜底 |

### 1.2 合入内容一览

| 来源 | 是否保留 | 备注 |
|------|----------|------|
| AOP readiness v2（训练/评测/counterfactual） | ✅ | 打榜默认 `AOP=off` |
| 双系统相对路径 fail-closed | ✅ | 去掉 `C:/newtask-pi`、`F:/pro2` |
| plan-protect | ✅ | 校准/INLINE 局禁止入口抢占 |
| perc top-K 点击候选 | ✅ | hybrid-click / IDDFS/BFS 搜索 |
| `LINGJING_R3_TLIMIT` | ✅ | 全量跑默认 20s，避免 unknown 烧 500s |
| engine_solver（unknown 白名单） | ✅ | lp85 本机复现 8/8 WIN |
| AOP takeover | ❌ 默认关 | masked ~45% ≠ 可接管 |

### 1.3 合并期回归与修复

合并后曾出现全量 25@200 ≈ **31.2**、**sb26 WIN→0**（ACTION7 刷屏）。

根因：

1. `EngineSolver` 在 `choose_action` 入口抢占 plan 局  
2. AOP controller 默认自动加载并末端改写  

修复要点（`+plan-protect`）：

- `_PLAN_PROTECTED_GAMES`：ls20/ar25/ft09/tr87/vc33/sb26/r11l/cd82/tn36/sc25/bp35/wa30  
- `_ENGINE_SOLVER_GAMES` 收窄为 unknown-only  
- AOP 仅 `_aop_mode() != "off"` 时加载；plan/INLINE 跳过改写  
- 尊重 `ENGINE_SOLVER=off`；环境目录优先 Starter `environment_files`

---

## 2. 跑测结果

### 2.1 条件

```text
分支 tip:     feature/aop-readiness-v2 (+plan-protect+perc-r3cap…)
命令:         scripts/play_local.py --max-steps 800
环境:         AOP=off
              LINGJING_R3_TLIMIT=20
日志:         data/full25_percr3cap_800.log
BUILD_TAG:    …+aop-readiness-v2+plan-protect+perc-r3cap…
```

### 2.2 总览

| 指标 | 旧基线（@200） | 本次全量（@800） | 变化 |
|------|----------------|------------------|------|
| Aggregate scorecard | **35.38** | **42.80** | **+7.42** |
| 整局 WIN | **7 / 25** | **10 / 25** | **+3** |
| 校准 7 局（cd82/ft09/r11l/sb26/tn36/tr87/vc33） | 7 WIN | 7 WIN | 零回退 |

涨分主因：`@800` 解锁 **ls20 / ar25** 全通 + **lp85** engine 8 关全通。

### 2.3 逐局明细（全量 25@800）

| 游戏 | levels | actions | state | 备注 |
|------|--------|---------|-------|------|
| ar25 | 8 | 276 | **WIN** | INLINE plan |
| bp35 | 1 | 800 | NOT_FINISHED | L0 plan；L1+ 未补 |
| cd82 | 6 | 73 | **WIN** | 校准 plan |
| cn04 | 0 | 800 | NOT_FINISHED | unknown |
| dc22 | 0 | 800 | NOT_FINISHED | unknown |
| ft09 | 6 | 81 | **WIN** | INLINE plan |
| g50t | 0 | 800 | NOT_FINISHED | unknown |
| ka59 | 0 | 800 | NOT_FINISHED | unknown |
| lf52 | 0 | 800 | NOT_FINISHED | unknown |
| **lp85** | **8** | **79** | **WIN** | engine_solver |
| ls20 | 7 | 309 | **WIN** | INLINE plan（@200 会截断低估） |
| m0r0 | 0 | 800 | NOT_FINISHED | unknown |
| r11l | 6 | 130 | **WIN** | 校准 plan |
| re86 | 0 | 800 | NOT_FINISHED | unknown |
| s5i5 | 0 | 800 | NOT_FINISHED | unknown |
| sb26 | 8 | 124 | **WIN** | plan-protect 后恢复 |
| sc25 | 4 | 800 | NOT_FINISHED | L0–L3 plan；L4+ 未补 |
| sk48 | 0 | 800 | NOT_FINISHED | unknown |
| sp80 | 0 | 800 | NOT_FINISHED | unknown |
| su15 | 1 | 800 | NOT_FINISHED | CEAX 破 1 关 |
| tn36 | 7 | 103 | **WIN** | 校准 plan |
| tr87 | 6 | 122 | **WIN** | INLINE plan |
| tu93 | 0 | 800 | NOT_FINISHED | unknown |
| vc33 | 7 | 167 | **WIN** | 校准 plan |
| wa30 | 3 | 800 | NOT_FINISHED | L0–L2；L3 在线规划失败 |

### 2.4 抽样复验（修复后）

| 范围 | 条件 | 结果 |
|------|------|------|
| 7 校准局 | `@800` AOP=off | **7/7 WIN**，aggregate 100.0 |
| lp85 + ls20 | `@800` | 均 WIN |
| cn04 / dc22 | `@800` R3=40s | 仍 0 关；IDDFS 已有 clicks（不再 NameError） |
| wa30 | `@800`（GBK 日志修复后） | 仍 3 关；L3 规划本身失败（非日志问题） |

---

## 3. 本轮代码 / 文档更新

### 3.1 Agent（`integrations/ARC-AGI-3-Kaggle-Starter/agent/my_agent.py`）

同步：`vendor/ARC-AGI-3-Agents/agents/templates/my_agent.py`

| 标签 | 内容 |
|------|------|
| `+plan-protect` | 校准局防 engine/AOP 抢占；引擎白名单收窄 |
| `+perc-r3cap` | `_perc_click_candidates` / `_build_search_click_acts`；`LINGJING_R3_TLIMIT`；plan 局跳过 R3 |
| `+iddfs-hasclick-fix` | 修复 IDDFS `has_click` NameError；空 `available_actions` 键盘兜底 |
| `+wa30-gbk-fix` | 去掉日志 emoji，避免 Windows GBK 控制台丢弃有效 wa30 plan |

### 3.2 文档

| 文件 | 作用 |
|------|------|
| `docs/AOP就绪v2整合评估_20261011.md` | 整合评估持续更新 |
| `docs/aop-readiness-v2合并跑测与更新_20261012.md` | **本文**：合并线 + 全量明细 + 更新清单 |
| `docs/双系统交榜提分升级_20261011.md` | 提分原则（既有） |
| `docs/双系统接缝分支更新_20261011.md` | 接缝变更（既有） |

### 3.3 证据日志（可选入库）

| 文件 | 说明 |
|------|------|
| `data/full25_percr3cap_800.log` | 全量 25@800 完整输出（scorecard 42.80） |
| `data/full25_planprotect_800.log` | plan-protect 早期中断跑（不完整，仅对照） |

### 3.4 明确未改 / 不提交

- `environment_files/*/metadata.json`：仅 Windows 路径分隔符与下载时间噪音，**不入库**  
- AOP 默认仍 **off**；未开 takeover 打榜  
- sc25 L5+ / wa30 L3+ / bp35 L2 / unknown 破零：**尚未**形成新确定性 plan

---

## 4. 复现命令

```bash
cd integrations/ARC-AGI-3-Kaggle-Starter

# 校准 7 局
set AOP=off
.\.venv\Scripts\python.exe scripts\play_local.py ^
  --game cd82,ft09,r11l,sb26,tn36,tr87,vc33 --max-steps 800

# 全量 25（unknown 搜索限时）
set AOP=off
set LINGJING_R3_TLIMIT=20
.\.venv\Scripts\python.exe scripts\play_local.py --max-steps 800
```

---

## 5. 下一步（提分）

1. **sc25 L5+ / wa30 L3+ / bp35 L2**：补确定性 plan（最大边际）  
2. **unknown 破零**：加长 R3 或接 shadow 专解（40s 内 cn04/dc22 仍 0）  
3. **su15**：已有 1 关，挖第 2 关  
4. 每次改动后重跑全量 25@800，守住 **10 WIN / ≥42.8**  
5. AOP：继续收数；holdout 过线前禁止 authority

---

## 6. Kaggle 官网提交（2026-10-12）

| 项 | 值 |
|----|-----|
| Kernel | [liaojiaxuan/arc-prize-2026-smart-router-v2-dpl](https://www.kaggle.com/code/liaojiaxuan/arc-prize-2026-smart-router-v2-dpl) |
| Kernel version | **2**（COMPLETE） |
| Competition submit ref | **57079926** |
| Message | `aop-readiness-v2 plan-protect perc-r3cap local42.8/10WIN AOP=off` |
| 提交时刻状态 | `PENDING`（Phase B 排队计分；当日额度已用尽） |
| 本地对照 | scorecard **42.80** / **10 WIN** @800 |

查分：

```bash
cd integrations/ARC-AGI-3-Kaggle-Starter
.\.venv\Scripts\kaggle.exe competitions submissions arc-prize-2026-arc-agi-3
```

---

## 7. 一句话

**`feature/aop-readiness-v2` = counterfactual tip + 双系统接缝 + plan-protect + perc/R3 加固；全量 25@800 = 42.80 / 10 WIN；已提交 Kaggle kernel v2 / ref 57079926（PENDING）；打榜 AOP=off。**
