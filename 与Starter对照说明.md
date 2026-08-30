# 课题探索版 vs ARC-AGI-3-Kaggle-Starter 对照说明

> 日期：2026-08-28  
> A = `灵境战队AGI课题探索`（GitHub Lingjing-Solo-）  
> B = `ARC-AGI-3-Kaggle-Starter`（本地比赛工程 + 增强 Agent）

---

## 1. 哪个更强？更好？

| 维度 | 更强的一方 | 说明 |
|------|------------|------|
| **过关 / 榜分能力** | **B · Starter 更强** | 全游戏摸底：B ≈ **0.14 分、1 关**；A = **0 分、0 关** |
| **架构清晰 / 论文叙事** | **A · 课题探索略干净** | 五层管道文档完整，适合课题汇报 |
| **能直接提交 Kaggle** | **B** | 自带官方 harness、`play_local`、notebook 打包 |
| **综合：比赛要提分** | **用 B 为主** | A 是骨架；B 才是「能跑、能过关」的版本 |

**结论：**

- **打比赛、提分 → Starter（B）更好、更强。**  
- **写报告、讲框架 → 课题探索（A）文档更整齐。**  
- **正确路线：以 B 为作战主干，把 A 的叙事/层命名对齐进去；不要反过来用 A 替换 B。**

---

## 2. 摸底成绩对比（同条件：25 局 × 400 步）

| 指标 | A 课题探索 | B Starter |
|------|------------|-----------|
| 总分（平均） | **0.00** | **≈ 0.14** |
| 总过关数 | **0** | **1**（ls20 L1） |
| 有分游戏 | 无 | 仅 ls20（单局 ≈ 3.57） |
| 典型 ls20 行为 | 一直向上 `ACTION1` | BFS + 旋转台，约十几步过 L1 |
| 耗时 | ~36s（策略极简） | ~500s（有求解器计算） |

数据文件：

- A：`ARC-AGI-3-Kaggle-Starter/ui/static/games_benchmark_team.json`  
- B：`ARC-AGI-3-Kaggle-Starter/ui/static/games_benchmark.json`

---

## 3. 结构差异（核心）

| 模块 | A 课题探索 | B Starter |
|------|------------|-----------|
| Agent 编排 | 五层：感知→场→探索→轻量规划→LLM 反思 | 六角色 + **优先专用 solver** |
| `ls20_solver` | **无** | **有**（键盘盘面 BFS / 旋转台） |
| `discrete_nav` | 无 | 有（通用四向导航） |
| 点击探索 | 默认关鼠标 | `BubbleClickPlanner`，可点 ACTION6 |
| 动作名 | UP/DOWN/LEFT/RIGHT/SPACE | ACTION1–6（官方） |
| 帧接入 | 偏 `.grid` | 原生 `FrameData.frame` |
| 官方 Agent 基类 | 需外层桥接 | `agent/my_agent.py` 已接好 |
| 评测工具 | 自测脚本为主 | `play_local` / 全游戏摸底 / 看板 / 提交流水线 |

决策优先级对比：

```text
A 课题探索（每步）
  反思(LLM?) → 轻量搜索 → 探索评分 → valid[0] 兜底
  └─ 实测常退化成一直选第一个动作（UP）

B Starter（每步）
  像 ls20？→ ls20_solver（专用）
  离散关？→ discrete_nav
  点击关？→ bubble_click
  否则 → 因果搜索 / 探索 / 顾问
```

**差别本质：**  
A 停在「通用世界模型框架」；B 在框架外加了「会过关的专用策略」。过关靠的是后者。

---

## 4. 各自适合干什么

### A · 课题探索（GitHub）

**适合：** 课题答辩、架构图、五层闭环说明、开源展示。  
**不适合：** 直接当当前最强提交、指望 clone 就能冲榜。

### B · Kaggle-Starter

**适合：** 本地全游戏摸底、迭代提分、打包提交。  
**短板：** 目前仍只有 ls20 过 1 关；离榜上 5.x / 6+ 还差「多游戏广度」。  
**文档 / 叙事** 不如 A 集中，但**实战能力明显更强**。

---

## 5. 怎么改进？（以 B 为主干）

### 原则

1. **作战代码以 Starter 为准**，不要退回只用课题探索。  
2. **课题探索仓库**继续作「公开叙事 + 同步核心思想」；把 B 里验证过的模块回灌进去。  
3. 提分靠 **多游戏各过 1 关**，不是死磕 ls20 L2（L2 单关对总分约 +0.2～0.3）。

### 具体步骤

| 优先级 | 做什么 | 预期 |
|--------|--------|------|
| P0 | 保持 B 的 `ls20_solver`，修 L2 可后置 | 保住现有 0.14 |
| P0 | B 上打开/加强点击探索，先让 `vc33` 等出现 levels>0 | 总分可见上涨 |
| P1 | 把 ls20 套路扩到其他 `keyboard` 关（wa30/g50t/tr87） | 多局有分 |
| P1 | 修通用层「卡住仍重复同一动作」 | A/B 通用探索都受益 |
| P2 | 将 B 的 solver + 官方动作/帧适配 **回灌** 到课题探索仓库 | 开源与实战一致 |
| P2 | 每次改完跑全游戏看板验收 | 用 `levels` / 总分说话 |

推荐命令（Starter 目录下）：

```powershell
# 打比赛用的摸底（B）
.\.venv\Scripts\python.exe scripts\benchmark_all_games.py 400 --agent agent/my_agent.py --out ui/static/games_benchmark.json --label "Starter"

# 看板
.\.venv\Scripts\python.exe scripts\games_benchmark_ui.py --port 8781
```

---

## 6. 一句话对照

| 问题 | 回答 |
|------|------|
| 有什么区别？ | A=框架骨架；B=框架 + 比赛适配 + ls20 专用解题 + 点击探索 |
| 哪个更强？ | **B 更强（能过关、有分）** |
| 哪个更好？ | 比赛选 **B**；写课题故事可引用 **A**，但代码以 B 为准 |
| 怎么改进？ | 在 B 上扩多游戏过关，再把成果同步回 A，而不是用 A 替换 B |

---

## 7. 相关文件

| 内容 | 路径 |
|------|------|
| 课题探索仓库 | `灵境战队AGI课题探索/` |
| Starter Agent | `ARC-AGI-3-Kaggle-Starter/agent/my_agent.py` |
| 课题探索桥接 | `ARC-AGI-3-Kaggle-Starter/agent/lingjing_team_agent.py` |
| 零关诊断报告 | `灵境战队AGI课题探索/全游戏摸底诊断报告.md` |
| 本对照说明 | `灵境战队AGI课题探索/与Starter对照说明.md` |
