# M1 整改 · Step 2：CEAX/画面感知链强化 —— 实验结论

> 2026-10-13 ｜ 对应 `M1整改方案_20261013.md` 第 ② 步
> 目标（拟定）：闭卷 25 局聚合分 **0.62 → ≥5**，pytest 全绿
> **结论：目标未达成，且经 4 组对照实验证明「不是调参问题，是能力缺口」。**

---

## 0. 头条

**Step 2 判定：未达标（诚实标注，公约铁律 4）。**

闭卷分仍是 **0.6200**。四组针对性对照实验**全部为负**，
说明 0.62 这个地板来自 CEAX 的**通用能力上限**，不是配置/策略调参能撬动的。

---

## 1. 先厘清计分口径（决定"值不值得做"）

`arc_agi/scorecard.py:185-206`：

```
单局分 = Σ(关卡分_i × 关卡序号_i) / Σ(关卡序号_i)     ← 关卡序号作权重（1..N）
关卡分_i = (基准步数 / 实际步数)² × 100，上限 115
总分   = 25 局单局分的算术平均
```

**关键推论**：当通关效率 ≥ 基准时，单局分 ≈ **加权通关比例**，
某局只过第 1 关（权重 1）而总局权重 Σ(1..N) 时，单局分 = 100/Σ(1..N)。

| 局 | 关卡数 N | Σ权重 | 只过 L1 的单局分 | 过 L1+L2 | 过 L1..L3 |
|---|---|---|---|---|---|
| r11l | 6 | 21 | 4.76 | 14.3 | 28.6 |
| tn36 / vc33 | 7 | 28 | 3.57 | 10.7 | 21.4 |
| lp85 / sb26 / s5i5 | 8 | 36 | 2.78 | 8.33 | 16.7 |
| su15 | 9 | 45 | 2.22 | 6.67 | 22.2(含L3) |
| lf52 | 10 | 55 | 1.82 | 5.45 | 10.9 |

⇒ 要打到 **总分 5**，需要 Σ单局分 = 125。
现状 Σ ≈ 13.3（6 局各过 1 关）。
**等价于「7 个 click 局全部过到 L1–L3」——那不是调参，是造一个通用 click 求解器。**
（这也是本步把目标从 5 下调的依据，见 §5。）

---

## 2. 四组对照实验（全部为负）

跑法：`LINGJING_CLOSED_BOOK=2`（strict 闭卷口径），25 局 × 400 步。

| # | 假设 | 改法 | 结果 | 判决 |
|---|---|---|---|---|
| 1 | 150 步早停/RESET 浪费预算 | `CEAX_GIVEUP_STEPS=0` | agg **0.5625**（↓0.058），关卡 5→7 | ❌ 更差：ls20 → `GAME_OVER`、vc33 掉到 0.01 |
| 2 | keyboard_click 局被误路由到点击 | 强制从 valid 里剔除 `ACTION6` | ar25 0关→0关；ka59 0关→0关 | ❌ 无变化 |
| 3 | click 局锁死在无效点击循环 | 禁 sticky + 强制纯探索 | lp85 1→1；tn36/lf52 0→0；**s5i5 1→0** | ❌ 无变化／更差 |
| 4 | R4 热点检测未接入 | `ARC_CLICK_HEAT=1`（`click_heatmap_v3.npz`） | lp85 仍 1 关但 **105 步**（原 12 步）；tn36 仍 0 | ❌ 无变化／更慢 |

产物：`state/bench_cb_strict_nogiveup_20261013.json`、
`state/_probe_force_kbd.py`、`state/_probe_explore_only.py`、
`state/_probe_valid_actions.py`、`state/_cb_*_console.log`。

### 实验 1 的额外信号（重要）
去掉早停后**关卡数反而变多（5→7）但总分下降**：
- tu93、vc33 多过了 L1，但 vc33 那一关效率极差（关卡分 ≈0.0145 → 单局分 0.01）；
- ls20 在没有 RESET 的情况下**打光命 → `GAME_OVER`**。

⇒ **"多过关"和"多拿分"是两件事**：计分对步数极度敏感（平方项）。
决策链真正的损失大头是**低效通关**，不是"过不了关"。

---

## 3. 两个结构性发现（比实验更重要）

### 3.1 ar25 报的可用动作是全集，CEAX 因此误判模态
实测首帧 `available_actions`：

| 局 | tags | 首帧 available_actions |
|---|---|---|
| **ar25** | keyboard_click | **[1,2,3,4,5,6,7]**（全集） |
| g50t | keyboard | [1,2,3,4,5] |
| lp85 | click | [6] |
| wa30 | keyboard | [1,2,3,4,5] |

ar25 的动作表（`_AR25_LEVEL_ACTIONS`）**只用 1–5，一次点击都没有**，
但环境把 `ACTION6/7` 也报成"可用"。CEAX `choose()` 见 `has_click=True`
就走进点击路径（`ceax_controller.py:1070`），**整局再没碰过键盘**。
25 局里 **13 局是 keyboard_click**，全部踩这个坑。

> 这是**真缺陷**，但实验 2 证明：修掉它也解不开 ar25（机制不是"走格子"）。
> 所以缺陷归缺陷，不能当提分理由。

### 3.2 仓里唯一的 AR25 感知模块**违反公约**
`lingjing_solo/perception/ar25_encoder.py` 文档字符串自己写着：

> *"Primary path is offline source parsing of `environment_files/ar25/*/ar25.py`"*
> *"Frame-only decoding is intentionally out of scope here"*

即：它直接**读引擎源码**（`DEFAULT_SOURCE` → `ar25/0c556536/ar25.py`）反解关卡，
并且硬编码了官方混淆 tag（`TAG_TARGET = "0001sruqbuvukh"` 等）。
`Ar25Field`（`BOUNCE_LIMIT == 12` "matches engine"）同理。

⇒ 公约《M1》白纸黑字「**不背答案、不看引擎内部**」——
**现有 AR25 感知链 100% 不合规**，且它的帧解码入口 `encode_from_parts`
**没有任何调用方**（等于没接进链路）。
**结论：仓里不存在任何"合法且已接通"的 AR25 感知。**

---

## 4. 为什么这些实验都没用（一句话）

CEAX 是**手写启发式**探索器：它能在"点一下就有反馈"的 click 局里瞎撞到 L1，
但**没有世界模型**——不会从像素里学规则、不能反推机制。
早停/锁死/热点这些都是**调度层**问题；调度得再好，
下层**根本没有可调度的推理**。所以修调度 = 0 收益。

---

## 5. 建议的 Step 2 重定义（三选一，请拍板）

| 方案 | 做什么 | 代价 | 期望 |
|---|---|---|---|
| **A · 窄而正**（推荐，最贴公约） | 目标从「25 局≥5」改为「**AR25 L1 闭卷通关**」。写一个**帧感知**版 Ar25Obs（禁止读源码）→ 接 `Ar25Field.enumerate_covering_configs` → pathify 出动作 | 中～大，真开发 | 兑现公约原文的 M1 全部目标；分不高但**合规且可解释** |
| **B · 治标** | 冻结 0.62 为诚实基线，直接转 Step③（CI 门禁反向 + 口径对齐） | 小 | 拿不到分，但把"不许拿罐头充数"焊死 |
| **C · 广而难** | 继续投通用 CEAX 能力（世界模型 / 规则归纳） | 很大，多会话 | 唯一能同时抬本地闭卷分和官方分的方向，但短期看不到数 |

**我的建议**：先做 **B**（半天，零风险，把口径钉死），再排 **A**（公约的原目标，可验收）。
C 不该在没有 A/B 的情况下开——否则又是一个"无法解释的数"。

---

## 6. 复现

```bash
cd integrations/ARC-AGI-3-Kaggle-Starter

# 实验1：不早停全量
LINGJING_CLOSED_BOOK=2 CEAX_GIVEUP_STEPS=0 .venv/Scripts/python.exe \
  scripts/benchmark_all_games.py 400 --agent agent/my_agent.py \
  --out ../../state/bench_cb_strict_nogiveup.json --label cb-strict-nogiveup

# 实验2/3：单点探针（都需在 import 前设 env）
python ../../state/_probe_force_kbd.py ar25 ka59
python ../../state/_probe_explore_only.py lp85 tn36 lf52 s5i5

# 实验4：热点
ARC_CLICK_HEAT=1 python ../../state/_cb_run.py lp85 strict
```
