# M1 整改 · Step 2B：闭卷 tu93 帧感知求解（已完成）

> 2026-10-13 ｜ 路线 A（窄而正）｜ 承接 Step2A 的闭卷 AR25 帧感知求解
> 目的：把**真·闭卷能力**从 0.7312 再抬一档，且**不引入任何新罐头**

---

## 0. 头条

| 指标 | 改前 | 改后 | Δ |
|---|---|---|---|
| **闭卷 headline（CB=2 聚合）** | **0.7312** | **1.1756** | **+0.4444（+60.8%）** |
| tu93 单局分 | 2.2222（0 关，部分分） | **13.3333（3 关）** | **+11.1111** |
| tu93 level_actions | `[18, 307, …]` | `[18, 11, 26, 345, …]` | 新增 L2/L3 |
| 其余 24 局 | —— | **逐局分数完全不变** | 0 |
| 罐头分（canned） | 52.5968 | 52.5968 | 0（未触碰） |

> **为什么聚合只涨 0.4444**：聚合分 = **25 局单局分的平均值**。tu93 单局 +11.1111 ⇒ 聚合 +11.1111/25 = **+0.4444**。
>
> **为什么不是 +0.5333**：改前 tu93 在 strict 下**已经吃到 L1 的部分分 2.2222**
> （`level_scores[0]=111.4198`、`level_actions[0]=18`，来自 CEAX 的键盘技能，见 §1）。
> 所以本轮**净增的是 L2 + L3**：(2+3)/45×100 = 11.1111（45 = 1+…+9，tu93 共 9 关）。
> headline 的净增量是 **+0.4444**，不是 +0.5333 —— 口径上宁可少报。

**关键**：这 0.4444 是**真·闭卷能力**的净增量——不看引擎、不背答案、不含任何新硬编码表。

---

## 1. 问题：tu93 之前能拿多少

tu93（TU93）是 **64×64 像素、3 px/格的迷宫游戏，共 9 关**，`baseline_actions=[19,16,34,42,123,80,14,23,111]`。

改前 strict（CB=2，层 2/3/4 全关）下 tu93 的表现（`state/_ab_one.py state/_pre_tu93_cb_my_agent.py tu93 strict`）：

```
step=0   L=0 ACTION4 ceax_kbd:ACTION4
step=1   L=0 ACTION2 ceax_kbd:ACTION2
step=2   L=0 ACTION2 ceax_kbd:ACTION2
         ceax-reset L1 (level transition)
step=18  L=1 ACTION3 ceax_kbd:ACTION3   skills=3
…
step=150 L=1 … ceax-giveup gid=tu93 L=1 连续 151 步零进展（阈值 150）→ RESET 重开换起点
RESULT levels=0 actions=800 state=NOT_FINISHED score=2.2222 level_actions=[18, 307, 0, …]
```

- **L1 被 CEAX 的键盘技能蒙到**（18 拍，恰好等于最短路径长度），计分卡吃下 `1/45×100 = 2.2222`；
- 之后 `ceax-giveup150` 触发 **RESET 重开**，`levels_completed` 被抹回 0，但已入账的 L1 分数还在；
- **L2 起彻底卡死**（307 拍零进展，然后反复 RESET）。

结论与 Step2A 相同：CEAX 是手写启发式、**没有世界模型**，不能从像素反推迷宫结构。要吃下 L2/L3，必须给 tu93 造一条纯帧感知链路。

---

## 2. 成果一：tu93 世界模型（全部由观测建立）

`lingjing_solo/closed_book/tu93_frame.py` —— 只吃 `FrameData`，**不读 `environment_files/`、不读引擎源码、不读混淆 tag**。
所有结论来自 `state/_probe_tu93_*.py`（30 个探针）。

### 2.1 几何

| 项 | 值 | 怎么得到的 |
|---|---|---|
| 帧 | 64×64，int8 调色板索引 | `_probe_tu93_raw.py` 原始像素 dump |
| 格 | **3 px/格** | 墙/目标边界恒落在 3 的整数倍上（`_probe_tu93_align.py` 全 9 关审计） |
| **节点** | 格坐标 **(奇, 奇)** | reset 帧玩家 bbox = (15,17,15,17)，完美落格 |
| 步长 | 一个动作 = 一个节点 = 2 格 = **6 px** | `_probe_tu93_walk.py` 定向行走 |
| 晶格相位 | **逐关不同**，每帧自校准 | `_probe_tu93_settled.py`：settle 后相位 L0 奇列 / L1、L4 偶列… |

### 2.2 地形（按格颜色）

| 色 | 含义 |
|---|---|
| 0 | 走廊 / 节点（玩家 9、目标 14、门 12/13、宝石 11/15 也画在节点上，用各自的色） |
| 2 | **连通格**（两个相邻节点之间的那一格，可通行） |
| 5 | **虚空**：堵住，也填满迷宫外 |
| 8 | **危险物**（见 2.4） |

**移动规则**：动作 A1=上(row−2) / A2=下(row+2) / A3=左(col−2) / A4=右(col+2)，
合法 **iff 两个节点之间的那个连通格不是虚空**。

### 2.3 两个必须处理的“读数陷阱”

1. **渲染滞后 1 拍**：`env.step` 返回的精灵位置落后一个动作（绘制动画没跟上）。
   → 求解器**自维护位置**，用解析出的迷宫**前向模拟**推进；帧读数只用于纠偏
   （读数必须等于「预测节点」或「刚离开的节点」，否则重新同步）。
2. **换关第一帧是上一关的滞留帧**（同样的滞后）。→ 检测到 level 变化时先发一个 **A5**，
   把新帧推出来再解析。

另外：**A5 是真 no-op**（framediff=0），而且**不推进游戏时钟** —— 所以它**不能**用来“等”巡逻物。

### 2.4 色 8 危险物（L1/L2 的核心机制）

`_probe_tu93_c8.py` / `_probe_tu93_c8shape.py` / `_probe_tu93_notch.py` / `_probe_tu93_entry.py`
/ `_probe_tu93_l2.py` / `_probe_tu93_l2dir.py` / `_probe_tu93_l2px.py` 逐条钉死：

- 危险物是 **3×3 的色 8 块，里嵌一颗异色像素**（多为色 15）；**那颗像素的位置给出“朝向”**。
- **从它面向的一侧进入 = 致死**；从别的侧进入 = **踩上去把它清掉**。
- **它正前方 2 格的节点致命**（“视线”致死），除非这个危险物已经被清掉。

因此 BFS 的状态必须是 **(节点, 已清危险物集合)** —— 直接把它当墙会导致永久无解
（目标被封在视线后）。危险物 ≤3 个 ⇒ 最多 8 种组合，状态空间很小。

### 2.5 色 15 巡逻物（L3）

`_probe_tu93_l3move.py`：L3 有一个色 15 物体**沿走廊往返巡逻**（观测到沿 col 13 上下），
**速度与玩家相同**（一动作一节点）。正面撞上必死，而 A5 又不推进时钟 ⇒ 要穿过去必须做
**时空联合规划**（预测它的相位、找交错窗口）。

---

## 3. 成果二：闭卷求解器

`lingjing_solo/closed_book/tu93_solver.py`，对外只有两个方法（与 `ar25_solver.py` 同构）：

```python
class Tu93ClosedBookSolver:
    def reset(self) -> None: ...
    def next_action(self, frame, levels) -> tuple[int, str] | None: ...
```

每拍：

1. level 变化 → 发 **A5**（把换关后的滞留帧顶掉），清空位置估计；
2. `parse_frame` → 晶格相位 + 节点地形 + 玩家/目标 + 危险物朝向；
3. 滞后补偿：读数 ∈ {预测位置, 上一位置} 则信任自维护位置，否则重同步；
4. 有巡逻物 → **原地 A5**（保守停滞，见 §7）；
5. 否则 `bfs_first_action(cells, pos, targets, facing, danger)`（**带状态 BFS**，允许“先踩敌人再通过”）；
6. 无路 → 能退就退，否则任一合法步；全无 → A5。

解析失败 → 返回 `None`，**正常落回 CEAX**，不改变既有行为。

### 离线证明（完全不用 agent）

两个独立的闭环验证脚本，都驱动**真实环境**、都不读罐头表：

```
$ python state/_verify_tu93_solver.py 400        # 直接用发布的求解器类
  >> level 0 -> 1 at step 18
  >> level 1 -> 2 at step 29
  >> level 2 -> 3 at step 55
RESULT levels=3 state=NOT_FINISHED steps=400 resyncs=0
```

`state/_verify_tu93_cb.py`（同逻辑的早期 harness）给出**逐拍一致**的结果。
步数 18 / +11 / +26 与 `baseline_actions[0..2] = 19/16/34` 同量级，符合“最短路径”预期。

---

## 4. 成果三：接入 agent（默认主链零回归）

**改动**：`agent/my_agent.py` 在 `_choose_action_inner` 的闭卷路径插入一段分支
（补丁脚本 `state/_tu93_cb_patch.py`，纯 ASCII + 显式 UTF-8，CRLF 保持；
改前备份 `state/_pre_tu93_cb_my_agent.py`，改后冻结快照 `state/_tu93_cb_my_agent.py`）：

```python
# --- CLOSED-BOOK: frame-only tu93 solver (no baked answers) ---
if _CLOSED_BOOK and gid == "tu93":
    ...   # 懒加载 Tu93ClosedBookSolver；失败即停用；异常一律吞掉
```

**精确 diff = +26 行、单 hunk**，插在 AR25 闭卷分支之后、EngineSolver 之前
（文件 129136 → 130343 B，`ast.parse` OK，CRLF 计数校验通过）。

- 只在 `_CLOSED_BOOK` 打开时生效 ⇒ **off 模式逐字节无影响**；
- 失败/解析不出 → 返回 `None` → 落回 CEAX；
- 与既有 INLINE 的 `self._tu93_idx/_tu93_level/_tu93_exhausted` 完全不相干（新属性 `_tu93_frame_solver`）。

### 零回归实测（单局，全新 scorecard）

| 模式 | 关卡 | 单局分 | 动作 | 状态 |
|---|---|---|---|---|
| `tu93 off`（主链 INLINE） | **9/9** | —— | **185** | **WIN** |
| `tu93 strict` 补丁**前** | 0/9 | 2.2222 | 800 | NOT_FINISHED |
| `tu93 strict` 补丁**后** | **3/9** | **13.3333** | 800 | NOT_FINISHED |
| `ar25 strict`（回归检查） | 1/8 | —— | 800 | NOT_FINISHED |

`off` 的 **185 拍与既有罐头基线逐拍一致**；`ar25` 闭卷分支未受影响。

### 整轮 25 局 A/B（`benchmark_all_games.py 400`）

```
改前 strict : aggregate=0.7312  levels=6   (state/bench_closed_book_strict_ar25_20261013.json)
改后 strict : aggregate=1.1756  levels=9   (state/bench_closed_book_strict_tu93_20261013.json)
```

逐局 diff（并集为“有变化的局”）：

| game | 改前 (关卡, 分) | 改后 (关卡, 分) | Δ |
|---|---|---|---|
| tu93 | (0, 2.22) | (3, 13.33) | **+11.1111** |
| 其余 24 局 | —— | **完全相同（bit-for-bit）** | **0** |

即 **唯一变化就是 tu93**，无任何副作用。

---

## 5. 成果四：把“合规”焊进门禁

### 5.1 单元测试（合成帧，不依赖引擎）

新增 `tests/test_tu93_frame.py`（**16 条**）：

- 直走廊解析（玩家/目标/首步方向/路径长度）；
- **晶格相位自恢复**：`offset ∈ {00,12,21,10,02}` 五组参数化；
- 连通格为虚空 ⇒ 边断开、BFS 返回无解；
- 需要绕路的图 ⇒ BFS 给出绕行路径；
- **危险物朝向**：面向侧的进入被拒，背面进入允许；
- **视线致死**：正前方 2 格不可入，除非先清掉危险物；
- **危险物封死唯一通路** ⇒ 无解；
- 巡逻物识别，且**危险物自身的朝向像素不被误判为巡逻物**；
- 形状不对 / 批维帧 / 观察对象解包 三种输入形态。

### 5.2 门禁扩展 `tests/test_closed_book_gate.py`

- 新增 **`test_no_baked_sequence_in_closed_book_modules`**：扫 `lingjing_solo/closed_book/*.py`，
  拒绝 ① 名字含 `LEVEL/PLAN/SOLUTION/SCRIPT/SEQUENCE/ANSWER` 的模块级常量，
  ② 任何 **≥15 个裸 int 的字面量**（那一定是动作序列）。→ 防止有人把 L3+ 用答案表“修好”。
- 新增 **`test_tu93_closed_book_branch_is_wired`**：AST 断言 agent 里**有且仅有一条**
  `_CLOSED_BOOK and gid == 'tu93'` 分支，且它引用了 `closed_book.tu93_solver`、`next_action`、并吞异常。
- 既有的 `test_closed_book_modules_never_read_the_engine` 用 glob 覆盖全部模块，
  **自动把 `tu93_frame.py` / `tu93_solver.py` 纳入**（禁 `environment_files` / `KNOWN_SOLUTIONS` /
  `arc_shadow` / `ar25.py` / `reference/` 与混淆 tag）。

### 5.3 变异自检（`state/_gate_mutation_check.py`，扩到 **7 个**变体）

```
[OK] inline_guard_removed        [OK] new_canned_table
[OK] engine_solver_unguarded     [OK] r3_unguarded
[OK] tu93_cb_ungated             [OK] closed_book_reads_engine
[OK] closed_book_baked_plan
[mutation] OK: all 7 mutants correctly rejected by the gate.
```

新增的两个变体正是本轮的两条新断言：**给 tu93 分支摘掉 `_CLOSED_BOOK` 门控**，
**往 `tu93_solver.py` 塞一张 20 个动作的答案表** —— 门禁都如期变红。

---

## 6. 口径已同步

`state/score_registry.json` 重算（`scripts/closed_book_score.py --write`），`--check` 通过：

```
canned  (主链)           52.5968   101 关  100.00%
tables  (CB=1)            2.0169    10 关    3.83%
strict  (CB=2) **         1.1756     9 关    2.24%   <- headline

分解: 硬编码动作表 50.5799 (96.17%) | R3 影子解法 0.8413 (1.60%) | 真·闭卷能力 1.1756 (2.24%)
官方 Kaggle 0.04 -> 罐头 1314.9x / 闭卷 29.4x（闭卷仍是远比罐头好的预测器）
```

历史档全部保留：`bench_closed_book_strict_20261013.json`（0.6200）、
`bench_closed_book_strict_ar25_20261013.json`（0.7312）。

---

## 7. 边界与遗留（诚实标注）

- **能力边界**：本轮求解器**精确覆盖 L0–L2**（迷宫 + 色 8 危险物视线）。
  **L3 起主动停滞（A5）**：L3 有**等速往返的巡逻物**，穿过去需要时空联合规划
  （预测相位、找交错窗口），而 A5 不推进时钟 ⇒ 原地等无效。**故意不做**，宁可不涨分也不试错送死。
- **L4+ 另有未建模机制**：宝石(11/15) → 开门(12/13)，本轮未解。
- 「停滞」是安全的：A5 是真 no-op，既不推进时钟也不改变 `levels_completed`，
  已到手的 L1–L3 不会被弄丢（本轮 400 拍零死亡、零 GAME_OVER）。
- `state/_probe_tu93_*.py` 里的**部分侦察探针**曾用罐头表快进（只为快速切到目标关 dump 帧），
  **绝不进入交付路径**；交付模块 `tu93_frame.py` / `tu93_solver.py` 由 §5.2 门禁静态保证不读引擎。
- 若日后要吃到 L3+，建议的下一步：把「巡逻物轨迹」建成**时间维展开的图**（node × tick mod period），
  在 BFS 里同时选“走/等”，而不是用 A5 盲等。

---

## 8. 复现

```bash
PY=integrations/ARC-AGI-3-Kaggle-Starter/.venv/Scripts/python.exe

# 帧语义观测（30 个探针）
$PY state/_probe_tu93_fullpix.py          # reset 帧全分辨率：节点格=色0、连通格=色2/色5
$PY state/_probe_tu93_c8shape.py          # 色8 是 3×3 缺一格（含色15 宝石）
$PY state/_probe_tu93_l2dir.py            # 色8 正前方 2 格节点致命
$PY state/_probe_tu93_l3move.py           # L3 色15 沿 col13 巡逻

# 离线证明（无 agent）
$PY state/_verify_tu93_solver.py 400      # 驱动发布版求解器类 -> levels=3
$PY state/_verify_tu93_cb.py 400         # 早期 harness，逐拍一致

# 单局 A/B（补丁前 vs 补丁后，全新 scorecard）
$PY state/_ab_one.py state/_pre_tu93_cb_my_agent.py tu93 strict   # 0 关 / 2.2222
$PY state/_ab_one.py state/_tu93_cb_my_agent.py     tu93 strict   # 3 关 / 13.3333

# 零回归
$PY state/_cb_run.py tu93-0768757b off     # 9/9 WIN 185 拍
$PY state/_cb_run.py ar25-0c556536 strict  # 1/8（未受影响）

# 整轮闭卷（25 局 x 400 拍）
OPERATION_MODE=offline LINGJING_CLOSED_BOOK=2 \
  $PY integrations/ARC-AGI-3-Kaggle-Starter/scripts/benchmark_all_games.py 400 \
  --out state/bench_closed_book_strict_tu93_20261013.json --label cb-strict-tu93

# 口径 / 门禁 / 变异
$PY scripts/closed_book_score.py --write state/score_registry.json
$PY scripts/closed_book_score.py --check
$PY -m pytest tests/test_closed_book_gate.py tests/test_tu93_frame.py tests/test_score_registry.py -q
$PY state/_gate_mutation_check.py
```
