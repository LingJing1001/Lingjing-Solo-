# M1 整改 · Step 2C：闭卷 tu93 时空规划（L4–L6 打通）+ 三个真 bug

> 2026-10-13 ｜ 路线 A（窄而正）｜ 承接 Step2B 的闭卷 tu93 L0–L3
> 目的：把**真·闭卷能力**再抬一档，全程**零罐头、零引擎读、零 per-level 硬编码**

---

## 0. 头条

| 指标 | 改前（Step2B） | 改后 | Δ |
|---|---|---|---|
| **闭卷 headline（CB=2 聚合）** | **1.1756** | **3.1312** | **+1.9556** |
| tu93 单局分 | 13.3333（L0–L2） | **62.2222（L0–L6）** | **+48.8889** |
| tu93 通关关数 | 3 | **7** | +4 |
| 其余 24 局 | —— | **逐局分数完全不变** | 0 |
| 罐头分（canned） | 52.5968 | 52.5968 | 0（未触碰） |

实测产物：`state/bench_closed_book_strict_tu93_20261013.json`
（`aggregate_score = 3.1311547552484926`，`total_levels_completed = 13`，tu93 `level_actions = [18, 11, 26, 18, 30, 31, 17, 249, 0]`）。

> 注意 `level_actions[7] = 249`：这 249 拍**全是 A5 安全 hold**（`why = "tu93-cb hold L7"`），
> **零死亡**因此成立（`state=NOT_FINISHED`）。逐拍实测见 `state/_trace_tu93_verbose.log`
> 与 §5 的机制表：L7 走的是 **st-plan 分支**（巡逻模型建成、`broken=False`），
> 但 `plan_path` 返回 `None`（模型把巡逻块当成"在列 12 的四格走廊里永久往返"，
> 而通往目标的唯一路线恰好要穿过这条走廊）⇒ 求解器不猜、原地 hold。

> **为什么聚合只涨 1.9556**：聚合分 = 25 局单局分的平均。
> tu93 单局 +48.8889 ⇒ 聚合 +48.8889/25 = **+1.9556**。
>
> **口径**：单局分 = `min( Σ level_score[i]·(i+1) / Σ(i+1) , 已通关权重/Σ权重 ×100 )`（官方 `scorecard.py`）。
> 通了 L0–L6 ⇒ `max_score = (1+2+…+7)/45×100 = 62.2222`。45 = 1+…+9（tu93 共 9 关）。
> headline 的净增量按**聚合值**报：**+1.9556**。

**关键**：这 1.9556 是**真·闭卷能力**的净增量——不看引擎内部、不背答案、不含任何新硬编码表。

---

## 1. 起点：Step2B 停在哪

Step2B 交付了 L0–L3：

```
>> level 0->1@18, 1->2@29, 2->3@55, 3->4@73      （400 拍零死亡）
```

当时**明确标注的边界**是：

- **L4** 在求解器内部 `_broken=True` 永久 hold —— 但离线 `plan_path` 对 L4 明明有效
  （29 步计划盲执行逐字节通关）。⇒ **模型是对的，读数逻辑是错的。**
- **L5** 主动拒止 —— 计划踩 4 次危险物致死，当时归因为「色 8 朝向不可读」，
  于是加了一道 `MAX_HAZARD_STEPS=2` 保守闸门。
- **L6/L7/L8** 未触及。

本轮把这三点全部推翻/解决。

---

## 2. 三个真 bug（都定位到了具体行）

### 2.1 L4 卡死 = `_reconcile` 用**位置列表** zip 比巡逻物

**症状**（`state/_dbg_solver4.py`，逐字）：

```
step 76 lv=4 A3 k=3 broken=True plan=22 seen=[(10,5),(10,9),(12,7)] pred=[(12,9),(10,7),(10,7),(8,9)]
```

**根因**：L4 有 4 个巡逻块。当两块落到**同一节点**时，帧里只剩 **3 格**（4 块塌成 3），
而 `pred` 仍是 4 项（含一个重复）。旧实现逐元素 `zip`：

```python
cands = [j for j in range(P) if all(p.at(j) == n for p, n in zip(patrols, seen))]
```

元素级一比就永远找不到相位 ⇒ `cands == []` ⇒ `_broken = True` ⇒ 永久 hold。

**证据**（`state/_dbg_l4reconcile.py`）：从 step2 起 `zip-cands=[]`，
而**集合**比较 `set-cands` 恒含真相位：

```
step2 k=2 seen=[(10,5),(10,9),(12,7)]
      zip-cands=[]   set-cands=[1,41]   expected=1  ZIP-FAIL SET-OK
```

**修**：比较**节点集合**，而不是位置列表。`np.where` 的行主序会随方块滑动而重排，
重叠又会把 4 格塌成 3 格 —— 两件事对集合比较都是隐形的。

### 2.2 L5 送死 = `_hazard_owner` **单值覆盖**（本轮最硬的一个）

**症状**：L5 的色 8 危险物朝向全都「读对了」，但计划仍死；而罐头参考解（28 步、踩 7 次危险物）
**活得很好**。同一个 `A1`、同一个 `k=21`，参考解活、我们死（`state/_dbg_l5diff.py`）。

逐节点 diff 给出答案：

```
node (10,8):  ref_cell=0  my_cell=8      # 参考解已清，我们没清
node (14,10): ref_cell=0  my_cell=8      # 同上
node (16,8):  ref_cell=8  my_cell=0      # 我们已清，参考解没清
```

**根因**：危险物的**视线格**可以同时被**多个**危险物盯上：

- `(14,10)` 朝 **W** ⇒ 视线 = `(14,8)`
- `(16,8)` 朝 **N** ⇒ 视线 = `(14,8)`（同一个格！）

旧实现按「视线格 → 单个 owner」存：

```python
owner[(er + dr, ec + dc)] = (er, ec)      # 后写覆盖
```

`(16,8)` 在遍历里排在 `(14,10)` 后面，**把 `(14,10)` 冲掉了**。
于是当 `(16,8)` 已被我们清掉、而 `(14,10)` 还活着时，模型认为 `(14,8)` 安全 ⇒ 走进视线 ⇒ `GAME_OVER`。
参考解在 move 19 先清了 `(14,10)`，所以它的 `(14,8)` 是真安全。

**修**：`owner[cell]` = **所有**盯它的危险物的集合；`_steps` 要求 **全部清完**才放行：

```python
watchers = owner.get((nr, nc))
if watchers and not watchers <= cleared:
    continue                       # 仍有活着的危险物盯着这一格
```

**同时删掉** Step2B 为绕这个 bug 加的 `MAX_HAZARD_STEPS=2` 保守闸门 ——
它把 L5 的**合法** 30 步解误判为「不可靠」而拒止。闸门是在给 bug 打补丁，bug 修好后它就是纯损失。

### 2.3 L6 = 巡逻物**会拐弯**，不存在 `k -> node` 函数

L3–L5 的巡逻块在**直走廊**上往返（bounce），位置是有效移动数 `k` 的确定函数 ⇒ 可以时空规划。
L6 的巡逻块沿**整个迷宫走廊绕圈**（`state/_dbg_l6path.py`）：

```
(6,8) → (8,8) → (10,8) → (10,10) → (12,10) → (14,10) → (14,12)
      → (12,12) → (12,14) → (10,14) → (8,14) → (6,14) → …
```

试过用「prefer straight，否则取唯一出口」重建，**在 `(10,8)` 就分叉**（非回退出口有 W 和 E 两个，
它选了 E）；换成「prefer straight，否则取离目标最近的出口」，**在 `(12,12)` 被反例推翻**
（直线 N 明明通着，它却拐 E）。⇒ **拐弯规则不自洽**，是关卡作者写死的固定环路。

**所以 L6 不能用时空规划**。见 §3。

---

## 3. L6 的解法：一条**可证的**反应式生存律

不去预测环路，而是利用一条帧里**永远可读**的事实：

> **巡逻块每「有效移动」推进 1 个节点，并把「下一步朝向」直接画在帧里**
> —— 3×3 方块里那枚异色像素的偏移即位方向。

于是得到一条**构造性安全**的规则：

> **玩家只要永不落在巡逻块「即将进入」的那一格，就永远不会死。**
> 落在它**当前**那一格是**交换**（它已经走了），不致死 —— 这条在 Step2B 就已实测确认。

实现：

- `tu93_frame.patrol_next_cells(frame, cells, offset)` —— 读每个巡逻块的当前格 + 朝向，给出下一格。
- `tu93_solver` 的 **reactive 分支**：巡逻模型失配（`_broken`）时不再永久 hold，而是走
  「避开 next cell 的 BFS」；直路被堵就侧移、优先回退，全堵才 A5。

这条路**不需要知道环路**，所以对拐弯巡逻物一样有效。

---

## 4. 验收（全部可复现）

### 4.1 闭环验证（驱动**真类**，帧进动作出，等同 agent 路径）

```
$ python state/_verify_tu93_solver.py 3000
  >> level 0 -> 1 at step 18
  >> level 1 -> 2 at step 29
  >> level 2 -> 3 at step 55
  >> level 3 -> 4 at step 73
  >> level 4 -> 5 at step 103
  >> level 5 -> 6 at step 134
  >> level 6 -> 7 at step 151
RESULT levels=7 state=NOT_FINISHED steps=3000 resyncs=0
```

- **L0–L3 步数零回退**（18/29/55/73 与 Step2B 完全一致）。
- **3000 拍零死亡**（`state=NOT_FINISHED`）。
- `resyncs=0`：全程没有一次位置对不上预测 —— 说明读数/模拟是一致的。
- **与 agent 主链逐关对齐**：25 局基准里 tu93 的 `level_actions = [18, 11, 26, 18, 30, 31, 17, 249, 0]`，
  前 7 项与上面的闭环跃迁**逐数吻合**（18 / 11 / 26 / 18 / 30 / 31 / 17）⇒
  离线 harness 与 shipped agent 走的是同一条路径，没有「harness 绿、主链红」的缝隙。

### 4.2 单测

`tests/test_tu93_frame.py` 从 15 个扩到 **26 个**（新增 11 个，全部合成 64×64 帧，不需要引擎）：

| 新增用例 | 钉住什么 |
|---|---|
| `test_hazard_owner_lists_every_watcher` | 一个视线格可被**多个**危险物盯（§2.2 的回归锁） |
| `test_line_of_sight_needs_every_watcher_cleared` | 只清一个盯梢者**不放行**，全清才放行 |
| `test_patrol_bounces_along_its_segment` | 巡逻块的 bounce 相位公式 |
| `test_build_patrols_reads_the_corridor_and_heading` | 走廊提取 + 朝向 + `patrol_next_cells` |
| `test_plan_path_never_lands_on_the_patroller` | 时空规划**永不**与巡逻块同格 |
| `test_plan_path_gives_up_when_the_patroller_seals_the_route` | 被封锁时返回 `None`（宁可不走） |
| `test_reconcile_tolerates_overlapping_patrols` | 重叠塌格不算模型坏（§2.1 的回归锁） |
| `test_reconcile_tolerates_an_occluded_patrol` | 巡逻块**钻到危险物下面**被遮住不算模型坏 |
| `test_reconcile_stands_down_when_a_patrol_is_somewhere_impossible` | 真失配仍要 stand down |
| `test_reactive_solver_never_enters_the_patrols_next_cell` | 反应式：唯一前进一步是死路时**选择不动** |

### 4.3 门禁 / 口径

- `tests/test_closed_book_gate.py` 9 个用例全绿（未新增任何 `_*_PLANS` / `_*_LEVEL_ACTIONS` 表）。
- **门禁仍在咬人**：`python state/_gate_mutation_check.py` → `[mutation] OK: all 7 mutants correctly rejected`。
- `scripts/closed_book_score.py --write state/score_registry.json` + `--check` 收口。
- **agent 侧零改动**：`my_agent.py` 是按模块路径 `from lingjing_solo.closed_book.tu93_solver import ...`
  取求解器的，所以 `state/_tu93_cb_my_agent.py` 快照与 shipped agent **逐字节相同**（无需重新冻结）。

> **口径收口（已解决）**：`closed_book_score.py` 的分解把 `R3 影子 = tables − strict`。
> 中途 strict 一度**超过**旧的 tables 文件（那个文件是 13:24 的，其 tu93 还是 0 关），
> 使差值显示为负。**已用正确的 runner 重跑 CB=1（tables）**，两侧现为同期：
>
> | 口径 | 分数 | 关卡 | WIN | 零关 |
> |---|---|---|---|---|
> | canned（主链） | 52.5968 | 101 | 12 | 8 |
> | tables（CB=1） | 4.5280 | 18 | 0 | 15 |
> | **strict（CB=2）** | **3.1312** | 13 | 0 | 18 |
>
> 分解：硬编码表 48.0688（91.39%）／R3 影子 **+1.3968**（2.66%）／**真·闭卷 3.1312（5.95%）**。
> 影子项的三个来源逐局可验：`cn04`、`g50t`、`sk48`（R3 影子解），其余 22 局两侧同分。
>
> ⚠️ **注意**：tables 侧的 tu93 同样是 **7 关 / 62.2222**——因为 agent 走的是模块路径
> `from lingjing_solo.closed_book.tu93_solver import ...`，闭卷求解器在任何 CB 模式下都被装进主链。
> 也就是说 tu93 这 62.2222 在两侧**不是**"R3 影子 vs 闭卷"的对照，而是同一份能力被计了两次；
> 真正的口径差只剩 cn04/g50t/sk48。
>
> `scripts/closed_book_score.py --write state/score_registry.json` + `--check`
> → `[closed-book-score] OK: registry matches bench artifacts.`

---

## 5. 诚实的边界（未解决，且**不试错送死**）

| 关 | 状态 | 已知事实 |
|---|---|---|
| **L7** | **st-plan 分支 → `plan_path=None` → hold**（安全） | 巡逻块建出的模型是「列 12 四格走廊往返」（`seg=[(4,12)..(10,12)], P=6`），但真实行为是在 `(6,12)` **连续 7 个有效移动不动**、之后才绕全图。用「有效移动 / A5 交替」实测：它**不是**有效移动数的函数。⇒ 拒绝猜测 |
| **L8** | 未触达 | L7 卡住，无观测 |

**处置**：L7 的**唯一路线必须穿过列 12 走廊**（`bfs` 忽略巡逻块给出 11 箭头解
`[4,4,1,1,4,4,1,1,1,3,3]`），而错误的 bounce 模型让时空搜索认为这条走廊**永久被封死**，
于是 `plan_path` 返回 `None` ⇒ 求解器 hold（A5 是免费的 no-op，不吃计时条）⇒ **零死亡风险**。
**这是一次「模型错、不是关卡无解」的失败**（`state/_dbg_l7why.py`），刻意不赌：宁可停在 L6，
也不在没搞懂的机制上试错。

另需说明：本轮**没有**引入宝石/门的消费逻辑。L5 的 12/13 格是**巡逻块**而非门
（它们有朝向像素、会移动，`state/_dbg_l5tiles.py`）；L7 的 `(6,12)` 同时像门又像巡逻块，
正是它行为异常的原因之一，尚未拆开。

---

## 6. 这一轮证明了什么 / 没证明什么

**证明**：

1. tu93 的**帧感知世界模型**足以覆盖 L0–L6：几何、地形、危险物（含**多重视线**）、
   巡逻物（直走廊 bounce + 拐弯），全部从帧里读出来，没有任何 per-level 常量。
2. 时空搜索（`(node, k mod P, cleared)` 上的 Dijkstra）配合**提交计划 + 逐拍重放**是对的；
   之前 L4 的失败纯粹是**对拍逻辑**（位置 vs 集合）写错。
3. 「永不落在巡逻块 next cell」是一条**可证**的安全律，不需要预测环路，因此对未知巡逻模式也成立。

**没证明**：

- **L7 的巡逻机制**（触发条件 / 是否与玩家位置耦合）—— 这是下一个真正的未知。
- **L8** 及门的消费机制。
- 计时条预算 L6/L7 的精确值仍靠 `round(64(T−k)/T)` 拟合，未逐关实测。

---

## 7. 复现清单

```bash
# 闭环（驱动真求解器）
python state/_verify_tu93_solver.py 3000

# 三个 bug 的证据脚本
python state/_dbg_l4reconcile.py     # §2.1  zip vs 集合
python state/_dbg_l5diff.py          # §2.2  同 k 同动作，一死一活 + 节点级 diff
python state/_dbg_l6path.py          # §2.3  拐弯环路
python state/_dbg_l7deep.py          # §5    L7 异常（7 拍不动）

# 单测 / 门禁
pytest tests/test_tu93_frame.py tests/test_closed_book_gate.py -q

# 25 局 strict 闭卷基准 + 口径收口
LINGJING_CLOSED_BOOK=2 python scripts/benchmark_all_games.py 400 \
    --out state/bench_closed_book_strict_tu93_20261013.json --label cb-strict-tu93
python scripts/closed_book_score.py --write state/score_registry.json
python scripts/closed_book_score.py --check
```
