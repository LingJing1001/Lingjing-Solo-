# M1 整改 · Step 2A：闭卷 AR25 帧感知求解（已完成）

> 2026-10-13 ｜ 路线 A（窄而正）｜ 承接 Step③ 的口径门禁
> 目的：把"真·闭卷能力"从 0.6200 抬上去，且**不引入任何新罐头**

---

## 0. 头条

| 指标 | 改前 | 改后 | Δ |
|---|---|---|---|
| **闭卷 headline（CB=2 聚合）** | **0.6200** | **0.7312** | **+0.1111（+17.9%）** |
| AR25 单局分 | 0.00（0 关） | **2.7778（1 关）** | **+2.7778** |
| 其余 24 局 | —— | **逐分完全不变** | 0 |
| 罐头分（canned） | 52.5968 | 52.5968 | 0（未触碰） |

> **为什么聚合只涨 0.1111**：聚合分 = **25 局单局分的平均值**，不是求和。
> 所以 AR25 单局 +2.7778 ⇒ 聚合 +2.7778/25 = **+0.1111**。
> （Step① 文档里的 52.60 也是均值口径，同一把尺。）

**关键**：这 0.1111 是**真·闭卷能力**的净增量——不看引擎、不背答案、不含任何新硬编码表。

---

## 1. 问题：为什么之前是 0

Step②/③ 的实验已把根因钉死：CEAX 是手写启发式、**没有世界模型**，不能从像素反推机制。
`strict` 下 5 局各拿 1 关全靠 CEAX + 画面热图，AR25 是 0 关（`keyboard_click` 局被锁进点击路径）。

于是路线 A 的目标：**给 AR25 造一条纯帧感知的真通用链路**，而不是再塞一张答案表。

---

## 2. 成果一：帧感知 AR25 编码器（合规）

`lingjing_solo/closed_book/ar25_frame.py` —— 只吃 `FrameData`，**不读 `environment_files/`、不读混淆 tag、不读引擎源码**。

### 渲染约定（全部由**观测**建立，见 `state/_probe_ar25_*.py`）

| 项 | 值 | 怎么得到的 |
|---|---|---|
| 棋盘 | **21×21 格，3 px/格**，原点 (0,0) | 对帧做连通域分析：轴是 3px 宽的整列、形状包围盒按 3px 对齐 |
| 右/下 1px 边框 | 步数环，须排除 | 颜色随动作步进变化，非棋面内容 |
| 色 0 | 选择光标（选中对象包围盒上的 1px L 形） | 该色只在被选形状的包围盒棱上出现 |
| 色 4 | **反射像** | 形状平移时同步镜像出现 |
| 色 5 | 可移动形状 | 跟随方向键整体平移 |
| 色 9 | 背景 | 占比 >89% |
| 色 10 | 镜像轴（整列 = 竖轴 / 整行 = 横轴） | 平移形状时反射面随之移动 |
| 色 11 | 目标格 | 胜利条件 = 被反射/直接覆盖 |

**镜像几何**：轴在格坐标 `a` 时，`c → 2a − c`（竖轴）/ `r → 2a − r`（横轴）。
（实测：轴在 col 10，形状 col 6 的镜像落在 col 14 ✓）

全部 9 关的调色板**恒为 {0,4,5,9,10,11}**（`state/_ar25_levels.txt`），所以这套词汇是稳定的。

`parse_frame()` 输出：轴（kind+coord）、形状连通域、目标格集合、光标格、**被选中的形状**（包围盒含光标的那个组件）。解析不明确时 `ok=False` → 上层放弃、交回 CEAX。

---

## 3. 成果二：闭卷平移求解器

`lingjing_solo/closed_book/ar25_solver.py`。

**核心推导**：找最小代价 `(dr, dc)`，使 `mirror(shape + (dr,dc)) ⊇ target`（代价 = 曼哈顿距离，与关卡玩法一致）。

**闭环执行**：每 tick 从**当前帧**重算一次，发一个动作；先横后纵；若上一动作帧没变化（撞墙/卡住）则换成先纵后横。

### 离线证明（完全不用 agent）

```
$ python state/_verify_ar25_cb.py ar25 400 9
RESET parse: ok=True axis=V10 shapes=[5] targets=5 cursor=5 selected=yes
  needed shift = (10, -5)
  step 15: ACTION2 -> levels=1
RESULT levels=1 actions=15
```

**15 步 = 恰好复现罐头串 `[3]*5 + [2]*10`**——但这次是**从像素推出来的**：
解析出"轴在 col 10、灰 L 在 (5,6..8)+(6..7,8)、目标在 (15..17,17..19)"，解出需要平移 `(dr=+10, dc=−5)`，于是 LEFT×5 再 DOWN×10。

---

## 4. 成果三：接入 agent（默认主链零回归）

**改动**：`agent/my_agent.py` 在 `_choose_action_inner` 的闭卷路径插入一段分支（补丁脚本 `state/_ar25_cb_patch.py`，ASCII + 显式 UTF-8，CRLF 保持；改前备份 `state/_pre_ar25_cb_my_agent.py`）：

```python
# --- CLOSED-BOOK: frame-only AR25 solver (no baked answers) ---
if _CLOSED_BOOK and gid == "ar25":
    ...   # 懒加载 Ar25ClosedBookSolver，失败即停用；异常一律吞掉
```

- 只在 `_CLOSED_BOOK` 打开时生效 ⇒ **off 模式逐字节无影响**。
- 失败/解析不出 → 返回 None，正常落回 CEAX，不改变现有行为。
- 文件 126350 → 127557 B，`ast.parse` OK，CRLF 计数校验通过。

### 零回归实测

| 模式 | 关卡 | 动作 | 状态 |
|---|---|---|---|
| `ar25 off`（主链） | **8/8** | **276** | WIN |
| `ar25 strict`（闭卷） | **1/8** | —— | NOT_FINISHED |

`off` 的 **276 拍与官方基线逐拍一致**。闭卷 1 关 = 新增能力。

---

## 5. 成果四：把"合规"焊进门禁

新增 `tests/test_ar25_closed_book.py`（5 条，合成帧、不依赖引擎）：

- 感知还原 L1 全要素（轴/形状/目标/光标/选中）
- `solve_shift == (10, -5)`，且该平移的镜像确实覆盖全部目标
- **闭环推演复现 `[3]*5 + [2]*10`**
- 无轴帧 → `ok=False` 且不动手；空帧安全

新增门禁 `test_closed_book_modules_never_read_the_engine`（在 `tests/test_closed_book_gate.py`）：
扫描 `lingjing_solo/closed_book/*.py`（**剥掉 docstring 后**），禁止出现
`environment_files`、`KNOWN_SOLUTIONS`、`arc_shadow`、`ar25.py`、`reference/`，以及形如 `"0001sruqbuvukh"` 的引擎混淆 tag。

**变异自检**（`state/_gate_mutation_check.py`）扩到 5 个变异体，全部被正确拒绝：

```
[OK] inline_guard_removed      [OK] new_canned_table
[OK] engine_solver_unguarded   [OK] r3_unguarded
[OK] closed_book_reads_engine
[mutation] OK: all 5 mutants correctly rejected by the gate.
```

---

## 6. 口径已同步

`state/score_registry.json` 重算（`scripts/closed_book_score.py --write`），`--check` 通过：

```
canned  (主链)    52.5968   101 关  100.00%
tables  (CB=1)     2.0169    10 关    3.83%
strict  (CB=2) **  0.7312     6 关    1.39%   <- headline

分解: 硬编码表 50.5799 (96.17%) | R3 影子 1.2857 (2.44%) | 真·闭卷 0.7312 (1.39%)
官方 Kaggle 0.04 -> 罐头 1314.9x / 闭卷 18.3x（闭卷仍是远比罐头好的预测器）
```

改前基线 `state/bench_closed_book_strict_20261013.json`（0.6200）保留存档。

---

## 7. 边界与遗留

- **能力边界**：本求解器覆盖「目标 = 单个可移动形状的镜像平移」这一类局。**L2 起交回 CEAX**
  （L2 需要 ACTION5 切换选择 / 多物体协同），故本轮只吃到 L1 的 1 关。
- 再往上要扩展：多形状择一 + CYCLE 选择 + 多轴联动。`Ar25Field.enumerate_covering_configs` 已有配置空间搜索，
  但**输入仍是违规的 `ar25_encoder.py`**；下一步应把它接到新的帧感知 `Ar25FrameObs` 上。
- `s5i5/tu93/lp85/r11l/sb26/su15/vc33` 在 strict 下各有 1 关级别的非零分（`levels_completed` 仍记 0，属计分口径内的部分分），
  它们来自 CEAX，本轮未动。

## 8. 复现

```bash
# 帧语义观测
python integrations/ARC-AGI-3-Kaggle-Starter/.venv/Scripts/python.exe state/_probe_ar25_cells.py ar25
# 离线证明（无 agent）
python integrations/ARC-AGI-3-Kaggle-Starter/.venv/Scripts/python.exe state/_verify_ar25_cb.py ar25 400 9
# 单局对比
python integrations/ARC-AGI-3-Kaggle-Starter/.venv/Scripts/python.exe state/_cb_run.py ar25 strict
python integrations/ARC-AGI-3-Kaggle-Starter/.venv/Scripts/python.exe state/_cb_run.py ar25 off
# 整轮闭卷
cd integrations/ARC-AGI-3-Kaggle-Starter
LINGJING_CLOSED_BOOK=2 .venv/Scripts/python.exe scripts/benchmark_all_games.py 400 \
  --agent agent/my_agent.py --out ../../state/bench_closed_book_strict_ar25_20261013.json --label cb-strict-ar25
# 门禁
python -m pytest tests/test_closed_book_gate.py tests/test_score_registry.py tests/test_ar25_closed_book.py -v
python state/_gate_mutation_check.py
```
