# M1 整改 · Step 3：门禁反向 + 口径对齐（已完成）

> 2026-10-13 ｜ 路线 **B**（治标）｜ 对应 `M1整改方案_20261013.md` 第 ③ 步
> 目的：把 0.62 钉成诚实基线，让"罐头分"再也不能冒充成绩

---

## 0. 一句话

**从此 CI 里有两道新闸门：一道保证 headline 分数必须是闭卷分，一道保证任何硬编码答案表都被 `_CLOSED_BOOK` 关住。**
原来的 CI（pytest + 路线回放）只会奖励"又过了几关"——**它天然鼓励把答案表做大**。这两道闸门正是把这个激励方向反过来。

| 闸门 | 文件 | 作用 |
|---|---|---|
| 口径对齐 | `scripts/closed_book_score.py --check` + `tests/test_score_registry.py` | `state/score_registry.json` 是唯一真相源；分数漂移而没同步 → 红 |
| 门禁反向 | `tests/test_closed_book_gate.py` | 任何未被 `not _CLOSED_BOOK` 包住的 baked 答案表 → 红 |

已接入 `.github/workflows/ci.yml` 的 `pytest` job（两步显式命名，排在原 `pytest tests/` 之前）。

---

## 1. 口径对齐：单一真相源

新增 `state/score_registry.json`（`schema: lingjing.score_registry/v1`），由脚本从三个 bench 产物重算生成：

```bash
python scripts/closed_book_score.py            # 打印分解表
python scripts/closed_book_score.py --write state/score_registry.json
python scripts/closed_book_score.py --check    # CI 用；漂移则 exit 1
```

**唯一 headline 指标 = `closed_book_strict`（CB=2）。** 其余口径只作为分解项存在。

```
口径                    分数      关卡  WIN  零关  占罐头比
----------------------------------------------------------------
canned  (主链)           52.5968   101   12     8   100.00%
tables  (CB=1)            2.0169    10    0    17    3.83%
strict  (CB=2) **         0.6200     5    0    20    1.18%

  ** = headline metric (closed_book_strict)

分解:
  硬编码动作表  50.5799  (96.17%)   = canned - tables
  R3 影子解法    1.3968  ( 2.66%)   = tables - strict
  真·闭卷能力    0.6200  ( 1.18%)   = strict
```

**为什么用差值就能分解**：三个口径是一条链，每级只关掉一层：

| 口径 | 保留的决策层 | 被关掉 |
|---|---|---|
| canned | ①②③④⑤ 全部 | —— |
| tables (CB=1) | ④ R3 + ⑤ CEAX | ② 内联预烘计划、③ EngineSolver |
| strict (CB=2) | ⑤ CEAX + 画面热图 | ②③④ |

所以 `canned − tables` 就是②③的净贡献（= 硬编码表和白名单），`tables − strict` 就是④的净贡献（= R3 影子解法/IDDFS），剩下的 `strict` 才是⑤这条**帧感知真通用链**的能力。

`tests/test_score_registry.py` 断言这些恒等式成立，并**重读 bench JSON 复核分数**——跑分一变而 registry 没更新，CI 立刻红。这堵死了"悄悄换个更大的数当成绩"。

---

## 2. 门禁反向：反罐头静态闸门

`tests/test_closed_book_gate.py`（**纯静态**：只读 `my_agent.py` 的文本/AST，不 import、不跑引擎，所以在无 `arcengine` 的 `pytest` job 里也能跑）。

六条断言：

| 测试 | 保证 |
|---|---|
| `test_closed_book_switch_exists` | `LINGJING_CLOSED_BOOK` 开关与 `_CLOSED_BOOK` / `_CLOSED_BOOK_STRICT` 两个模块级 flag 在位，且 strict ⇒ closed-book |
| `test_no_unregistered_canned_tables` | 模块级 `_*_PLANS` / `_*_LEVEL_ACTIONS` **必须**与冻结白名单**完全一致** |
| `test_baked_lookups_are_gated_in_decision_path` | `_choose_action_inner` 里每一处 baked 查表（含 `self.ls20` / `self.ar25`）都必须被 `if not _CLOSED_BOOK:` 包住 |
| `test_engine_solver_entry_is_gated` | ③ EngineSolver 分支必须带 `not _CLOSED_BOOK` |
| `test_r3_search_is_gated_under_strict` | ④ R3 入口必须带 `not _CLOSED_BOOK_STRICT` |
| `test_entry_shortcut_is_gated` | `choose_action` 入口捷径同样受闭卷开关约束 |

### 第二条才是"反向"的关键

它不是"检查分数高不高"，而是**检查有没有藏着新答案**：

- 任何人往 `my_agent.py` 里加一张新的 baked 表（`_XX_PLANS` / `_XX_LEVEL_ACTIONS`），**不改白名单就直接红**；
- 白名单里列了但表被删了，也红（防止白名单变成撒谎的记录）。

也就是说：**"加答案"从零成本变成必须显式登记的动作**。这就是把激励反过来。

冻结白名单（14 张，见 `EXPECTED_CANNED_TABLES`）：
`_VC33_PLANS` `_WA30_PLANS` `_SB26_PLANS` `_SU15_PLANS` `_R11L_PLANS` `_CD82_PLANS` `_TN36_PLANS` `_TR87_PLANS` `_SC25_PLANS` `_BP35_PLANS` `_TR87_LEVEL_ACTIONS` `_LS20_LEVEL_ACTIONS` `_AR25_LEVEL_ACTIONS` `_TU93_LEVEL_ACTIONS`

---

## 3. 门禁能真红吗：变异测试

> 一个永远不会红的门禁等于没有。用 `state/_gate_mutation_check.py` 造 4 个变异体反证。

```bash
python state/_gate_mutation_check.py
```

| 变异体 | 手法 | 结果 |
|---|---|---|
| `inline_guard_removed` | 把 `if not _CLOSED_BOOK:` 改成 `if True:` | ✅ 被 `test_baked_lookups_are_gated_in_decision_path` 抓住 |
| `new_canned_table` | 加一张未登记的 `_ZZ99_PLANS` | ✅ 被 `test_no_unregistered_canned_tables` 抓住 |
| `engine_solver_unguarded` | 去掉 EngineSolver 的 `and not _CLOSED_BOOK` | ✅ 被 `test_engine_solver_entry_is_gated` 抓住 |
| `r3_unguarded` | 去掉 R3 的 `and not _CLOSED_BOOK_STRICT` | ✅ 被 `test_r3_search_is_gated_under_strict` 抓住 |

```
[mutation] OK: all 4 mutants correctly rejected by the gate.
```

---

## 4. 接入 CI

`.github/workflows/ci.yml` 的 `pytest` job 新增两步（在原 `pytest tests/` 之前）：

```yaml
      - name: 口径对齐闸门（headline = 闭卷分）
        run: python scripts/closed_book_score.py --check

      - name: 门禁反向闸门（反罐头）
        run: python -m pytest tests/test_closed_book_gate.py tests/test_score_registry.py -v
```

新测试同时也在 `pytest tests/` 的全量收集里（472 → 482 用例），双保险。

---

## 5. 这次改了什么（零回归清单）

**新增（全部为测试/工具，不碰主链）**
- `scripts/closed_book_score.py` —— 口径分解 CLI
- `state/score_registry.json` —— 唯一真相源
- `tests/test_closed_book_gate.py` —— 反罐头静态门禁（6 断言）
- `tests/test_score_registry.py` —— 口径一致性门禁（4 断言）
- `state/_gate_mutation_check.py` —— 门禁变异自检
- `.github/workflows/ci.yml` —— 加两步

**未改动**：`agent/my_agent.py`、`lingjing_solo/**`、任何 bench 产物。闭卷/罐头两个口径的分数都逐字节不变。

---

## 6. 现状与遗留

- **headline = 0.6200（闭卷）**，且已被 CI 焊死为唯一可上报分数。
- 罐头分 52.5968 仍然存在，但它现在**只是分解项**，不再可能冒充成绩。
- **Step②（提闭卷分）仍未完成**——按路线 B 的定义，它被明确挂账：要动的是 ⑤ CEAX 帧感知链。
  原"AR25 L1 闭卷通关"的可选路线（旧方案 A）仍待排期，且已确认仓内 `ar25_encoder.py` 违反公约（直接读引擎源码），需重写为帧感知版。

## 7. 复现

```bash
# 口径
python scripts/closed_book_score.py
python scripts/closed_book_score.py --check          # exit 0
# 门禁
python -m pytest tests/test_closed_book_gate.py tests/test_score_registry.py -v
# 门禁自检（应全 OK）
python state/_gate_mutation_check.py
```
