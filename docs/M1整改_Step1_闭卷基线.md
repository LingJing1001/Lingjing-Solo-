# M1 整改 · Step 1：闭卷基线（已完成）

> 2026-10-13 ｜ 对应 `M1整改方案_20261013.md` 第 ① 步
> 目的：量出"去掉罐头后 agent 的真实能力地板"

---

## 0. 头条结论

| 口径 | 聚合分 | 总关卡 | WIN | 零关局 | 占罐头比 |
|---|---|---|---|---|---|
| **canned**（现有主链） | **52.5968** | 101 | 12 | 8 | 100% |
| **tables**（关 ②③ 硬编码表） | **2.0169** | 10 | 0 | 17 | 3.8% |
| **strict**（+关 ④ R3，M1 闭卷口径） | **0.6200** | 5 | 0 | 20 | **1.2%** |

> **现有 52.60 分里，97.8% 来自罐头**（96.2% 硬编码动作表 + 2.7% R3 影子解法），
> **真正的"闭卷推理能力"只占 1.2%（0.62 分）。**

---

## 1. 新增开关 `LINGJING_CLOSED_BOOK`

**默认不设 = 主链行为逐字节不变**（已验证，见 §3）。

| 取值 | 关闭 | 保留 |
|---|---|---|
| *(未设)* | —— | 全部（原样） |
| `1` / `tables` | ② 内联预烘计划层、③ EngineSolver 白名单 | ④ R3/Hybrid + ⑤ CEAX |
| `2` / `strict` | ② + ③ + ④ R3（shadow 预烘解法 + IDDFS 读引擎内部） | 仅 ⑤ CEAX + 画面热图点击 = **M1 闭卷** |

**为什么要两级**：`lingjing_solo/planning/arc_shadow_*.py`（26 个）**也是预烘答案**——`arc_shadow_ar25.py` 直接读 `arc_shadow.KNOWN_SOLUTIONS` 驱动引擎。只关 ②③，AR25 会从 R3 影子解法里**又拿到答案**，基线就废。

---

## 2. 改动点（5 处门控，补丁后行号）

| # | 位置 | 改动 |
|---|---|---|
| 1 | `my_agent.py:139–162` | 新增 `_CLOSED_BOOK` / `_CLOSED_BOOK_STRICT` + 启动打印 |
| 2 | `choose_action` 入口捷径 `:1320–1326` | 追加 `and not _CLOSED_BOOK` |
| 3 | `_choose_action_inner` ② 内联层（原 `1462–1699` → 现 `1483–1722`） | 整段包进 `if not _CLOSED_BOOK:`，缩进 +4 |
| 4 | ③ EngineSolver 分支 `:1726` | 追加 `and not _CLOSED_BOOK` |
| 5 | ④ R3 搜索条件 `:1844–1847` | 追加 `and not _CLOSED_BOOK_STRICT` |

**改动方式**：ASCII Python 脚本 + 显式 UTF-8 读写，换行 `chr(10)/chr(13)` 拼接，**未用编辑工具直写**（该文件曾被编辑器写坏成 UTF-16）。
补丁脚本 `state/_closed_book_patch.py`；**改前原件备份 `state/_pre_closed_book_my_agent.py`**（sha `1303b5f4…`，124153 B）。
验证：`py_compile` OK、`ast.parse` OK、锚点断言全过、CRLF 保持、`124153 → 126350 B`。

---

## 3. 零回归验证：AR25 单局三模式

| 模式 | 关卡 | 动作数 | 状态 | 耗时 |
|---|---|---|---|---|
| `off`（基线） | **8 / 8** | **276** | WIN | 3.9s |
| `tables` | 0 / 8 | 800（打满） | NOT_FINISHED | 19.4s |
| `strict` | 0 / 8 | 800（打满） | NOT_FINISHED | 17.7s |

`off` 的 **276 拍**与官方跑分 `bench_official_20261013.json` 的 ar25（276 拍 / 8 关 / 100 分）**逐拍一致** ⇒ 门控默认关闭、主链零影响。

---

## 4. 逐局三口径明细

| 局 | canned 关 | tables 关 | strict 关 | canned 分 | tables 分 | strict 分 |
|---|---|---|---|---|---|---|
| ar25 | 8 | 0 | 0 | 100.00 | 0.00 | 0.00 |
| bp35 | 2 | 0 | 0 | 6.67 | 0.00 | 0.00 |
| cd82 | 6 | 0 | 0 | 100.00 | 0.00 | 0.00 |
| cn04 | 2 | 3 | 0 | 8.25 | 28.57 | 0.00 |
| dc22 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 |
| ft09 | 6 | 0 | 0 | 100.00 | 0.00 | 0.00 |
| g50t | 1 | 1 | 0 | 3.57 | 3.57 | 0.00 |
| ka59 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 |
| lf52 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 |
| lp85 | 8 | 1 | 1 | 100.00 | 2.78 | 2.78 |
| ls20 | 7 | 0 | 0 | 100.00 | 0.00 | 0.00 |
| m0r0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 |
| r11l | 6 | 1 | 1 | 100.00 | 1.78 | 1.78 |
| re86 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 |
| s5i5 | 0 | 0 | 0 | 2.78 | 2.78 | 2.78 |
| sb26 | 8 | 1 | 1 | 100.00 | 2.78 | 2.78 |
| sc25 | 5 | 0 | 0 | 71.43 | 0.00 | 0.00 |
| sk48 | 0 | 1 | 0 | 0.00 | 2.78 | 0.00 |
| sp80 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 |
| su15 | 9 | 1 | 1 | 100.00 | 1.72 | 1.72 |
| tn36 | 7 | 0 | 0 | 100.00 | 0.00 | 0.00 |
| tr87 | 6 | 0 | 0 | 100.00 | 0.00 | 0.00 |
| tu93 | 9 | 0 | 0 | 100.00 | 2.22 | 2.22 |
| vc33 | 7 | 1 | 1 | 100.00 | 1.45 | 1.45 |
| wa30 | 4 | 0 | 0 | 22.22 | 0.00 | 0.00 |

**两个反直觉点**
1. `strict` 下 5 局仍各拿 1 关（lp85/r11l/sb26/su15/vc33）——来自 ⑤ CEAX + 画面热图，是**真·闭卷能力**（仅 0.62 分）。
2. `tables → strict` 只掉 1.40 分（cn04 −3 关、sk48 −1 关）⇒ **④ R3 那 26 个影子解法本身也基本是罐头**，贡献微乎其微。

---

## 5. 与官方 Kaggle 的关系

| 口径 | 分数 | 与官方 0.04 的比值 |
|---|---|---|
| canned 本地 | 52.5968 | **≈ 1315×** |
| strict 本地（闭卷） | 0.6200 | **≈ 15.5×** |
| 官方 Kaggle（隐藏集） | 0.04 | 1× |

⇒ **闭卷本地分是远比罐头分好的官方分预测器。** 这也反过来印证：本地 52.60 之所以不能预测 0.04，就是因为那 97.8% 是隐藏集上必然失效的罐头。

---

## 6. 结论与下一步

- **Step 1 完成**：闭卷地板已知 = **0.62**（M1 口径）/ **2.02**（关表口径）。
- **Step 2 的判断依据已经明确**：④ R3 不是救星（只值 1.40 分，且自带罐头）。要让闭卷分起来，必须动 ⑤ CEAX / 画面感知这条**真通用**的路，也就是公约里 R2/R3 该干而没干的活。
- 建议 Step 2 的验收线：**闭卷 25 局 ≥ 5 分**（当前 0.62），且 `pytest` 全绿。

---

## 7. 复现

```bash
cd integrations/ARC-AGI-3-Kaggle-Starter
LINGJING_CLOSED_BOOK=1 .venv/Scripts/python.exe scripts/benchmark_all_games.py 400 \
  --agent agent/my_agent.py --out ../../state/bench_closed_book_tables.json --label cb-tables
LINGJING_CLOSED_BOOK=2 .venv/Scripts/python.exe scripts/benchmark_all_games.py 400 \
  --agent agent/my_agent.py --out ../../state/bench_closed_book_strict.json --label cb-strict
```

单局调试：`state/_cb_run.py <game> <off|tables|strict>`（同进程；env 须在 import 前设）。

**产物**
- `state/bench_closed_book_strict_20261013.json`（cb-strict-400，250s）
- `state/bench_closed_book_tables_20261013.json`（cb-tables-400，488s）
- 日志 `state/_cb_strict_console.log` / `state/_cb_tables_console.log`
- 对照基线 `state/bench_official_20261013_tu93.json`（canned，52.5968）
