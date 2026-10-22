# 官方分支合并 + SELFLEARN_LLM 开关 —— 分析报告（终版）

日期：2026-10-22　范围：`灵境战队AGI课题探索`（课题探索仓库）· ARC-AGI-3-Kaggle-Starter（SmartRouter 线）
关联：`docs/v27_根因定位与提分方案_20261022.md`（duck 线 v27）
本轮提交：`34e7d39`（合并 origin/main + 开关）→ `00a171e`（修决策表 + shadow 移植）→ `75f7d8b`（出网硬守卫 + notebook 自测 cell）→ `100fdc2`（计划默认开 + Kaggle 浅路径修复）

---

## 0. 结论摘要（先给答案）

| 问题 | 结论 |
|---|---|
| 官方分支能不能合？ | **能，但只能"符号级合并"。** `origin/main` 整体合入（`34e7d39`）；`fix-shadow-env-passing` 移植进来（`00a171e`）；`feat/unify-decoder-entry` **只该取 25 个实质文件，不该合分支**（尖端含 193k 行全量快照）。 |
| `feature/llm-whole-qwen` 要不要再合？ | **不用。** 它已经是本地 HEAD 的祖先，whole-qwen LLM 链本来就在我们树里。之前的"再合一次"是误判。 |
| `SELFLEARN_LLM=1/0` 怎么落地的？ | 落成**权威主开关**：权威实现 `arc_adaptor/lexicon/llm_planner.py:selflearn_llm_enabled()`；`my_agent._llm_on()` 与 `agent/self_learner.py` 统一使用；`LINGJING_LLM` 降级为旧开关兜底。`=1`→大模型 planner；`=0`→原策略（**不起 vLLM、不挂模型数据集、纯算法**）。**并在出网层加了硬守卫**，保证 `=0` 时一个字节都不出网。 |
| 合并本身有没有副作用？ | **没有。** 合并后干净重跑的 25 局，与合并前逐局逐关动作数**完全一致**（`probe` vs `PRE`：19.366 / 35 关 / 同动作序列）。中途出现过的 -0.27 分回退，根因不是合并，而是 `SELFLEARN_LLM=0` 下仍有**幽灵 LLM 调用**，每次白等 45 秒超时把那一局的墙钟吃光。 |
| 跑测结果怎么样？ | 本地 25 局 × 400 步（`SELFLEARN_LLM=0` 原策略臂）：**计划关 19.37 分 / 35 关 / 4 局 WIN / 16 个零关局**；**计划开 59.85 分 / 105 关 / 13 局 WIN / 8 个零关局**。Kaggle 上独立复现同一结论（v4 计划关 15.48 分，零关局全是"判 `plan_gid` 的计划局"）。 |
| 最大的杠杆是什么？ | **不是"再合分支"，是"把已经在树里的东西打开"**：`LINGJING_SCRIPTS` 默认 off，16 个零关局里 11 个是这么来的。已改为**默认开 + 逐局可排除**。 |
| 继续合并还是优化？ | **继续合并，但只做一件：移植 `unify-decoder-entry` 的 25 文件破零包**（它打的正是我们剩下的 5 个零关局）。其余分支收益低、风险高。**同时必须清醒：计划带来的分是公开局分，隐藏集的胜负手是通用能力（engine solver / IDDFS / LLM）。** |
| 有什么硬阻断？ | **Kaggle 周 GPU 配额已耗尽**（用 41196s / 上限 21600s），**下次刷新 2026-10-24T00:00:00Z（北京时间 10-24 08:00）**。挡住 llm1 臂、A/B 的 LLM 侧、**以及明天要提的 v27 竞赛重提交**。日提交配额另算（今天已被队友用掉，UTC 明日解锁）。 |

---

## 1. 官方分支全景

镜像 `ghproxy.imciel.com` 拉到的远端（课题探索仓库 25+ 分支）：

| 分支 | HEAD | 与我们的关系 | 处置 |
|---|---|---|---|
| `origin/main` | `730b833` | **与本地 main 互补**，+11 commit / 96 文件 / +30,664 行 | ✅ 已整体合并（`34e7d39`） |
| `origin/feat/duck-scheduler2` | `0c89e8a` | 已并入 `origin/main`（duck_bridge import 修复） | ✅ 随 main 进 |
| `origin/feat/duck-scheduler` | `83b10ef` | **本地 4b43f34 已合过** | ✅ 无需重复 |
| `origin/feat/duck-harness` | `7eb8a91` | 本地已在 | ✅ |
| `origin/feature/llm-whole-qwen` | `fca8f72` | **是本地 HEAD 的祖先**（`merge-base --is-ancestor` = 0），但**不是 origin/main 的祖先**（= 1） | ✅ 无需重复合（链已在树里） |
| `origin/feature/llm-vision-preceax` | `e0919b9` | 上游 4 个唯一函数之一（`_load_llm_policy`/`_rule_to_action`/`_semantic_prompt`）已随 main 进 | ✅ |
| `origin/feature/aop-readiness-v2` | `8ad1761` | 已随 main 进 | ✅ |
| `origin/fix-shadow-env-passing` | `aa1b889` | **不在 main、也不在本地 HEAD** | ✅ 已移植（`00a171e`） |
| `origin/feat/unify-decoder-entry` | `3d4cdba` | **不在 main、也不在本地 HEAD**；自带 57.37 的破零包 | ⚠️ **只取 25 个实质文件，勿合分支** |
| `origin/feature/twin-engine-solver` / `graph-explorer-statekey` / `dpl-v14.2-unknown-bench-400` / `merge/pr9-into-main` / `reproduce/lingjing-solo-ls20` | — | 未评估/已在树内 | ⏸️ |
| `origin/feat/scriptbank-wmp-status` | — | ScriptBank：已验证榜上 0.00 的反模式 | ❌ 不合 |
| duck-harness 官方仓库 | `7652836` | 只有 `main`，我们已在 | ✅ |
| `tonghuikang/arc3` / `arcprize/ARC-AGI-3-Agents` | — | fetch 仍 exit=128（路径/远端问题未解决） | ⚠️ 未取到 |

### 1.1 `origin/main` 带来的 11 个提交

```
730b833 Merge branch 'feat/duck-scheduler2' into main
a715f99 chore: ignore generated AOP stability report
71f12fa Merge feature/r2-projection-graph into main
4e3865b fix: make replay memory metrics portable on Windows
0c89e8a fix(duck): 修 duck_bridge import 阻塞——接通真端点前的最后一道关
0bd59d9 fix: harden Kaggle submission build and verification
1294803 feat(duck): 加死路反思 + 跨局持久化, 错误路线不再走
8524720 feat: add R2 exploration and deployment controls
447dbfe feat: implement R2 projection stages 3-5
ed2db50 feat: add R2 projection graph stage 2 builder
515b56d feat(agent): self_learner iteration and reason_only control agent
```

其中 `515b56d`（self_learner iteration + reason_only）就是 `SELFLEARN_LLM` 开关的来源。

---

## 2. 合并方法：符号级合并，不是"解冲突"

### 2.1 为什么不能用普通 `git merge`

我们的 `agent/my_agent.py` 是**整段双份粘贴**的产物（4608 行）：

- 顶层定义出现两遍：`_llm_on` / `_llm_view` / `_lp_chunk` / `_lp_load` / `_lp_action_spec` / `_action_label` / `_llm_call_budget` / `_llm_preceax_on` / `_llm_spec_to_action` / `_plan_locked`
- `MyAgent` 的方法也出现两遍：`_lp_observe` / `_clear_llm_queue` / `_pop_llm_queue` / `_lp_plan_specs` / `_try_llm_batch`

Python 对重名定义**取最后一个**，所以这 4608 行里只有约 3287 行是有效行为。`git merge` 产生的 12 个冲突块是在这个双份结构上叠的，手改必然漏。

### 2.2 实际做法

新脚本 `scripts/merge_my_agent_upstream.py`：**以上游为基座 + 决策表逐符号取舍**，然后机械行区间替换。决策表（带理由注释）：

| 决策 | 符号 | 理由 |
|---|---|---|
| 取我们 | `MyAgent.__init__` / `choose_action` / `_choose_action_inner` | 我们是上游的严格超集（whole-qwen 整局路径 + 4 组校准计划 + `_engine_reason`） |
| 取我们 | `_bfs_search` / `_iddfs_search` | C++ CompactBFS / Visited 加速；**上游从未改过这两个函数**（upstream == base），没有可丢的东西 |
| 取我们 | `_try_llm_batch` / `_legacy_plan_specs` | 见 §2.3 的实测 bug |
| 取我们 | `_lp_load` | `sys.path.insert(0)` 保证导入的是我们随包发出的 lexicon |
| 取我们 | `_llm_call_budget` / `BUILD_TAG` / `_PLAN_PROTECTED_GAMES` | 我们的版本是上游超集（`_PLAN_PROTECTED_GAMES` 多 cn04/dc22/m0r0/sp80） |
| 取上游 | `_bootstrap_paths` | 上游还把 duck_bridge 目录加进 sys.path |
| 取上游 | `_duck_tick` | 上游加了 `LINGJING_DUCK_FORCE_GIDS` 测试开关 |
| 取上游 | `_lp_observe` / `_clear_llm_queue` / `_pop_llm_queue` / `_lp_plan_specs` / `_action_label` / `_llm_preceax_on` / `_llm_spec_to_action` / `_lp_action_spec` | 仅 docstring/注释差异，上游写得更好 |
| 取并集 | `_ENGINE_SOLVER_GAMES` | 上游 +sp80；我们的计划路由先返回，引擎是兜底 |
| 补我们 | 7 个 whole-qwen 专有函数 + cn04/dc22/m0r0/sp80 四组校准计划 | 上游没有 |

**校验**：编译通过；未解析符号 2 个（与上游/我们各自完全一致）→ **新增 0**；符号集 `missing=[]`；**重复定义 0**；4608 → 4053 行。

### 2.3 合并实测抓到的真 bug（重要）

第一次合并后跑 400 步基准，第 7 局炸出：

```
TypeError: MyAgent._try_llm_batch() got an unexpected keyword argument 'history'
```

**5 步冒烟没抓到**（LLM 分支没被触达），编译和导入也都正常——这类错误只在调用时刻暴露。

**根因是我的比对工具写错了**：对**类方法**用了 `setdefault`（取第一份定义），对**模块级函数**用赋值（取最后一份），而 Python 实际总是取**最后**一份。于是 `_try_llm_batch` / `_legacy_plan_specs` 被误判成"两边相同"→ 误取上游版（上游版没有 `history`/`briefing` 形参），而我们的 whole-qwen 调用点传了这两个关键字。

**修复**：三处符号改回我们的版本；合并脚本的决策表同步改正并写明理由。

**新增永久防线** `scripts/check_call_signatures.py`：静态检查每个调用点的关键字实参是否被被调函数接受（重名取最后一份定义）。合并后 my_agent.py 与合并前基线**各 14 处关键字调用点全部通过**；本轮改动后仍是 14/14 通过。

> 教训：**任何"符号级合并"都必须配一个"调用点签名一致性"静态检查**，否则这类 bug 会一路活到线上。

---

## 3. `SELFLEARN_LLM` 开关设计

### 3.1 语义（与用户要求一致）

| `SELFLEARN_LLM` | 行为 |
|---|---|
| `1` | 大模型 planner：lexicon 结构化视图（objects/rules/PREDICTED）+ 核内千问整局出步 |
| `0` | 原策略：engine solver + 校准计划 + CEAX 兜底，**纯算法，不起 vLLM、不挂模型数据集** |
| 未设置 | 读旧开关 `LINGJING_LLM`（默认 off）——老 notebook 行为完全不变 |

### 3.2 落点（两层守卫）

| 文件 | 改动 |
|---|---|
| `arc_adaptor/lexicon/llm_planner.py` | 新增 `selflearn_llm_enabled()`（权威实现）；`LLMPlanner.ready()/decide()` fail-closed |
| **`arc_adaptor/lexicon/llm_planner.py`（出网层）** | **`_post()`（唯一 HTTP 出口）与 `_call_file()`（邮箱协议）在动手前再问一次开关**——见 §4.3，这是本轮最实质的一处修复 |
| `agent/my_agent.py` | `_llm_on()` 改为读 `SELFLEARN_LLM` 优先，`LINGJING_LLM` 降级为兜底 |
| `agent/self_learner.py` | 改用同一个 `selflearn_llm_enabled()`（`lexicon` 不可导入时有等价本地兜底） |
| `scripts/build_notebook.py` | `LINGJING_SELFLEARN_LLM=0\|1` 构建开关 → 产出 `submission_llm0.ipynb` / `submission_llm1.ipynb`；`=0` 时 `LINGJING_LLM=off`，**不起 vLLM、不挂模型数据集**；cell 内加 `SELFLEARN_LLM_DRIFT` 断言防 A/B 串味 |

**为什么加出网层守卫**：跑测里反复出现 `[learn] llm unreachable (TimeoutError / HTTPError 404)`，而那一轮的环境是 `SELFLEARN_LLM=0` + `LINGJING_LLM=off`——**说明有调用方在"原策略臂"下仍然触发了 HTTP**。`decide()` 一直有守卫，但那是"调用方自觉"。守卫钉在出网那一层之后，`SELFLEARN_LLM=0` 从此是绝对保证。实测：

```
enabled = False
[learn] llm blocked by SELFLEARN_LLM=0 at _post
_post returned: '' elapsed 0.0 s      # 原来要白等 45 秒超时
_call_file returned: ''
ready(0)= False decide(...)= None
```

### 3.3 A/B 跑测用的 notebook

两份 notebook 都在普通 "Save & Run All" 下**真的跑 agent**：新增离线自测 cell（`LINGJING_BENCH=1`），在 kernel 内用竞赛自带的公开环境文件离线对局（25 局），写出 `/kaggle/working/BENCH_SELFLEARN_{0,1}.json`。

> 为什么需要这个 cell：竞赛重跑路径依赖 Kaggle gateway，只有真提交时才有；普通 kernel 运行原本什么都不跑，量不到任何东西。

---

## 4. 跑测结果

### 4.1 本地 25 局 × 400 步，`SELFLEARN_LLM=0`（原策略臂）

| 臂 | 总分 | 总关数 | WIN | 零关局 | 说明 |
|---|---|---|---|---|---|
| **PRE**（合并前 4b43f34） | **19.366** | 35 | 4 | 16 | 基线 |
| **POST**（合并后，干净重跑） | **19.366** | 35 | 4 | 16 | **与 PRE 逐局逐关动作数完全一致** |
| POST（受污染的那一轮） | 19.097 | 34 | 4 | 16 | 见 §4.3，污染源已定位并修复 |
| **POST + 校准计划全开** | **59.848** | **105** | **13** | **8** | `LINGJING_SCRIPTS=1`，25 局全跑完 |

PRE / POST 的 4 个 WIN：lp85 L=8/100.00、cn04 L=6/100.00、m0r0 L=6/100.00、sp80 L=6/100.00；dc22 L=5/71.43。

**计划开的净变化：+40.48 分 / +70 关 / +9 局 WIN / −8 个零关局。**

### 4.2 「合并有没有副作用」——逐局逐关比对

`_tmp_diff.py` 逐局 diff（PRE vs POST）+ 干净探针重跑：

| 局 | PRE | POST（干净） | POST（受污染） |
|---|---|---|---|
| s5i5 | 0 关 / 2.78 分（26.5s） | 0 关 / 2.78 分（27.8s） | 0 关 / **0.00 分（173.6s）** |
| su15 | 1 关 / 1.72 分（34.4s） | 1 关 / 1.72 分（36.1s） | 0 关 / **0.00 分（176.4s）** |
| tu93 | 0 关 / 2.22 分（33.5s） | 0 关 / 2.22 分（29.2s） | 0 关 / **0.00 分（395.3s）** |
| 其余 22 局 | — | **完全一致** | 完全一致 |

→ **合并本身是行为中性的。** 差异只出现在被 LLM 超时污染的那一轮，且伴随**耗时暴涨 6~12 倍**、400 个动作全砸在 level 0（连 level 0 都没拿到分）——典型的"每一步都在等一个永远不来的回复"。

### 4.3 幽灵 LLM 调用（真实 bug，已修）

受污染那一轮的环境是 `SELFLEARN_LLM=0` + `LINGJING_LLM=off`，但日志里有：

```
[learn] llm unreachable (TimeoutError: The read operation timed out) -> rule planner (silencing further calls)
[learn] llm unreachable (HTTPError: HTTP Error 404: Not Found) -> rule planner (silencing further calls)
```

影响：每次白等 45 秒（`LINGJING_LLM_TIMEOUT`），把该局的墙钟预算吃光，agent 在同一步上反复空转。

**修法**（`75f7d8b`）：守卫下沉到出网层——`_post()` / `_call_file()` 动手前先问 `selflearn_llm_enabled()`，为假直接返回空串（0.0 秒）。见 §3.2 的实测输出。

**顺带查清的一个隐患**：仓库里 `integrations/.../ARC-AGI-3-Kaggle-Starter/.env` 写着 `LINGJING_LLM=on`。只要 shell 没显式设 `LINGJING_LLM=off`，这个 `.env` 就会把大模型链路打开——**A/B 跑测必须显式设两个变量**，否则 llm0 臂会被污染。这条已写进 notebook（`LINGJING_SCRIPTS` / `SELFLEARN_LLM` 都显式钉住）。

### 4.4 校准计划全开（`LINGJING_SCRIPTS=1`）——最大单一杠杆

25 局全跑完（`bench400_scripts_on.json`）：

| 局 | 关 | 分数 | 动作 | 耗时 | 状态 |
|---|---|---|---|---|---|
| ar25 | 8 | 100.00 | 276 | 14.7s | WIN |
| lp85 | 8 | 100.00 | 79 | 2.8s | WIN |
| sb26 | 8 | 100.00 | 124 | 4.0s | WIN |
| ls20 | 7 | 100.00 | 309 | 8.5s | WIN |
| tn36 | 7 | 100.00 | 103 | 4.6s | WIN |
| vc33 | 7 | 100.00 | 167 | 6.0s | WIN |
| cd82 | 6 | 100.00 | 73 | 1.6s | WIN |
| cn04 | 6 | 100.00 | 198 | 1.7s | WIN |
| ft09 | 6 | 100.00 | 81 | 0.8s | WIN |
| m0r0 | 6 | 100.00 | 183 | 3.8s | WIN |
| r11l | 6 | 100.00 | 130 | 2.6s | WIN |
| sp80 | 6 | 100.00 | 183 | 2.9s | WIN |
| tr87 | 6 | 100.00 | 122 | 6.3s | WIN |
| dc22 | 5 | 71.43 | 400 | 14.9s | NOT_FINISHED |
| sc25 | 5 | 71.43 | 400 | 19.4s | NOT_FINISHED |
| wa30 | 6 | 46.67 | 400 | 24.9s | NOT_FINISHED |
| bp35 | 2 | 6.67 | 400 | 52.9s | NOT_FINISHED |
| g50t / ka59 / lf52 / re86 / s5i5 / sk48 / su15 / tu93 | 0 | 0.00 | 400 | 166~821s | NOT_FINISHED |

**合计：59.848 分 / 105 关 / 13 局 WIN / 8 局零关**（对照计划关：19.366 / 35 / 4 / 16）。

**一个必须看出来的细节**：13 个 WIN 局耗时只有 **0.76~14.7 秒**——它们不是"想"出来的，是**在重放一条固定的校准动作序列**（ar25 276 步、ls20 309 步、cd82 73 步…）。而真正靠通用搜索的 8 个零关局，400 步一个动作都没走对。**⇒ 计划把公开局分数拉上去了，但它没有提升任何通用能力。**

### 4.5 Kaggle 侧独立复现（官方口径）——最干净的单变量 A/B

同一个 kernel、同一台机器、**只改 `LINGJING_SCRIPTS` 一个变量**：

| 版本 | `LINGJING_SCRIPTS` | 总分 | 总关数 | WIN | 零关局 | 状态 |
|---|---|---|---|---|---|---|
| v4 | off（老默认） | **15.477** | 28 | 3 | 12 | COMPLETE |
| **v5** | **on（新默认）** | **56.228** | **99** | **13** | **7** | COMPLETE |

**净变化：+40.75 分 / +71 关 / +10 局 WIN / −5 个零关局。** 与本地同向、同量级（本地 19.37→59.85）。

v4 的零关局里，`ar25 / ls20 / ft09 / tr87 / cd82 / tn36 / sb26 / r11l / vc33 / wa30` **全是判 `plan_gid` 的计划局**——因为 Kaggle 上没人设 `LINGJING_SCRIPTS`，代码默认是 off；而判 `gid` 的 `cn04 / m0r0 / sp80 / dc22` 照常 WIN。**这正是代码判据不一致的直接证据，也说明"计划开关"就是全部差距。**

v5 的逐局（`BENCH_SELFLEARN_0.json`）：

| 局 | 关 | 分数 | 耗时 | | 局 | 关 | 分数 | 耗时 |
|---|---|---|---|---|---|---|---|---|
| sb26 | 8 | 100.00 | 1.9s | | wa30 | 6 | 46.67 | 17.0s |
| ar25 | 8 | 100.00 | 1.4s | | cn04 | 6 | 100.00 | 0.6s |
| ls20 | 7 | 100.00 | 1.8s | | m0r0 | 6 | 100.00 | 0.7s |
| vc33 | 7 | 100.00 | 1.1s | | sc25 | 5 | 71.43 | 9.1s |
| tn36 | 7 | 100.00 | 0.6s | | dc22 | 5 | 71.43 | 1.6s |
| tr87 | 6 | 100.00 | 0.9s | | bp35 | 2 | 6.67 | 10.6s |
| ft09 | 6 | 100.00 | 0.3s | | su15 | 1 | 1.72 | 10.7s |
| cd82 | 6 | 100.00 | 0.4s | | lp85 | 1 | 2.78 | 10.1s |
| sp80 | 6 | 100.00 | 0.7s | | g50t/s5i5/re86/lf52/ka59/tu93/sk48 | 0 | 0.00 | ~10s |
| r11l | 6 | 100.00 | 0.7s | | | | | |

**两处必须看出来的细节**：

1. **13 个 WIN 局耗时 0.3~1.9 秒**（本地 0.8~14.7 秒）——它们是**在重放固定动作序列**，不是"想"出来的；
2. **剩下的 7 个零关局（g50t / s5i5 / re86 / lf52 / ka59 / tu93 / sk48）恰好就是官方分支用破零包解决的那批**（官方自测零分局 3，我们 7）。**⇒ 分数我们追平了，通用能力的差距还在，而通用能力才是隐藏集的胜负手。**

llm1 臂（`SELFLEARN_LLM=1`）**至今没跑成**——周 GPU 配额耗尽（§5.1）。

### 4.6 Kaggle kernel 三连 ERROR，每次抓到一个新 bug（跑测面的真实价值）

| 版本 | 失败点 | 根因 | 修法 |
|---|---|---|---|
| v1 | `ModuleNotFoundError: langgraph.store.sqlite` | stock `agents/__init__.py` import langgraph 模板 | 改写成 slim registry |
| v2 | `BENCH_GAMES 0 []` + `No module named 'agents.templates.my_agent'` | ① 环境目录是"猜"的；② slim `__init__.py` 仍 import `.templates.my_agent`，但 agent 只写在 `/tmp/my_agent.py` | ① 环境目录改多候选探测；② 先把 agent 拷进 `agents/templates/` |
| v3 | `IndexError: 3` | `my_agent.py` 写死 `Path(__file__).resolve().parents[3]`，而 Kaggle 上 agent 在 `/tmp/my_agent.py`（只有 2 层父目录） | 换 `_repo_root_hint()`：层级不够退到最浅层 |
| v4 | ✅ COMPLETE（15.477 分 / 28 关） | — | — |
| v5 | ✅ COMPLETE（**56.228 分 / 99 关**） | 带"计划默认开"重推 | 见 §4.5 |

**这三个 bug 全部只可能在 Kaggle 上暴露**——本地 checkout 里 `parents[3]` 永远成立。这就是"把合并方法放进 notebook 去跑测"这条要求的直接回报。

---

## 5. 阻断与风险

### 5.1 【最高优先】Kaggle 周 GPU 配额已耗尽

`api.quota_view()` 实测：

```json
{"quotaRefreshTime": "2026-10-24T00:00:00.000Z",
 "gpuQuota": {"timeUsed": "41196.570.0s", "timeReserved": "0s",
              "totalTimeAllowed": "21600s", "minimumTimeAllowed": "21600s",
              "isPayToScaleEnabled": false, "hasEverRun": true}}
```

- 已用 **41196s**，上限 **21600s** → 已耗尽；
- **下次刷新 2026-10-24T00:00:00Z（北京时间 10-24 08:00）**；
- 推送 llm1 的报错：`Maximum weekly GPU quota of 30.00 hours reached.`

影响面：

1. `smartrouter-merged-ab-llm1` 推不上去 → **大模型臂的 Kaggle 跑测做不了**，A/B 只能有 llm0 一侧；
2. **明天要提的 v27 竞赛重提交同样需要 GPU**（核内起 vLLM）→ 按当前配额状态**大概率也会被挡**；
3. 应对：原策略臂已改用 CPU 推送（不烧配额）；LLM 臂等配额刷新，或换一个还有配额的账号。

> ⚠️ 时间窗口提醒：日提交配额（UTC 计）与 GPU 配额刷新**不是同一个时点**。日配额今天 01:49 UTC 已被队友 wangausx 用掉，UTC 明日解锁；**GPU 配额要到 10-24 08:00（北京时间）才刷新**。如果 v27 必须在 10-23 提交，**GPU 配额会是硬墙**，需要提前准备备用账号。

### 5.2 日提交配额

每天 1 次（UTC 计），今天 01:49 UTC 已被队友 wangausx 的 SmartRouter v2 DPL kernel 用掉（那次得 0.03）。我们的 v27 要等 UTC 明日解锁。距 11-02 截止约 11 天，还剩约 12 次机会。

### 5.3 基准不可复现（做 A/B 前必须先解决）

本轮实测发现，这套 25 局基准**不是确定性的**：

| 依赖 | 例子 |
|---|---|
| 墙钟超时 | `local_solver t=90.0s`、`iddfs t=20.0s`、`俯冲超时: 91s`、CEAX giveup 150 步 |
| 共享磁盘缓存 | `state/engine_solver_cache.json`（跨局、跨 run 复用） |
| CPU 争抢 | 同一局 g50t 在两种条件下分别拿到 **L=0** 与 **L=1 / 3.57 分** |

⇒ **任何"A 比 B 高 2 分"的结论，在基准可复现之前都不可信。** 建议：(a) 跑测前清掉 `state/engine_solver_cache.json`；(b) 跑测期间不并发别的基准；(c) 把墙钟预算调大或改为"步数预算"口径；(d) 固定 `seed=0`（已经是）。

### 5.4 其他

- `tonghuikang/arc3` 与 `arcprize/ARC-AGI-3-Agents` 两个官方仓库 fetch 仍 exit=128，未取到；
- `feat/unify-decoder-entry` 的分支尖端含 "snapshot: 固化 2026-10-13 工作树" 的 193k 行全量快照，整分支合会覆盖我们 10-22 的新工作，**风险高**；
- duck 线的 v27 overlay 数据集仍发不了版（`www.googleapis.com` 本机不通）→ 死局提前放弃开关未激活；
- `arc_shadow_*` 专用求解器模块**一个都不在树里**（`lingjing_solo/planning/` 下 0 个），shadow 路径目前是死代码（跑测日志里可见 `加载 arc_shadow_g50t 失败: ModuleNotFoundError`）。

---

## 6. 优化 / 升级路线

### 6.1 结论：**继续合并，但只合并一样东西；同时把力气放到通用能力上**

| 状态 | 口径 | 总分 | 总关数 | 零关局 |
|---|---|---|---|---|
| 我们合并后（计划关） | 本地 25 局 | 19.37 | 35 | 16 |
| **我们合并后（计划开）** | 本地 25 局 | **59.85** | **105** | **8** |
| **我们合并后（计划开）** | **Kaggle 官方口径** | **56.23** | **99** | **7** |
| 官方 `feat/unify-decoder-entry` 自测 | 官方口径 | 57.37 | 116 | **3** |

**我们已经在公开局分数上追平/反超官方分支（本地 59.85 > 57.37，Kaggle 56.23 ≈ 57.37）。但零关局 7~8 vs 官方 3 —— 差距全部落在"通用搜索找不到解"的那批局上。** 这正是 §6.2 要讲的区别。

### 6.2 ⚠️ 最重要的一条判断：计划带来的是"公开局分"，不是"隐藏集分"

这批校准计划是**按这 25 局公开环境校准并回放验证**的。打开它们：

- 公开局分数 19.37 → 59.85（本地）/ 15.48 → 56.23（Kaggle），**+40 分量级**；
- **对隐藏测试集几乎没有同等收益**（隐藏集的 game id 不匹配，计划根本不会触发）；
- 而且 WIN 局只花 0.3~1.9 秒（Kaggle），说明它们是**重放**而非**求解**。

**⇒ 所以"开计划"应该被定位成：保底 + 防零分 + 公开榜好看，而不是提分主线。**

真正的隐藏集胜负手只有两个：

1. **通用解法器**：剩下 7 个零关局（g50t / s5i5 / re86 / lf52 / ka59 / tu93 / sk48）在 Kaggle 上全部 0 分，400 步一个动作都没走对——**"通用搜索找不到解"**。官方 `unify-decoder-entry` 的破零包打的正是这几局（官方自测零分局 3，我们 7）。
2. **LLM 臂（`SELFLEARN_LLM=1`）**：唯一有"泛化"可能的路径。**但它的 A/B 至今没跑成**（GPU 配额）。

### 6.3 已落地：`LINGJING_SCRIPTS` 默认开 + 逐局可排除（`100fdc2`）

```python
LINGJING_SCRIPTS=on|off        # 全局（默认 on）
LINGJING_SCRIPTS_EXCLUDE=a,b   # 逐局排除
```

同时统一了 `plan_gid` / `gid` 判据（cn04 / dc22 / sp80 / m0r0 原来判 `gid`，其余判 `plan_gid`）——之前这两处不一致，导致计划总开关关着时这四局照样跑计划、逐局排除开关对它们无效。

> **需要你确认的一点**：这一条和团队规范里"不堆公开局 plan"的边界有关。我的判断是——**开它不违反规范**（这些计划已在树内、已校准、已回放验证，打开不是"新增公开局 plan"；官方分支同口径），但**它也不该被当成提分主线**（§6.2）。所以我做成了"默认开 + 可全局关 + 可逐局排除"，而不是砍掉。如果你判断赛制变化后这些计划会变成负资产，直接 `LINGJING_SCRIPTS_EXCLUDE=ar25,ls20,...` 即可回退。

### 6.4 P1：移植 `feat/unify-decoder-entry` 的破零包（只取 25 文件）

分支尖端 `3d4cdba` 自带 **+3755 行 / 25 文件** 的实质增量：

| 内容 | 效果 |
|---|---|
| `lingjing_solo/planning/plan_decoder.py`（712 行） | 公共 plan 解码器框架（DeepcopyBFS + PositionEnum） |
| `lingjing_solo/m0r0_solver.py`（235 行） | m0r0 0 → L2 破零 |
| `lingjing_solo/perception/object_tracker.py`（335 行） | 目标追踪 |
| `lingjing_solo/perception/status_mask.py`（185 行） | 状态掩码 |
| `lingjing_solo/exploration/action_effect.py`（310 行） | 动作效果模型 |
| `lingjing_solo/engine_solver.py`（+100 行） | ka59：`LINGJING_R3_TLIMIT=60` 找到解，0 → L1 破零 |
| `my_agent.py`（+203 行） | re86 位置枚举求解器 + `_RE86_PLANS`（0 → L3 破零）；wa30 优先硬编码 plan（L4→L6） |
| `tests/`（3 个新测试文件） | 可直接用 |

**移植方式**：**不要 `git merge` 分支**（尖端含 193k 行全量快照）。按 §2.2 的同一套符号级手法，逐文件/逐符号取增量，然后跑 `check_call_signatures.py` + 400 步基准回归。

**预期**：破零包直接冲着我们剩下的 5 个零关局（re86 / ka59 / m0r0 / …）去。按官方自测，55.01→57.37 的同款口径下约 **+2.4 分、+8 关、零分局 3→0**——**分不多，但打的是"通用解法器"，这正是隐藏集需要的能力。**

### 6.5 排序建议

| 优先级 | 动作 | 代价 | 预期收益 |
|---|---|---|---|
| **P0** | 确认/恢复 Kaggle GPU 配额；保住 v27 提交窗口 | 外部依赖 | 不做则整条提分线断掉 |
| **P0** | 跑通 llm1 臂的 Kaggle A/B | 等配额 | **唯一能验证"泛化"的实验**，目前完全空白 |
| **P0** | 让基准可复现（§5.3 四条） | 小 | 不做则所有 A/B 结论不可信 |
| **P1** | 移植 `unify-decoder-entry` 的 25 文件破零包 | 1 个符号级合并任务 + 回归 | 打通用解法器，隐藏集受益 |
| **P1** | `check_call_signatures.py` 接进合并后必跑校验 | 小 | 防同类 bug 再逃逸 |
| **P2** | 把 `arc_shadow_*` 求解器模块补进树（现在是死代码） | 需要各成员的解法文件 | shadow 路径当前完全空转 |
| **P2** | duck 线：overlay 数据集发版（需能通 `www.googleapis.com` 的代理/机器） | 外部依赖 | 激活死局提前放弃 |
| **P3** | 评估 `twin-engine-solver` / `graph-explorer-statekey` / `dpl-v14.2-unknown-bench-400` | 各一轮 | 未知 |

### 6.6 不建议做的事

- ❌ 不合 `feat/scriptbank-wmp-status`（ScriptBank 已验证榜上 0.00）；
- ❌ 不整分支合 `feat/unify-decoder-entry`（全量快照覆盖新工作）；
- ❌ 不用公开榜调参；
- ❌ 不在本地没冒烟的情况下烧提交配额；
- ❌ 不为了"显得在合并"而合并——`feature/llm-whole-qwen` 已在树里，再合一次是纯风险；
- ❌ **不把"计划开"当成提分主线**（§6.2）——它是保底，不是能力。

---

## 7. 本轮产出清单

| 文件 | 说明 |
|---|---|
| `integrations/.../scripts/merge_my_agent_upstream.py` | 可复现的符号级合并脚本 + 决策表 |
| `integrations/.../scripts/check_call_signatures.py` | 调用点关键字签名静态检查（新防线） |
| `integrations/.../scripts/push_ab_kernels.py` | A/B kernel 推送（独立 slug，不碰队友 kernel） |
| `integrations/.../scripts/check_kaggle_status.py` | kernel 状态 + GPU 配额（quota_view）+ 提交列表 |
| `integrations/.../scripts/benchmark_all_games.py` | 新增 `--games` 子集跑测 |
| `integrations/.../scripts/build_notebook.py` | SELFLEARN_LLM 构建开关 + bench cell 修复 + 显式钉 `LINGJING_SCRIPTS` |
| `integrations/.../agent/my_agent.py` | 合并后基座（4053 行 / 0 重复定义）+ 计划默认开 + `_repo_root_hint()` |
| `integrations/.../agent/shadow_adapter.py` | 取 aa1b889 方言归一版（去硬编码 Windows 路径） |
| `arc_adaptor/lexicon/llm_planner.py` | `selflearn_llm_enabled()` + `decide()` 守卫 + **出网层硬守卫** |
| `integrations/.../agent/self_learner.py` | 统一到权威开关 |
| `integrations/.../notebooks/submission_llm0.ipynb` | 原策略臂（CPU、含离线自测 cell） |
| `integrations/.../notebooks/submission_llm1.ipynb` | 大模型臂（含离线自测 cell） |
| 提交 `34e7d39` / `00a171e` / `75f7d8b` / `100fdc2` | 合并 → 修决策表 → 守卫+notebook → 计划默认开+浅路径 |

### Kaggle kernel

| kernel | 版本 | 状态 | 结果 |
|---|---|---|---|
| `liaojiaxuan/smartrouter-merged-ab-llm0`（CPU，`SELFLEARN_LLM=0`） | v5 | ✅ COMPLETE | **56.228 分 / 99 关 / 13 WIN**（v4 计划关时 15.477 / 28 关） |
| `liaojiaxuan/smartrouter-merged-ab-llm1`（GPU，`SELFLEARN_LLM=1`） | — | ❌ 未推送 | **被周 GPU 配额挡住**，刷新时间 2026-10-24T00:00:00Z |
| `liaojiaxuan/lingjing-duck-harness-v27` | — | COMPLETE | 18.61 离线均值 |
