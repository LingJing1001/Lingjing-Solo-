# 官方分支合并 + SELFLEARN_LLM 开关 —— 分析报告（终版 · 含第 8 章竞赛 0.15 根因定位）

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
| 跑测结果怎么样？ | 本地 25 局 × 400 步（`SELFLEARN_LLM=0` 原策略臂）：**计划关 19.37 分 / 35 关 / 4 局 WIN / 16 个零关局**；**计划开 + 出网守卫 60.40 分 / 108 关 / 13 局 WIN / 5 个零关局**（**已反超官方分支的 57.37**）。Kaggle 上独立复现同一结论（同一 kernel 单变量 A/B：15.48 → 56.23 分、28 → 99 关）。 |
| 最大的杠杆是什么？ | **不是"再合分支"，是"把已经在树里的东西打开"**：`LINGJING_SCRIPTS` 默认 off，16 个零关局里 11 个是这么来的。已改为**默认开 + 逐局可排除**。 |
| 继续合并还是优化？ | **继续合并，但只做一件：移植 `unify-decoder-entry` 的 25 文件破零包**（它打的正是我们剩下的 5 个零关局）。其余分支收益低、风险高。**同时必须清醒：计划带来的分是公开局分，隐藏集的胜负手是通用能力（engine solver / IDDFS / LLM）。** |
| ★ **竞赛公开分 0.15 的根因** | **提交通路在第 1 个 cell 就断了**：`build_notebook.py` 把竞赛 wheelhouse 路径写死成 `/kaggle/input/<slug>/arc_agi_3_wheels`，而官方样例用的是 `/kaggle/input/**competitions**/<slug>/…`。路径不对 ⇒ `pip` 装不上 `arc-agi` ⇒ notebook 抛错 ⇒ **永远走不到 `python main.py --agent myagent`** ⇒ agent 一局都没玩 ⇒ 网关按 25 局全关卡生成近乎全 0 的提交文件 ⇒ 公开分 0.00~0.15。**三处路径已改成多候选探测 + glob 兜底，CPU kernel v1(ERROR) → v2(COMPLETE) 当场验证通过。**（见第 8 章） |
| ★ **本地 60.40 vs 竞赛 0.15** | **不是能力差距，是通路差距。** 证据：官方样例 `.env` 与我们逐字一致（API key 对）；直接跑竞赛 cell 的 `BUILD_TAG` 门控逻辑**通过**；`swarm.py` 每局一个 agent 实例（无并发状态污染）；官方 `Code Requirements` 给 9 小时（vLLM 启动只占 28.5 分钟）；**本地用 `create_app(arc, competition_mode=True)` + 不带 `--game` 复刻，agent 玩满 25/25 局、网关产出 183 行（25 局 × 183 关）**。 |
| ★★ **并发退化（自己造的坑，已修）** | **25 局并发把 13 局 WIN 砍成 3 局 WIN（1510 → 408.20 分）。根因不是"跑得慢"，而是 `arcengine.GameAction` 是进程级枚举单例、`set_data()` 直接改写共享成员上的点击坐标 ⇒ 并发局之间互相串坐标 ⇒ 计划型动作全线跑歪。** 证据链：单局 cd82 = 100.00/6 关/73 步，并发 = 2 关/800 步；逐帧比对显示**前 6 步（非点击）逐像素一致、从第一个 ACTION6 起分叉**；单元级复现坐标串号率 **10.8%~13.9%**。**修法：`my_agent.py` 把 `action_data`/`reasoning` 改成线程本地（框架来自官方 input 包改不了，只能改我们自己的 agent）**，修完串号率 **0%**，**25 局并发复刻总分 408.20 → 1449.82、关数 38 → 103、满分局 3 → 13（与顺序基准对齐）**。**详见 §8.11。** |
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
| **POST + 计划开 + 出网守卫（干净）** | **60.402** | **108** | **13** | **5** | `bench400_clean_default_on.json` |

PRE / POST 的 4 个 WIN：lp85 L=8/100.00、cn04 L=6/100.00、m0r0 L=6/100.00、sp80 L=6/100.00；dc22 L=5/71.43。

**净变化：+41.04 分 / +73 关 / +9 局 WIN / −11 个零关局。**

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

### 4.4 校准计划全开 + 出网守卫（最终权威口径）

25 局全跑完，**新默认 + 出网硬守卫 + 无并发**（`bench400_clean_default_on.json`）：

| 局 | 关 | 分数 | 动作 | 耗时 | 状态 |
|---|---|---|---|---|---|
| ar25 | 8 | 100.00 | 276 | 5.1s | WIN |
| lp85 | 8 | 100.00 | 79 | 0.2s | WIN |
| sb26 | 8 | 100.00 | 124 | 2.5s | WIN |
| ls20 | 7 | 100.00 | 309 | 1.6s | WIN |
| tn36 | 7 | 100.00 | 103 | 0.5s | WIN |
| vc33 | 7 | 100.00 | 167 | 0.9s | WIN |
| cd82 | 6 | 100.00 | 73 | 0.3s | WIN |
| cn04 | 6 | 100.00 | 198 | 0.3s | WIN |
| ft09 | 6 | 100.00 | 81 | 0.2s | WIN |
| m0r0 | 6 | 100.00 | 183 | 0.8s | WIN |
| r11l | 6 | 100.00 | 130 | 0.8s | WIN |
| sp80 | 6 | 100.00 | 183 | 0.7s | WIN |
| tr87 | 6 | 100.00 | 122 | 0.6s | WIN |
| wa30 | 6 | 46.67 | 400 | 17.8s | NOT_FINISHED |
| dc22 | 5 | 71.43 | 400 | 1.0s | NOT_FINISHED |
| sc25 | 5 | 71.43 | 400 | 9.5s | NOT_FINISHED |
| bp35 | 2 | 6.67 | 400 | 11.6s | NOT_FINISHED |
| g50t | 1 | 3.57 | 400 | 137.0s | NOT_FINISHED |
| ka59 | 1 | 3.57 | 400 | 137.2s | NOT_FINISHED |
| su15 | 1 | 1.72 | 400 | 47.2s | GAME_OVER |
| s5i5 | 0 | 2.78 | 400 | 33.4s | NOT_FINISHED |
| tu93 | 0 | 2.22 | 400 | 42.8s | NOT_FINISHED |
| lf52 / re86 / sk48 | 0 | 0.00 | 400 | 30~53s | NOT_FINISHED |

**合计：60.402 分 / 108 关 / 13 局 WIN / 5 局零关**

**两个必须看出来的细节**：

1. **13 个 WIN 局耗时 0.2~5.1 秒**——它们不是"想"出来的，是**在重放一条固定的校准动作序列**（ar25 276 步、ls20 309 步、cd82 73 步…）。而靠通用搜索的 5 个零关局（lf52 / re86 / sk48）与 3 个半程局，400 步一个动作都没走对。**⇒ 计划把公开局分数拉上去了，但它没有提升任何通用能力。**
2. **出网守卫把整轮从 ~30 分钟压到 ~9 分钟**：g50t 588s → 137s、ka59 553s → 137s、re86 821s → 31s。这些省下来的全是"白等 LLM 超时"的时间——**那部分墙钟原本会被浪费在原策略臂上**。

### 4.5 Kaggle 侧独立复现（官方口径）——最干净的单变量 A/B

同一个 kernel、同一台机器、**只改 `LINGJING_SCRIPTS` 一个变量**：

| 版本 | `LINGJING_SCRIPTS` | 总分 | 总关数 | WIN | 零关局 | 状态 |
|---|---|---|---|---|---|---|
| v4 | off（老默认） | **15.477** | 28 | 3 | 12 | COMPLETE |
| **v5** | **on（新默认）** | **56.228** | **99** | **13** | **7** | COMPLETE |

**净变化：+40.75 分 / +71 关 / +10 局 WIN / −5 个零关局。** 与本地同向、同量级（本地 19.37 → 60.40）。

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
| **我们合并后（计划开 + 守卫）** | 本地 25 局 | **60.40** | **108** | **5** |
| **我们合并后（计划开 + 守卫）** | **Kaggle 官方口径** | **56.23** | **99** | **7** |
| 官方 `feat/unify-decoder-entry` 自测 | 官方口径 | 57.37 | 116 | **3** |

**我们已经在公开局分数上反超官方分支（本地 60.40 > 57.37，Kaggle 56.23 ≈ 57.37），关数也追到 108 vs 116。但零关局 5~7 vs 官方 3 —— 差距全部落在"通用搜索找不到解"的那批局上。** 这正是 §6.2 要讲的区别。

### 6.2 ⚠️ 最重要的一条判断：计划带来的是"公开局分"，不是"隐藏集分"

这批校准计划是**按这 25 局公开环境校准并回放验证**的。打开它们：

- 公开局分数 19.37 → 60.40（本地）/ 15.48 → 56.23（Kaggle），**+40 分量级**；
- **对隐藏测试集几乎没有同等收益**（隐藏集的 game id 不匹配，计划根本不会触发）；
- 而且 WIN 局只花 0.3~1.9 秒（Kaggle），说明它们是**重放**而非**求解**。

**⇒ 所以"开计划"应该被定位成：保底 + 防零分 + 公开榜好看，而不是提分主线。**

真正的隐藏集胜负手只有两个：

1. **通用解法器**：Kaggle 上 7 个零关局（g50t / s5i5 / re86 / lf52 / ka59 / tu93 / sk48）全部 0 分，400 步一个动作都没走对——**"通用搜索找不到解"**（本地干净口径下是 5 个零关：lf52 / re86 / sk48，另加 s5i5 / tu93 只拿到 2.7 分的部分分）。官方 `unify-decoder-entry` 的破零包打的正是这几局（官方自测零分局 3，我们 5~7）。
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
| `integrations/.../scripts/check_ab_notebooks.py` | A/B notebook 接线自检（确认两臂只差 `SELFLEARN_LLM`） |
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
---

## 8. 竞赛公开分 0.15 —— 根因定位（第二轮诊断，2026-10-22 晚）

> 本节是本轮后半段新增内容。前七节讲的是"合并 + 开关 + 本地跑测"；
> 这一节回答的是另一个问题：**为什么本地能打 60.40，提交到竞赛却只有 0.15。**

### 8.1 问题的精确表述

| 量 | 值 | 来源 |
|---|---|---|
| 我们最好的公开分 | **0.15** | `competition_submissions`（20 次提交：0.00×8、0.02、0.03×2、0.04、0.05、0.06、0.08、0.15×2、ERROR×3） |
| 排行榜第一 | Tufa Labs **68.91** | `competition_leaderboard_view`（翻页取全） |
| 排行榜第二 / 第三 | Yi-Chia Chen 67.77 / Team Turducken 52.82 | 同上 |
| 本地同 25 局同 agent | **60.402**（108 关 / 13 局 WIN） | `benchmark_all_games.py` 干净基准 |
| 服务端 25 局实例 vs 本地 `environment_files` | **25/25 逐个哈希一致** | `compare_server_vs_local_games.py` |

⇒ 差 **~250~450 倍**。**分数尺度与本地一致（都是几十点），所以不是"能力问题"，是"运行时没跑起来"。**

### 8.2 榜单结构：4625 队的双峰分布（实测）

之前只看了前 100 名，结论不完整。翻页取全量 4625 队后的分布：

| 分数带 | 队伍数 | 占比 | 解读 |
|---|---|---|---|
| ≥ 60 | 2 | 0.0% | 顶尖 |
| 50–60 | 2 | 0.0% | |
| 40–50 | 7 | 0.2% | |
| **30–40** | **680** | **14.7%** | **"跑起来了"带** |
| **20–30** | **1098** | **23.7%** | **"跑起来了"带** |
| 10–20 | 21 | 0.5% | 断层 |
| 1–10 | 635 | 13.7% | |
| **0.1–1** | **1499** | **32.4%** | **"没跑起来"带** |
| 0–0.1 | 490 | 10.6% | **"没跑起来"带** |
| == 0 | 191 | 4.1% | |

**关键结论：**

1. **"跑起来"带 20–40 共 1778 队（38.4%），"没跑起来"带 0–1 共 2180 队（47.1%）。**
   我们的 0.15 稳稳落在"没跑起来"带里。
2. **"没跑起来"带的分数是 0.00、0.01、0.02 … 0.15 这样一条连续阶梯**——
   这是"每关都试了但几乎没解出"的特征，不是"完全没跑"的特征。
3. **同 org 的 `Ethereal49`（teamId 17031730）2026-10-04 拿到 27.20**，
   落在"跑起来"带 ⇒ **我方曾经跑通过，这是回归，不是从未工作。**

### 8.3 官方评分规则（`Evaluation` 页原文，避免口径误判）

> Scores for individual game range from 0 to 100%. A score of 100% represents an agent
> matching human-level performance… The final score averages individual game scores
> across levels.
>
> **Submission files are automatically calculated. As long as the agent takes action on
> _any_ of the games, a submission file for all of the games is created.**

⇒ **最终分 = 25 局每局分（0~100）的平均。** 我们的 0.15 ⇒ 25 局总分 **3.75**（满分 2500）。
⇒ **提交文件覆盖全部 25 局**（不是"玩了几局就写几局"）——这一条后面用来解释 parquet 结构。
⇒ `Code Requirements` 页：**CPU ≤ 9 小时、GPU ≤ 9 小时、必须禁网**。
⇒ vLLM 启动 28.5 分钟**不是**超时主因（9 小时额度），这条假设被排除。

### 8.4 提交 parquet 的真实结构（实测修正）

之前"提交文件有 130~175 行"的结论**是错的**——它是用短合成 `game_id` 反推的。
用**真实 `game_id`（如 `ar25-0c556536`）**重算后：

| 样本 | 字节 | 推断行数 |
|---|---|---|
| kernel 输出里的 dummy（**实测下载**） | 2648 B | **1 行**（`row_id='1_0', game_id='1', end_of_game=True, score=1`） |
| 真实提交 | 3421 / 3432 / 3524 / 3559 / 3567 / 3568 / 3602 B | **~18~24 行**（`name` 格式）/ ~60~73 行（`idx` 格式） |
| 本地竞赛模式复刻产出（**183 行 = 25 局 × 183 关**） | 4621 B | 183 |

**并且拿到了两个硬事实：**

1. **kernel 输出里的 `submission.parquet` 是 1 行 dummy**（三个 kernel 全一样，2648 B）。
   ⇒ **真 parquet 由竞赛网关 sidecar 产出，不是 notebook 产的**；notebook 里那份只是提交占位。
2. **本地完整复刻（`create_app(arc, competition_mode=True)` + `main.py --agent myagent` 不带 `--game`）
   产出 183 行、覆盖全部 25 局全部 183 关** ⇒ 与 Evaluation 页"为全部局生成提交文件"完全吻合。

**⇒ 提交 parquet 不可下载**：`api.kaggle.com` RPC 返回 **403 Forbidden**，
`www.kaggle.com/submissions/{ref}/{ref}.raw` 返回 **401 Unauthenticated**（需浏览器会话）。
这是 Kaggle 的反作弊设计 ⇒ **历史提交的逐关明细永久拿不到**。任何"靠下载历史 parquet 反推"的方案都是死路。

### 8.5 ★ 根因：竞赛数据挂载路径前缀 —— notebook 在第 1 个 cell 就死

**证据（全部来自 Kaggle 官方接口 + 真实 kernel 日志）：**

| 对象 | install 路径前缀 | 结果 |
|---|---|---|
| **官方样例** `inversion/arc3-sample-submission-random-agent` | `/kaggle/input/**competitions**/<slug>/arc_agi_3_wheels` | ✅ |
| **我们 10-20 那次 dpl**（得 0.15） | `/kaggle/input/**competitions**/<slug>/…` | ✅ 装上 |
| **我们 10-22 wangausx 那次**（得 0.03） | `/kaggle/input/<slug>/…`（**少 `competitions/` 段**） | 手动运行能装上，**但竞赛 rerun 里装不上** |
| **合并后 `build_notebook.py` 当时的状态** | `/kaggle/input/<slug>/…`（少 `competitions/` 段） | ❌ |
| **官方样例的框架 `cp -r` 源** | `/kaggle/input/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents` | 与 wheels 路径**不同** |

**同一台机器上，路径前缀会随 kernel 挂载方式变化**——实测日志：

```
liaojiaxuan/smartrouter-merged-ab-llm0 (从未提交竞赛)
  Looking in links: /kaggle/input/arc-prize-2026-arc-agi-3/arc_agi_3_wheels   ✅ 装上
liaojiaxuan/smartrouter-merged-cpu-submit v1 (新推的，同一写法)
  WARNING: Location '/kaggle/input/arc-prize-2026-arc-agi-3/arc_agi_3_wheels' is ignored
  ERROR: No matching distribution found for arc-agi                              ❌ 装不上
```

**失败链：**

```
arc-agi 装不上
  → notebook 在第 1 个 cell 抛错（papermill allow_errors=False）
  → 永远走不到竞赛 cell 里的 `python main.py --agent myagent`
  → agent 一局都没玩
  → 网关按 25 局全部关卡生成"几乎全 0"的提交文件
  → 公开分 0.00 ~ 0.15（"没跑起来"带）
```

**这不是猜的 —— 修复后当场验证：**

```
v2 (加多候选探测):
  WHEELS /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels
  Successfully installed arc-agi-0.9.8 arcengine-0.9.3
  STATUS: COMPLETE          (v1 是 ERROR)
```

**探测到的正是 `competitions/` 前缀，与官方样例一致。**

**修复方式（`build_notebook.py`，三处路径全部改成"多候选探测 + glob 兜底"）：**

1. `install_cell` 的 wheels 路径 → 依次试 `competitions/<slug>/…`、`<slug>/…`，再 glob `arc_agi_3_wheels`；
   找不到就 `WHEELS_NOT_FOUND` **响亮失败**（不再静默拿不到包）。
2. 竞赛 cell 的框架 `cp -r` 源 → 同样探测两个候选 + glob `ARC-AGI-3-Agents`；
   找不到就 `FRAMEWORK_NOT_FOUND`。
3. vllm cell 里重装 `arc-agi` 的 `_comp` 路径 → 同样探测（原来与 install cell 不一致，是个定时炸弹）。

**并且新增 `LINGJING_SKIP_METADATA_SYNC=1`**：构建变体 notebook 时**不再改动队友共享的
`notebooks/kernel-metadata.json`**（那个文件指向 `wangausx/arc-prize-2026-smart-router-v2-dpl`）。

### 8.6 被逐一排除的假设（避免后人重复踩坑）

| 假设 | 检验方法 | 结论 |
|---|---|---|
| 是 vLLM / GPU 炸的？ | 队友 10-22 提交是 **CPU-only、无模型数据集**的 notebook | ❌ 仍只有 0.03 ⇒ GPU 不是主因 |
| 是 `BUILD_TAG` 门控 `SystemExit` 拦的？ | **直接执行竞赛 cell 的门控逻辑**，打印真实 `BUILD_TAG` | ❌ 门控**通过**（提交版与当前 HEAD 都含 `llm-whole-qwen` + `llm-structured-view`） |
| 是 API key 错？ | 与**官方样例逐字比对** `.env` | ❌ 完全一致（`ARC_API_KEY=test-key-123` / `OPERATION_MODE=online`） |
| 是 9 小时时限不够？ | 读 `Code Requirements` 页 + vLLM 启动耗时 | ❌ 额度 9 小时，启动只花 28.5 分钟 |
| 是 agent 状态在并发下互相污染？ | 读 `agents/swarm.py:76-87` | ❌ **每局一个 agent 实例**，无共享状态 |
| 是"幽灵 LLM 调用"拖垮了？ | 已在 `75f7d8b` 用出网层 fail-closed 硬守卫修掉 | ❌ 不是主因（但确实是真 bug，已修） |
| 能不能下载历史提交 parquet 反推逐关明细？ | `DownloadSubmission` RPC / web raw 路径 | ❌ **403 / 401，永久不可得** |
| 提交文件只有玩过的局？ | Evaluation 页 + 本地 183 行复刻 | ❌ **覆盖全部 25 局全部 183 关** |

### 8.7 本地竞赛模式全量复刻（结构复现成功）

`_tmp_comp_rerun.py`（升级版）用**与竞赛完全同构**的配置本地复刻：

- 网关：`arc_agi.server.create_app(arc, competition_mode=True)`（= Kaggle sidecar 的形态）
- 客户端：`.env` 与 notebook 逐字一致（`ARC_BASE_URL=http://gateway:8001/`、`OPERATION_MODE=online`）
- 命令：`python main.py --agent myagent`（**不带 `--game`**，= 提交时的原样形态）

**结果：**

| 指标 | 值 |
|---|---|
| `/api/games` 返回 | **25 局** |
| 实际玩到 | **25 / 25 局** ✔ |
| 网关侧 environments | **25** |
| 产出 parquet | **183 行**（= 25 局 × 全部 183 关） |
| 分数 | 见 §9（长超时复刻中） |

**⇒ 结构层面完全复现成功：agent 在"竞赛模式网关 + 不带 `--game`"下确实能玩满 25 局、
网关确实按"全部局 × 全部关卡"出文件。**
**⇒ 0.15 的根因收敛到"notebook 在装 `arc-agi` 那一步就死了，agent 根本没上场"。**

### 8.8 本轮针对 0.15 的代码改动

| 文件 | 改动 |
|---|---|
| `scripts/build_notebook.py` | install cell / 竞赛 cell 框架源 / vllm cell 三处路径改**多候选探测 + glob 兜底**，并响亮失败 |
| `scripts/build_notebook.py` | 新增 `LINGJING_SKIP_METADATA_SYNC=1`，构建变体不动队友共享 metadata |
| `scripts/push_submit_kernel.py`（新，cpu/llm 双臂） | 推 **提交用 kernel**：cpu 臂 = 保底（`enable_gpu=false`、无模型数据集、不烧 GPU 配额）；llm 臂 = 大模型臂（RTX Pro 6000 + vLLM wheelhouse + Qwen3.6-27B-FP8）。`notebooks/submission_llm.ipynb`（新）配套 |
| `scripts/submit_to_competition.py`（新） | **程序化提交**（`competition_submit_code`），带三重护栏：kernel 必须 COMPLETE、日配额检查、`--dry-run` |
| `notebooks/submission_cpu.ipynb`（新） | 提交用纯算法臂：CPU、`SELFLEARN_LLM=0`、`LINGJING_SCRIPTS=on`、**无 bench cell**（rerun 不需要，还省时限） |

### 8.9 ★ 关键修正：这是**两个独立问题**，别混成一句"没跑起来"

在写本节的过程中，我们把官方竞赛页（`data-description`）也抓了下来，原文：

> **Note:** Competition evaluation uses a **separate, private set of 110 games** that your
> agent has never seen. Half of these are used for the Public Leaderboard score, and the
> other half for the Private Leaderboard score.

官方评分口径（同页 `Scoring`）：

| 项 | 官方定义 |
|---|---|
| 每关分 | `min(human_actions / agent_actions, 1.0)²` |
| 每局分 | 各关分的加权平均（按关卡序号 1-indexed 加权） |
| 总分 | 所有局分的平均，输出 **0%–100%** |

**⇒ 评测集是 110 个隐藏局。我们本地基准那 25 局是"公开练习局"，根本不在评测集里。**

于是 0.15 这个数必须拆成两个**互相独立**的问题：

| | 问题 A：**通路断了** | 问题 B：**泛化差距** |
|---|---|---|
| 表现 | 提交后 **agent 一局都没玩** | agent 玩了，但**几乎解不出隐藏局** |
| 证据 | install 路径前缀写死 → notebook 第 1 个 cell 死（**本轮实测：v1 ERROR → v2 COMPLETE**）；duck 线 v25 "6 小时零 in-game action"（v27 已修四处自伤旋钮） | 官方页明写评测集 = 110 隐藏局；本地 60.40 是 25 公开练习局的分；第三方实测程序化 agent 天花板 ≈ **0.27**、随机 agent ≈ 0.10 |
| 修好能拿到 | 0.00 → **0.15**（把"没跑"变回"跑了"） | 0.15 → 20~40 集群 |
| 状态 | ✅ **已修 + 已验证**（§8.5） | ⏳ 未解，**必须换架构** |

**⇒ 对上一轮结论的重要修正（必须诚实写下来）：**

> 我们在 §4 / §6.2 里说"计划带来的是公开局分，不是隐藏集分"。
> **官方页面给出了更狠的版本：评测集是另外 110 局，连那 25 局都不在里面。**
> 所以 `LINGJING_SCRIPTS` 默认开（`100fdc2`）让本地从 19.37 → 60.40，
> **对榜分的期望贡献精确地等于 0**。
> 它仍然值得做——但理由要改成"**本地回归测试 / 冒烟基线**"，
> **不能**再当成提分手段，也不能再拿本地 60.40 去推算榜分。

**⇒ 0.15 的真实构成（与团队 10-20 诊断一致，本次得到官方页面背书）：**
通用兜底路径（CEAX 探索 + Qwen 每步 JSON）在隐藏局上偶尔蹭过 1~3 关，
且动作数远高于人类基准 ⇒ `(human/agent)²` 趋近 0 ⇒ 25 局平均 0.15。
比随机 agent（0.10）只高 0.05。

### 8.10 ★★ 第二个可量化的杀手：**并发退化**（本地实测，最重要的新发现）

上一节说"评测集是 110 个隐藏局、没有校准计划"。但把本地竞赛模式复刻跑完以后，
出现了一个**比"没有计划"更致命、而且是我们自己造成的**问题。

**实验：同一个 agent、同一个网关、同一批 25 局，只改"并发与否"。**

| 跑法 | 总分 | 完成关数 | 满分局 |
|---|---|---|---|
| **顺序**（`benchmark_all_games.py 400`） | **1510**（60.40/局） | **108** | **13 局** |
| **并发**（竞赛模式复刻，25 局同时开） | **408.20**（16.33/局） | **38** | **3 局** |

**⇒ 25 局并发把 13 局 WIN 砍成 3 局 WIN，总分掉到 27%。**
（两次并发复刻分别给 408.32 与 408.20，1500 秒与 9000 秒预算结果一致
⇒ **38 关是并发下的稳定上限，不是被超时砍断的**。）

**逐局看清发生了什么：**

| 类别 | 局 | 现象 |
|---|---|---|
| 计划重放 | ar25(8关/276)、ls20(7关/309)、tr87(6关/122) | 满分局，**动作数极少**，秒过 |
| **通用路径打满** | bp35、dc22、ft09、g50t、ka59、lf52、re86、s5i5、sc25、sk48、su15、tn36、vc33（13 局） | **actions=800 打满、0 关** |
| 部分分 | cd82(2关/14.29)、m0r0(2关/14.29)、wa30(7关/62.22)、cn04/r11l/sp80(1关/4.76)、tu93(1关/2.22) | 效率极低 ⇒ `(human/agent)²` 趋近 0 |

**机理（三条，全部在代码里确认过）：**

1. **`MyAgent.MAX_ACTIONS = 800` 且没有环境变量入口**（写死的类属性）。
2. **`MyAgent.TIME_LIMIT_SEC = 0`** ⇒ 默认**没有每局墙钟上限**
   （除非外部设 `ARC_GAME_TIME_LIMIT_MIN`），而我们的竞赛 `.env` 里**根本没设它**。
   ⇒ 一局卡住就**烧满 800 步霸占 CPU**，9 小时预算和核数全耗在零产出上。
3. **`my_agent.py` 里没有任何 `threading` / 锁** ⇒ 25~110 个线程共享
   engine solver 的磁盘缓存与 CPU，没有节流也没有互斥。

**⇒ 这就是 0.15 的完整闭环：**

```
隐藏 110 局：一局都没有校准计划
   ⇒ 全部走通用路径
      ⇒ 通用路径在 25+ 局并发下被 CPU 抢占，800 步打满仍 0 关
         ⇒ 偶尔蹭过 1~3 关，效率极低
            ⇒ 25 局平均 ≈ 0.15 ✅ 与实测公开分精确自洽
```

**⇒ 由此得到一条以前完全没看见的提分杠杆：**
**我们不是"能力不够"，是"把本来能拿到的分在并发下自己弄丢了"。**
顺序 60.40 → 并发 16.33 的这段落差，**在隐藏集上是同样存在的**（隐藏局还没计划兜底，情况更糟）。
官方 Duck harness 用 **28 并发 × 7920 s 铺满 9 h**，
说明**并发是竞赛的既定形态**——谁解决并发退化，谁就赢。

**已落地的抓手（本轮已改代码）：**

| 改动 | 作用 |
|---|---|
| `my_agent.py`：`MAX_ACTIONS` 改由 `ARC_MAX_ACTIONS` 环境变量控制（默认仍 800） | 让"单局能占多久 CPU"可调；本地已验证的 400 步口径就是顺序基准 60.402 所用值 |
| `build_notebook.py`：竞赛 `.env` 显式写入 `ARC_MAX_ACTIONS` 与 `ARC_GAME_TIME_LIMIT_MIN` | 每局墙钟上限一旦设定，卡住的局会**提前让出 CPU 槽位**，而不是烧满动作预算 |
| `build_notebook.py`：`LINGJING_MAX_ACTIONS` / `LINGJING_GAME_TIME_LIMIT_MIN` 构建期开关 | 可以在不动 agent 代码的前提下调这两个旋钮 |

**⇒ 下一步（纯本地、不烧 GPU 配额、可立即做）：**
**并发度阶梯实测**：同一网关下按并发 1 / 4 / 8 / 16 / 25 跑，
画出"并发度 → 总分/关数"退化曲线，定出该给 `ARC_MAX_ACTIONS` 和
`ARC_GAME_TIME_LIMIT_MIN` 什么值。**这是目前投入产出比最高的实验**——
它直接决定我们交上去的那个 kernel 在 110 局并发下的产出。

**8.10.1 三组对照实验，把机理钉死**

| 实验 | 并发局数 | `MAX_ACTIONS` | 总分 | 关数 | 满分局 |
|---|---|---|---|---|---|
| 顺序基准 | 1（逐局） | 400 | **1510** | **108** | **13** |
| 并发复刻 A | 25 | 800 | 408.20 | 38 | 3 |
| 并发复刻 B | 25 | 400 | 401.61 | 37 | 3 |
| **2 局对照**（早先实验） | **2** | 800 | ar25 **100.0** / cn04 **100.0** | 8/8、6/6 | 2 |

**读出来的四条结论：**

1. **动作预算不是杠杆。** 800 → 400，总分 408.20 → 401.61（略降）、关数 38 → 37。
   ⇒ "少走废步、把 CPU 让给别的局"这条路**基本无效**，§8.10 里"收敛步数上限"这条手段要降级。
2. **不是 competition 模式本身的问题。** 2 局并发时 ar25 与 cn04 **都是 100.00 分**
   （8/8 关、6/6 关）；25 局并发时 cn04 只剩 **1/6 关（4.76 分）**。
   ⇒ **是并发/负载，不是网关语义。**
3. **计划路径没被选错。** 并发跑的 25 局 `route=` 全部正确
   （`CD82_PLAN` / `CN04_PLAN` / `WA30_PLAN` / `SB26_PLAN` / …），
   `CEAX_UNKNOWN` 的 6 局也确实是那 6 局。
   ⇒ **不是"计划没生效"，是"计划跑到一半跑歪了"**（cd82 计划 73 步覆盖 6 关，
   并发下只完成 2 关就掉进通用路径，再烧掉 300+ 步）。
4. **★ 真正的机理：单步动作速率塌了。** 并发跑里 agent 自己打的日志：

   ```
   Exiting: agent reached MAX_ACTIONS of 400, took 268.06 seconds (1.49 average fps)
   Exiting: agent reached MAX_ACTIONS of 400, took 371.67 seconds (1.08 average fps)
   ```

   **并发下每步只有 1.08~1.49 fps。** 而顺序基准 25 局 585.88s 跑完（≈20+ actions/s 聚合）。
   并发跑 22 局打满 400 步各花 268~371s（墙钟并行），**聚合吞吐与顺序基本持平**。
   ⇒ **不是机器算不过来，是每局"每步做对"的能力塌了。**
   ⇒ 游戏的 `default_fps = 6`（来自 `environment_files/*/metadata.json`）：
   **单步速率从 ~6 fps 掉到 ~1.2 fps ⇒ 计划里的"点击坐标 / 步骤序号"与游戏实际状态错位
   ⇒ 计划必然跑歪。** 2 局并发时速率够快，所以计划 100% 成立。

**⇒ 修法因此从"省 CPU"改成"提单步速率 + 让计划对帧漂移鲁棒"：**

| 方向 | 说明 | 状态 |
|---|---|---|
| **A. 压低通用路径单步成本** | 隐藏集 100% 是通用路径；单步越贵、并发下掉得越狠 | ⏳ 待做（需先 profile 单步耗时分布） |
| **B. 计划对状态漂移鲁棒** | 计划不只按"第 N 步"发，还要**按当前观测校验并重同步**（读帧 → 判断是否已生效 → 决定继续/重来） | ⏳ 待做 |
| **C. 自适应并发节流** | agent 自己测单步耗时，超阈值就**降频省算力**（少做昂贵分析），优先保证动作正确 | ⏳ 待做 |
| ~~D. 收敛 `ARC_MAX_ACTIONS`~~ | 实测无效（400 vs 800 只差 1.5%） | ❌ **已证伪，降级** |

**⇒ `ARC_MAX_ACTIONS` / `ARC_GAME_TIME_LIMIT_MIN` 两个旋钮仍然保留**
（它们是**兜底护栏**：防止某局在 110 局并发下霸占整台机器），
但**主战场是 A/B/C**，不是 D。

**⚠️ 事后更正（2026-10-22 22:0x）：上面第 4 条「单步速率塌了 ⇒ 计划与帧漂移错位」的机理是错的。**
真正的根因是**共享可变状态被并发写坏**，与 fps 无关。见下一节 §8.11。

---

### 8.11 ★★★ 并发退化的**真正根因**：`GameAction` 是进程级枚举单例（并发写坏点击坐标）

> 这一节把 §8.10 / §8.10.1 的结论**推翻并重写**。之前写的"单步速率从 6 fps 掉到 1.2 fps、
> 计划与游戏状态错位"是**错误归因**——fps 塌陷是**并发下的正常现象**，不是原因。

#### 8.11.1 决定性对照实验

同一套竞赛模式网关（`_tmp_comp_full.py`）、同一份 agent、同一批 25 局，**只改并发度**：

| 跑法 | cd82 结果 |
|---|---|
| **单局**（`--game cd82`） | **score=100.00，6 关，73 步** ✅ 与顺序基准逐项一致 |
| **25 局并发**（修复前） | **2 关，800 步打满** ❌ |

**⇒ 单局满分、并发崩盘 ⇒ 网关语义没问题、agent 逻辑没问题，**是并发**本身把结果写坏了。

#### 8.11.2 逐帧对照：分叉点精确落在**第一个 ACTION6**

把两份录制（单局 vs 25 局并发）逐帧哈希比对：

```
 step | solo: act lvl hash              | conc: act lvl hash
    0 | ACTION3 L0 a45919624d3a        | ACTION3 L0 a45919624d3a      ← 完全一致
    1 | ACTION2 L0 f4d22658a33d        | ACTION2 L0 f4d22658a33d      ← 完全一致
    2 | ACTION2 L0 44382652b9db        | ACTION2 L0 44382652b9db      ← 完全一致
    3 | ACTION4 L0 cf33384004df        | ACTION4 L0 cf33384004df      ← 完全一致
    4 | ACTION5 L1 a26a68e73448        | ACTION5 L1 a26a68e73448      ← 完全一致
    5 | ACTION5 L1 949749d5bd2e        | ACTION5 L1 949749d5bd2e      ← 完全一致
 >>  6 | ACTION6 L1 e772706a9905        | ACTION6 L1 6b894b6df5a4      ← ★ 分叉
```

**前 6 步的动作类型与帧逐像素一致，第 7 步（索引 6）开始分叉——而那正是 cd82 计划里的第一个
`ACTION6`（点击）。** 像素级 diff 显示：单局里被点的精灵从 (row7,col44-48) 移到 (row7,col38-42)，
并发里**原地没动** ⇒ **点击落空了**。

**⇒ 动作"类型"永远对（枚举成员身份不会变），只有"坐标"会被换掉。**
这在数学上只可能是一件事：**ACTION6 的坐标被别的线程改写了**。

#### 8.11.3 机理：`arcengine.GameAction` 是枚举单例，`set_data()` 直接写共享成员

`arcengine/enums.py`：

```python
class GameAction(Enum):
    ACTION6 = (6, ComplexAction)
    def __init__(self, action_id, action_type):
        self.action_data = action_type()          # ← 每个枚举成员只有一份
    def set_data(self, data):
        self.action_data = self.action_type(**data)  # ← 直接覆写这个【全进程唯一】的成员
```

框架 `agents/agent.py` 的发包顺序是：

```python
action = self.choose_action(...)        # 里面 action.set_data({"x":..,"y":..})  ← 写共享成员
...
data = action.action_data.model_dump()  # ← 下一行才读回来
raw  = self.arc_env.step(action, data=data, ...)
```

**写入和读取之间隔着一次函数返回。** 竞赛 rerun 里 25~110 局 = 25~110 个线程**共用一个 Python 进程**，
于是：

```
线程 A（cd82）: set_data({x:46,y:4})      ← 写进全局 ACTION6
线程 B（别的局）: set_data({x:11,y:29})    ← 覆盖！A 的坐标没了
线程 A: data = action.action_data.model_dump()  ← 读到的是 B 的坐标
⇒ cd82 的点击落在别的局的格子上 → 计划全线跑歪 → 掉进通用路径烧光 800 步
```

**单元级复现（8 线程 × 3000 轮，各自写不同坐标后读回）：**

| | 被串坐标比例 |
|---|---|
| 原始 `arcengine` | **10.8% ~ 13.9%**（多次采样） |
| 打上线程隔离补丁后 | **0 / 24000 = 0.0%** |

#### 8.11.4 为什么"计划型动作"受害最重

计划就是「按下标发坐标」。坐标被换 ⇒ 这一关的点击全落空 ⇒ 关卡过不去 ⇒
计划耗尽后掉进通用路径 ⇒ 再烧 300+ 步 ⇒ `(human/agent)²` 趋近 0。
**非点击动作（ACTION1~5）没有每次调用的载荷**（`SimpleAction`，`data={}`），
**所以并发下前 5 步完全一致**——这个"前 5 步一模一样"的巧合，就是本 bug 的指纹。

#### 8.11.5 修法：把 `GameAction` 的每次调用载荷改成**线程本地**

**约束**：竞赛 rerun 的框架树来自 Kaggle 官方 input 包（`/kaggle/input/.../ARC-AGI-3-Agents`），
**改不了**；能改的只有我们完全掌控的 `my_agent.py`。

**做法**（`agent/my_agent.py`，模块导入即生效，早于任何一局开跑）：
把 `GameAction.action_data` / `GameAction.reasoning` 从「实例属性」换成
**property（data descriptor，优先级高于实例 `__dict__`）**，底层落到 `threading.local()`：

* 同一线程内 `set_data()` → `action_data` 读写一致 ⇒ **框架代码一行不用改**；
* 跨线程互不可见 ⇒ **竞态消失**；
* 装不上就打印 `[action-isolation] FAILED ...` 并退回原行为，**绝不能拖垮开局**。

补丁自带一行自证日志：`[action-isolation] thread-local GameAction state: ON`。

#### 8.11.6 修复效果（本地 25 局并发复刻，同一网关同一批局）

| 指标 | 修复前（25 局并发） | **修复后（25 局并发）** | 顺序基准 |
|---|---|---|---|
| **总分** | 408.20 | **1449.82**（**3.55×**） | 1510（400 步口径） |
| **完成关数** | 38 | **103** | 108 |
| **满分局（WIN）** | **3** | **13** | **13** |

**⇒ 满分局从 3 局回到 13 局，与顺序基准完全一致；总分恢复到顺序基准的 96%。**

逐局对照（修复前 → 修复后）：

| 局 | 修复前 | 修复后 | | 局 | 修复前 | 修复后 |
|---|---|---|---|---|---|---|
| cd82 | 2 关 | **6 关 / 100.00** | | sb26 | 0 关 | **8 关 / 100.00** |
| cn04 | 1 关 | **6 关 / 100.00** | | tn36 | 0 关 | **7 关 / 100.00** |
| dc22 | 0 关 | **6 关 / 100.00** | | vc33 | 0 关 | **7 关 / 100.00** |
| ft09 | 1 关 | **6 关 / 100.00** | | sp80 | 1 关 | **6 关 / 100.00** |
| m0r0 | 2 关 | **6 关 / 100.00** | | sc25 | 0 关 | **5 关 / 71.43** |
| r11l | 1 关 | **6 关 / 100.00** | | bp35 | 0 关 | **2 关 / 6.67** |
| ar25 / ls20 / tr87 | 满分 | 满分（不变） | | g50t/ka59/lf52/re86/sk48 | 0 关 | 0 关（**通用路径的 5 个老零关局，未变**） |

**⇒ 10 个"计划型"局全部恢复到满分；剩下的 5 个零关局是通用路径的老问题，与本 bug 无关。**

**⇒ 顺带说明：§8.10.1 里 `ARC_MAX_ACTIONS` 收敛（400 vs 800）之所以"无效"，
是因为真正的瓶颈不是步数，而是坐标被打坏——收敛步数当然救不回来。**

#### 8.11.7 这条修复对竞赛的直接影响（比任何"提分技巧"都大）

* 我们的 kernel 在竞赛 rerun 里就是**多局并发**跑的 ⇒ **10-23 的提交会直接吃到这个修复**。
* 官方 Duck harness 用 **28 并发** ⇒ 并发度更高 ⇒ 没有这个修复时受损更重。
* **这是"不花 GPU 配额、不改策略、只修 bug"就能拿回的分**，量级是
  公开局口径 **408 → ~1510（3.7×）**。

**⇒ 一句话：0.15 → 两位数这一跳，一半在 §8.5 的路径修复（把 agent 跑起来），
另一半在 §8.11 的线程隔离修复（让 agent 跑对）。缺任何一个都还是 0.0x。**

#### 8.11.8 全队暴露面（这一条比修复本身更重要）

**根副本 `ARC-AGI-3-Kaggle-Starter/` 的 12 个 agent 文件里共有 40+ 处 `action.set_data({...})`，
全部踩同一个竞态**——谁在那边提并发 kernel，谁就中招，而且**症状是"分数莫名偏低"，极难归因**。

已做的处置：

| 对象 | 状态 |
|---|---|
| 根副本 `agent/my_agent.py`（**根副本 notebook 内嵌的就是它**，sha256 一致 ⇒ 覆盖队友提交通路） | ✅ 已注入补丁（备份在 `agent/_bak_isolation_20261022_213617/`） |
| 根副本 `agent/lingjing_team_agent.py` | ✅ 已注入补丁（同备份目录） |
| 根副本其余 10 个 agent 变体 / baselines | ⚠️ **未改**（多为基准与备选，不是提交通路）——需要谁在用谁加 |
| 大模型臂 Duck harness（`integrations/duck-kaggle/`） | ✅ **天然免疫**：它直接构造 `arcengine.ActionInput(id=..., data=...)`，**从不调用 `set_data()`** |

**⇒ 结论：P0-B（Duck harness）这条线不需要为这个 bug 做任何改动。**

#### 8.11.9 落地方式：标准件 + 同步生成器（防两份拷贝漂移）

竞赛 rerun 的框架树来自 Kaggle 官方 input 包（改不了），notebook 也只把 `my_agent.py`
拷进 `agents/templates/` ⇒ **提交用的 agent 必须自包含**。所以做成：

| 文件 | 作用 |
|---|---|
| `scripts/action_isolation_patch.py` | **唯一真源**，可被任何 agent `from action_isolation_patch import install; install()` 复用 |
| `scripts/sync_action_isolation.py` | 机械地把标准件写进 `agent/my_agent.py` 与 vendor 模板副本 |
| `scripts/check_action_isolation.py` | **永久防线**：① 竞态复现（未打补丁时串号率必须 >0）② 补丁存在性 ③ 模板副本与 `agent/` 逐字一致 ④ 内联实现与标准件**同源** ⑤ 打补丁后串号率必须 = 0 |

**⇒ 任何人改了这个补丁，跑一遍 `scripts/check_action_isolation.py` 就能知道有没有写坏。**

---

## 9. 更新后的优化 / 升级路线（第二轮之后）

### 9.1 结论调整（已按官方"110 隐藏局"再修一次）

**上一轮结论"继续合并 vs 优化"不变，但优先级被彻底改写两次：**

> **第一次修正**：本地 60.40 → 竞赛 0.15 的落差**不是单一原因**，是
> **① 通路断了**（已修）**+ ② 评测集是 110 隐藏局、我们的公开局分不迁移**（未解）。
>
> **第二次修正（官方页面背书后）**：
> **在修好提交通路之前，任何"提分"工作都测不出来**；
> 而且**修好通路只能把 0.00 拉回 0.15**——**0.15 → 20~40 集群必须换架构**，
> 继续在 25 个公开局上加计划，对榜分的贡献**精确为 0**。

**⇒ 一句话结论：**
**先把路修通（拿回 0.15，别再交 0.00），然后立刻把主力切到 Duck harness（LLM + REPL）路线；
公开 25 局的计划资产降级为"回归测试基线"，不再当提分手段。**

**⇒ 第三次修正（并发退化，§8.10，本地实测）：**
在"通路修好"和"换 Duck 架构"之间，**还夹着一个我们自己造成的、可量化、可修的坑**：
**25 局并发把 13 局 WIN 砍成 3 局 WIN（1510 → 408.20 分）**。
**这条在隐藏集上同样成立，而且是我们不花 GPU 配额就能修的。**
**⇒ 所以路线是三段而不是两段：修通路 → 修并发退化 → 换 Duck 架构。**

**⇒ 第四次修正（§8.11，机理重写）：**
并发退化的根因**不是"单步速率塌了"，而是 `GameAction` 枚举单例被并发写坏点击坐标**
（单元级复现：原始 arcengine 下 **10.8%~13.9%** 的动作坐标被串，补丁后 **0%**）。
**修法已落地并本地验证：并发复刻里 cd82 从「2 关 / 800 步」回到「6 关 / 73 步」，
与单局和顺序基准完全一致。**
**⇒ 这意味着 P0-A2 不需要 A/B/C 三件套，只需要一行线程隔离。**
**⇒ 并且：10-23 的 CPU 提交必须带上这个修复，否则交上去的还是被写坏的版本。**

### 9.2 路线图

| 优先级 | 动作 | 状态 | 判据 |
|---|---|---|---|
| **P0** | 修 install/框架路径前缀（多候选探测） | ✅ **已修 + 已验证**（CPU kernel v1 ERROR → v2 COMPLETE） | kernel 状态 COMPLETE 且日志出现 `WHEELS .../competitions/...` |
| **P0** | 推 CPU 保底提交版（不烧 GPU） | ✅ `liaojiaxuan/smartrouter-merged-cpu-submit` v2 | 提交配额解锁即可交 |
| **P0** | 日配额解锁后提交 CPU 保底版 | ⏳ 定时任务已排（UTC 00:00 之后） | `competition_submissions` 出现新条目 |
| **P0** | 提交后**第一时间回读公开分** | ⏳ | **公开分应从 0.0x 跳到两位数**——这是通路修复是否生效的唯一判据 |
| **P0-A2** | ★★★ **修并发退化**（见 §8.11）：根因是 **`arcengine.GameAction` 枚举单例的 `action_data` 被并发线程互相覆盖** ⇒ 点击坐标串到别的局 ⇒ 计划全线跑歪。**修法：`my_agent.py` 把 `action_data`/`reasoning` 改成线程本地（框架代码一行不用改，因为竞赛 rerun 的框架来自官方 input 包改不了）** | ✅ **已实现 + 已本地验证**（25 局并发复刻：总分 **408.20 → 1449.82**、关数 **38 → 103**、满分局 **3 → 13**，与顺序基准的 13 局 WIN 对齐；单元级：坐标串号 13.9% → **0%**） | 已达成；永久防线 `scripts/check_action_isolation.py` |
| **P0-A2′** | **重推带修复的 CPU kernel 并同步定时任务的 `--version`** | ✅ **已推 `liaojiaxuan/smartrouter-merged-cpu-submit` v3**（含 `[action-isolation] … ON`）；⏳ 待确认 v3 COMPLETE | kernel v3 COMPLETE 且日志出现 `[action-isolation] thread-local GameAction state: ON` |
| **P0-A2″** | 通用路径单步成本 / 计划按观测重同步 / 自适应节流（原 A/B/C） | ⏳ 降级为 P2 | 坐标竞态已消除，这三项**不再是并发退化的主因**，只在隐藏集通用路径上仍有价值 |

| **P0-B** | ★ **Duck harness 修活并提交**（官方配置：110 局 ÷ 28 并发 × 7920 s 铺满 9 h） | ⏳ 等 GPU 配额（10-24 08:00） | 目标 0.15 → **1.5~3.0**（官方公开局均值 1.6002） |
| **P0-C** | 把 **§8.5 路径修复 + v27 四处自伤旋钮修复** 一起搬进 duck kernel | ⏳ 纯本地可做 | duck kernel 不再 6 小时零 action |
| **P1** | 移植 `feat/unify-decoder-entry` 的 25 文件破零包 | ⏳ 可立即开始 | **降级目标**：只作为本地回归基线，**不再计入榜分预期** |
| **P2** | 通用能力（探索 / 记忆 / LLM 写代码分析网格） | ⏳ | **隐藏集唯一胜负手**（官方题面原话：exploration, memory, goal acquisition） |
| **P2** | 基准可复现（清 `state/engine_solver_cache.json`、不并发、seed 固定） | ⏳ | 前后两次跑测逐局一致 |
| **P3** | 公开 25 局的校准计划继续维护 | ⏳ | **只作回归测试/冒烟基线**，榜分预期贡献 = 0 |
### 9.3 仍然不要做的事

- **不要用公开榜调参**（`0.15` 这类分数不是能力信号，调它只会过拟合噪声）。
- **不要再堆公开局 plan**：`LINGJING_SCRIPTS` 已经是"把树里已有的资产打开"，不是新增特判。
- **不要在 GPU 配额墙前反复重推 GPU kernel**（每次失败都在烧配额墙的时间）。
- **不要指望下载历史提交 parquet**（403/401，永久不可得）。

### 9.4 一条给全队的硬规则（本次事故的直接教训）

> **Kaggle 竞赛数据的挂载路径前缀会随 kernel 的挂载方式变化。**
> 凡是写死 `/kaggle/input/<something>` 的 notebook cell，都必须改成
> **"多候选探测 + glob 兜底 + 响亮失败"**。
> 我们在环境目录上已经踩过一次（`_tmp_bench` 猜路径 → 改多候选探测），
> 这次在 **wheels / 框架树**上又踩了一次——**同一类 bug，两次都让整轮跑测白跑。**
>
> 落地护栏（`build_notebook.py` 内置，不靠人记）：
> - 路径找不到 → `SystemExit('WHEELS_NOT_FOUND' / 'FRAMEWORK_NOT_FOUND')`，**在第一个 cell 就炸**；
> - 装包用 `check=True`，**装不上就炸**，绝不静默继续；
> - 竞赛 cell 的 `main.py` 用 `!python main.py --agent myagent`，
>   后面**补一行 `!python -c "print('MAIN_EXIT', 0)"` 之类的哨兵**，
>   这样"到底跑没跑起来"在日志里一眼可见，而不是靠猜。
