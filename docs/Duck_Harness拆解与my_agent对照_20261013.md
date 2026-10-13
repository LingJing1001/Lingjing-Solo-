# Duck Harness 拆解 与 `my_agent.py` 对照

> 日期：2026-10-13 · 用途：把冠军方案(LM+REPL)的真实骨架摊开，逐项对照本仓库主链，
> 判断哪些能直接借、哪些借不了。
> 来源：`github.com/Tufalabs/duck-harness`（MIT）· `tufalabs.ai/research/duck-harness/` ·
> Kaggle discussion `717133`（M1 技术报告）· `744792`（M2 前三 serving 对比）。

---

## 0. 一句话结论

冠军方案和本仓库**不是同一件事的两种做法，是两个不同的物种**：

- 冠军 = **LLM 在 REPL 里把游戏当编程题解**，harness 只负责"把状态喂好、把上下文管住、把动作执行掉"；
- 本仓库 = **一条手写的六级决策链**，靠预烘答案表 + 引擎源码求解 + CEAX 启发式。

两者唯一的交集是"输入都是 64×64 帧"，但从帧到动作之间发生的事完全不同。
本仓库的闭环里**没有任何模型**。

---

## 1. 血缘与版本

| 阶段 | 方案 | 模型 | 分数 |
|---|---|---|---|
| Preview | StochasticGoose（Dries Smit） | CNN+RL，只搜"改变帧的动作" | 预览赛第一；**主赛加动作效率计分后崩掉** |
| M1（06-30 截止） | **The Duck**（Tufa Labs） | Qwen3.6-27B-FP8 | **1.21%**（Kaggle LB）/ 1.6002±0.4475（25 局 ×20 次公开集） |
| M2（09-30） | Duck 血统三家 | Qwen3.8-Flash-Next | **27.9% / 23.8% / 22.5%** |
| 当前 | `yukailiu/arc-agi-3-champion-solution` | 同上 | **28.91** |

**关键读数：半年内 1.21% → 28.91%，底层模型基本没换，涨的全是 harness 工程。**
OpenAI 7/29 的受控实验是同一结论的最强证据：GPT-5.6 Sol 只改两个 harness 设置
（保留推理状态 + 用压缩替代截断），7.8% → 38.3%，token 还少 6×。

> 对照：本仓库同一竞赛 publicScore = **0.04**，`enable_gpu=false`，主链 96.17% 的分来自硬编码表。

---

## 2. The Duck 的骨架（基于源码，非二手转述）

### 2.1 仓库分层

```
duck-harness/
├── ARC3-Inference/        # The Duck 本体
│   └── inference/
│       ├── framework/solver.py        # HarnessSolver：TAAF 适配、动作执行、事件、转录
│       ├── agent/tool_agent.py        # OpenAI 兼容的工具调用循环
│       ├── agent/python_tool_sandbox.py  # 唯一的 python 工具（隔离运行时）
│       ├── agent/prompts.py           # ★ 全部提示词
│       ├── agent/runtime_state.py     # 注入沙箱的全局变量
│       ├── agent/vision_context.py    # 每轮附的网格图
│       ├── agent/action_names.py      # UP/DOWN/… ↔ ACTIONn 映射
│       └── utils/segmentation.py      # ★ 连通域分割（可原样拷进沙箱）
├── tufa-arc-agi-framework/  # TAAF：Benchmark / GameAPI 执行框架
├── example-run/             # 一次完整跑分（25 局 ×20 次）
└── taaf-duck-harness-kaggle-share.ipynb
```

### 2.2 一个回合发生什么

`solver.py` 起一个 `HarnessSolver`，把**最新帧、合法动作、历史、一个 python 工具**交给模型。
模型在 Python 里看板子、写代码、通过 `action(...)` 真正落动作；
引擎执行后**刷新沙箱变量**；模型继续。循环直到过关或超时。

### 2.3 动作空间（官方命名）

模型看到的是**具名动作**，不是 `ACTION1..7`：

- `UP` `DOWN` `LEFT` `RIGHT` `SPACE`
- `MOUSE(row=..., col=...)` —— **用 `row`/`col`，`x`/`y` 会被拒**

**⚠️ 注意：官方 `GameAction` 是 8 个（`RESET=0` + `ACTION1..7`，其中 `ACTION6=ComplexAction` 是唯一带 payload 的点击、`ACTION7=Undo`），
而 Duck 只暴露了 6 个 —— 它没有把 `Undo` 给模型。**
这是有意的设计取舍（不后悔、直接重规划），值得注意。

### 2.4 唯一的工具：`python`

从 `prompts.py` 原文：

> `- The only tool is python; call it with one ephemeral code string.`
> `- Every python tool call starts fresh. Re-import modules or re-define any custom utility logic you need.`
> `- The only importable standard-library modules are: bisect, collections, copy, fractions, functools, heapq, itertools, json, math, operator, random, re, statistics, string.`
> `- Each python tool call has a hard time limit of 30 seconds.`
> `- Tool responses are capped to about {tool_output_tokens} tokens.`

即：**无状态、白名单、30 秒、输出截断**。模型每个回合都要重写自己需要的工具函数。

### 2.5 沙箱里有什么（`runtime_state.py` + `prompts.py`）

| 变量 | 内容 |
|---|---|
| `current_frame` | `.ascii`（字母色码网格串）· `.step` · `.level` · `.shape` · `.segmentation` |
| `history` | 动作/帧快照列表；`history[i].frame` 是**该动作之后**的帧 |
| `previous_frame` | 最近一次真实动作**之前**的帧 |
| `transitions` / `last_transition` | 每个转移带 `.action/.before_frame/.after_frame/.result` |
| `last_action_result` | `board_changed` / `done` / `level_completed` / `game_over` / `run_complete` / `reward` / `valid_actions` |
| `valid_actions` | 当前合法动作名 |
| `action(actions)` | 执行 1..n 个动作，可循环里调多次 |

**最关键的一条设计选择（原文）：**

> `The raw numeric grid is intentionally not exposed. Use current_frame.segmentation as your primary view of the board`
> `The raw numeric grid is intentionally hidden from the Python tool. The preferred view is current_frame.segmentation; current_frame.ascii is there for small local checks.`

**即：Duck 不是"把像素丢给模型"，而是把像素先压成一套结构化对象再给模型。**
这和我们仓库的做法在方向上一致（都不直接用原始网格），但我们的压缩产物是
**16 个标量**（`aop_encoder.py`），而 Duck 的产物是**每个物体的 id/color/hash/pixels/boundary/children + 邻接表**。

### 2.6 分割器的字段（`utils/segmentation.py`，stdlib-only）

4-连通 flood fill → 每个同色连通块一个 node：

- `id`：按最上最左排序（64×64 内唯一）
- `color`：ARC 色码字符
- `hash`：**平移不变签名**（颜色 + 归一化后的形状）—— 同一形状在任意位置 hash 相同，
  用来**跨帧追踪同一物体**、或在同一帧里发现多个相同物体
- `pixels`：格数
- `boundary`：Moore 邻域顺时针外轮廓，**只保留拐点**
- `children`：被本物体**完整包住**的直接子物体（最内层包络者 = 父）
- `adjacency_list`：共享 4-邻边的 id 对

> 注意 `hash` 这个设计：它把"这个物体是什么形状"与"它在哪"解耦。
> 这正是我们 `data/aop` 里丢掉的东西（我们连帧都丢了）。

### 2.7 提示词工程（`prompts.py` 的 7 个块）

| 块名 | 干什么 |
|---|---|
| `TOOL_CALL_FORMAT_GUIDANCE` | 掐死工具调用格式，不许夹 markdown / 散文 |
| `GAME_OVERVIEW_ADDENDUM` | 定义"每回合 = 观察-规划-行动"；**目标是不只是过关卡，是通关整局** |
| `VISUAL_GAME_ADDENDUM` | 视觉先验 + **明确的失败模式警告**（见下） |
| `STRUCTURED_RUNTIME_STATE_ADDENDUM` | 逐个变量解释语义（含"`history[-1].frame` 就是当前帧，别自己跟自己比"这类坑） |
| `MULTIMODAL_CONTEXT_ADDENDUM` | 每轮附当前网格图；图与 `.ascii` 是同一帧的两种表示 |
| `PYTHON_ADDENDUM` | 方法学：优先 `segmentation`、鼓励 BFS/DFS/**beam search**、**禁止打印整帧** |
| `COMPACT_TOOL_SESSION_ADDENDUM` | 别省工具调用；每次 30 秒；输出会被截断 |

**两句最值钱的原文：**

> `A common failure mode is to mistake a segmented edge bar for clickable puzzle pieces. …
> If a repeated strip of small blocks sits flush against the border and actions only change that strip
> while the interior board stays the same, classify it as HUD/timer state … DON'T DO THIS!`

—— 这条反模式警告，**正是我们 `_eng_60` 类启发式在踩的坑**：本地 `_perc_click_candidates`
（物体中心 → 变化像素 → 稀有色采样）分不清"HUD 计时条"和"可点击物体"。
Duck 是把这条**写进提示词**让模型自己判断；我们是**写死在启发式里**，写错了就一直错。

> `IMPORTANT: Especially when the game is about making an agent navigate to a target, it is usually safer to
> write an explicit search algorithm such as BFS. … pathfinding, flood fill, BFS, DFS, beam search,
> shortest-path search … are all valid.`

—— 它把搜索**交给模型现场写代码**，而不是预置一个 `r3_kernel.py`。
换句话说：**我们仓库里那套 R3/BFS/IDDFS，在 Duck 里是"模型的一次输出"，不是一段固定代码。**

---

## 3. 逐项对照

| 维度 | Duck harness（28.91） | 本仓库 `my_agent.py`（0.04） |
|---|---|---|
| **决策者** | Qwen3.6/3.8，GPU 本地 vLLM | **无模型**（`enable_gpu=false`） |
| **控制流** | 模型写 Python → `action()` | 六级优先级链，命中即 return |
| **工具面** | 唯一 `python`，无状态，30s，import 白名单 | 无工具；内部函数调用 |
| **感知输入** | `segmentation`（对象图）+ `ascii` + 网格图；**原始数值网格刻意隐藏** | `click_heatmap` 热图 + `_perc_click_candidates`；AOP 16 标量（像素已剥） |
| **物体模型** | 每物体 `hash`（平移不变）+ `children` 包含树 + 邻接表 | 无持久物体标识；逐帧重算 |
| **跨帧推理** | `transitions` / `previous_frame` / `last_action_result.board_changed` | `stall` 计数 + `_frame_delta`；无显式转移表 |
| **搜索** | 模型现场写 BFS/beam search | 预置 `r3_kernel.py` + 26 个 `arc_shadow_*.py` + IDDFS（首次 600s / 二次 500s） |
| **上下文** | 驱逐最旧消息，无限对局；token 计量截断 | 无上下文概念 |
| **并发** | 16 局（Kaggle）/ 32 局（Slurm 2×B200） | 单局串行 |
| **时间预算** | 45~75 分钟/局，外加上限 | 400 步/局（另有 500s IDDFS 墙钟） |
| **分数来源** | harness 工程（并发/上下文/感知管线） | 硬编码表 96.17% + R3 2.44% + 闭卷 1.39% |
| **动作集** | 6 个具名（**无 Undo**） | `ACTION1..7`（含 Undo） |
| **本地可测性** | `make interactive GAME=...` 直接跑真引擎 | `make play-local`，但**跑不出云端口径** |

---

## 4. 我们能直接借的 5 条（按性价比排序）

**① 平移不变物体 hash（零依赖，可直接移植）**
`segmentation._object_hash` 只有 10 行：颜色 + 归一化形状 → sha1 前 16 位。
它能让我们把"这一帧的物体"和"上一帧的物体"对上，这是**现在完全缺失的能力**
（`_frame_delta` 只做像素差分，不认对象）。这一段是 stdlib-only，可以原样拷。

**② 包含树 `children` + 邻接表**
判别"这个物体是不是被那个物体包住"、"谁挨着谁"——正是解**嵌套/容器类谜题**
（bp35、sk48 这类）需要的结构，比我们现在按稀有颜色采样点击要靠谱。
同样 stdlib-only。

**③ 把"改了没变"这个信号提到一等公民**
Duck 的 `last_action_result` 里有 `board_changed`，且在提示词里反复强调
"先确认这步是不是只动了 HUD"。我们的 `stall` 计数是**粗糙的等价物**，
但缺"变更区域是 HUD 还是玩法区"的区分。这条纯逻辑，不需要模型。

**④ 提示词里那两条反模式，可以直接写成断言**
- HUD 边缘条 ≠ 可点击物体
- 对 `MOUSE`，`row`/`col` 顺序别搞反
这两条我们踩过（`_perc_click_candidates` 会往边缘条上采样）。
即使不上 LLM，也可以作为**启发式的硬护栏**，成本极低。

**⑤ 上下文驱逐 / 并发调度 —— 借不了，除非先改范式**
这两条只有在"有 LLM"时才有意义。它们占了冠军分差的绝大部分，
但**前提是先接受 LLM+REPL 这个范式**。

---

## 5. 借不了的硬约束（必须先认清）

> **17:10 事实核查修正**（依据官方 rules 存档 `state/kaggle_rules_pages.txt`）：
> 下面第 1/3 条的表述**把断网理解反了**。断网封死的是"调外部 API"，**不是 LLM 本身**；
> 官方恰恰为本地 LLM 铺了路 —— 断网 + 专属 RTX 6000 是**配套设计**，不是障碍。

1. **需要 GPU 与权重**：Qwen3.6-27B-FP8 / Qwen3.8-Flash-Next，单卡 96GB（RTX Pro 6000 / B200）。
   **这台机器官方免费给**：Code Requirements 允许 `GPU Notebook <= 9 hours run-time`；
   Upgraded accelerators 页原文「we've added **RTX 6000** machines to the ARC-AGI-3 hardware pool …
   Kaggle uses machine type `g4-standard-48`」「RTX are only available for notebooks attached to
   **this competition**」。⇒ 与 Duck 用的同款 96GB 卡，**算力成本 ≈ 0**。
   我们现在的 notebook 只是**还没打开**：`enable_gpu=false`、`dataset_sources=[]`、`model_sources=[]`。
2. **需要 vLLM / SGLang 服务栈**：含 FP8 KV cache、跨轮 prefix cache、Mamba 状态保留补丁。
   这是冠军分差里最"工程"的一块，且各家实现不同（M2 前三就是在这上面分高下）。
   **断网下额外多一道**：服务栈不能现装，必须预装或把 wheel 与依赖一并打进 dataset，
   用 `pip install --no-index --find-links=/kaggle/input/<wheel-ds>`。
3. **纯 API 路线在赛时不可行**：Code Requirements 原文 `Internet access disabled`，
   RTX 页亦写「all RTX sessions must have internet disabled」。Duck 支持的 OpenRouter 只能在
   **有网的开发机**上做离线验证，**不能进提交**。（论文提的"数万美元 API 成本"是开发期口径。）
   另据 General Rules §5.a：获奖提交必须 **open source system + open source model + open source
   weights**（OSI Open Source AI 定义）⇒ **闭源 API 路线连获奖资格都没有**。
   「本地开源权重」不是退而求其次，**是规则强制的唯一方向**。
4. **本地公开集分不能预测云端**：Tufa 自己的 M1 提交就有 0.77%~1.30% 的抖动；
   数据集方也明写 *"Offline levels do NOT predict the leaderboard"*。
   ⇒ **任何验证都必须在真引擎 + 多次重复上做**，单次跑分没有说服力。

---

## 6. 给本仓库的三个可选动作

| 选项 | 做什么 | 代价 | 能拿到的确定性 |
|---|---|---|---|
| **A（低风险）** | 移植 `_object_hash` + `children` + `adjacency_list` 到 `lingjing_solo/perception/`，替换/补充 `_perc_click_candidates` 的点击提案 | 纯 stdlib，1 天内可验证 | 中；能直接改善点击类关卡的命中率 |
| **B（中风险）** | 给 `my_agent.py` 加一个 `board_changed` / `HUD-vs-playfield` 判别，收紧 stall 判定与点击护栏 | 纯逻辑，无新依赖 | 中；减少"烧预算在计时条上"的浪费 |
| **C（高风险/高回报）** | 承认范式差异，起一条 **LLM+REPL 旁路**；赛时形态是 **C-Local**：权重作为 Kaggle Dataset/Model 挂载，在 notebook 内的 RTX 6000 上跑本地 serving，**全程不需要网络** | 算力≈0（官方白给 96GB RTX / 9h / 每日一次）；真实代价是**工程**：权重分发 + serving 栈离线安装 + 采样可复现 + 9h 吞吐 | 高，但工期是量级差异 |

**建议顺序：A → B →（用 A/B 的真实增益决定）C。**

---

## 附：参考链接

- 源码：https://github.com/Tufalabs/duck-harness
- 博客：https://tufalabs.ai/research/duck-harness/
- M1 技术报告：https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/717133
- M2 前三 serving 对比：https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/744792
- 冠军 notebook：https://www.kaggle.com/code/yukailiu/arc-agi-3-champion-solution
- 官方论文（本地副本）：`state/_paper_2603_24621.txt`
