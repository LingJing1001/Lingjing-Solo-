# `datawhalechina/llm-algo-leetcode` 评测与对本项目的作用

> 记录时间：2026-10-13 17:55　|　仓库：`github.com/datawhalechina/llm-algo-leetcode`
> 元信息（抓取时）：682 stars · 最后推送 2026-10-11 · 仓库体积 ~24.9 MB
> 许可：**notebook 内 Markdown 单元格 = CC BY 4.0；Code 单元格 = Apache-2.0**（按单元格类型分别适用）

---

## 0. 一句话结论

**名字误导性很强 —— 它不是 LeetCode 刷题库，是 Datawhale 的「大模型算法与系统」Notebook 教程。**
对本项目的作用是**单点的、但很精准的**：它的「推理优化」与「智能体基础设施」两个专题，
正好覆盖 **Option C-Local 的服务栈（vLLM/SGLang 那套）与 harness 工程（沙箱/轨迹/调度）**，
也就是我们上一轮认定"C 的真实成本所在"的那两块。**感知/分割/网格环境那层它一点忙都帮不上。**

---

## 1. 仓库讲什么

Notebook-first 的可运行教程，主线 `Part 00 → Part 05`，另有 13 个横向/领域专题。

| 部分 | 组别 | 内容定位 | 状态 |
|---|---|---|---|
| Part 00 前置知识与环境准备 | 0A~0E（5 组 / 20 节） | Python/NumPy、PyTorch 张量与自动求导、模型构建、训练直觉、调试与性能 | ✅ |
| Part 01 硬件、数学与系统 | 1A~1F（6 组 / 34 节） | 数值与规模、单设备执行/内存/访存、通信并行容错、执行模型与算子映射、编译后端、硬件适配 | ✅ |
| **Part 02 PyTorch 算法实战** | **10 组** | 2.1 Decoder/注意力 · 2.2 模型架构 · 2.3 SFT · 2.4 偏好优化 · 2.5 反向传播与训练内存 · **2.6 推理执行基础** · **2.7 推理策略与服务调度** · **2.8 量化与低比特适配** · 2.9 通信与并行 · 2.10 项目验证 | ✅ |
| Part 03 Triton 算子开发 | 5 组 / 15 节 | 基础、过渡、Attention 优化、推理优化、项目篇 | ✅ |
| Part 04 CUDA C++ 与系统优化 | 4 组 / 16 节 | CUDA 编程、系统级性能优化、分布式训练工程、架构视野 | 🛠 建设中 |
| Part 05 综合项目与架构决策 | — | 把机制、实验、架构决策收敛为可交付方案 | 🛠 建设中 |

**13 个专题**里，与本项目直接相关的是两个：

- **推理优化（Inference Optimization）** ✅ 持续优化 —— Task0~Task6
  - Task0 请求生命周期与阶段指标（prefill / decode / 终止）
  - Task1 注意力与执行分派（MHA/GQA → FlashAttention/SDPA → 后端 dispatch）
  - Task2 解码推进与候选验证（解码策略 → 投机解码 → MTP / EAGLE → 动态预算）
  - **Task3 KV Cache 表示、管理与复用**（PagedAttention → KV 生命周期与容量控制 → **Prefix Cache 匹配与复用** → **SGLang RadixAttention** → MLA / KV 量化）
  - **Task4 服务迭代预算、调度与异构协同**（迭代成本与 batch 上界 → 解码调度 → **Continuous Batching** → prefill/decode 阶段调度 → 异构 PD）
  - **Task5 低精度执行与单卡部署准入**（浮点基线 → 校准与误差 → W8A16 → 部署 → **量化产物准入 → Serving 复测**）
  - Task6 分布式 Serving、可靠性与 SLO 决策
- **智能体基础设施（Agent Infrastructure）** 🛠 建设中 —— Task0~Task6
  - Task0 智能体工作负载与执行链；Task1 轨迹数据生产与验证
  - **Task2 沙盒与执行环境**（工具契约 → 权限与资源 → 执行日志 → 环境快照）
  - **Task3 轨迹感知调度与上下文管理**（依赖状态 → 就绪队列 → 调度 → 重试与上下文复用）
  - **Task4 任务评估与可观测性**（任务身份 → 轨迹事件 → 错误分类 → 质量与成本报告）
  - Task5 框架与工具协议（MCP / 函数调用 / 业务 API）；Task6 Agentic RL

---

## 2. 对本项目的三层帮助

### 第一层：命中 C-Local 的「serving 栈」门槛 —— 最直接

上一轮确认 C-Local 的三项工程门槛是「权重挂载 / serving 离线安装 / 9h 吞吐」。
Duck harness 的服务端配置（KV 池 1M+ token、FP8 KV cache、跨轮 prefix cache、10~16 局并发波浪）
在这份教程里**每一条都有对应的可跑 notebook**：

| Duck 的配置 | 教程对应入口 |
|---|---|
| 跨轮 prefix cache | `Part 02 · 24 Prefix Cache Matching and Reuse`、`34 SGLang RadixAttention`、项目 `69 Prefix Caching Benchmark` |
| KV 池 / 容量 | `Part 02 · 37 KV Cache Lifecycle and Capacity Control`、`22 vLLM PagedAttention` |
| FP8 KV cache | `Part 02 · 54 KV Cache Quantization Strategies`、`Part 02 · 25 Quantization W8A16` |
| 10~16 局并发 | `Part 02 · 36 Decode Scheduling`、`INF-CONTINUOUS-BATCHING`、`38 Prefill/Decode Scheduling`、项目 `70 Serving Scheduler Benchmark` |
| 27B-FP8 塞进 96GB | `Part 02 · 67 Quantized Inference and Deployment`、`82 Quantization Artifact Evaluation Project`、`83 Quantized Serving Profile Benchmark` |

⇒ **不用盲抄 Duck**，可以先把这几个 notebook 跑通建立基准与直觉，再决定我们的 serving 参数。

### 第二层：Agent 专题 = Duck harness 的结构化说明书

Duck 的三块血肉（对象图感知 / 沙箱与提示词 / 上下文驱逐与并发调度），
Agent 专题的 Task2 / Task3 / Task4 几乎是逐条对齐，且用「工具契约 / 环境快照 / 轨迹事件 / 错误分类」
这套更规范的词汇重新表述。它还明确写了一句对我们很有用的话：

> 「框架名称不能替代任务成功率、轨迹正确性和资源成本证据。」

—— 正好可以拿来当 C 选型的判据，避免又掉进"看起来涨分"的坑。

### 第三层：它提供了一套我们正需要的「证据方法论」

这几天最大的教训是：canned 52.60 vs 官方 0.04、本地分不能预测云端、
**B 补丁"涨 3 局"其实是并行会话改动的误归因**。
这份教程反复强调的正是同一件事——benchmark / profiling / **配对复测** / 证据边界：

- Task5 的串行验收链：`浮点基线 → 量化对象与误差 → 产物 → 加载器 → 执行后端 → 计算内核或回退 → 单卡配对复测`
- 「场景或框架依赖发生变化时，建立新的任务身份和对照，**不沿用其他任务的训练或发布结论**」

与我们今天「冻结快照 → 同机同时刻 A/B → 逐局 diff」是同一思路，**验收模板可以直接借**。

---

## 3. 不该指望它帮上忙的部分（泼冷水）

1. **它和 ARC / 网格环境 / agent benchmark 毫无关系。** 感知层（分割器、平移不变 hash、HUD 判别）
   它一个字都没有 —— 那部分只能靠我们自己移植 Duck 或自研。
2. **Part 03 Triton / Part 04 CUDA 对本项目基本无用**（除非要写自定义 kernel）。
   vLLM/SGLang 已封装，我们不做算子级优化。
3. **Part 02 的 2.1~2.5（注意力/架构/SFT/DPO/反向传播）对本项目是绕过的** —— 我们不训练模型。
   真正要看的只有 **2.6 / 2.7 / 2.8**。
4. **Part 05 与多数主学习路线仍为「建设中」**，别指望有完整内容。
5. **环境门槛**：Part 00/01/02 基础练习可 CPU 跑，但「真实推理、训练和显存实验需要 GPU」
   （Part 03/04 必须 GPU）。要在本机复现 serving 类 notebook 需要 NVIDIA GPU。

---

## 4. 许可（已核对，可用）

README 的许可声明页写的是**混合许可**：

- `.ipynb` 内 **Markdown 单元格**（教程文字、公式、图示说明）→ **CC BY 4.0**
- `.ipynb` 内 **Code 单元格**（可执行实现）→ **Apache-2.0**

两者都与 ARC 竞赛的开源要求**兼容**（竞赛自身获奖提交用 CC-BY 4.0；Apache-2.0 是 OSI 认证许可）。
⇒ **读、改、借鉴代码都没有许可障碍**，只需按单元格类型分别遵守署名要求。

---

## 5. 三条可执行建议

1. **先算一道算术题再谈 C**：把 `24 Prefix Cache Matching and Reuse` + `34 RadixAttention`
   跑一遍，估出「LLM+REPL 每步重复前缀能省多少 prefill」。
   **prefix cache 命中率基本决定 9h 内能跑多少步** —— C 能不能落地是个吞吐算术问题，不是概念问题。
2. **借 Task5 的验收模板做模型选型**：把「浮点基线 → 量化 → 配对复测」套到
   「27B-FP8 vs 更小模型：同样 9h 预算下的**有效决策步数**」上。
3. **读 Agent 专题 Task3/Task4** 当作 Duck harness 的设计说明书（上下文驱逐、就绪队列、轨迹事件、错误分类），
   在动手写 C 的旁路之前先对齐术语与数据结构。

---

## 附：抓取到的原始材料

- `README.md`（37,555 B，含中英文版与完整资产/专题总览）
- `topic_discussion/agent/intro.md`（6,172 B）
- `topic_discussion/inference_optimization/intro.md`（15,514 B）
- `docs/guide.md`（14,107 B）
- `LICENSE`（CC BY 4.0 文本部分）
