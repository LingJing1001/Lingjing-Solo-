# 向冠军靠拢：Duck 骨架移植账

> 目标：不再维护自研启发式决策链，改为复现冠军（Tufa Labs 的 Duck harness）的
> 结构：**模型每轮现场写 Python 决策**，脚手架只提供读盘、花动作、拒绝、压缩。

## 一句话结论

架构已经同构，能跑通；**能力天花板不在代码里，在模型里**。
冠军的决策 100% 来自 128K 的 Qwen3.8-Flash-Next，本机最强是 0.6B。
把冠军的骨架接上 0.6B，会得到一台完全正确的空转机器。

## 冠军到底有什么

从 notebook（29 个 cell、498KB patch）里逐段读出来的，不是我猜的：

| 部件 | 它做什么 | 为什么关键 |
|---|---|---|
| `tool_agent` 循环 | 模型每轮拿到帧，可写 Python 查证，然后 `action(...)` | 决策是**现场写的**，不是预先写好的策略 |
| `python_tool_sandbox` | `python -I -S` 子进程，暴露 `current_frame`/`history`/`transitions`/`frame_diff`/`action` | 读和算免费，只有 `action()` 花预算 |
| `WORLD_MODEL_ADDENDUM` | "维护世界模型……低置信度时**写一个判别性探针**" | 这是全部的"推理策略"，没有目标检测器 |
| `noop_repeat_guard` | 已知无效的动作被拒，但**再发一次就放行** | 硬封锁会自我封闭，永远收不到反证 |
| `retained_functions` | 模型写的函数跨调用保留（按关卡作用域） | 跨关迁移的载体 |
| `SUMMARY_REQUEST_PROMPT` | 上下文快满时让**模型自己写摘要**，然后丢掉历史 | 摘要由理解局面的一方写，才不会丢关键 |
| `priority_scheduler` | 10 局并行 ×25 分钟，按剩余关卡奖励、按动作/token 衰减 | **27.89 分主要来自这里**，不是单盘打穿 |

关键否定事实：**冠军没有目标检测层、没有键语义学习、没有状态转移图**。
这三样是我们的发明，存在理由是"本机没有能现场推理的模型"。

## 我们移植了什么

`arc_adaptor/duck/`，8 个文件：

| 文件 | 行数 | 内容 | 来源 |
|---|---|---|---|
| `prompts.py` | 190 | 冠军 prompt 原文（标 `PORTER` 的都是逐字） | cell 3 `prompts.py` |
| `views.py` | 300 | `.ascii` / `.shape` / `.segmentation` / `frame_diff` | 重写（冠军原文在未参与 patch 的文件里） |
| `sandbox.py` | 420 | 一个 `python` 工具；`action()`；三类不可吞的拒绝；保留函数 | 结构照抄，实现重写 |
| `env.py` | 130 | 本地 `arc_agi` 引擎适配 | 新增（冠军用 docker gateway） |
| `agent.py` | 290 | 轮次循环、模型自写摘要、关卡迁移 | 重写（`ToolAgent` 类体在原仓库） |
| `chat.py` | 150 | OpenAI 兼容单次调用 + tool 支持 | 新增 |
| `deferred.py` | 460 | 可暂停驱动：每 tick 吐一个动作给 harness 计分 | 新增（冠军内建 yield/resume） |
| `memory.py` | 210 | 过关记忆：记录 + 要一行规则 + 钉在 system prompt 末尾 | 新增（lalalia9 的 `SOLVED_LEVEL_MEMORY_PATCH`） |

配套：

- `scripts/extract_champion.py` —— 把 notebook 里的 patch 拆成 `vendor/champion/`
  （新增文件完整还原；修改文件只能存"新增行片段"，因为原文件不在 notebook 里）
- `scripts/duck_play.py` —— 入口，两种后端
- `scripts/duck_bridge_selftest.py` —— 真引擎验证"动作经 `env.step` 才计分"
- `scripts/mock_duck_server.py` —— 离线 OpenAI 兼容端点，验证管线不依赖模型
- `tests/test_duck.py` —— 15 条，钉住语义
- `tests/test_duck_memory.py` + `tests/test_duck_driver_memory.py` —— 15 条，钉住记忆与接线

## 三个不可替代的设计决定

**1. 一轮在"动作执行"时结束，不在模型闭嘴时结束。**
否则模型会继续读一个它已经作废的棋盘。`executed_now` 是每轮的边界。

**2. 拒绝必须继承 `BaseException`。**
冠军的注释写得很清楚：模型习惯写 `try: action(...) except Exception:`，一次
把 1500 次连续拒绝当成了"整块棋盘都是惰性的"。可捕获的拒绝等于没有拒绝。

**3. 重复判定按"出发棋盘"记账。**
第一版按动作**之后**的棋盘记账，于是 `action(['DOWN','DOWN'])` 被当成重复拒绝——
任何正常的批量都被打死了。这个 bug 在 `tests/test_duck.py` 里有一条测试守着。

## 压缩不能切断 tool 调用链

`messages` 里出现一条 `tool` 消息、却没有对应的 `assistant.tool_calls`，所有
provider 都会 400，而且看起来像"模型坏了"。所以 `_compact()` 只有两条路：
模型给了摘要 → 整段替换成 `[system, 摘要, "Understood"]`；模型没给 → `_drop_oldest()`
只保留最后一条 assistant 及其之后的消息。两条路都不会留下孤立的 tool 消息。

## 实测（2026-10-19）

| 检查 | 结果 |
|---|---|
| `tests/test_duck.py` | **15/15** |
| `tests/test_goalsim.py` | 6/6（未被破坏） |
| `tests/test_view.py` | 7/7（未被破坏） |
| `--backend script`（离线片段序列） | 5 步，`frame_diff` 报 `delta [3,0]`，保留函数生效 |
| `--backend openai` + mock 端点 | 8 步，11 次调用，`marker_pos` 跨调用保留 |

2026-10-21，补上驱动层与记忆之后：

| 检查 | 结果 |
|---|---|
| `tests/test_duck.py` + `test_duck_memory.py` + `test_duck_driver_memory.py` | **30/30** |
| `scripts/duck_bridge_selftest.py --game cn04` | PASS，4 个动作，**计分卡也记 4** |
| `scripts/duck_bridge_selftest.py --game ar25` | PASS，同上 |
| `duck_play.py --backend openai`（mock 端点，ar25，8 步） | 8 动作，6840 tokens，`marker_pos` 保留 |

真实模型：**仍未接**。本机没有在跑的推理服务。但链路（HTTP → tool_calls → 沙箱
→ `env.step` → 计分卡）已经逐段验证过，缺的只是模型本身。

## 与冠军的真实差距

按"影响分数"排序，不是按代码量：

| 差距 | 影响 | 状态 |
|---|---|---|
| **没有优先级调度器** | 冠军 27.89 分的主要来源；我们仍是单盘全押 | 未做 |
| **模型能力** | 0.6B 写不出可用代码 → 骨架空转 | 待定 |
| `ToolAgent` 的 yield/resume/retry | 长跑稳定性和掉线恢复 | **已做**（`deferred.py`，见下） |
| 过关记忆（solved-level memory） | 关卡交界处不重学机制（lalalia9 版 +2.8） | **已做**（`memory.py`） |
| 死亡账本 | 同一盘面致死动作的拦截 | 未做（只做了 noop） |
| segmentation 的旋转/姿态/边界 | 认旋转对称、认贴边物体 | 简化 |
| 动画帧 | 瞬态像素，只在中间帧出现 | 未做（引擎不暴露） |
| 40k/12k 分块压缩 | 长局上下文利用效率 | 未做（整段摘要） |

## 已补上的两块：可暂停驱动 + 过关记忆

### `deferred.py` —— 每 tick 吐一个动作

计分卡只在 `env.step()` 里计数（`arc_agi/wrapper.py:187`），
`game.perform_action()` 不计。所以 duck 不能自己在沙箱里执行动作，必须把动作
交给 harness 花掉。`DuckDriver` 把回合循环改成生成器：

```
driver.poll()  -> ("ACTION2", {...})   # harness 用 env.step 花掉
driver.feed(raw)                        # harness 交回结果
driver.poll()  -> 下一个动作（同时投递上一个结果）
```

**`feed` 只暂存、下一次 `poll` 才投递**。改成在 `feed` 里 `send` 看起来等价，
其实必错：那时生成器已经停在下一个 `yield`，`poll` 自己的 `send(None)` 会把
`None` 塞给 `step_result`，动作结果静默丢失。自测
`scripts/duck_bridge_selftest.py` 会同时核对三件事：动作数 = harness 花掉的数 =
计分卡记的数。

### `memory.py` —— 过关记忆

过关时记下关号 / 动作数 / 获胜尝试最后 30 个动作 / 那一轮推理的尾巴，
接下来两轮向模型要一行 `Rule for level N:`，把记录钉在 **system prompt 末尾**。
两个设计约束：

- **只在历史被重建时才改写**（`pin_policy="drain"`，默认）：那时服务端本来就要
  重读上下文，改 system prompt 不额外花钱。`pin_policy="level"` 改成每次过关就
  写，适合本地从不会发生压缩的短跑，但会打断前缀缓存。
- **规则是"问出来的"，不是我们改写的**：模型没答就记 `not stated`。

`tests/test_duck_memory.py` 钉契约，`tests/test_duck_driver_memory.py` 钉接线
（含"RESET 后只记获胜那次尝试"、"死亡不算过关"）。


## 模型：唯一的硬分叉

冠军 `sglang serve` 跑 Qwen3.8-Flash-Next（INT4），需要大显存。本机是
MX250 2GB + 7.8GB RAM，连 1.7B 都 OOM。

三条路，代价不同，能力差一个数量级：

1. **接外部 API**（任意 70B 级模型）。能真正接近冠军，但闭卷比赛不能用。
   适合先把"架构是否成立"验证到底。
2. **本机换代码专用小模型**（如 Qwen2.5-Coder-1.5B-Instruct Q4）。冠军要的核心
   能力是**写代码查证**，不是世界知识；代码模型在这个尺寸上比同级通用模型强很多。
   1.5B Q4 约 1GB，2GB 显存勉强，是本机唯一有希望的方向。
3. **维持 0.6B 通用模型**。架构能跑，但不会写代码，等于没跑。

## 不要做的事

- **不要再往 `lexicon/` 里加启发式。** 那是"没有模型时的降级方案"，
  现在主路径换成 Duck，继续加层只会让两套逻辑互相遮蔽。
- **不要用单个盘的分数判断 Duck 是否有效。** 冠军的分数来自调度，
  单盘 0 关是正常的。
