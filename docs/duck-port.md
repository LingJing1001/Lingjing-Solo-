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
| `sandbox.py` | 460 | 一个 `python` 工具；`action()`；四类不可吞的拒绝（noop / 致死 / 重复 / 终局）；保留函数 | 结构照抄，实现重写 |
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
- `scripts/duck_router_selftest.py` —— 真引擎验证"duck 真的从 `my_agent` 手里接过方向盘"
- `scripts/mock_duck_server.py` —— 离线 OpenAI 兼容端点，验证管线不依赖模型
- `tests/test_duck.py` —— 15 条，钉住语义
- `tests/test_duck_death_ledger.py` —— 6 条，钉住死亡账本（noop 守卫的另一半）
- `tests/test_duck_memory.py` + `tests/test_duck_driver_memory.py` —— 15 条，钉住记忆与接线
- `integrations/ARC-AGI-3-Kaggle-Starter/agent/duck_bridge.py` —— 路由侧的缝（开关、预算、门禁、动作转换）
- `integrations/ARC-AGI-3-Kaggle-Starter/tests/test_duck_bridge.py` —— 14 条，钉住缝的每条判断

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
| `tests/test_duck_death_ledger.py` | **6/6**（致死动作记录、拒绝零成本、坚持放行、存活撤销、换关清空、noop 不误记） |
| `scripts/duck_bridge_selftest.py --game cn04` | PASS，4 个动作，**计分卡也记 4** |
| `scripts/duck_bridge_selftest.py --game ar25` | PASS，同上 |
| `duck_play.py --backend openai`（mock 端点，ar25，8 步） | 8 动作，6840 tokens，`marker_pos` 保留 |

2026-10-21 晚，接进路由之后：

| 检查 | 结果 |
|---|---|
| `tests/test_duck_bridge.py` | **14/14**（开关语义、门禁、每 tick 一动作、兜底、agent 侧转换） |
| `scripts/duck_router_selftest.py`（su15，真引擎） | PASS，3 个 duck 动作全经 `env.step`，**计分卡记 3**，脚本用完自动退出 |
| `play_local.py --game su15 --max-steps 25`（默认 `LINGJING_DUCK` 未设） | 1 关 / 1.72 分，日志里没有 duck 一行 —— 默认路径零改动 |

真实模型：**仍未接**。本机没有在跑的推理服务。但链路（HTTP → tool_calls → 沙箱
→ `env.step` → 计分卡 → `my_agent.choose_action` 提前 return）已经逐段验证过，
缺的只是模型本身。

## 接进路由：`agent/duck_bridge.py`

`_choose_action_inner` 本来就有三级降级：本地专用求解器 → shadow → IDDFS。
三条都空手时它现在会写一个 `_duck_request`，下一次 `choose_action` 由 duck 接手。

```
choose_action
  ├─ engine_solver（已知局）
  ├─ lp_observe（lexicon 看帧）
  ├─ _duck_tick ← 新增：命中就 return，绕开 v14/AOP 改写
  └─ _choose_action_inner
        └─ R3：local → shadow → IDDFS → 全空则 _duck_request = levels
```

三条硬约束：

1. **默认 off。** 44.57 是 duck 缺席时的成绩；默认开等于未经允许改数字，而且
   本机 8001 上任何 mock 网关都会让它"看起来能用"。`LINGJING_DUCK=auto` 才会
   按端口探测生效，`on` 强制生效。
2. **预算卡在 chat 外层**（`GuardedChat`）。duck 的调用跑在 `choose_action` 里，
   只在 tick 之间看墙钟挡不住一次 120 秒的阻塞；闸门放在每次调用之前，超时
   raise，duck 的 `_inner` 本来就有 `except Exception`，生成器自己收尾。
3. **动作必须在 v14/AOP 之前 return。** 那两层会改写动作（AOP 甚至会换成别的
   合法名），duck 的动作是被计分的原始意图，不能被"优化"。

配置（全部可缺省）：

| 变量 | 默认 | 含义 |
|---|---|---|
| `LINGJING_DUCK` | `off` | `off` / `auto` / `on` |
| `LINGJING_DUCK_PROBE` | `0.6` | auto 模式的 TCP 探测超时（秒） |
| `LINGJING_DUCK_BUDGET` | `300` | 每局 duck 墙钟预算（秒） |
| `LINGJING_DUCK_STEPS` / `_TURNS` | `80` / `24` | 动作与回合上限 |
| `LINGJING_DUCK_PIN` | `level` | 过关记忆何时进 system prompt；短局用 `level` |
| `LINGJING_DUCK_MEMORY` / `_TRANSFER` | `on` | 过关记忆 / 关卡迁移开关 |

## 四项的性价比排序

排序不按代码量，也不按"冠军做了什么"，而按**它直接作用于分数公式里的哪个量**：

```
分数 = (人类基准 / 本局动作数)²  ，25 局取平均
        └─ 动作数进分母 ─┘        └ 关卡数进分子 ┘
```

| 优先级 | 项 | 省下的是什么 | 作用于 | 状态 |
|---|---|---|---|---|
| **1** | 过关记忆 | 跨关重学机制的 token 与步数 | 分母（每关步数） | 已做 `memory.py` |
| **2** | 无效动作守卫（noop + 死亡账本） | 白花的动作预算；致死动作吃掉整次尝试 | 分母（更狠） | 已做 `sandbox.py` |
| **3** | 优先级调度器 | 死局不再耗尽全局预算 | 分子（救回几局） | **未做**，设计见下 |
| **4** | 保留函数 | 重复写代码的 token | 都不是，是写作成本 | 已做，**串行期无收益** |

为什么是这个顺序：

- **1 和 2 都改分母，且都只依赖引擎已经告诉我们的东西**（上一关走过的动作、
  这个盘面按过什么）。零模型能力要求 —— 0.6B 也能吃到，所以排前面。
- **2 比 1 更便宜**（每一步都在生效，1 一局只生效几次），但 1 的收益按关卡数
  线性累加，2 的收益取决于模型有多爱瞎按。真实部署里两者同时开，排序只决定
  先做谁。
- **3 的收益最大**（冠军 27.89 分的主要来源），但它改的是**跑法**不是模块：
  串行单局 → 多局轮转，预算分配、超时、日志、复现全要重做。所以它必须排在
  "不改变跑法就能拿到分"的两项之后。
- **4 在串行下确实不值钱**：一次调用能写完的函数，下次重写的 token 成本远小于
  "跨局共享函数库"要付的复杂度。它只在**同时维护 N 个上下文**时才成立 —— 那时
  省的是 N 倍的重复定义，还能让弱模型少犯"函数写坏了"的错。所以它跟着 3 走，
  3 不做，4 就没有场景。

### 2 的落地：死亡账本（本轮补上）

`_noop` 保一步，`_deadly` 保一局。规则和 noop 完全对称，实现在 `sandbox.py`：

- key 是**动作起始的盘面**，不是结束的 —— 否则重置回同一盘面永远命中不了；
- 连续第二次坚持就放行（反证必须做得到，否则和硬编码黑名单没区别）；
- 这一次活下来了 → 撤销该条（和 noop 在"板子真的动了"时撤记录同一条规则）；
- 换关清空（新关卡的死法不适用）。

`stop_reason` 是 `known_deadly`，`refusals["deadly"]` 单独计数，
`tests/test_duck_death_ledger.py` 6 条钉住上面每一条。

### 3 的设计（未做，先写清楚再动手）

关键认识：**`deferred.py` 已经把调度器需要的东西准备好了。** 它把"一局的回合"
暂停在 `yield` 上，于是多局交替推进不需要多进程、不需要线程 —— 只要多个
driver 轮流 poll。

```
lanes: {game_id: (driver, env)}
loop:
    gid = pick()                     # 按优先级
    nxt = lanes[gid].driver.poll()
    res = lanes[gid].env.step(nxt)   # 计分卡在这里计数
    lanes[gid].driver.feed(res)
    update_priority(gid, res)
```

优先级用**边际产出**，不用绝对分数：

```
gain(gid) = (近期关卡增量) / max(1, 近期动作数)
priority  = gain × 未完成关卡数 − 冷却罚分
```

三条必须写死的规则（否则调度器会让结果不可复现，比不做更糟）：

1. **lane 数设上限**（本地 ≤ 6，内存和 CPU 都撑不住 25 个 env）；
2. **步数配额按优先级切**，不按墙钟 —— 墙钟会让结果随机器负载漂移；
3. **每局的随机种子和初始状态要落盘**，否则"这次分数高"无法归因到调度。

工作量：调度器 + 优先级约 200 行，多 env 生命周期约 100 行，日志/复现约 100 行。
风险集中在"预算切分"上：切得太碎，每局都要重新热身；切得太粗，就等于没调度。

### 4 的定位修正

`retained` 保留在沙箱里（对单局仍有微弱的正收益，且测试已覆盖），但**不投入
任何额外工作**。等 3 落地、真的并行 N 局时再回来做"跨局共享函数库" ——
那时它才从"省 token"变成"省 N 倍 token + 降低弱模型的出错面"。

## 与冠军的其余差距

不按分数排序，因为这几项的收益依赖具体局：

| 差距 | 影响 | 状态 |
|---|---|---|
| **模型能力** | 0.6B 写不出可用代码 → 骨架空转 | 待定 |
| `ToolAgent` 的 yield/resume/retry | 长跑稳定性和掉线恢复 | **已做**（`deferred.py`） |
| 接进 `my_agent` 路由 | 未知局 R3 全空后有兜底 | **已做**（`duck_bridge.py`） |
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
