# 接通本地大模型 + 跨局反思记忆

这个文档说两件事：① 怎么把 duck 接到本地 OpenAI 兼容端点（如 Qwen3-8B/27B），
让关内反思/死亡反思/通关反思真正生效；② 跨局持久化怎么开（已写好的代码，
端点一接通就立刻能用，不需要额外改任何东西）。

---

## 1. 反思机制现状（接通前 vs 接通后）

| 反思层级 | LINGJING_DUCK=off（默认） | LINGJING_DUCK=on（接通后） |
|---|---|---|
| 关内对话压缩（长上下文） | 不触发（duck 不跑） | tokens≥60k 让模型自总结，丢细节留结论 |
| 死亡/重开反思 | 不触发 | "机制认知仍有效，正在做的无效，别重蹈覆辙" |
| 通关反思（Rule for level N） | 不触发 | 通关时记动作+推理尾，下关 system prompt 末尾钉规则块 |
| **跨局反思（同 game_id）** | **不触发** | **代码已写好，端点接通后立刻生效** |

跨局这块是补 lingjing_solo transfer 链路未接入 my_agent 的真实缺口——不是
替代它，是先有最小可工作版本，避免一次性引入整个 transfer 子系统。

---

## 2. 接 Qwen3 端点（OpenAI 兼容）

duck 默认连 `http://127.0.0.1:8001/v1`，model=`local`，key=`EMPTY`。端点配置
全靠环境变量，不开代码：

```bash
# 必填三件套
export LINGJING_LLM_URL=http://127.0.0.1:8001/v1     # 你的端点
export LINGJING_LLM_MODEL=qwen3-8b                   # 或 qwen3-27b / qwen2.5-7b-instruct 等
export LINGJING_DUCK=on                               # 强制接线（auto 探测失败会回退）

# 可选调优
export LINGJING_DUCK_BUDGET=300                      # 单局 LLM 调用总预算秒数
export LINGJING_DUCK_STEPS=80                        # duck 单关最大步数
export LINGJING_DUCK_TURNS=24                        # duck 单关最大回合数
export LINGJING_DUCK_MEMORY=on                       # 通关反思（默认 on）
export LINGJING_DUCK_TRANSFER=on                     # 关间 transfer（默认 on）
export LINGJING_DUCK_THINK=off                       # Qwen3 thinking 模式（短局建议 off）
export LINGJING_DUCK_PIN=level                       # pin 策略：level 立即生效
export LINGJING_DUCK_CROSS_DIR=runs/cross_game       # 跨局持久化目录（同 game_id 跨进程复用）
```

### 起服务的三种主流方式（按你机器条件选一种）

**A. vLLM（推荐，GPU 机器）**
```bash
# 装一次
pip install vllm
# 起 Qwen3-8B（GPU 显存 16GB+ 够用）
vllm serve Qwen/Qwen3-8B --port 8001 --max-model-len 32768 --gpu-memory-utilization 0.9
# 起 Qwen3-27B（需要 40GB+ 显存，或用 AWQ 量化版）
vllm serve Qwen/Qwen3-27B --port 8001 --max-model-len 32768 --quantization awq
```

**B. Ollama（CPU 也能跑，最简单）**
```bash
# 一次性装 ollama 后
ollama pull qwen3:8b                  # 或 qwen3:27b
# ollama 默认在 11434，duck 端点要改
export LINGJING_LLM_URL=http://127.0.0.1:11434/v1
export LINGJING_LLM_MODEL=qwen3:8b
```

**C. llama.cpp server（CPU/GPU 都行，最轻量）**
```bash
# 装好后，下 GGUF 量化文件
llama-server -m qwen3-8b-q4_k_m.gguf --port 8001 --ctx-size 32768
# 端点默认就是 http://127.0.0.1:8001/v1，model 任意填
```

---

## 3. 验证端点接通

端点起好之后，先做一次干跑（不跑分，只看路由是不是从 CEAX_UNKNOWN 变成
INLINE/DUCK 接管）：

```bash
cd integrations/ARC-AGI-3-Kaggle-Starter
export LINGJING_SCRIPTS=1
export LINGJING_DUCK=on
export LINGJING_LLM_URL=http://127.0.0.1:8001/v1
export LINGJING_LLM_MODEL=qwen3-8b
export LINGJING_DUCK_CROSS_DIR=runs/cross_game
export TMP=/d/tmp    # C 盘 100% 满时绕开

.venv312/Scripts/python.exe scripts/play_local.py --game ar25 \
    --max-steps 400 --max-minutes 3 --parallel 1 2>&1 | tail -20
```

启动日志里应看到：
- `[smart-router-v2...] boot game=ar25 ... route=INLINE:ar25`（已知局仍走预置计划表）
- 某局失败后 `[duck-bridge] start gid=<game_id> ...` —— 接到 LLM
- 通关后 `[duck-drv] rule recorded for level N: ...` —— Rule for level N 被记
- 进程退出前 `[duck-bridge] cross-game saved gid=<game_id> ...` —— 落盘
- 第二次同局跑时 `[duck-bridge] cross-game loaded N rule(s) gid=<game_id> ...` —— 读回

---

## 4. 跨局持久化代码做了什么（已实现）

落点：

| 文件 | 改动 |
|---|---|
| `arc_adaptor/duck/memory.py` | `LevelRecord.to_dict/from_dict`；`SolvedLevelMemory.load_from_dir/save_to_dir/merge_records/to_payload/_path_for` |
| `integrations/.../agent/duck_bridge.py` | `DuckBridge._cross_dir`（env `LINGJING_DUCK_CROSS_DIR`）；`_load_cross_game()` 启动时读回；`_save_cross_game()` 收尾时写盘；`atexit.register` 兜底进程退出；`_abandon` 注销避免重复 |
| `tests/test_cross_game_memory.py` | 6 项往返测试全过 |

设计要点：
1. **范围 = 同 game_id 跨进程**。sb26 学到的规则不会自动带到 sb27——那是
   lingjing_solo transfer 的"同族迁移"，更重，留作下一阶段。
2. **合并语义 = in-game wins**。disk 记录只填"我还没有的关"，不覆盖
   in-game 学的规则。`overwrite=True` 可改成"later run wins"，但默认不开。
3. **原子写**用 `tmp + os.replace`，避免半写文件被下次读回。
4. **atexit 兜底**：my_agent.is_done 三处 return True 都不调 shutdown，进程
   退出时 atexit 保证 driver 还活着的通关规则一定落盘。零侵入 my_agent。
5. **禁用开关**：`LINGJING_DUCK_CROSS_DIR=""`（空字符串）则关掉跨局，退化到
   today 的纯 in-memory 行为。一键回滚。

---

## 5. 下一步建议（按性价比）

- [ ] 真接上 Qwen3-8B 端点，跑一次 ar25 全程，看 `[duck-drv] rule recorded`
      日志验证通关反思生效
- [ ] 同一局跑两遍，看第二次 `[duck-bridge] cross-game loaded` 验证跨局复用
- [ ] 跨局扩到"同游戏系列"（sb26 → sb27）：需要游戏系列分类的领域知识，目前
      scope 之外
- [ ] 评估要不要把 lingjing_solo transfer 整链接进 my_agent（重型，5+ 文件，
      评分预期更高但风险大）
