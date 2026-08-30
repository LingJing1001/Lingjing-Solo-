# v1.0 根因诊断归档

本目录记录了 v1.0 中期发现并修复的核心 bug 的完整诊断过程。
保留这些脚本，以便任何"LLM 生成原地修改代码"或"重构 simulate"的改动能复盘。

## 时间线

### 1. 现象（中断前的判断，后被修正）
最初怀疑 "FakeLLM 生成的 step() 在 right/down 失效"。

### 2. 精确复现 → 推翻初步判断
`repro_rootcause.py` 发现：**实际失效的是 left/up，right/down 正确**。
方向依赖的错位 = 典型的"状态共享 + 调用顺序敏感"信号。

### 3. 二分隔离
`debugC_isolate.py`：`FakeLLM` 裸代码四方向 + 推箱**全部正确**。
→ bug 不在源码，而在 **WMP 链路（learn → compile → simulate）**。

### 4. 锁定元凶
`debugD_mutate.py`：实验 A/B/C 确认 `step()` 对输入 obj dict **原地修改**
（外层浅拷贝，内部 obj 仍共享引用）。同一 state 连续 right→left，
left 基于被污染的 (2,1) 而非 (1,1)。

### 5. 真正的性能元凶（意外发现）
`debugE/F`：排查首层短路时，发现 `_bfs` 的 `first_layer_best` return
写在 `for depth` 循环**之外**，导致即使首层已定最优，仍展开满 10 层
（实测 `_cached_successor` 调用 **1,398,100 次**，其中 7256 次真实 simulate）。
修复后 `sim_calls: 7256 → 4`。

## 修复

- `codegen.py`：`DynamicsProgram.simulate` 入口加 `_deepcopy_state`，
  保证传给 `step_fn` 的 state 与调用方完全独立（一处修复，Layer A/B 均受益）。
- `planner.py`：`_bfs` 首层结束后立即 `return`，不再展开深层。

## 回归防护

`tests/test_v10_state_isolation.py` 锁定这两个不变量：
1. simulate 不得修改调用方 state（含"right→left 连续调用"边界用例）
2. 推箱语义不被破坏

## 文件说明

| 文件 | 作用 |
|------|------|
| `debug.py`~`debugB.py` | 早期安全编译二分（v0.9 遗留，已归档） |
| `debugC_isolate.py` | 二分：裸代码 vs WMP 链路 |
| `debugD_mutate.py` | 确认原地修改（根因） |
| `debugE_diagnose.py` | 追踪首层短路未生效 |
| `debugF_bfs.py` | 量化 _cached_successor 调用爆炸 |
| `repro_rootcause.py` | 精确复现根因 |
| `verify_fix.py` | 修复验证（三测试） |
