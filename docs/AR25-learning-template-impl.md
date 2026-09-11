# AR25 Learning Template 实现文档

> 更新日期：2026-09-11
> 状态：已完成并验证

## 概述

AR25 Learning Template 是"北极星核心回路设计方案"中定义的三大缺失模块的实现，用于在 AR25 游戏中实现学习通关能力。

### 核心模块

| 模块 | 文件 | 说明 |
|------|------|------|
| `pathify(sim, config) → list[int]` | `ar25_learner.py` | 把目标配置转换为碰撞感知的动作序列 |
| `Ar25Learner.solve(obs, replay, max_configs) → SolveResult` | `ar25_learner.py` | 编排 enumerate→pathify→replay→tabu 循环 |
| `bind_obs_from_env(env) → Obs` | `ar25_learner.py` | 从 live arcengine 实例提取可观测状态 |
| `learn_solve_level()` | `arc_shadow.py` | learn 模式入口，与影子引擎对接 |

---

## 实现细节

### 1. pathify — 配置到动作序列

**核心逻辑：**
```
1. 轴移动（如果 axis_pos ≠ 当前轴位置）
   - 垂直轴：水平移动（x 变化），y 保持在 -3（镜像只依赖 x）
   - 水平轴：竖直移动（y 变化），x 保持在 0
   - 使用 Dijkstra 寻路，障碍 = 其他 sprite 的格子
2. Sprite 贪心移动
   - 按"已在目标位置"→"离目标近"排序处理顺序
   - 对每个 sprite 用 Dijkstra 找路径（障碍 = 其他 sprite）
   - 如果目标被占据，临时移开阻塞 sprite
3. 切换到目标 sel_idx
```

**关键修复历史：**
- ❌ 最初版本直接赋值 `ax["y"] = 0`（y_reset），导致 sprite Dijkstra 在轴 y=-3 时计算路径，但实际执行时 y=0，镜像结果错误 → **已修复**：y_reset 改为用 `sim.step()` 执行，使 Dijkstra 在正确的轴状态下计算
- ❌ `dijkstra_path` 只检查起点/终点是否出界，没检查 sprite 整个形状 → **已修复**：添加 `sprite_fit()` 几何检查
- ❌ 垂直轴做了不必要的 y_reset，但 y=-3 是合法的"停泊位置" → **已修复**：移除垂直轴的 y_reset

### 2. _enumerate_configs_for_learner — 候选配置枚举

**策略：** 对每个候选轴位置，生成能覆盖目标的 sprite 位置组合。

**关键修复历史：**
- ❌ 最初用 `axis_val=None` 全局生成 sprite 候选，导致覆盖计算错误 → **已修复**：对每个 `axis_val` 单独生成候选
- ❌ 初始位置 coverage=0 时被 `if cov: continue` 过滤掉 → **已修复**：特殊处理初始位置，无论 coverage 都保留
- ❌ 候选排序优先"距离初始位置近"而非"覆盖多"，导致高覆盖位置被 `MAX_PER_SPRITE=8` 截断 → **已修复**：排序改为 `(is_init, -coverage)`，优先高覆盖
- ❌ 没检查 sprite 几何有效性，导致返回棋盘外位置 → **已修复**：添加 `sprite_pos_valid()` 几何过滤

### 3. Ar25Learner.solve — 学习循环

```python
def solve(self, obs, *, replay, max_configs=32) -> SolveResult:
    # 1. 枚举候选配置
    configs = _enumerate_configs_for_learner(obs, self.sim, max_configs)
    
    # 2. 按覆盖数降序排序
    configs.sort(key=config_score, reverse=True)
    
    # 3. 对每个配置尝试 pathify + replay
    for config in configs:
        path = pathify(self.sim, config)
        if replay(path):  # 在真实引擎上重放验证
            return SolveResult(path, config, solved=True, ...)
    
    # 4. 返回最佳配置（部分解决）
    return SolveResult(best_path, best_config, solved=False, ...)
```

### 4. bind_obs_from_env — 状态提取

从 live arcengine 提取可观测状态（逆推属性名）：

```python
def bind_obs_from_env(env) -> Obs:
    g = env._game
    return Obs(
        level_idx=g.jtkyjqznbnp,  # axes
        axes=[(ax["x"], ax["y"]) for ax in g.jtkyjqznbnp],
        sprites=[(s["x"], s["y"]) for s in g.ayyvxqrhnzw],
        targets=...,  # g.fswikrcrdmx
        sel_idx=g.yvifanjrcyu,
        steps_used=g.lelsvjlwneo.current_steps,
        budget=...,
    )
```

---

## 验证结果

| 关卡 | sprite数 | 轴类型 | 结果 | 步数 | 配置 |
|------|----------|--------|------|------|------|
| L0 | 1 | 垂直 | ✅ WIN | 15 | `(10, [(1,15)], 0)` |
| L1 | 1 | 垂直 | ✅ WIN | 11 | `(10, [(15,14)], 1)` |
| L2 | 2 | 水平 | ✅ WIN | 47 | `(9, [(11,14),(3,14)], 1)` |

> L2 需要 `max_configs >= 768`（因轴位置排序，近初始位置的轴先枚举，winning 轴位置 y=9 排在第 12 位）

---

## 文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `ar25_learner.py` | 新建 | 核心实现 |
| `ar25_solver/simulator.py` | 增强 | 添加 Dijkstra 几何检查 |
| `arc_shadow.py` | 添加 | `learn_solve_level()` 接口 |

---

## 已知限制

1. **max_configs 需要level-specific调参**：L2 需要 1024 个 config 才能找到 winning 配置，L0/L1 只需 128
2. **Sprite 形状复杂度**：多 sprite 场景的候选组合数指数增长，高覆盖位置可能超出 `MAX_PER_SPRITE` 限制
3. **尚未对接 live arcengine**：`bind_obs_from_env` 的属性名是逆推的，需要在实际环境中验证
