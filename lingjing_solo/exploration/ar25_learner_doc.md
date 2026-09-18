# AR25 Learning Template — 枚举→Pathify→Replay 学习循环

> 更新日期：2026-09-11
> 状态：✅ 已验证 L0/L1/L2

## 架构位置

本模块位于 `lingjing_solo/exploration/` 的上游，与 `ExplorationEngine`（info_gain 驱动）形成**互补关系**：

| 组件 | 位置 | 职责 |
|------|------|------|
| `ExplorationEngine` | `explorer.py` | 信息增益驱动的主动探索 |
| `Ar25Learner` | `ar25_learner.py` | 枚举→Pathify→Replay 的确定性学习 |

## 核心循环

```
observe(state)
  ↓
enumerate_configs(obs, sim) → [(axis_pos, sprite_positions, sel_idx), ...]
  ↓
for config in configs (by coverage desc):
    path = pathify(sim, config)    # collision-aware 动作生成
    if replay(path):               # 在真实引擎验证
        return WIN
return PARTIAL
```

## 关键模块

### pathify(sim, config) → list[int]

把目标配置转换为动作序列，**不假设无碰撞**。

策略：
1. **轴移动**：垂直轴只水平移动（y 保持在 -3），水平轴只竖直移动
2. **Sprite 移动**：按 Dijkstra 寻路，障碍 = 其他 sprite 格子
3. **Collision 处理**：目标被占时临时移开阻塞 sprite

### _enumerate_configs_for_learner

对每个候选轴位置生成能覆盖目标的 sprite 位置组合：
- 几何过滤：sprite 所有 cell 必须在棋盘内
- 覆盖过滤：初始位置无论如何都保留
- 排序：`(is_init, -coverage)` 优先高覆盖

### Ar25Learner.solve

```python
def solve(self, obs, *, replay, max_configs=32) -> SolveResult:
    configs = _enumerate_configs_for_learner(obs, self.sim, max_configs)
    configs.sort(key=lambda c: coverage(c), reverse=True)
    for config in configs:
        path = pathify(self.sim, config)
        if replay(path):
            return SolveResult(path, config, solved=True, ...)
    return SolveResult(best_path, best_config, solved=False, ...)
```

## 验证结果

| Level | Sprites | Axis | Result | Steps |
|-------|---------|------|--------|-------|
| L0 | 1 | 垂直 | ✅ WIN | 15 |
| L1 | 1 | 垂直 | ✅ WIN | 11 |
| L2 | 2 | 水平 | ✅ WIN | 47 |

## Bug 修复记录

1. **Dijkstra 几何检查** — `dijkstra_path` 添加 `sprite_fit()` 验证 sprite 整个形状在棋盘内
2. **轴 y_reset** — 垂直轴不需 y_reset（y=-3 是合法停泊位，镜像只依赖 x）
3. **枚举覆盖计算** — 对每个 `axis_val` 单独生成候选（之前用 `axis_val=None` 导致反射计算错误）
4. **初始位置保留** — 无论 coverage 是否为 0 都保留初始位置
5. **候选排序** — 改为优先 coverage 而非距离初始位置

## 与 ExplorationEngine 的关系

- `ExplorationEngine` 用 **信息增益** 驱动探索（未知状态优先）
- `Ar25Learner` 用 **配置枚举** 驱动学习（已知目标优先）
- 两者可组合：先用 Learner 找可行配置，再用 Explorer 优化路径

## 文件对应

| 文件 | 职责 |
|------|------|
| `ar25_learner.py` | 核心实现 |
| `ar25_solver/simulator.py` | Dijkstra + 几何检查 |
| `arc_shadow.py` | `learn_solve_level()` 入口 |
