# AR25 剪枝对照规格（ARC-SAGE → Lingjing R2）

> 日期：2026-09-03  
> 来源：本地摘录 `reference/arcsage-ar25/`（上游 https://github.com/dp-web4/ARC-SAGE）  
> 目标：把「L7/L8 可解」的搜索范式落到 R2 感知 + 世界模型（`encoder` / `field`），并为规划侧提供可实现的剪枝契约。

## 0. 一句话结论

**不要在 ACTION1–5 的动作树上硬 DFS/Best-First 冲 L7。**  
ARC-SAGE 的可复现路径是：**配置空间暴力枚举（块坐标 × 镜轴坐标）→ 反射闭包判定覆盖 → 再反推短路径脚本**。  
Lingjing 现有报告中的贪心 Best-First（300s / 60 万节点超时）正是在错误搜索空间里烧预算。

## 1. 对照表

| 维度 | Lingjing 现状（`arc-agi-3-ar25-report.md`） | ARC-SAGE `ar25_solver` / `ar25_solve_v2` | R2 应对齐的规格 |
|------|---------------------------------------------|------------------------------------------|-----------------|
| 搜索对象 | 动作序列（UP/DOWN/LEFT/RIGHT/CYCLE） | **终态配置** `(piece_xy[], axis_coord[])` | `field` 提供「配置 → 覆盖集合」；规划只搜配置 |
| 深度含义 | 动作深度（易爆） | **反射 bounce 深度 ≤ 12**（引擎常量） | 反射 BFS `depth > 12` 剪掉；与动作深度解耦 |
| 胜利判定 | 引擎 `vplrhaovhr()` / 覆盖全部目标点 | 同构：`covered ⊇ targets` | `field.check_cover(state) -> bool + uncovered` |
| 状态键 | 曾漏选中对象 → 假解 | 配置层不依赖选中；执行层再管 SELECT | `state_key` 必须含：各 sprite 位姿/像素哈希、选中、旋转距离表 |
| 剪枝主轴 | `f=10h+depth`，h=未覆盖启发 | **轴坐标先穷举**（1 轴 O(W) / 2 轴 O(W·H)），无解再动块；固定块跳过 | 分层枚举：固定跳过 → 轴 → 块；剩余覆盖下界剪枝 |
| L7/L8 | 超时 / 未试 | 离线 brute-force 出配置 + **罐头序列**（含 L7/L8） | 先复现配置枚举；路径层可先罐头，再自动 pathify |
| 碰撞/旋转 | 引擎 oracle 隐式处理 | L7 注释：理想点 `(8,1)` 实际因碰撞在 `(8,11)` 通关 | 纯几何枚举后必须 **引擎回放校验**；旋转块用距离触发表 |
| NVARC/MindsAI | — | **不适用**（ARC-AGI-2 静态题） | 不要再当 AR25 DFS 蓝本 |

### ARC-SAGE 已给出的 L7/L8 配置锚点（`ar25_solve_v2.py`）

| 关 | 镜轴目标 | 块目标 | 备注 |
|----|----------|--------|------|
| L7 | H: y=5→7；V: x=3→12 | p1→(7,7)；p2 计划 (8,1)，实胜 (8,11) | 必须引擎回放 |
| L8 | H: y=5→11；V: x=3→12 | p1→(4,6)；p2→(16,3) | 60 墙，双轴双块 |

步数预算（`metadata.json` baseline）：L7=233，L8=73。引擎 `StepCounter`：L1–2=64，L3–5=128，L6–8=320。

## 2. 引擎事实（`0c556536/ar25.py`，与对照一致）

- 棋盘逻辑坐标约 **21×21**（非知识文档里的 8×8；渲染帧 64×64 为放大+UI）。
- 竖轴 tag `0054kgxrvfihgm`：`x' = 2ax − x`；横轴 `0002nuguepuujf`：`y' = 2ay − y`。
- 反射闭包：deque BFS，`ythhvclqmk = 12`；`visited` 去重；单轴反射标签过滤。
- 胜利：`vplrhaovhr()` → 合成覆盖图 `naxbskjmlg()` 上每个目标 `fswikrcrdmx` 格 `>= 0`。
- 可动块 tag `0006lxjtqggkmi`；固定 `0056icpryeujyf`；选中 `yvifanjrcyu`；旋转距离表 `ovoizfolxfq`。
- ACTION6/7 不扣步；ACTION5 扣步且切换选中。

## 3. 可落代码规格（R2 契约）

### 3.1 `encoder.py`（Layer 0）

输出结构化观测（不要只吐像素）：

```text
Ar25Obs:
  grid_w, grid_h
  targets: list[(x,y)]
  pieces:  list[{id, x, y, pixels_hash, tags, fixed}]
  axes:    list[{id, kind: V|H, x, y, fixed}]
  selected_id: optional
  steps_left: int
  rotate_dist: dict[id -> int]   # 对应 ovoizfolxfq
```

验收：L1 静态帧能稳定抽出目标点数与轴 kind；与本地引擎 reset 后对象计数一致。

### 3.2 `field.py`（Layer 1 世界模型）

必须实现（可对引擎做薄封装，但接口要纯函数化）：

```text
reflect_closure(pieces, axes, bounce_limit=12) -> set[(x,y)]
cover_report(pieces, axes, targets) -> {covered, uncovered, ok}
apply_config(obs, config) -> obs'          # 只改位姿，处理轴约束
estimate_path_cost(obs, config) -> int     # Σ|Δ| + SELECT 次数下界
is_futile(obs, config, steps_left) -> bool # cost 下界 > steps_left 则剪
```

反射实现要点（对齐引擎）：

1. 每个可反射块的非透明像素入队 depth=0。  
2. 对每条轴按 kind 生成镜像点；尊重 `0038pnuzypawco` / `reflect_horizontal_only`。  
3. `depth > 12` continue；坐标可出界但仍继续弹跳（引擎行为）。  
4. **缓存键**：`(frozenset(piece_pose), frozenset(axis_pose), pixels_version)`。

### 3.3 配置搜索（规划侧调用 field；R2 提供原语）

推荐搜索序（来自 `ar25_solver.solve_level`，并补齐联合枚举）：

```text
1. 若 cover(初始) OK → 空动作
2. 枚举可动轴坐标（V: x∈[0,W)，H: y∈[0,H)）；固定轴跳过
3. 仍无解：枚举可动块 (x,y)（可先轴对称候选 / 目标反投影，再全盘）
4. L7+：联合枚举 轴×块，但用剪枝：
   - 剩余未覆盖数单调不增才扩展（或允许临时变差的小 beam）
   - cost_lb = manhattan_moves + needed_selects > steps_left → 剪
   - 对称规范化：双轴配置按 (vx, hy) 排序去重
5. 每个 cover-OK 配置：引擎回放 pathify；失败则丢弃（防 L7 碰撞假解）
```

**禁止**：以「动作 DFS 深度上限」作为 L7 主策略。  
**允许**：pathify 阶段用短 BFS（状态=选中+各物体位姿），因配置已固定，分支远小于开放搜索。

### 3.4 假解防护（从 Lingjing 已踩坑提炼）

回放 / 线上执行前校验：

- `state_key` ⊇ 全部 sprite 位姿 + 像素（旋转后）+ `selected` + `ovoizfolxfq`  
- 每关求解前 `env.reset()` 隔离  
- 配置解必须在**同一引擎实例**逐步执行后 `levels_completed` +1 才算数

## 4. 落地里程碑（建议验收）

| 级别 | 标准 | 证据 |
|------|------|------|
| 及格 | `field.cover_report` 与引擎 `vplrhaovhr` 在 L1–L3 初始/已知解上一致 | 单测 |
| 良好 | 配置枚举复现 ARC-SAGE L7 锚点（或等价 cover-OK 配置）+ 本地回放通关 | 本地 log |
| 优秀 | L7 线上 scorecard WIN | scorecard id |

捷径（工程）：可直接移植 `reference/arcsage-ar25/solvers/ar25_solve_v2.py` 的 L7/L8 罐头做线上冲分，同时并行把配置枚举产品化——罐头证明可达，枚举证明可泛化。

## 5. 本地参考路径

```text
reference/arcsage-ar25/
  environment_files/ar25/0c556536/{ar25.py,metadata.json}
  solvers/ar25_solve_v2.py      # 8 关罐头 + 配置注释
  solvers/ar25_solver.py        # 源码解析 + 轴/块枚举骨架
  mechanics/ar25.md             # 机制说明（网格尺寸以引擎为准）
```

上游全文：https://github.com/dp-web4/ARC-SAGE  

## 6. 给 R1 的同步要点

1. NVARC/MindsAI ≠ AR25 DFS；已改对照 ARC-SAGE。  
2. L7 正解范式是 **配置枚举 + 反射闭包(12) + 引擎回放**，不是动作树深搜。  
3. 仓库已放入引擎摘录与规格；缺的是把旧求解器从 Best-First 切到本规格，或先罐头冲 L7 scorecard。
