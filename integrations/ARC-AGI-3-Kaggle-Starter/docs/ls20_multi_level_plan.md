# LS20 从 1 关提升到多关 — 改代码清单

> 当前基线：`levels_completed=1`，Level 1 约第 14 步通关，之后在 Level 2 卡住。

## 怎样算成功（先看懂这个）

| 结果 | `levels_completed` | `state` | 含义 |
|------|-------------------|---------|------|
| **完全成功** | = 7（或 `win_levels`） | `WIN` | 7 关全通 |
| **部分成功** | 1～6 | `NOT_FINISHED` | 过了若干关，步数用完 |
| **失败** | 0 | `GAME_OVER` | 第一关都没过 |
| **进行中** | 任意 | `NOT_FINISHED` | 还在玩 |

本地验证命令：

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_ls20.py 800
```

看 JSON 里的 `levels_completed` 和 `level_step_milestones`。

---

## 根因分析（为什么卡在 1 关）

1. **Level 2 旋转台机制比「踩 3 次」更复杂**（实测）：
   - 到达 `(49,45)` 自动 +90°；再 **2 次** 左出右进 → `270°` 且 `match=True`
   - **`match=True` 时离台任意方向都会触发重置**（传回 `(29,35)`、旋转归零）
   - `(44,40)` 是陷阱格，踏入后下一步必重置
   - 因此 L2 需要 **专用路径脚本**（先(goal)后(rot)或安全南向绕路），不能仅靠通用 BFS
2. **过关后 `_layout_wait`**：已加 4 帧等待 + agent 侧 `ACTION1` 占位，避免 `reset_level` 被误触发。
3. **移动平台 / 推箱子**：Level 4+ 需要 `_bfs_push`，部分关卡路径被错误判为不可达。
4. **形状台 / 调色台顺序**：`mod_tasks` 顺序与关卡要求不一致时会先走错修饰台。
5. **专用 solver 绕过通用学习**：ls20 不积累因果图，只能靠改 `ls20_solver.py`。

---

## 改代码清单（按优先级）

### P0 — 必做（Level 2 通关）

| # | 文件 | 改什么 | 怎么验证 |
|---|------|--------|----------|
| 1 | `lingjing_solo/planning/ls20_solver.py` | `_estimate_rot_entries`：Level 2 固定返回 **2**（非 3） | benchmark 出现 `level_step_milestones: {1:..., 2:...}` |
| 1b | 同上 | 新增 `_l2_trap_cells` 避开 `(44,40)`；平台等待时在 `(14,35)` 用 `ACTION1` 占位 | L2 不再频繁横向漂移 |
| 2 | 同上 | `looks_like_ls20()` 独立函数（修复 ImportError） | benchmark 能正常 import |
| 3 | `lingjing_solo/agent.py` | `_layout_wait` 期间不 `reset_level`，用 `ACTION1` 等待 | Level 1→2 切换稳定 |

### P1 — Level 3～4（颜色 + 形状）

| # | 文件 | 改什么 |
|---|------|--------|
| 5 | `ls20_solver.py` | `_find_color_pad` / `_estimate_color_entries`：按目标格颜色特征估调色次数 |
| 6 | 同上 | `_find_shape_pad`：`shape_toggles` 与 `_queue_modifier` 中形状台退出再进入逻辑 |
| 7 | 同上 | `mod_tasks` 排序：有形状台时先 shape 再 rot 再 color（按关卡几何检测） |
| 8 | 同上 | `_use_push_nav` + `_bfs_push`：Level 4 推障碍路径，见 `probe_l2_platform.py` |

### P2 — Level 5～7（双目标 + 迷雾）

| # | 文件 | 改什么 |
|---|------|--------|
| 9 | `ls20_solver.py` | 多目标 `goal_idx`：完成一个后自动 `goal_idx+=1`，不要清空全部 goals |
| 10 | 同上 | `_goal_wait_mode` + `PLATFORM_CYCLE`：移动平台关卡等待相位对齐 |
| 11 | 同上 | 迷雾关：BFS 仍用当前可见 grid，不依赖全图；或扩大 `_find_player` 鲁棒性 |
| 12 | `lingjing_solo/core/config.py` | `include_levels_in_hash=True` 确保过关后状态哈希区分关卡 |

### P3 — 测试与回归

| # | 文件 | 改什么 |
|---|------|--------|
| 13 | `scripts/probe_l2_rotation.py` | 单独测 Level 2 旋转 3 次后能否到达目标 |
| 14 | `scripts/probe_l2_platform.py` | 测推箱子路径 |
| 15 | `scripts/benchmark_ls20.py` | 默认 `max_steps=800`，输出每关里程碑 |
| 16 | 新建 `lingjing_solo/test_ls20_levels.py` | 断言 `levels_completed >= 2`（CI 回归） |

---

## 关键函数速查

```
ls20_solver.py
├── reset_level()          # 每关初始化：找玩家/目标/修饰台
├── _estimate_rot_entries() # 旋转台要踩几次
├── _queue_modifier()      # 去修饰台并计数
├── _queue_next()          # 去目标格
├── observe()              # 根据环境反馈更新计数
└── plan()                 # 每步输出一个 ACTION
```

---

## 推荐开发循环

```powershell
# 1. 改 ls20_solver.py
# 2. 快速测 Level 2
.\.venv\Scripts\python.exe scripts\probe_l2_rotation.py

# 3. 全关 benchmark
.\.venv\Scripts\python.exe scripts\benchmark_ls20.py 800

# 4. 可视化看每一步
.\.venv\Scripts\python.exe scripts\visualize_ls20.py --max-steps 800 --serve-only --open
```

---

## 目标里程碑

| 阶段 | `levels_completed` | 说明 |
|------|-------------------|------|
| 当前 | 1 | Level 1 OK |
| 下一步 | ≥ 2 | Level 2 旋转 3 次 + 导航 |
| 中期 | ≥ 4 | 形状台 + 推箱子 |
| 目标 | 7 + WIN | Kaggle 该游戏满分 |
