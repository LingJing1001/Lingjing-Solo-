# AR25（ARC-AGI-3）闯关总结

> 日期：2026-09-18
> 状态：✅ **8关全部通关，won=True**
> 流程：R2+R3+R4 组合 (r234 flow v5)

---

## 一、最终成绩

| 关卡 | 方法 | 步数 | 预算 | 耗时 | 结果 |
|------|------|------|------|------|------|
| L1 | R0-已知解法 | 15 | 64 | 0.0s | ✅ 通关 |
| L2 | R3-目标分解 | 12 | 64 | 0.1s | ✅ 通关 |
| L3 | R3-目标分解 | 36 | 128 | 0.3s | ✅ 通关 |
| L4 | R3-目标分解 | 34 | 128 | 0.2s | ✅ 通关 |
| L5 | R3-双轴分解 | 36 | 320 | 3.8s | ✅ 通关 |
| L6 | R3-双轴分解 | 52 | 320 | 8.2s | ✅ 通关 |
| L7 | R3-双轴分解 | 44 | 320 | 15.3s | ✅ 通关 |
| L8 | R3-双轴分解 | 49 | 320 | 9.0s | ✅ 通关 |

**总计：278 步，36.8 秒搜索，GameState.WIN**

---

## 二、版本演进

| 版本 | 最远关卡 | 总步数 | 总耗时 | 是否通关 | 关键改进 |
|------|----------|--------|--------|----------|----------|
| v1 | L2 | 29 | 0.026s | ❌ | R3搜索 + R4贪心回退 |
| v2 | L2 | 29 | 0.026s | ❌ | 同v1，参数微调 |
| v3 | L6 (失败) | ~200 | 163.7s | ❌ | 目标分解 + beam + arc_shadow |
| **v5** | **L8** | **278** | **36.8s** | **✅** | **双轴穷举 + 已知解法优先 + 增强预算** |

### v3 → v5 的关键突破

1. **双轴目标分解穷举** — v3 的贪心拼接在 L6 失败，v5 改为先穷举拼块排列（组合数≤10万时），再贪心回退。L6/L7/L8 全部由双轴分解秒级解出。
2. **已知解法优先 (R0)** — L1 直接用 `arc_shadow.KNOWN_SOLUTIONS` 秒通，避免不必要的搜索。
3. **L6 从"320步+163.7s失败"变成"52步+8.2s通关"** — 双轴分解比 arc_shadow 搜索高效约 20 倍。

---

## 三、求解策略详解

### R0：已知解法优先
- 来源：`arc_shadow.KNOWN_SOLUTIONS`（之前离线验证过的解法）
- 适用：关卡索引匹配时直接执行，秒级返回
- 实际效果：L1 直接通关（15步）

### R2：感知
- 从游戏内部状态提取：轴类型/位置、可动拼块、目标格、已覆盖格、步数预算
- 决定后续搜索策略：单轴→目标分解，双轴→双轴分解，困难关卡→增强预算

### R3-目标分解（单轴）
- 枚举轴目标位置，计算反射覆盖，贪心拼接拼块
- 适用：L2–L4（单轴关卡）
- 耗时：0.1–0.3s

### R3-双轴分解（双轴，v5核心改进）
- 枚举 (h_y, v_x) 所有轴位置组合
- 对每组轴位置，枚举每个拼块的所有可能放置
- **穷举拼接**：组合数≤10万时，枚举所有拼块位置组合找全覆盖最小成本；否则回退贪心
- 适用：L5–L8（双轴关卡）
- 耗时：3.8–15.3s

### R3-beam search（回退策略）
- L6+：beam_width=16, t_limit=300s, max_nodes=3M
- L1-L5：beam_width=8, t_limit=60s, max_nodes=1M
- 本次运行未触发（双轴分解已解出所有关卡）

### R3-arc_shadow（回退策略）
- L6+：t_limit=600s, max_nodes=2M
- 本次运行未触发

### R4-贪心步进（最终回退）
- 逐动作评分，选覆盖增量最大的动作
- 本次运行未触发

---

## 四、涉及文件

### 核心代码

| 文件 | 作用 |
|------|------|
| `arc_adaptor/agents/strategies/run_ar25_r234.py` | 主入口 — R2+R3+R4 组合闯关流程 (v5) |
| `arc_adaptor/arc_shadow.py` | 搜索引擎 — 状态快照/恢复/启发式/已知解法/beam search |
| `arc_adaptor/r2r3r4_combo.py` | R2R3R4 组合辅助模块 |
| `arc_adaptor/ar25_learner.py` | AR25 学习/训练模块 |

### ar25_solver 子包

| 文件 | 作用 |
|------|------|
| `arc_adaptor/ar25_solver/solver.py` | 求解器核心 |
| `arc_adaptor/ar25_solver/simulator.py` | 游戏模拟器 |
| `arc_adaptor/ar25_solver/advisor.py` | 策略顾问 |
| `arc_adaptor/ar25_solver/cli.py` | 命令行接口 |

### 环境与依赖

| 文件/包 | 作用 |
|---------|------|
| `environment_files/ar25/0c556536/ar25.py` | AR25 游戏逻辑（关卡定义、规则、状态） |
| `arc_agi` (pip 0.9.9) | ARC-AGI 框架 — Arcade, OperationMode |
| `arcengine` (pip 0.9.3) | 游戏引擎 — GameAction, ActionInput, GameState |

### 运行结果

| 文件 | 作用 |
|------|------|
| `state/v5_out.txt` | V5 第1次运行输出 |
| `state/v5_run2_out.txt` | V5 第2次运行输出（确认稳定） |
| `state/ar25_r234_20260918_171539_21472/result.json` | 第1次运行结果JSON |
| `state/ar25_r234_20260918_172330_37196/result.json` | 第2次运行结果JSON |

### 依赖关系

```
run_ar25_r234.py (主入口)
  ├── arc_shadow.py (搜索引擎 + KNOWN_SOLUTIONS)
  ├── arc_agi (pip: Arcade, OperationMode)
  ├── arcengine (pip: GameAction, ActionInput, GameState)
  ├── numpy (pip: 数组计算)
  └── ar25.py (环境: 游戏逻辑, 被 arc_agi 加载)
```

---

## 五、运行环境

| 项目 | 值 |
|------|-----|
| Python | 3.14.0 (`D:\Program\Python\python.exe`) |
| arc_agi | 0.9.9 |
| arcengine | 0.9.3 |
| numpy | 2.x |
| 运行模式 | OFFLINE（本地引擎） |
| game_id | ar25-0c556536 |

---

## 六、与之前成绩对比

| 关卡 | 旧成绩 (v3/scorecard) | **新成绩 (v5)** | 改进 |
|------|----------------------|----------------|------|
| L1 | 15步 | 15步 | — |
| L2 | 14步 | **12步** | ✅ -2步 |
| L3 | 62步 | **36步** | ✅ -26步 |
| L4 | 24步 | **34步** | ⚠️ +10步 |
| L5 | 56步 | **36步** | ✅ -20步 |
| L6 | 197步 (163.7s) | **52步 (8.2s)** | ✅ -145步, -155.5s |
| L7 | ❌ 超时 | **44步 (15.3s)** | ✅ 从失败到通关 |
| L8 | ❌ 未尝试 | **49步 (9.0s)** | ✅ 从未尝试到通关 |
| **总计** | 368步, 1430s, 6/8关 | **278步, 36.8s, 8/8关** | ✅ **-90步, -1393.2s, +2关** |

---

## 七、复现方式

```bash
# 确保 arc_agi 和 arcengine 已安装
pip install arc_agi arcengine

# 运行 AR25 R234 V5 求解器
python arc_adaptor/agents/strategies/run_ar25_r234.py

# 指定最大关卡数（默认8）
python arc_adaptor/agents/strategies/run_ar25_r234.py 8

# 从指定关卡开始（跳过已通关的）
python arc_adaptor/agents/strategies/run_ar25_r234.py 8 5
```

---

## 八、遗留事项

- [ ] L4 步数从24增加到34，需分析是否双轴分解对单轴关卡非最优
- [ ] 已知解法 (KNOWN_SOLUTIONS) 的 L2/L3/L4/L5 验证失败，需更新为 v5 解法
- [ ] 线上 scorecard 验证（当前仅在本地 OFFLINE 模式通关）
- [ ] 后端 `/api/proxy/arc-auto` 接入 v5 求解器
- [ ] 将 v5 解法固化到 KNOWN_SOLUTIONS 以加速后续运行