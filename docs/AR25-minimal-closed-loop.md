# AR25 最小可运行闭环

## 目标

把 AR25 的 Learning Template 接成一个真实可验证的最小闭环：

```text
observe → learner 提议 → official engine replay → 关卡推进验证 → 保存解
                         ↘ 失败时 official engine bounded search
```

## 本分支范围

- `Ar25Learner` 继续负责候选配置枚举、`pathify` 和 live replay。
- 几何候选全部被 live engine 拒绝时，`arc_shadow.learn_solve_level()` 回退到已有的 official-engine bounded search。
- 回退搜索前强制 `env.reset()`，保证搜索起点与后续 replay 起点一致。
- 只有检测到 `_current_level_index` 推进或真实 `WIN`，才返回路径并写入 solution。

## 当前验收结果

- [x] 纯 learner simulator：L0–L2 通过。
- [x] live offline AR25：`learn_solve_level(max_configs=128)` 推进 L1。
- [x] fallback 路径由真实 ARC engine replay 验证后返回。
- [x] `git diff --check` 和 AR25 模块 `py_compile` 通过。
- [ ] learner 几何路径本身在 live engine 上稳定通过；当前仍依赖 fallback。
- [ ] live L2/L3 以上使用本闭环连续推进。
- [ ] 修复仓库已有 `lingjing_solo/exploration/__init__.py:11` 语法错误后再跑全量 pytest。

## 后续顺序

1. 为 `learn_solve_level` 增加可注入的 engine/replay 单元测试，覆盖候选失败、reset、fallback、真实推进判定。
2. 将 live engine 的动作/坐标转移记录下来，修正 `pathify` 与真实碰撞语义的差异。
3. 在 offline engine 上连续验证 L1–L3，并记录每关 path、节点数、耗时和最终 level index。
4. 只有连续多关通过后，才把 fallback 从“安全兜底”升级为主要求解路径。
