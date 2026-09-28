# 冲榜整合_SmartRouter_v1

智能路由提交包：已跑通的方法走 INLINE，未知环境走 CEAX，禁止纯 CEAX 无地板。

## 快速结论

- 榜上 0 分 ≠ 本地不会玩；是 **Kaggle 交付/无地板事故**（见 `docs/DIRECTION.md`）。
- 本包锁定 **ls20 / ar25 / ft09 = 100**，预估练习等价 ≈ **12.55**，目标先脱离 0、再往 18 挖未知局。

## 目录

```
agent/my_agent.py     # SmartRouter（类名必须 MyAgent）
lingjing_solo/        # 直连 CeaxController 依赖
bench/                # 最近摸底 JSON
docs/DIRECTION.md     # 方向短文
docs/冲榜技术说明与四人分工.md  # 分工给 3 位伙伴（主文档）
scripts/smoke_floors.py
scripts/sync_into_starter.ps1
```

## 本地冒烟（三地板）

在已装好 `.venv` 的 Starter 里跑（本包复用其环境）：

```powershell
cd "f:\2026年工作文件\2026年比赛的项目\ARC奖2026 - ARC-AGI-3\冲榜整合_SmartRouter_v1"
..\ARC-AGI-3-Kaggle-Starter\.venv\Scripts\python.exe scripts\smoke_floors.py
```

期望日志：`route=INLINE:ls20|ar25|ft09` 且三局均为 `WIN`。

## 同步进 Starter 并提交

```powershell
.\scripts\sync_into_starter.ps1
cd ..\ARC-AGI-3-Kaggle-Starter
.\.venv\Scripts\python.exe scripts\build_notebook.py
# 再按 README 用 kaggle kernels push + Submit to Competition
```

Phase B 必须出现：

```
smart-router-v1+inline-ls20x7-ar25x8-ft09x6+ceax
MAIN_EXIT 0
```
