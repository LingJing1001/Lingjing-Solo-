# 0 分事故复盘与 fix0 修复

**事故提交**：ref `56051461`，`CEAX unknown v3.6 no plugins`，publicScore **0.00**  
**修复 BUILD_TAG**：`ceax-v3.6+inline-ls20ar25-fix0`

## 根因（按证据排序）

1. **交付链风险（高）**  
   Phase B 里原先 **先 `import MyAgent` 再改写 `agents/__init__.py`**。  
   竞赛包上游 `__init__` 会急切 import langgraph/smolagents 等，易导致校验失败或半初始化。  
   虽然 `!python main.py` 是子进程，但仍属脆弱顺序；fix0 改为 **先写瘦身 `__init__.py`，再用 in-process assert + `subprocess.run` 跑 main，失败即抬例外**。

2. **导入面过大（高）**  
   `from lingjing_solo.transfer import CeaxController` 会经 `transfer/__init__.py` 拉齐 alea/spectral/neural/unified。  
   任一模块在 Kaggle 环境异常 → 整 agent 起不来 → 全场 0。  
   fix0：**直连** `lingjing_solo.transfer.ceax_controller.CeaxController`。

3. **纯 CEAX 无 ls20 地板（中，已实证）**  
   本地复现：纯 CEAX 打 `ls20` @120 步 **0 关**；加 INLINE 后 **7/7 WIN**。  
   历史 publicScore **0.15** 指纹来自 INLINE/partial，不是 PluginRegistry。  
   公开评测若含练习集 id，纯 CEAX 会接近 0；若全是隐藏局，本地 focus8≈2.26 **不能**直接当 public 预期。

4. **无崩溃护栏（中）**  
   Swarm 每局一线程；`choose_action` 抛错会杀掉该局。fix0 全路径 try/except，永不空返回。

5. **预期分数误解（说明）**  
   focus8≈2.26 是 **本地 8 个练习局均值**，不是 Kaggle public 公式下的必然结果。  
   隐藏 ~110 局里练习局可能不出现；纯未知迁移失败时 public 可以是 0.00（与 `docs/隐藏110局分数推演与改进方案.md` 悲观情景一致）。

## fix0 改动

| 项 | 做法 |
|----|------|
| Agent | CEAX v3.6 + INLINE ls20 L1–L7 / ar25 L1（**无 PluginRegistry**） |
| Import | 直连 `ceax_controller`；`_norm_game` 内联 |
| Notebook | 先改写 `__init__.py` → CEAX_CHECK → `subprocess.run(main.py)` |
| Reasoning | 一律 `{"text": ...}` dict，兼容 gateway |
| 指纹 | `/kaggle/working/BUILD_TAG.txt` + 日志 `CEAX_CHECK ...` |

## 本地验收（fix0）

```
ls20  levels=7  WIN   (INLINE)
lp85  levels=1        (CEAX)
su15  levels=1        (CEAX)
三局 aggregate ≈ 34.8
```

## 提交

```powershell
cd ARC-AGI-3-Kaggle-Starter
.\scripts\submit.ps1 -Message "CEAX v3.6 + INLINE ls20/ar25 fix0 (no plugins)"
```

Phase B 日志必须出现：`CEAX_CHECK ceax-v3.6+inline-ls20ar25-fix0` 与 `MAIN_EXIT 0`。  
若仍为 0.00 且日志有指纹 → 问题在隐藏局零迁移，不是交付链。
