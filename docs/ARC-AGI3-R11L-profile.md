# ARC-AGI-3 R11L Game Profile（P1.1）

> 目标游戏：`r11l-495a7899`
> 建立时间：2026-09-02（PDT）
> 状态：**初始 profile 已建立；路线尚未验证**

## 已验证事实

| 项目 | 结果 | 证据 |
|---|---|---|
| 环境元数据 | title=`R11L`、tag=`click`、`win_levels=6` 的基线元数据 | `uv run python` 调用 `Arcade.get_environments()`，返回 `game_id='r11l-495a7899' title='R11L' tags=['click'] baseline_actions=[22, 33, 51, 26, 52, 49]` |
| Frame | 单通道 64×64；当前样本 dtype=`int8`，像素范围 0–15 | P0 reset probe；命令见下方 |
| 合法动作 | reset 后返回一个合法动作，原始值为 `6`，对应 `GameAction.ACTION6`/复杂动作 | P0 reset probe |
| 坐标契约 | `ACTION6` 使用 display-space `x,y`，范围均为 0–63；camera 再映射至 grid 坐标 | `arcengine.camera.Camera.display_to_grid()`；`r11l.py:1819-1827` |
| reset | reset 返回 `NOT_FINISHED`、`levels_completed=0`，并创建新的 scorecard/guid | P0 reset probe |
| 单击响应 | `ACTION6(x=32,y=32)` 服务端回显相同坐标；旧 diff 统计的 row=0,col=0 来自 leading channel 与错误目标检测，不能作为坐标错位证据 | Scorecard `d7b265f4-1d7a-4368-8bf0-e5336203ad39`；probe exit 0 |
| 可点击像素 | 游戏源码将未选中的 `sys_click` sprites remap 为 colour 3；colour 6 是 UI 状态样本，不是稳定目标标记 | `r11l.py:1511-1518`；本地 environment source |

## 尚未验证

- `ACTION6` 的坐标是否被服务端正确解释；当前三组坐标 `(0,0)`、`(17,47)`、`(32,32)` 的返回 frame 均只显示 row=0,col=0 变化，尚不能证明坐标语义。
- 首关正确 click、重复 reset 的确定性、level transition、6 关路线、game-over 条件。
- `level_0.json` route artifact 和离线 replay。

## 可复现实验命令

```bash
cd /srv/agent-platform/projects/ARC-AGI-3-Agents
uv run python tools/r11l_single_action_probe.py --x 32 --y 32
uv run python tools/r11l_single_action_probe.py --x 17 --y 47
```

probe 工具会输出 reset 前后 frame hash、shape、state、levels、合法动作、changed cells 和变化 bbox；异常返回非零退出码。API key 只从运行环境读取，不写入文档。

## 当前架构决策

- `r11l-*` 已由 `GameStrategyRegistry` 路由到 `R11LStrategy`。
- `R11LStrategy` 当前只做观测驱动的候选 marker click，不宣称已经解决 R11L。
- adapter 已支持策略返回带坐标的 action request；既有 LS20/Generic 字符串 action 契约保持不变。
- 在坐标/transition 语义得到独立 recording 证明前，不创建固定路线。
