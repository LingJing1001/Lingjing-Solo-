"""Kaggle ARC-AGI-3 Harness 适配层

官方 Starter 的接口只有两个方法：is_done(frames, latest_frame) / choose_action(frames, latest_frame)。
此适配器把 LingjingSoloAgent 包装成官方要求的 Agent 子类，让框架可一键放进：
    agent/my_agent.py

使用：
    from lingjing_solo.harness.kaggle_adapter import MyAgent
    # 在 Notebook 里继承 MyAgent 即可提交
"""
import numpy as np
from ..agent import LingjingSoloAgent
from ..core import SoloConfig, canonicalize


def to_game_action(name: str):
    """abstract action name → 引擎枚举；只在边界 adapter 里做（团队规范 §3.2 兼容要求）。

    转换逻辑原先在 `lingjing_solo/agent.py:_emit`，于是 planner 的返回类型取决于
    `arcengine` 装没装上。放在这里之后：planner 恒返回字符串，只有对接官方 harness 的
    这一层会要枚举；`arcengine` 不可用时原样返回名字（本地跑单测/无引擎环境要能用）。
    """
    key = canonicalize(name)
    try:
        from arcengine import GameAction
    except Exception:
        return key
    action = getattr(GameAction, key, None)
    if action is not None:
        return action
    if hasattr(GameAction, "from_name"):
        try:
            return GameAction.from_name(key)
        except Exception:
            pass
    return key


class MyAgent(LingjingSoloAgent):
    """Kaggle 评测入口：仅实现 is_done / choose_action，可改这一个文件。

    若官方 Starter 提供 Agent 基类，可改为 `class MyAgent(Agent, LingjingSoloAgent)`；
    这里保持框架无关，运行时动态适配。
    """
    def __init__(self, cfg=None, **kwargs):
        # 评测期无网络：默认不注入 LLM，走纯轻量路线
        if cfg is None:
            defaults = dict(
                llm_calls_per_game=0,      # 无网络时 LLM 预算归零，强制轻量规划
                enable_undo=False,
                return_game_action=True,   # 官方 harness 要枚举；转换在本层，不在 planner
            )
            defaults.update(kwargs)        # 调用方显式给的优先，别让关键字撞车
            cfg = SoloConfig(**defaults)
        super().__init__(cfg=cfg)

    # ---- 官方接口 1 ----
    def is_done(self, frames, latest_frame):
        return super().is_done(frames, latest_frame)

    # ---- 官方接口 2 ----
    def choose_action(self, frames, latest_frame, valid_actions=None):
        action = super().choose_action(frames, latest_frame, valid_actions=valid_actions)
        # planner 只给抽象名；这里才是"名字 → 引擎枚举"的边界
        return to_game_action(action) if self.cfg.return_game_action else action


def make_agent(llm_fn=None, cfg=None, **cfg_kwargs) -> MyAgent:
    """本地/带 LLM 调试时用；评测时直接实例化 MyAgent()。"""
    if cfg is None:
        cfg = SoloConfig(**cfg_kwargs)
    agent = MyAgent(cfg=cfg)
    if llm_fn is not None:
        agent.llm.inject_llm(llm_fn)
    return agent


# ---- 兼容官方 Starter 的帧格式约定 ----
def frames_to_grids(frames, latest_frame):
    """把 harness 传入的 frames 统一转成 numpy 网格列表。"""
    out = []
    for f in (frames or []):
        g = LingjingSoloAgent._to_grid(f)
        if g is not None:
            out.append(g)
    last = LingjingSoloAgent._to_grid(latest_frame)
    if last is not None and (not out or not np.array_equal(out[-1], last)):
        out.append(last)
    return out
