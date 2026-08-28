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
from ..core import SoloConfig


class MyAgent(LingjingSoloAgent):
    """Kaggle 评测入口：仅实现 is_done / choose_action，可改这一个文件。

    若官方 Starter 提供 Agent 基类，可改为 `class MyAgent(Agent, LingjingSoloAgent)`；
    这里保持框架无关，运行时动态适配。
    """
    def __init__(self, cfg=None, **kwargs):
        # 评测期无网络：默认不注入 LLM，走纯轻量路线
        if cfg is None:
            cfg = SoloConfig(
                llm_calls_per_game=0,      # 无网络时 LLM 预算归零，强制轻量规划
                enable_undo=False,
                **kwargs,
            )
        super().__init__(cfg=cfg)

    # ---- 官方接口 1 ----
    def is_done(self, frames, latest_frame):
        return super().is_done(frames, latest_frame)

    # ---- 官方接口 2 ----
    def choose_action(self, frames, latest_frame, valid_actions=None):
        return super().choose_action(frames, latest_frame, valid_actions=valid_actions)


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
