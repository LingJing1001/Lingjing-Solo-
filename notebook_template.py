"""Lingjing-Solo · Kaggle ARC-AGI-3 提交 Notebook 模板

使用方法：
1. 将整个 lingjing_solo/ 目录与本文件一起上传到 Kaggle Notebook
2. 在官方 Starter 中，把 agent/my_agent.py 改为：

       from notebook_template import MyAgent

3. 仅实现 is_done / choose_action 两个方法（官方接口约束）

注意：评测期【无网络】+【获奖须开源】，故默认 LLM 预算=0，走纯轻量路线；
      若需本地调试带 LLM，用 agent.LingjingSoloAgent.with_llm(...) 构造并注入调用函数。

本模板以 "## ---- CELL" 标记分块，可直接拆分粘贴进 Jupyter 单元格。
"""
import numpy as np
import sys, os

# 保证 lingjing_solo 包可被导入（Kaggle 工作目录）
sys.path.insert(0, os.path.dirname(__file__))

from lingjing_solo import SoloConfig
from lingjing_solo.harness import MyAgent, make_agent
from lingjing_solo.agent import LingjingSoloAgent


# %% ---- CELL 1: 配置（评测期无网络，LLM 预算归零）----
def build_config():
    return SoloConfig(
        grid_size=64,
        num_colors=16,
        cnn_feature_dim=128,
        llm_calls_per_game=0,        # 评测期无网络 → 纯轻量规划
        enable_undo=False,           # 默认禁用 UNDO（防滥用撤销浪费步数）
        enable_mouse=False,          # 默认仅方向键 + SPACE
        human_baseline_estimate=30,  # 人类步数预估（预算告急判定）
        loop_detect_window=6,
        lightweight_search_depth=6,
    )


# %% ---- CELL 2: 构造 Agent（单例，跨局由 harness 调用 reset）----
_cfg = build_config()

# 【可选】本地调试时注入 LLM 顾问（评测期请保持 None）
def _local_llm(snapshot, valid_actions):
    """占位：真实提交时替换为你的模型调用。"""
    return None

_agent = make_agent(llm_fn=None, cfg=_cfg)


# %% ---- CELL 3: 官方接口 —— 仅需改这一个文件 ----
class MyAgent(MyAgent):
    """直接继承 Kaggle 适配层 MyAgent，五层决策已内置。

    若官方 Starter 提供 Agent 基类（如 `from agent import Agent`），改为：
        class MyAgent(MyAgent, Agent): ...
    并把下面两个方法体保持为空（直接 super() 走父类实现）即可。
    """
    def is_done(self, frames, latest_frame):
        # 内置 WIN 检测 + 硬步数上限，避免无限循环拖低 RHAE 平方得分
        return super().is_done(frames, latest_frame)

    def choose_action(self, frames, latest_frame, valid_actions=None):
        # 内部走五层决策闭环：反思触发 → 轻量规划 → 探索评分 → 兜底随机
        return super().choose_action(frames, latest_frame, valid_actions)


# 评测入口：harness 会实例化 MyAgent()
Agent = MyAgent


# %% ---- CELL 4: 本地自检（make play-local 前可跑通）----
def _self_test():
    print("=" * 50)
    print("Lingjing-Solo self-test")
    a = make_agent(llm_fn=None, cfg=_cfg)
    a.reset()
    frames = []
    rng = np.random.RandomState(0)
    for t in range(20):
        grid = np.zeros((64, 64), dtype=np.int8)
        grid[10:14, 10:14] = 3
        grid[20:23, 20:25] = (t % 7) + 1
        frames.append(grid)
        act = a.choose_action(frames[:-1], grid)
        if a.is_done(frames, grid):
            break
    print(f"steps={a.step}, rules={len(a.field.rules)}, "
          f"visited={len(a.field.visited)}, llm_calls={a.llm.calls_used}")
    print("self-test OK ✅")


if __name__ == "__main__":
    _self_test()
