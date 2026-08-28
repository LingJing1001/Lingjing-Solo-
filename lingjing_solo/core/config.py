"""Lingjing-Solo 全局配置：所有可调超参集中于此。

设计原则：RHAE 步数经济优先 —— 任何超参的默认值都向着"少调 LLM、少走废步"倾斜。
"""
from dataclasses import dataclass, field


@dataclass
class SoloConfig:
    # ---------- 网格与环境 ----------
    grid_size: int = 64           # ARC-AGI-3 标准网格边长
    num_colors: int = 16          # 颜色通道数（one-hot 维度）

    # ---------- 感知层（Layer 0）----------
    cnn_feature_dim: int = 128    # CNN 编码输出特征维度
    delta_threshold: int = 1      # 像素变化绝对值阈值（判定"泡壁/ROI"）

    # ---------- 世界模型场（Layer 1）----------
    field_max_transitions: int = 5000     # 转移表容量上限，FIFO 淘汰
    field_max_rules: int = 128           # 规则假设集容量
    state_hash_history: int = 2000       # 已访问状态哈希集合容量（循环检测）

    # ---------- 探索引擎（Layer 2）----------
    probe_max_steps: int = 2             # 单次受控探测的最大步数
    rule_confidence_init: float = 0.5    # 新规则初始置信度
    rule_confidence_inc: float = 0.15    # 被证据支持时增幅
    rule_confidence_dec: float = 0.25    # 与证据冲突时降幅
    rule_confidence_min: float = 0.05    # 低于此值则淘汰

    # ---------- 规划执行（Layer 3）----------
    llm_calls_per_game: int = 8          # 每关 LLM 调用硬预算（RHAE 关键）
    lightweight_search_depth: int = 6    # 无 LLM 时 BFS/A* 最大搜索深度
    human_baseline_estimate: int = 30    # 人类步数预估（用于预算告急判定）
    budget_warn_ratio: float = 0.30      # 已用步数 / 人类预估 超过此值 → 告急

    # ---------- 反思触发器（Layer 4）----------
    loop_detect_window: int = 6          # 连续 N 步状态重复 → 判定循环
    reflection_min_interval: int = 5     # 两次 LLM 反思之间的最小步间隔

    # ---------- 其他 ----------
    enable_undo: bool = False            # 默认禁用 UNDO（避免 LLM 滥用撤销）
    enable_mouse: bool = False           # 默认禁用鼠标点击，仅用方向键
    seed: int = 42

    def __post_init__(self):
        # 动作空间：5 方向 + 可选 UNDO；鼠标独立
        self.allowed_actions = ["UP", "DOWN", "LEFT", "RIGHT", "SPACE"]
        if self.enable_undo:
            self.allowed_actions.append("UNDO")
