"""Lingjing-Solo 全局配置 —— 钱学森灵境转义超参。

设计原则：
1. RHAE 步数经济（少废步）
2. 面积律：高精度算力只花在泡壁清晰区
3. LLM 退居 L4 战略顾问
"""
from dataclasses import dataclass
from .actions import DEFAULT_SIMPLE


@dataclass
class SoloConfig:
    # ---------- 网格与环境 ----------
    grid_size: int = 64
    num_colors: int = 16

    # ---------- L0/L3 感知与泡壁 ----------
    cnn_feature_dim: int = 128
    delta_threshold: int = 1
    bubble_pad: int = 3                 # 泡壁扩张像素
    phi_block_size: int = 8             # Φ 粗粒分块边长（64/8=8 → 8×8 块）
    fog_explore_bonus: float = 0.8      # 迷雾高时鼓励探索
    curvature_explore_bonus: float = 0.5

    # ---------- L1/L2 世界模型场 ----------
    field_max_transitions: int = 5000
    field_max_rules: int = 128
    state_hash_history: int = 2000
    predict_majority_ratio: float = 0.6
    include_levels_in_hash: bool = True
    source_term_history: int = 256      # ΔJ 历史容量

    # ---------- L2 探索 ----------
    probe_max_steps: int = 2
    rule_confidence_init: float = 0.5
    rule_confidence_inc: float = 0.15
    rule_confidence_dec: float = 0.25
    rule_confidence_min: float = 0.05
    loop_action_penalty: float = 2.5
    noop_penalty: float = 1.5
    progress_bonus: float = 3.0
    unexplored_bonus: float = 1.2
    effect_ema_bonus: float = 1.5         # 高实测因果强度动作加分
    effect_ema_alpha: float = 0.25        # ΔJ 实测 EMA 平滑系数
    rha_penalty_base: float = 0.8           # RHA 平方惩罚基数
    rha_penalty_power: float = 2.0          # RHA 指数（默认平方）
    top_n_hypotheses: int = 8               # explorer 输出 Top-N 规则

    # ---------- 泡壁点击（ACTION6 / 面积律）----------
    enable_mouse: bool = True             # ARC 多数关需要点击
    enable_click_sweep: bool = True
    click_stall_threshold: int = 4        # 连续无变化 ≥N → 强制点泡壁
    click_warmup: int = 6                 # 早期探针点击上限
    click_grid_step: int = 4              # 泡壁内稀疏网格步长
    click_queue_max: int = 48
    click_max_consecutive: int = 3        # 最多连续点几次就改回方向键
    click_probe_every: int = 3            # 早期每 N 次 stall 探针一次
    click_cooldown: int = 8               # 点空后冷却步数

    # ---------- L3 规划 / 面积律预算 ----------
    # ---------- L3b 顾问（开放 LLM / 封闭符号）----------
    use_llm_advisor: bool = False           # False=Kaggle/封闭离线；True=联网 LLM 顾问
    llm_calls_per_game: int = 8
    symbolic_advisor_max_rounds: int = 6    # 封闭模式：每局最多扩增轮次
    symbolic_max_hypotheses: int = 12       # 每轮符号猜想上限（强剪枝）
    symbolic_combo_depth: int = 2           # 算子/动作组合深度

    lightweight_search_depth: int = 8
    enable_astar: bool = True               # L3a：A* 启发式搜索
    astar_info_gain_weight: float = 1.0     # A* 启发：信息增益权重
    plan_replan_every: int = 4
    human_baseline_estimate: int = 30
    budget_warn_ratio: float = 0.30
    hard_step_multiplier: float = 5.0

    # ---------- L4 自指监视（C 值）----------
    loop_detect_window: int = 6
    reflection_min_interval: int = 5
    c_phase_threshold: float = 0.72     # C 超此值触发战略反思
    c_history_len: int = 32
    rule_prune_threshold: float = 0.12  # 反思时清掉低于此置信度的规则

    # ---------- L6 CEAX Transfer（跨关/跨游戏技能迁移）----------
    enable_transfer: bool = True
    transfer_min_confidence: float = 0.75
    transfer_cf_blend: float = 0.55       # 反事实分与 explore 分融合权重
    transfer_priority_before_explore: bool = True  # 探索兜底前插入迁移建议

    # ---------- 其他 ----------
    enable_undo: bool = False
    # enable_mouse / enable_click_sweep 见上方「泡壁点击」
    seed: int = 42
    return_game_action: bool = True
    agent_id: str = "lingjing-etherealrealm-solo"

    def __post_init__(self):
        self.allowed_actions = list(DEFAULT_SIMPLE)
        if self.enable_undo:
            self.allowed_actions.append("ACTION7")
        if self.enable_mouse or self.enable_click_sweep:
            if "ACTION6" not in self.allowed_actions:
                self.allowed_actions.append("ACTION6")
