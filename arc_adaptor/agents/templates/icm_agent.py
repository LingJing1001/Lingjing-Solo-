"""ICM 内在动机 agent（深度接入版）—— 挂载在项目读图栈上的「探索头」。

定位
----
本模板把 ICM 探索机制**接到项目既有的读图栈**上，而不是另起一套：

  读图（与主链同一条路径）
    extract_grid / extract_state / extract_levels / extract_available_actions
      -> Frame
    PerceptionEncoder.perceive(prev, cur, ...)   -> PerceptionSnapshot
          .feature(128) .objects .delta_pixels .bubble .phi .curvature_proxy
    LevelStability.observe(grid)                 -> raw/board/hud delta, board_changed
    rerank_click_candidates(objects, shape, ...) -> 点击候选（chrome 降权、内部点）
    normalize_observation(...)                   -> NormalizedObservation（R4 审计契约）

  ICM（本文件新增的部分）
        观察向量 = 感知特征 ⊕ 标量摘要  ->  φ(MLP)  ->  forward / inverse 模型
        内在奖励 = ||fwd(φ(s), a) − φ(s')||²，**用 board_changed 门控**
        外在奖励 = levels_completed 增量
        策略     = 同一观察向量上的 A2C；ACTION6 的坐标不再用 4096 维头，
                   而是在 rerank_click_candidates 给出的 K 个候选点 + 「全域随机」之间选

相对「自己写一套读图」的版本，这里改掉的是三件具体的事：

  1. **内在奖励不再被 HUD 污染。** 原版把「原始像素是否变化」当新奇。但 ARC-AGI-3 的帧带
     常驻 HUD（计时器/步数/动画条），这些像素自己会动。`perception/stability.py` 记录了这个
     坑，`LevelStability.board_changed` 就是它的解药 —— 本模板用它门控内在奖励：
     棋盘没变（只有 HUD 在动）的转移拿 0 而非正的新奇分。
  2. **对象级点击。** ACTION6 的候选点来自 `rerank_click_candidates`（把背景/边框/条带这类
     chrome 降权，并用对象内部点而非 bbox 中心），策略只在这 K 个点里选。
  3. **状态/关卡/合法动作统一走 core 的提取器**，不再手工解析 payload。

它是**独立策略**：不参与 my_agent.py 的 SmartRouter 主链，不读任何预烘计划表，也不接
CEAX/R3 路由；适合作为「闭卷通用探索」路线的对照基线单独跑分。

依赖
----
torch >= 2.2（``pip install lingjing-solo[cnn]``）。torch 是可选依赖，因此本模块
**不要**在 ``arc_adaptor/agents/__init__.py`` 里做顶层 import —— 那会把 torch
不必要地拉进每个 Phase B 运行路径。

跑分
----
    cd integrations/ARC-AGI-3-Kaggle-Starter
    env -u NODE_OPTIONS -u ACC_PRODUCT_CONFIG_V3 .venv/Scripts/python.exe \\
      scripts/benchmark_all_games.py 400 \\
      --agent "<abs>/arc_adaptor/agents/templates/icm_agent.py" \\
      --out state/bench_icm_perc_400.json --label icm-perc-400
"""
from __future__ import annotations

import logging
import math
import os
import random
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from arcengine import FrameData, GameAction, GameState


def _bootstrap_src_path() -> None:
    """让 lingjing_solo 可 import。

    本文件有两种加载方式：作为 ``agents.templates`` 包的一部分（相对导入可用），
    或被 ``benchmark_all_games.py`` 用 ``spec_from_file_location`` 按路径加载
    （此时没有父包，必须靠 sys.path）。这里把仓库根补进 sys.path，两种方式都能跑。
    """
    import sys

    candidates: list[Path] = []
    configured = os.environ.get("LINGJING_SRC")
    if configured:
        p = Path(configured).expanduser().resolve()
        candidates.append(p.parent if p.name == "lingjing_solo" else p)
    here = Path(__file__).resolve()
    for up in (3, 4):
        if len(here.parents) > up:
            candidates.append(here.parents[up])
    candidates.append(Path.cwd())
    for c in candidates:
        try:
            if (c / "lingjing_solo").is_dir():
                s = str(c)
                if s not in sys.path:
                    sys.path.insert(0, s)
                return
        except OSError:
            continue


_bootstrap_src_path()

from lingjing_solo.core import (  # noqa: E402
    Frame,
    SoloConfig,
    extract_available_actions,
    extract_grid,
    extract_levels,
    extract_state,
    is_win_state,
    needs_reset,
)
from lingjing_solo.perception import (  # noqa: E402
    ActionRecord,
    LevelStability,
    PerceptionEncoder,
    normalize_observation,
    rerank_click_candidates,
)

try:                                                   # 包内导入（ARC checkout）
    from ..agent import Agent
except ImportError:                                    # 按路径加载（benchmark 脚本）
    from agents.agent import Agent                     # type: ignore[no-redef]

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
except ImportError as exc:  # pragma: no cover - 环境缺 torch 时给出可操作提示
    raise ImportError(
        "ICMAgent 需要 torch>=2.2；请安装 lingjing-solo[cnn] 或直接 pip install torch"
    ) from exc


# ----------------------------------------------------------------- 常量
NUM_ACTION_TYPES = 7                       # ACTION1..ACTION7（ACTION6 是坐标点击）
ACTION6_TYPE = 5                           # ACTION6 在 0 基索引里的位置
ACTION_VEC_DIM = NUM_ACTION_TYPES + 2      # 7 个动作 one-hot + 归一化 (x, y)
MAX_CANDS = 8                              # 点击候选上限（来自 rerank_click_candidates）
NUM_COORD_SLOTS = MAX_CANDS + 1            # 末位 = 全域均匀采样的兜底选项
ACTION_HISTORY_LEN = 8                     # 写进 R4 契约的动作历史长度

_CANON_INDEX = {f"ACTION{i}": i - 1 for i in range(1, NUM_ACTION_TYPES + 1)}


def _canon_index(name: Any) -> "int | None":
    """规范动作名 -> 0 基索引；非 ACTION* 或越界返回 None。"""
    c = str(name).strip().upper()
    idx = _CANON_INDEX.get(c)
    if idx is not None:
        return idx
    if c.startswith("ACTION"):
        try:
            i = int(c[len("ACTION"):]) - 1
        except ValueError:
            return None
        if 0 <= i < NUM_ACTION_TYPES:
            return i
    return None


class Cfg:
    """超参数集中在此，便于 A/B 对照实验。"""

    feat_dim = 128             # ICM 的 φ 维度
    hidden = 256               # MLP 隐层宽度
    lr_policy = 3e-4
    lr_icm = 1e-3
    gamma = 0.99
    n_step = 16                # 每收集这么多步做一次更新
    batch_size = 24            # 单次更新所需的最少 transition
    rollout_len = 32           # 单次更新最多采样多少条 transition
    w_ext = 1.0                # 外在奖励权重
    w_int = 0.05               # 内在奖励权重（已做 running 归一化）
    beta_inv = 0.2             # inverse model 损失权重
    beta_fwd = 1.0             # forward model 损失权重
    ent_coef = 0.01            # 熵奖励系数（正数 = 鼓励探索）
    vf_coef = 0.5              # 价值损失权重
    grad_clip = 1.0
    int_norm_warmup = 20       # 内在奖励归一化预热步数
    max_seconds = 8 * 3600 - 300


# ----------------------------------------------------------------- 读图栈
class ReadOut:
    """一次读图的完整结果（供决策与审计）。

    刻意不用 ``@dataclass``：``benchmark_all_games._load_agent`` 用
    ``spec_from_file_location`` 加载本文件但**不把模块注册进 sys.modules**，
    此时 dataclass 装饰器会在 ``sys.modules[cls.__module__]`` 上抛
    AttributeError。普通类没有这个约束。
    """

    __slots__ = ("frame", "snapshot", "stability", "feature", "candidates", "shape")

    def __init__(self, frame: Frame, snapshot: Any, stability: dict,
                 feature: np.ndarray, candidates: list, shape: tuple) -> None:
        self.frame = frame
        self.snapshot = snapshot
        self.stability = stability
        self.feature = feature                      # (obs_dim,) float32，ICM/策略的输入
        self.candidates = candidates                # [(row, col)]，点击候选
        self.shape = shape


class PerceptionStack:
    """项目读图栈的统一入口。

    与主链 ``lingjing_solo/agent.py`` 走的是同一条路径：``extract_*`` -> ``Frame``，
    再 ``PerceptionEncoder.perceive`` -> ``PerceptionSnapshot``。额外挂上
    ``LevelStability``（HUD/棋盘分离）与 ``rerank_click_candidates``（对象级点击点），
    并把结果归一化成 R4 契约留在 :attr:`observation` 供复核。
    """

    def __init__(self, cfg: SoloConfig, game_id: str = "unknown") -> None:
        self.cfg = cfg
        self.game_id = game_id
        self.encoder = PerceptionEncoder(cfg)
        self.perc_dim = int(getattr(cfg, "cnn_feature_dim", 128) or 128)
        # 观察向量 = [归一化感知特征(perc_dim)] + [13 个标量摘要]
        self.obs_dim = self.perc_dim + 13
        self.observation = None                    # 最近一次的 NormalizedObservation
        self.backend = "unknown"
        self.reset()

    # -- 生命周期 ------------------------------------------------------------
    def reset(self) -> None:
        self._prev_grid = None
        self._version = 0
        self._levels = None
        self._stab_shape = None
        self.stability = LevelStability((self.cfg.grid_size, self.cfg.grid_size))
        self._actions: deque = deque(maxlen=ACTION_HISTORY_LEN)
        self.observation = None

    def _ensure_stability(self, shape: tuple) -> None:
        """网格尺寸变化（换关/换局）时重建稳定性估计。"""
        if self._stab_shape != tuple(shape):
            self.stability = LevelStability(tuple(shape))
            self._stab_shape = tuple(shape)

    # -- 主入口 --------------------------------------------------------------
    def read(self, latest_frame: FrameData, step: int) -> "ReadOut | None":
        """把一帧 harness payload 变成 ReadOut；网格不可用时返回 None。"""
        grid = extract_grid(latest_frame)
        if grid is None:
            return None
        grid = np.asarray(grid)
        if grid.ndim != 2 or grid.size == 0:
            return None

        state = extract_state(latest_frame)
        levels = extract_levels(latest_frame)
        legal = extract_available_actions(latest_frame)
        frame = Frame(
            grid=grid,
            t=step,
            state=state,
            levels_completed=levels,
            available_actions=list(legal),
            raw=latest_frame,
        )

        if self._levels is not None and levels != self._levels:
            self.stability.reset()
        self._levels = levels
        self._ensure_stability(grid.shape)

        stab = self.stability.observe(grid)

        prev = self._prev_grid
        if prev is not None and prev.shape != grid.shape:
            prev = None                            # 尺寸变了：差分无意义，按首帧处理
        snap = self.encoder.perceive(
            prev, grid,
            version=self._version,
            tick=step,
            levels=levels,
            env_state=state,
        )
        if getattr(snap, "feature", None) is not None and self.backend == "unknown":
            self.backend = getattr(getattr(self.encoder, "_cnn", None), "backend", "none")

        cands = self._click_candidates(snap, stab, grid.shape)
        feat = self._feature(snap, stab, grid.shape)

        self.observation = self._normalise(grid, prev, legal, step)
        self._prev_grid = grid
        self._version += 1
        return ReadOut(frame=frame, snapshot=snap, stability=stab,
                       feature=feat, candidates=cands, shape=tuple(grid.shape))

    # -- R4 审计契约 ---------------------------------------------------------
    def _normalise(self, grid, prev, legal, step):
        try:
            return normalize_observation(
                grid,
                frame_id=f"{self.game_id}:{step}",
                previous_frame=prev,
                action_history=tuple(self._actions),
                legal_actions=tuple(legal),
                game_family="keyboard_click" if "ACTION6" in legal else "keyboard",
            )
        except Exception:                          # noqa: BLE001 - 契约失败不该拖停决策
            return None

    def record_action(self, name: str, coord) -> None:
        """记下刚发出的动作（outcome 待下一帧回填）。"""
        x = y = None
        if coord is not None:
            y, x = int(coord[0]), int(coord[1])     # coord 是 (row, col)
        self._actions.append(ActionRecord(action=str(name), x=x, y=y, outcome=None))

    def note_outcome(self, outcome: str) -> None:
        """回填最近一条动作的实测结果。"""
        if not self._actions:
            return
        last = self._actions[-1]
        if last.outcome is None:
            self._actions[-1] = ActionRecord(
                action=last.action, x=last.x, y=last.y, outcome=str(outcome)
            )

    # -- 点击候选 ------------------------------------------------------------
    def _click_candidates(self, snap, stab, shape) -> list:
        """对象级点击候选：chrome 降权 + 对象内部点；空则退回画面中心。"""
        cands: list = []
        objs = getattr(snap, "objects", None) or []
        if objs:
            settled = bool(stab.get("settled"))
            try:
                cands = rerank_click_candidates(
                    objs, shape,
                    k=MAX_CANDS,
                    animated=self.stability.animated_mask() if settled else None,
                    changed=self.stability.changed_ever() if settled else None,
                    settled=settled,
                )
            except Exception:                      # noqa: BLE001
                cands = []
        if not cands:
            h, w = shape
            cands = [(h // 2, w // 2)]
        return [(int(r), int(c)) for r, c in cands][:MAX_CANDS]

    # -- 观察向量 ------------------------------------------------------------
    def _feature(self, snap, stab, shape) -> np.ndarray:
        """感知特征（L2 归一化）⊕ 标量摘要，长度恰为 obs_dim。"""
        raw = getattr(snap, "feature", None)
        feats = np.zeros(self.perc_dim, dtype=np.float32)
        if raw is not None:
            arr = np.asarray(raw, dtype=np.float32).reshape(-1)
            n = min(arr.size, self.perc_dim)
            feats[:n] = arr[:n]
        feats = feats / (float(np.linalg.norm(feats)) + 1e-8)

        h, w = int(shape[0]), int(shape[1])
        area = float(max(1, h * w))
        objs = getattr(snap, "objects", None) or []
        areas = np.asarray([len(o.pixels) for o in objs] or [0], dtype=np.float32)
        bubble = getattr(snap, "bubble", None)
        phi = getattr(snap, "phi", None)
        scalars = np.asarray(
            [
                len(objs) / 16.0,
                float(areas.max()) / area,
                float(areas.mean()) / area,
                float(len(getattr(snap, "delta_pixels", None) or [])) / area,
                float(getattr(snap, "curvature_proxy", 0.0) or 0.0),
                float(stab.get("raw_delta", 0)) / area,
                float(stab.get("board_delta", 0)) / area,
                float(stab.get("hud_delta", 0)) / area,
                1.0 if stab.get("board_changed") else 0.0,
                1.0 if stab.get("settled") else 0.0,
                float(getattr(bubble, "fog_ratio", 0.0) or 0.0),
                float(getattr(bubble, "wall_entropy", 0.0) or 0.0),
                float(getattr(phi, "mean", 0.0) or 0.0) if phi is not None else 0.0,
            ],
            dtype=np.float32,
        )
        return np.concatenate([feats, scalars]).astype(np.float32)


# ----------------------------------------------------------------- 网络
class MLP(nn.Module):
    """两层 MLP，探索头/策略主体共用的骨架。"""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = Cfg.hidden) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):                                  # noqa: D102
        return self.net(x)


class PolicyNet(nn.Module):
    """Actor-Critic：动作头(7) + 坐标候选头(MAX_CANDS+1) + 价值头(1)。"""

    def __init__(self, obs_dim: int, hidden: int = Cfg.hidden) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden, NUM_ACTION_TYPES)
        self.coord_head = nn.Linear(hidden, NUM_COORD_SLOTS)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):                                  # noqa: D102
        z = self.trunk(x)
        return (self.action_head(z),
                self.coord_head(z),
                self.value_head(z).squeeze(-1),
                z)


class ICM(nn.Module):
    """Intrinsic Curiosity Module（Pathak et al. 2017 的骨架）。

    encoder phi      : 观察向量 -> 特征空间，由 inverse model 的梯度塑造
    inverse model    : 由 (phi(s), phi(s')) 预测动作，筛出「可控」的特征维度
    forward model    : 由 (phi(s), a) 预测 phi(s')，预测误差即内在奖励

    内在奖励因此是【状态级】且【连续】的，并随模型学会而自然衰减 —— 这正是原版
    二值「画面是否变化」缺少的性质。是否「有效」则由读图栈的 board_changed 判定。
    """

    def __init__(self, obs_dim: int, feat_dim: int = Cfg.feat_dim) -> None:
        super().__init__()
        self.encoder = MLP(obs_dim, feat_dim)
        self.inverse = nn.Sequential(
            nn.Linear(feat_dim * 2, Cfg.hidden), nn.ReLU(),
            nn.Linear(Cfg.hidden, NUM_ACTION_TYPES),
        )
        self.forward_model = nn.Sequential(
            nn.Linear(feat_dim + ACTION_VEC_DIM, Cfg.hidden), nn.ReLU(),
            nn.Linear(Cfg.hidden, feat_dim),
        )

    def intrinsic(self, obs, obs_next, a_vec, a_type):
        """返回 (r_i, inv_loss, fwd_loss)；r_i 形状 (B,)，逐样本。"""
        phi = self.encoder(obs)
        phi_next = self.encoder(obs_next)
        phi_next_d = phi_next.detach()                     # 目标端断梯度，训练更稳

        pred_next = self.forward_model(torch.cat([phi, a_vec], dim=-1))
        r_i = (pred_next - phi_next_d).pow(2).mean(dim=-1)
        fwd_loss = r_i.mean()

        inv_logits = self.inverse(torch.cat([phi, phi_next_d], dim=-1))
        inv_loss = F.cross_entropy(inv_logits, a_type)
        return r_i, inv_loss, fwd_loss


# ----------------------------------------------------------------- Agent
class ICMAgent(Agent):
    """ICM 探索头 + 项目读图栈（独立策略，不接主链）。"""

    MAX_ACTIONS = float("inf")
    VARIANT = "icm-perception"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1_000_000) + hash(self.game_id) % 1_000_000
        random.seed(seed)
        np.random.seed(seed % (2 ** 32 - 1))
        torch.manual_seed(seed % (2 ** 32 - 1))
        self.start_time = time.time()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 日志目录是**可选**的：默认不创建任何目录。bench 会把 agent 实例化 25 次，
        # 若无条件 mkdir，就会在仓库里留下一堆空目录。需要落盘时显式设 LINGJING_RUN_DIR。
        self.run_dir = None
        run_root = os.getenv("LINGJING_RUN_DIR")
        if run_root:
            self.run_dir = Path(run_root) / datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"ICMAgent.{self.game_id}")

        self.cfg = SoloConfig()
        self.perc = PerceptionStack(self.cfg, game_id=str(self.game_id))
        self.obs_dim = self.perc.obs_dim
        self.logger.info(
            "%s | device=%s | obs_dim=%d (perception %d + scalars 13) | run_dir=%s",
            self.VARIANT, self.device, self.obs_dim, self.perc.perc_dim, self.run_dir,
        )

        self.policy = PolicyNet(self.obs_dim).to(self.device)
        self.icm = ICM(self.obs_dim).to(self.device)
        self.opt_policy = optim.Adam(self.policy.parameters(), lr=Cfg.lr_policy)
        self.opt_icm = optim.Adam(self.icm.parameters(), lr=Cfg.lr_icm)

        self.pending: deque = deque(maxlen=8)
        self.buffer: deque = deque(maxlen=4096)

        # 注意：基类也会自增 action_counter，这里用独立计数器避免双重计数
        self.step_count = 0

        self._ri_count = 0
        self._ri_mean = 0.0
        self._ri_m2 = 0.0
        self.level = 0

    # ---------------------------------------------------------- 框架钩子
    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        try:
            if is_win_state(latest_frame):
                return True
        except Exception:                                  # noqa: BLE001
            pass
        return self._time_is_up()

    def _time_is_up(self) -> bool:
        return (time.time() - self.start_time) >= Cfg.max_seconds

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        try:
            return self._choose_action(latest_frame)
        except Exception as exc:                           # noqa: BLE001
            self.logger.warning("choose_action crashed: %r", exc)
            return self._fallback(latest_frame, repr(exc))

    # ---------------------------------------------------------- 动作构造
    def _fallback(self, latest_frame: FrameData, why: str) -> GameAction:
        legal = []
        try:
            legal = extract_available_actions(latest_frame)
        except Exception:                                  # noqa: BLE001
            pass
        name = legal[0] if legal else ""
        action = self._make_action(name) or GameAction.RESET
        action.reasoning = {"source": self.VARIANT, "why": "fallback", "error": why}
        return action

    @staticmethod
    def _make_action(name: str) -> "GameAction | None":
        if not name:
            return None
        try:
            return GameAction.from_name(name)
        except (KeyError, ValueError):
            return None

    def _build_action(self, a_type: int, coord, legal: list) -> GameAction:
        """把动作索引/坐标翻成合法 GameAction；不合法则回退到首个合法动作。"""
        name = f"ACTION{a_type + 1}"
        if name not in legal:
            name = legal[0] if legal else ""
        action = self._make_action(name)
        if action is None:
            return GameAction.RESET
        if action.is_complex() and coord is not None:
            row, col = int(coord[0]), int(coord[1])
            action.set_data({"game_id": self.game_id, "x": col, "y": row})
        return action

    # ---------------------------------------------------------- 奖励
    def _extrinsic_reward(self, levels_completed: int) -> float:
        """外在奖励：通关一次给 1.0（原版这条通道是死的，恒为 0）。"""
        if levels_completed != self.level:
            gained = levels_completed > self.level
            self.level = levels_completed
            if gained:
                self.logger.info("level up -> %d", levels_completed)
                return 1.0
        return 0.0

    def _normalise_intrinsic(self, r: float) -> float:
        """running 均值/方差归一化；否则内在奖励整体漂移，w_int 调不动。"""
        self._ri_count += 1
        delta = r - self._ri_mean
        self._ri_mean += delta / self._ri_count
        self._ri_m2 += delta * (r - self._ri_mean)
        if self._ri_count < Cfg.int_norm_warmup:
            return 0.0
        var = self._ri_m2 / max(1, self._ri_count - 1)
        std = math.sqrt(max(var, 1e-12))
        return (r - self._ri_mean) / (std + 1e-8)

    # ---------------------------------------------------------- 动作向量
    def _action_vec(self, a_type: int, coord, shape) -> torch.Tensor:
        v = torch.zeros(ACTION_VEC_DIM, dtype=torch.float32)
        v[a_type] = 1.0
        if a_type == ACTION6_TYPE and coord is not None:
            h, w = int(shape[0]), int(shape[1])
            row, col = int(coord[0]), int(coord[1])
            v[NUM_ACTION_TYPES] = col / max(1, w - 1)
            v[NUM_ACTION_TYPES + 1] = row / max(1, h - 1)
        return v

    # ---------------------------------------------------------- 掩码
    def _mask_for(self, legal: list) -> torch.Tensor:
        """合法动作 -> 7 类 logits 的加性掩码。"""
        mask = torch.full((NUM_ACTION_TYPES,), float("-inf"), device=self.device)
        for name in legal:
            idx = _canon_index(name)
            if idx is not None:
                mask[idx] = 0.0
        if torch.isinf(mask).all():
            mask = torch.zeros(NUM_ACTION_TYPES, device=self.device)
        return mask

    def _coord_mask(self, n: int) -> torch.Tensor:
        """前 n 个候选点 + 末位「全域随机」可用，其余置 -inf。"""
        mask = torch.full((NUM_COORD_SLOTS,), float("-inf"), device=self.device)
        if n > 0:
            mask[:n] = 0.0
        mask[MAX_CANDS] = 0.0
        return mask

    # ---------------------------------------------------------- 主决策
    def _choose_action(self, latest_frame: FrameData) -> GameAction:
        self.step_count += 1
        read = self.perc.read(latest_frame, self.step_count)
        if read is None:
            return self._fallback(latest_frame, "no-grid")

        if needs_reset(read.frame.state):
            self.pending.clear()
            self.perc.note_outcome("reset")
            reset = GameAction.RESET
            reset.reasoning = {"source": self.VARIANT, "why": "reset"}
            return reset

        legal = list(read.frame.available_actions)
        if not legal:
            return self._fallback(latest_frame, "no-legal-actions")

        obs = torch.from_numpy(read.feature).to(self.device)

        # ---- 1) 结算上一步：门控后的内在奖励 + 外在奖励 ----
        if self.pending:
            entry = self.pending.popleft()
            r_ext = self._extrinsic_reward(read.frame.levels_completed)
            with torch.no_grad():
                a_vec = self._action_vec(
                    entry["a_type"], entry["coord"], entry["shape"]
                ).unsqueeze(0).to(self.device)
                a_type_t = torch.tensor([entry["a_type"]], device=self.device)
                r_i_raw = float(
                    self.icm.intrinsic(
                        entry["obs"].unsqueeze(0), obs.unsqueeze(0), a_vec, a_type_t
                    )[0].item()
                )
            r_norm = self._normalise_intrinsic(r_i_raw)
            # 棋盘没动（只有 HUD 在动）的转移不给正向新奇 —— 这是接入读图栈的意义
            no_effect = bool(read.stability.get("settled")) and not bool(
                read.stability.get("board_changed")
            )
            entry["obs_next"] = obs
            entry["reward"] = Cfg.w_ext * r_ext + Cfg.w_int * (0.0 if no_effect else r_norm)
            entry["r_int_raw"] = r_i_raw
            entry["no_effect"] = no_effect
            self.buffer.append(entry)
            self.perc.note_outcome(
                "no_effect" if no_effect else ("changed" if read.stability.get("board_changed") else "pending")
            )

        # ---- 2) 选新动作：掩码后 softmax 采样 ----
        mask = self._mask_for(legal)
        with torch.no_grad():
            a_logits, c_logits, value, _ = self.policy(obs.unsqueeze(0))
            a_dist = torch.distributions.Categorical(logits=a_logits.squeeze(0) + mask)
            a_type = int(a_dist.sample().item())
            logp = float(a_dist.log_prob(torch.tensor(a_type, device=self.device)).item())

            slot = -1
            coord = None
            if a_type == ACTION6_TYPE:
                cmask = self._coord_mask(len(read.candidates))
                c_dist = torch.distributions.Categorical(logits=c_logits.squeeze(0) + cmask)
                slot = int(c_dist.sample().item())
                logp += float(c_dist.log_prob(torch.tensor(slot, device=self.device)).item())
                coord = self._coord_from_slot(slot, read)
            value = float(value.item())

        action = self._build_action(a_type, coord, legal)
        action.reasoning = {
            "source": self.VARIANT,
            "a_type": a_type,
            "p": round(math.exp(logp), 3),
            "v": round(value, 3),
            "cands": len(read.candidates),
            "slot": slot,
            "board": bool(read.stability.get("board_changed")),
        }
        self.perc.record_action(action.name, coord)

        self.pending.append({
            "obs": obs,
            "a_type": a_type,
            "a_slot": slot,
            "coord": coord,
            "shape": read.shape,
            "val": value,
        })

        # ---- 3) 定期更新 ----
        if self.step_count % Cfg.n_step == 0:
            self._update()

        return action

    def _coord_from_slot(self, slot: int, read: ReadOut):
        """候选槽 -> (row, col)；末位槽 = 全域均匀采样。"""
        if 0 <= slot < len(read.candidates):
            return read.candidates[slot]
        h, w = int(read.shape[0]), int(read.shape[1])
        return (random.randrange(h), random.randrange(w))

    # ---------------------------------------------------------- 学习
    def _update(self) -> None:
        if len(self.buffer) < Cfg.batch_size:
            return
        n = len(self.buffer)
        idx = np.random.choice(n, min(n, Cfg.rollout_len), replace=False)
        batch = [self.buffer[i] for i in idx]

        obs = torch.stack([b["obs"] for b in batch]).to(self.device)
        obs_next = torch.stack([b["obs_next"] for b in batch]).to(self.device)
        a_type = torch.tensor([b["a_type"] for b in batch], device=self.device)
        a_slot = torch.tensor([b["a_slot"] for b in batch], device=self.device)
        val_old = torch.tensor([b["val"] for b in batch], device=self.device)
        reward = torch.tensor([b["reward"] for b in batch], device=self.device)
        a_vec = torch.stack([
            self._action_vec(b["a_type"], b["coord"], b["shape"]) for b in batch
        ]).to(self.device)

        # 回报：片段内折扣累加（简化版 n-step return）
        returns = torch.zeros_like(reward)
        running = 0.0
        for i in reversed(range(len(batch))):
            running = float(reward[i].item()) + Cfg.gamma * running
            returns[i] = running
        adv = returns - val_old
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # --- 策略 + 价值（A2C）---
        a_logits, c_logits, value, _ = self.policy(obs)
        dist = torch.distributions.Categorical(logits=a_logits)
        logp = dist.log_prob(a_type)
        is6 = a_type == ACTION6_TYPE
        if bool(is6.any()):
            c_dist = torch.distributions.Categorical(logits=c_logits[is6])
            extra = torch.zeros_like(logp)
            extra[is6] = c_dist.log_prob(a_slot[is6].clamp(min=0))
            logp = logp + extra

        policy_loss = -(logp * adv.detach()).mean()
        value_loss = F.mse_loss(value, returns)
        entropy = dist.entropy().mean()                    # 真正的策略熵
        loss = policy_loss + Cfg.vf_coef * value_loss - Cfg.ent_coef * entropy

        self.opt_policy.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), Cfg.grad_clip)
        self.opt_policy.step()

        # --- ICM：特征空间由这里塑造 ---
        _, inv_loss, fwd_loss = self.icm.intrinsic(obs, obs_next, a_vec, a_type)
        icm_loss = Cfg.beta_inv * inv_loss + Cfg.beta_fwd * fwd_loss
        self.opt_icm.zero_grad()
        icm_loss.backward()
        nn.utils.clip_grad_norm_(self.icm.parameters(), Cfg.grad_clip)
        self.opt_icm.step()

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        if self.step_count % (Cfg.n_step * 10) == 0:
            n_noop = sum(1 for b in batch if b.get("no_effect"))
            self.logger.info(
                "step=%d pi=%.4f v=%.4f H=%.3f inv=%.4f fwd=%.4f r_int=%.4f noop=%d/%d",
                self.step_count, float(policy_loss), float(value_loss), float(entropy),
                float(inv_loss), float(fwd_loss),
                float(np.mean([b["r_int_raw"] for b in batch])), n_noop, len(batch),
            )

        self.buffer.clear()


# ----------------------------------------------------------------- bench 入口
# benchmark_all_games.py 用 spec_from_file_location 加载本文件，并取 mod.MyAgent
# （_load_agent 里写死了这个名字），所以这里必须暴露 MyAgent。
MyAgent = ICMAgent
