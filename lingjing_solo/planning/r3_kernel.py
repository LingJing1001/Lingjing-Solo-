"""统一 R3 内核（§3.2.1 / §7.1.1）。

R3Kernel 是文档 §3.2.1 要求的统一内核：消费 WorldView + SearchBackend，
经 T0–T3 分层搜索和 P1–P4 剪枝闸门，产出经 plan_contract.build_plan() 校验的 plan。

三条硬约束（§3.2.1）：
    1. 线上没有引擎对象：通过 SearchBackend.is_online 隔离，内核不接触引擎。
    2. 线上动作不可撤销：搜索是"预测 → 执行一步 → 校验 → 错误后重规划"。
    3. 先压缩搜索空间再搜动作：有 config_space() 时先搜配置层再 pathify。

R3 专项门槛对应：
    R3-A 线上边界  → backend.is_online 时禁止快照
    R3-B 双层搜索  → config_space() 非空时先配置层再动作层
    R3-C 候选与剪枝 → P1–P4 闸门
    R3-D 统一路由  → registry.resolve_planner 末端接入
    R3-E 分层证据  → plan validity + backend.evidence_tier
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Hashable, Sequence

from .world_view import WorldView, ActionSpec, ConfigSpace, action_edge_key
from .backends import SearchBackend
from .plan_contract import build_plan, register_planner


# ── P1–P4 剪枝契约 ──────────────────────────────────────────────

class PruningContract(str, Enum):
    """§3.2.1 统一剪枝契约。"""
    P1_NOOP = "p1_noop"                # key(child) == key(parent) 时丢弃
    P2_COUNTER_DOMINANCE = "p2_counter_dominance"  # 同 key 只保留计数器更优节点
    P3_DEATH = "p3_death"              # GAME_OVER 或关键资源下降时停止
    P4_BUDGET = "p4_budget"            # g + h <= budget，超预算 fail-closed


@dataclass(frozen=True)
class PruningViolation:
    """一次剪枝违规记录，供审计。"""
    contract: PruningContract
    reason: str
    state_hash: Hashable
    action: str
    divergence: str = ""


# ── 搜索预算 ────────────────────────────────────────────────────

@dataclass(frozen=True)
class SearchBudget:
    """R3 搜索预算（P4 闸门用）。

    同时受动作数和墙钟限制；超预算必须 fail-closed。
    """
    max_steps: int = 400
    max_nodes: int = 128000
    max_depth: int = 60
    time_ms: int = 30000

    def __post_init__(self) -> None:
        for name, val in (("max_steps", self.max_steps), ("max_nodes", self.max_nodes),
                          ("max_depth", self.max_depth), ("time_ms", self.time_ms)):
            if not isinstance(val, int) or val <= 0:
                raise ValueError(f"SearchBudget.{name} 必须是正整数，收到 {val}")


# ── R3Kernel ────────────────────────────────────────────────────

class R3Kernel:
    """统一 R3 内核：T0–T3 分层搜索 + P1–P4 剪枝闸门。

    用法：
        wv = AR25WorldView()           # 游戏专用 WorldView
        backend = LearnedGraphBackend() # 或 EngineBackend（离线）
        kernel = R3Kernel(wv, backend, budget=SearchBudget(max_steps=400))
        plan = kernel.plan(frame, legal_actions, state_hash)
    """

    def __init__(
        self,
        world_view: WorldView,
        backend: SearchBackend,
        *,
        budget: SearchBudget | None = None,
        planner_name: str = "generic",
        script_bank_fn: Any = None,
    ):
        self.wv = world_view
        self.backend = backend
        self.budget = budget or SearchBudget()
        self.planner_name = planner_name
        # T0 罐头路线回调：(game_id, level) -> list[str] | None
        self._script_bank_fn = script_bank_fn
        # 本 tick 的剪枝违规记录
        self.violations: list[PruningViolation] = []
        # 本 tick 的搜索统计
        self._stats: dict[str, Any] = {}

    # ── 主入口 ─────────────────────────────────────────────

    def plan(
        self,
        frame: Any,
        legal_actions: Sequence[str],
        state_hash: str,
        *,
        game_id: str = "",
        level: int = 0,
    ) -> dict[str, Any]:
        """生成一个 plan，经 plan_contract.build_plan() 出口。

        T0–T3 分层尝试，第一个非空的层胜出。每层经 P1–P4 闸门过滤。
        全部失败时返回 validity="none" 的空 plan（fail-closed）。
        """
        self.violations.clear()
        self._stats = {"layer_tried": [], "nodes": 0, "elapsed_ms": 0}
        t0 = time.monotonic()

        grids = self.wv.grids(frame)
        wv_key = self.wv.key(grids)
        counters = self.wv.counters(grids)

        # P3 死亡检查：dead frame 直接 fail-closed
        if self.wv.dead(frame):
            self.violations.append(PruningViolation(
                contract=PruningContract.P3_DEATH,
                reason="frame is dead (GAME_OVER or resource exhausted)",
                state_hash=state_hash,
                action="",
                divergence="dead frame cannot produce a plan",
            ))
            return self._build_none_plan(state_hash, "p3_death")

        # done 检查：已通关则不需要动作
        if self.wv.done(frame):
            return self._build_none_plan(state_hash, "already_done")

        candidate_actions: list[str] = []
        expected_goal = "unknown"
        cost: int | None = None
        validity = "candidate"
        evidence_refs: list[str] = []

        # ── T0 罐头路线 ────────────────────────────────────
        if self._script_bank_fn is not None:
            t0_actions = self._script_bank_fn(game_id, level)
            if t0_actions:
                # 罐头路线是已验证的，validity=verified_offline
                candidate_actions = [a for a in t0_actions if a in set(legal_actions)]
                if candidate_actions:
                    expected_goal = f"t0_script:{game_id}_L{level}"
                    validity = "verified_offline"
                    evidence_refs = [f"script_bank:{game_id}/L{level}"]
                    self._stats["layer_tried"].append("T0")
                    self._stats["t0_hit"] = True

        # ── T1 游戏专用 planner（WorldView.proposals）─────────
        if not candidate_actions:
            proposals = self.wv.proposals(grids, wv_key)
            # 过滤合法动作 + P1 noop 检查
            filtered = self._filter_proposals(
                proposals, wv_key, legal_actions, state_hash,
            )
            if filtered:
                candidate_actions = [p.name for p in filtered]
                expected_goal = f"t1_proposals:{self.planner_name}"
                self._stats["layer_tried"].append("T1")

        # ── T2 通用内核双层搜索 ──────────────────────────────
        if not candidate_actions:
            t2_actions = self._t2_search(grids, wv_key, legal_actions, state_hash, t0)
            if t2_actions:
                candidate_actions = t2_actions
                expected_goal = "t2_kernel:config_then_action"
                self._stats["layer_tried"].append("T2")

        # ── T3 探索兜底 ──────────────────────────────────────
        if not candidate_actions:
            t3_actions = self._t3_explore(grids, wv_key, legal_actions, state_hash, t0)
            if t3_actions:
                candidate_actions = t3_actions
                expected_goal = "t3_explore:info_gain"
                self._stats["layer_tried"].append("T3")

        # ── P4 预算检查 ──────────────────────────────────────
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        self._stats["elapsed_ms"] = elapsed_ms
        if elapsed_ms > self.budget.time_ms:
            self.violations.append(PruningViolation(
                contract=PruningContract.P4_BUDGET,
                reason=f"wall clock {elapsed_ms}ms > budget {self.budget.time_ms}ms",
                state_hash=state_hash,
                action="",
                divergence="timeout",
            ))
            return self._build_none_plan(state_hash, "p4_timeout")

        if not candidate_actions:
            return self._build_none_plan(state_hash, "no_candidates")

        # ── 经 plan_contract.build_plan() 出口 ───────────────
        search_budget = {
            "time_ms": self.budget.time_ms,
            "max_nodes": self.budget.max_nodes,
            "max_depth": self.budget.max_depth,
        }
        return build_plan(
            planner=self.planner_name,
            input_state_hash=state_hash,
            candidate_actions=candidate_actions,
            expected_goal=expected_goal,
            cost=cost,
            search_budget=search_budget,
            validity=validity,
            evidence_refs=evidence_refs,
            legal_actions=list(legal_actions),
        )

    # ── P1–P4 闸门 ──────────────────────────────────────────

    def _filter_proposals(
        self,
        proposals: list[ActionSpec],
        parent_key: Hashable,
        legal_actions: Sequence[str],
        state_hash: str,
    ) -> list[ActionSpec]:
        """对 proposals 做 P1 noop + 合法性过滤。"""
        legal_set = set(legal_actions)
        result: list[ActionSpec] = []
        for prop in proposals:
            # 合法性检查
            if prop.name not in legal_set:
                continue
            # P1 noop：预测的 child key == parent key → 丢弃但记录
            # 边键带点击坐标：只用 prop.name 时，同状态下任何一次点击都查同一条边，
            # noop 判据与"点哪里"无关（#9 坐标盲区）。
            edge_key = action_edge_key(prop.name, prop.payload_hint)
            predicted = self.backend.predict(state_hash, edge_key)
            if predicted is not None:
                # 用 WorldView 重建 key 做 noop 判断
                # 注意：predicted 是 state hash，不是 grids；这里用 hash 直接比
                # 完整的 P1 需要后端返回 grids，但后端只暴露 hash
                # 退而求其次：如果后端 predict 返回与当前相同的 hash，判为 noop
                if predicted == state_hash:
                    self.violations.append(PruningViolation(
                        contract=PruningContract.P1_NOOP,
                        reason=f"predict({edge_key}) returns same state hash",
                        state_hash=state_hash,
                        action=prop.name,
                        divergence=f"noop: key(child) == key(parent) edge={edge_key!r}",
                    ))
                    continue
            result.append(prop)
            self._stats["nodes"] += 1
            # P4 节点预算
            if self._stats["nodes"] > self.budget.max_nodes:
                self.violations.append(PruningViolation(
                    contract=PruningContract.P4_BUDGET,
                    reason=f"nodes {self._stats['nodes']} > max_nodes {self.budget.max_nodes}",
                    state_hash=state_hash,
                    action=prop.name,
                    divergence="node budget exhausted",
                ))
                break
        return result

    def _t2_search(
        self,
        grids: list[Any],
        wv_key: Hashable,
        legal_actions: Sequence[str],
        state_hash: str,
        t0: float,
    ) -> list[str]:
        """T2 双层搜索：配置层 → 动作层（R3-B）。"""
        config_space = self.wv.config_space(grids)
        if config_space is None:
            return []

        # 配置层搜索：找目标终态配置
        # 这里只做最简的 BFS：从当前配置找 goal_configs
        # 完整实现需要 config → action 的 pathify，这里返回空让 T3 兜底
        # （避免在未校验的配置上宣称找到路径）
        self._stats["config_space_nodes"] = len(config_space.nodes)
        return []

    def _t3_explore(
        self,
        grids: list[Any],
        wv_key: Hashable,
        legal_actions: Sequence[str],
        state_hash: str,
        t0: float,
    ) -> list[str]:
        """T3 探索兜底：信息增益候选 + bounded probe。"""
        # 选一个合法动作做探针（最简单的兜底）
        # 完整实现用 R4 explorer 的 information_gain 排序
        legal_list = [a for a in legal_actions if a and a != "RESET"]
        if not legal_list:
            return []
        # P4 预算检查
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        if elapsed_ms > self.budget.time_ms:
            self.violations.append(PruningViolation(
                contract=PruningContract.P4_BUDGET,
                reason=f"t3 explore timeout {elapsed_ms}ms",
                state_hash=state_hash,
                action="",
                divergence="timeout",
            ))
            return []
        return [legal_list[0]]

    # ── fail-closed 空 plan ──────────────────────────────────

    def _build_none_plan(self, state_hash: str, reason: str) -> dict[str, Any]:
        """validity=none 的空 plan（fail-closed）。"""
        return build_plan(
            planner=self.planner_name,
            input_state_hash=state_hash,
            candidate_actions=[],
            expected_goal=f"none:{reason}",
            cost=None,
            search_budget={
                "time_ms": self.budget.time_ms,
                "max_nodes": self.budget.max_nodes,
                "max_depth": self.budget.max_depth,
            },
            validity="none",
            evidence_refs=[],
        )

    # ── P2 计数器支配检查 ────────────────────────────────────

    def check_counter_dominance(
        self,
        key_a: Hashable,
        counters_a: dict[str, int],
        key_b: Hashable,
        counters_b: dict[str, int],
    ) -> bool:
        """P2：同 key 时 a 是否支配 b（a 的计数器全部 >= b）。

        Returns:
            True 如果 key_a == key_b 且 counters_a 全部 >= counters_b。
        """
        if key_a != key_b:
            return False
        return all(counters_a.get(k, 0) >= v for k, v in counters_b.items())

    # ── 统计 ────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        return dict(self._stats)


__all__ = [
    "PruningContract",
    "PruningViolation",
    "SearchBudget",
    "R3Kernel",
]
