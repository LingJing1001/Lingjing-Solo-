"""更新因果模型 —— 与 WorldModelField 同构的 (s, a, s') 因果图。

对局内：field.predict_graph / effect_ema 记录「动作→环境」。
工程侧：本模块记录「干预→提交结果」，供 R5 Layer 3b/4 反思与 LLM 顾问使用。
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


@dataclass
class CausalEdge:
    cause: str
    action: str
    effect: str
    weight: float = 1.0
    evidence: int = 1
    note: str = ""


@dataclass
class CausalHypothesis:
    id: str
    statement: str
    confidence: float
    status: str = "open"  # open | supported | contradicted | resolved
    linked_edges: list[str] = field(default_factory=list)


# 先验节点 / 干预 / 可观测效应（对应 C1–C6）
PRIOR_NODES = (
    "LOCAL_SCRIPT_OK",      # 本地 play 通关 / HARDCODED 有效
    "KERNEL_PUSHED",        # Phase A notebook push
    "SUBMIT_API_OK",        # competitions submit 成功
    "PHASE_B_BOUND",        # Phase B 绑到目标 kernel 版本
    "HARDCODED_LOADED",     # Phase B 日志出现 HARDCODED_CHECK / BUILD_TAG
    "MULTI_LEVEL_SCORE",    # 多关有效分（非≈1关）
    "SCORE_015",            # publicScore≈0.15 指纹
    "SCORE_ALIGNED",        # 与本地期望≈4.11 对齐
    "MISREPORT_SUBMITTED",  # 脚本误报已提交
    "QUOTA_BLOCK",          # CreateCodeSubmission 400
)

PRIOR_HYPOTHESES: list[CausalHypothesis] = [
    CausalHypothesis(
        "C1",
        "本地 ScriptBank/脚本有效 ≠ Phase B 一定加载同一代码",
        0.9,
        linked_edges=["LOCAL_SCRIPT_OK|assume_same_code|PHASE_B_BOUND"],
    ),
    CausalHypothesis(
        "C2",
        "kernels push（Phase A）≠ competitions submit 成功（Phase B）",
        0.95,
        linked_edges=["KERNEL_PUSHED|competitions_submit|SUBMIT_API_OK"],
    ),
    CausalHypothesis(
        "C3",
        "submit 脚本 API 失败仍打印 Submitted → 决策被误导",
        0.9,
        linked_edges=["QUOTA_BLOCK|misreport_success|MISREPORT_SUBMITTED"],
    ),
    CausalHypothesis(
        "C4",
        "publicScore≈0.15 ≈ 聚合后约 1 关有效贡献（脚本未充分生效指纹）",
        0.85,
        linked_edges=["PHASE_B_BOUND|run_partial_script|SCORE_015"],
    ),
    CausalHypothesis(
        "C5",
        "L1–L7 HARDCODED 已在 kernel，但 submit 400 → Phase B 从未跑本版",
        0.95,
        linked_edges=["KERNEL_PUSHED|competitions_submit|QUOTA_BLOCK"],
    ),
    CausalHypothesis(
        "C6",
        "榜上勾选/查看旧 Version → 看到的仍是旧因果结果，不是 L1–L7",
        0.9,
        linked_edges=["SCORE_015|select_old_version|SCORE_015"],
    ),
]


class UpdateCausalModel:
    """工程更新因果场：转移表 + predict_graph + effect_ema + 假设库。"""

    def __init__(self, *, max_transitions: int = 256) -> None:
        self.version = 0
        self.nodes: set[str] = set(PRIOR_NODES)
        self.transition_table: deque[CausalEdge] = deque(maxlen=max_transitions)
        self.predict_graph: dict[tuple[str, str], str] = {}
        self.effect_ema: dict[str, float] = defaultdict(float)
        self.pair_counts: Counter = Counter()
        self.hypotheses: dict[str, CausalHypothesis] = {
            h.id: CausalHypothesis(**h.__dict__) for h in PRIOR_HYPOTHESES
        }
        self.observed_states: set[str] = set()
        self._seed_prior_edges()

    def _seed_prior_edges(self) -> None:
        priors = [
            CausalEdge("LOCAL_SCRIPT_OK", "assume_same_code", "PHASE_B_BOUND", 0.2, 1, "C1 常不成立"),
            CausalEdge("KERNEL_PUSHED", "competitions_submit", "SUBMIT_API_OK", 0.5, 1, "C2 需成功才成立"),
            CausalEdge("SUBMIT_API_OK", "phase_b_rerun", "PHASE_B_BOUND", 0.9, 1, "绑定目标版本"),
            CausalEdge("PHASE_B_BOUND", "load_hardcoded", "HARDCODED_LOADED", 0.8, 1, "需日志确认"),
            CausalEdge("HARDCODED_LOADED", "run_l1_l7", "MULTI_LEVEL_SCORE", 0.85, 1, "期望路径"),
            CausalEdge("MULTI_LEVEL_SCORE", "aggregate_25", "SCORE_ALIGNED", 0.8, 1, "≈4.11"),
            CausalEdge("PHASE_B_BOUND", "run_partial_script", "SCORE_015", 0.9, 3, "C4 指纹"),
            CausalEdge("KERNEL_PUSHED", "competitions_submit", "QUOTA_BLOCK", 0.7, 2, "C5 实测"),
            CausalEdge("QUOTA_BLOCK", "misreport_success", "MISREPORT_SUBMITTED", 0.95, 1, "C3"),
            CausalEdge("SCORE_015", "select_old_version", "SCORE_015", 1.0, 2, "C6"),
            CausalEdge("LOCAL_SCRIPT_OK", "hardcode_l1_l7", "KERNEL_PUSHED", 0.9, 1, "整合进 notebook"),
        ]
        for e in priors:
            self._commit_edge(e, bump_version=False)
        self.version = 1

    def _commit_edge(self, edge: CausalEdge, *, bump_version: bool = True) -> None:
        self.nodes.add(edge.cause)
        self.nodes.add(edge.effect)
        key = (edge.cause, edge.action)
        existing = self.predict_graph.get(key)
        if existing is not None and existing != edge.effect:
            # 冲突：同一 (s,a) 不同后继 → 降低置信，保留新观测
            self.effect_ema[edge.action] *= 0.5
        self.predict_graph[key] = edge.effect
        self.transition_table.append(edge)
        self.pair_counts[key] += 1
        alpha = 0.35
        self.effect_ema[edge.action] = (1 - alpha) * self.effect_ema[edge.action] + alpha * edge.weight
        if bump_version:
            self.version += 1

    def observe_state(self, state: str) -> None:
        self.nodes.add(state)
        self.observed_states.add(state)

    def intervene(self, cause: str, action: str, effect: str, *, weight: float = 1.0, note: str = "") -> CausalEdge:
        edge = CausalEdge(cause, action, effect, weight=weight, note=note)
        self._commit_edge(edge)
        self.observe_state(cause)
        self.observe_state(effect)
        return edge

    def ingest_event(self, event: dict[str, Any]) -> list[CausalEdge]:
        """把更新时间线事件映射为因果干预。"""
        title = str(event.get("title", ""))
        detail = str(event.get("detail", ""))
        result = str(event.get("result", ""))
        blob = f"{title} {detail} {result}".lower()
        edges: list[CausalEdge] = []

        def add(c: str, a: str, e: str, w: float = 1.0, note: str = "") -> None:
            edges.append(self.intervene(c, a, e, weight=w, note=note or title[:80]))

        if "0.15" in blob or "0.14" in blob:
            add("PHASE_B_BOUND", "run_partial_script", "SCORE_015", 1.0, "实测榜分指纹")
            self.mark_hypothesis("C4", "supported", boost=0.05)

        if "400" in blob or "配额" in blob:
            add("KERNEL_PUSHED", "competitions_submit", "QUOTA_BLOCK", 1.0, "submit API 失败")
            self.mark_hypothesis("C5", "supported", boost=0.05)

        if "误报" in blob or ("submitted" in blob and "400" in blob):
            add("QUOTA_BLOCK", "misreport_success", "MISREPORT_SUBMITTED", 1.0, "误报已提交")
            self.mark_hypothesis("C3", "supported", boost=0.05)

        if "push" in blob and ("成功" in result or "push" in blob):
            add("LOCAL_SCRIPT_OK", "hardcode_l1_l7", "KERNEL_PUSHED", 0.9, "notebook 已含代码")

        if "l1–l7" in blob or "l1-l7" in blob or "ls20x7" in blob:
            add("LOCAL_SCRIPT_OK", "verify_local_win", "LOCAL_SCRIPT_OK", 1.0, "本地 7/7")
            if "未进榜" in result or "400" in result:
                add("KERNEL_PUSHED", "competitions_submit", "QUOTA_BLOCK", 1.0, "L1-L7 未绑定")
                self.mark_hypothesis("C1", "supported", boost=0.03)
                self.mark_hypothesis("C2", "supported", boost=0.05)

        if "version4" in blob or "v3" in blob and "0.15" in blob:
            add("SCORE_015", "select_old_version", "SCORE_015", 1.0, "旧版出分")
            self.mark_hypothesis("C6", "supported", boost=0.05)

        if "win" in blob and ("100" in blob or "7/7" in blob):
            self.observe_state("LOCAL_SCRIPT_OK")

        if not edges:
            # 弱观测：至少登记文本哈希态
            weak = "EVT_" + str(abs(hash(title)) % 10_000)
            add("LOCAL_SCRIPT_OK", "observe_update", weak, 0.2, title[:60])

        return edges

    def ingest_events(self, events: Iterable[dict[str, Any]]) -> None:
        for ev in events:
            self.ingest_event(ev)

    def mark_hypothesis(self, hid: str, status: str, *, boost: float = 0.0) -> None:
        h = self.hypotheses.get(hid)
        if not h:
            return
        h.status = status
        h.confidence = float(min(1.0, max(0.0, h.confidence + boost)))

    def counterfactual(self, *, do_action: str, from_state: str) -> dict[str, Any]:
        """干预查询：若在 from_state 执行 do_action，图上预测到哪。"""
        key = (from_state, do_action)
        predicted = self.predict_graph.get(key)
        alts = [
            (a, s2)
            for (s1, a), s2 in self.predict_graph.items()
            if s1 == from_state
        ]
        return {
            "from": from_state,
            "do": do_action,
            "predict": predicted,
            "alternatives": alts,
            "effect_ema": float(self.effect_ema.get(do_action, 0.0)),
        }

    def blocked_path_to_aligned(self) -> list[str]:
        """从 KERNEL_PUSHED 到 SCORE_ALIGNED 的断点诊断。"""
        path = [
            ("KERNEL_PUSHED", "competitions_submit", "SUBMIT_API_OK"),
            ("SUBMIT_API_OK", "phase_b_rerun", "PHASE_B_BOUND"),
            ("PHASE_B_BOUND", "load_hardcoded", "HARDCODED_LOADED"),
            ("HARDCODED_LOADED", "run_l1_l7", "MULTI_LEVEL_SCORE"),
            ("MULTI_LEVEL_SCORE", "aggregate_25", "SCORE_ALIGNED"),
        ]
        breaks: list[str] = []
        for s, a, expect in path:
            got = self.predict_graph.get((s, a))
            if got != expect:
                # 若观测到 QUOTA_BLOCK 等负面后继，标为断点
                if got is None:
                    breaks.append(f"缺失边 {s} --{a}--> {expect}")
                else:
                    breaks.append(f"偏离边 {s} --{a}--> {got}（期望 {expect}）")
                if got == "QUOTA_BLOCK" or "QUOTA_BLOCK" in self.observed_states:
                    breaks.append("当前卡点：QUOTA_BLOCK（竞赛提交未成功）")
                    break
        if "SCORE_015" in self.observed_states and "SCORE_ALIGNED" not in self.observed_states:
            breaks.append("观测到 SCORE_015 且未观测 SCORE_ALIGNED")
        if "MISREPORT_SUBMITTED" in self.observed_states:
            breaks.append("观测到 MISREPORT_SUBMITTED（决策曾被误导）")
        return breaks

    def snapshot(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "nodes": sorted(self.nodes),
            "observed_states": sorted(self.observed_states),
            "edges": [
                {
                    "cause": e.cause,
                    "action": e.action,
                    "effect": e.effect,
                    "weight": e.weight,
                    "note": e.note,
                }
                for e in list(self.transition_table)[-40:]
            ],
            "predict_graph": [
                {"state": s, "action": a, "next": nxt}
                for (s, a), nxt in sorted(self.predict_graph.items())
            ],
            "effect_ema": {k: round(v, 3) for k, v in sorted(self.effect_ema.items())},
            "hypotheses": [
                {
                    "id": h.id,
                    "statement": h.statement,
                    "confidence": round(h.confidence, 3),
                    "status": h.status,
                }
                for h in self.hypotheses.values()
            ],
            "path_breaks": self.blocked_path_to_aligned(),
            "cf_submit_ok": self.counterfactual(
                do_action="competitions_submit", from_state="KERNEL_PUSHED"
            ),
            "cf_load_hardcoded": self.counterfactual(
                do_action="load_hardcoded", from_state="PHASE_B_BOUND"
            ),
        }

    def render_for_prompt(self) -> str:
        snap = self.snapshot()
        hypo_lines = "\n".join(
            f"- {h['id']} [{h['status']}] conf={h['confidence']}: {h['statement']}"
            for h in snap["hypotheses"]
        )
        break_lines = "\n".join(f"- {b}" for b in snap["path_breaks"]) or "- （无显式断点）"
        graph_lines = "\n".join(
            f"- ({e['state']}) --{e['action']}--> ({e['next']})"
            for e in snap["predict_graph"][:30]
        )
        ema_lines = ", ".join(f"{k}={v}" for k, v in snap["effect_ema"].items())
        cf1 = snap["cf_submit_ok"]
        cf2 = snap["cf_load_hardcoded"]
        return f"""【UpdateCausalModel 快照 v{snap['version']}】
观测状态: {', '.join(snap['observed_states']) or '（无）'}

假设库:
{hypo_lines}

对齐路径断点（KERNEL_PUSHED → SCORE_ALIGNED）:
{break_lines}

因果图 predict_graph（节选）:
{graph_lines}

effect_ema: {ema_lines}

反事实:
- do(competitions_submit | KERNEL_PUSHED) → predict={cf1['predict']} ema={cf1['effect_ema']:.3f}
- do(load_hardcoded | PHASE_B_BOUND) → predict={cf2['predict']} ema={cf2['effect_ema']:.3f}
"""


__all__ = [
    "UpdateCausalModel",
    "CausalEdge",
    "CausalHypothesis",
    "PRIOR_NODES",
    "PRIOR_HYPOTHESES",
]
