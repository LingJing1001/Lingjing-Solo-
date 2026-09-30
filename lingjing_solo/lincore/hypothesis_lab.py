"""SSA-E2 · 假设实验室：LLM 高层假设生成 × 谱验证裁决。

分工（白皮书 §4 "LLM 慢思考"行）：
  LLM = System 2 提案器：输出结构化假设（哪个动作、作用在哪、预期什么效应）
  本实验室 = System 1 裁决器：把每个假设转成可证伪的线性约束，
      设计区分性实验 → 用效应向量 Δ 的块支撑集对比预期 → 支持/反驳计数

裁决数学（[S6] Ch1 内积 / Ch3 零空间）：
  agreement(Δ) = |supp(Δ) ∩ expect_blocks| / max(1, |supp(Δ) ∪ expect_blocks|)
  （Jaccard 型支撑集一致性；≥0.5 判支持，否则反驳）
  支持把假设推入"已验证列空间"（可入 ScriptBank），反驳即消元排除（[S6] Ch2）。

LLM 接入：from_llm_json() 把 LLM 的 JSON 建议转为 Hypothesis —— 无 LLM 时
实验室退化为纯数据驱动（经验区域先验），决策链不受影响。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .features import block_center


def extract_suggestions_json(text: str) -> Optional[dict]:
    """从 LLM 自由文本中提取建议 JSON（容错：找首个 [...] 或 {"suggestions": [...]}）。

    返回统一形如 {"suggestions": [...]}；解析失败返回 None。
    """
    if not text:
        return None
    try:
        import json as _json
        # 直接整体解析
        obj = _json.loads(text)
        if isinstance(obj, list):
            return {"suggestions": obj}
        if isinstance(obj, dict):
            return obj if "suggestions" in obj else None
    except Exception:
        pass
    # 截取首个 JSON 数组
    start = text.find("[")
    end = text.rfind("]")
    if 0 <= start < end:
        try:
            import json as _json
            arr = _json.loads(text[start : end + 1])
            if isinstance(arr, list):
                return {"suggestions": arr}
        except Exception:
            return None
    return None


@dataclass
class Hypothesis:
    """一条可证伪假设：ACTION a 作用于 xy，预期块集 expect_blocks 发生变化。"""

    hid: str
    text: str
    action: str
    xy: Optional[Tuple[int, int]] = None
    expect_blocks: Tuple[int, ...] = ()
    prior: float = 0.5
    trials: int = 0
    supports: int = 0
    refutes: int = 0
    last_agreement: float = 0.0
    source: str = "local"

    @property
    def confidence(self) -> float:
        """拉普拉斯平滑后验（先验 + 证据）。"""
        n = self.trials
        return (self.supports + 6.0 * self.prior) / (n + 6.0) if n else self.prior

    @property
    def exhausted(self) -> bool:
        return self.trials >= 4 or (self.trials >= 2 and self.refutes > self.supports)

    @property
    def verified(self) -> bool:
        return self.trials >= 2 and self.supports >= 2


class HypothesisLab:
    """假设登记 → 区分实验设计 → 谱裁决。"""

    def __init__(self, max_pending: int = 16) -> None:
        self.hyps: Dict[str, Hypothesis] = {}
        self.max_pending = int(max_pending)
        self._pending: List[str] = []
        self._last_tested: Optional[str] = None

    # ---------- 登记 ----------

    def register(
        self,
        action: str,
        text: str = "",
        xy: Optional[Tuple[int, int]] = None,
        expect_blocks: Sequence[int] = (),
        prior: float = 0.5,
        hid: Optional[str] = None,
        source: str = "local",
    ) -> Optional[Hypothesis]:
        hid = hid or f"h{len(self.hyps) + 1}_{int(action[-1]) if action[-1:].isdigit() else 0}"
        if hid in self.hyps:
            return self.hyps[hid]
        if len(self.hyps) >= self.max_pending:
            # 淘汰最差已结案假设腾位
            self._gc()
            if len(self.hyps) >= self.max_pending:
                return None
        h = Hypothesis(
            hid=hid,
            text=text or f"IF {action} at {xy} THEN change blocks {tuple(expect_blocks)}",
            action=str(action).upper(),
            xy=xy,
            expect_blocks=tuple(int(b) for b in expect_blocks),
            prior=float(min(1.0, max(0.0, prior))),
            source=source,
        )
        self.hyps[hid] = h
        self._pending.append(hid)
        return h

    @staticmethod
    def from_llm_json(payload: Any, blocks: int = 8, shape=(64, 64)) -> List[Hypothesis]:
        """LLM JSON → 假设列表。容错：任何畸形输入返回空。

        期望形如：
          {"suggestions": [{"action": "ACTION6", "x": 32, "y": 12,
                            "expect": "左上门亮起", "confidence": 0.7}]}
        expect_blocks 由 (x,y) 周边块自动推导（3×3 邻域）。
        """
        out: List[Hypothesis] = []
        try:
            items = payload.get("suggestions") if isinstance(payload, dict) else payload
            if not isinstance(items, list):
                return out
            for i, it in enumerate(items[:8]):
                if not isinstance(it, dict):
                    continue
                act = str(it.get("action", "ACTION6")).upper()
                x, y = int(it.get("x", 32)), int(it.get("y", 32))
                conf = float(it.get("confidence", 0.5))
                expect = _neighbor_blocks(x, y, shape, blocks)
                out.append(
                    Hypothesis(
                        hid=f"llm{i}_{int(act[-1]) if act[-1:].isdigit() else 0}",
                        text=str(it.get("expect", ""))[:120],
                        action=act,
                        xy=(x, y),
                        expect_blocks=expect,
                        prior=min(0.95, max(0.05, conf)),
                        source="llm",
                    )
                )
        except Exception:
            return []
        return out

    # ---------- 实验设计 ----------

    def design_experiment(self) -> Optional[Tuple[str, Optional[Tuple[int, int]], str]]:
        """选最有信息量且未结案的假设做区分实验（[S6] Ch2 消元判据的贪心版）。"""
        best, best_key = None, -1.0
        for hid in list(self._pending):
            h = self.hyps.get(hid)
            if h is None or h.exhausted or h.verified:
                continue
            key = h.prior + 0.1 * (4 - h.trials) / 4.0
            if key > best_key:
                best, best_key = h, key
        self._last_tested = best.hid if best else None
        if best is None:
            return None
        return best.action, best.xy, f"hyp_test:{best.hid}"

    # ---------- 谱裁决 ----------

    def verify(self, delta_blocks: Sequence[int]) -> Optional[Dict[str, Any]]:
        """对刚测试的假设裁决：Δ 的块支撑集 vs 预期块集。

        一致性 = 观测解释率 |obs∩exp| / max(1,|obs|)：预期是 LLM 坐标的
        3×3 邻域（近似语言），故只要求「实际发生的变化都被预期覆盖」，
        而非双向 Jaccard（那会过度惩罚坐标误差）。≥0.5 判支持。
        """
        hid = self._last_tested
        self._last_tested = None
        h = self.hyps.get(hid or "")
        if h is None:
            return None
        obs = set(int(b) for b in delta_blocks if int(b) >= 0)
        exp = set(h.expect_blocks)
        if not obs:
            agree = 0.0  # 无任何效应 → 无法支持
        else:
            agree = len(obs & exp) / max(1, len(obs))
        h.trials += 1
        h.last_agreement = round(float(agree), 3)
        if agree >= 0.5:
            h.supports += 1
            verdict = "support"
        else:
            h.refutes += 1
            verdict = "refute"
        return {
            "hid": h.hid,
            "verdict": verdict,
            "agreement": h.last_agreement,
            "confidence": round(h.confidence, 3),
            "verified": h.verified,
            "exhausted": h.exhausted,
        }

    def auto_from_mind(self, mind, shape=(64, 64), top_k: int = 2) -> int:
        """谱信号 → 显式可证伪假设（LLM 缺席时的 System 2 替身）。

        取特征技能图中高杠杆、未结案的区域，登记为
        「ACTION6 作用于该区域 → 该区域及其邻接块变化」假设。
        这把谱排序的隐式偏好升级为可记录/可导出 ScriptBank 的显式知识。
        """
        try:
            em = mind.eigen
            em.eigen_decompose()
            B = int(mind.blocks)
            lev = [
                (r, float(em.leverage(r)), float(em.trial_count(r)))
                for r in range(em.S.shape[0])
            ]
            lev = [x for x in lev if x[1] > 1e-6]
            lev.sort(key=lambda t: (-t[1], t[2]))
            added = 0
            for r, lv, _tr in lev:
                if added >= max(1, int(top_k)):
                    break
                hid = f"auto_r{int(r)}"
                h = self.hyps.get(hid)
                if h is not None and (h.exhausted or h.verified):
                    continue
                bi, bj = divmod(int(r), B)
                blocks = set()
                for dbi in (-1, 0, 1):
                    for dbj in (-1, 0, 1):
                        ni, nj = bi + dbi, bj + dbj
                        if 0 <= ni < B and 0 <= nj < B:
                            blocks.add(ni * B + nj)
                c = block_center(int(r), shape, B)
                if self.register(
                    action="ACTION6",
                    text=f"auto: high-leverage region r{int(r)} lev={lv:.2f}",
                    xy=c,
                    expect_blocks=tuple(sorted(blocks)),
                    prior=min(0.9, 0.3 + lv / 2.0),
                    hid=hid,
                ):
                    added += 1
            return added
        except Exception:
            return 0

    def verified_hypotheses(self) -> List[Hypothesis]:
        """已验证（可入技能库/列空间）的假设。"""
        return [h for h in self.hyps.values() if h.verified]

    def snapshot(self) -> Dict[str, Any]:
        return {
            "total": len(self.hyps),
            "pending": sum(
                1 for h in self.hyps.values() if not h.exhausted and not h.verified
            ),
            "verified": len(self.verified_hypotheses()),
            "top": [
                {
                    "hid": h.hid,
                    "action": h.action,
                    "xy": h.xy,
                    "conf": round(h.confidence, 2),
                    "t/s/r": f"{h.trials}/{h.supports}/{h.refutes}",
                }
                for h in sorted(self.hyps.values(), key=lambda z: z.confidence, reverse=True)[:5]
            ],
        }

    def _gc(self) -> None:
        dead = [hid for hid, h in self.hyps.items() if h.exhausted and not h.verified]
        for hid in dead:
            self.hyps.pop(hid, None)
            if hid in self._pending:
                self._pending.remove(hid)


def _neighbor_blocks(x: int, y: int, shape, blocks: int) -> Tuple[int, ...]:
    """(x,y) 的 3×3 像素邻域 → 块索引集合（LLM 坐标 → 谱块语言）。"""
    try:
        H, W = int(shape[0]), int(shape[1])
        bh = max(1, -(-H // blocks))
        bw = max(1, -(-W // blocks))
        out = set()
        for dy in (-2, 0, 2):
            for dx in (-2, 0, 2):
                nx, ny = min(W - 1, max(0, x + dx)), min(H - 1, max(0, y + dy))
                bi = min(blocks - 1, ny // bh)
                bj = min(blocks - 1, nx // bw)
                out.add(bi * blocks + bj)
        return tuple(sorted(out))
    except Exception:
        return ()
