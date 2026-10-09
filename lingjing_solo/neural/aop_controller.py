"""AOP Controller — 高置信预测接入控制路径。

与 AOPShadowObserver 的关键区别:
  - shadow: used_for_control 恒 False，只记录
  - controller: 高置信时 used_for_control=True，可覆盖动作

安全保证（fail-closed）:
  - 低置信（progressed 概率 < threshold）→ 回退 fallback_action
  - 预测动作不合法 → 回退
  - 任何异常 → 回退，不抛
  - aop_controller=None → 完全不影响 agent（零回归风险）

策略: 对每个合法动作编码 → 前向 → 取 progressed 概率 → 选最高。
高置信且优于 fallback → 覆盖；否则保留原有动作。
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Optional

import torch

from .aop_encoder import encode_label
from .aop_model import load_checkpoint


def _softmax(logits: torch.Tensor) -> torch.Tensor:
    """数值稳定的 softmax（沿 dim=-1）。"""
    x = logits - logits.max(dim=-1, keepdim=True).values
    exp_x = x.exp()
    return exp_x / exp_x.sum(dim=-1, keepdim=True)


class AOPController:
    """高置信 AOP 预测接入控制路径。

    Args:
        checkpoint_path: AOP 模型 checkpoint
        threshold: progressed 概率门控阈值，高于此才覆盖动作
        control_log_path: 可选，写控制决策日志 JSONL
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        threshold: float = 0.8,
        control_log_path: Optional[str | Path] = None,
    ) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold 必须在 (0, 1) 区间")
        self.model, self.config = load_checkpoint(checkpoint_path)
        self.model.eval()
        self.action_names = list(self.config.get("action_names", []))
        self.action_to_id = {name: i for i, name in enumerate(self.action_names)}
        self.action_count = max(2, len(self.action_names))
        self.model_version = str(self.config.get("version", "aop.model.v1"))
        self.threshold = threshold
        self.control_log_path = Path(control_log_path) if control_log_path else None
        if self.control_log_path is not None:
            self.control_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # 统计
        self.calls = 0
        self.overrides = 0
        self.fallbacks = 0
        self.exceptions = 0

    def advise(
        self,
        fallback_action: str,
        *,
        state: str,
        levels_completed: int,
        legal_actions: list[str],
        tick: int,
        step_id: int,
        episode_id: Optional[str] = None,
    ) -> tuple[str, dict[str, Any]]:
        """对每个合法动作打分，高置信时覆盖 fallback_action。

        Returns:
            (chosen_action, meta)
            meta: {used_for_control, confidence, fallback_action, chosen_action,
                   overrode, reason, exception}
        """
        self.calls += 1
        meta: dict[str, Any] = {
            "used_for_control": False,
            "confidence": 0.0,
            "fallback_action": fallback_action,
            "chosen_action": fallback_action,
            "overrode": False,
            "reason": "",
            "exception": None,
        }

        if not legal_actions:
            meta["reason"] = "no_legal_actions"
            self.fallbacks += 1
            return fallback_action, meta

        t0 = time.perf_counter()
        try:
            # 批量编码所有合法动作 → 一次前向（n 次 → 1 次）
            features_list = []
            for act in legal_actions:
                label = {
                    "observation_before": {
                        "state": state, "levels_completed": levels_completed,
                        "legal_actions": legal_actions, "tick": tick, "score": 0.0,
                    },
                    "requested_action": {"name": act, "id": None, "payload": {}},
                    "legal_actions": legal_actions,
                    "step_id": step_id,
                    "state_delta": {},
                }
                features_list.append(encode_label(
                    label,
                    action_id=self.action_to_id.get(act, 0),
                    action_count=self.action_count,
                ))
            batch = torch.stack(features_list)  # (n, 16)
            with torch.no_grad():
                out = self.model(batch)
            prog_probs = _softmax(out["progressed"])[:, 1].tolist()
            scored = sorted(zip(prog_probs, legal_actions), reverse=True)
            best_prob, best_action = scored[0]
            meta["confidence"] = best_prob

            if best_prob >= self.threshold and best_action in legal_actions:
                meta["used_for_control"] = True
                meta["chosen_action"] = best_action
                meta["overrode"] = best_action != fallback_action
                meta["reason"] = "high_confidence"
                self.overrides += 1
                self._log(episode_id, step_id, meta, t0)
                return best_action, meta
            else:
                meta["reason"] = "low_confidence" if best_prob < self.threshold else "illegal"
                self.fallbacks += 1
                self._log(episode_id, step_id, meta, t0)
                return fallback_action, meta

        except Exception as exc:  # noqa: BLE001 — 控制路径必须 fail-closed
            meta["reason"] = "exception"
            meta["exception"] = repr(exc)
            self.exceptions += 1
            self.fallbacks += 1
            self._log(episode_id, step_id, meta, t0)
            return fallback_action, meta

    def _log(self, episode_id, step_id, meta, t0) -> None:
        if self.control_log_path is None:
            return
        row = {
            "episode_id": episode_id, "step_id": step_id,
            "ts": time.time(), "latency_ms": (time.perf_counter() - t0) * 1000.0,
            **meta,
        }
        line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        with self._lock:
            with open(self.control_log_path, "a", encoding="utf-8") as fh:
                fh.write(line)

    def stats(self) -> dict[str, int]:
        return {"calls": self.calls, "overrides": self.overrides,
                "fallbacks": self.fallbacks, "exceptions": self.exceptions}
