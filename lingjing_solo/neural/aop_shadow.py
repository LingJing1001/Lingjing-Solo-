"""AOP Shadow Observer — 真实 agent loop 的旁路观测器。

设计原则（与 tools/aop/shadow_runtime.py 一致）：
- used_for_control 恒为 False：AOP 只做旁路记录，绝不改变权威动作。
- fail-closed：推理异常只记 exception 字段，不抛、不阻塞主流程。
- 记录 prediction / actual / latency_ms / exception / episode 标识。

用法:
    observer = AOPShadowObserver(checkpoint_path, output_path)
    observer.on_episode_reset(episode_id)
    observer.observe_predict(episode_id, step, label, authoritative_action)
    ...  # agent 执行动作、观测 outcome
    observer.observe_outcome(episode_id, step, actual={"action_effective": ..., "progressed": ...})
    observer.flush()  # episode 结束时把未回填的 pending 写出
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


class AOPShadowObserver:
    """旁路记录 AOP 模型对真实决策的预测，不参与控制。

    每步最终写一行 JSONL：
      {episode_id, step_id, authoritative_action, prediction, actual,
       latency_ms, exception, used_for_control:false, model_version, ts}

    observe_predict 时行存入 pending；observe_outcome 时补 actual 并落盘。
    flush() 把未回填的 pending 以 actual=null 写出（崩溃安全兜底）。
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        output_path: str | Path,
        *,
        action_names: Optional[list[str]] = None,
    ) -> None:
        self.model, self.config = load_checkpoint(checkpoint_path)
        self.model.eval()
        self.action_names = action_names or list(self.config.get("action_names", []))
        self.action_to_id = {name: i for i, name in enumerate(self.action_names)}
        self.action_count = max(2, len(self.action_names))
        self.model_version = str(self.config.get("version", "aop.model.v1"))
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self._pending: dict[tuple[str, int], dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._closed = False

    # ── episode 生命周期 ─────────────────────────────────────────────────
    def on_episode_reset(self, episode_id: str) -> None:
        """新 episode 开始：把上一 episode 未回填的 pending 落盘。"""
        self.flush()

    # ── 预测旁路 ─────────────────────────────────────────────────────────
    def observe_predict(
        self,
        episode_id: str,
        step_id: int,
        label: dict[str, Any],
        authoritative_action: str,
    ) -> None:
        """在权威动作确定后、执行前旁路调用。绝不改变 authoritative_action。"""
        ts = time.time()
        t0 = time.perf_counter()
        row: dict[str, Any] = {
            "episode_id": episode_id,
            "step_id": step_id,
            "authoritative_action": authoritative_action,
            "prediction": None,
            "actual": None,
            "latency_ms": None,
            "exception": None,
            "used_for_control": False,
            "model_version": self.model_version,
            "ts": ts,
        }
        try:
            action_id = self.action_to_id.get(authoritative_action, 0)
            features = encode_label(
                label, action_id=action_id, action_count=self.action_count
            ).unsqueeze(0)
            with torch.no_grad():
                outputs = self.model(features)
            row["prediction"] = {
                "action_effective": int(outputs["action_effective"].argmax(1).item()),
                "progressed": int(outputs["progressed"].argmax(1).item()),
                "action": (
                    self.action_names[idx]
                    if 0 <= (idx := int(outputs["action"].argmax(1).item())) < len(self.action_names)
                    else None
                ),
            }
        except Exception as exc:  # noqa: BLE001 — 旁路必须吞异常
            row["exception"] = repr(exc)
        row["latency_ms"] = (time.perf_counter() - t0) * 1000.0

        with self._lock:
            self._pending[(episode_id, step_id)] = row

    # ── 实际结果回填 ─────────────────────────────────────────────────────
    def observe_outcome(
        self,
        episode_id: str,
        step_id: int,
        actual: dict[str, Any],
    ) -> None:
        """agent 得知真实 outcome 后回填 actual 并落盘该行。"""
        key = (episode_id, step_id)
        with self._lock:
            row = self._pending.pop(key, None)
        if row is None:
            return
        row["actual"] = dict(actual)
        self._write_row(row)

    # ── 落盘 ─────────────────────────────────────────────────────────────
    def flush(self) -> None:
        """把所有未回填的 pending 以 actual=null 写出。"""
        with self._lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for row in pending:
            self._write_row(row)

    def close(self) -> None:
        self.flush()
        with self._lock:
            self._closed = True

    def _write_row(self, row: dict[str, Any]) -> None:
        with self._lock:
            if self._closed:
                return
        line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        with open(self.output_path, "a", encoding="utf-8") as fh:
            fh.write(line)

    # ── 统计 ─────────────────────────────────────────────────────────────
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)


def build_label_for_step(
    *,
    state: str,
    levels_completed: int,
    legal_actions: list[str],
    tick: int,
    step_id: int,
    action: str,
    score: float = 0.0,
) -> dict[str, Any]:
    """从 agent loop 的实时状态构造一个 encode_label 可用的 label dict。

    只含预测前可得的信息（observation_before / requested_action / legal_actions）。
    """
    return {
        "observation_before": {
            "state": state,
            "levels_completed": levels_completed,
            "legal_actions": legal_actions,
            "tick": tick,
            "score": score,
        },
        "requested_action": {"name": action, "id": None, "payload": {}},
        "legal_actions": legal_actions,
        "step_id": step_id,
        "state_delta": {},
    }
