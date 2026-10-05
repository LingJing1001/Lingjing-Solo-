"""
灵境引擎 V14.0 — persistence.recorder
======================================
状态记录与重放（确定性验证）。

功能：
- save_snapshot(state, path) : 保存完整状态（phi + agents）
- load_snapshot(path)        : 恢复状态
- compare_snapshots(s1, s2) : 比对两个快照是否一致
"""
from __future__ import annotations
from typing import Any, Dict
import json
import os


def save_snapshot(state: Dict[str, Any], path: str) -> None:
    """保存快照（phi + agent 状态）。"""
    import numpy as np
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "tick": int(state["tick"]),
        "t": float(state["t"]),
        "phi": state["phi"].astype(np.float64).tolist(),
        "agents": [
            {
                "id": int(a.id),
                "pos": list(map(int, a.pos)),
                "sub_pos": list(map(float, a.sub_pos)),
                "alive": bool(a.alive),
            }
            for a in state["agents"]
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f)


def load_snapshot(path: str) -> Dict[str, Any]:
    """加载快照。"""
    with open(path, "r") as f:
        return json.load(f)


def compare_snapshots(s1: Dict, s2: Dict, atol: float = 1e-10) -> Dict[str, Any]:
    """比对两个快照是否一致（用于确定性验证）。"""
    import numpy as np
    phi1 = np.array(s1["phi"])
    phi2 = np.array(s2["phi"])
    result = {
        "tick_match": s1["tick"] == s2["tick"],
        "t_match": np.isclose(s1["t"], s2["t"], atol=atol),
        "phi_match": np.allclose(phi1, phi2, atol=atol),
        "phi_max_err": float(np.max(np.abs(phi1 - phi2))),
        "agents_match": True,
    }
    if len(s1["agents"]) != len(s2["agents"]):
        result["agents_match"] = False
    else:
        for a1, a2 in zip(s1["agents"], s2["agents"]):
            if (a1["id"] != a2["id"] or
                list(a1["pos"]) != list(a2["pos"]) or
                not np.allclose(a1["sub_pos"], a2["sub_pos"], atol=atol)):
                result["agents_match"] = False
                break
    return result
