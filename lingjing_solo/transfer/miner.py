"""轨迹收割的纯逻辑半边：计划提取 / 格式归一 / 解析器 / 去重常量。

数据来源（按可靠性）：
  agent-plan   1.0  my_agent.py 里 10 套真机回放验证过的计划常量（ast 提取，不执行模块）
  route        1.0  integrations/.../tests/routes/*.json（3×WIN 门过的路线）
  solution     1.0  state/*_solution.json（离线求解器产物；tr87 不在 router 里，白赚一局）
  walk         0.6  无计划游戏上的种子随机游走（组件中心点击 + 键盘），给 LOGO 提供"未见游戏"
  probe        0.5  反事实探针：快照/恢复（generic_shadow）下试别的动作，引擎真值标签

去重键 = (is_probe, game_id, level, grid_hash_before, aid, x, y)——r2_transfers.json 里
那种同条记录重复写入的问题在这里结构性解决；同状态同动作但结局不同的罕见情况保留首见。

架构边界（tests/test_abstract_action_boundary.py）：lingjing_solo 包内只有 harness/
可以 import 引擎。本模块因此只放纯逻辑；真正上引擎回放/探针的收割器在
scripts/harvest_trajectories.py（边界侧脚本），训练入口 scripts/train_affordance.py。
状态签名用「去计数器摘要」：generic_state_key 的快照里 _action_count 每步必变，
会吞掉 no_op/state_only 的区分，所以显式剔除计数型键后再摘要。
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
ENV_DIR = ROOT / "environment_files"
AGENT_PY = ROOT / "integrations" / "ARC-AGI-3-Kaggle-Starter" / "agent" / "my_agent.py"
ROUTES_DIR = ROOT / "integrations" / "ARC-AGI-3-Kaggle-Starter" / "tests" / "routes"
STATE_DIR = ROOT / "state"
DEFAULT_OUT = STATE_DIR / "affordance_dataset.jsonl"

# 计划常量名 → 步格式：kbd=纯动作号，click=纯 ACTION6 点击 (x,y)，axy=(aid,x,y)
PLAN_VAR_FORMAT = {
    "_LS20_LEVEL_ACTIONS": ("kbd", "ls20"),
    "_AR25_LEVEL_ACTIONS": ("kbd", "ar25"),
    "_WA30_PLANS": ("kbd", "wa30"),
    "_VC33_PLANS": ("click", "vc33"),
    "_TN36_PLANS": ("click", "tn36"),
    "_R11L_PLANS": ("click", "r11l"),
    "_FT09_HARDCODED": ("click", "ft09"),
    "_SB26_PLANS": ("axy", "sb26"),
    "_CD82_PLANS": ("axy", "cd82"),
    "_BP35_PLANS": ("axy", "bp35"),
}
_COUNTER_RE = ("action", "step", "tick", "count", "frame", "guid")


# --------------------------------------------------------------- 纯函数部分


def extract_agent_plans(agent_py: Path = AGENT_PY) -> Dict[str, Dict[int, list]]:
    """从 my_agent.py 源码里 ast 提取计划常量（literal，不执行模块）。

    _FT09_HARDCODED 是类属性，ast.walk 一并把 ClassDef 里的 Assign 收进来。
    """
    tree = ast.parse(agent_py.read_text(encoding="utf-8"))
    out: Dict[str, Dict[int, list]] = {}
    for node in ast.walk(tree):
        # my_agent.py 的计划常量带类型注解（AnnAssign），普通赋值（Assign）一并兼容
        if isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        else:
            continue
        if value is None or not isinstance(target, ast.Name) or target.id not in PLAN_VAR_FORMAT:
            continue
        try:
            try:
                val = ast.literal_eval(value)
            except ValueError:
                # _AR25_LEVEL_ACTIONS 这类 [3]*5 + [2]*10 的字面表达式：
                # 无内建无名字的受限 eval（来源是仓库内自己的受信源码）
                val = eval(compile(ast.Expression(value), "<plan>", "eval"),
                           {"__builtins__": {}}, {})
        except Exception:
            continue
        out[target.id] = {int(k): list(v) for k, v in val.items()}
    return out


def counter_free_digest(snap: Dict[str, Any]) -> str:
    """快照摘要，剔除计数型键（值被判定为计数器：键名命中 _COUNTER_RE 且是标量）。

    复用 generic_shadow._encode 的无恒等编码（同一套编码，保证与搜索去重同语义）。
    """
    import hashlib

    from ..planning.search.generic_shadow import _encode

    payload = b"".join(
        _encode(k) + _encode(v)
        for k, v in sorted(snap.items())
        if not (isinstance(v, (int, float)) and any(t in str(k).lower() for t in _COUNTER_RE))
    )
    return hashlib.md5(payload).hexdigest()[:16]


def plan_steps(fmt: str, entry: Any) -> List[Dict[str, Optional[int]]]:
    """一条计划条目 → 归一化动作 spec 列表 [{aid, x, y}]。"""
    out: List[Dict[str, Optional[int]]] = []
    if fmt == "kbd":
        for aid in entry:
            out.append({"aid": int(aid), "x": None, "y": None})
    elif fmt == "click":
        for x, y in entry:
            out.append({"aid": 6, "x": int(x), "y": int(y)})
    elif fmt == "axy":
        for aid, x, y in entry:
            out.append({"aid": int(aid), "x": int(x) if int(aid) == 6 else None,
                        "y": int(y) if int(aid) == 6 else None})
    return out


def route_json_steps(actions: Sequence[dict]) -> List[Dict[str, Optional[int]]]:
    out = []
    for spec in actions:
        aid = int(str(spec["a"]).replace("ACTION", ""))
        out.append({"aid": aid,
                    "x": int(spec["x"]) if aid == 6 else None,
                    "y": int(spec["y"]) if aid == 6 else None})
    return out


def solution_json_steps(actions: Sequence[dict]) -> List[Dict[str, Optional[int]]]:
    out = []
    for spec in actions:
        aid = int(spec["action"])
        data = spec.get("data") or {}
        out.append({"aid": aid,
                    "x": int(data.get("x")) if aid == 6 and data else None,
                    "y": int(data.get("y")) if aid == 6 and data else None})
    return out


def gid_to_full_id(gid: str) -> Optional[str]:
    d = ENV_DIR / gid
    if not d.is_dir():
        return None
    for sub in sorted(d.iterdir()):
        if sub.is_dir():
            return f"{gid}-{sub.name}"
    return None


# --------------------------------------------------------------- 顶层编排


def load_route_plans() -> List[Tuple[str, str, Dict[int, List[Dict[str, Optional[int]]]]]]:
    """routes/*.json → [(full_id, gid, per_level)]。routes 是整局动作流，不分会，
    level 归属靠回放时的 levels_completed 动态判定（per_level 塞成 {0: 全部}）。"""
    out = []
    for p in sorted(ROUTES_DIR.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        full = d["game_id"]
        out.append((full, full.split("-")[0], {0: route_json_steps(d["actions"])}))
    return out


def load_solution_plans() -> List[Tuple[str, str, Dict[int, List[Dict[str, Optional[int]]]]]]:
    """state/*_solution.json → [(full_id, gid, per_level)]。"""
    out = []
    for name, kind in (("cd82_solution.json", "per_level"),
                       ("tn36_solution.json", "per_level"),
                       ("tr87_solution.json", "plans")):
        p = STATE_DIR / name
        if not p.is_file():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        full = d.get("game_id") or d.get("game")  # 解文件用的键是 "game"
        if not full:
            continue
        gid = full.split("-")[0]
        if kind == "per_level":
            # 全局动作流 + 每关步数 → 切成逐关计划
            steps = solution_json_steps(d["actions"])
            per_level: Dict[int, List] = {}
            i = 0
            for lv, n in enumerate(d["per_level_actions"]):
                per_level[lv] = steps[i:i + int(n)]
                i += int(n)
        else:
            per_level = {int(k): plan_steps("kbd", v) for k, v in d["plans"].items()}
        out.append((full, gid, per_level))
    return out
