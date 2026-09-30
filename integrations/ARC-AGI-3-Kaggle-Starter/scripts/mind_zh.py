"""Agent 心智状态 → 中文面板（play_ui / visualize_vc33 共用）。"""
from __future__ import annotations

import re
from typing import Any, Optional

ACTION_CN = {
    "RESET": "重置",
    "ACTION1": "向上 ↑",
    "ACTION2": "向下 ↓",
    "ACTION3": "向左 ←",
    "ACTION4": "向右 →",
    "ACTION5": "特殊5",
    "ACTION6": "点击",
    "ACTION7": "撤销",
    "INIT": "开局",
}

RATIONALE_CN = {
    "ls20_solver": "专用 ls20 解题器（BFS+旋转台）",
    "discrete_nav": "离散四向导航",
    "explore": "自主实验：试信息增益最高的动作",
    "search": "因果图搜索（BFS/A*）",
    "reflect": "困境反思后换策略",
    "advisor_hint": "顾问/符号猜想提示",
    "bubble_click": "泡壁区域点击探索",
    "bubble_click_lowgain": "低增益时点击试探",
    "fallback": "兜底随机",
    "reset": "重置关卡",
    "symbolic_give_up": "猜想耗尽，放弃本局",
    "act": "常规动作",
}

STATE_CN = {
    "WIN": "全胜通关 ✓",
    "NOT_FINISHED": "进行中（步数未用完）",
    "GAME_OVER": "失败（步数耗尽）",
    "NOT_PLAYED": "未开始",
    "IDLE": "待命",
    "STARTING": "启动中",
    "RUNNING": "对局中",
    "DONE": "已结束",
}

# 官方 baseline 步数 ≈ 人类参考；关卡数 = baseline_actions 长度
GAME_META: dict[str, dict[str, Any]] = {
    "ls20": {
        "title": "LS20",
        "total_levels": 7,
        "human_steps": [22, 123, 73, 84, 96, 192, 186],
        "mode": "键盘四向 + 专用解题器",
        "success_hint": "玩家块站到目标格，形状/颜色/旋转与目标一致",
    },
    "vc33": {
        "title": "VC33",
        "total_levels": 7,
        "human_steps": [7, 18, 44, 61, 131, 34, 152],
        "mode": "点击探索 + 因果图学习",
        "success_hint": "通过点击与探索解开关卡机关",
    },
}

VERDICT_CN = {
    "win": "✅ 任务完成",
    "partial": "⚠️ 部分完成",
    "fail": "❌ 任务失败",
    "running": "🔄 进行中",
    "idle": "⏸ 未开始",
}


def rationale_cn(key: str) -> str:
    k = (key or "").strip().lower()
    return RATIONALE_CN.get(k, f"策略：{key}")


def state_cn(st: str) -> str:
    name = str(st or "").split(".")[-1].upper()
    return STATE_CN.get(name, str(st))


def parse_lers(reason: str) -> dict[str, Any]:
    """从 my_agent 的 lers:... 字符串解析字段。"""
    out: dict[str, Any] = {}
    if not reason or not isinstance(reason, str):
        return out
    for part in reason.split("|"):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k == "eff":
            try:
                # eff={'ACTION1': 0.1, ...}
                pairs = re.findall(r"'?(ACTION\d)'?\s*:\s*([\d.]+)", v)
                out["eff"] = {a: float(x) for a, x in pairs}
            except Exception:
                out["eff_raw"] = v
        elif k == "why":
            out["why"] = v
        elif k in ("step", "v", "edges", "vis"):
            try:
                out[k] = int(v)
            except ValueError:
                out[k] = v
        elif k in ("C", "fog"):
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
        elif k == "act":
            out["act"] = v
        elif k.startswith("L"):
            try:
                out["level"] = int(k[1:])
            except ValueError:
                pass
    return out


def extract_mind(agent: Any, frame: Any = None, action_name: str = "") -> dict[str, Any]:
    """从 MyAgent + LingjingSoloAgent 提取结构化心智快照。"""
    brain = getattr(agent, "brain", None)
    if brain is None:
        return {}

    field = brain.field
    reflector = brain.reflector
    levels = int(getattr(frame, "levels_completed", 0) or 0) if frame else int(field.levels or 0)
    win_levels = getattr(frame, "win_levels", None) if frame else None

    c = float(getattr(reflector, "c_value", 0.0) or 0.0)
    fog = float(field.bubble.fog_ratio if field.bubble else 0.0)
    edges = len(field.predict_graph)
    visited = len(field.visited_set)
    transitions = len(field.transition_table)
    rules = len(field.rules)
    version = int(field.version)
    noop = int(field.noop_streak)
    delta_px = int(field.last_delta_pixels)

    eff_all = {}
    for i in range(1, 8):
        a = f"ACTION{i}"
        eff_all[a] = round(float(field.effect_strength(a)), 3)

    why = brain.last_rationale or ""
    why_cn = rationale_cn(why)
    act = action_name or ""

    ls20 = getattr(brain, "ls20", None)
    ls20_active = bool(getattr(ls20, "active", False)) if ls20 else False
    ls20_phase = getattr(ls20, "_phase", "") if ls20 else ""

    # 成功判定说明
    st_name = ""
    if frame and hasattr(frame, "state"):
        st_name = frame.state.name if hasattr(frame.state, "name") else str(frame.state)

    success_level = "partial"
    success_cn = "进行中"
    if st_name == "WIN":
        success_level = "win"
        success_cn = f"完全成功：通关全部关卡（{levels} 关）"
    elif levels > 0 and st_name == "NOT_FINISHED":
        success_level = "partial"
        success_cn = f"部分成功：已过 {levels} 关，还在继续"
    elif st_name == "GAME_OVER":
        success_level = "fail"
        success_cn = f"未通关：只过了 {levels} 关，步数耗尽"
    elif levels > 0:
        success_cn = f"已过 {levels} 关"

    return {
        "step_brain": int(brain.step),
        "levels": levels,
        "win_levels": win_levels,
        "version": version,
        "edges": edges,
        "visited": visited,
        "transitions": transitions,
        "rules": rules,
        "c_value": round(c, 3),
        "fog": round(fog, 3),
        "noop_streak": noop,
        "delta_pixels": delta_px,
        "why": why,
        "why_cn": why_cn,
        "action": act,
        "action_cn": ACTION_CN.get(act, act),
        "eff": eff_all,
        "ls20_active": ls20_active,
        "ls20_phase": ls20_phase,
        "conflict": bool(field.conflict_flag),
        "success_level": success_level,
        "success_cn": success_cn,
        "state_cn": state_cn(st_name),
        # 因果图增长解读
        "causal_hint": (
            f"因果边 {edges} 条 = 已学会「在某状态做某动作会到哪」；"
            f"转移记录 {transitions} 条；规则 {rules} 条"
        ),
        "experiment_hint": (
            "自主实验" if why in ("explore", "bubble_click", "bubble_click_lowgain") else
            "图搜索" if why == "search" else
            "专用逻辑" if why == "ls20_solver" else
            "反思换策" if why in ("reflect", "advisor_hint") else
            "其他"
        ),
    }


def mind_from_reasoning(reason: Any, parsed: Optional[dict] = None) -> dict[str, Any]:
    """仅用 reasoning 字符串补充解析（兼容旧数据）。"""
    text = ""
    if isinstance(reason, dict):
        text = str(reason.get("reason", reason))
    elif isinstance(reason, str):
        text = reason
    p = parse_lers(text)
    if parsed:
        p = {**parsed, **{k: v for k, v in p.items() if v is not None}}
    if p.get("why"):
        p["why_cn"] = rationale_cn(p["why"])
    return p


def build_verdict(
    agent: Any,
    frame: Any,
    game_id: str,
    step: int,
    max_steps: int,
    running: bool = False,
) -> dict[str, Any]:
    """生成可展示的成败判定 + 检查清单 + 失败原因（事实依据）。"""
    gid = (game_id or "").split("-")[0].lower()
    meta = GAME_META.get(gid, {
        "title": gid.upper(),
        "total_levels": 7,
        "human_steps": [],
        "mode": "通用",
        "success_hint": "达到 GameState.WIN 或 levels_completed 增加",
    })
    total = int(meta.get("total_levels", 7))
    mind = extract_mind(agent, frame, "")

    st_name = ""
    if frame and hasattr(frame, "state"):
        st_name = frame.state.name if hasattr(frame.state, "name") else str(frame.state)
    st_upper = st_name.split(".")[-1].upper()

    levels = int(mind.get("levels", 0) or 0)
    win_levels = mind.get("win_levels")
    target_levels = int(win_levels or total)

    if not running and st_upper == "NOT_PLAYED":
        verdict = "idle"
    elif st_upper == "WIN":
        verdict = "win"
    elif running and st_upper not in ("GAME_OVER", "WIN"):
        verdict = "running"
    elif levels > 0 and st_upper == "NOT_FINISHED" and step >= max_steps:
        verdict = "partial"
    elif st_upper == "GAME_OVER" or (not running and levels < target_levels and step >= max_steps):
        verdict = "fail" if levels == 0 else "partial"
    elif levels >= target_levels:
        verdict = "win"
    elif levels > 0:
        verdict = "partial"
    else:
        verdict = "fail" if st_upper == "GAME_OVER" else "running"

    progress_pct = min(100, round(100 * levels / max(1, target_levels)))

    checks: list[dict[str, Any]] = []
    checks.append({
        "id": "win_state",
        "label": "游戏状态 WIN",
        "pass": st_upper == "WIN",
        "detail": f"当前={st_upper}（WIN 才表示官方认定全通）",
    })
    checks.append({
        "id": "levels",
        "label": f"过关数 ≥ 目标 ({target_levels})",
        "pass": levels >= target_levels,
        "detail": f"已过 {levels} / {target_levels} 关",
    })
    checks.append({
        "id": "progress_signal",
        "label": "本步画面有变化",
        "pass": mind.get("delta_pixels", 0) > 0 or levels > 0,
        "detail": f"变化像素={mind.get('delta_pixels', 0)}（0=可能无效动作）",
    })
    checks.append({
        "id": "steps_budget",
        "label": "步数预算未耗尽",
        "pass": step < max_steps and st_upper != "GAME_OVER",
        "detail": f"已用 {step} / {max_steps} 步",
    })

    brain = getattr(agent, "brain", None)
    if gid == "ls20":
        ls = getattr(brain, "ls20", None) if brain else None
        checks.append({
            "id": "ls20_solver",
            "label": "ls20 专用解题器已激活",
            "pass": bool(getattr(ls, "active", False)) if ls else False,
            "detail": f"阶段={getattr(ls, '_phase', '?')} · "
            f"修饰={getattr(ls, 'mod_kind', '?')}",
        })
    else:
        checks.append({
            "id": "causal_growth",
            "label": "因果边在积累（边玩边学）",
            "pass": mind.get("edges", 0) >= 3 or mind.get("transitions", 0) >= 5,
            "detail": f"因果边={mind.get('edges', 0)} · 转移={mind.get('transitions', 0)}",
        })

    failure_reasons: list[str] = []
    if verdict == "win":
        failure_reasons.append("无失败项：任务已按官方规则完成。")
    elif verdict == "running":
        failure_reasons.append("对局进行中，尚未到最终判定。")
        if mind.get("noop_streak", 0) >= 4:
            failure_reasons.append(
                f"连续 {mind.get('noop_streak')} 步画面几乎无变化，可能在原地打转。"
            )
    else:
        if st_upper == "GAME_OVER":
            failure_reasons.append("步数/无效移动耗尽 → GAME_OVER（游戏内失败）。")
        if step >= max_steps:
            failure_reasons.append(f"达到最大步数上限 {max_steps}，Agent 被强制停止。")
        if levels < target_levels:
            failure_reasons.append(
                f"只过了 {levels} 关，未到目标 {target_levels} 关。"
            )
        if gid == "ls20" and brain:
            ls = brain.ls20
            if getattr(ls, "stall", 0) >= 3:
                failure_reasons.append("ls20 解题器多次移动失败（卡住/路径不通）。")
            if getattr(ls, "_await_level_layout", False):
                failure_reasons.append("过关后布局刷新中，解题器尚未重新锁定关卡。")
            if getattr(ls, "mod_kind", "") == "rot":
                pe = getattr(ls, "pad_entries", 0)
                tp = getattr(ls, "target_pad_entries", 0)
                if pe < tp:
                    failure_reasons.append(
                        f"旋转台次数不足：已踩 {pe}/{tp} 次，角度未对齐目标。"
                    )
        if mind.get("why") == "symbolic_give_up":
            failure_reasons.append("符号猜想耗尽，Agent 主动放弃本局。")
        if mind.get("c_value", 0) > 0.7:
            failure_reasons.append(
                f"困境 C 值高 ({mind.get('c_value')})：循环/迷雾/搜索耗尽。"
            )
        if not failure_reasons:
            failure_reasons.append("未达通关条件，需结合日志与画面继续排查。")

    human_ref = meta.get("human_steps", [])
    human_total = sum(human_ref) if human_ref else None

    return {
        "verdict": verdict,
        "verdict_cn": VERDICT_CN.get(verdict, verdict),
        "progress_pct": progress_pct,
        "levels": levels,
        "target_levels": target_levels,
        "total_levels": total,
        "state": st_upper,
        "checks": checks,
        "failure_reasons": failure_reasons,
        "game_title": meta.get("title", gid),
        "game_mode": meta.get("mode", ""),
        "success_hint": meta.get("success_hint", ""),
        "human_baseline_steps": human_total,
        "agent_steps": step,
        "max_steps": max_steps,
        "mind": mind,
    }
