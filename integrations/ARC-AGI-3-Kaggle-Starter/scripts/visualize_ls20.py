"""录制 ls20 求解过程并生成交互式步骤可视化。

用法:
  .\\.venv\\Scripts\\python.exe scripts\\visualize_ls20.py
  .\\.venv\\Scripts\\python.exe scripts\\visualize_ls20.py --max-steps 800 --serve
  .\\.venv\\Scripts\\python.exe scripts\\visualize_ls20.py --open

输出:
  ui/static/ls20_replay.json  — 每步网格 + 动作 + 标注
  浏览器打开 ls20_viewer.html 逐步回放
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))
sys.path.insert(0, str(ROOT))

import arc_agi
from arc_agi import OperationMode

from lingjing_solo.planning.ls20_solver import (
    _find_goal_markers,
    _find_player,
    _find_rot_pad,
    _find_shape_pad,
    _find_color_pad,
    _as_grid,
)

ACTION_CN = {
    "RESET": "重置关卡",
    "ACTION1": "向上 ↑（Y−5）",
    "ACTION2": "向下 ↓（Y+5）",
    "ACTION3": "向左 ←（X−5）",
    "ACTION4": "向右 →（X+5）",
    "ACTION5": "点击",
    "ACTION6": "撤销",
}

PHASE_CN = {
    "goal": "前往目标格",
    "pad": "前往修饰台",
    "wait": "等待平台/对齐",
}

MOD_CN = {
    "rot": "旋转台（+90°）",
    "shape": "形状台（换形状）",
    "color": "调色台（换颜色）",
    "none": "无修饰台",
}

LEVEL_INFO = [
    {
        "level": 1,
        "title": "旋转教学",
        "hint": "先到旋转台转 1 次（270°→0°），再站到目标格。",
        "goal": "StartRot=270° → GoalRot=0°",
        "pads": "旋转台 (19,30)",
        "target": "目标 (34,10)",
    },
    {
        "level": 2,
        "title": "远距旋转",
        "hint": "旋转台在右下角，需踩 3 次（0°→270°），再导航到上方目标。",
        "goal": "StartRot=0° → GoalRot=270°",
        "pads": "旋转台 (49,45)",
        "target": "目标 (14,40)",
    },
    {
        "level": 3,
        "title": "颜色+旋转",
        "hint": "先调色再旋转，颜色索引 12，旋转到 180°。",
        "goal": "StartColor=12, GoalRot=180°",
        "pads": "旋转台 (49,10)",
        "target": "目标 (54,50)",
    },
    {
        "level": 4,
        "title": "换形状",
        "hint": "先到形状台切换模板，再到达目标。",
        "goal": "StartShape=4，需形状台",
        "pads": "形状台",
        "target": "目标 (9,5)",
    },
    {
        "level": 5,
        "title": "调色",
        "hint": "旋转 + 调色，目标颜色索引 8。",
        "goal": "GoalColor=8",
        "pads": "旋转台 + 调色台",
        "target": "目标 (54,5)",
    },
    {
        "level": 6,
        "title": "双目标",
        "hint": "依次完成两个目标格（颜色/旋转组合）。",
        "goal": "GoalColor / GoalRot 列表",
        "pads": "旋转台 (34,40)",
        "target": "目标 ×2",
    },
    {
        "level": 7,
        "title": "迷雾",
        "hint": "视野受限（Fog），靠记忆与 BFS 导航。",
        "goal": "Fog=True",
        "pads": "旋转台 (54,10)",
        "target": "目标 (29,50)",
    },
]


def _grid_from_frame(frame) -> list[list[int]]:
    raw = frame.frame or []
    if not raw:
        return []
    layer = raw[-1]
    if isinstance(layer[0], list):
        return [[int(c) for c in row] for row in layer]
    return [[int(c) for c in row] for row in raw]


def _annotate_step(solver, grid_list: list[list[int]], action: str, rationale: str) -> dict:
    g = _as_grid(__import__("numpy").array(grid_list, dtype="int8"))
    player = _find_player(g) if g is not None else None
    goals = _find_goal_markers(g, player) if g is not None and player else []
    rot = _find_rot_pad(g, player) if g is not None else None
    shape = _find_shape_pad(g, player) if g is not None else None
    color = _find_color_pad(g, player) if g is not None else None

    phase = getattr(solver, "_phase", "goal")
    mod_kind = getattr(solver, "mod_kind", "none")
    mod_pad = getattr(solver, "mod_pad", None)
    goal_idx = getattr(solver, "goal_idx", 0)
    pad_entries = getattr(solver, "pad_entries", 0)
    shape_toggles = getattr(solver, "shape_toggles", 0)
    color_toggles = getattr(solver, "color_toggles", 0)
    target_pad = getattr(solver, "target_pad_entries", 0)

    act = action or "—"
    explain_parts: list[str] = []
    if act in ACTION_CN:
        explain_parts.append(f"执行：{ACTION_CN[act]}")
    if phase in PHASE_CN:
        explain_parts.append(f"阶段：{PHASE_CN[phase]}")
    if mod_kind in MOD_CN and mod_kind != "none":
        explain_parts.append(f"修饰：{MOD_CN[mod_kind]}")
        if mod_pad:
            explain_parts.append(f"台坐标 ({mod_pad[0]},{mod_pad[1]})")
        if mod_kind == "rot":
            explain_parts.append(f"已踩旋转台 {pad_entries}/{target_pad} 次")
        elif mod_kind == "shape":
            explain_parts.append(f"已换形状 {shape_toggles}/{target_pad} 次")
        elif mod_kind == "color":
            explain_parts.append(f"已调色 {color_toggles}/{target_pad} 次")
    if goals:
        cur = goals[min(goal_idx, len(goals) - 1)]
        explain_parts.append(f"当前目标 ({cur[0]},{cur[1]})")
    if player:
        explain_parts.append(f"玩家块左上角 ({player[0]},{player[1]})")
    if rationale and "ls20_solver" in rationale:
        explain_parts.append("策略：专用 ls20 BFS 求解器")

    return {
        "player": list(player) if player else None,
        "goals": [list(g) for g in goals],
        "goal_idx": goal_idx,
        "rot_pad": list(rot) if rot else None,
        "shape_pad": list(shape) if shape else None,
        "color_pad": list(color) if color else None,
        "mod_pad": list(mod_pad) if mod_pad else None,
        "mod_kind": mod_kind,
        "phase": phase,
        "pad_entries": pad_entries,
        "target_pad_entries": target_pad,
        "explain": " · ".join(explain_parts),
    }


def _describe_action(
    action: str,
    prev_player: tuple[int, int] | None,
    curr_player: tuple[int, int] | None,
) -> str:
    if not prev_player or not curr_player:
        return ACTION_CN.get(action, action)
    dx = curr_player[0] - prev_player[0]
    dy = curr_player[1] - prev_player[1]
    if dx == 0 and dy == 0:
        return f"{ACTION_CN.get(action, action)}（位置未变，可能踩台或等待）"
    return f"{ACTION_CN.get(action, action)} → 移动到 ({curr_player[0]},{curr_player[1]})"


def record_replay(max_steps: int = 800) -> dict:
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make("ls20")
    spec = importlib.util.spec_from_file_location("ma", ROOT / "agent" / "my_agent.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ag = mod.MyAgent(
        card_id="viz",
        game_id="ls20",
        agent_name="ls20-viz",
        ROOT_URL="x",
        record=False,
        arc_env=env,
        tags=[],
    )
    ag.MAX_ACTIONS = max_steps
    ag.timer = time.time()
    solver = ag.brain.ls20

    frames_data: list[dict] = []
    level_milestones: dict[int, int] = {}
    last_level = 0
    prev_player: tuple[int, int] | None = None

    # 初始帧
    init_frame = ag.frames[-1]
    init_grid = _grid_from_frame(init_frame)
    ann0 = _annotate_step(solver, init_grid, "INIT", "")
    frames_data.append(
        {
            "step": 0,
            "action": "INIT",
            "action_cn": "开局",
            "levels": int(init_frame.levels_completed or 0),
            "state": str(init_frame.state),
            "grid": init_grid,
            "rationale": "",
            **ann0,
        }
    )
    prev_player = tuple(ann0["player"]) if ann0.get("player") else None

    t0 = time.time()
    while not ag.is_done(ag.frames, ag.frames[-1]) and ag.action_counter <= max_steps:
        latest = ag.frames[-1]
        action_obj = ag.choose_action(ag.frames, latest)
        action_name = action_obj.name if hasattr(action_obj, "name") else str(action_obj)
        rationale = getattr(action_obj, "reasoning", "") or getattr(ag.brain, "last_rationale", "")
        if isinstance(rationale, dict):
            rationale = str(rationale.get("reason", rationale))

        frame = ag.take_action(action_obj)
        ag.append_frame(frame)
        ag.action_counter += 1

        grid = _grid_from_frame(frame)
        ann = _annotate_step(solver, grid, action_name, str(rationale))
        lv = int(frame.levels_completed or 0)
        if lv > last_level:
            level_milestones[lv] = ag.action_counter
            last_level = lv

        curr_player = tuple(ann["player"]) if ann.get("player") else None
        action_cn = _describe_action(action_name, prev_player, curr_player)
        prev_player = curr_player

        frames_data.append(
            {
                "step": ag.action_counter,
                "action": action_name,
                "action_cn": action_cn,
                "levels": lv,
                "state": str(frame.state),
                "grid": grid,
                "rationale": str(rationale)[:500],
                "level_milestone": lv > (frames_data[-1]["levels"] if frames_data else 0),
                **ann,
            }
        )

    elapsed = time.time() - t0
    final = ag.frames[-1]
    return {
        "game": "ls20",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "max_steps": max_steps,
        "total_steps": ag.action_counter,
        "levels_completed": int(final.levels_completed or 0),
        "elapsed_sec": round(elapsed, 2),
        "level_milestones": level_milestones,
        "level_info": LEVEL_INFO,
        "final_state": str(final.state),
        "frames": frames_data,
    }


def _serve_replay(port: int) -> None:
    from flask import Flask, send_from_directory

    static = ROOT / "ui" / "static"
    app = Flask(__name__, static_folder=str(static), static_url_path="/static")

    @app.get("/")
    def index():
        return send_from_directory(static, "ls20_viewer.html")

    @app.get("/api/replay")
    def replay():
        path = static / "ls20_replay.json"
        return path.read_text(encoding="utf-8")

    url = f"http://127.0.0.1:{port}"
    print(f"LS20 可视化 → {url}")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True, use_reloader=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-steps", type=int, default=800)
    p.add_argument("--out", type=Path, default=ROOT / "ui" / "static" / "ls20_replay.json")
    p.add_argument("--serve", action="store_true", help="录制后启动本地查看服务器")
    p.add_argument("--serve-only", action="store_true", help="仅启动服务器，使用已有 replay.json")
    p.add_argument("--open", action="store_true", help="录制后自动打开浏览器")
    p.add_argument("--port", type=int, default=8766)
    args = p.parse_args()

    if args.serve_only:
        replay_path = args.out
        if not replay_path.exists():
            raise SystemExit(f"回放文件不存在: {replay_path}，请先运行不带 --serve-only 的命令录制。")
        print(f"使用已有回放: {replay_path}")
        if args.open:
            threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{args.port}")).start()
        _serve_replay(args.port)
        return

    print(f"录制 ls20 求解过程（max_steps={args.max_steps}）…")
    data = record_replay(args.max_steps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(
        f"已保存 {args.out} | 步数={data['total_steps']} "
        f"关卡={data['levels_completed']}/7 耗时={data['elapsed_sec']}s"
    )
    print(f"关卡里程碑: {data['level_milestones']}")

    if args.serve:
        if args.open:
            threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{args.port}")).start()
        _serve_replay(args.port)
    elif args.open:
        viewer = ROOT / "ui" / "static" / "ls20_viewer.html"
        webbrowser.open(viewer.as_uri())
        print(f"已打开 {viewer}")
        print("若数据未加载，请用 --serve 启动服务器。")


if __name__ == "__main__":
    main()
