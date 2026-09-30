"""录制 vc33 自主实验 + 因果图增长，生成交互式可视化。

vc33 没有专用 solver，会走 explore/search，适合展示因果边增长。

用法:
  .\\.venv\\Scripts\\python.exe scripts\\visualize_vc33.py --max-steps 250
  .\\.venv\\Scripts\\python.exe scripts\\visualize_vc33.py --serve-only --open
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

from scripts.mind_zh import extract_mind, mind_from_reasoning, state_cn, ACTION_CN


def _grid_from_frame(frame) -> list[list[int]]:
    raw = frame.frame or []
    if not raw:
        return []
    layer = raw[-1]
    if isinstance(layer[0], list):
        return [[int(c) for c in row] for row in layer]
    return [[int(c) for c in row] for row in raw]


def record_vc33(max_steps: int = 250, grid_every: int = 5) -> dict:
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make("vc33")
    spec = importlib.util.spec_from_file_location("ma", ROOT / "agent" / "my_agent.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ag = mod.MyAgent(
        card_id="vc33-viz",
        game_id="vc33",
        agent_name="vc33-viz",
        ROOT_URL="x",
        record=False,
        arc_env=env,
        tags=[],
    )
    ag.MAX_ACTIONS = max_steps
    ag.timer = time.time()

    frames_data: list[dict] = []
    mind_history: list[dict] = []
    level_milestones: dict[int, int] = {}
    last_level = 0
    t0 = time.time()

    init = ag.frames[-1]
    init_mind = extract_mind(ag, init, "INIT")
    frames_data.append({
        "step": 0,
        "action": "INIT",
        "action_cn": "开局",
        "levels": 0,
        "state": str(init.state),
        "state_cn": state_cn(str(init.state)),
        "grid": _grid_from_frame(init),
        "mind": init_mind,
    })
    mind_history.append({
        "step": 0,
        "edges": init_mind.get("edges", 0),
        "visited": init_mind.get("visited", 0),
        "transitions": init_mind.get("transitions", 0),
        "rules": init_mind.get("rules", 0),
        "c": init_mind.get("c_value", 0),
        "why_cn": init_mind.get("why_cn", ""),
    })

    while not ag.is_done(ag.frames, ag.frames[-1]) and ag.action_counter <= max_steps:
        latest = ag.frames[-1]
        action_obj = ag.choose_action(ag.frames, latest)
        action_name = action_obj.name if hasattr(action_obj, "name") else str(action_obj)
        reason = getattr(action_obj, "reasoning", "")

        frame = ag.take_action(action_obj)
        ag.append_frame(frame)
        ag.action_counter += 1

        lv = int(frame.levels_completed or 0)
        if lv > last_level:
            level_milestones[lv] = ag.action_counter
            last_level = lv

        mind = extract_mind(ag, frame, action_name)
        mind = {**mind, **mind_from_reasoning(reason, mind)}

        mind_history.append({
            "step": ag.action_counter,
            "edges": mind.get("edges", 0),
            "visited": mind.get("visited", 0),
            "transitions": mind.get("transitions", 0),
            "rules": mind.get("rules", 0),
            "c": mind.get("c_value", 0),
            "why_cn": mind.get("why_cn", ""),
        })

        store_grid = (
            ag.action_counter <= 3
            or ag.action_counter % grid_every == 0
            or lv > frames_data[-1].get("levels", 0)
        )

        frames_data.append({
            "step": ag.action_counter,
            "action": action_name,
            "action_cn": ACTION_CN.get(action_name, action_name),
            "levels": lv,
            "state": str(frame.state),
            "state_cn": state_cn(str(frame.state)),
            "grid": _grid_from_frame(frame) if store_grid else None,
            "mind": mind,
            "level_up": lv > frames_data[-1].get("levels", 0),
        })

    elapsed = time.time() - t0
    final = ag.frames[-1]
    final_mind = extract_mind(ag, final, "")

    return {
        "game": "vc33",
        "title": "vc33 自主实验 + 因果图增长",
        "description": (
            "vc33 无专用解题器，Agent 通过自主实验积累因果边（predict_graph），"
            "观察 edges/visited/rules 曲线上升即表示「边玩边学」。"
        ),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "max_steps": max_steps,
        "total_steps": ag.action_counter,
        "levels_completed": int(final.levels_completed or 0),
        "elapsed_sec": round(elapsed, 2),
        "level_milestones": level_milestones,
        "final_state": str(final.state),
        "final_success_cn": final_mind.get("success_cn", ""),
        "mind_history": mind_history,
        "frames": frames_data,
        "success_guide": {
            "win": "state=WIN → 完全成功，关全通",
            "partial": "levels>0 且 NOT_FINISHED → 部分成功",
            "fail": "GAME_OVER → 失败，步数耗尽",
        },
    }


def _serve(port: int) -> None:
    from flask import Flask, send_from_directory

    static = ROOT / "ui" / "static"
    app = Flask(__name__, static_folder=str(static), static_url_path="/static")

    @app.get("/")
    def index():
        return send_from_directory(static, "vc33_viewer.html")

    @app.get("/api/replay")
    def replay():
        return (static / "vc33_replay.json").read_text(encoding="utf-8")

    url = f"http://127.0.0.1:{port}"
    print(f"vc33 因果可视化 → {url}")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True, use_reloader=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-steps", type=int, default=250)
    p.add_argument("--grid-every", type=int, default=5, help="每隔 N 步存一帧画面")
    p.add_argument("--out", type=Path, default=ROOT / "ui" / "static" / "vc33_replay.json")
    p.add_argument("--serve", action="store_true")
    p.add_argument("--serve-only", action="store_true")
    p.add_argument("--open", action="store_true")
    p.add_argument("--port", type=int, default=8767)
    args = p.parse_args()

    if args.serve_only:
        if not args.out.exists():
            raise SystemExit(f"回放不存在: {args.out}，请先录制。")
        print(f"使用已有: {args.out}")
        if args.open:
            threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{args.port}")).start()
        _serve(args.port)
        return

    print(f"录制 vc33（max_steps={args.max_steps}）…")
    data = record_vc33(args.max_steps, args.grid_every)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(
        f"已保存 {args.out} | 步数={data['total_steps']} "
        f"关卡={data['levels_completed']} 因果边终值={data['mind_history'][-1]['edges']}"
    )
    print(f"成功判定: {data['final_success_cn']}")

    if args.serve:
        if args.open:
            threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{args.port}")).start()
        _serve(args.port)
    elif args.open:
        webbrowser.open((ROOT / "ui" / "static" / "vc33_viewer.html").as_uri())


if __name__ == "__main__":
    main()
