"""统一实时仪表盘 — 双游戏对比 + 成败判定 + 因果面板（非录制，实时画面）。

用法:
  .\\.venv\\Scripts\\python.exe scripts\\dashboard_ui.py
  浏览器打开 http://127.0.0.1:8780
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
import threading
import time
import traceback
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
sys.path.insert(0, str(VENDOR))

from scripts.mind_zh import (
    ACTION_CN,
    extract_mind,
    mind_from_reasoning,
    state_cn,
    build_verdict,
    GAME_META,
)

from flask import Flask, Response, jsonify, request, send_from_directory


def load_dotenv() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


load_dotenv()

_lock = threading.Lock()
_seq = 0
_stop = threading.Event()
_workers: list[threading.Thread] = []


def _criteria_doc() -> list[dict[str, str]]:
    return [
        {
            "key": "WIN",
            "title": "完全成功",
            "desc": "游戏 state=WIN → 官方认定该游戏任务完成",
        },
        {
            "key": "levels",
            "title": "过关数",
            "desc": "levels_completed 增加 = 至少完成了一关（部分成功）",
        },
        {
            "key": "GAME_OVER",
            "title": "游戏内失败",
            "desc": "无效移动太多 / 步数耗尽 → GAME_OVER",
        },
        {
            "key": "max_steps",
            "title": "Agent 步数上限",
            "desc": "达到 play 脚本 max_steps 会强制停止（不等于游戏 WIN）",
        },
        {
            "key": "delta",
            "title": "画面是否变化",
            "desc": "delta_pixels=0 连续多步 → 动作可能无效或在打转",
        },
        {
            "key": "edges",
            "title": "因果边（vc33 等）",
            "desc": "edges 上升 = 正在记录「状态+动作→结果」（边玩边学）",
        },
        {
            "key": "ls20",
            "title": "ls20 专用",
            "desc": "走 ls20_solver，不看因果边；看旋转台次数、路径是否通",
        },
    ]


def _empty_slot(slot_id: str, game_id: str = "ls20") -> dict[str, Any]:
    return {
        "slot": slot_id,
        "running": False,
        "finished": False,
        "game_id": game_id,
        "step": 0,
        "levels": 0,
        "state": "IDLE",
        "state_cn": "未开始",
        "action": None,
        "action_cn": "—",
        "grid": [],
        "fps": 0.0,
        "mind": {},
        "mind_history": [],
        "verdict": {},
        "error": None,
        "log": deque(maxlen=80),
    }


STATE: dict[str, Any] = {
    "slots": {
        "a": _empty_slot("a", "ls20"),
        "b": _empty_slot("b", "vc33"),
    },
    "max_steps": 400,
    "delay_ms": 120,
    "criteria": _criteria_doc(),
}


def _publish_global(**kwargs: Any) -> None:
    global _seq
    with _lock:
        STATE.update(kwargs)
        _seq += 1
        STATE["_seq"] = _seq


def _publish_slot(slot_id: str, **kwargs: Any) -> None:
    global _seq
    with _lock:
        slot = STATE["slots"][slot_id]
        slot.update(kwargs)
        _seq += 1
        STATE["_seq"] = _seq


def _log_slot(slot_id: str, msg: str) -> None:
    global _seq
    with _lock:
        STATE["slots"][slot_id]["log"].appendleft({
            "t": time.strftime("%H:%M:%S"),
            "msg": msg,
        })
        _seq += 1
        STATE["_seq"] = _seq


def _grid_from_frame(frame) -> list[list[int]]:
    raw = frame.frame or []
    if not raw:
        return []
    layer = raw[-1]
    if isinstance(layer[0], list):
        return [[int(c) for c in row] for row in layer]
    return [[int(c) for c in row] for row in raw]


def load_agent_class():
    spec = importlib.util.spec_from_file_location(
        "user_agent_module", ROOT / "agent" / "my_agent.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load agent/my_agent.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.MyAgent


def run_slot(slot_id: str, game_id: str, max_steps: int, delay_ms: int) -> None:
    import arc_agi
    from arc_agi import OperationMode
    from arcengine import GameState

    gid = game_id.split("-")[0]
    try:
        _publish_slot(
            slot_id,
            running=True,
            finished=False,
            error=None,
            game_id=gid,
            step=0,
            levels=0,
            state="STARTING",
            state_cn="启动中",
            grid=[],
            mind_history=[],
        )
        _log_slot(slot_id, f"开始 {gid}，最多 {max_steps} 步（实时画面）")

        MyAgent = load_agent_class()
        MyAgent.MAX_ACTIONS = max_steps

        arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
        env = arc.make(gid)
        if env is None:
            raise RuntimeError(f"无法创建环境 {gid}")

        agent = MyAgent(
            card_id=f"dash-{slot_id}",
            game_id=gid,
            agent_name=f"Dashboard.{slot_id}.{gid}",
            ROOT_URL="http://localhost",
            record=False,
            arc_env=env,
            tags=["dashboard"],
        )

        mind_history: list[dict] = []
        last_levels = 0
        t0 = time.time()

        while (
            not agent.is_done(agent.frames, agent.frames[-1])
            and agent.action_counter <= max_steps
            and not _stop.is_set()
        ):
            latest = agent._convert_raw_frame_data(agent.arc_env.observation_space)
            action = agent.choose_action(agent.frames, latest)
            reason = getattr(action, "reasoning", None)
            frame = agent.take_action(action)
            if frame:
                agent.append_frame(frame)
            agent.action_counter += 1

            grid = _grid_from_frame(frame or latest)
            levels = int(getattr(frame, "levels_completed", 0) or 0) if frame else 0
            st = (
                frame.state.name
                if frame and hasattr(frame.state, "name")
                else str(getattr(frame, "state", "?"))
            )
            fps = agent.action_counter / max(time.time() - t0, 0.001)

            mind = extract_mind(agent, frame or latest, action.name)
            mind = {**mind, **mind_from_reasoning(reason, mind)}
            mind_history.append({
                "step": agent.action_counter,
                "edges": mind.get("edges", 0),
                "visited": mind.get("visited", 0),
                "rules": mind.get("rules", 0),
            })

            verdict = build_verdict(
                agent, frame or latest, gid,
                agent.action_counter, max_steps, running=True,
            )

            if levels > last_levels:
                _log_slot(
                    slot_id,
                    f"★ 过关！第 {levels} 关（第 {agent.action_counter} 步）",
                )
                last_levels = levels

            _publish_slot(
                slot_id,
                step=agent.action_counter,
                levels=levels,
                state=st,
                state_cn=state_cn(st),
                action=action.name,
                action_cn=ACTION_CN.get(action.name, action.name),
                grid=grid,
                fps=round(fps, 2),
                mind=mind,
                mind_history=mind_history,
                verdict=verdict,
            )

            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)

            if frame and frame.state is GameState.WIN:
                _log_slot(slot_id, f"全胜！{verdict['verdict_cn']}")
                break

        final = agent.frames[-1]
        final_verdict = build_verdict(
            agent, final, gid, agent.action_counter, max_steps, running=False,
        )
        _publish_slot(
            slot_id,
            running=False,
            finished=True,
            state=final.state.name if hasattr(final.state, "name") else str(final.state),
            state_cn=state_cn(str(final.state)),
            levels=int(final.levels_completed or 0),
            step=agent.action_counter,
            verdict=final_verdict,
            mind_history=mind_history,
        )
        _log_slot(
            slot_id,
            f"结束 · {final_verdict['verdict_cn']} · "
            f"{final_verdict['levels']}/{final_verdict['target_levels']} 关",
        )
    except Exception as e:
        _publish_slot(slot_id, running=False, finished=True, error=str(e))
        _log_slot(slot_id, f"错误: {e}")
        traceback.print_exc()


def start_compare(game_a: str, game_b: str, max_steps: int, delay_ms: int) -> None:
    global _workers
    if any(STATE["slots"][s]["running"] for s in ("a", "b")):
        raise RuntimeError("已有对局在运行，请先停止")
    _stop.clear()
    _publish_global(max_steps=max_steps, delay_ms=delay_ms)
    for sid, gid in (("a", game_a), ("b", game_b)):
        STATE["slots"][sid] = _empty_slot(sid, gid.split("-")[0])
    t_a = threading.Thread(
        target=run_slot, args=("a", game_a, max_steps, delay_ms), daemon=True
    )
    t_b = threading.Thread(
        target=run_slot, args=("b", game_b, max_steps, delay_ms), daemon=True
    )
    _workers = [t_a, t_b]
    t_a.start()
    t_b.start()


app = Flask(__name__, static_folder=str(ROOT / "ui" / "static"), static_url_path="/static")


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "dashboard.html")


@app.get("/api/state")
def api_state():
    with _lock:
        payload = deepcopy(STATE)
        for sid in payload["slots"]:
            payload["slots"][sid]["log"] = list(STATE["slots"][sid]["log"])
        payload["game_meta"] = GAME_META
    return jsonify(payload)


@app.get("/api/stream")
def api_stream():
    def gen():
        last = -1
        while True:
            with _lock:
                seq = STATE.get("_seq", 0)
                if seq != last:
                    last = seq
                    payload = deepcopy(STATE)
                    for sid in payload["slots"]:
                        payload["slots"][sid]["log"] = list(
                            STATE["slots"][sid]["log"]
                        )
                    payload["game_meta"] = GAME_META
                    yield f"data: {json.dumps(payload, default=str)}\n\n"
            time.sleep(0.04)

    return Response(gen(), mimetype="text/event-stream")


@app.post("/api/start")
def api_start():
    data = request.get_json(force=True, silent=True) or {}
    ga = str(data.get("game_a", "ls20")).split("-")[0]
    gb = str(data.get("game_b", "vc33")).split("-")[0]
    max_steps = int(data.get("max_steps", STATE.get("max_steps", 400)))
    delay_ms = int(data.get("delay_ms", STATE.get("delay_ms", 120)))
    try:
        start_compare(ga, gb, max_steps, delay_ms)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.post("/api/stop")
def api_stop():
    _stop.set()
    with _lock:
        for sid in STATE["slots"]:
            STATE["slots"][sid]["log"].appendleft({
                "t": time.strftime("%H:%M:%S"),
                "msg": "已请求停止",
            })
    return jsonify({"ok": True})


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8780)
    p.add_argument("--game-a", default="ls20")
    p.add_argument("--game-b", default="vc33")
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--delay-ms", type=int, default=120)
    p.add_argument("--autostart", action="store_true")
    args = p.parse_args()

    _publish_global(max_steps=args.max_steps, delay_ms=args.delay_ms)

    if args.autostart:
        threading.Timer(
            1.5,
            lambda: start_compare(
                args.game_a, args.game_b, args.max_steps, args.delay_ms
            ),
        ).start()

    url = f"http://127.0.0.1:{args.port}"
    print(f"统一实时仪表盘 → {url}")
    print(f"对比: {args.game_a} vs {args.game_b} · 实时画面，非录制")
    app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
