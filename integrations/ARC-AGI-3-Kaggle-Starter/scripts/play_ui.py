"""Aether Live UI — watch the agent play ARC-AGI-3 in the browser.

Usage:
  .\\.venv\\Scripts\\python.exe scripts\\play_ui.py
  .\\.venv\\Scripts\\python.exe scripts\\play_ui.py --game ls20 --max-steps 500 --port 8765

Then open http://127.0.0.1:8765
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
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
sys.path.insert(0, str(VENDOR))

from scripts.mind_zh import extract_mind, mind_from_reasoning, state_cn, ACTION_CN

from flask import Flask, Response, jsonify, request, send_from_directory

# ── dotenv ──────────────────────────────────────────────────────────────────

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
        existing = os.environ.get(k, "")
        weak = len(existing) < 16 or existing.lower() in {"sk-xxxx", "sk-xxx", "0"}
        if k and (k not in os.environ or weak):
            os.environ[k] = v


load_dotenv()

# ── shared runtime state ────────────────────────────────────────────────────

ARC_COLORS = {
    0: "#FFFFFF",
    1: "#CCCCCC",
    2: "#999999",
    3: "#666666",
    4: "#333333",
    5: "#000000",
    6: "#E53AA3",
    7: "#FF7BCC",
    8: "#F93C31",
    9: "#1E93FF",
    10: "#88D8F1",
    11: "#FFDC00",
    12: "#FF851B",
    13: "#921231",
    14: "#4FCC30",
    15: "#A356D6",
}

STATE: dict[str, Any] = {
    "running": False,
    "game_id": None,
    "step": 0,
    "levels": 0,
    "win_levels": None,
    "state": "IDLE",
    "action": None,
    "reasoning": None,
    "grid": [],
    "llm": {
        "enabled": False,
        "provider": None,
        "model": None,
        "calls": 0,
        "last_error": "",
    },
    "score_hint": None,
    "fps": 0.0,
    "error": None,
    "finished": False,
    "log": deque(maxlen=200),
    "mind": {},
    "mind_history": [],
}
_lock = threading.Lock()
_stop = threading.Event()
_worker: Optional[threading.Thread] = None
_seq = 0


def _publish(**kwargs: Any) -> None:
    global _seq
    with _lock:
        STATE.update(kwargs)
        _seq += 1
        STATE["_seq"] = _seq


def _log(msg: str) -> None:
    global _seq
    with _lock:
        STATE["log"].appendleft({"t": time.strftime("%H:%M:%S"), "msg": msg})
        _seq += 1
        STATE["_seq"] = _seq


def _grid_from_frame(frame) -> list[list[int]]:
    raw = frame.frame or []
    if not raw:
        return []
    layer = raw[-1]
    return [[int(c) for c in row] for row in layer]


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


def _log_cn(msg: str) -> None:
    """中文事件日志。"""
    _log(msg)


def run_game(game_id: str, max_steps: int, delay_ms: int) -> None:
    import arc_agi
    from arc_agi import OperationMode
    from arcengine import GameState

    try:
        _publish(
            running=True,
            finished=False,
            error=None,
            game_id=game_id,
            step=0,
            levels=0,
            state="STARTING",
            action=None,
            reasoning=None,
            grid=[],
        )
        _log_cn(f"开始游戏 {game_id}，最多 {max_steps} 步")

        MyAgent = load_agent_class()
        MyAgent.MAX_ACTIONS = max_steps

        arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
        env = arc.make(game_id)
        if env is None:
            raise RuntimeError(f"Could not create env for {game_id}")

        agent = MyAgent(
            card_id="ui-live",
            game_id=game_id,
            agent_name=f"AetherUI.{game_id}",
            ROOT_URL="http://localhost",
            record=False,
            arc_env=env,
            tags=["ui"],
        )

        llm = getattr(agent, "llm", None)
        _publish(
            llm={
                "enabled": bool(getattr(llm, "enabled", False)),
                "provider": getattr(llm, "provider", None),
                "model": getattr(llm, "model", None),
                "calls": 0,
                "last_error": "",
            }
        )
        _log_cn(
            f"LLM={'开' if getattr(llm, 'enabled', False) else '关'} "
            f"模型={getattr(llm, 'model', None) or '无'}"
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
            win_levels = getattr(frame, "win_levels", None) if frame else None
            st = (
                frame.state.name
                if frame and hasattr(frame.state, "name")
                else str(getattr(frame, "state", "?"))
            )
            elapsed = max(time.time() - t0, 0.001)
            fps = agent.action_counter / elapsed

            llm_info = {
                "enabled": bool(getattr(llm, "enabled", False)),
                "provider": getattr(llm, "provider", None),
                "model": getattr(llm, "model", None),
                "calls": int(getattr(llm, "calls", 0) or 0),
                "last_error": str(getattr(llm, "last_error", "") or ""),
            }

            mind = extract_mind(agent, frame or latest, action.name)
            mind = {**mind, **mind_from_reasoning(reason, mind)}
            mind_history.append({
                "step": agent.action_counter,
                "edges": mind.get("edges", 0),
                "visited": mind.get("visited", 0),
                "transitions": mind.get("transitions", 0),
                "rules": mind.get("rules", 0),
                "c": mind.get("c_value", 0),
            })

            if levels > last_levels:
                _log_cn(
                    f"★ 过关！第 {levels} 关完成（第 {agent.action_counter} 步）"
                )
                last_levels = levels

            _publish(
                step=agent.action_counter,
                levels=levels,
                win_levels=win_levels,
                state=st,
                state_cn=state_cn(st),
                action=action.name,
                action_cn=ACTION_CN.get(action.name, action.name),
                reasoning=reason,
                grid=grid,
                fps=round(fps, 2),
                llm=llm_info,
                mind=mind,
                mind_history=mind_history,
            )
            if agent.action_counter <= 3 or agent.action_counter % 20 == 0:
                _log_cn(
                    f"步 {agent.action_counter} · {mind.get('why_cn', '')} · "
                    f"因果边={mind.get('edges', 0)} · 已过{mind.get('levels', 0)}关"
                )

            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)

            if frame and frame.state is GameState.WIN:
                _log_cn(f"全胜通关！共 {levels} 关")
                break

        final = agent.frames[-1]
        final_levels = int(final.levels_completed or 0)
        final_st = final.state.name if hasattr(final.state, "name") else str(final.state)
        final_mind = extract_mind(agent, final, "")
        _publish(
            running=False,
            finished=True,
            state=final_st,
            state_cn=state_cn(final_st),
            levels=final_levels,
            step=agent.action_counter,
            mind=final_mind,
            mind_history=mind_history,
        )
        try:
            sc = arc.get_scorecard()
            score = sc.score if hasattr(sc, "score") else sc
            _publish(score_hint=score)
            _log_cn(
                f"结束 · 过了 {final_levels} 关 · "
                f"{final_mind.get('success_cn', '')} · 分数={score}"
            )
        except Exception:
            _log_cn(f"结束 · 过了 {final_levels} 关 · {final_mind.get('success_cn', '')}")
    except Exception as e:
        _publish(running=False, finished=True, error=str(e))
        _log(f"ERROR: {e}")
        traceback.print_exc()


def start_worker(game_id: str, max_steps: int, delay_ms: int) -> None:
    global _worker
    if STATE["running"]:
        raise RuntimeError("Already running")
    _stop.clear()
    _worker = threading.Thread(
        target=run_game, args=(game_id, max_steps, delay_ms), daemon=True
    )
    _worker.start()


# ── Flask app ───────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder=str(ROOT / "ui" / "static"), static_url_path="/static")


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/state")
def api_state():
    with _lock:
        payload = dict(STATE)
        payload["log"] = list(STATE["log"])
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
                    payload = dict(STATE)
                    payload["log"] = list(STATE["log"])
                    yield f"data: {json.dumps(payload, default=str)}\n\n"
            time.sleep(0.05)

    return Response(gen(), mimetype="text/event-stream")


@app.post("/api/start")
def api_start():
    data = request.get_json(force=True, silent=True) or {}
    game = str(data.get("game", "ls20")).split("-")[0]
    max_steps = int(data.get("max_steps", 500))
    delay_ms = int(data.get("delay_ms", 0))
    try:
        start_worker(game, max_steps, delay_ms)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.post("/api/stop")
def api_stop():
    _stop.set()
    _log_cn("已请求停止")
    return jsonify({"ok": True})


@app.get("/api/colors")
def api_colors():
    return jsonify(ARC_COLORS)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", default="ls20")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--delay-ms", type=int, default=0)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-autostart", action="store_true")
    args = p.parse_args()

    if not args.no_autostart:
        # slight delay so server is up
        threading.Timer(
            1.0, lambda: start_worker(args.game, args.max_steps, args.delay_ms)
        ).start()

    url = f"http://127.0.0.1:{args.port}"
    print(f"Aether Live UI → {url}")
    print(f"Autostart game={args.game} max_steps={args.max_steps}")
    app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
