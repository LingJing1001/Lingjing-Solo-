"""全游戏摸底结果 UI — 表格 + 分数条形图 + 提分优先级。

用法:
  .\\.venv\\Scripts\\python.exe scripts\\games_benchmark_ui.py --port 8781
  浏览器: http://127.0.0.1:8781
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify, request, send_from_directory

from scripts.benchmark_all_games import OUT_JSON, TEAM_OUT_JSON, run_all

app = Flask(__name__, static_folder=str(ROOT / "ui" / "static"))
_lock = threading.Lock()
_running = False
_last_error: str | None = None
_active_json = TEAM_OUT_JSON


def _read_data() -> dict:
    path = _active_json if _active_json.exists() else OUT_JSON
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data["_source_file"] = path.name
            return data
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "running": False,
        "games": [],
        "aggregate_score": 0,
        "total_games": 0,
        "message": "尚无数据，请点击「重新摸底」",
    }


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "games_benchmark.html")


@app.route("/api/data")
def api_data():
    return jsonify(_read_data())


@app.route("/api/run", methods=["POST"])
def api_run():
    global _running, _last_error
    with _lock:
        if _running:
            return jsonify({"ok": False, "error": "摸底正在运行中"}), 409
        _running = True
        _last_error = None

    body = request.get_json(silent=True) or {}
    max_steps = int(body.get("max_steps", 400))
    which = str(body.get("agent", "team"))
    if which == "team":
        agent_path = ROOT / "agent" / "lingjing_team_agent.py"
        out_path = TEAM_OUT_JSON
        label = "灵境战队AGI课题探索"
    else:
        agent_path = ROOT / "agent" / "my_agent.py"
        out_path = OUT_JSON
        label = "Starter my_agent (ls20 solver)"

    global _active_json
    _active_json = out_path

    def worker():
        global _running, _last_error
        try:
            report = run_all(
                max_steps=max_steps,
                agent_path=agent_path,
                out_json=out_path,
                agent_label=label,
            )
            report["aggregate_score_official"] = report["aggregate_score"]
            out_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            _last_error = f"{type(exc).__name__}: {exc}"
            logging.exception("benchmark failed")
        finally:
            with _lock:
                _running = False

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"ok": True, "max_steps": max_steps, "agent": label, "out": out_path.name})


@app.route("/api/status")
def api_status():
    with _lock:
        return jsonify({"running": _running, "error": _last_error})


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8781)
    p.add_argument("--run-on-start", action="store_true", help="启动时自动跑全游戏摸底")
    p.add_argument("--max-steps", type=int, default=400)
    args = p.parse_args()

    if args.run_on_start and not OUT_JSON.exists():
        def _bg():
            run_all(max_steps=args.max_steps)
        threading.Thread(target=_bg, daemon=True).start()

    print(f"Games benchmark UI: http://127.0.0.1:{args.port}")
    app.run(host="127.0.0.1", port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
