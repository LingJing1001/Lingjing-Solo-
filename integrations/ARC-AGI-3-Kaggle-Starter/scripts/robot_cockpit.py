"""全量测试 → 因果模型 → LLM 推演 一体化驾驶舱（可视化机器人）。

用法:
  .\\.venv\\Scripts\\python.exe scripts\\robot_cockpit.py --port 8799 --open
  浏览器: http://127.0.0.1:8799

流程按钮会：build_notebook → 全游戏摸底(带动作轨迹) → 更新因果图 → MiniMax 推演。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import threading
import time
import traceback
import webbrowser
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
sys.path.insert(0, str(VENDOR))

from flask import Flask, Response, jsonify, request, send_file, send_from_directory
from urllib.parse import quote

from lingjing_solo.llm_client import chat_completion, load_dotenv, resolve_credentials
from lingjing_solo.world_model.update_causal import UpdateCausalModel
from scripts.benchmark_all_games import run_all

STATIC = ROOT / "ui" / "static"
LIVE_PATH = STATIC / "robot_live.json"
BENCH_PATH = STATIC / "games_benchmark_submit_v7.json"
CAUSAL_PATH = ROOT / "R5更新反思报告_LLM_causal.json"
LLM_PATH = ROOT / "R5机器人推演_LLM.md"

app = Flask(__name__, static_folder=str(STATIC))
_lock = threading.Lock()
_state: dict[str, Any] = {
    "phase": "idle",
    "message": "待命。点击「启动全流程」开始。",
    "started_at": None,
    "finished_at": None,
    "error": None,
    "build": None,
    "bench": None,
    "causal": None,
    "llm": None,
    "actions": [],
    "events": [],
}
_running = False
_action_buf: deque[dict[str, Any]] = deque(maxlen=2500)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _llm_document(advice: str, report: Optional[dict[str, Any]] = None) -> str:
    agg = (report or {}).get("aggregate_score", "—")
    return (
        f"# R5 机器人推演（测试反哺）\n\n"
        f"- 时间: {_now()}\n"
        f"- aggregate: {agg}\n\n"
        f"{advice.rstrip()}\n"
    )


def _hydrate_llm_from_disk() -> None:
    """启动时若已有推演 MD，载入状态以便点击导出。"""
    if not LLM_PATH.exists():
        return
    try:
        text = LLM_PATH.read_text(encoding="utf-8")
    except OSError:
        return
    if not text.strip():
        return
    with _lock:
        _state["llm"] = {
            "model": "(磁盘缓存)",
            "markdown": text,
            "document": text,
            "path": str(LLM_PATH.relative_to(ROOT)),
            "filename": LLM_PATH.name,
        }


def _emit(phase: str, message: str, **extra: Any) -> None:
    with _lock:
        _state["phase"] = phase
        _state["message"] = message
        evt = {"t": _now(), "phase": phase, "message": message, **extra}
        _state.setdefault("events", []).append(evt)
        if len(_state["events"]) > 400:
            _state["events"] = _state["events"][-400:]
        if "actions" in extra:
            pass
        _state["actions"] = list(_action_buf)[-200:]
        snap = dict(_state)
    LIVE_PATH.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")


def _patch_agent_trace(agent: Any, game_id: str) -> None:
    orig = agent.choose_action

    def traced(frames, latest_frame):
        action = orig(frames, latest_frame)
        try:
            name = getattr(action, "name", str(action))
            levels = int(getattr(latest_frame, "levels_completed", 0) or 0)
            reason = str(getattr(action, "reasoning", "") or "")[:180]
            rec = {
                "t": _now(),
                "game_id": game_id,
                "step": int(getattr(agent, "action_counter", 0) or 0),
                "levels": levels,
                "action": name,
                "reasoning": reason,
            }
            _action_buf.append(rec)
            # 轻量刷新（每 8 步写一次，避免 IO 过密）
            if rec["step"] % 8 == 0:
                with _lock:
                    _state["actions"] = list(_action_buf)[-200:]
                    _state["message"] = (
                        f"对局 {game_id} step={rec['step']} "
                        f"L={levels} act={name}"
                    )
                LIVE_PATH.write_text(
                    json.dumps(_state, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        except Exception:  # noqa: BLE001
            pass
        return action

    agent.choose_action = traced  # type: ignore[method-assign]


def _instrument_run_all(max_steps: int) -> dict[str, Any]:
    """run_all with per-game agent patching via monkeypatch on loader."""
    import scripts.benchmark_all_games as bag

    agent_path = ROOT / "agent" / "my_agent.py"
    spec = importlib.util.spec_from_file_location("robot_agent", agent_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    Orig = mod.MyAgent
    orig_init = Orig.__init__

    def init_wrap(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        gid = str(kwargs.get("game_id") or getattr(self, "game_id", "") or "")
        _patch_agent_trace(self, gid.split("-")[0])

    Orig.__init__ = init_wrap  # type: ignore[method-assign]

    def patched_load(_path=None):
        return Orig

    old_load = bag._load_agent
    bag._load_agent = patched_load  # type: ignore[assignment]

    def on_progress(partial: dict[str, Any]) -> None:
        with _lock:
            _state["bench"] = partial
            _state["phase"] = "bench"
            done = partial.get("games_finished", 0)
            total = partial.get("total_games", 0)
            agg = partial.get("aggregate_score", 0)
            _state["message"] = (
                f"全量测试中 {done}/{total} · aggregate≈{agg} · "
                f"levels={partial.get('total_levels_completed', 0)}"
            )
            _state["actions"] = list(_action_buf)[-200:]
        LIVE_PATH.write_text(json.dumps(_state, ensure_ascii=False, indent=2), encoding="utf-8")
        BENCH_PATH.write_text(json.dumps(partial, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        report = bag.run_all(
            max_steps=max_steps,
            agent_path=agent_path,
            out_json=BENCH_PATH,
            agent_label=f"submit {getattr(mod, 'BUILD_TAG', 'my_agent')}",
            on_progress=on_progress,
        )
    finally:
        Orig.__init__ = orig_init  # type: ignore[method-assign]
        bag._load_agent = old_load  # type: ignore[assignment]
    return report


def _ingest_bench_to_causal(report: dict[str, Any]) -> UpdateCausalModel:
    causal = UpdateCausalModel()
    # 复用默认时间线种子后再喂测试结果
    from agent.r5_update_reflect_agent import DEFAULT_UPDATE_EVENTS

    for ev in DEFAULT_UPDATE_EVENTS:
        causal.ingest_event(ev)

    games = report.get("games") or []
    for g in games:
        gid = g.get("game_id")
        levels = int(g.get("levels_completed") or 0)
        score = float(g.get("game_score") or 0)
        state = str(g.get("state") or "")
        if gid == "ls20" and "WIN" in state:
            causal.intervene("HARDCODED_LOADED", "run_l1_l7", "MULTI_LEVEL_SCORE", weight=1.0, note="bench ls20 WIN")
            causal.observe_state("MULTI_LEVEL_SCORE")
            causal.mark_hypothesis("C5", "supported", boost=0.0)
        if gid == "ar25" and "WIN" in state:
            causal.intervene("HARDCODED_LOADED", "run_ar25_l1_l8", "MULTI_LEVEL_SCORE", weight=1.0, note="bench ar25 WIN")
            causal.observe_state("MULTI_LEVEL_SCORE")
        if score >= 99:
            causal.observe_state("LOCAL_SCRIPT_OK")
        if levels == 0 and "WIN" not in state:
            causal.intervene("PHASE_B_BOUND", "zero_game", "SCORE_015", weight=0.3, note=f"bench zero {gid}")

    agg = float(report.get("aggregate_score") or 0)
    if agg >= 7.5:
        causal.intervene("MULTI_LEVEL_SCORE", "aggregate_25", "SCORE_ALIGNED", weight=0.9, note=f"local agg={agg}")
        causal.observe_state("SCORE_ALIGNED")
    elif agg >= 3.5:
        causal.intervene("MULTI_LEVEL_SCORE", "partial_aggregate", "SCORE_015", weight=0.5, note=f"local agg={agg}")

    causal.intervene("LOCAL_SCRIPT_OK", "full_bench_done", "KERNEL_PUSHED", weight=0.8, note="packaged submit agent bench")
    return causal


def _llm_optimize(report: dict[str, Any], causal: UpdateCausalModel) -> str:
    load_dotenv(ROOT / ".env")
    key, base, model = resolve_credentials()
    top = sorted(report.get("games") or [], key=lambda g: -float(g.get("game_score") or 0))[:8]
    zeros = [g for g in (report.get("games") or []) if int(g.get("levels_completed") or 0) == 0][:10]
    recent_actions = list(_action_buf)[-40:]
    prompt = f"""你是 ARC-AGI-3 灵境机器人的在线推演顾问（R5 Layer 3b）。
刚完成提交版全量本地测试，请结合因果模型给出优化推演。

【测试摘要】
- agent: {report.get('agent')}
- aggregate: {report.get('aggregate_score')}
- official: {report.get('aggregate_score_official')}
- games: {report.get('total_games')} won={report.get('games_won')} levels={report.get('total_levels_completed')}
- elapsed_sec: {report.get('elapsed_sec')}

【高分局】
{json.dumps(top, ensure_ascii=False, indent=2)}

【零分/未过关优先】
{json.dumps(zeros, ensure_ascii=False, indent=2)}

【因果模型】
{causal.render_for_prompt()}

【最近动作轨迹样本】
{json.dumps(recent_actions, ensure_ascii=False, indent=2)}

请输出 Markdown：
## 1. 测试解读（本地证据 vs 榜上 0.15 因果）
## 2. 机器人当前能力边界（哪些游戏已硬编码通关）
## 3. 下一步优化动作（P0/P1，可执行）
## 4. 若不做会怎样 / 若做了会怎样
## 5. 对因果图的更新建议（新节点/边）
## 6. 一句话作战指令
"""
    if not key:
        return "（无 LLM Key，跳过推演）\n" + causal.render_for_prompt()
    _emit("llm", f"调用 {model} @ {base} 推演中…")
    return chat_completion(
        prompt,
        system="你是 ARC-AGI-3 竞赛机器人的战略推演顾问。用简体中文，完整输出全部标题。",
        max_tokens=4096,
        timeout=180,
    )


def _build_notebook() -> dict[str, Any]:
    import scripts.build_notebook as bn

    bn.main()
    nb = ROOT / "notebooks" / "submission.ipynb"
    src = nb.read_text(encoding="utf-8")
    tag = "submit-ls20x7+ar25x8"
    return {
        "path": str(nb.relative_to(ROOT)),
        "bytes": nb.stat().st_size,
        "has_tag": tag in src,
        "tag": tag,
    }


def run_pipeline(max_steps: int = 400) -> None:
    global _running
    _action_buf.clear()
    with _lock:
        _state.update(
            {
                "phase": "start",
                "message": "启动全流程",
                "started_at": _now(),
                "finished_at": None,
                "error": None,
                "build": None,
                "bench": None,
                "causal": None,
                "llm": None,
                "actions": [],
                "events": [],
            }
        )
    try:
        _emit("build", "打包 submission.ipynb（嵌入 my_agent HARDCODED）…")
        build_info = _build_notebook()
        with _lock:
            _state["build"] = build_info
        if not build_info.get("has_tag"):
            raise RuntimeError("notebook 未包含 submit-ls20x7+ar25x8，打包失败")
        _emit("build", f"打包完成 · {build_info['path']} · tag={build_info['tag']}")

        _emit("bench", f"开始全量摸底 max_steps={max_steps}…")
        report = _instrument_run_all(max_steps=max_steps)
        BENCH_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        with _lock:
            _state["bench"] = report
        _emit(
            "bench",
            f"摸底完成 aggregate={report.get('aggregate_score')} "
            f"levels={report.get('total_levels_completed')} "
            f"won={report.get('games_won')}",
        )

        _emit("causal", "测试结果反哺 UpdateCausalModel…")
        causal = _ingest_bench_to_causal(report)
        snap = causal.snapshot()
        CAUSAL_PATH.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
        with _lock:
            _state["causal"] = {
                "version": snap["version"],
                "observed_states": snap["observed_states"],
                "path_breaks": snap["path_breaks"],
                "hypotheses": snap["hypotheses"],
                "effect_ema": snap["effect_ema"],
                "predict_graph": snap["predict_graph"],
                "render": causal.render_for_prompt(),
            }
        _emit("causal", f"因果图 v{snap['version']} · 观测 {snap['observed_states']}")

        _emit("llm", "LLM 推演优化中…")
        advice = _llm_optimize(report, causal)
        doc = _llm_document(advice, report)
        LLM_PATH.write_text(doc, encoding="utf-8")
        with _lock:
            _state["llm"] = {
                "model": resolve_credentials()[2],
                "markdown": advice,
                "document": doc,
                "path": str(LLM_PATH.relative_to(ROOT)),
                "filename": LLM_PATH.name,
            }
        _emit("done", "全流程完成：打包 → 全量测 → 因果 → LLM 推演")
        with _lock:
            _state["finished_at"] = _now()
            _state["phase"] = "done"
    except Exception as exc:  # noqa: BLE001
        logging.exception("pipeline failed")
        with _lock:
            _state["error"] = f"{type(exc).__name__}: {exc}"
            _state["phase"] = "error"
            _state["message"] = _state["error"]
            _state["finished_at"] = _now()
        LIVE_PATH.write_text(json.dumps(_state, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        _running = False
        LIVE_PATH.write_text(json.dumps(_state, ensure_ascii=False, indent=2), encoding="utf-8")


@app.route("/")
def index():
    return send_from_directory(STATIC, "robot_cockpit.html")


@app.route("/api/live")
def api_live():
    with _lock:
        return jsonify(_state)


@app.route("/api/start", methods=["POST"])
def api_start():
    global _running
    body = request.get_json(silent=True) or {}
    max_steps = int(body.get("max_steps", 400))
    with _lock:
        if _running:
            return jsonify({"ok": False, "error": "已在运行"}), 409
        _running = True
    threading.Thread(target=run_pipeline, kwargs={"max_steps": max_steps}, daemon=True).start()
    return jsonify({"ok": True, "max_steps": max_steps})


@app.route("/api/llm/export.md")
def api_llm_export_md():
    """点击「导出 MD」：优先落盘文件，否则用内存中的推演正文。"""
    if LLM_PATH.exists() and LLM_PATH.stat().st_size > 0:
        resp = send_file(
            LLM_PATH,
            mimetype="text/markdown; charset=utf-8",
            as_attachment=True,
            download_name=LLM_PATH.name,
        )
    else:
        with _lock:
            llm = _state.get("llm") or {}
            text = (llm.get("document") or llm.get("markdown") or "").strip()
            fname = llm.get("filename") or LLM_PATH.name
        if not text:
            return jsonify({"ok": False, "error": "尚无 LLM 推演结果可导出"}), 404
        resp = Response(text + ("\n" if not text.endswith("\n") else ""), mimetype="text/markdown; charset=utf-8")
        resp.headers["Content-Disposition"] = (
            f"attachment; filename=\"{fname}\"; filename*=UTF-8''{quote(fname)}"
        )
        return resp
    # send_file 对非 ASCII 文件名在部分浏览器不稳，补 UTF-8 头
    fname = LLM_PATH.name
    resp.headers["Content-Disposition"] = (
        f"attachment; filename=\"R5_robot_llm.md\"; filename*=UTF-8''{quote(fname)}"
    )
    return resp


@app.route("/r5_causal.html")
def r5_causal():
    return send_from_directory(STATIC, "r5_causal.html")


def main() -> None:
    load_dotenv(ROOT / ".env")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8799)
    p.add_argument("--open", action="store_true")
    p.add_argument("--autorun", action="store_true", help="启动后自动跑全流程")
    p.add_argument("--max-steps", type=int, default=400)
    args = p.parse_args()

    STATIC.mkdir(parents=True, exist_ok=True)
    _hydrate_llm_from_disk()
    LIVE_PATH.write_text(json.dumps(_state, ensure_ascii=False, indent=2), encoding="utf-8")

    url = f"http://127.0.0.1:{args.port}/"
    print(f"[robot] cockpit → {url}")
    if args.open:
        webbrowser.open(url)
    if args.autorun:
        global _running
        with _lock:
            _running = True
        threading.Thread(
            target=run_pipeline, kwargs={"max_steps": args.max_steps}, daemon=True
        ).start()

    app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
