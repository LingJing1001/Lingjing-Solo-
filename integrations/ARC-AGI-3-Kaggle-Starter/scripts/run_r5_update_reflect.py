"""运行 R5 更新反思 Agent（接入 UpdateCausalModel；可选真实 LLM）。

用法:
  .\\.venv\\Scripts\\python.exe scripts\\run_r5_update_reflect.py
  .\\.venv\\Scripts\\python.exe scripts\\run_r5_update_reflect.py --llm --dump-causal
  .\\.venv\\Scripts\\python.exe scripts\\run_r5_update_reflect.py --llm --fallback-symbolic
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.r5_update_reflect_agent import (  # noqa: E402
    DEFAULT_UPDATE_EVENTS,
    R5UpdateReflectAgent,
)
from lingjing_solo.llm_client import load_dotenv, make_llm_fn, resolve_credentials  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=None, help="报告输出路径")
    p.add_argument("--llm", action="store_true", help="调用真实 LLM")
    p.add_argument(
        "--fallback-symbolic",
        action="store_true",
        help="LLM 失败时回退封闭符号复盘（默认：失败则 exit 1）",
    )
    p.add_argument("--dump-causal", action="store_true", help="写出因果 JSON 快照")
    args = p.parse_args()

    load_dotenv(ROOT / ".env")

    llm_fn = None
    llm_error = None
    if args.llm:
        key, base, model = resolve_credentials()
        print(f"[R5] LLM base={base} model={model} key={'SET' if key else 'MISSING'}")
        if not key:
            raise SystemExit("无法跑 --llm：请在 .env 配置 OPENAI_API_KEY 或 DEEPSEEK_API_KEY")
        raw_fn = make_llm_fn()

        def llm_fn(prompt: str) -> str:
            return raw_fn(prompt)

    agent = R5UpdateReflectAgent(llm_fn=llm_fn)
    agent.ingest_many(DEFAULT_UPDATE_EVENTS)
    print(
        f"[R5] causal v{agent.causal.version} "
        f"edges={len(agent.causal.predict_graph)} "
        f"observed={sorted(agent.causal.observed_states)}"
    )
    print(f"[R5] path_breaks={agent.causal.blocked_path_to_aligned()}")

    # 先落盘 prompt + causal，便于 API 失败后仍可人工/会话 LLM 续跑
    prompt_path = ROOT / "R5_llm_prompt.txt"
    prompt_path.write_text(agent.build_prompt(), encoding="utf-8")
    print(f"[R5] prompt -> {prompt_path}")

    try:
        report = agent.reflect()
    except Exception as exc:  # noqa: BLE001
        llm_error = str(exc)
        print(f"[R5] LLM FAILED: {llm_error}")
        if not (args.llm and args.fallback_symbolic):
            # 仍写出因果快照，方便续跑
            if args.dump_causal:
                snap_path = ROOT / "R5更新反思报告_causal.json"
                snap_path.write_text(
                    json.dumps(agent.causal.snapshot(), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(f"[R5] causal snapshot -> {snap_path}")
            raise SystemExit(
                "LLM 调用失败。可：1) 充值/换 Key 后重跑 --llm；"
                "2) 加 --fallback-symbolic；"
                "3) 用已生成的 R5_llm_prompt.txt 在会话里续答。"
            ) from exc
        print("[R5] falling back to symbolic advisor")
        agent.llm_fn = None
        report = agent.reflect()
        report.answer_md = (
            f"> API 失败回退符号模式：`{llm_error}`\n\n" + report.answer_md
        )

    out = Path(
        args.out
        or (ROOT / ("R5更新反思报告_LLM.md" if args.llm else "R5更新反思报告.md"))
    )
    out.write_text(report.as_markdown(), encoding="utf-8")

    if args.dump_causal:
        snap_path = out.with_name(out.stem + "_causal.json")
        snap_path.write_text(
            json.dumps(agent.causal.snapshot(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[R5] causal snapshot -> {snap_path}")

    print(f"[R5] role={agent.ROLE} layers={agent.LAYERS}")
    print(f"[R5] mode={report.mode} triggers={report.triggers}")
    print(f"[R5] wrote {out}")
    print()
    try:
        print(report.answer_md)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(report.answer_md.encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")


if __name__ == "__main__":
    main()
