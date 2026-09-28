"""Regenerate the wa30 baseline-vs-solver chart embedded in the SSA plan doc.

Numbers are the verified 2026-09-23 snapshot recorded in
docs/SSA白皮书关联核实与整合方案.md §6 — this script replots that snapshot,
it does not re-read bench/ or re-run anything. Output overwrites
docs/wa30_对照基线_20260923.png in place.

Usage (Starter venv):
    .venv/Scripts/python.exe ARC-AGI-3/scripts/wa30_baseline_chart.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PKG = Path(__file__).resolve().parents[1]
OUT = PKG / "docs" / "wa30_对照基线_20260923.png"


def main() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(8.4, 3.0), dpi=150)

    rows = ["定向求解器\n(scripts/wa30_carry_solver.py)", "Agent 链现状\n(SmartRouter play_local)"]
    vals = [26, 300]
    colors = ["#2e9e5b", "#c0392b"]
    bars = ax.barh(rows, vals, color=colors, height=0.52, zorder=3)

    # 游戏预算参考线（仅对求解器口径有意义：wa30 每关 200 步）
    ax.axvline(200, color="#555555", linestyle="--", linewidth=1.2, zorder=2)
    ax.text(200, 1.62, "wa30 每关步数预算 200", ha="center", va="bottom",
            fontsize=8.5, color="#444444")

    ax.bar_label(bars, labels=[
        "26 步 → 通关换关 · 26/200（三次复现序列一致）",
        "300 步耗尽 → 0 关 · score 0",
    ], padding=5, fontsize=9.5)

    ax.set_xlim(0, 340)
    ax.set_xlabel("动作数", fontsize=9.5)
    ax.set_title("wa30 L0 对照：agent 链 vs 机制定向求解（2026-09-23 实测）",
                 fontsize=11, pad=10)
    ax.grid(axis="x", color="#dddddd", linewidth=0.7, zorder=0)
    ax.tick_params(axis="y", labelsize=9.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)

    fig.text(0.01, -0.04,
             "注：agent 上限 300 = play_local --max-steps（MAX_ACTIONS）；求解器 26 步为游戏预算 200 内的\n"
             "通关解，逐动作与引擎断言一致、全新 reset 回放换关。数据：docs/SSA白皮书关联核实与整合方案.md §6。",
             fontsize=7.5, color="#666666", ha="left", va="top")

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
