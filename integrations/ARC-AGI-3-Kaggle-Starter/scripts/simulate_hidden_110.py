"""用本地公开局实证 + Monte Carlo + LLM，推演隐藏 110 局（55 public / 55 private）分数区间。

重要限制：
  正式评测 110 局本地不可得；本脚本是「能力外推模拟」，不是真跑隐藏集。

用法:
  .\\.venv\\Scripts\\python.exe scripts\\simulate_hidden_110.py
  .\\.venv\\Scripts\\python.exe scripts\\simulate_hidden_110.py --no-llm
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BENCH_DEFAULT = ROOT / "ui" / "static" / "games_benchmark_p1_coverage400.json"
OUT_JSON = ROOT / "ui" / "static" / "hidden110_simulation.json"
OUT_MD = ROOT / "docs" / "隐藏110局分数推演与改进方案.md"

N_HIDDEN = 110
N_PUBLIC = 55
N_PRIVATE = 55
N_MC = 4000


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tag_key(tags: list[str] | None) -> str:
    if not tags:
        return "unknown"
    t = sorted(tags)
    if "keyboard_click" in t or ("keyboard" in t and "click" in t):
        return "keyboard_click"
    if "click" in t and "keyboard" not in t:
        return "click"
    if "keyboard" in t:
        return "keyboard"
    return "+".join(t) or "unknown"


def load_bench(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def empirical_profile(bench: dict[str, Any]) -> dict[str, Any]:
    games = bench.get("games") or []
    by_tag: dict[str, list[dict[str, Any]]] = defaultdict(list)
    plugin_ids = {"ls20", "ar25"}
    non_plugin = []
    for g in games:
        gid = g.get("game_id")
        rec = {
            "game_id": gid,
            "tags": g.get("tags") or [],
            "tag": _tag_key(g.get("tags")),
            "levels_completed": int(g.get("levels_completed") or 0),
            "total_levels": int(g.get("total_levels") or 0),
            "game_score": float(g.get("game_score") or 0.0),
            "actions": int(g.get("actions") or 0),
            "state": g.get("state"),
            "is_plugin": gid in plugin_ids,
        }
        by_tag[rec["tag"]].append(rec)
        if not rec["is_plugin"]:
            non_plugin.append(rec)

    tag_dist = Counter(_tag_key(g.get("tags")) for g in games)
    n = max(len(games), 1)
    tag_probs = {k: v / n for k, v in tag_dist.items()}

    nonzero = [g for g in non_plugin if g["game_score"] > 0]
    # 经验：无插件局当前全部 0；用 Laplace 平滑给极小非零概率，避免 MC 全死
    p_any_level = (len(nonzero) + 0.5) / (len(non_plugin) + 1.0)
    # 若偶然过 1 关，参考部分分：用「1/total_levels * 粗略效率」上界估计 ~8–20
    partial_score_mean = 12.0
    partial_score_std = 6.0

    return {
        "n_public_practice": len(games),
        "aggregate_practice": bench.get("aggregate_score"),
        "plugin_wins": [g["game_id"] for g in games if g.get("game_id") in plugin_ids and (g.get("game_score") or 0) >= 99],
        "non_plugin_n": len(non_plugin),
        "non_plugin_nonzero": len(nonzero),
        "p_any_level_smoothed": p_any_level,
        "partial_score_mean": partial_score_mean,
        "partial_score_std": partial_score_std,
        "tag_probs": tag_probs,
        "by_tag_summary": {
            tag: {
                "n": len(rows),
                "mean_score": statistics.mean(r["game_score"] for r in rows) if rows else 0.0,
                "mean_levels": statistics.mean(r["levels_completed"] for r in rows) if rows else 0.0,
                "nonzero": sum(1 for r in rows if r["game_score"] > 0),
            }
            for tag, rows in by_tag.items()
        },
        "games": [
            {
                "game_id": g.get("game_id"),
                "tag": _tag_key(g.get("tags")),
                "levels_completed": g.get("levels_completed"),
                "total_levels": g.get("total_levels"),
                "game_score": g.get("game_score"),
                "state": g.get("state"),
            }
            for g in games
        ],
    }


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def sample_game_score(
    rng: random.Random,
    *,
    tag: str,
    scenario: str,
    profile: dict[str, Any],
) -> dict[str, Any]:
    """按情景抽样一局隐藏游戏得分（插件对未见 ID 默认不命中）。"""
    # 插件精确 ID 命中：隐藏集上视为 ~0（未见过同 ID）
    plugin_hit = 0.0
    p_level = float(profile["p_any_level_smoothed"])
    mu = float(profile["partial_score_mean"])
    sd = float(profile["partial_score_std"])

    if scenario == "pessimistic":
        # 与本地非插件一致：几乎不过关；偶发极低分
        p_level *= 0.3
        mu, sd = 6.0, 3.0
        # 对齐当前公开榜指纹 ~0.15：整体更冷
    elif scenario == "base":
        # 保留平滑后的极低过关率；略高于纯 0，贴近「已有 0.15」量级可解释
        p_level = max(p_level, 0.02)
        mu, sd = 10.0, 5.0
    elif scenario == "optimistic_transfer":
        # 假设 coverage/早停/族内迁移：5～12% 局能抠 ≥1 关
        p_level = 0.08
        if tag == "keyboard_click":
            p_level = 0.10
        elif tag == "click":
            p_level = 0.07
        mu, sd = 18.0, 10.0
    elif scenario == "after_p1_patch":
        # 早停+tu93 类近端插件思路外推：15% 非零，均值更高
        p_level = 0.15
        if tag == "keyboard_click":
            p_level = 0.20
        mu, sd = 28.0, 14.0
    else:
        raise ValueError(scenario)

    # 极低概率「整局高分」（误伤同类结构可脚本化）——乐观情景才开
    p_near_full = 0.0
    if scenario == "optimistic_transfer":
        p_near_full = 0.005
    elif scenario == "after_p1_patch":
        p_near_full = 0.015

    u = rng.random()
    if u < p_near_full:
        score = _clip(rng.uniform(70.0, 100.0), 0, 100)
        levels = max(1, int(score / 15))
        kind = "near_full_transfer"
    elif u < p_near_full + p_level:
        score = _clip(rng.gauss(mu, sd), 1.0, 55.0)
        levels = 1 if score < 25 else 2
        kind = "partial"
    else:
        score = 0.0
        levels = 0
        kind = "zero"

    return {
        "tag": tag,
        "game_score": round(score, 3),
        "levels_completed": levels,
        "kind": kind,
        "plugin_hit": plugin_hit,
    }


def run_mc(profile: dict[str, Any], scenario: str, n_trials: int = N_MC, seed: int = 42) -> dict[str, Any]:
    rng = random.Random(seed + hash(scenario) % 10000)
    tags = list(profile["tag_probs"].keys())
    weights = [profile["tag_probs"][t] for t in tags]
    # 若缺 unknown，补齐
    if not tags:
        tags, weights = ["unknown"], [1.0]

    public_means: list[float] = []
    private_means: list[float] = []
    full_means: list[float] = []
    nonzero_rates: list[float] = []

    for _ in range(n_trials):
        scores = []
        for i in range(N_HIDDEN):
            tag = rng.choices(tags, weights=weights, k=1)[0]
            scores.append(sample_game_score(rng, tag=tag, scenario=scenario, profile=profile)["game_score"])
        pub = statistics.mean(scores[:N_PUBLIC])
        priv = statistics.mean(scores[N_PUBLIC:])
        full = statistics.mean(scores)
        public_means.append(pub)
        private_means.append(priv)
        full_means.append(full)
        nonzero_rates.append(sum(1 for s in scores if s > 0) / N_HIDDEN)

    def pct(xs: list[float], p: float) -> float:
        ys = sorted(xs)
        i = int(_clip(p, 0, 1) * (len(ys) - 1))
        return ys[i]

    return {
        "scenario": scenario,
        "n_trials": n_trials,
        "public": {
            "mean": statistics.mean(public_means),
            "p10": pct(public_means, 0.10),
            "p50": pct(public_means, 0.50),
            "p90": pct(public_means, 0.90),
        },
        "private": {
            "mean": statistics.mean(private_means),
            "p10": pct(private_means, 0.10),
            "p50": pct(private_means, 0.50),
            "p90": pct(private_means, 0.90),
        },
        "full110": {
            "mean": statistics.mean(full_means),
            "p10": pct(full_means, 0.10),
            "p50": pct(full_means, 0.50),
            "p90": pct(full_means, 0.90),
        },
        "nonzero_rate": {
            "mean": statistics.mean(nonzero_rates),
            "p50": pct(nonzero_rates, 0.50),
        },
    }


def llm_advise(profile: dict[str, Any], mc: dict[str, Any], current_lb: float) -> str:
    from lingjing_solo.llm_client import chat_completion, load_dotenv, resolve_credentials

    load_dotenv(ROOT / ".env")
    key, base, model = resolve_credentials()
    if not key:
        return "（无 LLM Key，跳过顾问段）"

    prompt = f"""你是 ARC Prize 2026 / ARC-AGI-3 的竞赛计量与策略顾问。

【硬约束——必须遵守】
1. 正式评测是从未见过的约 110 局；一半≈55 局算 Public LB，一半 Private。本地只有约 25 个练习局。
2. 本地 ls20/ar25 满分 → 练习集 aggregate=8.0，**不能**直接当成隐藏 110 局得分。
3. 插件按 game_id 硬编码，在未见同 ID 的隐藏集上默认**不命中**。
4. 当前公开榜分数约 {current_lb}（旧提交），榜首约 7.51。分数尺度 0–100 的局均分。
5. 下面 Monte Carlo 是按标签分布外推的情景模拟，不是真跑了 110 局。

【本地实证】
{json.dumps(profile, ensure_ascii=False, indent=2)[:6000]}

【Monte Carlo 情景摘要】
{json.dumps(mc, ensure_ascii=False, indent=2)[:5000]}

请用简体中文输出完整 Markdown，必须含以下标题（不要省略）：
## 1. 结论摘要（给队长的 5 条子弹）
## 2. 隐藏 110 局分数区间（Public / Private / 全量）
- 对每个情景给出你「校准后」的区间（可修正 MC，但要说明修正理由）
- 明确：交上当前 v11 插件包后，Public 最可能落在哪
## 3. 与榜首 7.51 的差距拆解
## 4. 为何本地 8.0 不能兑换榜上 8
## 5. 改进方案（按期望 ΔPublic 排序的 P0/P1/P2）
- 每条：动作、负责人席位建议、验收指标、期望提分
## 6. 两周实验设计（如何用练习集逼近隐藏集泛化）
## 7. 风险与反模式
## 8. 一句话作战指令
"""
    print(f"[sim110] LLM {model} @ {base} …")
    return chat_completion(
        prompt,
        model=model,
        system="你是严谨的竞赛计量顾问。区分练习集与隐藏评测集；给出可执行分工，拒绝虚假精确。",
        max_tokens=4096,
        timeout=180,
    )


def render_md(
    profile: dict[str, Any],
    mc_all: dict[str, Any],
    llm_text: str,
    current_lb: float,
) -> str:
    lines = [
        "# 隐藏 110 局分数推演与改进方案",
        "",
        f"> 生成时间：`{_now()}`  ",
        f"> 性质：**能力外推模拟**（非正式隐藏集实测；本地仅 {profile['n_public_practice']} 练习局）  ",
        f"> 当前公开榜参考：`{current_lb}`；榜首参考：`7.51`",
        "",
        "## 0. 方法说明",
        "",
        "1. **实证层**：读取本地全量摸底（max_steps=400）得到标签分布与非插件过关率。",
        "2. **Monte Carlo**：按练习集标签比例抽样 110 局×4000 次，拆 55 Public / 55 Private。",
        "3. **情景**：pessimistic / base / optimistic_transfer / after_p1_patch。",
        "4. **LLM**：在硬约束下校准区间并给改进分工。",
        "",
        "### 本地实证要点",
        "",
        f"- 练习集 aggregate = **{profile['aggregate_practice']}**（插件局 {profile['plugin_wins']}）",
        f"- 非插件局：{profile['non_plugin_n']} 局，非零 **{profile['non_plugin_nonzero']}**",
        f"- 平滑后 P(≥1 关) ≈ **{profile['p_any_level_smoothed']:.4f}**",
        f"- 标签分布：`{json.dumps(profile['tag_probs'], ensure_ascii=False)}`",
        "",
        "### Monte Carlo 原始结果（未校准）",
        "",
        "| 情景 | Public mean | Public P50 [P10–P90] | Private P50 | 全110 P50 | 非零局率 mean |",
        "|------|-------------|----------------------|-------------|-----------|---------------|",
    ]
    for name, block in mc_all.items():
        pub, priv, full, nz = block["public"], block["private"], block["full110"], block["nonzero_rate"]
        lines.append(
            f"| `{name}` | {pub['mean']:.2f} | "
            f"**{pub['p50']:.2f}** [{pub['p10']:.2f}–{pub['p90']:.2f}] | "
            f"{priv['p50']:.2f} | {full['p50']:.2f} | {nz['mean']*100:.1f}% |"
        )
    lines += [
        "",
        "---",
        "",
        llm_text.strip() if llm_text.strip() else "_（LLM 未运行）_",
        "",
        "---",
        "",
        "## 附录 · 练习集逐局快照",
        "",
        "| game | tag | levels | score | state |",
        "|------|-----|--------|-------|-------|",
    ]
    for g in sorted(profile["games"], key=lambda x: (-(x["game_score"] or 0), x["game_id"] or "")):
        lines.append(
            f"| {g['game_id']} | {g['tag']} | {g['levels_completed']}/{g['total_levels']} | "
            f"{g['game_score']} | {g['state']} |"
        )
    lines += [
        "",
        f"原始 JSON：`{OUT_JSON.relative_to(ROOT).as_posix()}`",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", type=Path, default=BENCH_DEFAULT)
    ap.add_argument("--current-lb", type=float, default=0.15, help="当前公开榜分数")
    ap.add_argument("--trials", type=int, default=N_MC)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    bench = load_bench(args.bench)
    profile = empirical_profile(bench)
    scenarios = ["pessimistic", "base", "optimistic_transfer", "after_p1_patch"]
    mc_all = {
        s: run_mc(profile, s, n_trials=args.trials, seed=args.seed) for s in scenarios
    }

    llm_text = ""
    if not args.no_llm:
        try:
            llm_text = llm_advise(profile, mc_all, args.current_lb)
        except Exception as exc:  # noqa: BLE001
            llm_text = (
                f"## LLM 调用失败\n\n`{type(exc).__name__}: {exc}`\n\n"
                "请检查 MINIMAX_API_KEY / 代理后重跑；下方仍保留 Monte Carlo 结果。\n"
            )

    payload = {
        "generated_at": _now(),
        "disclaimer": "Proxy simulation only; competition 110 games are not available locally.",
        "current_public_lb": args.current_lb,
        "leaderboard_first_ref": 7.51,
        "profile": profile,
        "monte_carlo": mc_all,
        "llm_markdown": llm_text,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(render_md(profile, mc_all, llm_text, args.current_lb), encoding="utf-8")
    print(f"[sim110] wrote {OUT_JSON}")
    print(f"[sim110] wrote {OUT_MD}")
    for s, b in mc_all.items():
        print(
            f"  {s:22s} public_p50={b['public']['p50']:.2f} "
            f"private_p50={b['private']['p50']:.2f} nz={b['nonzero_rate']['mean']*100:.1f}%"
        )


if __name__ == "__main__":
    main()
