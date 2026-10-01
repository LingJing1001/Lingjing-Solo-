"""25 局跑分：affordance 排序器在贪心探索里的实跑成绩（对齐比赛口径）。

四个策略，同一副骨架（候选生成、stuck 处理、预算），只有打分不同——
差值就是学习组件的净贡献：
  random        候选里均匀乱试（探索地板）
  color_prior   按训练集颜色命中率查表（"颜色查表"基线）
  ranker        部署权重零样本打分（LOGO k=0 形态）
  ranker_adapt  同上 + 每攒 24 发本局已观测结局做一次 k-shot 校准（LOGO k>0 形态）

口径：每局 300 步预算，候选 = 组件中心点击（top6）+ 键盘动作；
click-first（每 5 步插一手未试过的键盘动作做覆盖）；同屏同动作不重复试；
无可试候选或 GAME_OVER → 整局重开（计一步）；指标 = 途中最高 levels_completed。

输出：state/afford_play_bench.json + stdout 对比表。
"""
from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import logging  # noqa: E402

from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import ActionInput, GameAction  # noqa: E402

from lingjing_solo.transfer.affordance import (  # noqa: E402
    AffordanceRanker, click_feature_dict, color_prior_train, record_label,
)
from lingjing_solo.transfer.miner import ENV_DIR, gid_to_full_id  # noqa: E402
from lingjing_solo.transfer.trajectory import (  # noqa: E402
    StepRecord, classify_outcome, click_context, connected_components, grid_digest,
    grid_of, read_jsonl, screen_delta,
)

DATASET = ROOT / "state" / "affordance_dataset.jsonl"
OUT_JSON = ROOT / "state" / "afford_play_bench.json"
BUDGET = 300
ADAPT_EVERY = 24
POLICIES = ("random", "color_prior", "ranker", "ranker_adapt")


def load_model_and_prior():
    records = read_jsonl(DATASET)
    model = AffordanceRanker(epochs=15).fit(records)
    prior = color_prior_train(records)
    return model, prior


class Player:
    """单局单策略的贪心探索器。所有策略共享骨架，只换打分。"""

    def __init__(self, arc: Any, full_id: str, policy: str,
                 model: Optional[AffordanceRanker] = None,
                 prior: Optional[Dict[int, float]] = None,
                 budget: int = BUDGET, seed: int = 0) -> None:
        self.env = arc.make(full_id, seed=seed, save_recording=False)
        self.game = getattr(self.env, "_game", None)
        self.full_id = full_id
        self.gid = full_id.split("-")[0]
        self.policy = policy
        self.model = model
        self.prior = prior or {}
        self.budget = budget
        self.rng = random.Random(f"{policy}:{full_id}")
        self.tried: set = set()
        self.shots: list = []  # 本局已观测的点击结局（校准用）

    # -- 候选与打分 --------------------------------------------------------

    def _candidates(self, grid: np.ndarray) -> List[Tuple[int, Optional[int], Optional[int]]]:
        comps = connected_components(grid)
        cands = [(6, int(c["cx"]), int(c["cy"]))
                 for c in sorted(comps, key=lambda c: -int(c["area"]))[:6]]
        return cands

    def _kb_candidates(self, frame: Any) -> List[Tuple[int, None, None]]:
        return [(int(a), None, None) for a in (frame.available_actions or [])
                if int(a) not in (6, 7)]

    def _score_click(self, grid: np.ndarray, x: int, y: int) -> float:
        if self.policy == "color_prior":
            return self.prior.get(int(grid[y, x]), 0.5)
        if self.policy in ("ranker", "ranker_adapt"):
            return self.model.score(click_context(grid, x, y))
        return 0.5  # random 不看分

    def _pick(self, grid: np.ndarray, frame: Any) -> Tuple[int, Optional[int], Optional[int]]:
        click_cands = self._candidates(grid)
        kb = self._kb_candidates(frame)
        untried_clicks = [c for c in click_cands
                          if ("c", grid_digest(grid), c) not in self.tried]
        untried_kb = [c for c in kb
                      if ("k", grid_digest(grid), c) not in self.tried]
        step_kind = "click"
        if self.policy == "random":
            pool = [(c, "c") for c in untried_clicks] + [(c, "k") for c in untried_kb]
            if not pool:
                return (-1, None, None)
            return self.rng.choice(pool)[0]
        # click-first：键盘只做覆盖（每 5 步或点击枯竭时插一手）；键盘枯竭则留在点击
        if untried_kb and ((not untried_clicks) or self.rng.random() < 0.2):
            step_kind = "kb"
        if step_kind == "kb":
            return self.rng.choice(untried_kb)
        if not untried_clicks:
            return (-1, None, None)
        if self.policy in ("ranker", "ranker_adapt"):
            scored = sorted(untried_clicks,
                            key=lambda c: -self._score_click(grid, c[1], c[2]))
            return scored[0]
        return self.rng.choice(untried_clicks)

    # -- 主循环 ------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        frame = self.env.reset()
        if frame is None:
            return {"gid": self.gid, "policy": self.policy, "levels": 0,
                    "steps": 0, "resets": 0, "clicks": 0, "adapt_events": 0,
                    "error": "reset failed"}
        levels_max = int(frame.levels_completed)
        steps = resets = clicks = adapt_events = 0
        hidden = self._hidden()
        while steps < self.budget:
            steps += 1
            if frame.state.name in ("WIN", "GAME_OVER"):
                frame = self.env.reset()
                hidden = self._hidden()
                resets += 1
                self.tried.clear()
                continue
            grid = grid_of(frame.frame)
            spec = self._pick(grid, frame)
            if spec[0] == -1:  # 本屏全试过 → 整局重开（计一步）
                frame = self.env.reset()
                hidden = self._hidden()
                resets += 1
                self.tried.clear()
                continue
            aid, x, y = spec
            is_click = aid == 6
            levels_before = int(frame.levels_completed)
            try:
                if is_click:
                    frame = self.env.step(GameAction.ACTION6, data={"x": x, "y": y})
                else:
                    frame = self.env.step(getattr(GameAction, f"ACTION{aid}"))
            except Exception:
                frame = None
            if frame is None:
                self.tried.add(("c" if is_click else "k", grid_digest(grid), spec))
                continue
            self.tried.add(("c" if is_click else "k", grid_digest(grid), spec))
            hidden_after = self._hidden()
            outcome = classify_outcome(
                levels_before=levels_before,
                levels_after=int(frame.levels_completed),
                state_after=frame.state.name,
                delta_count=len(screen_delta(grid, grid_of(frame.frame))),
                hidden_changed=bool(hidden) and hidden != hidden_after,
                sig_changed=False,
            )
            hidden = hidden_after
            levels_max = max(levels_max, int(frame.levels_completed))
            clicks += is_click
            if is_click and self.policy == "ranker_adapt":
                self.shots.append(_Shot(grid, x, y, outcome))
                if clicks % ADAPT_EVERY == 0 and self._adapt():
                    adapt_events += 1
        return {"gid": self.gid, "policy": self.policy, "levels": levels_max,
                "steps": steps, "resets": resets, "clicks": clicks,
                "adapt_events": adapt_events}

    def _hidden(self) -> str:
        if self.game is None:
            return ""
        try:
            hs = self.game._get_hidden_state()
            return grid_digest(np.asarray(hs).astype(np.int64) % 251) if hs is not None else ""
        except Exception:
            return ""

    def _adapt(self) -> bool:
        labeled = [s for s in self.shots[-96:]]
        labels = {s.outcome != "no_op" for s in labeled}
        if len(labels) < 2:
            return False
        recs = [s.to_record(self.gid) for s in labeled]
        adapted = self.model.adapt(recs, epochs=3)
        self.model = adapted
        return True


class _Shot:
    """live 观测到的一次点击结局，够 adapt() 用即可。"""

    def __init__(self, grid: np.ndarray, x: int, y: int, outcome: str) -> None:
        self.ctx = click_context(grid, x, y)
        self.outcome = outcome

    def to_record(self, gid: str) -> StepRecord:
        return StepRecord(game_id=gid, gid=gid, level_idx=0, step_idx=0,
                          action={"id": 6, "x": 1, "y": 1}, source="live",
                          reliability=1.0, outcome=self.outcome,
                          grid_hash_before="live", context=self.ctx)


def main() -> int:
    t0 = time.time()
    model, prior = load_model_and_prior()
    logging.getLogger("arc_agi").setLevel(logging.CRITICAL)
    logging.getLogger("arc_agi.scorecard").setLevel(logging.CRITICAL)
    arc = Arcade(environments_dir=str(ENV_DIR), logger=logging.getLogger("bench"),
                 operation_mode=OperationMode.OFFLINE)
    games = sorted(p.name for p in ENV_DIR.iterdir() if p.is_dir())
    results: List[Dict[str, Any]] = []
    for gid in games:
        full = gid_to_full_id(gid)
        if full is None:
            continue
        for policy in POLICIES:
            m = model if policy in ("ranker", "ranker_adapt") else None
            p = Player(arc, full, policy, model=m, prior=prior)
            r = p.run()
            results.append(r)
            print(f"  {r['gid']:6s} {r['policy']:12s} levels={r['levels']} "
                  f"steps={r['steps']} resets={r['resets']} "
                  f"adapt={r['adapt_events']}", flush=True)
    OUT_JSON.write_text(json.dumps(results, indent=1), encoding="utf-8")

    print(f"\n== 25 局跑分（budget={BUDGET}）==")
    print(f"{'game':8s}" + "".join(f"{p:>14s}" for p in POLICIES))
    by_game: Dict[str, Dict[str, int]] = {}
    for r in results:
        by_game.setdefault(r["gid"], {})[r["policy"]] = r["levels"]
    for gid in sorted(by_game):
        row = by_game[gid]
        print(f"{gid:8s}" + "".join(f"{row.get(p, 0):>14d}" for p in POLICIES))
    print("-" * 64)
    tot = {p: sum(r["levels"] for r in results if r["policy"] == p) for p in POLICIES}
    nz = {p: sum(1 for r in results if r["policy"] == p and r["levels"] > 0)
          for p in POLICIES}
    print(f"{'总关数':6s}" + "".join(f"{tot[p]:>14d}" for p in POLICIES))
    print(f"{'破零局':6s}" + "".join(f"{nz[p]:>13d}/25" for p in POLICIES))
    print(f"用时 {time.time() - t0:.0f}s → {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
