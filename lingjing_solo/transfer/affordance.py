"""Affordance 排序器：点击上下文特征 → 「有没有效果 / 是否有效」概率，LOGO 评测。

标签（引擎真值，来自 trajectory.py 的效果分类）：
  has_effect    outcome != no_op                       —— 探索期"这个点会动吗"
  productive    outcome ∈ {state_only, advance, win}   —— "这个点碰到机制了吗"
                   （screen 是负类：移动/相机会让像素几乎必变，收割实测占 81%）

特征全部来自 click_context（无绝对坐标语义，可跨布局迁移）：颜色 one-hot、
3×3 邻域统计、所属连通域几何、盘面统计。模型 = lincore 的 LogisticSGD
（动量 + L2，流式友好），刻意保持小——热图 v1 的教训是目标失配与标签毒化，
不是模型容量。

评测：leave-one-game-out（按 gid 留出）。两个指标：
  pairwise   同一基态（grid_hash_before）内 正分 > 负分 的比例（排序质量）
  top1       基态内得分最高的候选是正例的比例（部署形态：给探索器提 top-1）
基线：color-prior（训练集里该颜色的有效率）与 global prior——模型必须
打得过"按颜色查表"才算学到了跨游戏结构。
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from ..lincore.gradient import LogisticSGD
from .trajectory import StepRecord

EFFECT_COLORS = tuple(range(16))
# productive 头的正类：触及机制（引擎隐藏状态/进度），而非只动像素。
# screen 在相机会/移动类游戏里几乎必然发生（收割实测占 81%），含它则标签失去判别力。
MECHANISM_OUTCOMES = frozenset({"state_only", "advance", "win"})


def click_feature_dict(ctx: Optional[dict]) -> Optional[Dict[str, float]]:
    """click_context → 定长命名特征。非点击上下文返回 None。"""
    if not ctx or ctx.get("kind") != "click":
        return None
    f: Dict[str, float] = {}
    color = int(ctx.get("click_color", 0))
    for c in EFFECT_COLORS:
        f[f"color{c}"] = 1.0 if color == c else 0.0
    f["on_bg"] = 1.0 if ctx.get("on_bg") else 0.0
    f["rel_x"] = float(ctx.get("rel_x", 0.5))
    f["rel_y"] = float(ctx.get("rel_y", 0.5))
    f["n_objects"] = min(int(ctx.get("n_objects", 0)), 64) / 64.0
    f["n_colors"] = int(ctx.get("n_colors", 0)) / 16.0

    hood = ctx.get("hood3x3") or []
    bg = int(ctx.get("bg", 0))
    if hood:
        f["hood_bg_frac"] = sum(1 for h in hood if h == bg) / len(hood)
        f["hood_same_frac"] = sum(1 for h in hood if h == color) / len(hood)
        f["hood_distinct"] = len({h for h in hood if h >= 0}) / 9.0
        f["hood_oob"] = sum(1 for h in hood if h < 0) / len(hood)
    else:
        f["hood_bg_frac"] = f["hood_same_frac"] = f["hood_distinct"] = f["hood_oob"] = 0.0

    obj = ctx.get("obj")
    if obj:
        f["has_obj"] = 1.0
        f["obj_area_log"] = math.log1p(int(obj["area"])) / 8.0
        f["obj_fill"] = int(obj["area"]) / max(1, int(obj["bw"]) * int(obj["bh"]))
        f["obj_bbox_log"] = math.log1p(int(obj["bw"]) * int(obj["bh"])) / 10.0
        f["obj_color_ncomps"] = min(int(obj["n_comps_of_color"]), 8) / 8.0
        f["obj_color_cells_log"] = math.log1p(int(obj["color_cells_total"])) / 10.0
    else:
        f["has_obj"] = 0.0
        f["obj_area_log"] = f["obj_fill"] = f["obj_bbox_log"] = 0.0
        f["obj_color_ncomps"] = f["obj_color_cells_log"] = 0.0
    return f


def record_label(rec: StepRecord, head: str) -> int:
    if head == "has_effect":
        return 1 if rec.outcome != "no_op" else 0
    if head == "productive":
        return 1 if rec.outcome in MECHANISM_OUTCOMES else 0
    raise ValueError(f"unknown head {head}")


class AffordanceRanker:
    """两个逻辑回归头（has_effect / productive），特征键在 fit 时定型。

    训练目标 = BCE + rank_weight × 组内 pairwise 排序损失（RankNet 式）。
    排序项把 (x_pos − x_neg, y=1) 喂回同一个 LogisticSGD——这在数学上恰好等价于
    −log σ(s_pos − s_neg) 的梯度（∂L/∂w = −σ(−d)(x_pos−x_neg)），所以不用第二套
    优化器；代价是共享动量状态，实测无碍（特征有界、lr 小）。

    组 = 同一基态 (game_id, level_idx, grid_hash_before) 的候选点击集合——部署时
    探索器正是"在当前屏幕上选一个点"，组内排序才是对齐的指标。

    诊断注记（2026-09-30 LOGO）：ft09/tu93 零样本 auc=0.000 不是排序崩坏，是
    组内正负对只有 8/1 个 + 逐局特征翻转（ft09 机制点击在同色组件多的 Busy 盘面、
    tu93 点什么都触发）。零样本跨不过按局翻转，k-shot 在线校准（adapt）是正解——
    live agent 本来就在持续观测自己动作的结局（CEAX ClickTarget 同款逻辑）。
    """

    HEADS = ("has_effect", "productive")

    def __init__(self, lr: float = 0.08, l2: float = 1e-4, epochs: int = 15,
                 seed: int = 0, rank_weight: float = 1.0,
                 pairs_per_group: int = 16) -> None:
        self.lr, self.l2, self.epochs, self.seed = lr, l2, epochs, seed
        self.rank_weight = rank_weight
        self.pairs_per_group = pairs_per_group
        self.keys: List[str] = []
        self.models: Dict[str, LogisticSGD] = {}

    def _vector(self, fd: Dict[str, float]) -> List[float]:
        return [fd.get(k, 0.0) for k in self.keys]

    def _bce_rank_pass(self, m: LogisticSGD, labeled: Sequence[Tuple[Dict[str, float], int]],
                       pairs: Sequence[Tuple[Dict[str, float], Dict[str, float]]],
                       rng) -> None:
        for fd, y in labeled:
            m.partial_fit(self._vector(fd), y)
        if self.rank_weight > 0:
            for fpos, fneg in pairs:
                for _ in range(int(self.rank_weight)):
                    # (x+ − x−, y=1) ≡ RankNet 梯度，见类 docstring
                    m.partial_fit([a - b for a, b in
                                   zip(self._vector(fpos), self._vector(fneg))], 1.0)

    def _group_pairs(self, records: Sequence[StepRecord], head: str, rng):
        """组内 (正特征, 负特征) 对，每组采样上限 pairs_per_group。"""
        by_group: Dict[Tuple, List[StepRecord]] = defaultdict(list)
        feats: Dict[int, Dict[str, float]] = {}
        for r in records:
            fd = click_feature_dict(r.context)
            if fd is None:
                continue
            feats[id(r)] = fd
            by_group[(r.game_id, r.level_idx, r.grid_hash_before)].append(r)
        pairs = []
        for recs in by_group.values():
            pos = [feats[id(r)] for r in recs if record_label(r, head) == 1]
            neg = [feats[id(r)] for r in recs if record_label(r, head) == 0]
            if not pos or not neg:
                continue
            all_pairs = [(p, n) for p in pos for n in neg]
            if len(all_pairs) > self.pairs_per_group:
                all_pairs = rng.sample(all_pairs, self.pairs_per_group)
            pairs.extend(all_pairs)
        return pairs

    def fit(self, records: Sequence[StepRecord]) -> "AffordanceRanker":
        feats: List[Dict[str, float]] = []
        for r in records:
            fd = click_feature_dict(r.context)
            if fd is not None:
                feats.append(fd)
        if not feats:
            raise ValueError("no click records to fit")
        self.keys = sorted({k for fd in feats for k in fd})
        import random as _random

        rng = _random.Random(self.seed)
        for head in self.HEADS:
            m = LogisticSGD(dim=len(self.keys), lr=self.lr, l2=self.l2)
            labeled = [(fd, record_label(r, head)) for r in records
                       if (fd := click_feature_dict(r.context)) is not None]
            pairs = self._group_pairs(records, head, rng)
            for _ in range(self.epochs):
                rng.shuffle(labeled)
                rng.shuffle(pairs)
                self._bce_rank_pass(m, labeled, pairs, rng)
            self.models[head] = m
        return self

    def adapt(self, shots: Sequence[StepRecord], epochs: int = 3) -> "AffordanceRanker":
        """k-shot 在线校准：用本局前 k 个已观测结局微调一份副本。

        这是部署形态——live agent 每走一步都拿到引擎真值（CEAX 的 ClickTarget
        就在做这件事），评测里不给它这个信息才是失真。shot 太少或全同标签时
        原样返回（没有可学的信号就别硬掰权重）。
        """
        if not self.keys or not shots:
            return self
        labels = {head: [record_label(r, head) for r in shots
                         if click_feature_dict(r.context) is not None]
                  for head in self.HEADS}
        out = AffordanceRanker(lr=self.lr, l2=self.l2, epochs=epochs, seed=self.seed,
                               rank_weight=self.rank_weight,
                               pairs_per_group=self.pairs_per_group)
        out.keys = list(self.keys)
        import random as _random

        rng = _random.Random(self.seed + 1)
        for head in self.HEADS:
            m = LogisticSGD(dim=len(self.keys), lr=self.lr, l2=self.l2)
            m.w = self.models[head].w.copy()
            labeled = [(fd, record_label(r, head)) for r in shots
                       if (fd := click_feature_dict(r.context)) is not None]
            pairs = self._group_pairs(shots, head, rng)
            if len({y for _, y in labeled}) < 2 and not pairs:
                continue  # 全同标签：零信号，保留全局权重
            for _ in range(epochs):
                rng.shuffle(labeled)
                rng.shuffle(pairs)
                self._bce_rank_pass(m, labeled, pairs, rng)
            out.models[head] = m
        missing = [h for h in self.HEADS if h not in out.models]
        for h in missing:
            out.models[h] = self.models[h]  # 共享未适配头
        return out

    def score(self, ctx: Optional[dict], head: str = "productive") -> float:
        fd = click_feature_dict(ctx)
        m = self.models.get(head)
        if fd is None or m is None:
            return 0.5
        return m.predict_proba(self._vector(fd))

    @property
    def ready(self) -> bool:
        return bool(self.models) and all(m.ready for m in self.models.values())


# --------------------------------------------------------------- 评测


def _groups(records: Sequence[StepRecord]) -> Dict[Tuple, List[StepRecord]]:
    g: Dict[Tuple, List[StepRecord]] = defaultdict(list)
    for r in records:
        if click_feature_dict(r.context) is None:
            continue
        g[(r.gid, r.level_idx, r.grid_hash_before)].append(r)
    return g


def _pairwise_auc(groups, score_fn) -> float:
    wins, total = _pairwise_wins(
        list(groups.values()) if hasattr(groups, "values") else groups, score_fn)
    return wins / total if total else 0.5


def _y(r: StepRecord) -> int:
    return record_label(r, "productive")


def _top1(groups, score_fn) -> float:
    """每个含正例的基态里，得分最高候选为正例的比例。"""
    hits = total = 0
    for recs in groups.values():
        if not any(_y(r) == 1 for r in recs):
            continue
        total += 1
        best = max(recs, key=score_fn)
        hits += 1 if _y(best) == 1 else 0
    return hits / total if total else 0.0


def color_prior_train(train: Sequence[StepRecord]) -> Dict[int, float]:
    """基线：训练集里各颜色点击的 productive 率（Laplace 平滑）。"""
    num: Dict[int, int] = defaultdict(int)
    den: Dict[int, int] = defaultdict(int)
    for r in train:
        ctx = r.context
        if not ctx or ctx.get("kind") != "click":
            continue
        c = int(ctx.get("click_color", -1))
        den[c] += 1
        num[c] += _y(r)
    return {c: (num[c] + 1.0) / (den[c] + 2.0) for c in den}


def _sample_shots(clicks: List[StepRecord], k: int, gid: str) -> List[StepRecord]:
    """带种子的随机 k-shot（按局确定性）。

    不取"数据集前 k 条"：那会拿走计划局开局路线步（全负例、掏空探针对比组），
    评出一堆 pair=0 的退化数字（2026-09-30 实测教训）。随机采样模拟 live
    探索者跨状态累积观测的现实——每个状态都可能贡献一发已判定的结局。
    """
    import random as _random

    rng = _random.Random(f"shots:{gid}")
    if len(clicks) <= k:
        return list(clicks)
    return rng.sample(clicks, k)


def logo_eval(records: Sequence[StepRecord], min_test_clicks: int = 20,
              epochs: int = 15, adapt_shots: int = 0) -> Dict[str, Dict[str, float]]:
    """逐游戏留出评测。返回 {gid: {metric}} + pooled（合并全部留出组，各用本折模型）。

    adapt_shots > 0 时模拟部署形态：把留出局按采集顺序的前 k 个点击结局交给
    adapt() 做在线校准，再用其余记录评测（shots 不进评测组）。ft09/tu93 这类
    逐局特征翻转的游戏，零样本有地板，k-shot 才是真实可部署的数字。
    颜色基线同样喂 shots（Laplace 更新），保证对照公平。
    """
    gids = sorted({r.gid for r in records})
    per: Dict[str, Dict[str, float]] = {}
    pooled: List[Tuple[List[StepRecord], object]] = []
    for gid in gids:
        test = [r for r in records if r.gid == gid]
        train = [r for r in records if r.gid != gid]
        clicks = [r for r in test if click_feature_dict(r.context) is not None]
        n_click = len(clicks)
        if n_click < min_test_clicks:
            per[gid] = {"n_clicks": float(n_click), "skipped": 1.0}
            continue
        shots = _sample_shots(clicks, adapt_shots, gid) if adapt_shots else []
        shot_ids = {id(r) for r in shots}
        eval_recs = [r for r in test if id(r) not in shot_ids]
        model = AffordanceRanker(epochs=epochs).fit(train)
        if shots:
            model = model.adapt(shots)
        prior = color_prior_train(train + shots)
        n_train_click = sum(1 for r in train if click_feature_dict(r.context) is not None)
        glob = sum(_y(r) for r in train) / max(1, n_train_click)

        def s_model(r: StepRecord, _m=model) -> float:
            return _m.score(r.context)

        def s_prior(r: StepRecord, _p=prior) -> float:
            return _p.get(int(r.context.get("click_color", -1)), 0.5)

        def s_glob(r: StepRecord, _g=glob) -> float:
            return _g

        groups = _groups(eval_recs)
        pooled.append((list(groups.values()), s_model))
        wins, n_pairs = _pairwise_wins(list(groups.values()), s_model)
        per[gid] = {
            "n_clicks": float(n_click),
            "n_shots": float(len(shots)),
            "n_pairs": float(n_pairs),
            "auc_model": _pairwise_auc(groups, s_model),
            "auc_color_prior": _pairwise_auc(groups, s_prior),
            "auc_global": _pairwise_auc(groups, s_glob),
            "top1_model": _top1(groups, s_model),
            "top1_color_prior": _top1(groups, s_prior),
            "top1_global": _top1(groups, s_glob),
        }
    flat_groups = [g for gs, _ in pooled for g in gs]
    pooled_metrics = {"n_groups": float(len(flat_groups))}
    if flat_groups:
        wins = total = top_hits = top_total = 0
        for gs, s_fn in pooled:
            w, t = _pairwise_wins(gs, s_fn)
            wins += w
            total += t
            top_hits += _top1_hits(gs, s_fn)
            top_total += sum(1 for g in gs if any(_y(r) == 1 for r in g))
        pooled_metrics["auc_model"] = wins / total if total else 0.5
        pooled_metrics["top1_model"] = top_hits / top_total if top_total else 0.0
        pooled_metrics["n_pairs"] = float(total)
    per["pooled"] = pooled_metrics
    return per


def _pairwise_wins(groups, score_fn) -> Tuple[float, float]:
    """(wins, total)：同基态内 (正, 负) 对，正分>负分记 1、平记 0.5。"""
    wins = total = 0.0
    for recs in groups:
        pos = [score_fn(r) for r in recs if _y(r) == 1]
        neg = [score_fn(r) for r in recs if _y(r) == 0]
        for p in pos:
            for n in neg:
                total += 1
                wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins, total


def _top1_hits(groups, score_fn) -> int:
    hits = 0
    for recs in groups:
        if recs and any(_y(r) == 1 for r in recs):
            hits += 1 if _y(max(recs, key=score_fn)) == 1 else 0
    return hits
