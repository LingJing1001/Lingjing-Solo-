"""自动推理引擎：把"我该怎么做"降解成"我现在还不知道什么，怎么用最少的动作问出来"。

它不执行任何一条为某个游戏写的规则。它只做三件事：

  1. **归纳**  每一次 (画面, 按键, 新画面) 都被拆成 谁动了 / 动了多少 / 还改了什么
               —— 分别进 KeySemantics 和 EffectBook。
  2. **归因**  过关那一刻的转移是唯一免费的正样本；从这里倒推出"目标"是什么陈述，
               下一个关卡把这条陈述重新定位（concept, not coordinates）。
  3. **提问**  决策 = 挑一个信息量最大的实验：
                 · 有键从没按过      → 按它（语义未知）
                 · 语义有冲突        → 换个局面再按（判别性实验）
                 · 目标已知          → 用学到的语义把位移分解成按键序列
                 · 目标未知          → 去碰还没碰过的结构（副作用未知）
               每一步都会先说一句理由，所以行为是可审计的，不是黑箱。

关键点：这里没有"目标 = 稀有 + 静止 + 内部"，没有"ACTION1 是上"。
所有具体内容都来自数据；代码里只有**假设空间**。
"""
from __future__ import annotations

import os

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Optional

from ..judge import Judge, cases
from . import goals as G
from . import world as W
from .goals import Goal
from .hypo import EffectBook, KeySemantics, Motion, fit
from .world import Cell, Grid

INF = 10 ** 9


@dataclass
class Trial:
    g0: Grid
    act: str
    g1: Grid
    moved: bool
    moved_ok: bool
    reward: bool = False
    dead: bool = False


class Reasoner:
    """一个只靠推理活着的 agent 的大脑。"""

    def __init__(self, max_notes: int = 600) -> None:
        self.keys = KeySemantics()
        self.effects = EffectBook()
        # 判断层：不打分、不选动作，只回答"我现在对某某有多确定"。
        # 它吃的是**计数**（票型、冲突数、推不动次数），不是关卡知识；
        # 它带走的跨关记忆只有一个校准参数，不是任何具体结论。
        # JUDGE_CAL 给一个路径就能让温度跨进程累积 —— 这是唯一"越用越准"的东西。
        self.judge = Judge(path=os.environ.get("JUDGE_CAL"))
        self.goal: Optional[Goal] = None
        self.body_sig: Optional[tuple] = None     # (normalise(body), colour_map)
        self.body: frozenset[Cell] = frozenset()
        # 身份 → 当前占的格子。ar25 有两个体（左房间一块、右房间一块），
        # 用一个 self.body 表示不了：只记一个，另一个就永远够不着目标。
        self.bodies: dict[tuple, frozenset[Cell]] = {}
        self.level: int = -1
        self.steps: int = 0
        self.trials: list[Trial] = []
        self.notes: list[str] = []
        self._max_notes = max_notes

        # 关卡内的短期状态
        self._blocked: set[tuple] = set()        # (身份, 键, 位置) —— 这个体在这儿推不动
        self._visited_targets: set[Cell] = set()
        self._last: Optional[tuple[Grid, str]] = None
        self._dead_acts: set[str] = set()   # 一按就死的键（可以不止一个）
        self._need_reset = False
        self._first_grid: Optional[Grid] = None
        # "这个键在别处按过也不动"：角落里的 noop 不能当"这个键没用"的证据
        self._pending: Optional[tuple[str, dict[tuple, Cell]]] = None
        self._noop_at: dict[tuple, Cell] = {}    # 身份 → 上次按这个键没动的位置
        self._noop_act: dict[tuple, str] = {}
        # 背景减除：哪些格子是"环境自己在动"（只从"没有整体平移"的转移里学）
        self._noop_freq: dict[Cell, int] = {}
        self._noop_n: int = 0
        self._ambient: frozenset[Cell] = frozenset()
        # 本关已经报过"中间帧有变化"的键，免得每拍重复刷同一句
        self._anim_seen: set[str] = set()
        # 跨关保留：按键语义与"这类游戏的目标句式"是概念，不随关卡作废
        self.transfer_hits = 0
        # 跨关保留：见过的"会动的身份"。目标不可能是会动的块 —— 这是迁移时
        # 排除候选的依据，也是 ar25 从 L0 到 L1 挑错目标的直接原因。
        self._movers: set[tuple] = set()
        # 目标陈述先搬到这里挂着，等认出体再落坐标
        self._seed: Optional[Goal] = None
        self._seed_tries = 0
        # 判别实验的"上次问它时票数是多少"：问过而票数没变，说明这个问题
        # 在那个局面下问不出新东西 —— 换问题，而不是按步数冷却。
        self._probe_seen: dict[str, int] = {}
        # 待结算的判断：emit 时存下"我预测按它会怎样"，observe 时对账。
        # 这是校准层唯一的真值来源 —— 不是人标���，是下一帧自己打脸。
        self._pending_judge: Optional[tuple] = None
        self._judge_miss = 0

    # ──────────────────────────────────────────────────────
    # 记忆（说人话）
    # ──────────────────────────────────────────────────────
    def say(self, msg: str) -> None:
        self.notes.append(msg)
        if len(self.notes) > self._max_notes:
            del self.notes[: len(self.notes) - self._max_notes]

    # ──────────────────────────────────────────────────────
    # 关卡切换
    # ──────────────────────────────────────────────────────
    def new_level(self, g: Grid) -> None:
        self.level += 1
        self.steps = 0
        self.body = frozenset()
        # 体的**位置**是本关的，身份（形状+配色）跨关保留 —— 和按键语义一样的处理：
        # 概念带走，坐标重新定位。
        self.bodies = {}
        self._blocked.clear()
        self._visited_targets.clear()
        self._noop_at.clear()
        self._noop_act.clear()
        self._last = None
        self._need_reset = False
        self._first_grid = g
        self._anim_seen.clear()
        # 判别实验的排期是本关的（换了关，局面全变了，上次问到哪儿不作数）
        self._probe_seen.clear()
        self._pending_judge = None
        # 目标也一样：**陈述**带走，坐标重新落定。但落定需要知道"我在哪"，
        # 而新关第一帧往往还认不出体 —— 那就先挂着，认出来再搬（见 _try_transfer）。
        if self.goal is not None:
            self._seed = self.goal
            self._seed_tries = 0
            self.goal = None
        self._try_transfer(g)
        if self.goal is None and self._seed is not None:
            self.say("[迁移] 新关还没认出体 → 目标先挂起，认出后自动重新定位")
        # 一关学完顺手存一次校准：跨关、跨盘、跨进程都能接着用
        if self.judge.path is not None:
            try:
                self.judge.save()
            except Exception:  # noqa: BLE001 —— 存不下去不能拖垮对局
                pass

    # ──────────────────────────────────────────────────────
    # 观测
    # ──────────────────────────────────────────────────────
    def observe(self, g0: Grid, act: str, g1: Grid,
                reward: bool = False, dead: bool = False,
                transient: Optional[dict] = None) -> Optional[Motion]:
        self.steps += 1
        if not g0 or not g1:
            return None
        if act == "RESET":
            # 重置把画面整体换掉，这一拍里"谁动了"没有任何意义：实测 ar25
            # 从这里把整列墙登记成了一个会左右移动的"体"，之后每次寻路都要
            # 在它身上试一遍。重置只作废位置记忆，不产出任何结论。
            self._pending = None
            self._blocked.clear()
            self._visited_targets.clear()
            self._last = (g1, act)
            return None

        m = fit(g0, g1, ignore=self._ambient, prefer=self.body_sig)
        moved = m is not None and m.moved
        # 只关心"我跟踪的那一块动没动"：另一块动了但我在墙这边被挡住，
        # 对这个键的语义来说仍然是"没动"。
        fmoved = moved and m.focus is not None and m.move != (0, 0)
        # transient：动作的中间帧改过、末帧又改回去的格子。它们不影响 g1，
        # 所以 fit 看不见；但它证明这个键**有可见效果**，不能记成"按了没用"。
        # 没有这一条，一个"只发一次扫描、不改变局面"的键会被永久判成 inert。
        unclear = ((m is not None) and (not moved) and not dead and not reward) \
            or bool(transient and (transient.get("transient_pixels") or 0))

        if transient and not dead and not reward:
            tp = transient.get("transient_pixels") or 0
            if tp and act not in self._anim_seen:
                self._anim_seen.add(act)
                bbox = transient.get("transient_bbox")
                self.say(f"[动画] {act} 在中间帧改了 {tp} 格又改回去"
                         f"（范围 {bbox}）—— 末帧里看不到，别把它当无效键")

        # ── 背景减除的证据来源：**没有整体平移的那些转移** ──────────
        # 画面没被任何块搬动，却还是变了 —— 这变化与按键无关（计时器、精灵动画、
        # 状态栏）。这是最干净的噪声样本，比"统计所有步"准得多：后者会把
        # 真正走过很多遍的路径也误判成噪声。
        if unclear:
            self._noop_n += 1
            for p in W.diff(g0, g1):
                self._noop_freq[p] = self._noop_freq.get(p, 0) + 1
            if self._noop_n >= 4:
                h, w = W.dims(g0)
                cand = frozenset(p for p, k in self._noop_freq.items()
                                 if k / self._noop_n >= 0.6)
                # 超过三成格子都在"自发"变化，那就不是噪声，是我解释错了
                if len(cand) <= 0.3 * h * w:
                    self._ambient = cand

        # ── 谁动了：按身份（形状+配色）分别记账 ─────────────────────
        # ar25 的 A3 让左块向西、右块向东。如果只盯着"一个聚焦块"，
        # 这个键的位移永远有两个互相矛盾的票，agent 会一直做判别实验而永不收敛。
        # 按身份分开以后，每个体自己的表都是唯一确定的。
        imoves: dict[tuple, Cell] = m.ident_moves if m is not None else {}
        seen_ids: set[tuple] = set()
        if m is not None:
            for C, d in m.motions.items():
                i = m.idents.get(C)
                if i is None:
                    continue
                seen_ids.add(i)
                self.keys.observe(act, d, ident=i)
                self.bodies[i] = C
                self._movers.add(i)
                if len(self.bodies) > 1 and self.steps <= 3:
                    self.say(f"[归纳] 记下了第 {len(self.bodies)} 个可动体："
                             f"{W.cells_text(C)}（{W.dir_text(d)}）")
        # 已知但这一拍没动的体：它被挡住了，或者这个键不控制它。
        # 只在"确实有别的块动了"时才记 —— 整拍没动的证据已经在 pending 里结算过，
        # 这里再记一次会把同一个位置重复计成"这个键对它无效"。
        if imoves:
            for i in list(self.bodies):
                if i not in seen_ids:
                    self.keys.observe(act, None, ident=i)

        # "在这个位置按它不动"≠"这个键没用"：角落会被墙挡住，换一处可能就动了。
        if self._pending is not None:
            p_act, p_pos = self._pending
            if p_act == act:
                for i, pos in p_pos.items():
                    if i in seen_ids:
                        self._noop_at.pop(i, None)
                        self._noop_act.pop(i, None)
                    else:
                        self._noop_at[i] = pos
                        self._noop_act[i] = act
                        # 从这个位置按它推不动这一块 —— 别再往这个方向撞
                        self._blocked.add((i, act, pos))
                self._pending = None

        # 全局表只在这拍**只有一个体在动**时才记：多个体反向移动时，
        # 全局表必然分裂成两条互斥的位移，它作为"兜底先验"就失效了。
        # 那种情况下各体自己的表已经把信息记全了。
        d = m.move if fmoved else None
        if dead or reward:
            self.keys.observe(act, None if dead else d, dead=dead, unclear=unclear)
        elif len(imoves) <= 1:
            self.keys.observe(act, d, unclear=unclear)

        self.trials.append(Trial(g0, act, g1, fmoved, bool(fmoved), reward, dead))
        # 上一拍按下前的预测，现在有实际结果了 → 结算给校准层
        self._settle_judge(imoves, fmoved, dead)

        if moved and not dead:
            if len(m.motions) > 1:
                self.say(f"[归纳] 一次动作有 {len(m.motions)} 块各自动了：{m.text()}")
            if m.body:
                sig = (W.normalise(m.body), W.colour_map(g0, m.body))
                if self.body_sig != sig and fmoved:
                    self.body_sig = sig
                    self.say(f"[归纳] 跟踪的可动体 = {W.cells_text(m.body)}，"
                             f"形状 {len(m.body)} 格")
            if fmoved:
                self.body = m.body_new
            if m.side:
                self.effects.observe(m.side, m.bodies)

        if dead:
            self._dead_acts.add(act)
            self._need_reset = True
            self.say(f"[失败] {act} 导致终止 —— 这条动作以后要绕开")
            return m

        if reward:
            self._induce_goal(g0, g1, m)

        self._last = (g1, act)
        return m

    def _induce_goal(self, g0: Grid, g1: Grid, m: Optional[Motion]) -> None:
        """从过关那一刻倒推目标。多体游戏里先问一句"哪个体真的达成了"。

        `g1` 必须是**胜利瞬间**那一帧，不能是下一关的开局 —— 后者与 g0 属于两个
        不同的世界，从它们身上减出来的任何东西都是幻觉。
        """
        tries: list[tuple[frozenset[Cell], Cell]] = []
        if m is not None and m.motions:
            tries = list(m.motions.items())
        else:
            tries = [(self._guess_body(g0), (0, 0))]
        side = m.side if m else {}

        # 免费的判别器，两条必须同时成立：
        #   · 过关**后**那一帧里，这条陈述成立；
        #   · 过关**前**那一帧里，它不成立。
        # 只留前一条的话，任何"永远成立"的空话都会胜出 —— 实测漏掉后半条时，
        # ar25 归纳出"碰到 36格@6-17,1-19"（整块画布），因为那个永远成立。
        for C, d in tries:
            after = W.shift(C, d)
            for c in G.induce(g0, g1, frozenset(C), d, side, level=self.level):
                if c.done(g1, after) and not c.done(g0, frozenset(C)):
                    self.goal = c
                    self.say(f"[归因] 过关事件 → 目标假设「{c.text()}」"
                             f"（依据：{c.note}；由 {W.cells_text(C)} 的这次移动解释；"
                             f"过关前不成立、过关后成立）")
                    return
        cands = G.induce(g0, g1, tries[0][0], tries[0][1], side, level=self.level)
        if not cands:
            self.say("[归因] 过关但画面几乎没变 —— 目标可能与视觉无关")
            return
        self.say(f"[归因] 没有哪条陈述能同时满足「过关前不成立、过关后成立」"
                 f"（备选 {len(cands)} 条）—— 这一关先不设目标，继续探索")

    # ──────────────────────────────────────────────────────
    # 认得当前的可动体（可以不止一个）
    # ──────────────────────────────────────────────────────
    def _find_by_ident(self, g: Grid, ident: tuple) -> Optional[frozenset[Cell]]:
        """拿身份（形状 + 配色）在当前画面里重新定位那个体。"""
        shape, cols = ident
        bg = W.bg_colour(g)
        comps = W.components(g, conn=4, same_colour=True, bg=bg)
        for s in comps:
            if W.normalise(s) == shape and W.colour_map(g, s) == cols:
                return s
        # 颜色被改写/吸收时退一步只比形状
        for s in comps:
            if W.normalise(s) == shape:
                return s
        return None

    def _locate_all(self, g: Grid) -> None:
        """把每个记过的体重新定位到当前画面。认不出的体就放弃（它消失了）。"""
        if not g:
            return
        for i in list(self.bodies):
            s = self._find_by_ident(g, i)
            if s is None:
                self.bodies.pop(i, None)
            else:
                self.bodies[i] = s

    # ── 目标跨关：陈述带走，坐标等认出体以后再落 ──────────────
    def _mover_cells(self, g: Grid) -> frozenset[Cell]:
        """当前画面上"已知会动的块"占的格子 —— 目标必须避开它们。"""
        if not self._movers or not g:
            return frozenset()
        bg = W.bg_colour(g)
        out: set[Cell] = set()
        for s in W.components(g, conn=4, same_colour=True, bg=bg):
            if (W.normalise(s), W.colour_map(g, s)) in self._movers:
                out |= s
        return frozenset(out)

    def _try_transfer(self, g: Grid) -> bool:
        """把挂着的目标陈述落到本关的坐标上。成功返回 True。"""
        seed = self._seed
        if seed is None or not g:
            return False
        bodies = [b for _i, b in self._bodies_by_priority()]
        if not bodies:
            guess = self._guess_body(g)
            bodies = [guess] if guess else []
        if not bodies:
            self._seed_tries += 1
            return False
        avoid = self._mover_cells(g)
        for b in bodies:
            cand = seed.relocate(g, b, avoid=avoid)
            if cand is None:
                continue
            if cand.done(g, b):
                # 新关一开局这条陈述就成立 → 它是空话，不是目标
                self.say(f"[迁移] 「{cand.text()}」在本关开头已经成立 → 不是目标，丢掉")
                self._seed = None
                return False
            self.goal = cand
            self._seed = None
            self.transfer_hits += 1
            self.say(f"[迁移] 上一关的目标陈述搬到本关：{cand.text()}"
                     f"（避开 {len(avoid)} 个已知会动的格子）")
            return True
        self._seed_tries += 1
        if self._seed_tries >= 6:
            self.say(f"[迁移] 「{seed.text()}」在本关试了 6 次都落不下来 → 放弃，回到探索")
            self._seed = None
        return False

    def _guess_body(self, g: Grid) -> frozenset[Cell]:
        """没有运动证据时，用"学过的那块形状"在当前画面里重新定位。"""
        if not g:
            return frozenset()
        if self.body_sig is not None:
            s = self._find_by_ident(g, self.body_sig)
            if s is not None:
                return s
        if self.bodies:
            # 多体时长的那块更可能是玩家；同长时取坐标靠前的，保证确定性
            return sorted(self.bodies.values(), key=lambda s: (-len(s), W.bbox(s)))[0]
        if self.body and all(0 <= r < len(g) and 0 <= c < len(g[0])
                             for r, c in self.body):
            # 形状认不出时，退到"上一帧它还在那儿"——块被吸收、被推走都靠这条兜底
            return self.body
        return frozenset()

    def locate(self, g: Grid) -> frozenset[Cell]:
        b = self._guess_body(g)
        if b:
            self.body = b
            return b
        return self.body

    def rebase(self, g: Grid) -> None:
        """原生画布尺寸变了 —— 这一关的**坐标**记忆整体作废，概念保留。

        尺度和位置是两件事：尺度（一格几像素）变了会让每个坐标的含义都变，
        所以位置类记忆（块在哪、哪儿推不动、哪个结构碰过了）全部清掉，
        形状和配色、按键语义、目标句式这些与坐标无关的结论留着。
        正常流程不会走到这里（感知在关内钉住尺寸），这是兜底。
        """
        self.bodies.clear()
        self.body = frozenset()
        self._blocked.clear()
        self._visited_targets.clear()
        self._noop_at.clear()
        self._noop_act.clear()
        self._pending = None
        self._noop_freq.clear()
        self._noop_n = 0
        self._ambient = frozenset()
        if self.goal is not None:
            self._seed = self.goal
            self.goal = None
        self._try_transfer(g)
        if self.goal is None and self._seed is not None:
            self.say(f"[尺度] 原生画布变成 {W.dims(g)}，目标「{self._seed.text()}」"
                     f"暂未落定 → 等认出体")
        self.say(f"[尺度] 原生画布变成 {W.dims(g)} —— 位置记忆作废，语义保留")

    # ──────────────────────────────────────────────────────
    # 决策
    # ──────────────────────────────────────────────────────
    def decide(self, g: Grid, acts: list[str]) -> str:
        acts = [a for a in acts if a not in self._dead_acts] or list(acts)
        if self._need_reset:
            self._need_reset = False
            return "RESET"
        self._locate_all(g)
        self.locate(g)                 # 兼容接口：更新"最像玩家"的那一个
        # 上一关的目标还挂着，现在（可能）认出体了 → 再试着落地
        if self.goal is None and self._seed is not None:
            self._try_transfer(g)

        def emit(act: str) -> str:
            self._pending = (act, {i: W.origin(b) for i, b in self.bodies.items() if b})
            self._note_prediction(act)
            return act

        # ① 有键从没做出过任何可归因的事 —— 不按它就永远不知道它能干什么
        unknown = [a for a in acts
                   if not self._touched(a)
                   and a not in self.keys.death
                   and self.keys.unclear.get(a, 0) < 2]
        if unknown:
            self.say(f"[提问] {unknown[0]} 的语义未知 → 按一次")
            return emit(unknown[0])

        # ①b 某个体在某个位置按这个键没动：可能只是被墙挡住，换一处再验
        stale = self._stale(acts)
        if stale is not None:
            i, a, pos = stale
            self.say(f"[提问] {a} 在 {pos} 推不动 "
                     f"{W.cells_text(self.bodies.get(i, frozenset()))}"
                     f" —— 可能只是被墙挡住，换一处再验")
            return emit(a)

        # ② 各体的表还都没定，而全局票互相矛盾 → 挑一个最该验的键问一次
        #
        # 这里以前是 `if conflicted and self.steps % 3 == 0`。那个 `% 3` 是纯粹
        # 人写的节流阀，它存在的原因只有一个：我们没有"这次问还有没有信息量"的
        # 度量。现在有了 —— 问由两个计数决定，不由步数决定：
        #     · 冲突票越多越值得问
        #     · 上次问它时票数是多少，这次还是多少，就说明问不出新东西
        conflicted = [a for a in acts
                      if not any(self.keys.known(a, i) for i in self.bodies)
                      and len(self.keys.votes.get(a, {})) > 1]
        if conflicted:
            a = self._pick_probe(conflicted)
            if a is not None:
                votes = self.keys.votes.get(a) or Counter()
                self.say(f"[提问] {a} 的位移出现分歧 {dict(votes)} "
                         f"→ 换个局面再问一次（不是按步数排期）")
                return emit(a)

        # ③ 目标已知 —— 看**每个**体能不能走到，选最近的那一对
        if self.goal is not None and self.bodies:
            a = self._pursue(g, acts)
            if a is not None:
                return emit(a)

        # ④ 目标未知 —— 去碰还没碰过的结构
        a = self._explore(g, acts)
        if a is not None:
            return emit(a)

        # ⑤ 什么都不成立：挑最少用过的键
        counts = Counter(t.act for t in self.trials[-40:])
        return emit(min(acts, key=lambda x: (counts[x], x)))

    # ── 判断层的真值闭环 ────────────────────────────────────
    def _note_prediction(self, act: str) -> None:
        """按下去之前，先记下"我预测它会怎么动"。

        这是校准层唯一的真值来源：不是人标的数据，是**下一帧自己打脸**。
        训练数据靠人标的项目，最后都会变成"标了多少就用多少"；靠现实的，会越用越准。
        """
        votes = self.keys.votes.get(act)
        if not votes:
            self._pending_judge = None
            return
        ans = self.judge.ask(cases.displacement(act, votes, who="全体"))
        self._pending_judge = ((ans, act, ans.choice)
                               if ans.choice is not None else None)

    def _settle_judge(self, imoves: dict, moved: bool, dead: bool) -> None:
        if self._pending_judge is None:
            return
        ans, act, want = self._pending_judge
        self._pending_judge = None
        if dead or not moved or not imoves:
            # 没动（被挡）或死了，这一拍没有"实际结果"可比 —— 不算它错
            return
        actual = list(imoves.values())
        if want in actual:
            self.judge.learn(ans, want)
            return
        if actual[0] in ans.raw:
            self.judge.learn(ans, actual[0])
        else:
            # 预测的位移根本不在候选里 —— 这是后端先验的问题，不是校准的问题
            self._judge_miss += 1
        self.say(f"[校验] {act} 预测走 {want}，实际 {actual} → 这条判断记错一次")

    # ── 判别性实验：问哪一个 ────────────────────────────────
    def _pick_probe(self, conflicted: list[str]) -> Optional[str]:
        """几个键都还在打鼓，下一次该验哪个？

        交给判断层，理由全部是计数：冲突票数、总票数、上次问它之后票数有没有涨。
        票数没涨 = 上一次在这个局面问它没拿到新信息，换一个问题比重复问更有用。
        """
        cands = []
        for a in conflicted:
            votes = self.keys.votes.get(a) or Counter()
            total = sum(votes.values())
            conflict = total - (votes.most_common(1)[0][1] if votes else 0)
            stale = self._probe_seen.get(a, -1) >= total
            cands.append((a, conflict, total, stale))
        q = cases.probe_target(cands)
        pick = self.judge.pick(q, at=0.5)
        if pick is None:
            return None
        self._probe_seen[pick] = next((n for a, _c, n, _p in cands if a == pick), 0)
        return pick

    # ── 这个键有没有做出过任何可归因的事 ────────────────────
    def _touched(self, act: str) -> bool:
        if self.keys.votes.get(act) or self.keys.noop.get(act):
            return True
        for i in self.bodies:
            if self.keys.by_id.get(i, {}).get(act):
                return True
            if self.keys.noop_id.get(i, {}).get(act):
                return True
        return False

    def _stale(self, acts: list[str]) -> Optional[tuple[tuple, str, Cell]]:
        """某个体对这个键记过"推不动"，但它现在换位置了 —— 值得再试一次。

        什么时候放弃再试，以前写的是 `noop_count >= 3`（"三个位置都推不动"）。
        现在问判断层，证据是两条计数：推不动几次、这个键在别处有没有动过。
        验算：动过(2 票支持) + 推不动 2 次 → 0.6，过；推不动 3 次 → 0.5，不过。
        和原来的 3 次上限等价，但它是从证据算出来的，不是定死的常数。
        """
        for i, b in self.bodies.items():
            a = self._noop_act.get(i)
            if a is None or a not in acts:
                continue
            pos = self._noop_at.get(i)
            if pos is None or pos == W.origin(b):
                continue
            if self.keys.best(a, i)[0] is not None:
                continue          # 这个体已经学到这个键了，不用再验
            noops = self.keys.noop_count(a, i)
            moved_elsewhere = bool(self.keys.votes.get(a) or
                                   self.keys.by_id.get(i, {}).get(a))
            q = cases.worth_recheck(a, noops, moved_elsewhere,
                                    who=W.cells_text(b))
            # 门槛 0.5 = "证据打平就试一次"：换位置试错的代价是一步，
            # 错过一个能动的键的代价是整局卡住，所以这里该松不该紧。
            if not self.judge.yes(q, at=0.5):
                continue          # 这个键在这个体上确实不动，不值得再花步子
            return (i, a, pos)
        return None

    # ── 某个体当前的"位移 → 键" ────────────────────────────
    def _steps_for(self, ident: tuple, acts: list[str], origin: Cell) -> dict[Cell, str]:
        """这个体手上所有**学到的**位移，不止 1 格。

        以前只放 1 格的步子，理由是"一格格走才好定位"。实测 ar25 的 L1 里
        一次 A3 就跳 2 格，于是这个体一个能用的步子都没有，寻路直接放弃，
        退回"算位移再分解"，而 2 格的步子永远对不上 1 格的残差 —— 卡死。
        寻路本来就是在真实画面上做 BFS（不穿实心物），步子多大都成立，
        限制成 1 格没有任何依据。
        """
        out: dict[Cell, str] = {}
        best_n: dict[Cell, int] = {}
        for a in acts:
            d, n, _c = self.keys.best(a, ident)
            if not d or d == (0, 0):
                continue
            if (ident, a, origin) in self._blocked:
                continue
            if d not in out or n > best_n[d]:
                out[d] = a
                best_n[d] = n
        return out

    def _bodies_by_priority(self):
        """长的那块优先（玩家通常比装饰大），同长按坐标，保证确定性。"""
        return sorted(((i, b) for i, b in self.bodies.items() if b),
                      key=lambda kv: (-len(kv[1]), W.bbox(kv[1])))

    # ── 寻路：在真实画面上走一步 ──────────────────────────
    def _nav_step(self, g: Grid, body: frozenset[Cell], tgt: frozenset[Cell],
                  steps: dict[Cell, str], onto: bool = False
                  ) -> tuple[str, Optional[str], int]:
        """返回 (状态, 该按的键, 还差几步)。

        为什么必须寻路，而不是"算位移再分解"：算位移会得出"往东 3 格"，
        但东边第 2 格是墙 —— 那条路根本不通，agent 会一直顶墙直到超时，
        而且它以为自己在执行计划。寻路用的只是"我不会穿过实心东西"这条
        最基本的常识，不含任何关卡知识。

        位移表不从 self.keys 里现取，而是由调用方按**具体的体**给出 ——
        ar25 的两个体对同一个键的位移是相反的，用同一张表会算出一个不存在的路。
        """
        if not body or not tgt or not steps:
            return ("no_keys", None, -1)
        h, w = W.dims(g)
        bg = W.bg_colour(g)
        shape = tuple(sorted(W.normalise(body)))
        base = W.origin(body)
        own = set(body)
        tgt_set = set(tgt)

        def footprint(o: Cell) -> list[Cell]:
            return [(o[0] + dr, o[1] + dc) for dr, dc in shape]

        def ok(o: Cell) -> bool:
            for r, c in footprint(o):
                if not (0 <= r < h and 0 <= c < w):
                    return False
                if (r, c) in tgt_set or (r, c) in own:
                    continue
                if g[r][c] != bg:
                    return False
            return True

        def near(cells: list[Cell]) -> bool:
            for r, c in cells:
                if (r - 1, c) in tgt_set or (r + 1, c) in tgt_set \
                        or (r, c - 1) in tgt_set or (r, c + 1) in tgt_set:
                    return True
            return False

        def done(o: Cell) -> bool:
            cells = footprint(o)
            if onto:
                hit = sum(1 for p in cells if p in tgt_set)
                return hit >= max(1, int(0.9 * len(cells)))
            return near(cells)

        if not ok(base):
            return ("unreachable", None, -1)
        if done(base):
            return ("ok", None, 0)

        seen = {base}
        q = deque([(base, "", 0)])
        while q:
            o, first, dist = q.popleft()
            for d, a in steps.items():
                nxt = (o[0] + d[0], o[1] + d[1])
                if nxt in seen or not ok(nxt):
                    continue
                seen.add(nxt)
                if done(nxt):
                    return ("ok", first or a, dist + 1)
                q.append((nxt, first or a, dist + 1))
        return ("unreachable", None, -1)

    # ── 朝目标走：每个体各算一遍，取最近的 ──────────────────
    def _pursue(self, g: Grid, acts: list[str]) -> Optional[str]:
        goal = self.goal
        assert goal is not None
        tgt = goal.target
        if not tgt:
            return None
        onto = goal.kind == "coincide"
        best: Optional[tuple[int, tuple, str]] = None

        for i, b in self._bodies_by_priority():
            if goal.done(g, b):
                continue          # 这个体已经达成了，换另一个去完成
            st, a, d = self._nav_step(g, b, tgt, self._steps_for(i, acts, W.origin(b)),
                                      onto=onto)
            if st == "ok" and a is not None and (best is None or d < best[0]):
                best = (d, i, a)
        if best is not None:
            d, i, a = best
            self.say(f"[计划] 目标「{goal.text()}」→ "
                     f"{W.cells_text(self.bodies[i])} 还剩 {d} 步，第一步 {a}")
            return a

        # 寻路用不上（没有 1 格的键）：退回"算位移 + 分解"。每拍重算，不缓存。
        for i, b in self._bodies_by_priority():
            ba = W.origin(b)
            vec = (W.origin(tgt)[0] - ba[0], W.origin(tgt)[1] - ba[1])
            if goal.kind == "touch":
                vec = _step_toward(vec, 1)
            if vec == (0, 0):
                self._visited_targets.add(ba)
                continue
            disp: dict[str, Cell] = {}
            for act in acts:
                d, _n, _c = self.keys.best(act, i)
                if d and (i, act, ba) not in self._blocked:
                    disp[act] = d
            plan = decompose(vec, disp)
            if not plan:
                continue
            self.say(f"[计划] 目标「{goal.text()}」→ 位移 {vec} "
                     f"= {'+'.join(plan[:8])}{'...' if len(plan) > 8 else ''}")
            return plan[0]
        return None

    # ── 探索：去碰没碰过的结构 ────────────────────────────
    def _explore(self, g: Grid, acts: list[str]) -> Optional[str]:
        bodies = self._bodies_by_priority()
        if not bodies:
            return None
        bg = W.bg_colour(g)
        occupied: set[Cell] = set()
        for _i, b in bodies:
            occupied |= b
        comps = [s for s in W.components(g, conn=4, same_colour=True, bg=bg)
                 if not (s & occupied)]
        cands = []
        for s in comps:
            o = W.origin(s)
            if o in self._visited_targets:
                continue
            cands.append((len(s), o, s))
        # 小的优先：能"碰一下"的东西通常小（按钮、道具、孪生体），通高的线是墙，
        # 撞上去只会白挨。这是"越小的结构越可能是机制载体"这个先验，不是关卡知识。
        cands.sort(key=lambda x: (x[0], x[1]))

        if not any(self._steps_for(i, acts, W.origin(b)) for i, b in bodies):
            return None      # 还没有能走 1 格的键，先别动

        for _n, _o, tgt in cands:
            best: Optional[tuple[int, tuple, str]] = None
            stuck: list[tuple[tuple, frozenset[Cell]]] = []
            for i, b in bodies:
                st, a, d = self._nav_step(g, b, tgt,
                                          self._steps_for(i, acts, W.origin(b)))
                if st == "ok" and a is not None and (best is None or d < best[0]):
                    best = (d, i, a)
                elif st == "ok":
                    stuck.append((i, b))
            if best is not None:
                d, i, a = best
                self.say(f"[探索] 目标未知 → {W.cells_text(self.bodies[i])} 去碰 "
                         f"{W.cells_text(tgt)}（{d} 步，第一步 {a}）")
                return a
            # 已经贴住了：探索要的不是"靠近"，是**压进去看它怎么反应**。
            for i, b in stuck:
                a2 = self._press_toward(g, b, tgt, acts, i)
                if a2 is not None:
                    self.say(f"[探索] {W.cells_text(b)} 已贴住 {W.cells_text(tgt)} "
                             f"→ 压进去看反应")
                    return a2
            self._visited_targets.add(W.origin(tgt))
            self.say(f"[探索] {W.cells_text(tgt)} 走不到 —— 被隔开，换下一个")
        return None

    def _press_toward(self, g: Grid, body: frozenset[Cell], tgt: frozenset[Cell],
                      acts: list[str], ident: Optional[tuple] = None) -> Optional[str]:
        """朝目标按一下。用**这个体学到的**单位位移选键，猜不到方向就不按。"""
        ba, ta = W.origin(body), W.origin(tgt)
        axes = sorted(((abs(ta[0] - ba[0]), 0), (abs(ta[1] - ba[1]), 1)), reverse=True)
        for _m, axis in axes:
            s = _sgn((ta[0] - ba[0]) if axis == 0 else (ta[1] - ba[1]))
            if s == 0:
                continue
            d = (s, 0) if axis == 0 else (0, s)
            for a in acts:
                dd, _n, _c = self.keys.best(a, ident)
                if dd == d and (ident, a, ba) not in self._blocked:
                    return a
        return None

    # ── 反馈：现实打脸 ────────────────────────────────────
    def note_blocked(self, act: str, body: frozenset[Cell],
                     ident: Optional[tuple] = None) -> None:
        if not body:
            return
        if ident is None:
            for i, b in self.bodies.items():
                if b == body:
                    ident = i
                    break
        self._blocked.add((ident if ident is not None else body, act, W.origin(body)))

    # ──────────────────────────────────────────────────────
    # 给人和给模型看的推理记录
    # ──────────────────────────────────────────────────────
    def text(self, limit: int = 8) -> str:
        goal = self.goal.text() if self.goal else (
            f"挂起（等认出体后落到本关）: {self._seed.text()}"
            if self._seed is not None else "未知（还在探索）")
        lines = [
            f"关卡 L{self.level}  已走 {self.steps} 步  试验 {len(self.trials)} 次",
            f"按键语义: {self.keys.text()}",
            f"副作用:   {self.effects.text()}",
            f"可动体:   {W.cells_text(self.body) if self.body else '未确认'}"
            + (f"（共 {len(self.bodies)} 个体）" if len(self.bodies) > 1 else ""),
            f"目标:     {goal}",
            f"判断层:   {self.judge.text()}"
            + (f"  先验落空 {self._judge_miss} 次" if self._judge_miss else ""),
        ]
        return "\n".join(lines)

    def recent(self, n: int = 6) -> list[str]:
        return self.notes[-n:]


# ─────────────────────────────────────────────────────────────
# 位移分解：把"要移动 3 格东 2 格南"变成按键序列
# ─────────────────────────────────────────────────────────────
def decompose(vec: Cell, disp: dict[str, Cell], limit: int = 40) -> list[str]:
    """BFS 在"位移格点"上找最短按键序列。

    不假设 ACTION1 是上、也不假设一格一按 —— 只用**学到的**位移集合。
    找不到就返回空（宁可不动，也不瞎按）。
    """
    if vec == (0, 0):
        return []
    if not disp:
        return []
    from collections import deque
    start = (0, 0)
    seen = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if len(seen) > 4000:
            break
        for a, d in disp.items():
            nxt = (cur[0] + d[0], cur[1] + d[1])
            if abs(nxt[0]) > abs(vec[0]) + 3 or abs(nxt[1]) > abs(vec[1]) + 3:
                continue
            if nxt in seen:
                continue
            seen[nxt] = (cur, a)
            if nxt == vec:
                seq = []
                node = nxt
                while seen[node] is not None:
                    prev, act = seen[node]
                    seq.append(act)
                    node = prev
                return list(reversed(seq))[:limit]
            q.append(nxt)
    return []


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def _step_toward(vec: Cell, k: int) -> Cell:
    def sgn(x: int) -> int:
        return (x > 0) - (x < 0)
    return (sgn(vec[0]) * min(abs(vec[0]), k), sgn(vec[1]) * min(abs(vec[1]), k))


def _blank_like(g: Grid) -> Grid:
    return tuple(tuple(0 for _ in row) for row in g)
