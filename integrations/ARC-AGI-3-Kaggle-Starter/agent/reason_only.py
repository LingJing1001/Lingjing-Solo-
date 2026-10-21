"""只用推理层闯关的 agent —— 一个用来证明"规则不是人写的"的对照组。

它和 self_learner 的区别不是参数，是**结构**：

    self_learner   目标层(goalsim) / 导航层(nav) / 状态图(graph) / 规则前瞻(worldsim)
                   每一层都是一段人为某个游戏写的规则，按优先级瀑布式下漏

    reason_only    只有一个 Reasoner：按键语义、可动体、副作用、目标，
                   全部由这一局的 (画面, 按键, 新画面) + 过关信号归纳出来

所以它的成绩可以直接回答一个问题：**把人的规则全拿掉，还能不能动**。
跑法：

    SELFLEARN_MAX_ACTIONS=400 .venv312/Scripts/python.exe \
        scripts/play_local.py --agent agent/reason_only.py --game ar25 --max-steps 400
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_FILE = Path(__file__).resolve()
_ROOT = _FILE.parents[3] if len(_FILE.parents) > 3 else _FILE.parent
for cand in (_ROOT, Path.cwd()):
    if (cand / "arc_adaptor" / "induction" / "engine.py").is_file():
        sys.path.append(str(cand))
        break

from agents.agent import Agent  # noqa: E402
from arcengine import FrameData, GameAction, GameState  # noqa: E402

from arc_adaptor.induction import Perception, Reasoner  # noqa: E402
from arc_adaptor.induction import world as W  # noqa: E402
from arc_adaptor.duck.champion import animation as AN  # noqa: E402

SIMPLE = ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5")


def _grid(fr: FrameData) -> W.Grid:
    layers = getattr(fr, "frame", None) or []
    arrs = [np.asarray(x) for x in layers if np.asarray(x).size]
    if not arrs:
        return ()
    return W.to_grid(arrs[-1])


def _px(fr: FrameData) -> tuple:
    """末帧的像素，不可变，用作动画链的起点。"""
    layers = getattr(fr, "frame", None) or []
    arrs = [np.asarray(x) for x in layers if np.asarray(x).size]
    if not arrs:
        return ()
    return tuple(tuple(int(v) for v in row) for row in arrs[-1].tolist())


def _animation(before_px: tuple, fr: FrameData) -> Optional[dict]:
    """一次动作的动画链里，有没有末帧看不到的变化。

    只取 `frame[-1]` 会漏掉一类动作：中途改了画面、结束时又改回去。
    实测 25 个盘连续按 8 个动作：bp35 每步都返回 3–5 帧、瞬态 14–79 格，
    g50t / tu93 只在特定动作多帧，sk48 / lf52 多帧但瞬态为 0（纯插值，
    末帧已包含结果）。单帧的 19 个盘（含 ar25）在这里直接返回 None，
    不花任何时间。
    """
    layers = getattr(fr, "frame", None) or []
    if os.environ.get("REASON_DEBUG_ANIM"):
        print(f"[anim-dbg] layers={len(layers)} before_px={len(before_px)}", flush=True)
    if len(layers) < 2 or not before_px:
        return None
    try:
        chain = AN.collapse_frames(before_px, AN.normalize_frames(layers))
        out = AN.summarize_animation(chain)
        if os.environ.get("REASON_DEBUG_ANIM"):
            print(f"[anim-dbg] chain={len(chain)} summary={out}", flush=True)
        return out
    except Exception as exc:  # noqa: BLE001 —— 视图出错不能拖垮对局
        if os.environ.get("REASON_DEBUG_ANIM"):
            print(f"[anim-dbg] EXC {type(exc).__name__}: {exc}", flush=True)
        return None


def _act_id(a) -> int:
    if isinstance(a, int):
        return a
    s = str(getattr(a, "name", a)).strip()
    if s.upper().startswith("ACTION"):
        s = s[6:]
    try:
        return int(s)
    except ValueError:
        return 0


class MyAgent(Agent):
    """没有规则，只有归纳器的 agent。"""

    MAX_ACTIONS: int = int(os.environ.get("REASON_MAX_ACTIONS", "600"))

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.brain = Reasoner()
        self.percept = Perception()
        self._lv: Optional[int] = None
        self._pending: Optional[tuple[str, W.Grid, tuple]] = None
        self._log_at = 0
        self._acts: list[str] = list(SIMPLE)
        self._frames_seen = 0
        self._shape: Optional[tuple[int, int]] = None

    # ── 过关那一步的"胜利瞬间" ─────────────────────────────
    def _win_grid(self, latest: FrameData) -> W.Grid:
        """过关这一步的帧链里，倒数第二张 = 胜利瞬间；最后一张已经是下一关。

        实测 ar25 过关（第 20 步 ACTION3）：这一**帧对象内部的** layers 有 3 张，
        前两张是胜利瞬间（块已经压在目标上），第三张是 L1 的开局。所以胜利帧是
        `latest.frame[-2]`，不是"上一步的那一帧" —— 后者是 `frames` 历史列表里
        的倒数第二个元素，拿到的是**上一回合**的画面，和过关前完全相同，
        归纳出来的东西必然自相矛盾（实跑就是这样：前后两帧一模一样）。

        而且它要用**旧锁**解码（属于上一关，不能进新关的帧历史）。
        只有 1 张时返回 ()：这一步直接把画面换成下一关，没有胜利瞬间可看。
        """
        layers = getattr(latest, "frame", None) or []
        if len(layers) < 2:
            return ()
        try:
            return self.percept.decode(W.to_grid(layers[-2]))
        except Exception:  # noqa: BLE001 —— 视图出错不能拖垮对局
            return ()

    # ── 动作名 → GameAction ────────────────────────────────
    def _emit(self, name: str) -> GameAction:
        if name == "RESET":
            a = GameAction.RESET
        else:
            a = getattr(GameAction, name, GameAction.ACTION1)
        a.reasoning = {"text": f"reason:{name}"}
        return a

    def _available(self, latest: FrameData) -> list[str]:
        aa = getattr(latest, "available_actions", None) or []
        acts = [f"ACTION{_act_id(x)}" for x in aa if _act_id(x)]
        # 只保留方向/功能键；点击类另走一条路，这里不做
        acts = [a for a in acts if a in SIMPLE]
        if not acts:
            acts = list(SIMPLE[:4])
        return acts

    # ── 主循环 ────────────────────────────────────────────
    def choose_action(self, frames: list[FrameData], latest: FrameData) -> GameAction:
        raw = _grid(latest)
        cur = self.percept.see(raw)
        lv = int(getattr(latest, "levels_completed", 0) or 0)
        self._frames_seen += 1
        if os.environ.get("REASON_DEBUG_DIMS"):
            print(f"[dims] raw={W.dims(raw)} cur={W.dims(cur)} k={self.percept.k} "
                  f"off={self.percept.offset} relocks={self.percept.relocks} "
                  f"rejected={self.percept.rejected}", flush=True)

        if self._lv is None:
            self._lv = lv
            self.brain.new_level(cur)
            print(f"[reason] 开局 L{lv} 像素 {W.dims(raw)} → 原生 {W.dims(cur)}"
                  f"（{self.percept.text()}）", flush=True)
            print(f"         {self.percept.note(cur)}", flush=True)
            if os.environ.get("REASON_DUMP_GRID"):
                print(f"[grid] L{lv}\n{W.to_text(cur)}", flush=True)

        # ① 结算上一次动作：过关/死亡/无事发生，都从这里喂给归纳器
        reward = self._lv is not None and lv > self._lv
        if self._pending is not None:
            act, prev, prev_px = self._pending
            dead = latest.state == GameState.GAME_OVER
            tr = _animation(prev_px, latest)
            # 过关那一步返回的帧序列里，**最后一帧已经是下一关的画面**。
            # 拿它当"过关后的画面"去和"过关前的画面"配对，等于拿两个不同世界
            # 做减法 —— 归纳出来的目标必然是幻觉（实测 ar25：得到"碰到
            # 36格@6-17,1-19"这种整块画布的东西）。真正的胜利帧在 frames[-2]。
            win = self._win_grid(latest) if reward else ()
            if reward and os.environ.get("REASON_DEBUG_WIN"):
                print(f"[win-dbg] frames={len(frames)} win={W.dims(win)} "
                      f"prev={W.dims(prev)} cur={W.dims(cur)} "
                      f"ambient={len(self.brain._ambient)}", flush=True)
                if win and os.environ.get("REASON_DEBUG_WIN") == "2":
                    print("[win-dbg] 过关前:", flush=True)
                    print(W.to_text(prev), flush=True)
                    print("[win-dbg] 胜利瞬间:", flush=True)
                    print(W.to_text(win), flush=True)
            if reward and not win:
                print("[reason] ★ 过关，但这一步只带回一帧（直接跳到下一关）"
                      "→ 这次转移横跨两关，不用它归纳目标", flush=True)
            else:
                self.brain.observe(prev, act, win or cur, reward=reward,
                                   dead=dead, transient=tr)
            self._pending = None
            if reward:
                self._lv = lv
                self._shape = None            # 新关：尺度重新学
                self.percept.reset_scale()
                self.brain.new_level(cur)
                print(f"[reason] ★ 过关 → L{lv}\n{self.brain.text()}", flush=True)
                if os.environ.get("REASON_DEBUG_WIN"):
                    for line in self.brain.recent(4):
                        print(f"[win-dbg]   {line}", flush=True)
                if os.environ.get("REASON_DUMP_GRID"):
                    print(f"[grid] L{lv}\n{W.to_text(cur)}", flush=True)
            elif dead:
                print(f"[reason] ✗ {act} 致死，本关重来", flush=True)

        # ①b 尺度换了：位置记忆不能再沿用，否则旧坐标会越界
        if not reward and self._shape is not None and W.dims(cur) != self._shape:
            print(f"[reason] 画布 {self._shape} → {W.dims(cur)}，清掉位置记忆", flush=True)
            self.brain.rebase(cur)
        if not reward:
            self._shape = W.dims(cur)

        # ② 让归纳器决定
        self._acts = self._available(latest)
        act = self.brain.decide(cur, self._acts)
        self._pending = (act, cur, _px(latest))

        if self.action_counter - self._log_at >= 40:
            self._log_at = self.action_counter
            print(f"[reason] t={self.action_counter}\n{self.brain.text()}", flush=True)
            for line in self.brain.recent(3):
                print(f"         {line}", flush=True)

        every = int(os.environ.get("REASON_DUMP_EVERY", "0") or 0)
        if every and self.action_counter % every == 0:
            print(f"[dump] t={self.action_counter} L{self.brain.level} "
                  f"目标={self.brain.goal.text() if self.brain.goal else '-'}", flush=True)
            print(W.to_text(cur), flush=True)
            for i, b in self.brain.bodies.items():
                print(f"   体 {W.cells_text(b)} 键: {self.brain.keys.body_text(i)}",
                      flush=True)
        return self._emit(act)

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        if latest_frame.state == GameState.WIN:
            print(f"[reason] WIN\n{self.brain.text()}", flush=True)
            for line in self.brain.notes[-25:]:
                print(f"    {line}", flush=True)
        return latest_frame.state == GameState.WIN
