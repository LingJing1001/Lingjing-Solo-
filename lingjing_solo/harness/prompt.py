"""提示词拼装：**字节稳定的共享前缀** + 易变后缀。

这是 C-Local 旁路的**第二块**，也是容量模型给出的最关键一条结论的落点：

    "前缀缓存值钱的不是省 prefill（只有 1% 量级），是省显存 ——
     公共前缀只存一份 → 腾出显存 → 抬高并发 → 轮数变少。"

要把这条兑现，前缀必须满足两个条件：

1. **跨回合字节不变**（同一局内 t=1..T 完全一致）；
2. **跨局字节不变**（不同游戏共用同一份前缀）。

于是本模块把提示词**物理拆成两段**：:data:`SYSTEM_PREFIX` 是模块级常量字符串，
构造后永不参与拼接；:func:`build_prompt` 只负责生成后缀。
:func:`prompt_stats` 给出 ``shared_prefix_tokens``，可直接填进
``capacity.Workload(shared_prefix_tokens=...)`` 重跑容量估算。

设计上还有一个刻意的取舍：**把"变化的部分"全部挤出前缀**。
- 不做 "第 3 回合" 这类回合号（会破坏跨回合稳定性）→ 回合信息放后缀。
- 不在前缀里放游戏名 → 跨局共享。
- 记忆（scratchpad）放后缀，且只放**键名列表**而不是值（值放文本末尾，
  这样记忆膨胀也不会顶掉前面的稳定段）。
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .object_graph import Graph, render

__all__ = [
    "SYSTEM_PREFIX",
    "PromptParts",
    "build_prompt",
    "prompt_stats",
    "estimate_tokens",
    "prefix_signature",
]


# --------------------------------------------------------------------------- #
# 稳定段（模块级常量，永不参与拼接）
# --------------------------------------------------------------------------- #

# 注意：这段字符串是**逐字节被哈希的**。任何编辑都会让所有已缓存的
# KV 块失效；改动成本在心里有个数（见 docs 里的 prefix_signature 记录）。
SYSTEM_PREFIX = """You are an agent solving an interactive grid puzzle, one action at a time.
You never see raw pixels. You see a list of OBJECTS extracted from the current frame.
Your only tool is a Python REPL. You decide by writing code, not prose.

=== OBSERVATION: THE OBJECT GRAPH ===
Each turn you receive a block like:

    FRAME 64x64 objects=7 adjacency=3
    #0 c=0 n=4096 r=31 c=31 h=8f2a11de bbox=0,0-63,63 kids=[1,2]
    #1 c=5 n=48   r=31 c=30 h=77aa0011
    adj 0-1 1-2 0-3

Field meanings:
- `#id`   stable index of the object within THIS frame (top-left reading order).
- `c=`    one hex digit 0..f: the object's colour.
- `n=`    number of cells the object covers.
- `r=`,`c=`  the centre (row, col) of the object's bounding box. Use these for clicking
             and for relative direction between two objects.
- `h=`    a TRANSLATION-INVARIANT shape signature (colour + normalized shape).
          Two objects with the same `h` are the SAME KIND of thing, even if they moved.
          This is how you recognise an object across turns: match on `h`, not on `#id`.
- `bbox=` bounding box as `r0,c0-r1,c1` (inclusive).
- `kids=[...]` ids of objects directly enclosed by this one (a containment tree).
- `adj a-b`  pairs of objects that physically touch (share an edge).

Reading the graph:
- Large `n` (hundreds to thousands) = background or wall or border. Usually inert.
- Small `n` (1..50) = a sprite: player, token, gem, button, cursor, switch.
- An object with `kids` is a CONTAINER; something acts on it or through it.
- Objects that MOVE between turns (same `h`, different `r`/`c`) are the dynamic ones.

=== ACTIONS ===
Call `action(name)` exactly once per turn to choose your move. Names:
    "UP", "DOWN", "LEFT", "RIGHT"      step one cell in that direction
    "SPACE"                             the neutral/interact key
    "CLICK", with row=..., col=...      click a specific grid cell
`CLICK` is used on a target cell, typically the centre of a small object:
    action("CLICK", row=o["r"], col=o["c"])
Choose exactly one action. If you call more than once, only the LAST call counts.
There is no Undo action. Every action advances the game.

=== THE REPL ===
Available names inside your code block:
- `objects`  : list of dicts, one per object, with keys
               id, color, hash, n, r, c, bbox, children, layer.
- `history`  : list of past steps, oldest first. Each is a dict with keys
               turn, action, moved (bool), changed (bool), objects_before, objects_after.
- `memory`   : a dict YOU own. It persists across turns within a level. Use it to
               record hypotheses, counters, or a plan. Values must be JSON-safe
               (int / float / str / bool / list / dict / None).
- `action(name, row=None, col=None)` : choose an action (see above).
- `log(msg)` : append a note to your reasoning trace (does not affect the game).

Rules:
- The REPL is STATELESS between turns except for `memory`.
  Recompute anything you need from `objects` and `history` every turn.
- Allowed imports: math, collections, itertools, functools, heapq, bisect,
  statistics, re, json, copy, operator, string, random, typing, dataclasses, enum.
  Any other import is blocked. There is no file, network, or process access.
- Wall-clock limit: 30 seconds. If you time out, the turn is wasted.
- Always end by calling `action(...)`. Emitting code without an action wastes the turn.

=== RESPONSE FORMAT ===
Reply with ONE fenced python block and nothing else. Example:

```python
small = [o for o in objects if 1 <= o["n"] <= 50]
if small:
    t = max(small, key=lambda o: o["n"])
    log(f"clicking {t['hash']} at {t['r']},{t['c']}")
    action("CLICK", row=t["r"], col=t["c"])
else:
    action("SPACE")
```

=== HOW TO PLAY WELL ===
- Form a hypothesis, then design the smallest code that TESTS it. Prefer an action
  that distinguishes two explanations over one that just moves.
- Track what changed: compare `objects` now against the previous turn in `history`.
  An object that changed colour or count is a strong signal about the rules.
- Reuse `memory` so you do not re-derive the same facts every turn.
- When something works once, record the exact conditions in `memory` and repeat them.
- When nothing changes for several turns, you are stuck. Do something else: try
  `SPACE`, try clicking a different object, try the opposite direction.
- Prefer specific targets over the screen centre. The centre of the frame is almost
  never a meaningful target; click the centre of an actual small object instead.
"""


# --------------------------------------------------------------------------- #
# 后缀（易变段）
# --------------------------------------------------------------------------- #


@dataclass
class PromptParts:
    prefix: str
    suffix: str
    turn: int = 0

    @property
    def text(self) -> str:
        return self.prefix + self.suffix

    @property
    def prefix_chars(self) -> int:
        return len(self.prefix)

    @property
    def suffix_chars(self) -> int:
        return len(self.suffix)


def estimate_tokens(text: str) -> int:
    """词表无关的 token 估算（≈4 字符/token）。

    这是给容量模型用的**量级**估计，不是精确计数：真正上线要用目标模型的
    tokenizer 复测一遍（见 docs 的"已知局限"）。它只用来判断
    "共享前缀到底是 2k 还是 12k token"这种量级问题。
    """
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _fmt_history(history: Sequence[Dict[str, Any]], tail: int) -> List[str]:
    if not history:
        return ["  (no steps yet)"]
    h = history[-tail:] if tail > 0 else history
    out = []
    for rec in h:
        t = rec.get("turn")
        a = rec.get("action")
        moved = "moved" if rec.get("moved") else "stayed"
        changed = "board_changed" if rec.get("changed") else "board_same"
        out.append(f"  t={t} {a} {moved} {changed}")
    if tail > 0 and len(history) > tail:
        out.insert(0, f"  ...({len(history) - tail} earlier steps omitted)")
    return out


def _fmt_memory(memory: Dict[str, Any]) -> List[str]:
    if not memory:
        return ["  (empty)"]
    out = []
    for k in sorted(memory):  # 排序 -> 稳定
        v = memory[k]
        s = repr(v)
        if len(s) > 160:
            s = s[:157] + "..."
        out.append(f"  {k} = {s}")
    return out


def build_prompt(
    graph: Graph,
    history: Sequence[Dict[str, Any]] = (),
    last_result: Optional[str] = None,
    memory: Optional[Dict[str, Any]] = None,
    turn: int = 0,
    max_objects: int = 0,
    max_adjacency: int = 0,
    history_tail: int = 12,
) -> PromptParts:
    """拼出 :class:`PromptParts`。``prefix`` 恒等于 :data:`SYSTEM_PREFIX`。"""
    lines: List[str] = []
    lines.append(f"\n=== CURRENT TURN (t={turn}) ===")
    lines.append(render(graph, max_objects=max_objects, max_adjacency=max_adjacency))

    lines.append("")
    lines.append("=== RECENT HISTORY ===")
    lines.extend(_fmt_history(history, history_tail))

    if last_result is not None:
        lines.append("")
        lines.append("=== LAST ACTION RESULT ===")
        lines.append(f"  {last_result}")

    lines.append("")
    lines.append("=== YOUR MEMORY ===")
    lines.extend(_fmt_memory(memory or {}))

    lines.append("")
    lines.append("=== NOW: write one python block that calls action(...) ===")

    return PromptParts(prefix=SYSTEM_PREFIX, suffix="\n".join(lines) + "\n", turn=turn)


def prefix_signature() -> str:
    """共享前缀的 sha1 前 16 位。改动 :data:`SYSTEM_PREFIX` 后它会变 —— 
    记录在文档里，用来判断"这次改动会不会让全部 KV 缓存失效"。"""
    return hashlib.sha1(SYSTEM_PREFIX.encode("utf-8")).hexdigest()[:16]


def prompt_stats(
    graph: Optional[Graph] = None,
    history: Sequence[Dict[str, Any]] = (),
    memory: Optional[Dict[str, Any]] = None,
    turn: int = 0,
) -> Dict[str, Any]:
    """共享前缀 / 后缀的字符数与 token 估算 —— 直接喂 ``capacity.Workload``。"""
    prefix = SYSTEM_PREFIX
    if graph is None:
        suffix = ""
    else:
        suffix = build_prompt(graph, history=history, memory=memory, turn=turn).suffix

    p_tok = estimate_tokens(prefix)
    s_tok = estimate_tokens(suffix)
    return {
        "prefix_chars": len(prefix),
        "suffix_chars": len(suffix),
        "prefix_tokens": p_tok,
        "suffix_tokens": s_tok,
        "total_tokens": p_tok + s_tok,
        "shared_prefix_tokens": p_tok,
        "prefix_signature": prefix_signature(),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _load_real_frame():
    """尽量从 data/aop/episodes 取一帧真帧；取不到就返回 None。"""
    import glob
    import json
    import os

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pat = os.path.join(root, "data", "aop", "episodes", "*", "recording.jsonl")
    for path in sorted(glob.glob(pat)):
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    data = rec.get("data") or {}
                    if data.get("frame"):
                        return os.path.basename(os.path.dirname(path)), data["frame"]
        except Exception:
            continue
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="object_graph+prompt", description="共享前缀与提示词统计")
    ap.add_argument("--suffix", action="store_true", help="同时统计真实帧的后缀开销")
    ap.add_argument("--max-objects", type=int, default=0)
    ap.add_argument("--max-adjacency", type=int, default=0)
    ap.add_argument("--quiet", action="store_true", help="只输出键值行")
    a = ap.parse_args(argv)

    graph = None
    game = None
    if a.suffix:
        got = _load_real_frame()
        if got:
            game, frame = got
            from .object_graph import build_graph

            graph = build_graph(frame)

    st = prompt_stats(graph=graph, turn=0)
    if not a.quiet:
        print("提示词分段统计")
        print("-" * 56)
    print(f"prefix_signature      {st['prefix_signature']}")
    print(f"shared_prefix_chars   {st['prefix_chars']}")
    print(f"shared_prefix_tokens  {st['shared_prefix_tokens']}")
    if graph is not None:
        print(f"frame_game            {game}")
        print(f"frame_objects         {graph.n_objects}")
        print(f"suffix_chars          {st['suffix_chars']}")
        print(f"suffix_tokens         {st['suffix_tokens']}")
        print(f"total_tokens          {st['total_tokens']}")
    print("")
    print(f"capacity.Workload(shared_prefix_tokens={st['shared_prefix_tokens']})")

    if a.suffix and graph is not None and not a.quiet:
        print("")
        print(render(graph, max_objects=a.max_objects, max_adjacency=a.max_adjacency))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
