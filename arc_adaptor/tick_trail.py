"""逐 tick 证据写入器（团队规范 §5.2 recording JSONL）。

为什么要单独一个模块，而不是像 LS20 运行器（`run_ls20_r2r3.py:447-513`）那样把
`build_tick` 摊在主循环里：AR25 的引擎推进分散在四处（`run_names` 的罐头回放与校验回放、
R4 贪心步进、主循环补放），而其中**只有"真的留在引擎上的那些动作"才算证据**——beam 与
状态搜索在 finally 里 `_restore` 回入口帧，它们打过的每一帧都只是搜索展开，写进 recording
等于把"搜索尝试过"冒充"环境发生过"（§10）。所以这里必须支持：

    begin_attempt → record… → commit_attempt   （成功：整段进本关缓冲，落盘由 flush_level 决定）
    begin_attempt → record… → abort_attempt    （失败回滚：整段丢弃，一行都不留）

第二个结构性约束：AR25 的 plan 是**每关一份**、覆盖整条路径，而且它在 `r234_solve` 里
晚于路径成功才造出来（validity 取决于有没有真通关）。所以 tick 行不能"当场"写——那样
本关第一行永远拿不到自己的 plan。这里的做法是：已提交的转移先进本关缓冲，
`attach_plan(plan)` + `flush_level()` 时才编号落盘，并把完整 plan 只盖在本关第一行
（其余行带 `plan_id` 作为 join 键；每行重复一份整关 plan 会让 recording 臃肿八倍）。

顺带做一件哈希审计：每个已提交动作都比对「语义 `state_hash` 变没变」和「搜索键
`engine_key_hash` 变没变」，两个方向的分歧分开计数（`r2_ar25` 的模块 docstring 说明了
为什么搜索键不能当状态身份）。前者变、后者不变 = 新哈希比搜索键更细（预期，`_state_key`
不含旋转距离）；后者变、前者不变 = 新哈希可能漏了一类可变状态，必须进 `report.json` 的
limitations，而不是被"这局跑通了"掩盖。

依赖只有 stdlib + `evidence_compat` + `r2_ar25`（后两个本身 stdlib-only），所以 Tier A
测试在没有 `arcengine` 的解释器里也能跑，不必像 ② 的回归测试那样整份 skip。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import evidence_compat as ev
import r2_ar25


class TickTrailError(RuntimeError):
    """证据写入顺序不对（缺基线、没 begin 就 commit、没 flush 就 close）时抛出。"""


@dataclass(frozen=True)
class _Committed:
    """一条已提交（真留在引擎上）的转移。tick 号要到 flush 时才定得下来。"""

    before: dict[str, Any]
    action: str
    after: dict[str, Any]
    frame: Any
    transition: dict[str, Any]


class TickTrail:
    def __init__(self, path: Path | str, *, run_id: str, episode_id: str,
                 legal_actions: tuple[str, ...] | list[str], game_id: str = "ar25") -> None:
        self.path = Path(path)
        self.run_id = run_id
        self.episode_id = episode_id
        self.game_id = game_id
        self.legal_actions = [str(a) for a in legal_actions]
        self._handle = self.path.open("w", encoding="utf-8")
        self._attempt: list[_Committed] = []
        self._level: list[_Committed] = []
        self._baseline_obs: dict[str, Any] | None = None
        self._flushed_obs: dict[str, Any] | None = None
        self._in_attempt = False
        self._plan: dict[str, Any] | None = None
        self._baseline_written = False
        self.ticks_written = 0
        self.discarded = 0
        self.actions_committed = 0
        self.audit: dict[str, Any] = {
            "steps_only": 0,                  # 语义没变、只扣一步（单拼块时的 ACTION5）
            "hash_split_engine_key_only": 0,   # 搜索键变了、语义哈希没变 → 新哈希可能欠收
            "hash_split_semantic_only": 0,     # 语义哈希变了、搜索键没变 → 新哈希更细
            "chain_breaks": 0,                 # 引擎位置和最后一条证据对不上 → 有动作没进证据
            "plan_link": {"plans": 0, "linked": 0},   # plan.input_state_hash 是否就是本关首行的入口态
            "examples": [],
        }

    # ── 基线 ─────────────────────────────────────────────────
    def baseline(self, obs: dict[str, Any], frame: Any) -> dict[str, Any]:
        """tick 0：`replay_recording` 要求首行的 `requested_action` 是 RESET（或缺省）。"""
        if self._baseline_written:
            raise TickTrailError("baseline 只能写一次")
        row = ev.build_tick(
            run_id=self.run_id, episode_id=self.episode_id, tick=0, frame=frame,
            requested_action={"name": "RESET", "payload": {}}, settled_frame=True,
            state="RESET", levels_completed=int(obs["levels_completed"]), score=None,
            legal_actions=self.legal_actions, state_hash=obs["state_hash"],
            evidence_refs=[obs["observation_id"]],
            game_specific={"semantic_state": obs["game_specific"]["semantic_state"],
                           "level_index": obs["levels_completed"],
                           "steps_left": obs["game_specific"]["steps_left"]})
        self._write({"data": row, "observation": obs})
        self._baseline_written = True
        self._baseline_obs = obs
        self.ticks_written = 1
        return row

    # ── 尝试（一次"打上去再决定要不要留着"的段落）────────────
    def begin_attempt(self) -> None:
        if self._in_attempt:
            raise TickTrailError("上一段尝试还没 commit/abort")
        self._attempt = []
        self._in_attempt = True

    def commit_attempt(self) -> int:
        """引擎确实前进到了这里：本段转入关缓冲，等 plan 到位再落盘。"""
        if not self._in_attempt:
            raise TickTrailError("没有进行中的尝试可提交")
        self._in_attempt = False
        n = len(self._attempt)
        self._level.extend(self._attempt)
        self._attempt = []
        self.actions_committed += n
        return n

    def abort_attempt(self) -> int:
        """引擎被 `_restore` 回滚：一行都不留，但记下丢弃行数以便核对总量。"""
        n = len(self._attempt)
        self._attempt = []
        self._in_attempt = False
        self.discarded += n
        return n

    def record(self, before_obs: dict[str, Any], action: str,
               after_obs: dict[str, Any], frame: Any) -> _Committed:
        """登记一条**已提交**（真留在引擎上）的转移；失败段落由 abort_attempt 整段丢弃。

        调用方的两条义务：
          1. 这个动作确实改到了引擎上（否则别 record，或记下后 abort）；
          2. `after_obs` 是动作之后从引擎重采的观测——它的 `tick`（等于 flush_level 之后
             本行的行号）、`state_hash`、`game_specific.level_index` 都是**输出态**身份，
             拿入口态复用会让逐 tick 报告把"本 tick 完成后的状态"写成"开始前的状态"。
        """
        if not self._in_attempt:
            raise TickTrailError("record 前必须 begin_attempt（回滚过的动作不该进证据）")
        if not self._baseline_written:
            raise TickTrailError("recording 缺基线行：先写 baseline()")
        if action not in self.legal_actions:
            raise ev.EvidenceValidationError(f"illegal action: {action}")
        trans = r2_ar25.transition(before_obs, action, after_obs)
        self._audit_hash(trans, action)
        item = _Committed(before_obs, action, after_obs, frame, trans)
        self._attempt.append(item)
        return item

    # ── 本关落盘 ─────────────────────────────────────────────
    def attach_plan(self, plan: dict[str, Any] | None) -> None:
        """收下本关的 plan，并当场核对它声明的 `input_state_hash` 就是本关的入口态。

        §3.2 字段③ 的意义全在这里：plan 必须说明"相对哪个局面成立"。对不上不代表 plan 一定
        错，但代表这份 plan 和这份 recording 之间没有绑定关系，九字段就退化成自说自话。
        """
        self._plan = plan
        if plan is None:
            return
        link = self.audit["plan_link"]
        link["plans"] += 1
        # 入口态 = 本关第一条已提交转移的 before 观测；整关一步没动（如已胜再进）时就是当前态。
        entry = self._level[0].before if self._level else self.last_obs
        if entry is not None and entry["state_hash"] == plan.get("input_state_hash"):
            link["linked"] += 1
        elif len(link.setdefault("examples", [])) < 5:
            link["examples"].append({
                "plan_id": plan.get("plan_id"),
                "input_state_hash": plan.get("input_state_hash"),
                "entry_state_hash": None if entry is None else entry["state_hash"],
            })

    def check_continuity(self, state_hash: str) -> bool:
        """运行器每关入口调一次：引擎当前局面必须就是最后一条证据的终态。

        对不上说明有动作真的改了引擎却没被记进 recording（例如某层搜索不回滚却被当成回滚）。
        这不抛异常——抛在这里会让跑测因记账问题中断，而记账问题该以 `chain_breaks` 的形式
        进 `report.json` 并让 verdict 降级，判成 BLOCKED 而不是"没跑完"。
        """
        last = self.last_obs
        if last is not None and last["state_hash"] == state_hash:
            return True
        self.audit["chain_breaks"] += 1
        if len(self.audit["examples"]) < 5:
            self.audit["examples"].append({
                "why": "引擎当前 state_hash 与最后一条证据的终态不符：有动作没进 recording",
                "recorded": None if last is None else last["state_hash"],
                "engine": state_hash,
            })
        return False

    def flush_level(self) -> int:
        """把本关已提交的转移编号落盘；完整 plan 只盖在第一行，其余行只带 plan_id。"""
        if self._in_attempt:
            raise TickTrailError("flush_level 前要先 commit/abort 当前尝试")
        plan, self._plan = self._plan, None
        for index, item in enumerate(self._level):
            # 编号从 ticks_written 起算：baseline 占 tick 0 且已计入 ticks_written，
            # 再加 1 会让第一行变成 tick 2（tick 1 凭空消失，replay 的连续编号检查就红）。
            tick_no = self.ticks_written + index
            # 完整 plan 只进第一行（每行重复整关 plan 会让 recording 臃肿八倍），但 plan_id
            # 是外键：其余行不带它，事后就无法把某一步 join 回它所属的那份计划。
            row = self._build_tick(item, tick_no,
                                   plan if index == 0 else None,
                                   plan.get("plan_id") if plan else None)
            record: dict[str, Any] = {"data": row, "observation": item.before,
                                      "transition": item.transition}
            if index == 0 and plan:
                record["plan"] = plan
            self._write(record)
        n = len(self._level)
        if n:
            # 缓冲清空后引擎还停在最后一行的终态；不记这里，下一关的 last_obs 会退回基线，
            # 于是 check_continuity 从第二关起每关都误报一次 chain_break。
            self._flushed_obs = self._level[-1].after
        self._level = []
        self.ticks_written += n
        return n

    def close(self) -> dict[str, Any]:
        """收尾：没 flush 的本关行也要落盘（否则失败的最后一关会丢证据）。"""
        if self._in_attempt:
            self.abort_attempt()
        pending = len(self._level)
        if pending:
            self.flush_level()
        self._handle.close()
        final = self.last_obs
        return {"path": self.path.name, "ticks": self.ticks_written,
                "actions": self.actions_committed, "discarded_rows": self.discarded,
                "flushed_at_close": pending,
                "final_state_hash": None if final is None else final["state_hash"],
                "format": "lingjing-evidence-v1 build_tick（含基线行），每行包在 data 字段里",
                "backend": ev.BACKEND, "hash_audit": dict(self.audit)}

    # ── 内部 ─────────────────────────────────────────────────
    @property
    def plan_hash_linked(self) -> bool:
        """每一份已 attach 的 plan 都对得上本关入口态（一条没 attach 时按"无声明"记 True）。

        运行器把它并进 `report.json` 的判据；`plans` 计数另处给出，避免"没有 plan"被读成
        "plan 都对上了"。
        """
        link = self.audit["plan_link"]
        return link["plans"] == link["linked"]

    @property
    def last_obs(self) -> dict[str, Any] | None:
        """**与引擎当前位置对得上**的那条观测：尝试末条 → 本关缓冲末条 → 已落盘末条 → 基线。

        有了它，调用方不必自己维护 `prev_obs`，回滚也自带正确性：`abort_attempt` 丢掉缓冲后
        这里自动退回尝试开始前的观测，跟 `_restore` 之后的引擎是同一个局面。
        `flush_level` 之后本关缓冲是空的，引擎却还停在最后一行的终态——所以要接着看
        `_flushed_obs`，否则会退回基线，让下一关的 `check_continuity` 误报。
        """
        for buffer in (self._attempt, self._level):
            if buffer:
                return buffer[-1].after
        return self._flushed_obs if self._flushed_obs is not None else self._baseline_obs

    @property
    def next_tick(self) -> int:
        """本条 record 落盘时会拿到的 tick 号（observation 的 id 用它，避免两套编号）。

        与 `flush_level` 的编号同一套口径：基线已占 tick 0 并计入 `ticks_written`，
        所以下一行是 `ticks_written + 未落盘行数`，不再额外 +1。
        """
        return self.ticks_written + len(self._level) + len(self._attempt)

    def _build_tick(self, item: _Committed, tick_no: int, plan: dict[str, Any] | None,
                    plan_id: str | None = None) -> dict[str, Any]:
        request: dict[str, Any] = {"name": item.action, "payload": {}}
        try:                                     # §5.2 的 id 是引擎动作编号；读不出来就不填
            request["id"] = int(item.action.replace("ACTION", ""))
        except ValueError:
            pass
        after = item.after
        return ev.build_tick(
            run_id=self.run_id, episode_id=self.episode_id, tick=tick_no, frame=item.frame,
            requested_action=request, settled_frame=True, state=after["state"],
            levels_completed=int(after["levels_completed"]), score=None,
            legal_actions=self.legal_actions, state_hash=after["state_hash"],
            plan_id=plan_id if plan_id is not None else (plan or {}).get("plan_id"),
            plan=plan,
            evidence_refs=[item.before["observation_id"]],
            game_specific={"semantic_state": after["game_specific"]["semantic_state"],
                           "level_index": after["levels_completed"],
                           "steps_left": after["game_specific"]["steps_left"]})

    def _audit_hash(self, trans: dict[str, Any], action: str) -> None:
        semantic_moved = trans["moved"]
        before_key, after_key = trans["before_engine_key"], trans["after_engine_key"]
        key_moved = None if before_key is None or after_key is None else before_key != after_key
        if not semantic_moved and trans["steps_delta"] < 0:
            self.audit["steps_only"] += 1
        if key_moved and not semantic_moved:
            self.audit["hash_split_engine_key_only"] += 1
            if len(self.audit["examples"]) < 5:
                self.audit["examples"].append(
                    {"action": action, "before": trans["before_hash"],
                     "why": "搜索键变了、语义哈希没变：新哈希可能少收了一类可变状态"})
        elif semantic_moved and key_moved is False:
            self.audit["hash_split_semantic_only"] += 1

    def _write(self, record: dict[str, Any]) -> None:
        self._handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._handle.flush()
