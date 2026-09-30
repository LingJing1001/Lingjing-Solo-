"""统一 R3 plan 出口（团队规范《最小可运行闭环》§3.2 的九字段契约）。

为什么这个文件必须在 `lingjing_solo/` 而不是 `arc_adaptor/`：
`pyproject.toml` 里 `packages = ["lingjing_solo"]`，ARC checkout 的 venv 用 editable
安装只暴露这一个包（`_editable_impl_lingjing_solo.pth` 内容就是本仓库根目录）。
契约放 `arc_adaptor/` 的结果是"离线跑测能 import、线上策略 import 不到"，
那正是第二份真源。

分工（本模块只描述，不做决策）：
    R3 各 planner  →  build_plan()  →  validate_plan()  →  调用方执行
`candidate_actions` 只允许 **abstract action name**（如 ``"ACTION3"``）；abstract →
`GameAction` / `ActionInput` 的转换归 ARC boundary adapter（规则 5），本模块不碰引擎枚举，
这样带 payload 的复杂动作（实测 `GameAction.ACTION6.is_complex() == True`）不会因为
"plan 里塞了个数字"而丢掉 x/y。

fail-closed：任何一条断言不过就抛 `PlanContractError`，绝不"缺字段就当默认值"。
理由见 2026-09-19 的教训——旧 R2 把 HUD 步数条当目标时也没报错，于是一路错到决策层。
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from ..sica.rule_lifecycle import ComplexityBudget, measure_complexity
PLAN_SCHEMA = "lingjing-r3-plan-v1"

#: 团队规范 §3.2 `:121-132` 要求的九个字段，一个都不能少。
REQUIRED_FIELDS: tuple[str, ...] = (
    "plan_id", "planner", "input_state_hash", "candidate_actions", "expected_goal",
    "cost", "search_budget", "validity", "evidence_refs",
)

#: `validity` 的合法取值；`none` 表示"这 tick 没产出任何可执行计划"（决策失败，不是没跑）。
VALIDITY_STATES: frozenset[str] = frozenset({"candidate", "verified_offline", "expired", "none"})

#: planner 注册表：名字 → 这层在干什么。**新 planner 必须登记在这里**，
#: 否则 `validate_plan` 会拒绝——防止各运行器自造 planner 名导致统计口径漂移（§7.5 命名漂移条）。
PLANNER_NAMES: dict[str, str] = {
    "t0_known_solution": "已验证罐头解（分数下限，优先级最高）",
    "astar": "网格/位置 A*（抽象动作，只做寻路）",
    "state_search": "显式状态空间搜索（按联合状态键去重）",
    "beam": "启发式 beam search",
    "generic": "无游戏专用知识时的兜底 planner",
}


class PlanContractError(RuntimeError):
    """plan 不符合 §3.2 契约。抛出来比"降级放行"有用：静默的 plan 等于没有闸门。"""


def register_planner(name: str, description: str) -> None:
    """注册一个 planner 名。已存在但描述不同则报错，避免同名两义。"""
    if not name or not name.strip():
        raise PlanContractError("planner 名不能为空")
    existing = PLANNER_NAMES.get(name)
    if existing is not None and existing != description:
        raise PlanContractError(f"planner {name} 已注册为 {existing!r}，不能改描述")
    PLANNER_NAMES[name] = description


def normalize_budget(budget: Mapping[str, Any] | None) -> dict[str, int]:
    """预算键名统一到规范示例的 `time_ms`/`max_nodes`/`max_depth`。

    运行器历史上写的是 `time_s`（`arc_adaptor/agents/strategies/run_ls20_r2r3.py:373-375`），
    这里集中换算一次，而不是让两边各写各的——`search_budget` 一旦有两种口径，
    "搜索预算有界"这条门槛就没法机器检查。
    """
    src = dict(budget or {})
    out: dict[str, int] = {}
    if "time_ms" in src:
        out["time_ms"] = int(src["time_ms"])
    elif "time_s" in src:
        out["time_ms"] = int(round(float(src["time_s"]) * 1000))
    for key in ("max_nodes", "max_depth"):
        if key in src:
            out[key] = int(src[key])
    return out


def build_plan(*, planner: str, input_state_hash: str, candidate_actions: Sequence[str],
               expected_goal: str, cost: int | None, search_budget: Mapping[str, Any] | None,
               validity: str, evidence_refs: Iterable[Any] | None = None,
               plan_id: str | None = None, subgoals: Sequence[str] | None = None,
               legal_actions: Sequence[str] | None = None,
               complexity_budget: Mapping[str, Any] | None = None,
               complexity_cost: float | None = None,
               **extra: Any) -> dict[str, Any]:
    """组装并校验一份 plan。`extra` 供各游戏挂自己的诊断字段（如 `solver_phase`）。

    `legal_actions` 是当帧合法动作集合：给了就当场校验候选动作是否越界（§7.1 门槛「动作属
    合法集」），并且**留在 plan 里**——plan 与它是在哪个合法集合下成立的必须一起可审计，
    否则事后换一个 legal 集合重放就无法复现当初的判定。
    """
    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA,
        "plan_id": plan_id or f"{input_state_hash[:12]}-{planner}",
        "planner": planner,
        "input_state_hash": input_state_hash,
        # 不做 str() 兜底：把 GameAction 枚举或裸 int 静默转成 "3" 正好绕过②的类型断言，
        # 于是"abstract name"这条规则只在写代码时被遵守、落盘后没人管。原样交给 validate。
        "candidate_actions": list(candidate_actions),
        "expected_goal": expected_goal,
        "cost": cost,
        "search_budget": normalize_budget(search_budget),
        "validity": validity,
        "evidence_refs": list(evidence_refs or []),
    }
    if legal_actions is not None:
        plan["legal_actions"] = list(legal_actions)
    if subgoals is not None:
        plan["subgoals"] = list(subgoals)
    if complexity_budget is not None:
        plan["complexity_budget"] = dict(complexity_budget)
    if complexity_cost is not None:
        plan["complexity_cost"] = float(complexity_cost)
    plan.update(extra)
    return validate_plan(plan, legal_actions=plan.get("legal_actions"))


def validate_plan(plan: Any, *, legal_actions: Sequence[str] | None = None,
                  require_registered_planner: bool = True) -> dict[str, Any]:
    """六条断言（§3.2 + §7.1 R3 门槛）。`legal_actions=None` 表示调用方没当帧合法集，跳过②。

    `require_registered_planner`：生成 plan 时必须为 True（⑤ 防命名漂移）。**审计已落盘的
    recording 时要为 False**——注册表是运行器在 import 时填的，独立审计进程没 import 它就
    是空的，拿它当读取门槛会让旧 recording 再也读不了（违反 §7.2「不覆盖既有 recording」）。
    """
    if plan is None:
        raise PlanContractError("plan 为 None：R3 这 tick 没产出计划")
    if not isinstance(plan, dict):
        raise PlanContractError(f"plan 必须是 dict，收到 {type(plan).__name__}")

    # ① 九字段齐全
    missing = [f for f in REQUIRED_FIELDS if f not in plan]
    if missing:
        raise PlanContractError(f"plan 缺字段 {missing}（§3.2 九字段）")

    # ③ 输入状态哈希：没有它，plan 无法说明"相对哪个局面"
    if not isinstance(plan["input_state_hash"], str) or not plan["input_state_hash"]:
        raise PlanContractError("input_state_hash 必须是非空字符串")

    if not isinstance(plan["plan_id"], str) or not plan["plan_id"]:
        raise PlanContractError("plan_id 必须是非空字符串")
    if not isinstance(plan["expected_goal"], str) or not plan["expected_goal"]:
        raise PlanContractError("expected_goal 必须是非空字符串（目标分解的产物，不能留常量占位）")

    # ② 候选动作必须是 abstract name，且（给了合法集时）全部落在其中。
    #    调用方没传合法集时回退到 plan 自带的那份：从 recording 里读出来的 plan 已经存了
    #    legal_actions，事后重放也要能查这一条，否则门槛只在生成时有效、落盘后失效。
    actions = plan["candidate_actions"]
    if not isinstance(actions, list) or not all(isinstance(a, str) and a for a in actions):
        raise PlanContractError("candidate_actions 必须是 list[str]（abstract action name）")
    if legal_actions is None:
        legal_actions = plan.get("legal_actions")
    if legal_actions is not None:
        allowed = set(legal_actions)
        illegal = [a for a in actions if a not in allowed]
        if illegal:
            raise PlanContractError(f"候选动作不在当帧合法集合内: {illegal} ⊄ {sorted(allowed)}")

    # ④ 搜索预算有界：三个键都得是正数
    budget = plan["search_budget"]
    if not isinstance(budget, dict) or not budget:
        raise PlanContractError("search_budget 必须是非空 dict（预算无界 = 不可验收）")
    bad = {k: v for k, v in budget.items()
           if not isinstance(v, int) or isinstance(v, bool) or v <= 0}
    if bad:
        raise PlanContractError(f"search_budget 存在非正/非整数键: {bad}")

    # ⑤ planner 必须登记在注册表里（仅生成时；见上面 docstring 的说明）
    planner = plan["planner"]
    if require_registered_planner and planner not in PLANNER_NAMES:
        raise PlanContractError(
            f"未登记的 planner {planner!r}；先调用 register_planner()（已注册: {sorted(PLANNER_NAMES)}）")

    # ⑥ validity 取值 + verified_offline 必须带证据
    validity = plan["validity"]
    if validity not in VALIDITY_STATES:
        raise PlanContractError(f"validity {validity!r} ∉ {sorted(VALIDITY_STATES)}")
    if validity == "verified_offline" and not plan["evidence_refs"]:
        raise PlanContractError("validity=verified_offline 必须带非空 evidence_refs")
    if validity == "none" and actions:
        raise PlanContractError("validity=none 却带候选动作：自相矛盾的 plan 比没有 plan 更坏")
    if plan["cost"] is not None and (not isinstance(plan["cost"], int)
                                     or isinstance(plan["cost"], bool) or plan["cost"] < 0):
        raise PlanContractError(f"cost 必须是非空 int 或 None，收到 {plan['cost']!r}")

    # R3：复杂度由结构计算，不能只相信 planner 自报的数字。
    profile = measure_complexity({"candidate_actions": actions, "expected_goal": plan["expected_goal"],
                                  "subgoals": plan.get("subgoals", [])})
    supplied_cost = plan.get("complexity_cost")
    if supplied_cost is not None and abs(float(supplied_cost) - profile.cost) > 1e-9:
        raise PlanContractError(f"complexity_cost 与结构度量不一致: {supplied_cost} != {profile.cost}")
    plan["complexity_cost"] = profile.cost
    raw_budget = plan.get("complexity_budget")
    if raw_budget is not None:
        if not isinstance(raw_budget, dict) or "max_cost" not in raw_budget:
            raise PlanContractError("complexity_budget 必须包含 max_cost")
        try:
            budget = ComplexityBudget(float(raw_budget["max_cost"]),
                                      int(raw_budget["max_depth"]) if "max_depth" in raw_budget else None)
        except (TypeError, ValueError) as exc:
            raise PlanContractError("complexity_budget 数值非法") from exc
        if not budget.accept(profile):
            raise PlanContractError(f"复杂度超过预算: cost={profile.cost}, budget={raw_budget}")

    return plan
