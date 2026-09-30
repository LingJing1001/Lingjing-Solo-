"""R5 Update Reflect Agent — Layer 3b LLM顾问 + Layer 4 反思触发。

对应分工表：
  R5: LLM+反思 | Layer 3b + Layer 4 | LLM 顾问、反思触发 Prompt 设计

对「所有更新」做结构化反思：做错了什么 / 下一步 / 反事实 / 正事实 / 因果关联。
默认封闭符号复盘（不依赖外网 LLM）；可选注入 llm_fn。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from lingjing_solo.reflection.prompt import build_r5_update_prompt
from lingjing_solo.reflection.skill import R5_UPDATE_SKILL
from lingjing_solo.world_model.update_causal import UpdateCausalModel


PRIOR_CAUSAL = """C1 本地通关 ≠ Phase B 加载同一 agent
C2 kernels push ≠ competitions submit 成功
C3 submit 脚本曾误报 Submitted（API 400 仍绿字）
C4 publicScore≈0.15 ≈ 聚合后约 1 关有效（25 游戏均值）
C5 L1–L7 在 kernel 内，但 CreateCodeSubmission 400 → Phase B 未跑本版
C6 榜上 Version4 = INLINE v3 ls20×2，仍 0.15；与 L1–L7 无关
"""


DEFAULT_UPDATE_EVENTS: list[dict[str, Any]] = [
    {
        "when": "2026-08-27",
        "title": "首次提交 lingjing_solo v0.3.1",
        "detail": "几乎无可靠脚本通路",
        "result": "publicScore 0.14",
    },
    {
        "when": "2026-08-30 ~ 09-01",
        "title": "ScriptBank v1/v2（ls20×2 + ar25×1）",
        "detail": "依赖 planning/data JSON 或 package 路径在 Phase B 落地",
        "result": "连续 publicScore 0.15（脚本未真正生效的典型指纹）",
    },
    {
        "when": "2026-09-03 12:48 CST",
        "title": "INLINE hardcoded v3（ls20×2+ar25×1）竞赛提交成功",
        "detail": "声称脚本写进 my_agent；对应 Notebook Version 4",
        "result": "COMPLETE → 仍是 0.15（有效贡献仍≈1关）",
    },
    {
        "when": "2026-09-03 ~13:10 CST",
        "title": "整合 GitHub L1–L7 进 my_agent + ScriptBank + notebook v5/v6",
        "detail": "本地 play_local ls20=7/7 WIN score=100；期望聚合≈4.11",
        "result": "kernels push 成功；competitions submit 400 — 未进榜",
    },
    {
        "when": "2026-09-03 13:10 CST",
        "title": "submit.ps1 误报",
        "detail": "CreateCodeSubmission 400 后仍打印 Submitted",
        "result": "团队误以为 L1–L7 已打榜",
    },
    {
        "when": "2026-09-03 18:50 CST",
        "title": "用户截图确认榜分",
        "detail": "最新成功提交仍是 Version4 v3 描述；勾选 Version3 也是 0.15",
        "result": "确认：看到的 0.15 不是 L1–L7 的成绩",
    },
    {
        "when": "2026-09-03 18:53+ CST",
        "title": "再次尝试 submit kernel v6",
        "detail": "PENDING 已结束仍 400",
        "result": "配额或其他 API 限制；L1–L7 仍未绑定 Phase B",
    },
]


@dataclass
class ReflectionReport:
    prompt: str
    answer_md: str
    triggers: list[str] = field(default_factory=list)
    mode: str = "symbolic"
    generated_at: str = ""
    causal_render: str = ""

    def as_markdown(self) -> str:
        ts = self.generated_at or datetime.now(timezone.utc).isoformat()
        causal = self.causal_render or "（无）"
        return (
            f"# R5 更新反思报告（Layer 3b + Layer 4 + UpdateCausalModel）\n\n"
            f"- 生成时间：{ts}\n"
            f"- 模式：{self.mode}\n"
            f"- 触发：{', '.join(self.triggers) or '无'}\n\n"
            f"## 因果模型摘要\n\n```text\n{causal}\n```\n\n"
            f"---\n\n{self.answer_md}\n\n"
            f"---\n\n<details><summary>完整反思 Prompt</summary>\n\n"
            f"```text\n{self.prompt}\n```\n\n</details>\n"
        )


class R5UpdateReflectAgent:
    """工程更新反思 Agent（非对局逐步执行器）。"""

    ROLE = "R5: LLM+反思"
    LAYERS = "Layer 3b + Layer 4"
    FOCUS = "LLM 顾问、反思触发 Prompt 设计"

    def __init__(
        self,
        llm_fn: Optional[Callable[[str], str]] = None,
        causal: Optional[UpdateCausalModel] = None,
    ) -> None:
        self.llm_fn = llm_fn
        self.causal = causal or UpdateCausalModel()
        self.events: list[dict[str, Any]] = []
        self.triggers: list[str] = []

    def reset(self) -> None:
        self.events.clear()
        self.triggers.clear()
        self.causal = UpdateCausalModel()

    def ingest(self, event: dict[str, Any]) -> None:
        self.events.append(dict(event))
        self.causal.ingest_event(event)

    def ingest_many(self, events: list[dict[str, Any]]) -> None:
        for e in events:
            self.ingest(e)

    def layer4_evaluate(self) -> list[str]:
        """Layer 4：从事件结果推断反思触发标签。"""
        triggers: list[str] = []
        blob = " ".join(
            f"{e.get('title','')} {e.get('detail','')} {e.get('result','')}"
            for e in self.events
        ).lower()
        if "0.15" in blob or "0.14" in blob:
            triggers.append("score_stuck_015")
        if "400" in blob or "配额" in blob:
            triggers.append("quota_block")
        if "误报" in blob or "submitted" in blob and "400" in blob:
            triggers.append("submit_misreport")
        if "未进榜" in blob or "未绑定" in blob or "phase b 未" in blob:
            triggers.append("phase_b_not_bound")
        if "push 成功" in blob and "submit" in blob and "400" in blob:
            triggers.append("phase_b_not_bound")
        # 去重保序
        seen: set[str] = set()
        out: list[str] = []
        for t in triggers:
            if t not in seen:
                seen.add(t)
                out.append(t)
        self.triggers = out
        return out

    def build_prompt(self, *, prior_causal: str = PRIOR_CAUSAL) -> str:
        if not self.triggers:
            self.layer4_evaluate()
        return build_r5_update_prompt(
            self.events,
            triggers=self.triggers,
            prior_causal=prior_causal,
            causal_snapshot=self.causal.render_for_prompt(),
        )

    def symbolic_advise(self, prompt: str) -> str:
        """封闭模式：基于事件时间线的确定性因果复盘（不调用外网）。"""
        has_l17_local = any("L1–L7" in str(e) or "L1-L7" in str(e) for e in self.events)
        has_submit_fail = any("400" in str(e.get("result", "")) for e in self.events)
        has_015 = any("0.15" in str(e.get("result", "")) for e in self.events)
        return f"""## 1. 我做错了什么

1. **把「kernels push / notebook 有代码」当成了「竞赛 Phase B 已跑本版」** —— 这是最大混淆。
2. **信任了 submit.ps1 的绿色 Submitted** —— 实际 `CreateCodeSubmission` 已 400，L1–L7 从未进队。
3. **过早宣布测试–打榜对齐完成** —— 本地 ls20=100 只证明 HARDCODED 内容对；不能证明榜上跑的是同一绑定。
4. **对 Version4（v3 ls20×2）仍出 0.15 的根因跟进不足** —— 即便「写进 my_agent」，Phase B 有效贡献仍≈1关，说明 L2+ 或 agent 注册/game_id/执行路径仍可能断链；不能假设「hardcode 就一定全生效」。

## 2. 我接下来怎么做

**P0**
1. 等竞赛提交配额恢复后，**只提交 kernel 含 `submit-ls20x7+ar25x1` 的版本**，message 必须含 `L1-L7`。
2. 提交成功后查 Submissions 列表：必须出现**新条目**；打开 Phase B 日志确认 `HARDCODED_CHECK` / `ls20_levels=7`。
3. 在日志确认前，**不要**再向用户宣称「已打榜对齐」。

**P1**
4. 保持 `submit.ps1` 失败即 exit 1（已修）；加一道门禁：submit 后立刻 `competitions submissions` 校验 description。
5. 针对 Version4 仍 0.15：拉 Phase B 日志做 **agent 是否加载 / HARDCODED 是否打印 / ls20 过几关** 的三联检查，修断链后再扩关。

## 3. 如果不这么做会怎么样

- 继续 push notebook 而不成功 `competitions submit` → **榜分永远停在旧 Version4 的 0.15**。
- 继续用旧描述/旧勾选版本 → 团队会误判「整合失败/算法不行」，实际是 **绑定错误**。
- 不查 Phase B 日志 → 无法区分「脚本没加载」vs「脚本加载了但关卡序列不对」→ 重复无效提交烧配额。

## 4. 如果这么做会怎么样

- 若 Phase B 真正跑到 L1–L7 HARDCODED 且公开集仍含 ls20/ar25 → **期望 publicScore ≈4.11 量级**（本地证据：ls20=100，ar25≈2.78）。
- 若日志有 HARDCODED_CHECK 但仍 ≈0.15 → 因果切换到「评测环境差异 / game_id 不匹配 / 隐藏集不同」，再开新因果支路，而不是再改本地 ScriptBank 路径空想。
- 若 submit 持续 400 → 分数不会变；唯一损失是时间，需换日配额或 Web UI 手动 Submit。

## 5. 跟之前的因果相关吗

**高度相关，是同一条因果链的延伸，不是新病。**

| 先验 | 本次是否命中 |
|------|----------------|
| C1 本地≠Phase B 同码 | 命中：本地 L1–L7 通，榜上从未跑该码 |
| C2 push≠submit | 命中：push v6 成功，submit 400 |
| C3 误报 Submitted | 命中：已发生并误导决策 |
| C4 0.15≈约1关 | 命中：Version4 仍 0.15，与 v1/v2 同指纹 |
| C5 L1–L7 未绑定 Phase B | 命中：榜上无 L1–L7 描述条目 |
| C6 看到的 0.15 是旧版 | 命中：截图 Version4 = v3 ls20×2 |

补充：Version4「已 hardcode」仍 0.15，说明 **C4 的「脚本未充分生效」支路仍未关闭**；L1–L7 提交成功后若仍 0.15，才需要升级为「评测集/环境差异」新假设。

## 6. 一句话判决

**错在「未确认 Phase B 绑定就宣称对齐」；下一步是成功挂上带 L1–L7 的竞赛提交并用日志验 `HARDCODED_CHECK`——否则分数会继续钉死在 0.15，与本地 100 分无关。**

---
（本段由 R5UpdateReflectAgent 封闭符号模式生成；Prompt 技能：{R5_UPDATE_SKILL[:40]}…）
本地整合证据：has_l17_local={has_l17_local}, has_submit_fail={has_submit_fail}, has_015={has_015}
"""

    def reflect(
        self,
        *,
        use_default_timeline: bool = False,
        prior_causal: str = PRIOR_CAUSAL,
    ) -> ReflectionReport:
        if use_default_timeline and not self.events:
            self.ingest_many(DEFAULT_UPDATE_EVENTS)
        self.layer4_evaluate()
        prompt = self.build_prompt(prior_causal=prior_causal)
        if self.llm_fn is not None:
            answer = self.llm_fn(prompt)
            mode = "llm"
        else:
            answer = self.symbolic_advise(prompt)
            mode = "symbolic"
        return ReflectionReport(
            prompt=prompt,
            answer_md=answer,
            triggers=list(self.triggers),
            mode=mode,
            generated_at=datetime.now(timezone.utc).isoformat(),
            causal_render=self.causal.render_for_prompt(),
        )


# 兼容对局框架：若被当作模板 Agent 误加载，给出明确身份
class MyAgent:  # noqa: N801 — 仅防误用，不用于 Kaggle 主提交
    """占位：R5 更新反思不是竞赛主 Agent；请用 agent/my_agent.py。"""

    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "R5UpdateReflectAgent 是工程反思顾问，不能作为竞赛 MyAgent。"
            "请使用 agent/my_agent.py（BUILD_TAG=submit-ls20x7+ar25x1）。"
        )
