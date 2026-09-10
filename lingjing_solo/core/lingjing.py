"""钱学森「灵境」纲领 → ARC-AGI-3 工程转义表。

致敬钱学森先生「灵境」构想：人机深度结合、统一事实源、局部高精度观测。
本模块只做**可运行的工程映射**，不在评测中求解真实场方程 / 度规张量。

核心一句话（继承原方案）：
    空间积分压缩泡壁边界，边界按需投影四维时空。
ARC 转义：
    未观测区粗粒化；变化区（泡壁）高精度；动作即源项 ΔJ 写回 Φ。
"""
from __future__ import annotations

# ---------- 协议对照（PDF 协议 1-11 → Solo 层）----------
PROTOCOL_MAP = {
    1: {
        "name": "时空基底",
        "solo": "WorldModelField + 64×64 网格状态哈希",
        "status": "active",
    },
    2: {
        "name": "信息密度 Φ 与可求解量",
        "solo": "PhiDensity（颜色/变化熵粗粒化）→ PerceptionSnapshot",
        "status": "active",
    },
    3: {
        "name": "信息密度决定弯曲",
        "solo": "∇Φ 梯度 → 探索方向偏置（非真实曲率求解）",
        "status": "active_lite",
    },
    4: {
        "name": "Agent 求解场 → 感知",
        "solo": "PerceptionEncoder.perceive() → PerceptionSnapshot",
        "status": "active",
    },
    5: {
        "name": "决策写回信息场",
        "solo": "inject_source_term(ΔJ) + 环境 step 为真相演化",
        "status": "active",
    },
    6: {
        "name": "多智能体共识 / 版本号",
        "solo": "单 Agent：field.version 单调递增；环境帧即唯一事实源",
        "status": "active_solo",
    },
    7: {
        "name": "场动力学 / 面积律",
        "solo": "泡壁 ROI 高精度 + 迷雾区粗粒；LLM/搜索预算节制",
        "status": "active_lite",
    },
    8: {
        "name": "Agent 身份与记忆",
        "solo": "trajectory + C 值（自指复杂度代理）",
        "status": "active",
    },
    9: {
        "name": "迷雾区展开",
        "solo": "Agent 步入新 ROI 时提升清晰度 / 保留粗粒统计",
        "status": "active_lite",
    },
    10: {
        "name": "外部接入接口",
        "solo": "LLMPlanner / harness.MyAgent / GameAction 适配",
        "status": "active",
    },
    11: {
        "name": "人类交互接口",
        "solo": "评测期关闭；本地可注入 llm_fn / Logger",
        "status": "deferred",
    },
}

# L0-L6 与 Solo 目录映射
LAYER_MAP = {
    "L0": ("比特池 / 网格基元", "perception + grid bytes"),
    "L1": ("因果图 / 转移记忆", "world_model.transition_*"),
    "L2": ("场与几何（轻量）", "PhiDensity + gradient bias"),
    "L3": ("泡壁投影感知", "BubbleWall + PerceptionSnapshot"),
    "L4": ("自指监视", "ReflectionTrigger.C"),
    "L5": ("观测者上行反馈", "SourceTerm / choose_action"),
    "L6": ("跨任务技能迁移", "transfer.TransferLayer / SkillLibrary"),
}

MANIFESTO = (
    "继承钱学森「灵境」：以人为主、人机结合；"
    "Agent 不靠互发消息同步，而读写统一可版本化的信息场 Φ；"
    "算力按泡壁面积分配，而非全域体积暴力。"
)


def protocol_status_summary() -> str:
    lines = [MANIFESTO, "", "协议转义状态："]
    for k, v in PROTOCOL_MAP.items():
        lines.append(f"  P{k:02d} {v['name']}: [{v['status']}] → {v['solo']}")
    return "\n".join(lines)
