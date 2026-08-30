"""
build_arch.py — 生成 Lingjing-Solo v1.0 架构图
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.font_manager as fm

# 中文字体
for fp in ["/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
           "/usr/share/fonts/wenquanyi/wqy-microhei/wqy-microhei.ttc"]:
    try:
        fm.fontManager.addfont(fp)
        break
    except Exception:
        pass
plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(15, 10))
ax.set_xlim(0, 15)
ax.set_ylim(0, 10)
ax.axis("off")

def box(x, y, w, h, title, body, color):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12",
                       linewidth=1.5, edgecolor="#333", facecolor=color, alpha=0.95)
    ax.add_patch(b)
    ax.text(x + w/2, y + h - 0.32, title, ha="center", va="top",
            fontsize=10, fontweight="bold", color="#111")
    ax.text(x + w/2, y + h/2 - 0.25, body, ha="center", va="center",
            fontsize=8.2, color="#222", linespacing=1.45)

def arrow(x1, y1, x2, y2, label="", color="#444"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                        linewidth=1.6, color=color)
    ax.add_patch(a)
    if label:
        ax.text((x1+x2)/2 + 0.05, (y1+y2)/2 + 0.22, label, fontsize=7.5,
                color=color, ha="center")

C_ENV  = "#dceef8"   # 环境
C_PER  = "#e6f5ea"   # 感知
C_WM   = "#fff2cc"   # 世界模型（核心）
C_PLAN = "#fce4d6"   # 规划
C_REF  = "#e8e0f5"   # 反思
C_TELE = "#f0f0f0"   # 观测
C_KAG  = "#d5f0e8"   # Kaggle

# 标题
ax.text(7.5, 9.6, "Lingjing-Solo v1.0 架构", ha="center", fontsize=16, fontweight="bold")
ax.text(7.5, 9.15, "可执行关系型世界模型 · 模拟器内零步规划 · 状态隔离（浅拷贝根因已修复）",
        ha="center", fontsize=9, color="#555")

# 顶层：Kaggle 烟囱口
box(5.3, 8.15, 4.4, 0.75, "Kaggle Agent (agent.py)", "is_done / choose_action · 八级决策 · 无网络降级", C_KAG)

# 环境
box(0.4, 0.35, 3.0, 1.0, "ARC-AGI-3 隐藏环境", "64×64 / 零指令 / 无网络\nRHAE 平方惩罚", C_ENV)

# L0 感知
box(4.0, 0.35, 3.0, 1.0, "L0 Perception", "CNN 编码 + 差分 ROI\n连通域分割 · ObjectTracker", C_PER)

# L1 世界模型场 Φ（核心，放大）
box(7.9, 0.35, 3.2, 1.9, "L1 Φ 场 / WMP v2 (codegen.py)",
    "转移表 (s,a)→s'\n关系规则 · push/联动\nLLM Layer A ⇄ 规则 Layer B\n"
    "simulate = 深拷输入 (v1.0 修复)", C_WM)

# 关系归纳
box(11.8, 0.35, 2.8, 1.9, "关系归纳 (induction)\nrelations.py",
    "ADJACENT / IN_FRONT_OF\n假设-检验 · 置信度晋升\n冲突检测 · CEGIS 重建", C_WM)

# L2 探索
box(0.4, 3.0, 3.0, 1.15, "L2 Exploration", "信息增益 + 对象拉力\n主动探索 · probe · Retrodict 逆推", C_PER)

# L3 规划（核心，放大）
box(4.0, 3.0, 3.0, 1.15, "L3 Planning (planner.py)", "BFS + 转移图缓存\n深度自适应 · 浅拷贝\n★首层短路(7256→4)", C_PLAN)

# 四级后继
box(7.9, 3.0, 3.2, 1.15, "四级后继优先级", "①转移表 ②WMP模拟器\n③对象级 ④探索评分", C_PLAN)

# L4 反思
box(11.8, 3.0, 2.8, 1.15, "L4 Reflection", "循环/冲突/预算\n漂移检测 · LLM 战略顾问(≤8次)", C_REF)

# 观测层（v1.0 新增，高亮）
box(4.0, 5.0, 7.1, 0.95, "Telemetry 观测层 (telemetry/) — v1.0 新增",
    "每步决策来源 · sim_calls · 漂移 · 步数  →  JSONL 落盘（喂回分析）", C_TELE)

# 不变量条
box(0.4, 5.0, 3.0, 0.95, "v1.0 核心不变量",
    "simulate 深拷输入\n调用方 state 永不污染", "#ffe0e0")

# 数据流：环境→感知→世界模型
arrow(1.9, 1.35, 4.0, 1.05)
arrow(7.0, 1.35, 7.9, 1.6)
arrow(10.9, 1.3, 11.8, 1.6)

# 上行：感知→规划→反思
arrow(5.5, 1.6, 5.5, 3.0)
arrow(5.5, 3.55, 9.5, 3.55)

# 下行：规划→动作
arrow(5.5, 3.0, 5.5, 2.4, "动作", "#c0392b")
arrow(5.5, 2.4, 1.9, 1.35)

# Kaggle 烟囱口 ↔ 规划
arrow(7.5, 8.15, 7.5, 6.95, "choose_action")
arrow(8.5, 6.0, 11.0, 4.15, "漂移")
arrow(11.0, 3.55, 9.5, 5.0)

# 观测层汇入
arrow(7.5, 5.0, 7.5, 5.95, "record", "#7f8c8d")

# 图例
legend = [
    ("★ 性能修复：首层短路 sim_calls 7256→4", "#c0392b"),
    ("✓ 正确性修复：_deepcopy_state 隔离输入", "#c0392b"),
    ("● 新增：Telemetry JSONL（跑分数据喂回）", "#2c3e50"),
]
for i, (txt, c) in enumerate(legend):
    ax.text(0.5, 8.55 - i*0.35, txt, fontsize=8.5, color=c, fontweight="bold")

plt.tight_layout()
plt.savefig("/data/workspace/lingjing_v10/lingjing_v10_architecture.png", dpi=150,
            bbox_inches="tight", facecolor="white")
print("saved: lingjing_v10_architecture.png")
