"""绘制 Lingjing-Solo 五层架构图 (ASCII + PNG 双版本)

PNG 用 matplotlib；若环境无中文字体则自动退化。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

try:
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    HAS_CN = True
except Exception:
    HAS_CN = False


def draw():
    fig, ax = plt.subplots(figsize=(12, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.axis("off")

    title = "Lingjing-Solo · 单 Agent 世界模型场架构"
    ax.text(5, 15.4, title, ha="center", va="center", fontsize=15, fontweight="bold")

    layers = [
        ("Kaggle 烟囱口", "is_done() / choose_action()\nAgent 编排 (agent.py)", "#37474F", 13.6, 14.7),
        ("Layer 4 · 反思触发器", "循环检测 | 规则冲突 | 步数预算告急\n→ 打包 Φ 摘要触发 LLM 反思", "#AB47BC", 12.0, 13.1),
        ("Layer 3 · 规划执行层", "轻量 BFS/A* (无 LLM)\nLLM 战略顾问 (每关硬预算, 默认 8 次)", "#EC407A", 10.2, 11.3),
        ("Layer 2 · 探索与假设引擎", "信息增益评分 | 规则归纳 | 目标推断\n「单位步数信息增益最大」对冲 RHAE 平方惩罚", "#FF7043", 8.4, 9.5),
        ("Layer 1 · 世界模型场 Φ", "网格状态 | 转移表 (s,a)→s'\n规则假设集(置信度) | 目标假设 | 循环检测集", "#42A5F5", 6.2, 7.6),
        ("Layer 0 · 感知编码层", "帧差分 Δ(泡壁) | 连通域分割 | CNN/直方图特征\n64×64 → 128 维 (不开 LLM)", "#66BB6A", 4.0, 5.4),
        ("环境 · ARC-AGI-3", "隐藏/回合制/零指令 · 64×64·16 色\n动作: 5方向 + UNDO + 可选 MOUSE", "#8D6E63", 2.0, 3.2),
    ]

    # 主干：环境 → L0 → L1 → L2 → L3 → L4 → 烟囱口（自下而上）
    for name, desc, color, y0, y1 in layers:
        box = FancyBboxPatch((1.0, y0), 8.0, y1 - y0, boxstyle="round,pad=0.02,rounding_size=0.15",
                             linewidth=1.5, edgecolor="white", facecolor=color, alpha=0.92)
        ax.add_patch(box)
        ax.text(5, (y0 + y1) / 2 + 0.25, name, ha="center", va="center",
                fontsize=11, color="white", fontweight="bold")
        ax.text(5, (y0 + y1) / 2 - 0.25, desc, ha="center", va="center", fontsize=8.5, color="white")

    # 决策流箭头（自下而上 = 数据/感知上行；自上而下 = 动作下行）
    for i in range(len(layers) - 1):
        y_top = layers[i][4]      # 上层底部 ≈ 下层顶部，近似用边界
        y_bot = layers[i + 1][3]
        # 上行感知流（蓝）
        ax.annotate("", xy=(3.0, y_bot + 0.05), xytext=(3.0, y_top - 0.05),
                    arrowprops=dict(arrowstyle="-|>", color="#1565C0", lw=2.0))
        # 下行动作流（红）
        ax.annotate("", xy=(7.0, y_top - 0.05), xytext=(7.0, y_bot + 0.05),
                    arrowprops=dict(arrowstyle="-|>", color="#C62828", lw=2.0))

    ax.text(3.0, 15.0, "感知↑", ha="center", fontsize=8, color="#1565C0")
    ax.text(7.0, 15.0, "动作↓", ha="center", fontsize=8, color="#C62828")

    # 右侧：Φ 场内部结构标注
    ax.text(5, 0.7, "Φ 场核心 invariants:  转移因果链 · 规则置信度 · ROI 局部高精度 · 版本化状态哈希",
            ha="center", fontsize=8, color="#1565C0", style="italic")

    plt.tight_layout()
    plt.savefig("/data/workspace/lingjing_solo_architecture.png", dpi=150, bbox_inches="tight")
    print("saved: /data/workspace/lingjing_solo_architecture.png")


if __name__ == "__main__":
    draw()
