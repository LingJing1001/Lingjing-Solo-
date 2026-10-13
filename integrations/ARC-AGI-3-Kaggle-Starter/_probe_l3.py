"""验证 wa30 L3 的精灵构成。"""
import sys, os, copy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ["OPERATION_MODE"] = "offline"
os.environ.setdefault("LINGJING_SRC", str(ROOT.parent / "lingjing_solo"))
sys.path.insert(0, str(ROOT.parent))

import arc_agi
from arc_agi import OperationMode

arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
env = arc.make("wa30")
g = env._game

# 走到 L3（跳过 L0-L2）
from arcengine import ActionInput as AI, GameAction as GA

# 用 _WA30_PLANS 跳到 L3
plans = {
    0: [1, 1, 5, 1, 1, 5, 4, 4, 4, 1, 5, 2, 3, 3, 5, 3, 3, 3, 3, 3, 1, 5, 4, 4, 4, 5],
    1: [2, 2, 2, 2, 2, 1, 1, 4, 4, 4, 4, 4, 4, 5, 2, 2, 3, 3, 3, 3, 2, 3, 3, 5,
        1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 5, 3, 4],
    2: [1, 1, 1, 1, 4, 5, 4, 4, 4, 5, 3, 3, 3, 3, 3, 3, 1, 4, 5, 4, 4, 4, 4, 4, 4,
        5, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 2, 4, 5, 4, 4, 4, 4, 4, 5, 3, 4, 3, 4,
        3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3,
        4],
}
for lv_idx in range(3):
    for aid in plans[lv_idx]:
        g.perform_action(AI(id=GA.from_id(aid), data={}), raw=True)

lv = g.current_level
print(f"=== L3 状态 ===")
print(f"level_index={g.level_index} state={g._state}")
print(f"budget={getattr(getattr(g, 'kuncbnslnm', None), 'dbdarsgrbj', None)}")

# 全部精灵
sprites = lv.get_sprites()
print(f"\n=== 精灵总数: {len(sprites)} ===")

# 按标签分组
from collections import Counter
tag_count = Counter()
tag_sprites = {}
for sp in sprites:
    tag = sp.tag if hasattr(sp, 'tag') else f"id={id(sp)}"
    tag_count[tag] += 1
    tag_sprites.setdefault(tag, []).append(sp)

# 已知标签
KNOWN = {
    "wbmdvjhthc": "玩家",
    "geezpjgiyd": "箱子",
    "fsjjayjoeg": "落区",
    "bnzklblgdk": "blocker",
    "zqxwgacnue": "locked?",
}
print("\n=== 精灵构成 ===")
for tag, count in tag_count.most_common():
    name = KNOWN.get(tag, f"未知({tag[:8]}...)")
    print(f"  {name}: {count}  tag={tag}")
    if name.startswith("未知"):
        for sp in tag_sprites[tag]:
            print(f"    pos=({sp.x},{sp.y}) collidable={sp.is_collidable} w={sp.width} h={sp.height}")

# 玩家
player = lv.get_sprites_by_tag("wbmdvjhthc")[0]
print(f"\n=== 玩家: ({player.x}, {player.y}, rot={player.rotation}) ===")

# 箱子
blocks = lv.get_sprites_by_tag("geezpjgiyd")
print(f"=== 箱子: {[(b.x, b.y) for b in blocks]} ===")

# 落区
zone = set()
for sp in lv.get_sprites_by_tag("fsjjayjoeg"):
    zone |= {(sp.x + i, sp.y + j) for i in range(sp.width) for j in range(sp.height)}
print(f"=== 落区: {len(zone)} 格 ===")

# 4 格对齐的落位
valid = sorted([(x, y) for x, y in zone if x % 4 == 0 and y % 4 == 0])
print(f"=== 4格对齐落位: {valid} ===")

# NPC 位置
print(f"\n=== NPC 分析 ===")
for tag, sps in tag_sprites.items():
    if tag not in KNOWN and not tag.startswith("wbmd") and not tag.startswith("gee") and not tag.startswith("fsj") and not tag.startswith("bnz"):
        print(f"  tag={tag}: {[(sp.x, sp.y) for sp in sps]}")
