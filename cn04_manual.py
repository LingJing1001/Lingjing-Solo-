# -*- coding: utf-8 -*-
"""cn04 手动求解器 — 交互式玩游戏，记录点击序列。

用法:
    python cn04_manual.py <level>

示例:
    python cn04_manual.py 0  # 玩 L1
    python cn04_manual.py 5  # 玩 L6

操作:
    1. 点击 sprite 选中它
    2. 按方向键移动
    3. 按 R 旋转
    4. 按 Q 退出并保存解法
"""
import os, sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KAGGLE = ROOT / "integrations" / "ARC-AGI-3-Kaggle-Starter"
sys.path.insert(0, str(KAGGLE))
os.environ.setdefault("LINGJING_SRC", str(ROOT / "lingjing_solo"))
sys.path.insert(0, str(ROOT))
VENDOR = KAGGLE / "vendor" / "ARC-AGI-3-Agents"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))

import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction, ActionInput
import numpy as np


def make_env(gid="cn04"):
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    return arc.make(gid)


def render_frame(g):
    """渲染当前帧。"""
    frame = g.current_level.get_sprites()
    grid = np.full((20, 20), -1, dtype=int)
    for s in frame:
        if not s.is_visible:
            continue
        rendered = s.render()
        if rendered is None:
            continue
        for y in range(rendered.shape[0]):
            for x in range(rendered.shape[1]):
                if rendered[y, x] >= 0:
                    gx, gy = s.x + x, s.y + y
                    if 0 <= gx < 20 and 0 <= gy < 20:
                        grid[gy, gx] = rendered[y, x]
    return grid


def print_grid(grid, selected=None):
    """打印网格。"""
    print()
    for y in range(20):
        row = []
        for x in range(20):
            if selected and selected.x <= x < selected.x + selected.width and selected.y <= y < selected.y + selected.height:
                row.append("*")
            elif grid[y, x] == -1:
                row.append(".")
            elif grid[y, x] == 0:
                row.append("0")
            elif grid[y, x] == 8:
                row.append("8")
            elif grid[y, x] == 13:
                row.append("3")
            else:
                row.append(str(grid[y, x]))
        print("  " + " ".join(row))
    print()


def find_click_pos(g, sprite):
    """找能选中 sprite 的显示坐标。"""
    rendered = sprite.render()
    if rendered is None:
        return None
    non_tp = np.argwhere(rendered >= 0)
    if len(non_tp) == 0:
        return None
    y, x = non_tp[0]
    gx, gy = int(sprite.x + x), int(sprite.y + y)
    g2d = {}
    for dy in range(64):
        for dx in range(64):
            res = g.camera.display_to_grid(dx, dy)
            if res is not None and res not in g2d:
                g2d[res] = (dx, dy)
    if (gx, gy) in g2d:
        return g2d[(gx, gy)]
    return gx, gy


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("level", type=int, default=0, help="关卡编号 0-5")
    p.add_argument("--out", default=None, help="输出文件路径")
    args = p.parse_args()

    env = make_env("cn04")
    frame = env.reset()
    g = env._game

    # 跳到目标关
    for _ in range(args.level):
        g.next_level()

    print(f"\n=== cn04 L{args.level+1} ===")
    print(f"MaxSteps: {g.ojcsxidcz}")
    print(f"操作说明:")
    print(f"  1. 输入 'click <sprite_id>' 点击选中 sprite (0 或 1)")
    print(f"  2. 输入 'move <dir>' 移动: up/down/left/right")
    print(f"  3. 输入 'rotate' 旋转")
    print(f"  4. 输入 'win' 检查是否过关")
    print(f"  5. 输入 'save' 保存解法")
    print(f"  6. 输入 'quit' 退出")
    print()

    actions = []
    step_count = 0

    while True:
        # 渲染
        grid = render_frame(g)
        selected = g.xseexqzst
        print_grid(grid, selected)

        print(f"Steps: {step_count}/{g.ojcsxidcz}")
        if selected:
            print(f"Selected: {selected.name} at ({selected.x},{selected.y}) rot={selected.rotation}")
        else:
            print(f"Selected: None")

        # 检查 win
        if g.sjwqloivve():
            print("\n*** WIN! ***")
            print(f"Total steps: {step_count}")
            print(f"Actions: {actions}")
            break

        # 输入
        cmd = input("> ").strip().lower()

        if cmd == "quit":
            break
        elif cmd == "win":
            print(f"win: {g.sjwqloivve()}")
            continue
        elif cmd == "save":
            out = args.out or f"cn04_l{args.level+1}_plan.json"
            with open(out, "w", encoding="utf-8") as f:
                json.dump({"level": args.level, "actions": actions}, f, ensure_ascii=False, indent=2)
            print(f"Saved to {out}")
            continue
        elif cmd.startswith("click "):
            try:
                sid = int(cmd.split()[1])
                sprites = [s for s in g.current_level.get_sprites() if s.is_visible]
                if 0 <= sid < len(sprites):
                    s = sprites[sid]
                    click = find_click_pos(g, s)
                    if click:
                        action = ActionInput(id=GameAction.ACTION6, data={"x": click[0], "y": click[1]}, reasoning=None)
                        g.perform_action(action)
                        actions.append(("click", click[0], click[1]))
                        step_count += 1
                        print(f"Clicked sprite {sid} at {click}")
                    else:
                        print(f"Cannot find click position for sprite {sid}")
                else:
                    print(f"Invalid sprite id: {sid}")
            except Exception as e:
                print(f"Error: {e}")
            continue
        elif cmd == "rotate":
            if g.xseexqzst:
                action = ActionInput(id=GameAction.ACTION5, data={}, reasoning=None)
                g.perform_action(action)
                actions.append(("rotate",))
                step_count += 1
                print(f"Rotated to {g.xseexqzst.rotation}")
            else:
                print("No sprite selected")
            continue
        elif cmd.startswith("move "):
            try:
                dir = cmd.split()[1]
                if dir in ("up", "down", "left", "right"):
                    if g.xseexqzst:
                        act_id = {"up": 1, "down": 2, "left": 3, "right": 4}[dir]
                        action = ActionInput(id=GameAction.from_id(act_id), data={}, reasoning=None)
                        g.perform_action(action)
                        actions.append(("move", act_id))
                        step_count += 1
                        print(f"Moved {dir} to ({g.xseexqzst.x},{g.xseexqzst.y})")
                    else:
                        print("No sprite selected")
                else:
                    print(f"Invalid direction: {dir}")
            except Exception as e:
                print(f"Error: {e}")
            continue
        elif cmd == "reset":
            env.reset()
            for _ in range(args.level):
                g.next_level()
            actions = []
            step_count = 0
            print("Reset")
            continue
        else:
            print(f"Unknown command: {cmd}")
            print(f"  可用命令: click 0/1, move up/down/left/right, rotate, win, save, quit, reset")


if __name__ == "__main__":
    main()
