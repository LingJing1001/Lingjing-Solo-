"""
debugD_mutate.py — 验证假设：多次 simulate 调用间，传入的 state 是否被原地修改。

推理：FakeLLM 的 step 用 s['objects'] = dict(state['objects']) —— 这是「外层 dict 拷贝」，
但内部的每个 obj（{"x":..,"y":..,"role":..}）仍是**同一引用**（浅拷贝）。
如果 step 里对 obj 做 o['x']=... 原地修改，那原始 state 就会被污染。
同一份 state 连续调 right 再调 left：left 看到的 avatar 已不是 (1,1) 而是 (2,1)，
于是 left 算出 (2-1,1)=(1,1) —— 表现为「left 方向失效」。

这是「浅拷贝 + 原地修改」的经典 bug，正好解释「left/up 失效、right/down 正确」的 asymmetry。
"""
import sys, os, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from lingjing_solo.world_model.codegen import FakeLLM


SRC = FakeLLM().generate("ignored")


def make_step():
    _RUNTIME_SAFE = {
        "dict": dict, "list": list, "tuple": tuple, "set": set,
        "len": len, "isinstance": isinstance,
        "range": range, "enumerate": enumerate, "zip": zip,
        "min": min, "max": max, "sum": sum, "abs": abs, "all": any,
        "True": True, "False": False, "None": None,
    }
    ns = {"__builtins__": {}}
    ns.update(_RUNTIME_SAFE)
    ns.update({"UP": "up", "DOWN": "down", "LEFT": "left", "RIGHT": "right"})
    exec(compile(SRC, "<x>", "exec"), ns)
    return ns["step"]


def fresh():
    return {
        "objects": {
            "a": {"x": 1, "y": 1, "role": "avatar"},
            "b": {"x": 2, "y": 1, "role": "box"},
        },
        "avatar_id": "a",
        "extras": {},
    }


def main():
    step = make_step()

    print("=" * 60)
    print("实验 A：同一份 state 连续调用 right → left")
    print("=" * 60)
    s = fresh()
    print(f"  初始: a={s['objects']['a']}")
    out1 = step(s, "right")
    print(f"  right 后: a={s['objects']['a']}  (state 是否被原地改?)")
    out2 = step(s, "left")
    print(f"  left 后:  a={s['objects']['a']}")
    print(f"  left 返回值 avatar = ({out2['objects']['a']['x']},{out2['objects']['a']['y']})")
    print(f"  若 state 被污染：left 基于 (2,1) 算 → 返回 (1,1) ❌")
    print(f"  若未污染：     left 基于 (1,1) 算 → 返回 (0,1) ✓")

    print()
    print("=" * 60)
    print("实验 B：每次传全新拷贝（对照组）")
    print("=" * 60)
    s2 = fresh()
    out_r = step(copy.deepcopy(s2), "right")
    print(f"  right 返回 avatar = ({out_r['objects']['a']['x']},{out_r['objects']['a']['y']})")
    out_l = step(copy.deepcopy(s2), "left")
    print(f"  left  返回 avatar = ({out_l['objects']['a']['x']},{out_l['objects']['a']['y']})  "
          "← 应为 (0,1)")

    print()
    print("=" * 60)
    print("实验 C：检查 step 是否拷贝了 obj（或原地改）")
    print("=" * 60)
    s3 = fresh()
    obj_a_before = s3["objects"]["a"]
    step(s3, "right")
    obj_a_after = s3["objects"]["a"]
    print(f"  同一 dict 对象? {obj_a_before is obj_a_after}")
    print(f"  a['x'] 被改? {obj_a_before['x']} (原应为 1)")
    if obj_a_before["x"] != 1:
        print("  ❯❯❯ 根因确认：step 原地修改了输入的 obj dict（浅拷贝 bug）")
    print("=" * 60)


if __name__ == "__main__":
    main()
