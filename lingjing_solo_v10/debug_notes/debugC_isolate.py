"""
debugC_isolate.py — 二分隔离：剥离 WMP 归纳，直接测 FakeLLM 生成的 step()

目的：确认到底是「FakeLLM 代码本身有 bug」还是「WMP 链路（learn→compile→simulate）改写了行为」。
"""
import sys, os, ast
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from lingjing_solo.world_model.codegen import FakeLLM, _safe_compile


FAKE_SRC = FakeLLM().generate("ignored prompt")


def run(src, state, action):
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
    exec(compile(src, "<x>", "exec"), ns)
    return ns["step"](state, action)


BASE = {
    "objects": {"a": {"x": 1, "y": 1, "role": "avatar"},
                "b": {"x": 2, "y": 1, "role": "box"}},
    "avatar_id": "a",
    "extras": {},
}


def main():
    print("=" * 60)
    print("隔离测试：直接 exec FakeLLM 源码，不经过 WMP")
    print("=" * 60)
    print("\n--- 安全编译检查 ---")
    try:
        _safe_compile(FAKE_SRC)
        print("  ✓ _safe_compile 通过")
    except Exception as e:
        print(f"  ❌ 编译失败: {e}")

    step = None
    try:
        step = run(FAKE_SRC, dict(objects={"a": {"x": 1, "y": 1, "role": "avatar"},
                                          "b": {"x": 2, "y": 1, "role": "box"}},
                                  avatar_id="a", extras={}), "right")
        print("  ✓ exec 成功，step 可调用")
    except Exception as e:
        print(f"  ❌ exec 失败: {e}")

    print("\n--- 裸代码各方向（avatar=(1,1), box=(2,1)）---")
    dirs = {"right": (1, 0), "left": (-1, 0), "down": (0, 1), "up": (0, -1)}
    failures = []
    for act, (dx, dy) in dirs.items():
        import copy
        s = copy.deepcopy(BASE)
        out = run(FAKE_SRC, s, act)
        av = out["objects"]["a"]
        exp = (1 + dx, 1 + dy)
        okk = (av["x"], av["y"]) == exp
        print(f"  {act:6s}: avatar=({av['x']},{av['y']}) expected={exp}  {'✓' if okk else '❌'}")
        if not okk:
            failures.append(act)

    print("\n--- 裸代码推箱 (right, avatar 与 box 相邻) ---")
    s = copy.deepcopy(BASE)
    out = run(FAKE_SRC, s, "right")
    print(f"  avatar=({out['objects']['a']['x']},{out['objects']['a']['y']})  "
          f"box=({out['objects']['b']['x']},{out['objects']['b']['y']})")

    print("\n" + "=" * 60)
    if failures:
        print(f"结论：裸代码在 {failures} 方向已失效 → bug 在 FakeLLM 源码本身")
    else:
        print("结论：裸代码全对 → bug 在 WMP 链路（learn/compile/simulate 改写）")
    print("=" * 60)


if __name__ == "__main__":
    main()
