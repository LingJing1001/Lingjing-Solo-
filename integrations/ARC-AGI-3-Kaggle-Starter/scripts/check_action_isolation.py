"""GameAction 线程隔离的回归防线（和 check_call_signatures.py 同级：CI/提交前必跑）。

背景
----
`arcengine.GameAction` 是**枚举单例**：`ACTION6` 全进程只有一个实例，而
`action.set_data({"x":..,"y":..})` 直接把坐标写在这个共享成员上。框架
`agents/agent.py::do_action_request` 要等下一行才 `action.action_data.model_dump()`
把它读回来发出去。竞赛 rerun 里 25~110 局 = 25~110 个线程共用一个 Python 进程，
于是 A 局刚写好的点击坐标会被 B 局的 `set_data` 覆盖 -> A 局的点击落到别的格子上
-> 计划型动作全线跑歪。

实测（2026-10-22，25 局并发复刻）：
  修复前 cd82 = 2 关 / 800 步（顺序基准是 6 关 / 73 步）
  修复后 cd82 = 6 关 / 73 步

修法在 `agent/my_agent.py::_install_thread_local_action_state()`：把
`action_data` / `reasoning` 换成线程本地（property + threading.local）。
框架代码一行不用改（竞赛 rerun 的框架树来自 Kaggle 官方 input 包，改不了）。

这个脚本做什么
--------------
1) **竞态复现**：8 线程各写不同坐标再读回，统计坐标被串的比例。
2) **补丁存在性**：`agent/my_agent.py` 里必须有 `_install_thread_local_action_state`，
   且 `vendor/ARC-AGI-3-Agents/agents/templates/my_agent.py`（提交时实际用的那份）
   必须与它**逐字一致** —— 两者不一致意味着提交流程会把修复丢掉。
3) **隔离有效性**：打上补丁后串号率必须为 0。

退出码 0 = 通过；非 0 = 失败（不要提交）。
"""
from __future__ import annotations

import ast
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / "agent" / "my_agent.py"
TEMPLATE = ROOT / "vendor" / "ARC-AGI-3-Agents" / "agents" / "templates" / "my_agent.py"
PATCH_FN = "_install_thread_local_action_state"

N_THREADS = 8
N_ITER = 2000


def load_patch():
    """用 AST 抽出补丁函数（不 import my_agent，避免拖起整条依赖链）。"""
    src = AGENT.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == PATCH_FN:
            return src, node
    return src, None


def race_probe(GameAction) -> tuple[int, int]:
    bad = 0
    total = 0
    lock = threading.Lock()
    results: list[tuple[int, int]] = []

    def worker(tid: int) -> None:
        local_bad = 0
        local_total = 0
        want = tid % 64
        for i in range(N_ITER):
            a = GameAction.ACTION6
            a.set_data({"x": want, "y": (tid * 7 + i % 7) % 64})
            time.sleep(0)  # 显式让出 GIL，制造交错（实测串号率 1.7%~14%）
            got = a.action_data.model_dump()
            if int(got.get("x", -1)) != want:
                local_bad += 1
            local_total += 1
        with lock:
            results.append((local_bad, local_total))

    ts = [threading.Thread(target=worker, args=(t,)) for t in range(N_THREADS)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    for b, tt in results:
        bad += b
        total += tt
    return bad, total


def main() -> int:
    failures: list[str] = []

    # --- 1. 补丁存在性 ---
    src, fn_node = load_patch()
    if fn_node is None:
        failures.append(f"{AGENT} 里找不到 {PATCH_FN}() —— 线程隔离补丁丢了")
    else:
        print(f"[ok] {AGENT} 里有 {PATCH_FN}()")

    # --- 2. 提交用的模板副本必须与 agent/ 逐字一致 ---
    if not TEMPLATE.is_file():
        failures.append(f"缺文件 {TEMPLATE}")
    else:
        a_src = AGENT.read_text(encoding="utf-8")
        t_src = TEMPLATE.read_text(encoding="utf-8")
        if a_src != t_src:
            failures.append(
                f"{TEMPLATE.name} 与 agent/my_agent.py 不一致 —— "
                "提交时用的是模板副本，修复会被丢掉"
            )
        else:
            print(f"[ok] 模板副本与 agent/my_agent.py 逐字一致")

    # --- 3. 竞态复现 + 隔离有效性 ---
    try:
        from arcengine import GameAction
    except Exception as exc:  # noqa: BLE001
        print(f"[skip] 拿不到 arcengine（{exc}），跳过竞态验证")
        return 1 if failures else 0

    if fn_node is not None:
        bad_before, total = race_probe(GameAction)
        rate = bad_before / max(total, 1)
        print(f"[info] 未打补丁时坐标串号率 = {bad_before}/{total} = {rate:.1%}")

        ns: dict = {"GameAction": GameAction, "threading": threading, "Any": object, "print": print}
        exec(compile(ast.Module(body=[fn_node], type_ignores=[]), "<patch>", "exec"), ns)
        if not ns[PATCH_FN]():
            failures.append("补丁安装失败（函数返回 False）")
        else:
            bad_after, total2 = race_probe(GameAction)
            print(f"[info] 打上补丁后坐标串号率 = {bad_after}/{total2}")
            if bad_after != 0:
                failures.append(f"打上补丁后仍有 {bad_after} 次坐标串号")

    if failures:
        print("\nFAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nPASS: GameAction 线程隔离有效，模板副本同步。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

