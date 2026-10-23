"""v28 notebook 的高保真验证 —— 不是静态查字符串，是**真跑一遍 cell 8 的逻辑**。

为什么需要这个脚本
------------------
v28 的核心改动之一是把 vLLM 的 admit limit (`--max-num-seqs`) 从 8 抬到 16。
cell 8 里写死 8 的地方有**三处**：

    A. `_vllm_extra` 注入分支 —— 上游没有 --enforce-eager 时走这里
    B. `for` 循环的替换值      —— 上游自带 --max-num-seqs 时走这里
    C. `else` fallback         —— 上游完全没提时走这里

实测上游 `setup_commands.json`（10076 字符）里 `--max-num-seqs` 出现 **0 次**、
`--enforce-eager` 出现 **0 次** ⇒ 运行时恒走 A 分支 ⇒ **A 才是真正生效的那一处**。
只改 B/C 的话，v28 的并发提升会在 vLLM 的 admission gate 上排队，100% 落空。

而 `patch_v28.py` 里那个手写的精确锚点只改了 B/C（A 的字面量被拆成两行、
换了一种引号组合），所以「锚点命中 0 次」这个报错其实掩盖了更深的问题。

本脚本的做法：把生成出来的 notebook 里 cell 8 的 admit-limit 逻辑**原样切出来**，
喂**真实的上游 setup_commands.json** 执行一遍，看最终命令里的 `--max-num-seqs`
到底是几。同时校验：

    1. 所有 code cell 语法可编译（notebook 顶层 await 的误报用 v27 基线对照判定）
    2. 三处写死值确实都成了 16
    3. 运行期守卫（LINGJING_MAX_NUM_SEQS / LINGJING_MAX_NUM_SEQS_BAD）在位
    4. cell 8 里 base64 出来的 solver.py 与本地文件逐字节一致、sha 对得上

用法（starter venv python）：
    python verify_v28.py            # 失败退出码 1
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V28 = HERE / "push_v28" / "lingjing-duck-harness-v28.ipynb"
V27 = HERE / "push_v27" / "lingjing-duck-harness-v27.ipynb"
SETUP = ROOT / "integrations" / "duck-kaggle" / "_verify_bundle" / "setup_commands.json"
SOLVER = ROOT / "integrations" / "duck-harness" / "ARC3-Inference" / "inference" / "framework" / "solver.py"
EXPECT_SEQS = 16

fail: list[str] = []


def syntax_errors(cells: list) -> dict:
    out = {}
    for idx, c in enumerate(cells):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        if src.lstrip().startswith("%") or src.lstrip().startswith("!"):
            continue
        try:
            compile(src, f"cell{idx}", "exec")
        except SyntaxError as e:
            out[idx] = str(e)
    return out


def main() -> int:
    if not V28.is_file():
        print(f"[FAIL] 缺 {V28} —— 先跑 patch_v28.py")
        return 1

    nb = json.loads(V28.read_text(encoding="utf-8"))
    cells = nb["cells"]
    print(f"[1] notebook {V28.name}  {V28.stat().st_size} bytes, {len(cells)} cells")

    base = syntax_errors(json.loads(V27.read_text(encoding="utf-8"))["cells"])
    for idx, c in enumerate(cells):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        if src.lstrip().startswith("%") or src.lstrip().startswith("!"):
            print(f"    cell{idx}: 跳过（magics）")
            continue
        errs = syntax_errors(cells).get(idx)
        if errs is None:
            print(f"    cell{idx}: 语法 OK ({len(src)} chars)")
        elif idx in base:
            print(f"    cell{idx}: 与 v27 同款顶层 await 误报（基线一致）-> {errs[:50]}")
        else:
            fail.append(f"cell{idx} 语法错误: {errs}")
            print(f"    cell{idx}: 语法错误 {errs}")

    cell8 = "".join(cells[8].get("source", []))

    # ---- 2) 真跑一遍 cell 8 的 admit-limit 逻辑 ----
    print("\n[2] 用真实上游 setup_commands.json 跑 cell 8 的 admit-limit 逻辑")
    lines = cell8.split("\n")
    start = next(i for i, ln in enumerate(lines) if ln.strip() == "_nl = chr(10)")
    # 取最后一个 _patched_cmds.append(_cmd)：前面那个在 continue 分支里
    end = max(i for i, ln in enumerate(lines) if ln.strip() == "_patched_cmds.append(_cmd)")
    block = "\n".join(lines[start:end + 1])
    print(f"    取片段行 {start + 1}..{end + 1}（{len(block)} chars）")

    cmds = json.loads(SETUP.read_text(encoding="utf-8"))
    ns = {"json": json, "os": __import__("os"), "sys": sys, "Path": Path,
          "_cmds": cmds, "_patched_cmds": [], "WHEELHOUSE": Path("/tmp/nonexistent")}
    try:
        exec(compile(block, "cell8-sim", "exec"), ns)
    except Exception as e:  # noqa: BLE001
        fail.append(f"cell8 仿真异常: {type(e).__name__}: {e}")
        print(f"    仿真异常: {type(e).__name__}: {e}")
    else:
        patched = ns["_patched_cmds"]
        print(f"    patched cmds = {len(patched)}")
        for i, c in enumerate(patched):
            if not isinstance(c, str):
                continue
            hits = re.findall(r"--max-num-seqs',\s*'(\d+)'", c)
            print(f"    cmd[{i}] --max-num-seqs = {hits}")
            if hits != [str(EXPECT_SEQS)]:
                fail.append(f"cmd[{i}] --max-num-seqs 期望 ['{EXPECT_SEQS}'] 实际 {hits}")
            if "--max-num-seqs" not in c:
                fail.append(f"cmd[{i}] 根本没有 --max-num-seqs")
            j = c.find("--max-num-seqs")
            if j >= 0:
                print("      片段:", repr(c[max(0, j - 60): j + 40]))

    # ---- 3) 三处写死值 + 运行期守卫 ----
    print("\n[3] 静态检查：三处写死值 + 运行期守卫")
    for m in re.finditer(r".*'--max-num-seqs'.*", cell8):
        print("   ", m.group(0).strip())
    guard = "LINGJING_MAX_NUM_SEQS" in cell8
    guard_assert = "LINGJING_MAX_NUM_SEQS_BAD" in cell8
    print(f"    守卫块: {guard} | 断言: {guard_assert}")
    if not guard:
        fail.append("缺运行期守卫 LINGJING_MAX_NUM_SEQS")
    if not guard_assert:
        fail.append("缺运行期守卫断言 LINGJING_MAX_NUM_SEQS_BAD")
    for ln in cell8.split("\n"):
        if "'--max-num-seqs'" in ln and "'8'" in ln:
            fail.append(f"仍有写死 8: {ln.strip()}")
            print("    [!!] 仍有写死 8:", ln.strip())

    # ---- 4) solver.py 覆盖 ----
    print("\n[4] solver.py base64 覆盖校验")
    m = re.search(r'_lj_b64\.b64decode\("([A-Za-z0-9+/=]+)"\)', cell8)
    if not m:
        fail.append("cell8 找不到 solver b64")
    else:
        blob = base64.b64decode(m.group(1))
        local = SOLVER.read_bytes()
        print(f"    解码长度 {len(blob)}  本地 solver.py {len(local)}")
        print(f"    解码 sha {hashlib.sha256(blob).hexdigest()[:16]}")
        print(f"    本地 sha {hashlib.sha256(local).hexdigest()[:16]}")
        if blob != local:
            fail.append("base64 解码结果与本地 solver.py 不一致")
        sm = re.search(r'_lj_sha\s*=\s*"([0-9a-f]+)"', cell8)
        if sm and sm.group(1) != hashlib.sha256(local).hexdigest():
            fail.append("cell8 内记录的 sha 与本地 solver.py 不一致")
        else:
            print("    cell8 记录的 sha 与本地一致 ✓")

    print("\n" + "=" * 60)
    if fail:
        print(f"[FAIL] {len(fail)} 项")
        for f in fail:
            print("  -", f)
        return 1
    print("[ok] v28 notebook 全部高保真验证通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
