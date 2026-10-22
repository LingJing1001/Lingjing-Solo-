"""把 `scripts/action_isolation_patch.py` 的标准实现同步进 agent（自包含要求）。

为什么需要它
------------
竞赛 rerun 的框架树来自 Kaggle 官方 input 包（改不了），notebook 也只会把
`my_agent.py` 拷进 `agents/templates/`。所以**提交用的 agent 必须自包含**：
标准实现得有一份内联拷贝留在 `agent/my_agent.py` 里。

但"两份拷贝"必然漂移。本脚本把标准件作为**唯一真源**，机械地把它写进：
  1. `agent/my_agent.py`
  2. `vendor/ARC-AGI-3-Agents/agents/templates/my_agent.py`（提交时实际用的那份）
并且 `scripts/check_action_isolation.py` 会校验三者一致。

用法:
    python scripts/sync_action_isolation.py         # 写入
    python scripts/sync_action_isolation.py --check # 只校验不写入
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCH_SRC = ROOT / "scripts" / "action_isolation_patch.py"
TARGETS = [
    ROOT / "agent" / "my_agent.py",
    ROOT / "vendor" / "ARC-AGI-3-Agents" / "agents" / "templates" / "my_agent.py",
]

CANON_NAME = "install"
INLINE_NAME = "_install_thread_local_action_state"


def canonical_source() -> str:
    """从标准件里取出 install() 的源码（改名为内联名）。"""
    src = PATCH_SRC.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == CANON_NAME:
            text = ast.get_source_segment(src, node)
            if not text:
                raise SystemExit("取不出 install() 的源码")
            # 改名 + 去掉装饰器痕迹（本来就没有）
            lines = text.splitlines()
            for i, line in enumerate(lines):
                if line.startswith("def install("):
                    lines[i] = line.replace("def install(", f"def {INLINE_NAME}(", 1)
                    break
            else:
                raise SystemExit("install() 定义没找到")
            return "\n".join(lines)
    raise SystemExit(f"{PATCH_SRC} 里找不到 {CANON_NAME}()")


def current_inline(target: Path) -> str | None:
    src = target.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == INLINE_NAME:
            return ast.get_source_segment(src, node)
    return None


def replace_inline(target: Path, new_src: str) -> bool:
    src = target.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == INLINE_NAME:
            old = ast.get_source_segment(src, node)
            if old is None:
                raise SystemExit(f"{target}: 取不出 {INLINE_NAME}() 的原文")
            if old == new_src:
                return False
            target.write_text(src.replace(old, new_src, 1), encoding="utf-8")
            return True
    raise SystemExit(f"{target} 里找不到 {INLINE_NAME}()")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只校验不写入")
    args = ap.parse_args()

    canon = canonical_source()
    ok = True
    for t in TARGETS:
        if not t.is_file():
            print(f"[skip] {t} 不存在")
            continue
        cur = current_inline(t)
        if cur == canon:
            print(f"[ok] {t.name}: 与标准件一致")
            continue
        if args.check:
            print(f"[FAIL] {t.name}: 与标准件不一致（跑 sync_action_isolation.py 修）")
            ok = False
            continue
        replace_inline(t, canon)
        print(f"[write] {t.name}: 已同步标准件")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
