"""
灵境引擎 V14.2 — 统一回归入口
==================================
分层运行全部测试套件，退出码供 CI 使用。

V14.2 变更：
- 新增 tests/test_v14.py（pytest 可选，可裸跑），补齐历史覆盖
- 契约层新增 C7（Agent 写场契约）、C8（shape fail-fast）、C9（配置 dataclass）
- 一律不依赖 pytest：纯 unittest 兼容 + 自运行入口
"""
from __future__ import annotations
import sys, os, subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

SUITES = [
    ("契约测试   test_contracts.py  (C1-C9)",  "tests/test_contracts.py"),
    ("端到端     test_e2e.py",                "tests/test_e2e.py"),
    ("接口自检   test_scenario_contract.py",   "tests/test_scenario_contract.py"),
    ("历史覆盖   test_v14.py",                 "tests/test_v14.py"),
]


def run_module(path: str) -> bool:
    """以独立进程运行一个测试模块（利用其 __main__ 自运行入口）。"""
    full = os.path.join(ROOT, path)
    print(f"\n────────────────────────────────────────────────────────────")
    print(f"▶ {os.path.basename(path)}")
    print(f"────────────────────────────────────────────────────────────")
    result = subprocess.run([sys.executable, full], cwd=ROOT)
    return result.returncode == 0


def main() -> int:
    results = []
    for name, path in SUITES:
        ok = run_module(path)
        results.append((name, ok))

    print("\n" + "=" * 60)
    print("回归汇总")
    print("=" * 60)
    pass_count = 0
    for name, ok in results:
        flag = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {flag}  {name}")
        pass_count += ok

    total_tests = sum(_count_tests(p) for _, p in SUITES)
    print(f"\n总计: {pass_count}/{len(results)} 套件通过 "
          f"（约 {total_tests} 项断言）")
    return 0 if pass_count == len(results) else 1


def _count_tests(path: str) -> int:
    """粗略统计套件内 test_ 函数数量，用于汇总展示。"""
    full = os.path.join(ROOT, path)
    try:
        with open(full, encoding="utf-8") as fh:
            return sum(1 for line in fh if line.startswith("def test_"))
    except OSError:
        return 0


if __name__ == "__main__":
    sys.exit(main())
