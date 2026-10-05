"""Lingjing engine V14.2 regression entry point.

Run with ``python -m lingjing_solo.regress``. Each suite executes in a
fresh interpreter while importing the installed ``lingjing_solo`` package.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SUITES = [
    ("契约测试   test_contracts.py  (C1-C9)", "lingjing_solo.tests.test_contracts"),
    ("端到端     test_e2e.py", "lingjing_solo.tests.test_e2e"),
    ("接口自检   test_scenario_contract.py", "lingjing_solo.tests.test_scenario_contract"),
    ("历史覆盖   test_v14.py", "lingjing_solo.tests.test_v14"),
]


def run_module(module: str) -> bool:
    """Run one suite in an isolated interpreter."""
    print("\n" + "─" * 60)
    print(f"▶ {module.rsplit('.', 1)[-1]}.py")
    print("─" * 60)
    result = subprocess.run([sys.executable, "-m", module], cwd=ROOT.parents[1])
    return result.returncode == 0


def main() -> int:
    results = [(name, run_module(module)) for name, module in SUITES]
    print("\n" + "=" * 60)
    print("回归汇总")
    print("=" * 60)
    for name, ok in results:
        print(f"  {'✅ PASS' if ok else '❌ FAIL'}  {name}")
    total_tests = sum(_count_tests(module) for _, module in SUITES)
    pass_count = sum(ok for _, ok in results)
    print(f"\n总计: {pass_count}/{len(results)} 套件通过（约 {total_tests} 项断言）")
    return 0 if pass_count == len(results) else 1


def _count_tests(module: str) -> int:
    """Count top-level test functions for the summary."""
    path = ROOT.joinpath(*module.split('.')[2:]).with_suffix('.py')
    try:
        return sum(1 for line in path.read_text().splitlines() if line.startswith("def test_"))
    except OSError:
        return 0


if __name__ == "__main__":
    sys.exit(main())
