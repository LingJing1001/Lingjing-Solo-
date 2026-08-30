"""
run_all_v10.py — v1.0 全量回归（根因修复后）

覆盖：
  - test_v10_state_isolation  (新增：simulate 输入隔离回归，锁定浅拷贝根因)
  - test_v10_performance       (缓存/浅拷贝/自适应深度/首层短路/兼容性)
  - test_v10_telemetry_agent    (观测层 + Agent 集成)
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

MODULES = [
    "tests.test_v10_state_isolation",
    "tests.test_v10_performance",
    "tests.test_v10_telemetry_agent",
]

loader = unittest.TestLoader()
suite = unittest.TestSuite()
loaded = 0
for mod in MODULES:
    try:
        suite.addTests(loader.loadTestsFromName(mod))
        loaded += 1
    except Exception as e:
        print(f"[WARN] 无法加载 {mod}: {e}", file=sys.stderr)

print(f"[info] 加载 {loaded}/{len(MODULES)} 个测试模块\n", file=sys.stderr)

runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
