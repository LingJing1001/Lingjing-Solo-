"""SolvedLevelMemory 跨局持久化的最小往返测试。

不依赖 LLM 端点：构造两条假通关记录 → save_to_dir → 用一个全新实例
load_from_dir → 验证 round-trip 一致 + merge 不覆盖已存在的关。
"""
import sys
import shutil
import tempfile
from pathlib import Path

# 把 arc_adaptor 摆到 sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arc_adaptor.duck.memory import SolvedLevelMemory, LevelRecord

TMP = Path(tempfile.mkdtemp(prefix="cross-game-test-"))
try:
    # ---- 1) 第一次跑：模拟通关 level 0/1/2，记三条规则 ----
    mem1 = SolvedLevelMemory(enabled=True)
    mem1.record(0, ["ACTION3", "ACTION3", "ACTION3", "ACTION2", "ACTION6",
                    "ACTION3"] * 4, reasoning="pushed crates east, lever south")
    mem1.levels[-1].rule = "Rule for level 0: push all crates two cells east, then step on the south lever"
    mem1.record(1, ["ACTION3", "ACTION4", "ACTION6"] * 6, reasoning="collected keys in row 2 first")
    mem1.levels[-1].rule = "Rule for level 1: clear row 2 keys before touching the door"
    mem1.record(2, ["ACTION5", "ACTION2"] * 12, reasoning="")
    mem1.levels[-1].rule = ""  # 模型没答 rule 的关：not stated

    saved = mem1.save_to_dir(str(TMP), "ar25")
    assert saved, "save_to_dir should return True when levels present"
    disk_path = TMP / "ar25.json"
    assert disk_path.exists(), f"disk file missing at {disk_path}"
    print(f"[1] saved 3 levels to {disk_path} ({disk_path.stat().st_size} bytes)")

    # ---- 2) 模拟第二次跑同一局：全新实例，load_from_dir 读回 ----
    mem2 = SolvedLevelMemory(enabled=True)
    loaded = mem2.load_from_dir(str(TMP), "ar25")
    assert loaded == 3, f"expected 3 records loaded, got {loaded}"
    assert len(mem2.levels) == 3, f"mem2 has {len(mem2.levels)} levels, expected 3"
    # 比对每条记录字段
    for orig, got in zip(mem1.levels, mem2.levels):
        assert orig.level == got.level, f"level mismatch {orig.level} vs {got.level}"
        assert orig.rule == got.rule, f"rule mismatch at L{orig.level}"
        assert orig.n_actions == got.n_actions
        assert orig.actions == got.actions, f"actions tail mismatch at L{orig.level}"
        assert orig.reasoning == got.reasoning
    print(f"[2] round-trip OK: 3 records loaded, all fields match")
    print(f"    L0 rule: {mem2.levels[0].rule[:60]}...")
    print(f"    L2 rule: '{mem2.levels[2].rule}' (should be empty)")

    # ---- 3) merge 语义：mem2 已有 L1，再 load 同样的 L1 不应覆盖 ----
    mem2.levels[1].rule = "OVERWRITTEN BY IN-GAME LEARNING"
    loaded_again = mem2.load_from_dir(str(TMP), "ar25")
    assert loaded_again == 0, f"merge should add 0 (all levels already present), got {loaded_again}"
    assert mem2.levels[1].rule == "OVERWRITTEN BY IN-GAME LEARNING", "in-game rule should not be clobbered by disk"
    print(f"[3] merge semantics OK: in-game learning wins over disk record")

    # ---- 4) 不存在的 game_id 应返回 0，不报错 ----
    mem3 = SolvedLevelMemory(enabled=True)
    n = mem3.load_from_dir(str(TMP), "nonexistent-game-id")
    assert n == 0
    print(f"[4] missing game_id returns 0, no exception")

    # ---- 5) bad JSON should not crash ----
    bad = TMP / "corrupt.json"
    bad.write_text("{ this is not json", encoding="utf-8")
    n = mem3.load_from_dir(str(TMP), "corrupt")
    assert n == 0, f"corrupt file should return 0, got {n}"
    print(f"[5] corrupt JSON returns 0, no exception")

    # ---- 6) DuckBridge 挂钩子：用一个 fake driver 验证 _save/_load 接口 ----
    class FakeMemory:
        enabled = True
        reasoning_chars = 900
        def __init__(self):
            self.levels = []
        def record(self, level, actions, reasoning=""):
            self.levels.append(LevelRecord(level=level, n_actions=len(actions),
                                           actions=list(actions), reasoning=reasoning))
        def load_from_dir(self, dir_, gid):
            m = SolvedLevelMemory(enabled=True)
            return m.load_from_dir(dir_, gid)
        def save_to_dir(self, dir_, gid):
            m = SolvedLevelMemory(enabled=True)
            m.levels = self.levels
            return m.save_to_dir(dir_, gid)

    class FakeDriver:
        def __init__(self):
            self.memory = FakeMemory()

    # Import DuckBridge（off 模式仍然可以构造，只是 enabled=False）
    sys.path.insert(0, str(ROOT / "integrations" / "ARC-AGI-3-Kaggle-Starter" / "agent"))
    from duck_bridge import DuckBridge

    os_environ_backup = None
    import os
    os_environ_backup = dict(os.environ)
    os.environ["LINGJING_DUCK"] = "off"  # 让 DuckBridge.enabled=False，不真起 driver
    os.environ["LINGJING_DUCK_CROSS_DIR"] = str(TMP)

    bridge = DuckBridge("ar25")
    assert not bridge.enabled, "DuckBridge should be off"
    assert bridge._cross_dir == str(TMP), f"cross_dir={bridge._cross_dir!r}"

    # 手动塞一个 fake driver 模拟 _start 成功后的状态
    bridge._driver = FakeDriver()
    bridge._driver.memory.record(5, ["A"] * 10, reasoning="learned in this run")
    bridge._driver.memory.levels[-1].rule = "Rule for level 5: disk persistence test"

    # 触发 save：模拟 _abandon 的开头
    saved = bridge._save_cross_game()
    assert saved, "DuckBridge._save_cross_game should return True"
    disk = TMP / "ar25.json"
    import json
    payload = json.loads(disk.read_text(encoding="utf-8"))
    assert any(r["level"] == 5 for r in payload["levels"]), "L5 rule not on disk"
    print(f"[6] DuckBridge hooked: cross_dir={bridge._cross_dir!r} L5 saved via _save_cross_game()")

    # 清掉 env 还原
    os.environ.clear()
    os.environ.update(os_environ_backup)

    print("\n=== ALL TESTS PASSED ===")
finally:
    shutil.rmtree(TMP, ignore_errors=True)


def test_failed_routes():
    """FailedRoutesMemory 单元测试：dedup、合并语义、跨局往返、capture 抓避坑线。"""
    from arc_adaptor.duck.memory import FailedRoutesMemory, FailedRoute
    TMP = Path(tempfile.mkdtemp(prefix="cross-game-fail-"))
    try:
        # ---- 1) 第一次跑：模拟在 L2 死两次、L3 死一次 ----
        f1 = FailedRoutesMemory(enabled=True)
        r1 = f1.record(2, ["A", "B", "C", "D", "E", "F"] * 2 + ["X", "Y", "Z"], "game_over")
        assert r1 is not None and r1.n_actions_total == 15
        r2 = f1.record(3, ["P", "Q", "R"] * 5, "game_over")
        assert r2 is not None
        # 死路同一关同一最后3步签名 = 重复，不记
        r1_dup = f1.record(2, ["Z", "Y", "X"] * 5 + ["X", "Y", "Z"], "game_over")
        assert r1_dup is None, "duplicate dead-end (same last-3 sig) should be skipped"
        # 同关不同死路（最后3步不同）才记新条目
        r1_diff = f1.record(2, ["Z", "Y", "X"] * 5 + ["A", "B", "C"], "game_over")
        assert r1_diff is not None, "different dead-end should be added"
        assert len(f1.routes) == 3, f"expected 3 routes, got {len(f1.routes)}"
        print(f"[F1] recorded 3 routes for L2(2)+L3(1); dedup OK")

        # ---- 2) save_to_dir 落盘 ----
        saved = f1.save_to_dir(str(TMP), "ar25")
        assert saved and (TMP / "ar25.failures.json").exists()
        print(f"[F2] saved to {TMP / 'ar25.failures.json'}")

        # ---- 3) 全新实例 load_from_dir 读回 ----
        f2 = FailedRoutesMemory(enabled=True)
        n = f2.load_from_dir(str(TMP), "ar25")
        assert n == 3, f"expected 3 loaded, got {n}"
        assert len(f2.routes) == 3
        # 字段全对
        for orig, got in zip(f1.routes, f2.routes):
            assert orig.level == got.level
            assert orig.last_actions == got.last_actions
            assert orig.reason == got.reason
            assert orig.n_actions_total == got.n_actions_total
        print(f"[F3] round-trip OK: 3 routes loaded, all fields match")

        # ---- 4) block() 生成系统 prompt 的 KNOWN DEAD-ENDS 块 ----
        block = f2.block()
        assert "KNOWN DEAD-ENDS" in block
        assert "Level 2" in block and "Level 3" in block
        assert "do not replay" in block.lower() or "dead-end" in block.lower()
        print(f"[F4] block() generates dead-ends prompt ({len(block)} chars)")

        # ---- 5) rewrite() 把块钉在 base_system 末尾，幂等 ----
        base = "Base system prompt."
        rewritten = f2.rewrite(base)
        assert rewritten.startswith(base)
        assert "KNOWN DEAD-ENDS" in rewritten
        twice = f2.rewrite(rewritten)  # 幂等：再调一次不重复
        assert twice == rewritten or twice.count("KNOWN DEAD-ENDS") == 1
        print(f"[F5] rewrite() idempotent, block pinned at end of base_system")

        # ---- 6) merge 语义：disk 只填新条目，不覆盖 in-game 记的 ----
        f2.routes[0].avoid = "IN-GAME-WRITTEN AVOID LINE"
        n_again = f2.load_from_dir(str(TMP), "ar25")
        assert n_again == 0, f"merge should add 0 (all dup), got {n_again}"
        assert f2.routes[0].avoid == "IN-GAME-WRITTEN AVOID LINE"
        print(f"[F6] merge semantics OK: in-game avoid-line wins over disk")

        # ---- 7) begin_turn + capture：模拟模型回答 "Dead-end for level N:" ----
        f3 = FailedRoutesMemory(enabled=True)
        f3.record(5, ["A", "B", "C"] * 4, "game_over")
        ask = f3.begin_turn()
        assert ask is not None and "Dead-end for level 5" in ask
        avoid_text = ("Sure. Dead-end for level 5: pushing the crate east then "
                      "south triggers the spike trap; try west first.")
        captured = f3.capture(avoid_text)
        assert captured is not None and "spike trap" in captured
        assert f3.routes[0].avoid == captured
        print(f"[F7] begin_turn + capture OK: avoid-line extracted = '{captured[:50]}...'")

        # ---- 8) begin_turn 给 1 次机会（ask_turns=1），模型不答就放弃 ----
        f4 = FailedRoutesMemory(enabled=True, ask_turns=1)
        f4.record(6, ["X", "Y", "Z"] * 3, "game_over")
        ask1 = f4.begin_turn()
        assert ask1 is not None
        ask2 = f4.begin_turn()  # 已用完次数
        assert ask2 is None, "begin_turn should give up after ask_turns"
        print(f"[F8] begin_turn gives up after ask_turns=1")

        # ---- 9) max_per_level 上限：同关死路太多挤掉最老 ----
        f5 = FailedRoutesMemory(enabled=True, max_per_level=2)
        f5.record(7, ["A1", "A2", "A3"], "game_over")
        f5.record(7, ["B1", "B2", "B3"], "game_over")
        f5.record(7, ["C1", "C2", "C3"], "game_over")  # 挤掉 A
        assert len(f5.routes) == 2
        assert all(r.level == 7 for r in f5.routes)
        print(f"[F9] max_per_level=2: 3rd route evicts oldest, kept {len(f5.routes)}")

        print("\n=== FAILED ROUTES TESTS PASSED ===")
    finally:
        shutil.rmtree(TMP, ignore_errors=True)


def test_failed_routes():
    """FailedRoutesMemory 单元测试：dedup、合并语义、跨局往返、capture 抓避坑线。"""
    from arc_adaptor.duck.memory import FailedRoutesMemory
    TMP = Path(tempfile.mkdtemp(prefix="cross-game-fail-"))
    try:
        # ---- 1) 第一次跑：模拟在 L2 死两次、L3 死一次 ----
        f1 = FailedRoutesMemory(enabled=True)
        r1 = f1.record(2, ["A", "B", "C", "D", "E", "F"] * 2 + ["X", "Y", "Z"], "game_over")
        assert r1 is not None and r1.n_actions_total == 15
        r2 = f1.record(3, ["P", "Q", "R"] * 5, "game_over")
        assert r2 is not None
        # 死路同一关同一最后3步签名 = 重复，不记
        r1_dup = f1.record(2, ["Z", "Y", "X"] * 5 + ["X", "Y", "Z"], "game_over")
        assert r1_dup is None, "duplicate dead-end (same last-3 sig) should be skipped"
        # 同关不同死路（最后3步不同）才记新条目
        r1_diff = f1.record(2, ["Z", "Y", "X"] * 5 + ["A", "B", "C"], "game_over")
        assert r1_diff is not None, "different dead-end should be added"
        assert len(f1.routes) == 3, f"expected 3 routes, got {len(f1.routes)}"
        print(f"[F1] recorded 3 routes for L2(2)+L3(1); dedup OK")

        # ---- 2) save_to_dir 落盘 ----
        saved = f1.save_to_dir(str(TMP), "ar25")
        assert saved and (TMP / "ar25.failures.json").exists()
        print(f"[F2] saved to {TMP / 'ar25.failures.json'}")

        # ---- 3) 全新实例 load_from_dir 读回 ----
        f2 = FailedRoutesMemory(enabled=True)
        n = f2.load_from_dir(str(TMP), "ar25")
        assert n == 3, f"expected 3 loaded, got {n}"
        assert len(f2.routes) == 3
        for orig, got in zip(f1.routes, f2.routes):
            assert orig.level == got.level
            assert orig.last_actions == got.last_actions
            assert orig.reason == got.reason
            assert orig.n_actions_total == got.n_actions_total
        print(f"[F3] round-trip OK: 3 routes loaded, all fields match")

        # ---- 4) block() 生成系统 prompt 的 KNOWN DEAD-ENDS 块 ----
        block = f2.block()
        assert "KNOWN DEAD-ENDS" in block
        assert "Level 2" in block and "Level 3" in block
        assert "do not replay" in block.lower() or "dead-end" in block.lower()
        print(f"[F4] block() generates dead-ends prompt ({len(block)} chars)")

        # ---- 5) rewrite() 把块钉在 base_system 末尾，幂等 ----
        base = "Base system prompt."
        rewritten = f2.rewrite(base)
        assert rewritten.startswith(base)
        assert "KNOWN DEAD-ENDS" in rewritten
        twice = f2.rewrite(rewritten)  # 幂等：再调一次不重复
        assert twice == rewritten or twice.count("KNOWN DEAD-ENDS") == 1
        print(f"[F5] rewrite() idempotent, block pinned at end of base_system")

        # ---- 6) merge 语义：disk 只填新条目，不覆盖 in-game 记的 ----
        f2.routes[0].avoid = "IN-GAME-WRITTEN AVOID LINE"
        n_again = f2.load_from_dir(str(TMP), "ar25")
        assert n_again == 0, f"merge should add 0 (all dup), got {n_again}"
        assert f2.routes[0].avoid == "IN-GAME-WRITTEN AVOID LINE"
        print(f"[F6] merge semantics OK: in-game avoid-line wins over disk")

        # ---- 7) begin_turn + capture：模拟模型回答 "Dead-end for level N:" ----
        f3 = FailedRoutesMemory(enabled=True)
        f3.record(5, ["A", "B", "C"] * 4, "game_over")
        ask = f3.begin_turn()
        assert ask is not None and "Dead-end for level 5" in ask
        avoid_text = ("Sure. Dead-end for level 5: pushing the crate east then "
                      "south triggers the spike trap; try west first.")
        captured = f3.capture(avoid_text)
        assert captured is not None and "spike trap" in captured
        assert f3.routes[0].avoid == captured
        print(f"[F7] begin_turn + capture OK: avoid-line extracted = '{captured[:50]}...'")

        # ---- 8) begin_turn 给 1 次机会（ask_turns=1），模型不答就放弃 ----
        f4 = FailedRoutesMemory(enabled=True, ask_turns=1)
        f4.record(6, ["X", "Y", "Z"] * 3, "game_over")
        ask1 = f4.begin_turn()
        assert ask1 is not None
        ask2 = f4.begin_turn()  # 已用完次数
        assert ask2 is None, "begin_turn should give up after ask_turns"
        print(f"[F8] begin_turn gives up after ask_turns=1")

        # ---- 9) max_per_level 上限：同关死路太多挤掉最老 ----
        f5 = FailedRoutesMemory(enabled=True, max_per_level=2)
        f5.record(7, ["A1", "A2", "A3"], "game_over")
        f5.record(7, ["B1", "B2", "B3"], "game_over")
        f5.record(7, ["C1", "C2", "C3"], "game_over")  # 挤掉 A
        assert len(f5.routes) == 2
        assert all(r.level == 7 for r in f5.routes)
        print(f"[F9] max_per_level=2: 3rd route evicts oldest, kept {len(f5.routes)}")

        print("\n=== FAILED ROUTES TESTS PASSED ===")
    finally:
        shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    # 主测试跑完后再跑失败路线测试
    test_failed_routes()
