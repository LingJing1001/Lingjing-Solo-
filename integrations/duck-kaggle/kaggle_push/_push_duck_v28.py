"""Push the Duck v28 kernel（独立 slug，不动已在榜上的 v27/v25 kernel）。

v28 与 v27 的差异（全部记在 push_v28/kernel-metadata.json + notebook 里）
---------------------------------------------------------------------
1. **覆盖率**（v28 的唯一杠杆）：
   solver.concurrency 8→16、max_runtime_s_per_game 5400→7920（官方值）、
   vLLM admit limit (--max-num-seqs) 8→16。
   官方 Duck 是 28 并发 × 7920 s 铺满 9 h 窗口（110 局 ÷ 28 = 4 波 × 7920 = 31680 s
   ≈ 32400 s）；我们 v27 是 8 并发 × 5400 s ⇒ 只能覆盖约 45/110 局（41%）。
   官方评分 = 110 局局分平均 ⇒ 没玩到的局直接算 0，覆盖率是分数的直接乘数。
2. **提前放弃开关真正生效**：用 notebook 内 base64 覆盖 overlay 的 solver.py
   （overlay 数据集发版走 www.googleapis.com，本机不通，overlay 卡在 v5）。
3. **短 bench**：非竞赛 rerun 时每局上限压到 900 s，推一次 kernel 从 8.3 h 降到 ~1 h。
4. **路径探测**：wheels / environment_files 改「多候选 + glob 兜底 + 响亮失败」。

前置：
    python patch_v28.py          # 生成 push_v28/lingjing-duck-harness-v28.ipynb + metadata
    python _push_duck_v28.py     # 推送（starter venv python，有 kaggle 包）
"""
from __future__ import annotations

import time
from pathlib import Path

PUSH_DIR = Path(__file__).resolve().parent / "push_v28"
KERNEL = "liaojiaxuan/lingjing-duck-harness-v28"


def main() -> int:
    from kaggle.api.kaggle_api_extended import KaggleApi

    nb = PUSH_DIR / "lingjing-duck-harness-v28.ipynb"
    meta = PUSH_DIR / "kernel-metadata.json"
    for f in (nb, meta):
        if not f.is_file():
            print(f"MISSING {f} —— 先跑 patch_v28.py", flush=True)
            return 2

    for i in range(1, 9):
        print(f"PUSH {i}", flush=True)
        try:
            api = KaggleApi()
            api.authenticate()
            resp = api.kernels_push(str(PUSH_DIR))
            ver = None
            if resp is not None:
                d = resp.to_dict() if hasattr(resp, "to_dict") else getattr(resp, "__dict__", {})
                print("PUSH_RESP", d, flush=True)
                for key in ("versionNumber", "version_number", "currentVersionNumber"):
                    if isinstance(d, dict) and d.get(key) is not None:
                        print(key, d[key], flush=True)
                        if ver is None:
                            ver = d[key]
            time.sleep(2)
            st = api.kernels_status(KERNEL)
            print("STATUS", st, flush=True)
            print("OK", KERNEL, "version", ver, flush=True)
            print("URL", f"https://www.kaggle.com/code/{KERNEL}", flush=True)
            if ver is not None:
                (PUSH_DIR / "_last_push_version_v28.txt").write_text(str(ver), encoding="utf-8")
            return 0
        except Exception as e:
            print(type(e).__name__, e, flush=True)
            time.sleep(6 + i)
    print("PUSH_EXHAUSTED", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
