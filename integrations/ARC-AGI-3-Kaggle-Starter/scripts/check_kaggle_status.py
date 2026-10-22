"""查 Kaggle kernel 状态 + 周 GPU 配额 + 提交列表。

方法名对 kaggle 2.2.4：kernels_status / competition_submissions / quota_view。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

SLUGS = [
    ("liaojiaxuan", "smartrouter-merged-ab-llm0"),
    ("liaojiaxuan", "smartrouter-merged-ab-llm1"),
    ("liaojiaxuan", "lingjing-duck-harness-v27"),
]

for owner, slug in SLUGS:
    try:
        st = api.kernels_status(f"{owner}/{slug}")
        print(f"{owner}/{slug}: status={st.status}")
    except Exception as exc:  # noqa: BLE001
        print(f"{owner}/{slug}: ERROR {type(exc).__name__}: {str(exc)[:200]}")

print("--- quota_view ---")
try:
    q = api.quota_view()
    print(q)
except Exception as exc:  # noqa: BLE001
    print("quota ERROR", type(exc).__name__, str(exc)[:300])

print("--- 提交列表 ---")
try:
    subs = api.competition_submissions("arc-prize-2026")
    for s in list(subs)[:8]:
        print(f"  {getattr(s, 'ref', '?')} score={getattr(s, 'publicScore', '?')} "
              f"desc={str(getattr(s, 'description', ''))[:70]}")
except Exception as exc:  # noqa: BLE001
    print("  submissions ERROR", type(exc).__name__, str(exc)[:300])
