"""核验三份 notebook 的 A/B 接线是否与构建开关一致。"""
import json
import re
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks"

for name in ("submission.ipynb", "submission_llm0.ipynb", "submission_llm1.ipynb"):
    j = json.load(open(NB / name, encoding="utf-8"))
    src = "\n".join("".join(c.get("source", [])) for c in j["cells"])
    sl = re.findall(r"SELFLEARN_LLM'\]\s*=\s*'(\d)'", src)
    llm = re.findall(r"LINGJING_LLM'\]\s*=\s*'(\w+)'", src)
    sc = re.findall(r"LINGJING_SCRIPTS',\s*'(\w+)'", src)
    drift = "SELFLEARN_LLM_DRIFT" in src
    print(f"{name:24s} cells={len(j['cells']):2d} bench={'BENCH_SELFLEARN' in src} "
          f"SELFLEARN_LLM={sl} LINGJING_LLM={llm} SCRIPTS={sc} drift_guard={drift} "
          f"chars={len(src)}")
