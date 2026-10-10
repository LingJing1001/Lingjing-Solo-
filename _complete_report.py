"""补全 holdout 报告: confidence_distribution + invalid_action_rate。"""
import json, sys
from pathlib import Path
import torch
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from lingjing_solo.neural.aop_model import load_checkpoint
from lingjing_solo.neural.aop_encoder import encode_label
from tools.aop.dataset import split_labels
from tools.aop.train import read_labels, targets

# 读数据
labels_path = ROOT / "data" / "aop" / "labels" / "collected_10eps_labels.jsonl"
ckpt_path = ROOT / "data" / "aop" / "collected_10eps.pt"
report_path = ROOT / "data" / "aop" / "holdout_report.json"

labels = read_labels(labels_path)
train, valid = split_labels(labels, 0.2)
model, config = load_checkpoint(ckpt_path)
model.eval()
action_names = list(config["action_names"])
action_to_id = {name: i for i, name in enumerate(action_names)}
action_count = max(2, len(action_names))

# 计算
confidences = []
invalid_count = 0
total_count = 0

with torch.no_grad():
    for label in valid:
        try:
            action_name = label["requested_action"]["name"]
            if action_name not in action_to_id:
                continue
            features = encode_label(label, action_id=action_to_id[action_name], action_count=action_count).unsqueeze(0)
            outputs = model(features)
            
            # confidence: softmax(action) 的 max 概率
            action_logits = outputs["action"][0]
            action_probs = torch.softmax(action_logits, dim=0)
            max_conf = float(action_probs.max().item())
            confidences.append(max_conf)
            
            # invalid action: 预测的 action 不在 legal_actions 里
            pred_action_idx = int(action_logits.argmax().item())
            pred_action_name = action_names[pred_action_idx] if pred_action_idx < len(action_names) else "UNKNOWN"
            legal_actions = label.get("legal_actions", [])
            if legal_actions and pred_action_name not in legal_actions:
                invalid_count += 1
            total_count += 1
        except Exception:
            continue

# 统计
confidences = np.array(confidences)
confidence_distribution = {
    "mean": float(confidences.mean()),
    "median": float(np.median(confidences)),
    "p25": float(np.percentile(confidences, 25)),
    "p50": float(np.percentile(confidences, 50)),
    "p75": float(np.percentile(confidences, 75)),
    "p90": float(np.percentile(confidences, 90)),
    "min": float(confidences.min()),
    "max": float(confidences.max()),
    "std": float(confidences.std()),
}
invalid_action_rate = invalid_count / max(1, total_count)

# 更新报告
report = json.loads(report_path.read_text(encoding="utf-8"))
report["confidence_distribution"] = confidence_distribution
report["invalid_action_rate"] = invalid_action_rate
report["invalid_count"] = invalid_count
report["total_valid_samples"] = total_count

with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print("=== 补全后的评估报告 ===")
print(json.dumps(report, indent=2, ensure_ascii=False))
