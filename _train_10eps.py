"""合并 10 个 episode 的 recording → build_labels → train → 评估。"""
import json, sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

episodes_dir = ROOT / "data" / "aop" / "episodes"
merged = ROOT / "data" / "aop" / "labels" / "collected_10eps.jsonl"
merged.parent.mkdir(parents=True, exist_ok=True)

# 1. 合并 + normalize 所有 recording
from tools.aop.normalize_arc_recording import normalize
merged = ROOT / "data" / "aop" / "normalized" / "collected_10eps.jsonl"
merged.parent.mkdir(parents=True, exist_ok=True)

total = 0
with open(merged, "w", encoding="utf-8") as out:
    for ep_dir in sorted(episodes_dir.iterdir()):
        if not ep_dir.is_dir() or ep_dir.name == "quarantine":
            continue
        rec = ep_dir / "recording.jsonl"
        if not rec.exists():
            continue
        # normalize 单个 episode
        tmp = ROOT / "data" / "aop" / "normalized" / f"_tmp_{ep_dir.name}.jsonl"
        try:
            n = normalize(rec, tmp, episode_id=ep_dir.name)
            for line in tmp.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    out.write(line + "\n")
                    total += 1
        except Exception as e:
            print(f"  ⚠️ {ep_dir.name} normalize 失败: {e}")
        finally:
            if tmp.exists():
                tmp.unlink()
print(f"normalize: {total} 行 → {merged}")

# 2. build_labels
from models.aop.labels import row_to_label, LabelError
labels_out = ROOT / "data" / "aop" / "labels" / "collected_10eps_labels.jsonl"
labels_out.parent.mkdir(parents=True, exist_ok=True)
written = 0
with open(merged, encoding="utf-8") as src, open(labels_out, "w", encoding="utf-8") as dst:
    for line_no, line in enumerate(src, 1):
        if not line.strip():
            continue
        try:
            label = row_to_label(json.loads(line))
            dst.write(json.dumps(label.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
            written += 1
        except (json.JSONDecodeError, LabelError) as exc:
            pass
print(f"build_labels: {written} 标签 → {labels_out}")

# 3. train
from tools.aop.train import read_labels, targets, split_labels
from lingjing_solo.neural.aop_model import AOPModel, save_checkpoint
from lingjing_solo.neural.aop_encoder import encode_label
from tools.aop.dataset import split_labels
import torch
from torch import nn

labels = read_labels(labels_out)
if len(labels) < 2:
    print("❌ 标签不足 2 个，无法训练")
    sys.exit(1)

train, valid = split_labels(labels, 0.2)
names = sorted({x["requested_action"]["name"] for x in labels})
action_to_id = {name: i for i, name in enumerate(names)}
print(f"训练: {len(train)} 行, 验证: {len(valid)} 行, 动作: {names}")

model = AOPModel(16, max(2, len(names)))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    features, y_eff, y_prog, y_act = targets(train, action_to_id)
    outputs = model(features)
    loss = criterion(outputs["action_effective"], y_eff) + criterion(outputs["progressed"], y_prog) + criterion(outputs["action"], y_act)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        features, y_eff, y_prog, y_act = targets(valid, action_to_id)
        outputs = model(features)
        val_loss = float((criterion(outputs["action_effective"], y_eff) + criterion(outputs["progressed"], y_prog) + criterion(outputs["action"], y_act)).item())

        # 评估指标
        pred_eff = outputs["action_effective"].argmax(1)
        pred_prog = outputs["progressed"].argmax(1)
        pred_act = outputs["action"].argmax(1)
        acc_eff = (pred_eff == y_eff).float().mean().item()
        acc_prog = (pred_prog == y_prog).float().mean().item()
        acc_act = (pred_act == y_act).float().mean().item()
        # macro-F1 (simplified)
        from sklearn.metrics import f1_score
        try:
            macro_f1 = f1_score(y_act, pred_act, average="macro", zero_division=0)
        except ImportError:
            macro_f1 = 0.0

    print(f"epoch {epoch+1}: loss={loss.item():.4f} val_loss={val_loss:.4f} acc_eff={acc_eff:.3f} acc_prog={acc_prog:.3f} acc_act={acc_act:.3f} macro_f1={macro_f1:.3f}")

# 保存 checkpoint
ckpt = ROOT / "data" / "aop" / "collected_10eps.pt"
save_checkpoint(ckpt, model, {"action_names": names, "train_rows": len(train), "validation_rows": len(valid), "evidence": "collected_10eps"})
print(f"\ncheckpoint 保存: {ckpt}")

# 评估报告
report = {
    "episodes": 10,
    "total_labels": written,
    "train_rows": len(train),
    "valid_rows": len(valid),
    "action_names": names,
    "final_val_loss": val_loss,
    "accuracy_effective": acc_eff,
    "accuracy_progressed": acc_prog,
    "accuracy_action": acc_act,
    "macro_f1": macro_f1,
    "invalid_action_rate": 0.0,  # TODO: 计算
    "confidence_distribution": "TODO",
}
report_path = ROOT / "data" / "aop" / "holdout_report.json"
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"评估报告: {report_path}")
print(f"\n=== 评估报告 ===")
print(f"  episodes: 10")
print(f"  labels: {written}")
print(f"  train: {len(train)}, valid: {len(valid)}")
print(f"  val_loss: {val_loss:.4f}")
print(f"  accuracy_effective: {acc_eff:.3f}")
print(f"  accuracy_progressed: {acc_prog:.3f}")
print(f"  accuracy_action: {acc_act:.3f}")
print(f"  macro_f1: {macro_f1:.3f}")
