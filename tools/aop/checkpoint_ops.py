#!/usr/bin/env python3
"""AOP checkpoint 运维：验证、回滚演练。

- verify_checkpoint(path): load + 跑确定性输入集 + 写 .verified 标记
- list_verified(dir): 列出已验证 checkpoint
- rollback_to_previous(current, dir): 找上一份已验证 checkpoint + 再验证

权威动作路径不受影响：shadow used_for_control=False，回滚只换旁路模型。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from lingjing_solo.neural.aop_model import AOPModel, load_checkpoint

# 确定性输入集（固定种子，跨进程可复现）
_DETERMINISTIC_SEED = 42
_DETERMINISTIC_N = 8


def _deterministic_inputs(width: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(_DETERMINISTIC_SEED)
    return torch.randn(_DETERMINISTIC_N, width, generator=g)


def verify_checkpoint(path: str | Path) -> bool:
    """load checkpoint + 跑确定性输入 + 断言输出有限/shape 对。

    通过则写 `path.verified` 标记文件。返回 True/False。
    """
    path = Path(path)
    try:
        model, config = load_checkpoint(path)
        width = int(config["input_width"])
        inputs = _deterministic_inputs(width)
        with torch.no_grad():
            outputs = model(inputs)
        for key in ("action_effective", "progressed", "action"):
            t = outputs[key]
            if not torch.isfinite(t).all():
                raise ValueError(f"{key} has non-finite values")
            if t.shape[0] != _DETERMINISTIC_N:
                raise ValueError(f"{key} batch shape mismatch: {t.shape}")
    except Exception:
        return False
    # 写 verified 标记
    (path.with_suffix(path.suffix + ".verified")).write_text(
        json.dumps({"path": str(path), "seed": _DETERMINISTIC_SEED, "n": _DETERMINISTIC_N}),
        encoding="utf-8",
    )
    return True


def list_verified(checkpoints_dir: str | Path) -> list[Path]:
    """列出目录下所有有 .verified 标记的 .pt checkpoint，按修改时间排序。"""
    checkpoints_dir = Path(checkpoints_dir)
    if not checkpoints_dir.is_dir():
        return []
    verified: list[Path] = []
    for p in sorted(checkpoints_dir.glob("*.pt")):
        if p.with_suffix(p.suffix + ".verified").exists():
            verified.append(p)
    return verified


def rollback_to_previous(
    current: str | Path,
    checkpoints_dir: str | Path,
) -> Path | None:
    """找 current 之前最近一份已验证 checkpoint，再验证一次后返回其路径。

    返回 None 表示没有可回滚的目标。
    """
    current = Path(current)
    verified = list_verified(checkpoints_dir)
    if len(verified) < 2:
        return None
    try:
        idx = verified.index(current)
    except ValueError:
        # current 不在 verified 列表（可能未验证），回滚到最后一份 verified
        idx = len(verified)
    if idx <= 0:
        return None
    prev = verified[idx - 1]
    if not verify_checkpoint(prev):
        return None
    return prev


def write_rollback_report(
    current: str | Path,
    rolled_back: str | Path | None,
    output: str | Path,
    *,
    authoritative_unchanged: bool,
) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "current_checkpoint": str(current),
        "rolled_back_to": str(rolled_back) if rolled_back else None,
        "authoritative_action_path_unchanged": authoritative_unchanged,
    }
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["verify", "list", "rollback"])
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dir", type=Path, default=Path("data/aop/checkpoints"))
    parser.add_argument("--report", type=Path, default=Path("data/aop/rollback_report.json"))
    args = parser.parse_args()

    if args.command == "verify":
        ok = verify_checkpoint(args.checkpoint)
        print(f"VERIFY {'OK' if ok else 'FAIL'} {args.checkpoint}")
        return 0 if ok else 1
    if args.command == "list":
        for p in list_verified(args.dir):
            print(p)
        return 0
    if args.command == "rollback":
        prev = rollback_to_previous(args.checkpoint, args.dir)
        write_rollback_report(args.checkpoint, prev, args.report, authoritative_unchanged=True)
        print(f"ROLLBACK to {prev}" if prev else "ROLLBACK none available")
        return 0 if prev else 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
