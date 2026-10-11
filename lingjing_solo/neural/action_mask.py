"""Fail-closed legal-action masking for AOP action logits."""
from __future__ import annotations

from collections.abc import Sequence
import torch


def _legal_indices(legal_actions: Sequence[str], action_names: Sequence[str]) -> list[int]:
    index = {name: i for i, name in enumerate(action_names)}
    unknown = sorted({name for name in legal_actions if name not in index})
    if unknown:
        raise ValueError(f"unknown legal action(s): {unknown}")
    indices = sorted({index[name] for name in legal_actions})
    if not indices:
        raise ValueError("legal_actions must not be empty")
    return indices


def mask_action_logits(
    logits: torch.Tensor,
    legal_actions: Sequence[str],
    action_names: Sequence[str],
) -> torch.Tensor:
    """Set illegal logits to -inf without mutating the input tensor."""
    if logits.ndim not in (1, 2):
        raise ValueError("action logits must be 1D or 2D")
    if logits.shape[-1] != len(action_names):
        raise ValueError("logit width does not match action_names")
    indices = _legal_indices(legal_actions, action_names)
    masked = logits.clone()
    legal = torch.zeros(len(action_names), dtype=torch.bool, device=logits.device)
    legal[indices] = True
    masked[..., ~legal] = float("-inf")
    return masked


def choose_masked_action(
    logits: torch.Tensor,
    legal_actions: Sequence[str],
    action_names: Sequence[str],
) -> str:
    masked = mask_action_logits(logits, legal_actions, action_names)
    if masked.ndim != 1:
        raise ValueError("choose_masked_action expects one action-logit vector")
    return action_names[int(masked.argmax().item())]
