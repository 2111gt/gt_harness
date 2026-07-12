"""
Compatibility shims so optional ML stacks keep working across version skew.

Especially: some transformers releases removed
``find_pruneable_heads_and_indices`` from ``transformers.pytorch_utils`` while
older sentence-transformers still import it.
"""

from __future__ import annotations

from typing import List, Set, Tuple


def apply_transformers_shims() -> str:
    """
    Patch transformers if needed. Safe to call multiple times.

    Returns a short status string.
    """
    try:
        import transformers.pytorch_utils as pu
    except Exception as exc:  # noqa: BLE001
        return f"transformers unavailable: {exc}"

    if hasattr(pu, "find_pruneable_heads_and_indices"):
        return "ok"

    def find_pruneable_heads_and_indices(
        heads: List[int],
        n_heads: int,
        head_size: int,
        already_pruned_heads: Set[int],
    ) -> Tuple[Set[int], "object"]:
        """Minimal re-implementation used only for import compatibility."""
        import torch

        mask = torch.ones(n_heads, head_size)
        heads_set = set(heads) - set(already_pruned_heads)
        for head in heads_set:
            head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        return heads_set, index

    pu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices  # type: ignore[attr-defined]
    return "patched find_pruneable_heads_and_indices"


def ensure_ml_compat() -> str:
    """Apply all known shims; call before loading ST / transformers models."""
    return apply_transformers_shims()
