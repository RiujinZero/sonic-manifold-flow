"""Lexical (non-neural) retrieval baseline for the Phase 1 representation sweep.

design-v0.3.md SS9 Phase 1's exit gate requires the selected Qwen
representation to "materially exceed random and lexical-only controls on
paraphrased held-out queries." This is the lexical control: retrieval by raw
token overlap, with no model involved at all, so any margin the selected
representation shows is attributable to something beyond surface wording.
"""

from __future__ import annotations


def jaccard_token_similarity(first: str, second: str) -> float:
    """Word-overlap similarity in [0, 1]; 0 if either text has no tokens."""
    first_tokens = set(first.lower().split())
    second_tokens = set(second.lower().split())
    if not first_tokens or not second_tokens:
        return 0.0
    return len(first_tokens & second_tokens) / len(first_tokens | second_tokens)


def lexical_retrieval_topk_accuracy(
    queries: list[str], bank: list[str], true_index: list[int], *, k: int
) -> float:
    """Fraction of queries whose true bank match ranks in the top-k by
    `jaccard_token_similarity`, mirroring `retrieval_topk_accuracy`'s
    protocol so the two controls are directly comparable."""
    if not queries or not bank:
        raise ValueError("queries and bank must be non-empty")
    if len(queries) != len(true_index):
        raise ValueError("true_index must have one entry per query")
    if not 0 < k <= len(bank):
        raise ValueError("k must be in [1, bank size]")
    hits = 0
    for query, target in zip(queries, true_index, strict=True):
        scored = sorted(
            range(len(bank)),
            key=lambda index: jaccard_token_similarity(query, bank[index]),
            reverse=True,
        )
        if target in scored[:k]:
            hits += 1
    return hits / len(queries)
