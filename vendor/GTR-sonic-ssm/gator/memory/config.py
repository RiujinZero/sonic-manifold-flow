"""Configuration for Surprise-Gated Block Memory (SGBM), design spec section 21."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["SurpriseMemoryConfig"]


@dataclass
class SurpriseMemoryConfig:
    fused_dim: int
    key_dim: int = 16
    value_dim: int = 32
    memory_slots: int = 32
    quarantine_slots: int = 8
    retrieval_topk: int = 4
    exact_cache_blocks: int = 16
    surprise_ema: float = 0.99
    surprise_momentum_beta: float = 0.90
    merge_threshold: float = 0.82
    write_threshold: float = 0.55
    quarantine_threshold: float = 0.20
    promotion_confirmations: int = 2
    memory_dropout: float = 0.10
    base_decay: float = 1e-3

    def __post_init__(self) -> None:
        if self.fused_dim <= 0:
            raise ValueError("fused_dim must be positive")
        if self.key_dim <= 0:
            raise ValueError("key_dim must be positive")
        if self.value_dim <= 0:
            raise ValueError("value_dim must be positive")
        if self.memory_slots <= 0:
            raise ValueError("memory_slots must be positive")
        if self.quarantine_slots < 0:
            raise ValueError("quarantine_slots must be nonnegative")
        if not 0 < self.retrieval_topk <= self.memory_slots:
            raise ValueError("retrieval_topk must be in (0, memory_slots]")
        if self.exact_cache_blocks < 0:
            raise ValueError("exact_cache_blocks must be nonnegative")
        if not 0.0 < self.surprise_ema < 1.0:
            raise ValueError("surprise_ema must be in (0, 1)")
        if not 0.0 <= self.surprise_momentum_beta <= 1.0:
            raise ValueError("surprise_momentum_beta must be in [0, 1]")
        if not -1.0 <= self.merge_threshold <= 1.0:
            raise ValueError("merge_threshold must be a cosine similarity in [-1, 1]")
        if not 0.0 <= self.quarantine_threshold <= self.write_threshold <= 1.0:
            raise ValueError(
                "thresholds must satisfy "
                "0 <= quarantine_threshold <= write_threshold <= 1"
            )
        if self.promotion_confirmations < 1:
            raise ValueError("promotion_confirmations must be at least 1")
        if not 0.0 <= self.memory_dropout <= 1.0:
            raise ValueError("memory_dropout must be in [0, 1]")
        if self.base_decay < 0.0:
            raise ValueError("base_decay must be nonnegative")
