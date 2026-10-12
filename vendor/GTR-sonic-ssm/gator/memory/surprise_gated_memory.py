"""SurpriseGatedMemory: the top-level read/write API (SGBM design spec section 21).

Wires together every Phase B-D module into the reference
``read()``/``write()`` interface:

- ``read()`` scores and top-k-blends stable memory (``retrieval.py``) and
  bumps usage on the slots that were actually used.
- ``write()`` runs the fixed write-gate formula (``controller.py``) to
  decide, per batch lane, whether a candidate is ignored, quarantined, or
  merged/allocated into stable memory (``store.py``/``quarantine.py``), and
  only that lane's memory changes — matching every Phase B function's
  ``active_mask`` no-op guarantee for the other two decisions.
- ``maintain()`` is periodic housekeeping (decay/forget, quarantine
  promotion, quarantine expiry) that a caller invokes once per block
  alongside — not inside — ``write()``, since the spec's own section 22
  pseudocode does not show maintenance running as part of every write.

Every mutating call is ``torch.no_grad()`` (design spec section 21: "the
online memory-state update itself runs under torch.no_grad()"). Deriving
the query, key, and value vectors from an actual GTR-L forward pass is
Phase F — this module only consumes them.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from gator.memory.config import SurpriseMemoryConfig
from gator.memory.controller import (
    WriteDecision,
    WriteGateCoefficients,
    commit_strength_from_gate,
    decide_write,
    write_gate_strength,
)
from gator.memory.quarantine import (
    expire_quarantine,
    promote_confirmed,
    write_or_confirm_quarantine,
)
from gator.memory.retrieval import bump_retrieval_usage, retrieve_top_k
from gator.memory.state import SurpriseMemoryState
from gator.memory.store import decay_stable_memory, write_stable_memory

__all__ = ["SurpriseGatedMemory"]


class SurpriseGatedMemory(nn.Module):
    def __init__(
        self,
        config: SurpriseMemoryConfig,
        *,
        gate_coefficients: WriteGateCoefficients | None = None,
        retrieval_temperature: float = 1.0,
        retrieval_strength_weight: float = 0.0,
        retrieval_usage_weight: float = 0.0,
        retrieval_age_weight: float = 0.0,
        retrieval_usage_threshold: float = 0.05,
        eviction_gamma: float = 1e-3,
        reliability_ema_decay: float = 0.9,
        quarantine_reliability_ema_decay: float = 0.5,
        quarantine_max_age_unconfirmed: float = 8.0,
        quarantine_max_age_confirmed: float | None = None,
        decay_time_constant: float = 10.0,
        decay_forget_threshold: float = 1e-4,
    ) -> None:
        super().__init__()
        self.config = config
        self.gate_coefficients = gate_coefficients or WriteGateCoefficients()
        self.retrieval_temperature = retrieval_temperature
        self.retrieval_strength_weight = retrieval_strength_weight
        self.retrieval_usage_weight = retrieval_usage_weight
        self.retrieval_age_weight = retrieval_age_weight
        self.retrieval_usage_threshold = retrieval_usage_threshold
        self.eviction_gamma = eviction_gamma
        self.reliability_ema_decay = reliability_ema_decay
        self.quarantine_reliability_ema_decay = quarantine_reliability_ema_decay
        self.quarantine_max_age_unconfirmed = quarantine_max_age_unconfirmed
        self.quarantine_max_age_confirmed = quarantine_max_age_confirmed
        self.decay_time_constant = decay_time_constant
        self.decay_forget_threshold = decay_forget_threshold

    def read(self, query: Tensor, state: SurpriseMemoryState) -> tuple[Tensor, dict[str, Tensor]]:
        """Score, top-k select, and blend stable memory (design spec section 13)."""

        retrieved, metrics = retrieve_top_k(
            query,
            state,
            top_k=self.config.retrieval_topk,
            temperature=self.retrieval_temperature,
            strength_weight=self.retrieval_strength_weight,
            usage_weight=self.retrieval_usage_weight,
            age_weight=self.retrieval_age_weight,
        )
        bump_retrieval_usage(
            state,
            metrics["retrieval_indices"],
            metrics["retrieval_weights"],
            threshold=self.retrieval_usage_threshold,
        )
        return retrieved, metrics

    @torch.no_grad()
    def write(
        self,
        candidate_key: Tensor,
        candidate_value: Tensor,
        surprise: Tensor,
        reliability: Tensor,
        state: SurpriseMemoryState,
        *,
        novelty: Tensor | None = None,
    ) -> SurpriseMemoryState:
        """Gate a candidate write into ignore / quarantine / stable memory.

        ``surprise`` is the already-momentum-blended combined surprise
        (``surprise_metrics["combined"]`` in the spec's section 22
        pseudocode). ``novelty`` is optional and defaults to zero: the
        spec's reference ``write()`` signature (section 21) only takes one
        combined surprise value, but the write-gate formula in section 11
        weights semantic novelty a second time on top of its contribution
        inside "combined" — pass it explicitly to reproduce that formula
        exactly, or omit it to let the gate be driven by ``surprise`` alone.
        """

        batch_size = candidate_key.shape[0]
        if novelty is None:
            novelty = torch.zeros(batch_size, device=candidate_key.device)

        write_gate = write_gate_strength(surprise, novelty, coefficients=self.gate_coefficients)
        commit_strength = commit_strength_from_gate(write_gate, reliability)
        decision = decide_write(commit_strength, self.config)

        stable_mask = decision == int(WriteDecision.STABLE)
        quarantine_mask = decision == int(WriteDecision.QUARANTINE)

        if bool(stable_mask.any()):
            write_stable_memory(
                state,
                candidate_key=candidate_key,
                candidate_value=candidate_value,
                commit_strength=commit_strength,
                reliability=reliability,
                surprise=surprise,
                config=self.config,
                active_mask=stable_mask,
                eviction_gamma=self.eviction_gamma,
                reliability_ema_decay=self.reliability_ema_decay,
            )

        if bool(quarantine_mask.any()):
            write_or_confirm_quarantine(
                state,
                candidate_key=candidate_key,
                candidate_value=candidate_value,
                write_strength=commit_strength,
                reliability=reliability,
                config=self.config,
                active_mask=quarantine_mask,
                eviction_gamma=self.eviction_gamma,
                reliability_ema_decay=self.quarantine_reliability_ema_decay,
            )

        return state

    @torch.no_grad()
    def maintain(self, state: SurpriseMemoryState) -> dict[str, Tensor]:
        """Periodic housekeeping: decay/forget, quarantine promotion, expiry.

        Intended to be called once per block by the training/inference loop
        alongside (not inside) :meth:`write` — see module docstring.
        """

        forgotten = decay_stable_memory(
            state,
            self.config,
            time_constant=self.decay_time_constant,
            forget_threshold=self.decay_forget_threshold,
        )
        promoted = promote_confirmed(state, self.config, eviction_gamma=self.eviction_gamma)
        expired = expire_quarantine(
            state,
            max_age_unconfirmed=self.quarantine_max_age_unconfirmed,
            max_age_confirmed=self.quarantine_max_age_confirmed,
        )
        return {"forgotten": forgotten, "promoted": promoted, "expired": expired}
