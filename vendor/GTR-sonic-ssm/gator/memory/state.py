"""Per-batch-lane Surprise-Gated Block Memory state, design spec sections 5 and 21.

``SurpriseMemoryState`` is pure tensor state — no learned parameters live
here. It is allocated per training/inference run, reset per batch lane at
episode boundaries, and mutated only under ``torch.no_grad()`` by the store/
retrieval/quarantine mechanics built on top of it (Phase B).
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from gator.memory.config import SurpriseMemoryConfig

__all__ = ["SurpriseMemoryState"]


class SurpriseMemoryState:
    """Stable memory, quarantine, and running surprise state for one batch.

    Shapes (B = batch size, M = memory_slots, Q = quarantine_slots, K =
    key_dim, V = value_dim, D = fused_dim), matching the design spec's
    reference API (section 21):

    ``keys [B,M,K]``, ``values [B,M,V]``, ``strength/age/usage/reliability/
    surprise/valid [B,M]``; ``quarantine_keys [B,Q,K]``, ``quarantine_values
    [B,Q,V]``, ``quarantine_strength/confirmations/age/valid [B,Q]``;
    ``surprise_momentum [B]``; ``previous_block_summary [B,D]``.
    """

    def __init__(
        self,
        *,
        keys: Tensor,
        values: Tensor,
        strength: Tensor,
        age: Tensor,
        usage: Tensor,
        reliability: Tensor,
        surprise: Tensor,
        valid: Tensor,
        quarantine_keys: Tensor,
        quarantine_values: Tensor,
        quarantine_strength: Tensor,
        quarantine_confirmations: Tensor,
        quarantine_age: Tensor,
        quarantine_valid: Tensor,
        surprise_momentum: Tensor,
        previous_block_summary: Tensor,
    ) -> None:
        self.keys = keys
        self.values = values
        self.strength = strength
        self.age = age
        self.usage = usage
        self.reliability = reliability
        self.surprise = surprise
        self.valid = valid
        self.quarantine_keys = quarantine_keys
        self.quarantine_values = quarantine_values
        self.quarantine_strength = quarantine_strength
        self.quarantine_confirmations = quarantine_confirmations
        self.quarantine_age = quarantine_age
        self.quarantine_valid = quarantine_valid
        self.surprise_momentum = surprise_momentum
        self.previous_block_summary = previous_block_summary
        self._validate_shapes()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def allocate(
        cls,
        batch_size: int,
        config: SurpriseMemoryConfig,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "SurpriseMemoryState":
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        B, M, Q, K, V, D = (
            batch_size,
            config.memory_slots,
            config.quarantine_slots,
            config.key_dim,
            config.value_dim,
            config.fused_dim,
        )
        zeros = lambda *shape: torch.zeros(*shape, device=device, dtype=dtype)
        return cls(
            keys=zeros(B, M, K),
            values=zeros(B, M, V),
            strength=zeros(B, M),
            age=zeros(B, M),
            usage=zeros(B, M),
            reliability=zeros(B, M),
            surprise=zeros(B, M),
            valid=torch.zeros(B, M, device=device, dtype=torch.bool),
            quarantine_keys=zeros(B, Q, K),
            quarantine_values=zeros(B, Q, V),
            quarantine_strength=zeros(B, Q),
            quarantine_confirmations=torch.zeros(B, Q, device=device, dtype=torch.int64),
            quarantine_age=zeros(B, Q),
            quarantine_valid=torch.zeros(B, Q, device=device, dtype=torch.bool),
            surprise_momentum=zeros(B),
            previous_block_summary=zeros(B, D),
        )

    def _validate_shapes(self) -> None:
        batch_size = self.keys.shape[0]
        expectations = {
            "keys": (batch_size, None, None),
            "values": (batch_size, None, None),
            "strength": (batch_size, None),
            "age": (batch_size, None),
            "usage": (batch_size, None),
            "reliability": (batch_size, None),
            "surprise": (batch_size, None),
            "valid": (batch_size, None),
            "quarantine_keys": (batch_size, None, None),
            "quarantine_values": (batch_size, None, None),
            "quarantine_strength": (batch_size, None),
            "quarantine_confirmations": (batch_size, None),
            "quarantine_age": (batch_size, None),
            "quarantine_valid": (batch_size, None),
            "surprise_momentum": (batch_size,),
            "previous_block_summary": (batch_size, None),
        }
        for name, expected_rank_shape in expectations.items():
            tensor = getattr(self, name)
            if tensor.ndim != len(expected_rank_shape):
                raise ValueError(
                    f"{name} must have {len(expected_rank_shape)} dims, "
                    f"got shape {tuple(tensor.shape)}"
                )
            if tensor.shape[0] != batch_size:
                raise ValueError(
                    f"{name} batch dimension {tensor.shape[0]} does not "
                    f"match keys' batch dimension {batch_size}"
                )
        if self.keys.shape[1] != self.values.shape[1]:
            raise ValueError("keys and values must share the same slot count")
        if self.quarantine_keys.shape[1] != self.quarantine_values.shape[1]:
            raise ValueError(
                "quarantine_keys and quarantine_values must share the same slot count"
            )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def batch_size(self) -> int:
        return self.keys.shape[0]

    @property
    def memory_slots(self) -> int:
        return self.keys.shape[1]

    @property
    def quarantine_slots(self) -> int:
        return self.quarantine_keys.shape[1]

    @property
    def device(self) -> torch.device:
        return self.keys.device

    @property
    def dtype(self) -> torch.dtype:
        return self.keys.dtype

    def _field_names(self) -> tuple[str, ...]:
        return (
            "keys",
            "values",
            "strength",
            "age",
            "usage",
            "reliability",
            "surprise",
            "valid",
            "quarantine_keys",
            "quarantine_values",
            "quarantine_strength",
            "quarantine_confirmations",
            "quarantine_age",
            "quarantine_valid",
            "surprise_momentum",
            "previous_block_summary",
        )

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def reset_lanes(self, lane_mask: Tensor) -> None:
        """Zero out state for the batch lanes where ``lane_mask`` is True.

        Lanes where ``lane_mask`` is False are left completely untouched —
        this is what keeps unrelated sequences in the same batch from
        leaking state into each other across an episode boundary (design
        spec unit test #3).
        """

        if lane_mask.shape != (self.batch_size,):
            raise ValueError(
                f"lane_mask must have shape ({self.batch_size},), got "
                f"{tuple(lane_mask.shape)}"
            )
        mask = lane_mask.to(dtype=torch.bool, device=self.device)
        if not bool(mask.any()):
            return
        for name in self._field_names():
            tensor = getattr(self, name)
            zero_fill = torch.zeros_like(tensor[mask])
            tensor[mask] = zero_fill

    @torch.no_grad()
    def detach_(self) -> "SurpriseMemoryState":
        """Detach every tensor in place — the online no-grad boundary.

        Ensures a write derived from this state can never carry an autograd
        graph back through a prior block (design spec unit test #12).
        """

        for name in self._field_names():
            tensor = getattr(self, name)
            setattr(self, name, tensor.detach())
        return self

    def clone(self) -> "SurpriseMemoryState":
        kwargs = {name: getattr(self, name).clone() for name in self._field_names()}
        return SurpriseMemoryState(**kwargs)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Tensor]:
        return {name: getattr(self, name).detach().clone() for name in self._field_names()}

    def load_state_dict(self, state: Mapping[str, Tensor]) -> None:
        expected = set(self._field_names())
        missing = expected - set(state)
        if missing:
            raise KeyError(f"missing SurpriseMemoryState fields: {sorted(missing)}")
        for name in self._field_names():
            setattr(self, name, state[name].clone())
        self._validate_shapes()
