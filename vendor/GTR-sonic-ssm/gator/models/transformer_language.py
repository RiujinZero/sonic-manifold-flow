"""One-shot non-autoregressive Transformer language baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class OneShotTransformerLanguageConfig:
    """Configuration for simultaneous blank-slot response formation."""

    vocab_size: int
    d_model: int = 38
    n_heads: int = 2
    ff_width: int = 76
    n_layers: int = 4
    context_tokens: int = 64
    response_tokens: int = 32
    dropout: float = 0.0
    mask_token_id: int = 3
    blank_token_id: int | None = None
    forbidden_output_token_ids: tuple[int, ...] = ()

    architecture: str = "one_shot_transformer"

    def __post_init__(self) -> None:
        if self.vocab_size <= 0 or self.d_model <= 0:
            raise ValueError("vocab_size and d_model must be positive")
        if self.n_heads <= 0 or self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.ff_width <= 0 or self.n_layers <= 0:
            raise ValueError("ff_width and n_layers must be positive")
        if self.context_tokens <= 0 or self.response_tokens <= 0:
            raise ValueError("context_tokens and response_tokens must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0 <= self.mask_token_id < self.vocab_size:
            raise ValueError("mask_token_id must be inside the vocabulary")
        if self.blank_token_id is not None:
            if not 0 <= self.blank_token_id < self.vocab_size:
                raise ValueError("blank_token_id must be inside the vocabulary")
            if self.blank_token_id in self.forbidden_output_token_ids:
                raise ValueError("blank_token_id cannot be forbidden at readout")
        if len(set(self.forbidden_output_token_ids)) != len(
            self.forbidden_output_token_ids
        ):
            raise ValueError("forbidden_output_token_ids must be unique")
        if any(
            token_id < 0 or token_id >= self.vocab_size
            for token_id in self.forbidden_output_token_ids
        ):
            raise ValueError("forbidden output token IDs must be inside the vocabulary")

    def to_dict(self) -> dict:
        return asdict(self)


class OneShotTransformerLanguageModel(nn.Module):
    """Encode context and all blank output queries in one noncausal pass.

    Output predictions are never fed back. Every response slot is represented
    by a learned query and all slots can interact bidirectionally through the
    encoder. The language-model projection is tied to the token embedding.
    """

    def __init__(self, config: OneShotTransformerLanguageConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.register_buffer(
            "_forbidden_output_token_ids",
            torch.tensor(config.forbidden_output_token_ids, dtype=torch.long),
            persistent=False,
        )
        self.position_embedding = nn.Embedding(
            config.context_tokens + config.response_tokens, config.d_model
        )
        self.segment_embedding = nn.Embedding(2, config.d_model)
        self.output_queries = nn.Parameter(
            torch.empty(config.response_tokens, config.d_model)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.ff_width,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=config.n_layers, norm=nn.LayerNorm(config.d_model)
        )
        self.input_norm = nn.LayerNorm(config.d_model)
        self.lm_bias = nn.Parameter(torch.zeros(config.vocab_size))
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        nn.init.normal_(self.segment_embedding.weight, std=0.02)
        nn.init.normal_(self.output_queries, std=0.02)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        context_ids: torch.Tensor,
        context_mask: torch.Tensor,
        *,
        output_slots: int,
        temperature: float | None = None,
    ) -> torch.Tensor:
        del temperature
        batch, context_length = context_ids.shape
        if context_length > self.config.context_tokens:
            raise ValueError("context exceeds configured context_tokens")
        if output_slots > self.config.response_tokens:
            raise ValueError("output_slots exceeds configured response_tokens")

        device = context_ids.device
        context_positions = torch.arange(context_length, device=device)
        output_positions = torch.arange(
            context_length, context_length + output_slots, device=device
        )
        context = self.token_embedding(context_ids)
        context = context + self.position_embedding(context_positions)[None]
        context = context + self.segment_embedding.weight[0]
        queries = self.output_queries[:output_slots][None].expand(batch, -1, -1)
        queries = queries + self.position_embedding(output_positions)[None]
        queries = queries + self.segment_embedding.weight[1]
        states = self.input_norm(torch.cat([context, queries], dim=1))

        output_valid = torch.ones(batch, output_slots, dtype=torch.bool, device=device)
        valid = torch.cat([context_mask.bool(), output_valid], dim=1)
        states = self.encoder(states, src_key_padding_mask=~valid)
        output_states = states[:, context_length:]
        logits = output_states @ self.token_embedding.weight.t() + self.lm_bias
        if self._forbidden_output_token_ids.numel():
            logits = logits.index_fill(
                -1, self._forbidden_output_token_ids, float("-inf")
            )
        return logits
