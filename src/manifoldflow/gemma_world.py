"""Gemma-native continuous latent prefix for retrieved world state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class PreparedGemmaInput:
    inputs_embeds: Tensor
    attention_mask: Tensor
    labels: Tensor | None
    memory_tokens: int


class LatentMemoryPort(nn.Module):
    """Map typed world values into weighted Gemma embedding tokens."""

    def __init__(
        self,
        state_dimension: int = 256,
        key_dimension: int = 128,
        hidden_dimension: int = 640,
        bottleneck_dimension: int = 64,
        type_count: int = 8,
    ) -> None:
        super().__init__()
        self.state_dimension = state_dimension
        self.key_dimension = key_dimension
        self.hidden_dimension = hidden_dimension
        self.value_projection = nn.Sequential(
            nn.Linear(state_dimension, bottleneck_dimension),
            nn.GELU(),
            nn.Linear(bottleneck_dimension, hidden_dimension),
        )
        self.identity_projection = nn.Sequential(
            nn.Linear(key_dimension, bottleneck_dimension),
            nn.GELU(),
            nn.Linear(bottleneck_dimension, hidden_dimension),
        )
        self.type_embedding = nn.Embedding(type_count, hidden_dimension)
        self.recency_projection = nn.Linear(1, hidden_dimension, bias=False)
        self.confidence_projection = nn.Linear(1, hidden_dimension, bias=False)
        self.gate = nn.Parameter(torch.tensor(-2.0))

    def forward(
        self,
        values: Tensor,
        *,
        identity_keys: Tensor | None = None,
        retrieval_weights: Tensor | None = None,
        confidence: Tensor | None = None,
        recency: Tensor | None = None,
        type_ids: Tensor | None = None,
    ) -> Tensor:
        if values.ndim != 3 or values.shape[-1] != self.state_dimension:
            raise ValueError(
                f"values must have shape [batch, memory, {self.state_dimension}]"
            )
        batch, count, _ = values.shape
        device = values.device
        dtype = values.dtype
        if retrieval_weights is None:
            retrieval_weights = torch.ones((batch, count), device=device, dtype=dtype)
        if confidence is None:
            confidence = torch.ones_like(retrieval_weights)
        if recency is None:
            recency = torch.zeros_like(retrieval_weights)
        if type_ids is None:
            type_ids = torch.zeros((batch, count), device=device, dtype=torch.long)
        expected = (batch, count)
        if any(
            item.shape != expected
            for item in (retrieval_weights, confidence, recency, type_ids)
        ):
            raise ValueError("all memory metadata must have shape [batch, memory]")
        token = self.value_projection(values)
        if identity_keys is not None:
            if identity_keys.shape != (batch, count, self.key_dimension):
                raise ValueError(
                    "identity_keys must have shape "
                    f"[batch, memory, {self.key_dimension}]"
                )
            token = token + self.identity_projection(identity_keys.to(dtype))
        token = token + self.type_embedding(type_ids)
        token = token + self.recency_projection(recency[..., None].to(dtype))
        token = token + self.confidence_projection(confidence[..., None].to(dtype))
        weight = retrieval_weights[..., None].to(dtype).clamp(0.0, 1.0)
        return token * weight * torch.sigmoid(self.gate)


class GemmaQueryProjector(nn.Module):
    """Learned frozen-Gemma question route into the visual basin key space."""

    def __init__(self, hidden_dimension: int = 640, key_dimension: int = 128) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension // 2),
            nn.GELU(),
            nn.Linear(hidden_dimension // 2, key_dimension),
        )

    def forward(self, hidden: Tensor) -> Tensor:
        return torch.nn.functional.normalize(self.projection(hidden.float()), dim=-1)


class GemmaWorldModel(nn.Module):
    """Wrap a causal LM and inject memory tokens after its first input token."""

    def __init__(self, model: nn.Module, port: LatentMemoryPort) -> None:
        super().__init__()
        self.model = model
        self.port = port
        embedding = model.get_input_embeddings()
        if embedding.embedding_dim != port.hidden_dimension:
            raise ValueError("port hidden dimension must match model embeddings")

    def prepare_inputs(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        memory_values: Tensor,
        *,
        memory_keys: Tensor | None = None,
        labels: Tensor | None = None,
        memory_mask: Tensor | None = None,
        retrieval_weights: Tensor | None = None,
        confidence: Tensor | None = None,
        recency: Tensor | None = None,
        type_ids: Tensor | None = None,
    ) -> PreparedGemmaInput:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError(
                "input_ids and attention_mask must have shape [batch, seq]"
            )
        if memory_values.shape[0] != input_ids.shape[0]:
            raise ValueError("memory batch must match input batch")
        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError("labels must match input_ids")
        memory = self.port(
            memory_values,
            identity_keys=memory_keys,
            retrieval_weights=retrieval_weights,
            confidence=confidence,
            recency=recency,
            type_ids=type_ids,
        )
        token_embeddings = self.model.get_input_embeddings()(input_ids)
        memory = memory.to(dtype=token_embeddings.dtype)
        combined = torch.cat(
            (token_embeddings[:, :1], memory, token_embeddings[:, 1:]), dim=1
        )
        if memory_mask is None:
            memory_mask = torch.ones(
                memory.shape[:2],
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
        if memory_mask.shape != memory.shape[:2]:
            raise ValueError("memory_mask must have shape [batch, memory]")
        combined_mask = torch.cat(
            (attention_mask[:, :1], memory_mask, attention_mask[:, 1:]), dim=1
        )
        combined_labels = None
        if labels is not None:
            ignored = torch.full(
                memory.shape[:2], -100, device=labels.device, dtype=labels.dtype
            )
            combined_labels = torch.cat((labels[:, :1], ignored, labels[:, 1:]), dim=1)
        return PreparedGemmaInput(
            inputs_embeds=combined,
            attention_mask=combined_mask,
            labels=combined_labels,
            memory_tokens=memory.shape[1],
        )

    def forward(self, **kwargs: Any) -> Any:
        prepared = self.prepare_inputs(**kwargs)
        return self.model(
            inputs_embeds=prepared.inputs_embeds,
            attention_mask=prepared.attention_mask,
            labels=prepared.labels,
        )

    @torch.inference_mode()
    def encode_query(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = output.hidden_states[-1]
        positions = attention_mask.long().sum(dim=1).sub(1).clamp_min(0)
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[rows, positions]

    @torch.inference_mode()
    def generate(
        self, *, generation_kwargs: dict[str, Any] | None = None, **kwargs: Any
    ) -> Tensor:
        prepared = self.prepare_inputs(**kwargs)
        options = dict(generation_kwargs or {})
        return self.model.generate(
            inputs_embeds=prepared.inputs_embeds,
            attention_mask=prepared.attention_mask,
            **options,
        )


def load_gemma_world_model(
    model_name: str = "google/gemma-3-270m-it",
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> GemmaWorldModel:
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Gemma requires the world extra: pip install -e '.[world]'"
        ) from error
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    hidden = int(model.get_input_embeddings().embedding_dim)
    return GemmaWorldModel(model, LatentMemoryPort(hidden_dimension=hidden).to(device))


def enable_upper_layer_lora(
    wrapper: GemmaWorldModel,
    *,
    rank: int = 8,
    alpha: int = 16,
) -> GemmaWorldModel:
    """Enable the plan's gated LoRA stage after adapters-only evaluation."""
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("LoRA requires PEFT from the world extra") from error
    layer_count = int(wrapper.model.config.num_hidden_layers)
    upper_layers = list(range(max(0, layer_count - 6), layer_count))
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        layers_to_transform=upper_layers,
        layers_pattern="layers",
    )
    wrapper.model = get_peft_model(wrapper.model, config)
    return wrapper
