"""Frozen Qwen3.5-4B `RepresentationAdapter` (design-v0.3.md Phase 0, SS4).

Loads Qwen3.5-4B frozen, captures one selected decoder layer's hidden state
through a targeted forward hook (SS4.3: "the measured implementation should
capture only the selected layer through a targeted hook", not
`output_hidden_states=True` for every layer), and projects it to a
deterministic D128 key via `FixedSeededProjection`. Requires the `qwen`
extra (`pip install -e '.[qwen]'`) for `transformers`/`accelerate`, which
this module only imports lazily so the rest of the package stays usable
without them - matching the existing `vision`/`stream`/`ann` extras'
lazy-import pattern in `backbone.py` and `huggingface_stream.py`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from manifoldflow.representation import (
    FixedSeededProjection,
    RepresentationAdapter,
    layer_index_for_fraction,
    pool_hidden_states,
)

_DEFAULT_MODEL_NAME = "Qwen/Qwen3.5-4B"


@dataclass(frozen=True)
class QwenRepresentationConfig:
    """Defaults are Phase 1's selected representation (SS9 Phase 1 exit
    gate, `docs/results/qwen-phase1-representation-sweep.md`): 75% depth,
    mean pooling reached 100% held-out paraphrase retrieval versus 75%
    lexical and 0% random controls, and dominated last-token pooling at
    every layer depth tried."""

    model_name: str = _DEFAULT_MODEL_NAME
    revision: str | None = None
    layer_fraction: float = 0.75
    pooling: str = "mean"
    projection_seed: int = 0
    output_dim: int = 128
    dtype: str = "bfloat16"
    device: str | None = None
    max_new_tokens: int = 128

    def __post_init__(self) -> None:
        if self.pooling not in ("last_token", "mean"):
            raise ValueError("pooling must be 'last_token' or 'mean'")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")


@dataclass
class _CapturedHiddenState:
    tensor: Tensor | None = None


class QwenRepresentationAdapter(RepresentationAdapter):
    """Frozen Qwen3.5-4B behind the modality-independent `RepresentationAdapter`."""

    def __init__(self, config: QwenRepresentationConfig | None = None) -> None:
        self.config = config or QwenRepresentationConfig()
        transformers = _require_transformers()
        device = self.config.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        dtype = getattr(torch, self.config.dtype)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.config.model_name, revision=self.config.revision
        )
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            revision=self.config.revision,
            dtype=dtype,
        ).to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        num_layers = self.model.config.num_hidden_layers
        hidden_size = self.model.config.hidden_size
        self.layer_index = layer_index_for_fraction(
            num_layers, self.config.layer_fraction
        )
        self.device = device

        self.projection = FixedSeededProjection(
            hidden_size, self.config.output_dim, seed=self.config.projection_seed
        ).to(device)

        self._captured = _CapturedHiddenState()
        target_layer = self._decoder_layers()[self.layer_index]
        self._hook_handle = target_layer.register_forward_hook(self._capture_hook)

    def _decoder_layers(self) -> Any:
        # transformers' causal-LM wrappers expose the base decoder as
        # `.model` and its layer stack as `.model.layers`; verified against
        # the installed Qwen3.5 implementation rather than assumed.
        return self.model.model.layers

    def _capture_hook(self, module: Any, inputs: Any, output: Tensor) -> None:
        self._captured.tensor = output

    @torch.no_grad()
    def encode_event(self, text: str) -> Tensor:
        messages = [{"role": "user", "content": text}]
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        self._captured.tensor = None
        self.model.model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            use_cache=False,
        )
        hidden_states = self._captured.tensor
        if hidden_states is None:
            raise RuntimeError("forward hook did not capture a hidden state")

        pooled = pool_hidden_states(
            hidden_states.float(),
            encoded["attention_mask"],
            pooling=self.config.pooling,
        )
        key = self.projection(pooled)
        return key.squeeze(0).to("cpu", dtype=torch.float32)

    @torch.no_grad()
    def generate(self, prompt: str, memories: list[str]) -> str:
        content = prompt
        if memories:
            lines = "\n".join(f"- {memory}" for memory in memories)
            content = f"<persistent_memory>\n{lines}\n</persistent_memory>\n\n{prompt}"
        messages = [{"role": "user", "content": content}]
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        output_ids = self.model.generate(
            **encoded,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=False,
        )
        generated = output_ids[0, encoded["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def clear_runtime_context(self) -> None:
        # Each `encode_event`/`generate` call runs a fresh, independent
        # forward pass with no retained KV cache on `self`, so there is no
        # per-session state to clear - satisfying design-v0.3.md SS8's
        # fresh-KV-cache isolation requirement by construction.
        self._captured.tensor = None

    def manifest(self) -> dict[str, Any]:
        chat_template = self.tokenizer.chat_template or ""
        return {
            "model_name": self.config.model_name,
            "revision": self.config.revision,
            "model_revision_hash": _short_hash(
                getattr(self.model.config, "_commit_hash", None)
                or self.config.model_name
            ),
            "chat_template_hash": _short_hash(chat_template),
            "num_hidden_layers": self.model.config.num_hidden_layers,
            "hidden_size": self.model.config.hidden_size,
            "layer_index": self.layer_index,
            "layer_fraction": self.config.layer_fraction,
            "pooling": self.config.pooling,
            "projection_seed": self.config.projection_seed,
            "output_dim": self.config.output_dim,
            "normalization": "l2",
            "dtype": self.config.dtype,
        }


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _require_transformers() -> Any:
    try:
        import transformers
    except ImportError as error:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "The Qwen representation adapter requires the qwen extra: "
            "python -m pip install -e '.[qwen]'"
        ) from error
    return transformers
