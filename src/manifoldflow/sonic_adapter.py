"""Frozen Sonic SSM `RepresentationAdapter` (design-v0.4.md's renamed and
substantially expanded Phase 7, "Sonic portability").

Design-v0.3.md's own Phase 7 exit gate was narrower than what design-v0.4
now asks for: "Implement the same representation-adapter interface for
Sonic... run the identical episodic harness... compare memory curves
independently of generation quality." This adapter satisfies exactly that
narrower claim - a real, checkpointed Sonic behind the same
`RepresentationAdapter` interface `QwenRepresentationAdapter` already
implements, reusing the same `pool_hidden_states`/`FixedSeededProjection`
primitives unchanged. Design-v0.4's own vision (Sonic trained *with*
ManifoldFlow as native recurrent state from step zero, not probed after
the fact) is a much larger undertaking - joint training infrastructure,
not an inference adapter - and is not what this file does.

Sonic is architecturally a state-space model, not a transformer: there is
no fixed-width KV cache and no discrete "decoder layer" boundary in the
attention sense, but `SonicSSMLayer.forward` still returns `(hidden,
state)` per layer exactly like a hookable transformer block, so the same
targeted-forward-hook pattern `QwenRepresentationAdapter` uses still
applies unchanged.

`docs/results/sonic-phase1-representation-sweep.md` ran the same sweep
Qwen's Phase 1 ran (same dataset, same three depth fractions, same two
pooling rules) against this real checkpoint and found the identical
qualitative pattern - mean pooling dominates last-token pooling at every
depth, and 75% depth is the single best cell - so `layer_fraction=0.75`
(hooked, matching that sweep's methodology) is this adapter's default,
not `forward_hidden`'s own final post-`final_norm` output. That extra
normalization is applied only to the last layer inside `forward_hidden`,
which would make a "final-layer" choice inconsistent with the sweep's
uniform raw-hook comparison across depths; encoding therefore hooks
`model.layers[layer_index]` directly, exactly as the sweep did.

The tokenizer (`gtr_arw_english_50k`, vendored at
`resources/tokenizers/gtr_arw_english_50k/`) has no chat template - it is
a plain pretraining tokenizer with `<|begin_of_text|>`/`<|end_of_text|>`
only, confirmed via its own `tokenizer_config.json`. `encode_event`
therefore tokenizes raw text directly rather than applying a chat
template the way Qwen's adapter does.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from manifoldflow.representation import (
    FixedSeededProjection,
    RepresentationAdapter,
    layer_index_for_fraction,
    pool_hidden_states,
)
from manifoldflow.sonic_ssm import SonicSSMConfig, SonicSSMLanguageModel

_DEFAULT_TOKENIZER_DIR = (
    Path(__file__).resolve().parents[2]
    / "resources"
    / "tokenizers"
    / "gtr_arw_english_50k"
)


@dataclass(frozen=True)
class SonicRepresentationConfig:
    """Defaults are Phase 1's selected representation
    (`docs/results/sonic-phase1-representation-sweep.md`): 75% depth,
    mean pooling reached 91.7% held-out paraphrase retrieval versus 75%
    lexical and 0% random controls, and dominated last-token pooling at
    every depth tried - matching Qwen's own Phase 1 result exactly."""

    checkpoint_path: Path
    tokenizer_dir: Path = _DEFAULT_TOKENIZER_DIR
    layer_fraction: float = 0.75
    pooling: str = "mean"
    projection_seed: int = 0
    output_dim: int = 128
    device: str | None = None
    max_new_tokens: int = 128

    def __post_init__(self) -> None:
        if self.pooling not in ("last_token", "mean"):
            raise ValueError("pooling must be 'last_token' or 'mean'")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")


class SonicRepresentationAdapter(RepresentationAdapter):
    """A real, checkpointed Sonic SSM behind `RepresentationAdapter`."""

    def __init__(self, config: SonicRepresentationConfig) -> None:
        self.config = config
        transformers = _require_transformers()
        device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._checkpoint = torch.load(
            config.checkpoint_path, map_location="cpu", weights_only=False
        )
        self.model_config = SonicSSMConfig(**self._checkpoint["model_config"])
        self.model = SonicSSMLanguageModel(self.model_config).to(device)
        self.model.load_state_dict(self._checkpoint["model"])
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(config.tokenizer_dir)
        )
        if len(self.tokenizer) != self.model_config.vocab_size:
            raise ValueError(
                f"tokenizer vocab size {len(self.tokenizer)} does not match "
                f"model vocab_size {self.model_config.vocab_size}"
            )

        self.device = device
        self.projection = FixedSeededProjection(
            self.model_config.d_model, config.output_dim, seed=config.projection_seed
        ).to(device)

        self.layer_index = layer_index_for_fraction(
            self.model_config.n_layers, config.layer_fraction
        )
        self._captured: Tensor | None = None
        self._hook_handle = self.model.layers[self.layer_index].register_forward_hook(
            self._capture_hook
        )

    def _capture_hook(self, module: Any, inputs: Any, output: Any) -> None:
        self._captured = output[0]  # SonicSSMLayer.forward returns (hidden, state)

    @torch.no_grad()
    def encode_event(self, text: str) -> Tensor:
        input_ids = self._encode_ids(text)
        self._captured = None
        self.model.forward_hidden(input_ids)
        if self._captured is None:
            raise RuntimeError("forward hook did not capture a hidden state")
        attention_mask = torch.ones_like(input_ids)
        pooled = pool_hidden_states(
            self._captured.float(), attention_mask, pooling=self.config.pooling
        )
        key = self.projection(pooled)
        return key.squeeze(0).to("cpu", dtype=torch.float32)

    @torch.no_grad()
    def generate(self, prompt: str, memories: list[str]) -> str:
        content = prompt
        if memories:
            lines = "\n".join(f"- {memory}" for memory in memories)
            content = f"<persistent_memory>\n{lines}\n</persistent_memory>\n\n{prompt}"
        input_ids = self._encode_ids(content)

        hidden, states = self.model.forward_hidden(input_ids)
        logits = self.model.logits_from_hidden(hidden[:, -1:, :])
        generated: list[int] = []
        next_id = int(logits[0, -1].argmax().item())
        eos_id = self.tokenizer.eos_token_id
        for _ in range(self.config.max_new_tokens):
            if eos_id is not None and next_id == eos_id:
                break
            generated.append(next_id)
            step_input = torch.tensor([[next_id]], device=self.device)
            hidden, states = self.model.forward_hidden(step_input, states)
            logits = self.model.logits_from_hidden(hidden)
            next_id = int(logits[0, -1].argmax().item())
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def clear_runtime_context(self) -> None:
        # Each encode_event/generate call starts from states=None (a fresh
        # prefill), so there is no retained recurrent state on `self` to
        # clear - the SSM analogue of Qwen adapter's fresh-KV-cache note.
        pass

    def manifest(self) -> dict[str, Any]:
        return {
            "architecture": "sonic_ssm",
            "checkpoint_path": str(self.config.checkpoint_path),
            "checkpoint_format": self._checkpoint.get("format"),
            "checkpoint_update": self._checkpoint.get("update"),
            "checkpoint_tokens_seen": self._checkpoint.get("tokens_seen"),
            "tokenizer_dir": str(self.config.tokenizer_dir),
            "tokenizer_sha256": self._checkpoint.get("tokenizer", {}).get("sha256"),
            "d_model": self.model_config.d_model,
            "n_layers": self.model_config.n_layers,
            "layer_index": self.layer_index,
            "layer_fraction": self.config.layer_fraction,
            "pooling": self.config.pooling,
            "projection_seed": self.config.projection_seed,
            "output_dim": self.config.output_dim,
            "normalization": "l2",
        }

    def _encode_ids(self, text: str) -> Tensor:
        ids = self.tokenizer.encode(text, add_special_tokens=True)
        return torch.tensor([ids], device=self.device)


def _require_transformers() -> Any:
    try:
        import transformers
    except ImportError as error:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "The Sonic representation adapter requires the qwen extra "
            "(shared tokenizer dependency): python -m pip install -e '.[qwen]'"
        ) from error
    return transformers
