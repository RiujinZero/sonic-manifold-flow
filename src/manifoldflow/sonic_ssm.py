"""Sonic sparse-mode language model built around its custom SSM block.

The recurrent block is extracted from ``Sonic v1.4 CPP/hedgehog_core.py``.
Sonic's token-local sparse mode gate is retained, while persistent simplex
memory, geometry retrieval, semantic-boundary routing, and all external memory
systems are intentionally excluded.  The surrounding lexical shell uses the
same factorized, tied vocabulary design as the production Gator run so the 50K
tokenizer does not consume the entire parameter budget.

---

Vendored verbatim (byte-identical, diffed at copy time) from
``GTR-sonic-ssm/gator/models/sonic_ssm.py`` in the separate
``C:\\Users\\user\\Documents\\projects\\GTR-sonic-ssm`` repo,
per design-v0.4's Phase 7 (design-v0.3.md's "Sonic portability" renamed
and substantially expanded): this project reads Sonic's hidden states as
`RepresentationAdapter`'s modality-independent interface (`representation.py`)
requires; it does not modify or re-derive Sonic's own architecture. Do not
hand-edit this file - if the upstream model changes, re-copy it rather than
patching a fork here. The only external dependency (`gator_linear.
_hgrn_compiled.compiled_chunk_hgrn`, used solely by the non-default
`scan_backend="hgrn"` Triton path this project never selects) is vendored
alongside it at `src/gator_linear/`, a deliberately minimal subset of the
upstream `gator_linear` package - see that package's own docstring.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from gator_linear._hgrn_compiled import compiled_chunk_hgrn


@dataclass(frozen=True)
class SonicSSMConfig:
    """Configuration for the Sonic sparse-mode SSM language model."""

    vocab_size: int = 50_000
    d_model: int = 1_504
    lexical_dim: int = 256
    n_layers: int = 29
    state_dim: int = 64
    n_heads: int = 1
    num_modes: int = 120
    sparsity_target: float = 0.78
    gate_residual: bool = True
    dense_a: bool = False
    dt_min: float = 0.001
    dt_max: float = 0.1
    activation_checkpointing: bool = True
    checkpoint_every: int = 1
    # Read/write external memory bank (disabled when memory_num_slots == 0).
    # Cross-attention over a small fixed-size bank of memory_num_slots slots
    # rather than over the growing sequence, hooked in once after layer
    # memory_layer_index (0-indexed) so it stays O(seq_len * num_slots) --
    # constant in context length -- instead of reintroducing an O(n^2) /
    # growing-KV-cache cost.
    memory_num_slots: int = 0
    memory_slot_dim: int = 8
    memory_layer_index: Optional[int] = None
    # Very negative by default so the gate starts closed enough to be a
    # near-no-op when grafting onto an already-trained checkpoint. Training
    # fully from scratch has no such checkpoint to protect, so it's fine
    # (often better) to open this up, e.g. to -1.0 or 0.0.
    memory_gate_init_bias: float = -4.0
    # False (default): keys/initial content/read-out/write-candidate are
    # frozen random buffers -- only the gate is trainable, so gradient never
    # gets a chance to shape *what* gets written, only *whether*. True:
    # those become ordinary trainable parameters, shaped by the plain
    # end-task loss like everything else in the model (still no auxiliary
    # loss on the memory content itself).
    memory_trainable_content: bool = False
    # False (default): writes/reads go through separate learned/frozen
    # projections into a memory_slot_dim-wide bottleneck (typically much
    # narrower than d_model), which the model has to learn to make
    # meaningful from scratch. True: skip those projections entirely --
    # the memory holds the hidden state's own representation directly
    # (requires memory_slot_dim == d_model). No compress/decompress step,
    # no separately-learned content space to get right.
    memory_identity_content: bool = False

    def __post_init__(self) -> None:
        if self.vocab_size <= 1:
            raise ValueError("vocab_size must be greater than one")
        if self.d_model <= 0 or self.lexical_dim <= 0:
            raise ValueError("model and lexical widths must be positive")
        if self.n_layers <= 0 or self.state_dim <= 0:
            raise ValueError("layer count and state width must be positive")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be evenly divisible by n_heads")
        if self.state_dim % self.n_heads != 0:
            raise ValueError("state_dim must be evenly divisible by n_heads")
        if self.dense_a and self.n_heads != 1:
            raise ValueError("multi-head splitting is only supported for the diagonal (non-dense) transition")
        if self.num_modes <= 0:
            raise ValueError("num_modes must be positive")
        if not 0 <= self.sparsity_target < 1:
            raise ValueError("sparsity_target must be in [0, 1)")
        if not 0 < self.dt_min <= self.dt_max:
            raise ValueError("expected 0 < dt_min <= dt_max")
        if self.checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive")
        if self.memory_num_slots < 0:
            raise ValueError("memory_num_slots cannot be negative")
        if self.memory_num_slots > 0:
            if self.memory_slot_dim <= 0:
                raise ValueError("memory_slot_dim must be positive when memory is enabled")
            if self.memory_layer_index is None:
                raise ValueError("memory_layer_index is required when memory is enabled")
            if not 0 <= self.memory_layer_index < self.n_layers:
                raise ValueError("memory_layer_index must index an existing layer")
            if self.memory_identity_content and self.memory_slot_dim != self.d_model:
                raise ValueError("memory_identity_content requires memory_slot_dim == d_model")
        elif self.memory_layer_index is not None:
            raise ValueError("memory_layer_index is only meaningful when memory_num_slots > 0")

    def to_dict(self) -> dict:
        return asdict(self)


def estimated_sonic_ssm_parameter_count(config: SonicSSMConfig) -> int:
    """Return the exact parameter count without allocating the model."""

    d_model = config.d_model
    state_dim = config.state_dim
    lexical = config.vocab_size * config.lexical_dim
    lexical += (
        0
        if config.lexical_dim == d_model
        else config.lexical_dim * d_model
    )
    lexical += config.vocab_size
    transition = state_dim * state_dim if config.dense_a else state_dim
    n_heads = config.n_heads
    block = (
        2 * d_model  # pre-SSM LayerNorm
        + d_model * d_model  # input projection (always dense, shared across heads)
        + 3 * d_model * state_dim // n_heads  # B, C, and delta projections (block-diagonal across heads)
        + state_dim  # delta bias
        + state_dim * d_model // n_heads  # output projection (block-diagonal across heads)
        + d_model  # learned residual skip
        + transition
    )
    sparse_gate = (
        2 * d_model * config.num_modes  # selector and mode-value projection
        + config.num_modes  # learned per-mode scale
        + 1  # learned temperature
    )
    final_norm = 2 * d_model
    # The write gate is always trainable. Keys/initial content are
    # trainable too when memory_trainable_content=True, else they're fixed
    # buffers -- excluded here either way, matching model.parameter_count
    # (which sums nn.Module.parameters() and so also excludes buffers).
    # Read-out/write-candidate projections don't exist at all in
    # memory_identity_content mode (the hidden state is used directly).
    memory = 0
    if config.memory_num_slots > 0:
        memory = d_model + 1
        if config.memory_trainable_content:
            k, slot_dim = config.memory_num_slots, config.memory_slot_dim
            memory += k * d_model + k * slot_dim  # keys + initial content
            if not config.memory_identity_content:
                memory += 2 * d_model * slot_dim  # read-out and write-candidate
    return lexical + config.n_layers * (block + sparse_gate) + final_norm + memory


def sonic_ssm_memory_buffer_element_count(config: SonicSSMConfig) -> int:
    """Return the element count of the memory block's fixed (untrained) buffers."""

    if config.memory_num_slots <= 0 or config.memory_trainable_content:
        return 0
    k, slot_dim, d_model = config.memory_num_slots, config.memory_slot_dim, config.d_model
    if config.memory_identity_content:
        return k * d_model + k * slot_dim  # keys + initial content only
    return (
        k * d_model  # static per-slot address keys
        + k * slot_dim  # fixed initial memory content
        + d_model * slot_dim  # fixed read-out projection
        + slot_dim * d_model  # fixed write-candidate projection
    )


class SonicSparseGate(nn.Module):
    """Sonic's token-local straight-through top-K mode selector.

    The gate chooses a sparse set of learned modes independently for every
    token. During training, its forward values remain hard top-K while its
    gradients follow a temperature-controlled softmax. This is the original
    Sonic ``num_modes`` mechanism; it is not the persistent simplex-memory
    subsystem.
    """

    def __init__(self, config: SonicSSMConfig) -> None:
        super().__init__()
        self.num_modes = int(config.num_modes)
        self.sparsity_target = float(config.sparsity_target)
        self.mode_selector = nn.Linear(
            config.d_model,
            config.num_modes,
            bias=False,
        )
        self.mode_scale = nn.Parameter(torch.ones(config.num_modes))
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.mode_value = nn.Linear(
            config.num_modes,
            config.d_model,
            bias=False,
        )

        nn.init.orthogonal_(self.mode_selector.weight)
        with torch.no_grad():
            self.mode_value.weight.copy_(self.mode_selector.weight.t())

    @property
    def active_modes(self) -> int:
        return max(1, int(self.num_modes * (1.0 - self.sparsity_target)))

    def mode_activations(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return the hard/ST sparse mode activations used by ``forward``."""

        scores = self.mode_selector(hidden)
        top_indices = scores.topk(self.active_modes, dim=-1).indices
        hard = torch.zeros_like(scores).scatter_(-1, top_indices, 1.0)
        if self.training:
            soft = torch.softmax(scores / self.temperature.clamp(min=0.1), dim=-1)
            selected = hard - soft.detach() + soft
        else:
            selected = hard
        return selected * self.mode_scale

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.mode_value(self.mode_activations(hidden))


class _ChunkedLinearCrossEntropy(torch.autograd.Function):
    """Tiled tied-readout cross entropy without a full ``[tokens, vocab]``.

    This is Sonic's chunked linear-CE path, extended with the production
    vocabulary bias.  Forward and backward hold at most one vocabulary tile.
    """

    @staticmethod
    def forward(ctx, hidden, weight, bias, labels, chunk_rows, ignore_index):
        valid_count = (labels != ignore_index).sum().clamp(min=1)
        total = hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, hidden.shape[0], int(chunk_rows)):
            chunk_hidden = hidden[start : start + int(chunk_rows)]
            chunk_labels = labels[start : start + int(chunk_rows)]
            logits = F.linear(
                chunk_hidden.float(),
                weight.float(),
                bias.float(),
            )
            log_normalizer = torch.logsumexp(logits, dim=-1)
            correct = logits.gather(
                1, chunk_labels.clamp(min=0).unsqueeze(1)
            ).squeeze(1)
            mask = (chunk_labels != ignore_index).to(logits.dtype)
            total = total + ((log_normalizer - correct) * mask).sum()
        ctx.save_for_backward(hidden, weight, bias, labels)
        ctx.chunk_rows = int(chunk_rows)
        ctx.ignore_index = int(ignore_index)
        ctx.valid_count = valid_count
        return total / valid_count.to(total.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        hidden, weight, bias, labels = ctx.saved_tensors
        scale = (grad_output / ctx.valid_count.to(grad_output.dtype)).float()
        grad_hidden = torch.zeros_like(hidden)
        grad_weight = torch.zeros_like(weight, dtype=torch.float32)
        grad_bias = torch.zeros_like(bias, dtype=torch.float32)
        for start in range(0, hidden.shape[0], ctx.chunk_rows):
            chunk_hidden = hidden[start : start + ctx.chunk_rows]
            chunk_labels = labels[start : start + ctx.chunk_rows]
            logits = F.linear(
                chunk_hidden.float(),
                weight.float(),
                bias.float(),
            )
            probabilities = torch.softmax(logits, dim=-1)
            clamped = chunk_labels.clamp(min=0).unsqueeze(1)
            probabilities.scatter_add_(
                1,
                clamped,
                torch.full_like(clamped, -1.0, dtype=probabilities.dtype),
            )
            probabilities *= (
                chunk_labels != ctx.ignore_index
            ).to(probabilities.dtype).unsqueeze(1)
            probabilities *= scale
            grad_hidden[start : start + ctx.chunk_rows] = (
                probabilities @ weight.float()
            ).to(grad_hidden.dtype)
            grad_weight += probabilities.transpose(0, 1) @ chunk_hidden.float()
            grad_bias += probabilities.sum(dim=0)
        return (
            grad_hidden,
            grad_weight.to(weight.dtype),
            grad_bias.to(bias.dtype),
            None,
            None,
            None,
        )


class SonicSSMBlock(nn.Module):
    """Sonic's input-selective diagonal state-space recurrence.

    For each token ``t``::

        delta_t = softplus(W_delta x_t)
        a_t     = exp(delta_t * A)
        b_t     = delta_t * W_B x_t
        state_t = a_t * state_(t-1) + b_t
        y_t     = W_out((W_C x_t) * state_t) + D * x_t

    The diagonal recurrence is evaluated by Sonic's associative doubling scan
    during prefill/training and naturally becomes a single recurrent update
    during cached decoding.
    """

    def __init__(self, config: SonicSSMConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.state_dim = config.state_dim
        self.dense_a = config.dense_a
        self.n_heads = int(config.n_heads)
        self.scan_backend = "doubling"
        self.packed_projections = False

        self.norm = nn.LayerNorm(config.d_model)
        self.in_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        if self.n_heads == 1:
            self.b_proj = nn.Linear(config.d_model, config.state_dim, bias=False)
            self.c_proj = nn.Linear(config.d_model, config.state_dim, bias=False)
            self.dt_proj = nn.Linear(config.d_model, config.state_dim, bias=True)
            self.out_proj = nn.Linear(config.state_dim, config.d_model, bias=False)
        else:
            # Block-diagonal (grouped 1x1-conv) heads: each of the n_heads
            # groups reads only its own d_model/n_heads channel slice and
            # writes only its own state_dim/n_heads state slice -- a genuine
            # multi-head SSM (Mamba2-style) rather than one shared dense
            # state_dim-wide recurrence. in_proj/norm/skip stay dense so
            # heads still mix through the surrounding residual stream.
            self.b_proj = nn.Conv1d(
                config.d_model, config.state_dim, kernel_size=1,
                groups=self.n_heads, bias=False,
            )
            self.c_proj = nn.Conv1d(
                config.d_model, config.state_dim, kernel_size=1,
                groups=self.n_heads, bias=False,
            )
            self.dt_proj = nn.Conv1d(
                config.d_model, config.state_dim, kernel_size=1,
                groups=self.n_heads, bias=True,
            )
            self.out_proj = nn.Conv1d(
                config.state_dim, config.d_model, kernel_size=1,
                groups=self.n_heads, bias=False,
            )
        self.skip = nn.Parameter(torch.ones(config.d_model))

        if config.dense_a:
            initial = -torch.eye(config.state_dim)
            initial = initial + 0.01 * torch.randn_like(initial)
            self.a_matrix = nn.Parameter(initial)
            self.register_parameter("log_a", None)
        else:
            initial = torch.linspace(
                math.log(config.dt_max),
                math.log(config.dt_min),
                config.state_dim,
            )
            self.log_a = nn.Parameter(initial)
            self.register_parameter("a_matrix", None)

        for projection in (
            self.in_proj,
            self.b_proj,
            self.c_proj,
            self.dt_proj,
            self.out_proj,
        ):
            nn.init.xavier_uniform_(projection.weight)
        dt_initial = torch.exp(
            torch.rand(config.state_dim)
            * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        )
        with torch.no_grad():
            self.dt_proj.bias.copy_(torch.log(torch.expm1(dt_initial)))

    def _project(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Apply a projection that is a Linear (n_heads=1) or a grouped
        1x1 Conv1d (n_heads>1, which expects [batch, channels, sequence])."""

        if self.n_heads == 1:
            return layer(x)
        return layer(x.transpose(1, 2)).transpose(1, 2)

    def transition(self) -> torch.Tensor:
        if self.dense_a:
            assert self.a_matrix is not None
            return self.a_matrix
        assert self.log_a is not None
        return -torch.exp(self.log_a)

    def configure_execution(
        self,
        *,
        scan_backend: str = "doubling",
        packed_projections: bool = False,
    ) -> None:
        """Select checkpoint-neutral execution optimizations.

        These controls deliberately are not model parameters or buffers, so
        they do not change checkpoint keys, parameter counts, or model
        semantics.  The HGRN kernel has no initial-state gradient; reject that
        use explicitly instead of silently truncating a differentiable carry.
        """

        if scan_backend not in {"doubling", "hgrn"}:
            raise ValueError("scan_backend must be 'doubling' or 'hgrn'")
        if scan_backend == "hgrn" and self.dense_a:
            raise ValueError("the HGRN scan backend requires a diagonal transition")
        if packed_projections and self.n_heads != 1:
            raise ValueError("packed projections currently require n_heads=1")
        self.scan_backend = scan_backend
        self.packed_projections = bool(packed_projections)

    @staticmethod
    def parallel_diagonal_scan(
        decay: torch.Tensor,
        update: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> torch.Tensor:
        """Inclusive associative scan for ``s_t = decay_t*s_(t-1)+update_t``."""

        if decay.shape != update.shape or decay.ndim != 3:
            raise ValueError("decay and update must share [batch, sequence, state]")
        if initial_state.shape != (decay.shape[0], decay.shape[2]):
            raise ValueError("initial_state must have shape [batch, state]")
        if decay.shape[1] == 0:
            return update

        update = torch.cat(
            [
                decay[:, :1] * initial_state.unsqueeze(1) + update[:, :1],
                update[:, 1:],
            ],
            dim=1,
        )
        sequence_length = decay.shape[1]
        shift = 1
        while shift < sequence_length:
            previous_decay = decay[:, :-shift]
            previous_update = update[:, :-shift]
            tail_decay = decay[:, shift:]
            update = torch.cat(
                [
                    update[:, :shift],
                    update[:, shift:] + tail_decay * previous_update,
                ],
                dim=1,
            )
            decay = torch.cat(
                [decay[:, :shift], tail_decay * previous_decay], dim=1
            )
            shift *= 2
        return update

    def forward(
        self,
        hidden: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden.ndim != 3 or hidden.shape[-1] != self.d_model:
            raise ValueError("hidden must have shape [batch, sequence, d_model]")
        residual = hidden
        projected = self.in_proj(self.norm(hidden))
        batch, sequence_length, _ = projected.shape
        if sequence_length == 0:
            if state is None:
                state = torch.zeros(
                    batch,
                    self.state_dim,
                    device=hidden.device,
                    dtype=torch.float32,
                )
            return residual, state

        if self.packed_projections:
            packed_weight = torch.cat(
                (self.b_proj.weight, self.c_proj.weight, self.dt_proj.weight),
                dim=0,
            )
            packed = F.linear(projected, packed_weight)
            write, read, delta_logits = packed.split(self.state_dim, dim=-1)
            delta = F.softplus(delta_logits + self.dt_proj.bias)
        else:
            write = self._project(self.b_proj, projected)
            read = self._project(self.c_proj, projected)
            delta = F.softplus(self._project(self.dt_proj, projected))
        transition = self.transition()

        if self.dense_a:
            assert transition.ndim == 2
            if state is None:
                state = torch.zeros(
                    batch,
                    self.state_dim,
                    device=hidden.device,
                    dtype=transition.dtype,
                )
            outputs = []
            identity = torch.eye(
                self.state_dim,
                device=hidden.device,
                dtype=transition.dtype,
            ).unsqueeze(0)
            for token_index in range(sequence_length):
                token_delta = delta[:, token_index].to(transition.dtype)
                discrete_a = identity + token_delta.unsqueeze(-1) * transition
                state = (
                    torch.bmm(state.to(transition.dtype).unsqueeze(1), discrete_a)
                    .squeeze(1)
                    + token_delta * write[:, token_index].to(transition.dtype)
                )
                token_read = read[:, token_index].to(state.dtype) * state
                outputs.append(
                    self._project(self.out_proj, token_read.unsqueeze(1))
                )
            ssm_output = torch.cat(outputs, dim=1)
        else:
            assert transition.ndim == 1
            log_decay = delta.to(transition.dtype) * transition.view(1, 1, -1)
            update = delta.to(log_decay.dtype) * write.to(log_decay.dtype)
            if state is None:
                state = torch.zeros(
                    batch,
                    self.state_dim,
                    device=hidden.device,
                    dtype=log_decay.dtype,
                )
            state = state.to(log_decay.dtype)
            if self.scan_backend == "hgrn":
                if state.requires_grad:
                    raise RuntimeError(
                        "the HGRN scan backend cannot differentiate through "
                        "a carried initial state; use scan_backend='doubling' "
                        "for truncated-BPTT"
                    )
                states, final_state = compiled_chunk_hgrn(
                    update,
                    log_decay,
                    initial_state=state,
                    output_final_state=True,
                )
                assert final_state is not None
                state = final_state
            else:
                states = self.parallel_diagonal_scan(
                    torch.exp(log_decay), update, state
                )
                state = states[:, -1]
            ssm_output = self._project(self.out_proj, read.to(states.dtype) * states)

        skipped = self.skip.view(1, 1, -1) * projected
        return residual + ssm_output + skipped, state


class SonicReadWriteMemory(nn.Module):
    """Cross-attention read/write over a small, fixed-size external memory bank.

    Every token attends over ``num_slots`` fixed memory slots (not over the
    growing sequence), so cost is ``O(seq_len * num_slots)`` -- constant in
    context length, unlike a growing KV-cache. Read and write share the same
    softmax attention weights, computed from the token's hidden state against
    a set of per-slot keys. The write is a gated blend (GRU/NTM-style
    erase-add via the same diagonal associative scan the SSM block itself
    uses) rather than a hard overwrite: ``memory_t = (1 -
    attention*gate)*memory_{t-1} + (attention*gate)*candidate_t``.

    By default (``memory_trainable_content=False``) the block itself is not
    trainable -- keys, the initial memory content, the read-out projection,
    and the write-candidate projection are all fixed random buffers, not
    ``nn.Parameter``s, so they never appear in ``model.parameters()`` and
    never reach the optimizer. Only ``write_gate`` is a real parameter:
    gradient descent decides *when* to write into (and read out of) an
    otherwise-fixed random substrate, not what the substrate itself looks
    like. When ``memory_trainable_content=True``, keys/initial content
    become ordinary trainable parameters instead -- gradient descent gets a
    chance to shape *what* gets written too, via the plain end-task loss
    (still no auxiliary loss on the memory content itself; that's a
    separate, bigger change).

    ``memory_identity_content=True`` goes further: there is no separate
    read-out/write-candidate projection at all (requires
    ``memory_slot_dim == d_model``) -- the memory holds the hidden state's
    own representation directly, read back out directly. No lossy
    compress-then-decompress step, no separately-learned content space that
    has to be made meaningful from scratch.

    The write gate's bias starts very negative either way so the gate
    starts near zero -- the block begins near a no-op on a checkpoint that
    never had it, then eases in during training as the gate learns to open.
    """

    def __init__(self, config: SonicSSMConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.num_slots = int(config.memory_num_slots)
        self.slot_dim = int(config.memory_slot_dim)
        self.trainable_content = bool(config.memory_trainable_content)
        self.identity_content = bool(config.memory_identity_content)

        keys = torch.empty(self.num_slots, config.d_model)
        nn.init.normal_(keys, std=0.02)
        memory_init = torch.zeros(self.num_slots, self.slot_dim)

        if self.trainable_content:
            self.keys = nn.Parameter(keys)
            self.memory_init = nn.Parameter(memory_init)
        else:
            self.register_buffer("keys", keys)
            self.register_buffer("memory_init", memory_init)

        if not self.identity_content:
            read_out_weight = torch.empty(config.d_model, self.slot_dim)
            write_candidate_weight = torch.empty(self.slot_dim, config.d_model)
            nn.init.xavier_uniform_(write_candidate_weight)
            if self.trainable_content:
                # Zero-init read-out: with the content trainable, this is a
                # genuine "starts as a no-op, gradient decides if it's
                # worth opening" init rather than a permanent dead end
                # (unlike the frozen case, where a zero read-out could
                # never move).
                self.read_out_weight = nn.Parameter(torch.zeros_like(read_out_weight))
                self.write_candidate_weight = nn.Parameter(write_candidate_weight)
            else:
                nn.init.xavier_uniform_(read_out_weight)
                self.register_buffer("read_out_weight", read_out_weight)
                self.register_buffer("write_candidate_weight", write_candidate_weight)

        self.write_gate = nn.Linear(config.d_model, 1, bias=True)
        nn.init.xavier_uniform_(self.write_gate.weight)
        with torch.no_grad():
            self.write_gate.bias.fill_(config.memory_gate_init_bias)

    def forward(
        self,
        hidden: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden.ndim != 3 or hidden.shape[-1] != self.d_model:
            raise ValueError("hidden must have shape [batch, sequence, d_model]")
        batch, sequence_length, _ = hidden.shape
        if state is None:
            state = self.memory_init.unsqueeze(0).expand(batch, -1, -1)
        if sequence_length == 0:
            return hidden, state

        scores = torch.einsum("bsd,kd->bsk", hidden, self.keys) / math.sqrt(self.d_model)
        attention = torch.softmax(scores, dim=-1)

        candidate = hidden if self.identity_content else F.linear(hidden, self.write_candidate_weight)
        gate = torch.sigmoid(self.write_gate(hidden))
        erase = attention * gate

        decay = (1.0 - erase).unsqueeze(-1).expand(-1, -1, -1, self.slot_dim)
        update = erase.unsqueeze(-1) * candidate.unsqueeze(2)
        decay_flat = decay.reshape(batch, sequence_length, self.num_slots * self.slot_dim)
        update_flat = update.reshape(batch, sequence_length, self.num_slots * self.slot_dim)
        initial_flat = state.reshape(batch, self.num_slots * self.slot_dim)

        memory_states_flat = SonicSSMBlock.parallel_diagonal_scan(
            decay_flat, update_flat, initial_flat.to(decay_flat.dtype)
        )
        memory_states = memory_states_flat.reshape(
            batch, sequence_length, self.num_slots, self.slot_dim
        )

        read = torch.einsum("bsk,bskd->bsd", attention.to(memory_states.dtype), memory_states)
        # Stashed (not returned) so a training loop can attach an auxiliary
        # loss directly to what the memory actually produced at each
        # position, without changing this module's forward signature or
        # threading an extra return value through every caller.
        self.last_read = read
        self.last_attention = attention
        read_contribution = (
            read.to(hidden.dtype) if self.identity_content
            else F.linear(read.to(hidden.dtype), self.read_out_weight)
        )
        new_hidden = hidden + read_contribution
        new_state = memory_states[:, -1]
        return new_hidden, new_state


class SonicSSMLayer(nn.Module):
    """Original Sonic ordering: sparse mode modulation, then recurrent block."""

    def __init__(self, config: SonicSSMConfig) -> None:
        super().__init__()
        self.sparse_gate = SonicSparseGate(config)
        self.ssm = SonicSSMBlock(config)
        self.gate_residual = bool(config.gate_residual)
        self._ssm_call = self.ssm.forward

    def compile_ssm(self, *, mode: str = "default") -> None:
        """Compile only the recurrent block, leaving sparse top-K eager."""

        self._ssm_call = torch.compile(
            self.ssm.forward,
            mode=mode,
            fullgraph=False,
            dynamic=False,
        )

    def compile_layer(self, *, mode: str = "default") -> None:
        """Compile sparse routing and recurrence as one full graph."""

        self._ssm_call = self.ssm.forward
        self.forward = torch.compile(  # type: ignore[method-assign]
            self.forward,
            mode=mode,
            fullgraph=True,
            dynamic=False,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mode_update = self.sparse_gate(hidden)
        gated = hidden + mode_update if self.gate_residual else mode_update
        return self._ssm_call(gated, state)


class SonicSSMLanguageModel(nn.Module):
    """A tied-vocabulary causal LM with Sonic sparse modes and SSM blocks."""

    def __init__(self, config: SonicSSMConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.lexical_dim)
        if config.lexical_dim == config.d_model:
            self.lexical_projection = nn.Identity()
        else:
            self.lexical_projection = nn.Linear(
                config.lexical_dim, config.d_model, bias=False
            )
        self.layers = nn.ModuleList(
            SonicSSMLayer(config) for _ in range(config.n_layers)
        )
        self.memory = (
            SonicReadWriteMemory(config) if config.memory_num_slots > 0 else None
        )
        self.memory_layer_index = config.memory_layer_index
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.gradient_checkpointing = bool(config.activation_checkpointing)
        self.checkpoint_every = int(config.checkpoint_every)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        if isinstance(self.lexical_projection, nn.Linear):
            nn.init.orthogonal_(self.lexical_projection.weight)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def lexical_parameter_count(self) -> int:
        projection = (
            self.lexical_projection.weight.numel()
            if isinstance(self.lexical_projection, nn.Linear)
            else 0
        )
        return self.token_embedding.weight.numel() + projection + self.lm_bias.numel()

    def compile_ssm_blocks(self, *, mode: str = "default") -> None:
        """Compile each recurrent block without changing checkpoint key names."""

        for layer in self.layers:
            layer.compile_ssm(mode=mode)

    def configure_execution(
        self,
        *,
        scan_backend: str = "doubling",
        packed_projections: bool = False,
    ) -> None:
        """Configure checkpoint-neutral execution choices on every layer."""

        for layer in self.layers:
            layer.ssm.configure_execution(
                scan_backend=scan_backend,
                packed_projections=packed_projections,
            )
            layer._ssm_call = layer.ssm.forward

    def compile_layers(self, *, mode: str = "default") -> None:
        """Compile every complete Sonic layer without changing state keys."""

        for layer in self.layers:
            layer.compile_layer(mode=mode)

    def gradient_checkpointing_enable(self, every: int = 1) -> None:
        if every <= 0:
            raise ValueError("checkpoint interval must be positive")
        self.gradient_checkpointing = True
        self.checkpoint_every = int(every)

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    def forward_hidden(
        self,
        input_ids: torch.Tensor,
        states: Optional[tuple[Optional[torch.Tensor], ...]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        expected_states = len(self.layers) + (1 if self.memory is not None else 0)
        if states is None:
            states = (None,) * expected_states
        if len(states) != expected_states:
            raise ValueError(
                "one recurrent state is required per SSM layer, plus one more "
                "for the memory block when it is enabled"
            )
        layer_states = states[: len(self.layers)]
        memory_state = states[-1] if self.memory is not None else None

        hidden = self.lexical_projection(self.token_embedding(input_ids))
        next_states: list[torch.Tensor] = []
        checkpoint_layers = (
            self.gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
        )
        for index, (layer, state) in enumerate(
            zip(self.layers, layer_states, strict=True)
        ):
            if checkpoint_layers and index % self.checkpoint_every == 0:
                hidden, next_state = checkpoint(
                    layer,
                    hidden,
                    state,
                    use_reentrant=False,
                )
            else:
                hidden, next_state = layer(hidden, state)
            next_states.append(next_state)
            if self.memory is not None and index == self.memory_layer_index:
                hidden, memory_state = self.memory(hidden, memory_state)
        hidden = self.final_norm(hidden)
        if self.memory is not None:
            next_states.append(memory_state)
        return hidden, tuple(next_states)

    def lexical_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        if isinstance(self.lexical_projection, nn.Linear):
            return F.linear(hidden, self.lexical_projection.weight.t())
        return hidden

    def logits_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(
            self.lexical_hidden(hidden),
            self.token_embedding.weight,
            self.lm_bias,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        states: Optional[tuple[Optional[torch.Tensor], ...]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        hidden, next_states = self.forward_hidden(input_ids, states)
        return self.logits_from_hidden(hidden), next_states

    def chunked_cross_entropy(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor,
        *,
        chunk_rows: int = 256,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        if chunk_rows <= 0:
            raise ValueError("chunk_rows must be positive")
        if labels.shape != hidden.shape[:2]:
            raise ValueError("labels must match hidden batch and sequence dimensions")
        lexical = self.lexical_hidden(hidden).reshape(-1, self.config.lexical_dim)
        return _ChunkedLinearCrossEntropy.apply(
            lexical,
            self.token_embedding.weight,
            self.lm_bias,
            labels.reshape(-1).long(),
            int(chunk_rows),
            int(ignore_index),
        )


__all__ = [
    "SonicReadWriteMemory",
    "SonicSparseGate",
    "SonicSSMBlock",
    "SonicSSMConfig",
    "SonicSSMLayer",
    "SonicSSMLanguageModel",
    "estimated_sonic_ssm_parameter_count",
    "sonic_ssm_memory_buffer_element_count",
]
