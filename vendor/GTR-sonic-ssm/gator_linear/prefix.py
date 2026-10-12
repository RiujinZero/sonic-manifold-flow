"""Simultaneous modular-prefix task shell for the linear-scaling Gator."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch
from torch import Tensor, nn

from .core import (
    GatorLinearBlockCache,
    GatorLinearConfig,
    GatorLinearCore,
    detach_block_cache,
    fixed_address_code,
)
from gator.memory import SurpriseGatedMemory, SurpriseMemoryConfig
from gator.memory.block_predictor import BlockExpectationPredictor
from gator.memory.block_summarizer import BlockSummarizer
from gator.memory.reliability import (
    combined_reliability,
    consensus_reliability,
    entropy_reliability,
    quality_reliability,
)
from gator.memory.retrieval import StreamMemoryInjector
from gator.memory.runtime import MemoryRuntime
from gator.memory.surprise import (
    SurpriseNormalizer,
    combine_surprise,
    context_continuity,
    predictive_surprise,
    semantic_novelty,
    stream_disagreement,
    surprise_momentum_update,
)


@dataclass(frozen=True)
class GatorLinearPrefixConfig:
    """Task-level settings wrapping the linear core."""

    core: GatorLinearConfig = field(default_factory=GatorLinearConfig)
    # Surprise-Gated Block Memory (see Papers/GTRL_PROCOIM_TRACE.md and the
    # SGBM design spec it links). Disabled by default: when None, no memory
    # submodules are constructed and forward_cached behaves exactly as
    # before this integration existed.
    memory: SurpriseMemoryConfig | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.core, GatorLinearConfig):
            raise TypeError("core must be a GatorLinearConfig")
        if self.core.vocab_size < 16:
            raise ValueError("vocab_size must be at least 16 for the prefix task")
        if self.memory is not None and not isinstance(self.memory, SurpriseMemoryConfig):
            raise TypeError("memory must be a SurpriseMemoryConfig or None")
        if self.memory is not None and self.memory.fused_dim != self.core.width:
            raise ValueError(
                "memory.fused_dim must equal core.width (the memory branch "
                "reads/writes the same width as the model's stream state)"
            )

    @property
    def vocab_size(self) -> int:
        return self.core.vocab_size


class GatorLinearPrefixModel(nn.Module):
    """Form every modular-prefix output in one simultaneous linear pass."""

    def __init__(self, config: GatorLinearPrefixConfig) -> None:
        super().__init__()
        if not isinstance(config, GatorLinearPrefixConfig):
            raise TypeError("config must be a GatorLinearPrefixConfig")
        self.config = config
        core_config = config.core
        width = core_config.width

        self.token_embedding = nn.Embedding(core_config.vocab_size, width)
        self.output_query = nn.Parameter(torch.zeros(width))
        self.slot_type = nn.Embedding(2, width)
        self.address_projection = nn.Linear(
            core_config.address_width, width, bias=False
        )
        # Every stream starts from the same evidence but a different learned
        # offset, so streams can diverge into complementary interpretations
        # instead of collapsing onto one another.
        self.stream_embedding = nn.Parameter(
            torch.empty(core_config.streams, width)
        )
        self.core = GatorLinearCore(core_config)

        # Late fusion: small per-stream adapters preserve the
        # primary/corrector distinction, then one vocabulary projection runs
        # over the fused hidden state instead of once per stream.
        self.stream_adapter = nn.Parameter(
            torch.empty(core_config.streams, width, width)
        )
        self.fusion_score = nn.Linear(width, 1)
        # A bias vector over the vocabulary can express the target marginal
        # on its own, at zero cost to the input pathway. With the observed
        # collapse -- every stream emitting one constant token while loss
        # sits at exactly ln(12), the uniform 12-way marginal -- that bias is
        # the cheapest way to stay there. Dropping it forces any marginal to
        # be produced through the fused representation, so the input path
        # carries it or nothing does.
        self.vocabulary = nn.Linear(
            width, core_config.vocab_size, bias=core_config.readout_bias
        )
        if core_config.tie_token_embeddings:
            # EvoToken turns output beliefs back into inputs through
            # token_embedding.weight. Sharing the classifier with that table
            # keeps both directions in one token geometry and avoids paying
            # for a second [vocabulary, width] matrix. Assignment preserves a
            # single Parameter object, so optimizers update it exactly once.
            self.vocabulary.weight = self.token_embedding.weight
        # Populated on every forward so the trainer can add the route losses
        # without a second pass.
        self.latest_route_statistics: dict[str, Tensor] = {}
        self.latest_route_statistics_scope = "full_sequence"

        # Surprise-Gated Block Memory (opt-in). Every submodule here is only
        # constructed when config.memory is set, so a model built with the
        # default config (memory=None) has zero new parameters and zero new
        # codepaths exercised in forward_cached.
        self.memory_config = config.memory
        if self.memory_config is not None:
            memory_config = self.memory_config
            # route_stat_dim=2 (route_entropy, route_balance) and
            # indicator_dim=2 (mean top-1 confidence, mean output entropy)
            # are v1 placeholders: GatorLinearCore currently only exposes
            # batch-averaged scalar route statistics, not per-lane
            # occupancy, so route_surprise cannot yet be computed per lane
            # (see _memory_write_from_formation). Richer per-lane route/
            # quality features are a natural follow-up once this wiring is
            # validated, not a design ceiling.
            self._memory_route_stat_dim = 2
            self._memory_indicator_dim = 2
            self.memory_query_projection = nn.Linear(2 * width, memory_config.key_dim)
            self.memory_key_projection = nn.Linear(
                2 * width + memory_config.value_dim, memory_config.key_dim
            )
            self.memory_value_projection = nn.Linear(
                2 * width + self._memory_route_stat_dim, memory_config.value_dim
            )
            self.memory_block_predictor = BlockExpectationPredictor(
                context_dim=width,
                cache_dim=width,
                memory_dim=memory_config.value_dim,
                output_dim=width,
            )
            self.memory_block_summarizer = BlockSummarizer(
                token_embedding_dim=width,
                hidden_dim=width,
                stream_width=width,
                route_stat_dim=self._memory_route_stat_dim,
                indicator_dim=self._memory_indicator_dim,
                output_dim=width,
            )
            self.memory_stream_injector = StreamMemoryInjector(
                stream_count=core_config.streams,
                stream_width=width,
                value_dim=memory_config.value_dim,
                dropout=memory_config.memory_dropout,
            )
            self.surprise_gated_memory = SurpriseGatedMemory(memory_config)
            # Surprise-feature normalizers (EMA mean/var, spec section 9)
            # are plain running statistics, not learned parameters, so they
            # live outside the nn.Module parameter tree -- same treatment
            # as self.latest_route_statistics above.
            self._memory_predictive_normalizer = SurpriseNormalizer()
            self._memory_streams_normalizer = SurpriseNormalizer()
            self._memory_novelty_normalizer = SurpriseNormalizer()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        width = self.config.core.width
        nn.init.normal_(self.token_embedding.weight, std=0.5)
        nn.init.zeros_(self.output_query)
        nn.init.normal_(self.slot_type.weight, std=0.5)
        nn.init.normal_(self.stream_embedding, std=0.5)
        identity = torch.eye(width).expand(
            self.config.core.streams, width, width
        )
        with torch.no_grad():
            self.stream_adapter.copy_(
                identity + torch.randn_like(identity) * 0.02
            )
        if self.vocabulary.bias is not None:
            nn.init.zeros_(self.vocabulary.bias)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def procoim_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        """Return disjoint semantic blocks for forward-only perturbations.

        The tied token/vocabulary weight is yielded only once by
        ``named_parameters()`` and is deliberately assigned to ``embedding``.
        The default Procoim schedule omits that group, keeping the token table
        frozen for the first generated-prefix experiments.
        """

        groups: dict[str, list[nn.Parameter]] = {
            "routing": [],
            "receiver_memory": [],
            "node_compute": [],
            "stream_mix": [],
            "readout": [],
            "embedding": [],
        }
        embedding_prefixes = (
            "token_embedding.",
            "output_query",
            "slot_type.",
            "address_projection.",
            "stream_embedding",
        )
        readout_prefixes = (
            "fusion_score.",
            "vocabulary.",
        )
        stream_mix_prefixes = (
            "stream_adapter",
            "core.stream_ingress_bias",
        )
        routing_markers = (
            "core.ingress_",
            ".router_query.",
            ".edge_key",
            ".edge_bias",
            ".stream_edge_bias",
            ".bus_in",
            ".bus_out",
            ".bus_gate.",
        )
        receiver_markers = (
            ".write_down",
            ".write_up",
            ".read_down",
            ".read_up",
            ".read_gate",
            ".write_projection.",
            ".write_gate.",
            ".mixer.",
            ".read_projection.",
            ".context_norm.",
            ".attn_",
        )
        node_markers = (
            ".basis_weight",
            ".basis_bias",
            ".node_coefficients",
            ".update_gate.",
            ".delta_norm.",
        )
        for name, parameter in self.named_parameters():
            if name.startswith(embedding_prefixes):
                group = "embedding"
            elif name.startswith(readout_prefixes):
                group = "readout"
            elif name.startswith(stream_mix_prefixes):
                group = "stream_mix"
            elif any(marker in name for marker in routing_markers):
                group = "routing"
            elif any(marker in name for marker in receiver_markers):
                group = "receiver_memory"
            elif any(marker in name for marker in node_markers) or (
                name.startswith("core.relays.") and ".norm." in name
            ):
                group = "node_compute"
            else:
                group = "stream_mix"
            groups[group].append(parameter)
        return groups

    def initial_hidden(
        self,
        observed: Tensor,
        *,
        output_slots: int,
        output_embeddings: Tensor | None = None,
        output_position_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return initial stream states ``[B, L, S, D]`` and addresses.

        ``output_position_ids[physical] = logical`` optionally assigns a
        logical output address to each physical output tensor slot. This is
        used only by position-starvation controls: targets and soft-token
        embeddings are packed with the same mapping, so the permutation is
        identifiable rather than random label noise.
        """

        if observed.ndim != 2:
            raise ValueError("observed must have shape [batch, slots]")
        if output_slots <= 0:
            raise ValueError("output_slots must be positive")
        core_config = self.config.core
        batch, observed_length = observed.shape
        length = observed_length + output_slots
        device = observed.device
        dtype = self.token_embedding.weight.dtype

        evidence = self.token_embedding(observed)
        if output_embeddings is None:
            queries = self.output_query.view(1, 1, -1).expand(
                batch, output_slots, -1
            )
        else:
            expected = (batch, output_slots, core_config.width)
            if output_embeddings.shape != expected:
                raise ValueError(
                    "output_embeddings must have shape "
                    f"{expected}, got {tuple(output_embeddings.shape)}"
                )
            if output_embeddings.device != observed.device:
                raise ValueError(
                    "output_embeddings and observed must be on the same device"
                )
            if not output_embeddings.is_floating_point():
                raise TypeError("output_embeddings must be floating point")
            queries = output_embeddings.to(dtype=dtype)
        slots = torch.cat((evidence, queries), dim=1)

        kinds = torch.zeros(length, device=device, dtype=torch.long)
        kinds[observed_length:] = 1
        slots = slots + self.slot_type(kinds).unsqueeze(0)

        addresses = fixed_address_code(
            length,
            core_config.address_width,
            device=device,
            dtype=dtype,
            reference_length=(
                core_config.context_length
                if core_config.block_causal
                else None
            ),
        )
        if output_position_ids is not None:
            if (
                output_position_ids.ndim != 1
                or output_position_ids.shape[0] != output_slots
                or output_position_ids.dtype != torch.long
                or output_position_ids.device != device
            ):
                raise TypeError(
                    "output_position_ids must be long [output_slots] on the "
                    "observed device"
                )
            if not torch.compiler.is_compiling():
                expected = torch.arange(output_slots, device=device)
                if not torch.equal(output_position_ids.sort().values, expected):
                    raise ValueError(
                        "output_position_ids must be a permutation"
                    )
            output_addresses = addresses[observed_length:].index_select(
                0, output_position_ids
            )
            addresses = torch.cat(
                (addresses[:observed_length], output_addresses), dim=0
            )
        slots = slots + self.address_projection(addresses).unsqueeze(0)

        hidden = slots.unsqueeze(2) + self.stream_embedding.view(
            1, 1, core_config.streams, -1
        )
        return hidden, addresses

    def _single_kind_hidden(
        self,
        embeddings: Tensor,
        *,
        slot_kind: int,
        address_offset: int | Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Build committed or proposal slots at absolute cached positions."""

        if not self.config.core.block_causal:
            raise ValueError("cached slots require block_causal=True")
        if embeddings.ndim != 3:
            raise ValueError("embeddings must have shape [batch, slots, width]")
        batch, length, width = embeddings.shape
        core_config = self.config.core
        if width != core_config.width or length <= 0:
            raise ValueError("embeddings have the wrong slot or width shape")
        if slot_kind not in (0, 1):
            raise ValueError("slot_kind must be 0 (committed) or 1 (proposal)")
        embeddings = embeddings.to(dtype=self.token_embedding.weight.dtype)
        kinds = torch.full(
            (length,), slot_kind, device=embeddings.device, dtype=torch.long
        )
        addresses = fixed_address_code(
            length,
            core_config.address_width,
            device=embeddings.device,
            dtype=embeddings.dtype,
            offset=address_offset,
            reference_length=core_config.context_length,
        )
        slots = (
            embeddings
            + self.slot_type(kinds).unsqueeze(0)
            + self.address_projection(addresses).unsqueeze(0)
        )
        hidden = slots.unsqueeze(2) + self.stream_embedding.view(
            1, 1, core_config.streams, -1
        )
        return hidden, addresses

    def _committed_hidden(
        self,
        token_ids: Tensor,
        *,
        address_offset: int | Tensor,
    ) -> tuple[Tensor, Tensor]:
        if token_ids.ndim != 2 or token_ids.dtype != torch.long:
            raise TypeError("committed token_ids must be long [batch, slots]")
        return self._single_kind_hidden(
            self.token_embedding(token_ids),
            slot_kind=0,
            address_offset=address_offset,
        )

    def _proposal_hidden(
        self,
        *,
        batch_size: int,
        output_slots: int,
        address_offset: Tensor,
        output_embeddings: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        if output_embeddings is None:
            embeddings = self.output_query.view(1, 1, -1).expand(
                batch_size, output_slots, -1
            )
        else:
            expected = (batch_size, output_slots, self.config.core.width)
            if output_embeddings.shape != expected:
                raise ValueError(
                    "output_embeddings must have shape "
                    f"{expected}, got {tuple(output_embeddings.shape)}"
                )
            if not output_embeddings.is_floating_point():
                raise TypeError("output_embeddings must be floating point")
            embeddings = output_embeddings
        return self._single_kind_hidden(
            embeddings,
            slot_kind=1,
            address_offset=address_offset,
        )

    def output_logits(
        self,
        hidden: Tensor,
        *,
        output_slots: int,
    ) -> tuple[Tensor, Tensor]:
        """Apply the shared coalition readout to any relay hidden state.

        Returning both logits and fusion weights lets experimental training
        supervise intermediate relays without adding per-relay vocabulary
        heads. The terminal path calls the same helper, so this refactor does
        not change existing model mathematics.
        """

        if hidden.ndim != 4:
            raise ValueError(
                "hidden must have shape [batch, slots, streams, width]"
            )
        if output_slots <= 0 or output_slots > hidden.shape[1]:
            raise ValueError(
                "output_slots must be in [1, hidden sequence length]"
            )
        weights = torch.softmax(self.fusion_score(hidden).squeeze(-1), dim=-1)
        # The adapter and vocabulary head are position-independent, and only
        # the final output slots contribute logits. Slice before the two large
        # projections so observed-context positions are not transformed only
        # to be discarded immediately afterward. Keep the full fusion weights
        # for existing diagnostics.
        output_hidden = hidden[:, -output_slots:]
        adapted = torch.einsum(
            "blsd,sde->blse", output_hidden, self.stream_adapter
        )
        fused = torch.einsum(
            "blsd,bls->bld", adapted, weights[:, -output_slots:]
        )
        return self.vocabulary(fused), weights

    def write_gate_schedule(
        self,
        *,
        batch_size: int,
        observed_length: int,
        output_slots: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Silence output slots as mailbox *writers* early, ramp to full.

        Matches gator_path/prefix.py's validated controlled-writing
        schedule: observed slots always write at full strength; output
        slots start at 0 (they are still blank queries at relay 0, so
        writing them would pollute the shared mailbox with noise) and ramp
        linearly to 1 by the final relay, once they have had a chance to
        read a real answer. Only what a slot *contributes* is gated -- every
        slot always reads normally, per relay, from relay 0.
        """

        relays = self.config.core.relays
        length = observed_length + output_slots
        schedule = torch.ones(
            relays, batch_size, length, device=device, dtype=dtype
        )
        denominator = max(relays - 1, 1)
        fractions = (
            torch.arange(relays, device=device, dtype=dtype) / denominator
        )
        schedule[:, :, observed_length:] = fractions.view(-1, 1, 1)
        return schedule

    @torch.no_grad()
    def prefill_cache(
        self,
        committed_ids: Tensor,
        *,
        block_size: int,
        temperature: float = 1.0,
    ) -> GatorLinearBlockCache:
        """Encode committed token blocks into a constant-size relay cache."""

        if not self.config.core.block_causal:
            raise ValueError("prefill_cache requires block_causal=True")
        if committed_ids.ndim != 2 or committed_ids.dtype != torch.long:
            raise TypeError("committed_ids must be long [batch, slots]")
        if block_size <= 0 or committed_ids.shape[1] % block_size != 0:
            raise ValueError(
                "committed length must be divisible by positive block_size"
            )
        hidden, addresses = self._committed_hidden(
            committed_ids, address_offset=0
        )
        result = self.core(
            hidden,
            addresses,
            temperature=temperature,
            block_size=block_size,
            return_cache=True,
        )
        _, _, statistics, cache = result
        self.latest_route_statistics = statistics
        self.latest_route_statistics_scope = "full_sequence"
        return detach_block_cache(cache)

    def _memory_read_and_inject(
        self,
        hidden: Tensor,
        memory: MemoryRuntime,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        """Read SGBM before relay 0 and inject it into every proposal stream.

        Design spec section 4: memory is retrieved once per block from a
        query built off the running context/cache summaries, then injected
        into every private work stream through a gated residual — the same
        retrieved vector broadcast across every position in the active
        block (not per-position retrieval).
        """

        query_input = torch.cat([memory.context_summary, memory.cache_summary], dim=-1)
        query = torch.nn.functional.normalize(
            self.memory_query_projection(query_input), dim=-1, eps=1e-8
        )
        retrieved, read_metrics = self.surprise_gated_memory.read(query, memory.state)

        batch, output_slots, streams, width = hidden.shape
        flat_hidden = hidden.reshape(batch * output_slots, streams, width)
        flat_retrieved = (
            retrieved.unsqueeze(1).expand(batch, output_slots, -1).reshape(batch * output_slots, -1)
        )
        injected = self.memory_stream_injector(flat_hidden, flat_retrieved)
        hidden = injected.reshape(batch, output_slots, streams, width)
        return hidden, retrieved, read_metrics

    def _memory_write_from_formation(
        self,
        *,
        final_hidden: Tensor,
        logits: Tensor,
        memory: MemoryRuntime,
        retrieved: Tensor,
    ) -> dict[str, Tensor]:
        """Compute surprise/reliability from the just-formed proposal and write.

        Runs entirely on the *proposed* (not yet accepted) block, matching
        design spec section 22's pseudocode — whether the block is later
        accepted into the committed cache (``append_cache``) is a separate
        decision this never touches, preserving the "no target leakage"
        invariant (section 6.2): memory written here can only affect the
        *next* call to ``forward_cached``, never this one.
        """

        committed_tokens = logits.argmax(dim=-1).detach()
        token_embeddings = self.token_embedding(committed_tokens)
        pooled_stream = final_hidden.mean(dim=1)  # [B, S, D], pooled over block positions
        # v1 approximation of the coalition-fused hidden state (the exact
        # fusion-weighted vector is available via output_logits, but
        # re-deriving it here would duplicate that computation for a
        # feature that only needs to be roughly representative).
        fused_hidden = final_hidden.mean(dim=(1, 2))

        route_entropy = self.latest_route_statistics.get(
            "route_entropy", torch.zeros((), device=final_hidden.device)
        )
        route_balance = self.latest_route_statistics.get(
            "route_balance", torch.zeros((), device=final_hidden.device)
        )
        batch = final_hidden.shape[0]
        route_stats = torch.stack([route_entropy, route_balance]).view(1, 2).expand(batch, 2)

        probabilities = torch.softmax(logits, dim=-1)
        top1_confidence = probabilities.max(dim=-1).values.mean(dim=1)
        prediction_entropy = (
            -(probabilities.clamp_min(1e-8) * probabilities.clamp_min(1e-8).log())
            .sum(dim=-1)
            .mean(dim=1)
        )
        indicators = torch.stack([top1_confidence, prediction_entropy], dim=-1)

        actual_summary = self.memory_block_summarizer(
            token_embeddings=token_embeddings,
            fused_hidden=fused_hidden,
            stream_states=pooled_stream,
            route_stats=route_stats,
            indicators=indicators,
        )
        predicted_summary = self.memory_block_predictor(
            memory.context_summary, memory.cache_summary, retrieved
        )

        candidate_key = torch.nn.functional.normalize(
            self.memory_key_projection(
                torch.cat([memory.context_summary, actual_summary.detach(), retrieved.detach()], dim=-1)
            ),
            dim=-1,
            eps=1e-8,
        )
        pooled_stream_vector = pooled_stream.mean(dim=1)
        candidate_value = self.memory_value_projection(
            torch.cat([actual_summary.detach(), pooled_stream_vector.detach(), route_stats], dim=-1)
        )

        predictive = predictive_surprise(predicted_summary, actual_summary.detach())
        streams_score = stream_disagreement(pooled_stream)
        # route_surprise (design spec section 7.3) needs a per-lane node
        # occupancy distribution; GatorLinearCore currently only returns
        # batch-averaged scalar route statistics, so this stays zero until
        # the core is extended to expose per-lane occupancy (tracked as a
        # follow-up, not silently faked here).
        route_score = torch.zeros(batch, device=final_hidden.device)
        novelty = semantic_novelty(candidate_key, memory.state.keys, memory.state.valid)
        continuity = context_continuity(memory.state.previous_block_summary, actual_summary.detach())

        predictive_norm = self._memory_predictive_normalizer.normalize(predictive)
        streams_norm = self._memory_streams_normalizer.normalize(streams_score)
        novelty_norm = self._memory_novelty_normalizer.normalize(novelty)

        memory.state.surprise_momentum = surprise_momentum_update(
            memory.state.surprise_momentum, predictive_norm, continuity
        )
        raw_combined, blended_combined = combine_surprise(
            prediction=predictive_norm,
            streams=streams_norm,
            route=route_score,
            novelty=novelty_norm,
            momentum=memory.state.surprise_momentum,
        )

        consensus = consensus_reliability(streams_score)
        entropy_rel = entropy_reliability(probabilities, vocab_size=self.config.core.vocab_size)
        # v1: no blank/repeat/forbidden-token detection wired up yet (the
        # prefix task has no such concept); neutral quality until a language
        # task supplies real rates.
        neutral = torch.zeros(batch, device=final_hidden.device)
        quality = quality_reliability(neutral, neutral, neutral)
        reliability = combined_reliability(consensus, entropy_rel, quality)

        self.surprise_gated_memory.write(
            candidate_key=candidate_key.detach(),
            candidate_value=candidate_value.detach(),
            surprise=blended_combined.detach(),
            reliability=reliability.detach(),
            state=memory.state,
            novelty=novelty.detach(),
        )
        memory.state.previous_block_summary = actual_summary.detach()
        memory.update_summaries(token_embeddings.mean(dim=1).detach())

        return {
            "predictive_surprise": predictive.detach(),
            "stream_disagreement": streams_score.detach(),
            "route_surprise": route_score.detach(),
            "semantic_novelty": novelty.detach(),
            "context_continuity": continuity.detach(),
            "combined_surprise": raw_combined.detach(),
            "blended_surprise": blended_combined.detach(),
            "reliability": reliability.detach(),
        }

    def _cached_proposal_hidden(
        self,
        cache: GatorLinearBlockCache,
        *,
        output_embeddings: Tensor | None,
        temperature: float,
        return_diagnostics: bool,
        memory: MemoryRuntime | None = None,
    ) -> tuple[
        Tensor,
        list[dict[str, Tensor]],
        dict[str, Tensor],
        Tensor | None,
        dict[str, Tensor] | None,
    ]:
        """Run one proposal block while leaving committed cache immutable."""

        if not self.config.core.block_causal:
            raise ValueError("cached proposals require block_causal=True")
        if not isinstance(cache, GatorLinearBlockCache):
            raise TypeError("cache must be a GatorLinearBlockCache")
        if len(cache.relays) != self.config.core.relays:
            raise ValueError("cache relay count does not match model")
        if memory is not None and self.memory_config is None:
            raise ValueError(
                "memory was supplied but this model was built with memory=None"
            )
        batch_size = cache.relays[0].mixer.node_sum.shape[0]
        hidden, addresses = self._proposal_hidden(
            batch_size=batch_size,
            output_slots=cache.block_size,
            address_offset=cache.next_position,
            output_embeddings=output_embeddings,
        )
        retrieved = None
        read_metrics = None
        if memory is not None:
            hidden, retrieved, read_metrics = self._memory_read_and_inject(hidden, memory)
        schedule = None
        if self.config.core.controlled_writing:
            schedule = self.write_gate_schedule(
                batch_size=batch_size,
                observed_length=0,
                output_slots=cache.block_size,
                device=hidden.device,
                dtype=self.token_embedding.weight.dtype,
            )
        result = self.core(
            hidden,
            addresses,
            temperature=temperature,
            write_gate_schedule=schedule,
            return_diagnostics=return_diagnostics,
            block_size=cache.block_size,
            cache=cache,
        )
        final_hidden, records, statistics = result
        self.latest_route_statistics = statistics
        self.latest_route_statistics_scope = "active_block"
        return final_hidden, records, statistics, retrieved, read_metrics

    def forward_cached(
        self,
        cache: GatorLinearBlockCache,
        *,
        output_embeddings: Tensor | None = None,
        temperature: float = 1.0,
        return_diagnostics: bool = False,
        memory: MemoryRuntime | None = None,
    ) -> Tensor | tuple[Tensor, dict[str, object]]:
        """Predict the next block without advancing or mutating ``cache``.

        ``memory`` is optional and defaults to ``None``, in which case this
        method's behavior (and the sequence of tensor operations it runs)
        is completely unchanged from before Surprise-Gated Block Memory
        existed. Passing a :class:`MemoryRuntime` (only valid when this
        model was built with ``config.memory`` set) reads memory before
        relay 0 and writes a candidate back afterward — see
        ``_memory_read_and_inject``/``_memory_write_from_formation``.
        """

        hidden, records, statistics, retrieved, read_metrics = self._cached_proposal_hidden(
            cache,
            output_embeddings=output_embeddings,
            temperature=temperature,
            return_diagnostics=return_diagnostics,
            memory=memory,
        )
        logits, weights = self.output_logits(
            hidden, output_slots=cache.block_size
        )
        memory_metrics: dict[str, Tensor] = {}
        if memory is not None:
            memory_metrics = self._memory_write_from_formation(
                final_hidden=hidden,
                logits=logits,
                memory=memory,
                retrieved=retrieved,
            )
            memory_metrics.update(read_metrics)
        if not return_diagnostics:
            return logits
        diagnostics: dict[str, object] = {
            "relays": records,
            "fusion_weights": weights.detach(),
            "route_statistics": statistics,
            "route_statistics_scope": "active_block",
        }
        if memory is not None:
            diagnostics["memory"] = memory_metrics
        return logits, diagnostics

    @torch.no_grad()
    def append_cache(
        self,
        cache: GatorLinearBlockCache,
        accepted_ids: Tensor,
        *,
        temperature: float = 1.0,
    ) -> GatorLinearBlockCache:
        """Atomically commit one accepted discrete block into a new cache.

        Accepted IDs are recomputed as observed slots with full write strength.
        Proposal/mask embeddings and their controlled-write states are never
        persisted into the language history.
        """

        if not self.config.core.block_causal:
            raise ValueError("append_cache requires block_causal=True")
        if accepted_ids.ndim != 2 or accepted_ids.dtype != torch.long:
            raise TypeError("accepted_ids must be long [batch, block_size]")
        batch_size = cache.relays[0].mixer.node_sum.shape[0]
        if accepted_ids.shape != (batch_size, cache.block_size):
            raise ValueError(
                "accepted_ids must match cache batch and block_size"
            )
        hidden, addresses = self._committed_hidden(
            accepted_ids, address_offset=cache.next_position
        )
        result = self.core(
            hidden,
            addresses,
            temperature=temperature,
            block_size=cache.block_size,
            cache=cache,
            return_cache=True,
        )
        _, _, statistics, next_cache = result
        self.latest_route_statistics = statistics
        self.latest_route_statistics_scope = "active_block"
        return detach_block_cache(next_cache)

    def stream_logits(
        self,
        observed: Tensor,
        *,
        output_slots: int,
        temperature: float = 1.0,
    ) -> tuple[Tensor, Tensor]:
        """Return per-stream logits and fusion weights without fusing.

        Used to measure whether the terminal streams form a *complementary
        coalition* or a winner-take-all particle. The reference architecture
        settles on the former: three material lanes scoring 33% alone, 83% as
        the best pair, and 99% together (GATOR_PATH_DESIGN_SPEC.md).
        Preserving that is an acceptance criterion, so it has to be
        measurable rather than assumed.

        Returns:
            Logits ``[B, output_slots, S, V]`` and weights ``[B, L, S]``.
        """

        hidden, addresses = self.initial_hidden(
            observed, output_slots=output_slots
        )
        schedule = None
        if self.config.core.controlled_writing:
            batch, observed_length = observed.shape
            schedule = self.write_gate_schedule(
                batch_size=batch,
                observed_length=observed_length,
                output_slots=output_slots,
                device=observed.device,
                dtype=self.token_embedding.weight.dtype,
            )
        hidden, _, _ = self.core(
            hidden,
            addresses,
            temperature=temperature,
            write_gate_schedule=schedule,
            block_size=(
                output_slots if self.config.core.block_causal else None
            ),
        )
        adapted = torch.einsum("blsd,sde->blse", hidden, self.stream_adapter)
        weights = torch.softmax(self.fusion_score(hidden).squeeze(-1), dim=-1)
        # The vocabulary head is linear, so applying it per stream and then
        # combining is equivalent to fusing first -- which is exactly what
        # makes per-stream ablation meaningful.
        per_stream = self.vocabulary(adapted)[:, -output_slots:]
        return per_stream, weights

    def forward(
        self,
        observed: Tensor,
        *,
        output_slots: int,
        output_embeddings: Tensor | None = None,
        output_position_ids: Tensor | None = None,
        temperature: float = 1.0,
        return_diagnostics: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, object]]:
        """Return logits for the output slots only."""

        hidden, addresses = self.initial_hidden(
            observed,
            output_slots=output_slots,
            output_embeddings=output_embeddings,
            output_position_ids=output_position_ids,
        )
        schedule = None
        if (
            self.config.core.block_causal
            and observed.shape[1] % output_slots != 0
        ):
            raise ValueError(
                "block_causal observed length must be divisible by output_slots"
            )
        if self.config.core.controlled_writing:
            batch, observed_length = observed.shape
            schedule = self.write_gate_schedule(
                batch_size=batch,
                observed_length=observed_length,
                output_slots=output_slots,
                device=observed.device,
                dtype=self.token_embedding.weight.dtype,
            )
        hidden, records, statistics = self.core(
            hidden,
            addresses,
            temperature=temperature,
            write_gate_schedule=schedule,
            return_diagnostics=return_diagnostics,
            block_size=(
                output_slots if self.config.core.block_causal else None
            ),
        )
        self.latest_route_statistics = statistics
        self.latest_route_statistics_scope = "full_sequence"

        logits, weights = self.output_logits(
            hidden, output_slots=output_slots
        )
        if not return_diagnostics:
            return logits
        return logits, {
            "relays": records,
            "fusion_weights": weights.detach(),
            "route_statistics": statistics,
        }
