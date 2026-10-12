"""Progressive soft-token refinement for GTR-L.

This module adapts EvoToken-DLM's four-state trajectory to GTR-L without
changing the legacy query-only forward path:

``MASK -> mask-aware soft token -> pure soft token -> hard decode``.

Two execution modes intentionally share the same state update:

``relay``
    Treat each GTR-L relay as one refinement step. Occupancy and private
    hidden state continue across the trajectory, and a detached top-k token
    belief is injected between relays. This is the efficient, GTR-L-native
    experiment.

``full_pass``
    Re-run the complete GTR-L stack for every refinement step, rebuilding
    ingress from the new soft token embeddings. This is the more faithful
    EvoToken control and costs ``steps * relays`` relay evaluations.

Token IDs and probabilities are detached between refinement steps in both
modes. Embedding lookup still uses the live token table at every step, just
as in the released EvoToken implementation, so the embedding parameters
receive ordinary input-path gradients without backpropagating through the
discrete trajectory decisions.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .core import GatorLinearBlockCache, load_balance, route_entropy
from .prefix import GatorLinearPrefixModel


def balanced_promotion_counts(
    output_slots: int,
    steps: int,
) -> tuple[int, ...]:
    """Distribute output-token promotions across refinement stages.

    Every slot is promoted exactly once. Remainder promotions are spread
    across the trajectory from left to right, which keeps early feedback
    available and naturally permits zero-promotion stages when
    ``output_slots < steps``.
    """

    if (
        not isinstance(output_slots, int)
        or isinstance(output_slots, bool)
        or output_slots <= 0
    ):
        raise ValueError("output_slots must be a positive integer")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise ValueError("steps must be a positive integer")

    base, remainder = divmod(output_slots, steps)
    counts = [base] * steps
    if remainder:
        for index in range(remainder):
            stage = index * steps // remainder
            counts[stage] += 1
    return tuple(counts)


def sample_training_schedule(
    targets: Tensor,
    *,
    mask_token_id: int,
    steps: int,
    transition_choices: Sequence[int] | None = None,
    promotion_counts: Sequence[int] | None = None,
    slot_order: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor, Tensor, tuple[int, ...]]:
    """Sample a CTS corruption and per-stage token-promotion schedule.

    ``promotion_counts`` may contain zero and need not be uniform. When it is
    omitted, the legacy ``transition_choices`` API remains available; if both
    are omitted, every slot is promoted once using
    :func:`balanced_promotion_counts`. ``slot_order`` optionally supplies one
    explicit permutation per batch row, making fixed and cyclic promotion
    controls possible without changing the default random schedule.
    """

    if targets.ndim != 2 or targets.dtype != torch.long:
        raise TypeError("targets must be a long tensor with [batch, slots] shape")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if transition_choices is not None and promotion_counts is not None:
        raise ValueError(
            "provide transition_choices or promotion_counts, not both"
        )

    batch, slots = targets.shape
    if slot_order is not None:
        if (
            slot_order.shape != targets.shape
            or slot_order.dtype != torch.long
            or slot_order.device != targets.device
        ):
            raise TypeError(
                "slot_order must be long, match targets, and share its device"
            )
        expected = torch.arange(slots, device=targets.device).expand(batch, -1)
        if not torch.equal(slot_order.sort(dim=1).values, expected):
            raise ValueError("each slot_order row must be a permutation")
    if promotion_counts is not None:
        counts = tuple(int(value) for value in promotion_counts)
    elif transition_choices is not None:
        choices = tuple(int(value) for value in transition_choices)
        if not choices or any(value <= 0 for value in choices):
            raise ValueError("transition choices must be positive")
        choice_index = int(
            torch.randint(
                len(choices),
                (1,),
                device=targets.device,
                generator=generator,
            ).item()
        )
        counts = (choices[choice_index],) * steps
    else:
        counts = balanced_promotion_counts(slots, steps)

    if len(counts) != steps:
        raise ValueError("promotion_counts must contain one value per step")
    if any(value < 0 for value in counts):
        raise ValueError("promotion counts cannot be negative")
    total_promotions = sum(counts)
    if total_promotions <= 0 or total_promotions > slots:
        raise ValueError(
            "promotion counts must promote between one and all output slots"
        )
    max_initial_visible = slots - total_promotions

    initial_pure = torch.zeros_like(targets, dtype=torch.bool)
    promotions = torch.zeros(
        steps,
        batch,
        slots,
        device=targets.device,
        dtype=torch.bool,
    )
    for row in range(batch):
        order = (
            torch.randperm(
                slots, device=targets.device, generator=generator
            )
            if slot_order is None
            else slot_order[row]
        )
        visible_count = int(
            torch.randint(
                max_initial_visible + 1,
                (1,),
                device=targets.device,
                generator=generator,
            ).item()
        )
        initial_pure[row, order[:visible_count]] = True
        scheduled = order[
            visible_count : visible_count + total_promotions
        ]
        cursor = 0
        for stage, count in enumerate(counts):
            selected = scheduled[cursor : cursor + count]
            promotions[stage, row, selected] = True
            cursor += count

    initial_ids = torch.full_like(targets, mask_token_id)
    initial_ids = torch.where(initial_pure, targets, initial_ids)
    return initial_ids, initial_pure, promotions, counts


def sample_masked_denoising_inputs(
    targets: Tensor,
    *,
    mask_token_id: int,
    minimum_mask_rate: float = 0.5,
    maximum_mask_rate: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Create a masked-denoising bootstrap batch with at least one mask/row."""

    if targets.ndim != 2 or targets.dtype != torch.long:
        raise TypeError("targets must be a long tensor with [batch, slots] shape")
    if not 0.0 <= minimum_mask_rate <= maximum_mask_rate <= 1.0:
        raise ValueError("mask rates must satisfy 0 <= minimum <= maximum <= 1")
    batch, slots = targets.shape
    rates = torch.rand(
        batch,
        1,
        device=targets.device,
        generator=generator,
    )
    rates = minimum_mask_rate + rates * (
        maximum_mask_rate - minimum_mask_rate
    )
    masked = (
        torch.rand(
            batch,
            slots,
            device=targets.device,
            generator=generator,
        )
        < rates
    )
    missing = ~masked.any(dim=1)
    if bool(missing.any()):
        rows = missing.nonzero(as_tuple=False).squeeze(-1)
        positions = torch.randint(
            slots,
            (rows.numel(),),
            device=targets.device,
            generator=generator,
        )
        masked[rows, positions] = True
    input_ids = torch.where(
        masked,
        torch.full_like(targets, mask_token_id),
        targets,
    )
    return input_ids, masked


def masked_trajectory_cross_entropy(
    stage_logits: Tensor,
    targets: Tensor,
    initially_visible: Tensor,
    *,
    visible_weight: float = 0.0,
    position_weights: Tensor | None = None,
    worst_position_weight: float = 0.0,
    worst_position_temperature: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Score an EvoToken trajectory without letting copied tokens dominate.

    Every stage is supervised, but positions that were visible as
    ground-truth inputs receive ``visible_weight`` relative to positions that
    began masked. A zero weight therefore trains only tokens the trajectory
    actually had to infer while still supervising them after promotion.

    ``position_weights`` applies an additional nonnegative weight to each
    physical output slot. Zero excludes a position, which supports tail-only
    probes. ``worst_position_weight`` mixes the weighted mean with a
    weight-normalized smooth maximum over active per-position losses. The
    smooth maximum uses a temperature-normalized log-sum-exp, so it approaches
    the weighted mean at high temperature and the worst active position at low
    temperature.

    Returns the normalized weighted loss and the unreduced per-token loss for
    diagnostics.
    """

    if stage_logits.ndim != 4:
        raise ValueError(
            "stage_logits must have [stages, batch, slots, vocabulary] shape"
        )
    if targets.ndim != 2 or targets.dtype != torch.long:
        raise TypeError("targets must be a long tensor with [batch, slots] shape")
    if initially_visible.shape != targets.shape:
        raise ValueError("initially_visible must match targets")
    if initially_visible.dtype != torch.bool:
        raise TypeError("initially_visible must be boolean")
    if tuple(stage_logits.shape[1:3]) != tuple(targets.shape):
        raise ValueError("stage_logits batch and slot dimensions must match targets")
    if not 0.0 <= visible_weight <= 1.0:
        raise ValueError("visible_weight must be in [0, 1]")
    if not 0.0 <= worst_position_weight <= 1.0:
        raise ValueError("worst_position_weight must be in [0, 1]")
    if worst_position_temperature <= 0.0:
        raise ValueError("worst_position_temperature must be positive")

    stages = stage_logits.shape[0]
    expanded_targets = targets.unsqueeze(0).expand(stages, -1, -1)
    token_loss = F.cross_entropy(
        stage_logits.reshape(-1, stage_logits.shape[-1]),
        expanded_targets.reshape(-1),
        reduction="none",
    ).view_as(expanded_targets)

    visible = initially_visible.unsqueeze(0).expand_as(token_loss)
    visibility_weights = torch.where(
        visible,
        torch.full_like(token_loss, visible_weight),
        torch.ones_like(token_loss),
    )
    if position_weights is None:
        resolved_position_weights = torch.ones(
            targets.shape[1], device=targets.device, dtype=token_loss.dtype
        )
        weights = visibility_weights
    else:
        if (
            position_weights.ndim != 1
            or position_weights.shape[0] != targets.shape[1]
            or not position_weights.is_floating_point()
            or position_weights.device != targets.device
        ):
            raise TypeError(
                "position_weights must be floating [slots] on the target device"
            )
        if (
            not torch.compiler.is_compiling()
            and bool((position_weights < 0).any())
        ):
            raise ValueError("position_weights cannot be negative")
        resolved_position_weights = position_weights.to(token_loss.dtype)
        weights = visibility_weights * resolved_position_weights.view(
            1, 1, -1
        )
    denominator = weights.sum()
    if not torch.compiler.is_compiling() and not bool(denominator > 0):
        raise ValueError("the trajectory must contain at least one scored token")
    mean_loss = (token_loss * weights).sum() / denominator.clamp_min(1.0)
    if worst_position_weight == 0.0:
        loss = mean_loss
    else:
        per_position_denominator = visibility_weights.sum(dim=(0, 1))
        per_position_loss = (
            (token_loss * visibility_weights).sum(dim=(0, 1))
            / per_position_denominator.clamp_min(1.0)
        ).float()
        smooth_weights = resolved_position_weights.float()
        active = smooth_weights > 0
        active_weight = smooth_weights.sum().clamp_min(
            torch.finfo(smooth_weights.dtype).tiny
        )
        log_weights = smooth_weights.clamp_min(
            torch.finfo(smooth_weights.dtype).tiny
        ).log()
        scaled = (
            per_position_loss / worst_position_temperature + log_weights
        ).masked_fill(~active, -torch.inf)
        smooth_worst = worst_position_temperature * (
            torch.logsumexp(scaled, dim=0) - active_weight.log()
        )
        loss = (
            (1.0 - worst_position_weight) * mean_loss
            + worst_position_weight * smooth_worst
        )
    return loss, token_loss


def position_gradient_statistics(
    position_losses: Tensor,
    parameter_groups: Sequence[Sequence[Tensor]],
) -> tuple[Tensor, Tensor]:
    """Return exact per-position group norms and cosine-to-prefix values.

    The diagnostic performs one vector-Jacobian product for every singleton
    position loss and every nontrivial cumulative prefix. Polarization then
    recovers the cosine between position ``i`` and the combined gradient from
    preceding positions without retaining ``positions * parameters`` worth
    of gradients. Only one VJP result is live at a time, and ``autograd.grad``
    leaves ordinary ``parameter.grad`` buffers untouched.

    Returns ``(norms, cosine_to_prefix)`` with shape ``[groups, positions]``.
    The first cosine and any zero-norm comparison are NaN.
    """

    if position_losses.ndim != 1 or not position_losses.is_floating_point():
        raise TypeError("position_losses must be a floating one-dimensional tensor")
    if position_losses.numel() <= 0:
        raise ValueError("position_losses cannot be empty")
    groups = tuple(tuple(group) for group in parameter_groups)
    if not groups or any(not group for group in groups):
        raise ValueError("parameter_groups must contain nonempty groups")
    parameters = tuple(parameter for group in groups for parameter in group)
    if any(not parameter.requires_grad for parameter in parameters):
        raise ValueError("all diagnostic parameters must require gradients")
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise ValueError("diagnostic parameter groups must be disjoint")
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for group in groups:
        next_cursor = cursor + len(group)
        ranges.append((cursor, next_cursor))
        cursor = next_cursor

    def group_squares(scalar_loss: Tensor) -> Tensor:
        gradients = torch.autograd.grad(
            scalar_loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        totals: list[Tensor] = []
        for start, stop in ranges:
            total = torch.zeros(
                (), device=position_losses.device, dtype=torch.float32
            )
            for gradient in gradients[start:stop]:
                if gradient is not None:
                    total = total + gradient.detach().float().square().sum()
            totals.append(total)
        return torch.stack(totals)

    singleton_squares = torch.stack(
        [group_squares(loss) for loss in position_losses]
    )
    prefix_squares = [singleton_squares[0]]
    for index in range(1, position_losses.shape[0]):
        prefix_squares.append(group_squares(position_losses[: index + 1].sum()))
    prefix_squares_tensor = torch.stack(prefix_squares)
    norms = singleton_squares.clamp_min(0).sqrt()
    cos_prefix = torch.full_like(norms, torch.nan)
    if position_losses.shape[0] > 1:
        numerator = 0.5 * (
            prefix_squares_tensor[1:]
            - prefix_squares_tensor[:-1]
            - singleton_squares[1:]
        )
        denominator = (
            prefix_squares_tensor[:-1].clamp_min(0).sqrt()
            * singleton_squares[1:].clamp_min(0).sqrt()
        )
        valid = denominator > 0
        cos_prefix[1:] = torch.where(
            valid,
            (numerator / denominator.clamp_min(1e-30)).clamp(-1.0, 1.0),
            torch.full_like(numerator, torch.nan),
        )
    return norms.transpose(0, 1), cos_prefix.transpose(0, 1)


def topk_token_distribution(
    logits: Tensor,
    *,
    top_k: int,
    valid_token_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return sparse beliefs plus confidence and retained probability mass.

    Probabilities are computed in float32 and renormalized over the retained
    tokens. ``confidence`` is the top-1 probability before top-k
    renormalization; ``retained_mass`` measures how much of the legal
    distribution survived filtering.
    """

    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, slots, vocabulary]")
    vocabulary = logits.shape[-1]
    if (
        valid_token_mask.ndim != 1
        or valid_token_mask.shape[0] != vocabulary
        or valid_token_mask.dtype != torch.bool
    ):
        raise TypeError(
            "valid_token_mask must be boolean with [vocabulary] shape"
        )
    if not torch.compiler.is_compiling():
        legal_count = int(valid_token_mask.sum().item())
        if top_k <= 0 or top_k > legal_count:
            raise ValueError("top_k must be in [1, number of valid tokens]")

    legal = valid_token_mask.to(device=logits.device)
    filtered = logits.detach().float().masked_fill(
        ~legal.view(1, 1, -1), -torch.inf
    )
    values, token_ids = filtered.topk(top_k, dim=-1)
    log_normalizer = torch.logsumexp(filtered, dim=-1)
    full_top_probabilities = torch.exp(values - log_normalizer.unsqueeze(-1))
    retained_mass = full_top_probabilities.sum(dim=-1)
    probabilities = torch.softmax(values, dim=-1)
    confidence = full_top_probabilities[..., 0]
    return token_ids, probabilities, confidence, retained_mass


def soft_token_embeddings(
    token_ids: Tensor,
    probabilities: Tensor,
    embedding_weight: Tensor,
) -> Tensor:
    """Evaluate a sparse probability-weighted token embedding lookup."""

    if token_ids.ndim != 3 or token_ids.dtype != torch.long:
        raise TypeError("token_ids must be long with [batch, slots, top_k] shape")
    if probabilities.shape != token_ids.shape:
        raise ValueError("probabilities must match token_ids")
    if not probabilities.is_floating_point():
        raise TypeError("probabilities must be floating point")
    if embedding_weight.ndim != 2:
        raise ValueError("embedding_weight must have [vocabulary, width] shape")
    if not torch.compiler.is_compiling():
        if bool((probabilities < 0).any()):
            raise ValueError("probabilities cannot be negative")
        row_sums = probabilities.float().sum(dim=-1)
        if not torch.allclose(
            row_sums,
            torch.ones_like(row_sums),
            atol=2e-4,
            rtol=2e-4,
        ):
            raise ValueError("probabilities must sum to one")
    candidates = F.embedding(token_ids, embedding_weight)
    return (
        candidates
        * probabilities.to(dtype=candidates.dtype).unsqueeze(-1)
    ).sum(dim=-2)


class EvoTokenRefiner(nn.Module):
    """Run continuous token evolution either across relays or full passes."""

    def __init__(
        self,
        model: GatorLinearPrefixModel,
        *,
        mask_token_id: int,
        valid_token_mask: Tensor,
        steps: int = 4,
        top_k: int = 3,
        mode: str = "relay",
        feedback_scale: float = 0.5,
    ) -> None:
        super().__init__()
        if not isinstance(model, GatorLinearPrefixModel):
            raise TypeError("model must be a GatorLinearPrefixModel")
        if mode not in {"relay", "full_pass"}:
            raise ValueError("mode must be relay or full_pass")
        if steps <= 0:
            raise ValueError("steps must be positive")
        if mode == "relay" and steps != model.config.core.relays:
            raise ValueError("relay mode requires steps == model relay count")
        if model.config.core.causal:
            raise ValueError(
                "EvoToken refinement requires bidirectional active-block mixing"
            )
        if not 0 <= mask_token_id < model.config.core.vocab_size:
            raise ValueError("mask_token_id must be inside the vocabulary")
        if not torch.isfinite(torch.tensor(feedback_scale)):
            raise ValueError("feedback_scale must be finite")
        if mode == "full_pass" and feedback_scale != 1.0:
            raise ValueError("full_pass mode requires feedback_scale=1")
        if (
            valid_token_mask.ndim != 1
            or valid_token_mask.shape[0] != model.config.core.vocab_size
            or valid_token_mask.dtype != torch.bool
        ):
            raise TypeError(
                "valid_token_mask must be boolean with [vocabulary] shape"
            )
        if bool(valid_token_mask[mask_token_id]):
            raise ValueError("the mask token cannot be a valid decoded token")
        if top_k <= 0 or top_k > int(valid_token_mask.sum()):
            raise ValueError("top_k exceeds the legal vocabulary")

        self.model = model
        self.mask_token_id = int(mask_token_id)
        self.steps = int(steps)
        self.top_k = int(top_k)
        self.mode = mode
        self.feedback_scale = float(feedback_scale)
        self.register_buffer(
            "valid_token_mask",
            valid_token_mask.clone(),
            persistent=False,
        )

    def _alpha(
        self,
        alpha: Tensor,
        *,
        stage: int,
        batch: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        if alpha.ndim == 0:
            value = alpha.expand(batch)
        elif alpha.shape == (self.steps, batch):
            value = alpha[stage]
        else:
            raise ValueError("alpha must be scalar or have [steps, batch] shape")
        if (
            not torch.compiler.is_compiling()
            and bool(((value < 0.0) | (value > 1.0)).any())
        ):
            raise ValueError("alpha values must be in [0, 1]")
        return value.to(device=device, dtype=dtype).view(batch, 1, 1)

    def _promote(
        self,
        confidence: Tensor,
        pure: Tensor,
        *,
        scheduled: Tensor | None,
        inference_promotion_tokens: int,
    ) -> Tensor:
        if scheduled is not None:
            if scheduled.shape != pure.shape or scheduled.dtype != torch.bool:
                raise TypeError(
                    "each promotion schedule stage must be boolean [batch, slots]"
                )
            if (
                not torch.compiler.is_compiling()
                and bool((scheduled & pure).any())
            ):
                raise ValueError("a schedule cannot promote an already-pure token")
            return scheduled
        if inference_promotion_tokens < 0:
            raise ValueError("inference promotion count cannot be negative")
        promoted = torch.zeros_like(pure)
        if inference_promotion_tokens == 0:
            return promoted
        unresolved = ~pure
        if (
            not torch.compiler.is_compiling()
            and bool(
                (unresolved.sum(dim=1) < inference_promotion_tokens).any()
            )
        ):
            raise ValueError("not enough unresolved tokens for inference promotion")
        selection_score = confidence.masked_fill(~unresolved, -torch.inf)
        indices = selection_score.topk(
            inference_promotion_tokens, dim=-1
        ).indices
        promoted.scatter_(1, indices, True)
        return promoted

    def _evolve(
        self,
        logits: Tensor,
        pure: Tensor,
        best_ids: Tensor,
        best_confidence: Tensor,
        *,
        alpha: Tensor,
        stage: int,
        scheduled_promotions: Tensor | None,
        inference_promotion_tokens: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        token_ids, probabilities, confidence, retained_mass = (
            topk_token_distribution(
                logits,
                top_k=self.top_k,
                valid_token_mask=self.valid_token_mask,
            )
        )
        promoted = self._promote(
            confidence,
            pure,
            scheduled=scheduled_promotions,
            inference_promotion_tokens=inference_promotion_tokens,
        )
        next_pure = pure | promoted
        candidates = token_ids[..., 0]
        better = next_pure & (confidence > best_confidence)
        next_best_ids = torch.where(better, candidates, best_ids)
        next_best_confidence = torch.where(
            better, confidence, best_confidence
        )

        distribution_embedding = soft_token_embeddings(
            token_ids.detach(),
            probabilities.detach(),
            self.model.token_embedding.weight,
        )
        mix = self._alpha(
            alpha,
            stage=stage,
            batch=logits.shape[0],
            dtype=distribution_embedding.dtype,
            device=distribution_embedding.device,
        )
        mask_embedding = self.model.token_embedding.weight[
            self.mask_token_id
        ].view(1, 1, -1)
        mask_aware = (
            mix * mask_embedding + (1.0 - mix) * distribution_embedding
        )
        next_embeddings = torch.where(
            next_pure.unsqueeze(-1),
            distribution_embedding,
            mask_aware,
        )
        return (
            next_pure,
            next_best_ids,
            next_best_confidence,
            next_embeddings,
            retained_mass,
        )

    def _relay_forward(
        self,
        observed: Tensor,
        current_embeddings: Tensor,
        pure: Tensor,
        best_ids: Tensor,
        best_confidence: Tensor,
        *,
        alpha: Tensor,
        promotion_schedule: Tensor | None,
        inference_promotion_counts: tuple[int, ...],
        output_position_ids: Tensor | None,
        temperature: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        output_slots = current_embeddings.shape[1]
        if (
            self.model.config.core.block_causal
            and observed.shape[1] % output_slots != 0
        ):
            raise ValueError(
                "block_causal observed length must be divisible by output slots"
            )
        hidden, addresses = self.model.initial_hidden(
            observed,
            output_slots=output_slots,
            output_embeddings=current_embeddings,
            output_position_ids=output_position_ids,
        )
        core = self.model.core
        width_mask = core.stream_width_mask.view(
            1, 1, *core.stream_width_mask.shape
        )
        hidden = hidden * width_mask
        occupancy = core.initial_occupancy(hidden)
        schedule = None
        if core.config.controlled_writing:
            schedule = self.model.write_gate_schedule(
                batch_size=observed.shape[0],
                observed_length=observed.shape[1],
                output_slots=output_slots,
                device=observed.device,
                dtype=self.model.token_embedding.weight.dtype,
            )

        stage_logits: list[Tensor] = []
        retained_masses: list[Tensor] = []
        entropies: list[Tensor] = []
        loads: list[Tensor] = []
        for stage, relay in enumerate(core.relays):
            entropies.append(route_entropy(occupancy))
            loads.append(occupancy.mean(dim=(0, 1, 2)))
            write_gate = (
                None
                if schedule is None
                else schedule[stage][:, :, None, None]
            )
            if core.config.block_causal:
                relay_result = relay(
                    hidden,
                    occupancy,
                    addresses,
                    core.edge_destinations,
                    temperature=temperature,
                    write_gate=write_gate,
                    block_size=output_slots,
                    cache_length=torch.zeros(
                        (), device=hidden.device, dtype=torch.long
                    ),
                    return_cache=True,
                )
                hidden, occupancy, _, _ = relay_result
            else:
                hidden, occupancy, _ = relay(
                    hidden,
                    occupancy,
                    addresses,
                    core.edge_destinations,
                    temperature=temperature,
                    write_gate=write_gate,
                )
            hidden = hidden * width_mask
            logits, _ = self.model.output_logits(
                hidden, output_slots=output_slots
            )
            stage_logits.append(logits)
            scheduled = (
                None
                if promotion_schedule is None
                else promotion_schedule[stage]
            )
            (
                pure,
                best_ids,
                best_confidence,
                next_embeddings,
                retained_mass,
            ) = self._evolve(
                logits,
                pure,
                best_ids,
                best_confidence,
                alpha=alpha,
                stage=stage,
                scheduled_promotions=scheduled,
                inference_promotion_tokens=inference_promotion_counts[stage],
            )
            retained_masses.append(retained_mass)
            if stage + 1 < self.steps and self.feedback_scale != 0.0:
                delta = (
                    next_embeddings - current_embeddings
                ).to(dtype=hidden.dtype)
                output_hidden = hidden[:, -output_slots:] + (
                    self.feedback_scale * delta.unsqueeze(2)
                )
                hidden = torch.cat(
                    (hidden[:, :-output_slots], output_hidden), dim=1
                )
                hidden = hidden * width_mask
            current_embeddings = next_embeddings

        entropies.append(route_entropy(occupancy))
        loads.append(occupancy.mean(dim=(0, 1, 2)))
        self.model.latest_route_statistics = {
            "route_entropy": torch.stack(entropies).mean(),
            "route_balance": load_balance(torch.stack(loads)),
        }
        self.model.latest_route_statistics_scope = "full_sequence"
        return (
            torch.stack(stage_logits),
            best_ids,
            torch.stack(retained_masses),
        )

    def _relay_forward_cached(
        self,
        cache: GatorLinearBlockCache,
        current_embeddings: Tensor,
        pure: Tensor,
        best_ids: Tensor,
        best_confidence: Tensor,
        *,
        alpha: Tensor,
        promotion_schedule: Tensor | None,
        inference_promotion_counts: tuple[int, ...],
        temperature: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Refine one active block against an immutable committed cache."""

        output_slots = cache.block_size
        batch = current_embeddings.shape[0]
        hidden, addresses = self.model._proposal_hidden(
            batch_size=batch,
            output_slots=output_slots,
            address_offset=cache.next_position,
            output_embeddings=current_embeddings,
        )
        core = self.model.core
        width_mask = core.stream_width_mask.view(
            1, 1, *core.stream_width_mask.shape
        )
        hidden = hidden * width_mask
        occupancy = core.initial_occupancy(hidden)
        schedule = None
        if core.config.controlled_writing:
            schedule = self.model.write_gate_schedule(
                batch_size=batch,
                observed_length=0,
                output_slots=output_slots,
                device=hidden.device,
                dtype=self.model.token_embedding.weight.dtype,
            )

        stage_logits: list[Tensor] = []
        retained_masses: list[Tensor] = []
        entropies: list[Tensor] = []
        loads: list[Tensor] = []
        for stage, relay in enumerate(core.relays):
            entropies.append(route_entropy(occupancy))
            loads.append(occupancy.mean(dim=(0, 1, 2)))
            write_gate = (
                None
                if schedule is None
                else schedule[stage][:, :, None, None]
            )
            relay_result = relay(
                hidden,
                occupancy,
                addresses,
                core.edge_destinations,
                temperature=temperature,
                write_gate=write_gate,
                block_size=output_slots,
                cache=cache.relays[stage],
                cache_length=cache.next_position,
                return_cache=True,
            )
            hidden, occupancy, _, _ = relay_result
            hidden = hidden * width_mask
            logits, _ = self.model.output_logits(
                hidden, output_slots=output_slots
            )
            stage_logits.append(logits)
            scheduled = (
                None
                if promotion_schedule is None
                else promotion_schedule[stage]
            )
            (
                pure,
                best_ids,
                best_confidence,
                next_embeddings,
                retained_mass,
            ) = self._evolve(
                logits,
                pure,
                best_ids,
                best_confidence,
                alpha=alpha,
                stage=stage,
                scheduled_promotions=scheduled,
                inference_promotion_tokens=inference_promotion_counts[stage],
            )
            retained_masses.append(retained_mass)
            if stage + 1 < self.steps and self.feedback_scale != 0.0:
                delta = (next_embeddings - current_embeddings).to(
                    dtype=hidden.dtype
                )
                hidden = hidden + self.feedback_scale * delta.unsqueeze(2)
                hidden = hidden * width_mask
            current_embeddings = next_embeddings

        entropies.append(route_entropy(occupancy))
        loads.append(occupancy.mean(dim=(0, 1, 2)))
        self.model.latest_route_statistics = {
            "route_entropy": torch.stack(entropies).mean(),
            "route_balance": load_balance(torch.stack(loads)),
        }
        self.model.latest_route_statistics_scope = "active_block"
        return (
            torch.stack(stage_logits),
            best_ids,
            torch.stack(retained_masses),
        )

    def _full_pass_forward(
        self,
        observed: Tensor,
        current_embeddings: Tensor,
        pure: Tensor,
        best_ids: Tensor,
        best_confidence: Tensor,
        *,
        alpha: Tensor,
        promotion_schedule: Tensor | None,
        inference_promotion_counts: tuple[int, ...],
        output_position_ids: Tensor | None,
        temperature: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        output_slots = current_embeddings.shape[1]
        stage_logits: list[Tensor] = []
        retained_masses: list[Tensor] = []
        for stage in range(self.steps):
            logits = self.model(
                observed,
                output_slots=output_slots,
                output_embeddings=current_embeddings,
                output_position_ids=output_position_ids,
                temperature=temperature,
            )
            stage_logits.append(logits)
            scheduled = (
                None
                if promotion_schedule is None
                else promotion_schedule[stage]
            )
            (
                pure,
                best_ids,
                best_confidence,
                current_embeddings,
                retained_mass,
            ) = self._evolve(
                logits,
                pure,
                best_ids,
                best_confidence,
                alpha=alpha,
                stage=stage,
                scheduled_promotions=scheduled,
                inference_promotion_tokens=inference_promotion_counts[stage],
            )
            retained_masses.append(retained_mass)
        return (
            torch.stack(stage_logits),
            best_ids,
            torch.stack(retained_masses),
        )

    def _full_pass_forward_cached(
        self,
        cache: GatorLinearBlockCache,
        current_embeddings: Tensor,
        pure: Tensor,
        best_ids: Tensor,
        best_confidence: Tensor,
        *,
        alpha: Tensor,
        promotion_schedule: Tensor | None,
        inference_promotion_counts: tuple[int, ...],
        temperature: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        stage_logits: list[Tensor] = []
        retained_masses: list[Tensor] = []
        for stage in range(self.steps):
            logits = self.model.forward_cached(
                cache,
                output_embeddings=current_embeddings,
                temperature=temperature,
            )
            stage_logits.append(logits)
            scheduled = (
                None
                if promotion_schedule is None
                else promotion_schedule[stage]
            )
            (
                pure,
                best_ids,
                best_confidence,
                current_embeddings,
                retained_mass,
            ) = self._evolve(
                logits,
                pure,
                best_ids,
                best_confidence,
                alpha=alpha,
                stage=stage,
                scheduled_promotions=scheduled,
                inference_promotion_tokens=inference_promotion_counts[stage],
            )
            retained_masses.append(retained_mass)
        return (
            torch.stack(stage_logits),
            best_ids,
            torch.stack(retained_masses),
        )

    def forward(
        self,
        observed: Tensor,
        initial_ids: Tensor,
        initial_pure: Tensor,
        *,
        alpha: Tensor,
        promotion_schedule: Tensor | None = None,
        inference_promotion_counts: Sequence[int] | None = None,
        inference_transition_tokens: int = 0,
        output_position_ids: Tensor | None = None,
        temperature: float = 1.0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return per-stage logits, historical-best IDs, and retained mass."""

        if observed.ndim != 2 or observed.dtype != torch.long:
            raise TypeError("observed must be long with [batch, slots] shape")
        if (
            initial_ids.ndim != 2
            or initial_ids.dtype != torch.long
            or initial_ids.shape[0] != observed.shape[0]
        ):
            raise TypeError(
                "initial_ids must be long with [batch, output_slots] shape"
            )
        if initial_pure.shape != initial_ids.shape or initial_pure.dtype != torch.bool:
            raise TypeError("initial_pure must be boolean and match initial_ids")
        if promotion_schedule is not None and (
            promotion_schedule.shape
            != (self.steps, *initial_ids.shape)
            or promotion_schedule.dtype != torch.bool
        ):
            raise TypeError(
                "promotion_schedule must be boolean [steps, batch, output_slots]"
            )
        if promotion_schedule is None:
            if (
                inference_promotion_counts is not None
                and inference_transition_tokens != 0
            ):
                raise ValueError(
                    "provide inference_promotion_counts or "
                    "inference_transition_tokens, not both"
                )
            if inference_promotion_counts is not None:
                resolved_promotion_counts = tuple(
                    int(value) for value in inference_promotion_counts
                )
            elif inference_transition_tokens > 0:
                resolved_promotion_counts = (
                    inference_transition_tokens,
                ) * self.steps
            else:
                resolved_promotion_counts = balanced_promotion_counts(
                    initial_ids.shape[1],
                    self.steps,
                )
            if (
                len(resolved_promotion_counts) != self.steps
                or any(value < 0 for value in resolved_promotion_counts)
                or sum(resolved_promotion_counts) != initial_ids.shape[1]
                or (
                    not torch.compiler.is_compiling()
                    and bool(initial_pure.any())
                )
            ):
                raise ValueError(
                    "inference requires an all-mask start and nonnegative "
                    "promotion counts that sum to output_slots"
                )
        else:
            if (
                inference_promotion_counts is not None
                or inference_transition_tokens != 0
            ):
                raise ValueError(
                    "inference promotion counts cannot accompany "
                    "a training promotion_schedule"
                )
            resolved_promotion_counts = (0,) * self.steps

        current_embeddings = F.embedding(
            initial_ids, self.model.token_embedding.weight
        )
        pure = initial_pure.clone()
        best_ids = initial_ids.clone()
        best_confidence = torch.zeros_like(initial_ids, dtype=torch.float32)
        kwargs = {
            "alpha": alpha,
            "promotion_schedule": promotion_schedule,
            "inference_promotion_counts": resolved_promotion_counts,
            "output_position_ids": output_position_ids,
            "temperature": temperature,
        }
        if self.mode == "relay":
            return self._relay_forward(
                observed,
                current_embeddings,
                pure,
                best_ids,
                best_confidence,
                **kwargs,
            )
        return self._full_pass_forward(
            observed,
            current_embeddings,
            pure,
            best_ids,
            best_confidence,
            **kwargs,
        )

    def forward_cached(
        self,
        cache: GatorLinearBlockCache,
        initial_ids: Tensor,
        initial_pure: Tensor,
        *,
        alpha: Tensor,
        promotion_schedule: Tensor | None = None,
        inference_promotion_counts: Sequence[int] | None = None,
        inference_transition_tokens: int = 0,
        temperature: float = 1.0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Refine one block against ``cache`` without committing proposals."""

        if not self.model.config.core.block_causal:
            raise ValueError("forward_cached requires block_causal=True")
        if not isinstance(cache, GatorLinearBlockCache):
            raise TypeError("cache must be a GatorLinearBlockCache")
        batch = cache.relays[0].mixer.node_sum.shape[0]
        if (
            initial_ids.ndim != 2
            or initial_ids.dtype != torch.long
            or initial_ids.shape != (batch, cache.block_size)
        ):
            raise TypeError(
                "initial_ids must be long [cache batch, cache block_size]"
            )
        if initial_pure.shape != initial_ids.shape or initial_pure.dtype != torch.bool:
            raise TypeError("initial_pure must be boolean and match initial_ids")
        if promotion_schedule is not None and (
            promotion_schedule.shape
            != (self.steps, *initial_ids.shape)
            or promotion_schedule.dtype != torch.bool
        ):
            raise TypeError(
                "promotion_schedule must be boolean [steps, batch, output_slots]"
            )
        if promotion_schedule is None:
            if (
                inference_promotion_counts is not None
                and inference_transition_tokens != 0
            ):
                raise ValueError(
                    "provide inference_promotion_counts or "
                    "inference_transition_tokens, not both"
                )
            if inference_promotion_counts is not None:
                resolved_promotion_counts = tuple(
                    int(value) for value in inference_promotion_counts
                )
            elif inference_transition_tokens > 0:
                resolved_promotion_counts = (
                    inference_transition_tokens,
                ) * self.steps
            else:
                resolved_promotion_counts = balanced_promotion_counts(
                    cache.block_size, self.steps
                )
            if (
                len(resolved_promotion_counts) != self.steps
                or any(value < 0 for value in resolved_promotion_counts)
                or sum(resolved_promotion_counts) != cache.block_size
                or (
                    not torch.compiler.is_compiling()
                    and bool(initial_pure.any())
                )
            ):
                raise ValueError(
                    "inference requires an all-mask start and nonnegative "
                    "promotion counts that sum to block_size"
                )
        else:
            if (
                inference_promotion_counts is not None
                or inference_transition_tokens != 0
            ):
                raise ValueError(
                    "inference promotion counts cannot accompany "
                    "a training promotion_schedule"
                )
            resolved_promotion_counts = (0,) * self.steps

        current_embeddings = F.embedding(
            initial_ids, self.model.token_embedding.weight
        )
        pure = initial_pure.clone()
        best_ids = initial_ids.clone()
        best_confidence = torch.zeros_like(initial_ids, dtype=torch.float32)
        kwargs = {
            "alpha": alpha,
            "promotion_schedule": promotion_schedule,
            "inference_promotion_counts": resolved_promotion_counts,
            "temperature": temperature,
        }
        if self.mode == "relay":
            return self._relay_forward_cached(
                cache,
                current_embeddings,
                pure,
                best_ids,
                best_confidence,
                **kwargs,
            )
        return self._full_pass_forward_cached(
            cache,
            current_embeddings,
            pure,
            best_ids,
            best_confidence,
            **kwargs,
        )
