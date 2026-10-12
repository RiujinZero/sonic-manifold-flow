"""GTR-L: linear-scaling Gator.

Dense hypotheses, linear communication. See ``core`` for the relay and
``Papers/GATOR_LINEAR_DESIGN.md`` for the design rationale.
"""

from .coalition import coalition_report, effective_voters, format_report
from .core import (
    GatorLinearBlockCache,
    GatorLinearConfig,
    GatorLinearCore,
    LinearAttentionBlockCache,
    LinearRelay,
    LinearRelayBlockCache,
    LinearSequenceBlockCache,
    LinearSequenceMixer,
    detach_block_cache,
    fixed_address_code,
    straight_through_commit,
)
from .prefix import GatorLinearPrefixConfig, GatorLinearPrefixModel
from .evotoken import (
    EvoTokenRefiner,
    balanced_promotion_counts,
    masked_trajectory_cross_entropy,
    position_gradient_statistics,
    sample_masked_denoising_inputs,
    sample_training_schedule,
    soft_token_embeddings,
    topk_token_distribution,
)

__all__ = [
    "GatorLinearConfig",
    "GatorLinearCore",
    "GatorLinearBlockCache",
    "GatorLinearPrefixConfig",
    "GatorLinearPrefixModel",
    "EvoTokenRefiner",
    "LinearRelay",
    "LinearRelayBlockCache",
    "LinearSequenceBlockCache",
    "LinearSequenceMixer",
    "LinearAttentionBlockCache",
    "coalition_report",
    "balanced_promotion_counts",
    "masked_trajectory_cross_entropy",
    "position_gradient_statistics",
    "sample_masked_denoising_inputs",
    "sample_training_schedule",
    "soft_token_embeddings",
    "straight_through_commit",
    "topk_token_distribution",
    "effective_voters",
    "detach_block_cache",
    "fixed_address_code",
    "format_report",
]
