"""Public API for the ManifoldFlow mathematical core."""

from manifoldflow.aggregation import aggregate_flow_metrics
from manifoldflow.ann_benchmark import (
    benchmark_hnsw_routing,
    energy_reranked_top1,
    exact_energy_top1,
    exact_euclidean_top1,
    latency_distribution_ms,
)
from manifoldflow.backbone import Frozen224Backbone, imagenet_eval_transform
from manifoldflow.bank import BasinState
from manifoldflow.calibration import (
    DecoderCalibrationResult,
    NeutralDecoderCalibrationResult,
    calibrate_decoder,
    calibrate_neutral_decoder,
    evaluate_flow_ablation,
)
from manifoldflow.checkpoint import (
    latest_version,
    load_checkpoint,
    publish_checkpoint,
    rollback,
)
from manifoldflow.conditional import (
    ConditionalFlowResult,
    ConditionalGateCalibration,
    ConditionalGateSpec,
    RoutedFlowEvidence,
    apply_conditional_gate,
    calibrate_conditional_gate,
    calibrate_conditional_gate_grid,
    conditional_flow,
    routed_flow_evidence,
    semantic_gate_name,
    semantic_hybrid_name,
)
from manifoldflow.continual import (
    ContinualStreamSpec,
    class_conditioned_shift,
    evaluate_continual_latent_stream,
)
from manifoldflow.continual_aggregation import aggregate_continual_metrics
from manifoldflow.flow import analytic_topk_flow, basin_energy, positive_rate
from manifoldflow.gemma_world import (
    GemmaQueryProjector,
    GemmaWorldModel,
    LatentMemoryPort,
)
from manifoldflow.huggingface_stream import (
    batches_from_rows,
    streaming_imagenet_batches,
)
from manifoldflow.index import (
    AnnMainDeltaIndex,
    ExactIndex,
    ExactMainDeltaIndex,
    HnswIndex,
)
from manifoldflow.initialization import (
    InitializedBasinBank,
    initialize_basin_bank,
    mini_batch_kmeans,
    nearest_center_assignment,
)
from manifoldflow.matched_baselines import evaluate_matched_baselines
from manifoldflow.models import CompactImageAutoencoder
from manifoldflow.normalization import FrozenDiagonalWhitening
from manifoldflow.persistent_context import PersistentBasinContext
from manifoldflow.prototype import (
    nearest_prototype_class_scores,
    nearest_prototype_contraction,
    rbf_prototype_class_scores,
    rbf_prototype_contraction,
)
from manifoldflow.routing import FlowResult, exact_bank_flow, rerank_topk
from manifoldflow.selection import select_operating_points
from manifoldflow.sharded_latents import extract_sharded_latents, load_sharded_latents
from manifoldflow.streaming import (
    NoveltyProposal,
    NoveltyStager,
    OnlineBasinMemory,
    OnlineUpdateResult,
    calibrate_novelty_threshold,
)
from manifoldflow.streaming_baselines import OnlineClassCentroid, OnlinePrototypeMemory
from manifoldflow.synthetic_scale import (
    synthetic_gaussian_mixture_bank,
    synthetic_query_latents,
)
from manifoldflow.world_events import TabletopEventEngine
from manifoldflow.world_memory import WorldMemory, WorldStateEncoder
from manifoldflow.world_simulator import (
    ScenarioKind,
    SimulatorConfig,
    TabletopSimulator,
)
from manifoldflow.world_synthetic_training import (
    OracleWorldBatch,
    OracleWorldExample,
    collate_oracle_examples,
    load_oracle_examples,
)
from manifoldflow.world_types import (
    EventType,
    ObservationEvent,
    RelationRecord,
    VisibilityState,
    WorldEvent,
    WorldHit,
    WorldRecord,
    WorldSnapshot,
)

__all__ = [
    "BasinState",
    "AnnMainDeltaIndex",
    "CompactImageAutoencoder",
    "ConditionalFlowResult",
    "ConditionalGateCalibration",
    "ConditionalGateSpec",
    "ContinualStreamSpec",
    "DecoderCalibrationResult",
    "ExactIndex",
    "ExactMainDeltaIndex",
    "FlowResult",
    "Frozen224Backbone",
    "FrozenDiagonalWhitening",
    "HnswIndex",
    "InitializedBasinBank",
    "GemmaWorldModel",
    "GemmaQueryProjector",
    "LatentMemoryPort",
    "NeutralDecoderCalibrationResult",
    "NoveltyProposal",
    "NoveltyStager",
    "OnlineBasinMemory",
    "OnlineClassCentroid",
    "OnlinePrototypeMemory",
    "OnlineUpdateResult",
    "OracleWorldBatch",
    "OracleWorldExample",
    "PersistentBasinContext",
    "TabletopEventEngine",
    "TabletopSimulator",
    "RoutedFlowEvidence",
    "ScenarioKind",
    "SimulatorConfig",
    "EventType",
    "ObservationEvent",
    "RelationRecord",
    "VisibilityState",
    "WorldEvent",
    "WorldHit",
    "WorldMemory",
    "WorldRecord",
    "WorldSnapshot",
    "WorldStateEncoder",
    "apply_conditional_gate",
    "analytic_topk_flow",
    "aggregate_flow_metrics",
    "aggregate_continual_metrics",
    "basin_energy",
    "batches_from_rows",
    "benchmark_hnsw_routing",
    "calibrate_decoder",
    "calibrate_conditional_gate",
    "calibrate_conditional_gate_grid",
    "calibrate_neutral_decoder",
    "calibrate_novelty_threshold",
    "class_conditioned_shift",
    "collate_oracle_examples",
    "conditional_flow",
    "exact_bank_flow",
    "evaluate_flow_ablation",
    "evaluate_matched_baselines",
    "evaluate_continual_latent_stream",
    "energy_reranked_top1",
    "exact_energy_top1",
    "exact_euclidean_top1",
    "extract_sharded_latents",
    "imagenet_eval_transform",
    "initialize_basin_bank",
    "latency_distribution_ms",
    "latest_version",
    "load_checkpoint",
    "load_oracle_examples",
    "load_sharded_latents",
    "mini_batch_kmeans",
    "nearest_center_assignment",
    "nearest_prototype_class_scores",
    "nearest_prototype_contraction",
    "positive_rate",
    "publish_checkpoint",
    "rerank_topk",
    "rbf_prototype_class_scores",
    "rbf_prototype_contraction",
    "rollback",
    "routed_flow_evidence",
    "select_operating_points",
    "semantic_gate_name",
    "semantic_hybrid_name",
    "streaming_imagenet_batches",
    "synthetic_gaussian_mixture_bank",
    "synthetic_query_latents",
]
