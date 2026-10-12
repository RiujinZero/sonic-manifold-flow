"""Opt-in diagnostics for understanding Gator training dynamics."""

from .adamw_weight_trace import AdamWTraceSnapshot, AdamWWeightTracer
from .procoim_weight_trace import ProcoimTraceSnapshot, ProcoimWeightTracer

__all__ = [
    "AdamWTraceSnapshot",
    "AdamWWeightTracer",
    "ProcoimTraceSnapshot",
    "ProcoimWeightTracer",
]
