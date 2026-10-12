"""Surprise-Gated Block Memory (SGBM) for GTR-L.

A Titans-inspired memory subsystem: recent blocks stay in the existing
exact block cache, older/important information compresses into a
surprise-gated session memory that is writable at inference (no backward
pass), and a small quarantine buffer holds high-surprise-but-unconfirmed
events until they are either promoted or expire.

See the design specification this package implements for the full
rationale, state definitions, and phased build order.

Implementation status: Phases A-E (config/state, store/retrieval/
quarantine mechanics, surprise/reliability signals, block predictor/
summarizer/controller, and the top-level SurpriseGatedMemory read/write
API). Not yet implemented: actual GTR-L forward-pass integration (Phase F)
and episodic training support (Phase G) — see TaskList.
"""

from gator.memory.config import SurpriseMemoryConfig
from gator.memory.state import SurpriseMemoryState
from gator.memory.surprise_gated_memory import SurpriseGatedMemory

__all__ = ["SurpriseMemoryConfig", "SurpriseMemoryState", "SurpriseGatedMemory"]
