"""Modality-independent memory boundary (design-v0.3.md SS5, SS15) and its
implementation through SS9 Phase 5 ("Continual write lifecycle").

`BasinMemory` stores inspectable text payloads keyed by geometric basins.
It deliberately does *not* reuse `OnlineBasinMemory`/`NoveltyStager`
(`streaming.py`) for basin creation: that machinery requires
`persistence>=2` repeated, spatially-compatible observations before
promoting a new basin - correct for the vision domain's noise-robust
online learning, but the opposite of SS5.1's "recent microbasins support
**one-shot acquisition**." A single stated fact must be immediately
retrievable, not wait for the same fact to recur three times. This was
discovered, not assumed (Phase 2): an early version delegated to
`OnlineBasinMemory.update()` and every single-shot write past the first
came back rejected. Reusing `BasinState.append_statistics`/
`update_winners` and `ExactMainDeltaIndex` directly (both already used by
`OnlineBasinMemory` itself) reimplements only the creation policy, not the
geometry primitives. Repeated evidence still reinforces an existing basin
via the normal Welford update - only *creation* is one-shot. Matches
SS6.4's read/write separation rule: writes always come from the raw key,
never a context-blended query.

Phase 5 adds the write lifecycle SS5.1/SS5.3/SS7.3 describe:

- **Trust-tier durability** (SS5.3): "Qwen's unsupported generated claims
  must not become durable semantic memory." `write()` rejects any
  `Source.trust_tier` outside `DURABLE_TRUST_TIERS` - an ephemeral claim
  is simply not persisted here at all, rather than this module growing a
  second, separate ephemeral working-memory store (out of scope for this
  pass; recorded as a real limitation, not hidden).
- **Repetition-based promotion** (SS5.1): a basin's *active* (non-
  superseded, non-expired) record count decides its `tier()` - `"recent"`
  (a microbasin, one-shot, provisional) until `promotion_threshold`
  reinforcing writes land on it, then `"consolidated"`.
- **Expiration** (SS5.1's "expires -> discard"): `expire_stale_microbasins()`
  marks still-`"recent"` basins whose records have gone
  `microbasin_max_age` writes without reinforcement as expired - their
  records stop being returned by `read()`/`payloads_for()`, but the
  underlying geometric basin is left in the index. Removing basin *rows*
  is SS5.2's prune/tombstone/quantize consolidator, explicitly Phase 6
  ("one operation at a time after versioning"), not this phase's job.
- **Supersession** (SS7.3): `write(..., supersedes=(basin_id, index))`
  marks a specific prior record superseded and links the new one to it.
  There is no automatic "this new fact corrects that old one" detection -
  real entity/relation extraction is out of scope here, so the caller
  (whoever already knows a correction is happening, e.g. because a user
  said "actually...") must identify which record to supersede, typically
  via a prior `read()`. `read()`/`payloads_for()` exclude superseded
  records by default, matching "retrieval excludes superseded values by
  default while retaining history for auditing."

Phase 6 adds the structural side SS5.2/SS9 describe as "snapshot the
active bank... rebuild and atomically swap the main index": both
`prune_expired_basins()` and `merge_compatible_recent_basins()` build an
entirely new `BasinState`/index from a plan computed up front, rather
than mutating rows in place - the rebuild-based model the design doc
itself specifies, not an invented alternative. `consolidate()` (SS15's
interface) runs both in sequence and returns the new bank version.
Merging combines sufficient statistics via the standard parallel-Welford
formula (exact, not an approximation) and only ever considers `"recent"`
tier basins, never `"consolidated"` ones - SS5.1's "without immediately
deforming a mature semantic centroid" applies to merging as much as it
does to creation. **Split is deliberately not implemented**: SS5.2 itself
says split needs "streaming subcentroids/sketch... before split logic,"
infrastructure this project does not have yet - attempting it without
that foundation would be guessing, not implementing the design.

`qwen-phase6-consolidation-scale.md` measured merge at 77.7 seconds for
10,000 basins - long enough that running it inline on a live session
would stall every read for over a minute, contradicting this project's
whole speed emphasis. `consolidate_in_background()` runs consolidation on
a daemon thread instead - the "sleep state" this memory enters
periodically rather than a synchronous step in the request path. `read()`
never touches the consolidation lock and keeps working throughout,
against whichever `_bank`/`_index` reference is currently live - CPython's
GIL makes a plain attribute reassignment (`self._bank = new_bank`) atomic,
so a concurrent read sees either the fully-old or fully-new bank, never a
torn one, without this needing an explicit read lock. `write()` *does*
take the lock and raises rather than blocks or silently drops the write if
consolidation is already running - a write racing a live rebuild has
nowhere correct to land (the rebuild's snapshot was already taken from the
state before the write happened), and rejecting loudly is safer than
losing it silently. Real concurrent reads-during-consolidation are tested
directly, not just checked immediately before and after
(`docs/results/qwen-phase6b-background-consolidation.md`) - closing a gap
the original Phase 6 benchmark left open by construction, since a
synchronous `consolidate()` call has no window for concurrent access to
even occur.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
from torch import Tensor

from manifoldflow.bank import BasinState
from manifoldflow.index import ExactMainDeltaIndex

DURABLE_TRUST_TIERS = frozenset(
    {"user", "corrected", "imported", "verified", "confirmed"}
)


@dataclass(frozen=True)
class Source:
    """Provenance for one write (SS5.3: only certain trust tiers are durable)."""

    trust_tier: str


@dataclass
class MemoryRecord:
    """One inspectable, versioned entry at a basin (SS5.2's record schema,
    scoped to the fields Phase 5's lifecycle actually needs)."""

    text: str
    trust_tier: str
    created_step: int
    superseded: bool = False
    expired: bool = False
    supersedes: tuple[int, int] | None = None

    @property
    def active(self) -> bool:
        return not self.superseded and not self.expired


@dataclass(frozen=True)
class MemoryHit:
    basin_id: int
    score: float
    payloads: list[str]


class ContinuousMemory(ABC):
    """Read/write boundary between a `RepresentationAdapter` and a memory bank.

    Any implementation (this bank now; a two-tier, context-aware,
    consolidating one in later phases) exposes the same five operations so
    the representation side of the system never depends on which memory
    implementation is behind it.
    """

    @abstractmethod
    def read(self, raw_key: Tensor, *, session_id: str, top_k: int) -> list[MemoryHit]:
        """Return up to `top_k` basins ranked by relevance to `raw_key`."""

    @abstractmethod
    def write(self, raw_key: Tensor, payload: str, source: Source) -> str:
        """Store `payload` at the basin `raw_key` routes to; return its id."""

    @abstractmethod
    def advance_context(self, raw_key: Tensor, *, session_id: str) -> None:
        """Update any per-session temporal bias. No-op until context exists."""

    @abstractmethod
    def reset_context(self, session_id: str) -> None:
        """Clear any per-session temporal bias."""

    @abstractmethod
    def consolidate(self) -> int:
        """Run any pending structural maintenance; return the bank version."""


@dataclass
class BasinMemory(ContinuousMemory):
    """Basin-keyed payload store with the Phase 5 write lifecycle.

    Lazily seeds its `BasinState` from the first write, since `BasinState`
    requires at least one basin (`bank.py`'s `__post_init__`) and this
    memory has no pre-existing corpus to initialize centers from - every
    fact it knows about arrived through `write()`. `advance_context` and
    `reset_context` are no-ops: session-scoped persistent context is
    layered on top by `persistent_context.PersistentBasinContext` /
    `adaptive_context.AdaptiveBasinContext`, not reimplemented here.
    """

    dimension: int
    novelty_threshold: float
    max_basins: int = 100_000
    promotion_threshold: int = 2
    microbasin_max_age: int = 50
    merge_threshold: float | None = None
    _bank: BasinState | None = field(default=None, init=False, repr=False)
    _index: ExactMainDeltaIndex | None = field(default=None, init=False, repr=False)
    _records: dict[int, list[MemoryRecord]] = field(
        default_factory=dict, init=False, repr=False
    )
    _last_touched_step: dict[int, int] = field(
        default_factory=dict, init=False, repr=False
    )
    _step: int = field(default=0, init=False, repr=False)
    _consolidation_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")
        if self.novelty_threshold < 0:
            raise ValueError("novelty_threshold must be non-negative")
        if self.promotion_threshold <= 0:
            raise ValueError("promotion_threshold must be positive")
        if self.microbasin_max_age <= 0:
            raise ValueError("microbasin_max_age must be positive")
        if self.merge_threshold is None:
            self.merge_threshold = self.novelty_threshold
        elif self.merge_threshold < 0:
            raise ValueError("merge_threshold must be non-negative")

    @classmethod
    def from_bulk(
        cls, centers: Tensor, payloads: list[str], source: Source, **kwargs: object
    ) -> BasinMemory:
        """Construct directly from a batch of centers/payloads (SS5.3's
        "imported trusted material"), one basin per row - vectorized bank
        construction, not a loop of individual `write()` calls. `write()`'s
        per-call index query makes it unsuitable for bulk loading at real
        scale: `ExactMainDeltaIndex` only folds delta rows back into the
        main index on `consolidate()`, so an uninterrupted `write()` loop
        leaves an ever-growing, never-rebuilt delta and degrades toward
        O(n^2) - the same reason `synthetic_scale.py`/`ann_scale_sweep.py`
        build large synthetic banks directly rather than one write at a
        time."""
        if centers.ndim != 2 or centers.shape[0] == 0:
            raise ValueError("centers must have shape [basins, d] with basins > 0")
        if len(payloads) != centers.shape[0]:
            raise ValueError("payloads must have one entry per center row")
        if source.trust_tier not in DURABLE_TRUST_TIERS:
            raise ValueError(
                f"trust_tier {source.trust_tier!r} is not durable (SS5.3); "
                f"only {sorted(DURABLE_TRUST_TIERS)} may create semantic memory"
            )
        memory = cls(dimension=centers.shape[1], **kwargs)  # type: ignore[arg-type]
        # from_statistics with count=1 per row, not from_centers
        # (count=0) - each row is one real observation, and merge's
        # parallel-Welford combine hits 0/0 on two never-reinforced
        # count=0 basins (see write()'s matching comment).
        cloned_centers = centers.detach().clone()
        memory._bank = BasinState.from_statistics(
            cloned_centers,
            torch.ones(cloned_centers.shape[0], dtype=torch.long),
            torch.zeros_like(cloned_centers),
        )
        memory._index = ExactMainDeltaIndex(memory._bank.mu)
        for basin_id, payload in enumerate(payloads):
            memory._step += 1
            memory._records[basin_id] = [
                MemoryRecord(
                    text=payload,
                    trust_tier=source.trust_tier,
                    created_step=memory._step,
                )
            ]
            memory._last_touched_step[basin_id] = memory._step
        return memory

    @property
    def basin_count(self) -> int:
        return 0 if self._bank is None else self._bank.mu.shape[0]

    @property
    def centers(self) -> Tensor:
        """Current basin centers, for wrapping this memory's routing in a
        `persistent_context.PersistentBasinContext` once the bank is built
        (Phase 3): that class takes a fixed `centers` tensor and index at
        construction, matching this static (`alpha=1`) bank's own
        assumption that writes finish before a context sweep begins."""
        if self._bank is None:
            raise RuntimeError("no basins exist yet; call write() first")
        return self._bank.mu

    @property
    def index(self) -> ExactMainDeltaIndex:
        if self._index is None:
            raise RuntimeError("no basins exist yet; call write() first")
        return self._index

    @torch.no_grad()
    def write(
        self,
        raw_key: Tensor,
        payload: str,
        source: Source,
        *,
        supersedes: tuple[int, int] | None = None,
    ) -> str:
        if raw_key.shape != (self.dimension,):
            raise ValueError(f"raw_key must have shape [{self.dimension}]")
        if not payload:
            raise ValueError("payload must be non-empty")
        if source.trust_tier not in DURABLE_TRUST_TIERS:
            raise ValueError(
                f"trust_tier {source.trust_tier!r} is not durable (SS5.3); "
                f"only {sorted(DURABLE_TRUST_TIERS)} may create semantic memory"
            )
        if supersedes is not None:
            target_basin, target_index = supersedes
            target_records = self._records.get(target_basin, [])
            if not 0 <= target_index < len(target_records):
                raise ValueError(f"no record at supersedes={supersedes!r}")

        if not self._consolidation_lock.acquire(blocking=False):
            raise RuntimeError(
                "cannot write while consolidation is in progress (sleep state); "
                "retry after it finishes"
            )
        try:
            return self._write_locked(raw_key, payload, source, supersedes=supersedes)
        finally:
            self._consolidation_lock.release()

    def _write_locked(
        self,
        raw_key: Tensor,
        payload: str,
        source: Source,
        *,
        supersedes: tuple[int, int] | None,
    ) -> str:
        """The actual write, run under `_consolidation_lock` so it can
        never interleave with a `consolidate()` rebuild in progress."""
        self._step += 1
        key = raw_key.detach().clone()

        if self._bank is None:
            # from_statistics with count=1, not from_centers (count=0,
            # meant for "seeded but not yet observed") - this basin
            # represents one real write, and a real bug (found via the
            # concurrency test below) showed why the distinction matters:
            # merging two never-reinforced count=0 basins hits 0/0 in the
            # parallel-Welford combine.
            self._bank = BasinState.from_statistics(
                key[None, :], torch.tensor([1]), torch.zeros_like(key)[None, :]
            )
            self._index = ExactMainDeltaIndex(self._bank.mu)
            basin_id = 0
        else:
            assert self._index is not None
            winner = int(self._index.query(key[None, :], 1).item())
            center = self._bank.mu[winner].to(dtype=key.dtype)
            squared_distance = float((key - center).square().sum().item())
            if squared_distance <= self.novelty_threshold:
                self._bank.update_winners(key[None, :], torch.tensor([winner]))
                self._index.upsert(
                    torch.tensor([winner]), self._bank.mu[winner][None, :]
                )
                basin_id = winner
            elif self._bank.mu.shape[0] >= self.max_basins:
                raise RuntimeError("write rejected: bank is at max_basins capacity")
            else:
                basin_id = self._bank.append_statistics(
                    key, count=1, m2=torch.zeros_like(key)
                )
                self._index.upsert(
                    torch.tensor([basin_id]), self._bank.mu[basin_id][None, :]
                )

        if supersedes is not None:
            target_basin, target_index = supersedes
            self._records[target_basin][target_index].superseded = True

        record = MemoryRecord(
            text=payload,
            trust_tier=source.trust_tier,
            created_step=self._step,
            supersedes=supersedes,
        )
        self._records.setdefault(basin_id, []).append(record)
        self._last_touched_step[basin_id] = self._step
        return str(basin_id)

    def write_always_new(self, raw_key: Tensor, payload: str, source: Source) -> str:
        """Equivalent to `write()` with `supersedes=None`, but skips the
        nearest-existing-basin novelty check entirely - only valid when
        `self.novelty_threshold == 0.0` (asserted below), in which case
        `write()`'s own `squared_distance <= self.novelty_threshold`
        branch is unreachable anyway (a squared distance between two
        distinct float embeddings is never exactly <= 0.0), so every write
        already takes the "append a new basin" branch regardless of the
        neighbor search's result - that search (one `.item()` for the
        nearest id, one more for the squared distance) is pure unused
        overhead in that configuration. Exists for callers juggling many
        banks per step (this is exactly ManifoldFlow's per-lane training
        loop's situation, which always constructs its banks with
        `novelty_threshold=0.0`) where those two guaranteed-irrelevant
        CPU-GPU syncs, paid once per bank per step, were a real, confirmed
        stall source."""
        if self.novelty_threshold != 0.0:
            raise ValueError(
                "write_always_new requires novelty_threshold == 0.0 (the "
                "novelty check it skips is only provably a no-op then) - "
                "use write() for a bank with a nonzero threshold"
            )
        if raw_key.shape != (self.dimension,):
            raise ValueError(f"raw_key must have shape [{self.dimension}]")
        if not payload:
            raise ValueError("payload must be non-empty")
        if source.trust_tier not in DURABLE_TRUST_TIERS:
            raise ValueError(
                f"trust_tier {source.trust_tier!r} is not durable (SS5.3); "
                f"only {sorted(DURABLE_TRUST_TIERS)} may create semantic memory"
            )
        if not self._consolidation_lock.acquire(blocking=False):
            raise RuntimeError(
                "cannot write while consolidation is in progress (sleep state); "
                "retry after it finishes"
            )
        try:
            self._step += 1
            key = raw_key.detach().clone()
            if self._bank is None:
                self._bank = BasinState.from_statistics(
                    key[None, :], torch.tensor([1]), torch.zeros_like(key)[None, :]
                )
                self._index = ExactMainDeltaIndex(self._bank.mu)
                basin_id = 0
            else:
                assert self._index is not None
                if self._bank.mu.shape[0] >= self.max_basins:
                    raise RuntimeError("write rejected: bank is at max_basins capacity")
                basin_id = self._bank.append_statistics(
                    key, count=1, m2=torch.zeros_like(key)
                )
                self._index.upsert(
                    torch.tensor([basin_id]), self._bank.mu[basin_id][None, :]
                )
            record = MemoryRecord(
                text=payload,
                trust_tier=source.trust_tier,
                created_step=self._step,
                supersedes=None,
            )
            self._records.setdefault(basin_id, []).append(record)
            self._last_touched_step[basin_id] = self._step
            return str(basin_id)
        finally:
            self._consolidation_lock.release()

    def tier(self, basin_id: int) -> str:
        """`"recent"` (microbasin, provisional) or `"consolidated"`
        (promoted, SS5.1) based on how many active records a basin has."""
        active_count = sum(1 for r in self._records.get(basin_id, []) if r.active)
        return "consolidated" if active_count >= self.promotion_threshold else "recent"

    def active_records(self, basin_id: int) -> list[tuple[int, MemoryRecord]]:
        """`(index, record)` pairs for a basin's active (current, not
        superseded or expired) records, indices stable for use as a
        `write(..., supersedes=...)` target."""
        return [
            (index, record)
            for index, record in enumerate(self._records.get(basin_id, []))
            if record.active
        ]

    def expire_stale_microbasins(self) -> list[int]:
        """Mark still-`"recent"` basins untouched for `microbasin_max_age`
        writes as expired (SS5.1: "expires -> discard"). Returns the basin
        ids affected. Only marks records inactive; the geometric basin
        itself is left in the index (SS5.2's structural pruning is Phase
        6, not this)."""
        expired_basin_ids = []
        for basin_id, records in self._records.items():
            if self.tier(basin_id) != "recent":
                continue
            last_touched = self._last_touched_step.get(basin_id, 0)
            if self._step - last_touched <= self.microbasin_max_age:
                continue
            newly_expired = False
            for record in records:
                if record.active:
                    record.expired = True
                    newly_expired = True
            if newly_expired:
                expired_basin_ids.append(basin_id)
        return expired_basin_ids

    @torch.no_grad()
    def read(
        self, raw_key: Tensor, *, session_id: str = "", top_k: int = 4
    ) -> list[MemoryHit]:
        del session_id  # no session-scoped context in this bank
        if raw_key.shape != (self.dimension,):
            raise ValueError(f"raw_key must have shape [{self.dimension}]")
        if self._bank is None:
            return []
        assert self._index is not None
        k = min(top_k, self._bank.mu.shape[0])
        ids = self._index.query(raw_key[None, :], k).squeeze(0)
        centers = self._bank.mu[ids].to(dtype=raw_key.dtype)
        # Keys are L2-normalized (representation.py), so dot product is
        # cosine similarity and ranks identically to Euclidean nearest
        # neighbor here - there is no separate "RBF ranking" to validate
        # against for top-k retrieval order, only for a weighted mixture
        # over multiple hits, which this static read does not need.
        scores = (centers * raw_key[None, :]).sum(dim=-1)
        return [
            MemoryHit(
                basin_id=int(basin_id),
                score=float(score),
                payloads=self.payloads_for(int(basin_id)),
            )
            for basin_id, score in zip(ids.tolist(), scores.tolist(), strict=True)
        ]

    @torch.no_grad()
    def read_raw(self, raw_key: Tensor, *, top_k: int) -> tuple[Tensor, Tensor] | None:
        """Same nearest-neighbor lookup `read()` performs, but returns the
        raw `(ids, scores)` GPU tensors instead of converting them to a
        `list[MemoryHit]` - `read()`'s `ids.tolist()`/`scores.tolist()`
        forces a CPU-GPU sync on every call, and a caller juggling many
        independent banks at once (e.g. one per training-loop lane) can
        stack many banks' `read_raw()` results and pay that sync exactly
        once for the whole batch instead of once per bank. Returns `None`
        when the bank is empty, mirroring `read()`'s empty-list case."""
        if raw_key.shape != (self.dimension,):
            raise ValueError(f"raw_key must have shape [{self.dimension}]")
        if self._bank is None:
            return None
        assert self._index is not None
        k = min(top_k, self._bank.mu.shape[0])
        ids = self._index.query(raw_key[None, :], k).squeeze(0)
        centers = self._bank.mu[ids].to(dtype=raw_key.dtype)
        scores = (centers * raw_key[None, :]).sum(dim=-1)
        return ids, scores

    def advance_context(self, raw_key: Tensor, *, session_id: str) -> None:
        del raw_key, session_id

    def reset_context(self, session_id: str) -> None:
        del session_id

    def _rebuild_from_groups(self, groups: list[list[int]]) -> dict[int, int]:
        """Rebuild bank/index/records from basin-id groups - each group's
        members combine into one new basin via the exact parallel-Welford
        merge formula for (center, count, m2). Singleton groups just
        survive under a (possibly renumbered) id. SS9 Phase 6: "snapshot
        the active bank... rebuild and atomically swap the main index."
        """
        assert self._bank is not None
        old_to_new: dict[int, int] = {}
        centers: list[Tensor] = []
        counts: list[int] = []
        m2s: list[Tensor] = []
        new_records: dict[int, list[MemoryRecord]] = {}
        new_last_touched: dict[int, int] = {}

        for new_id, group in enumerate(groups):
            combined_center = self._bank.mu[group[0]].clone()
            combined_count = int(self._bank.count[group[0]].item())
            combined_m2 = self._bank.m2[group[0]].clone()
            for old_id in group[1:]:
                other_center = self._bank.mu[old_id]
                other_count = int(self._bank.count[old_id].item())
                other_m2 = self._bank.m2[old_id]
                total = combined_count + other_count
                delta = other_center - combined_center
                combined_center = combined_center + delta * (other_count / total)
                combined_m2 = (
                    combined_m2
                    + other_m2
                    + delta.square() * (combined_count * other_count / total)
                )
                combined_count = total
            centers.append(combined_center)
            counts.append(max(1, combined_count))
            m2s.append(combined_m2)

            merged_records: list[MemoryRecord] = []
            latest_touch = 0
            for old_id in group:
                old_to_new[old_id] = new_id
                merged_records.extend(self._records.get(old_id, []))
                latest_touch = max(latest_touch, self._last_touched_step.get(old_id, 0))
            new_records[new_id] = merged_records
            new_last_touched[new_id] = latest_touch

        new_bank = BasinState.from_statistics(
            torch.stack(centers),
            torch.tensor(counts, dtype=torch.long),
            torch.stack(m2s),
        )
        new_bank.version = self._bank.version + 1
        self._bank = new_bank
        self._index = ExactMainDeltaIndex(new_bank.mu)
        self._records = new_records
        self._last_touched_step = new_last_touched
        return old_to_new

    def prune_expired_basins(self) -> dict[int, int]:
        """Remove basins with no active records (SS5.1: "expires ->
        discard"), rebuilding the bank/index without them. Returns
        old_basin_id -> new_basin_id for surviving basins whose id
        changed. A no-op if nothing is expired."""
        if self._bank is None:
            return {}
        surviving = [
            basin_id
            for basin_id in range(self._bank.mu.shape[0])
            if any(record.active for record in self._records.get(basin_id, []))
        ]
        if len(surviving) == self._bank.mu.shape[0]:
            return {}
        if not surviving:
            self._bank = None
            self._index = None
            self._records = {}
            self._last_touched_step = {}
            return {}
        return self._rebuild_from_groups([[basin_id] for basin_id in surviving])

    def merge_compatible_recent_basins(
        self, merge_threshold: float, *, chunk_size: int = 2000
    ) -> dict[int, int]:
        """Greedily merge pairs of `"recent"`-tier basins within
        `merge_threshold` squared distance of each other (SS5.1: "merge
        compatible microbasins"). `"consolidated"` basins are never
        touched. Each basin merges at most once per call - a chain of
        three mutually close basins needs two `consolidate()` calls, kept
        simple deliberately rather than guessing at transitive-merge
        semantics. Returns old_basin_id -> new_basin_id; a no-op (`{}`) if
        nothing qualifies.

        Pairwise distance is computed in row chunks, not one
        `n_recent x n_recent` matrix - a full matrix at real bank scale
        (100K recent basins would need a 100,000 x 100,000 float matrix,
        40GB) is not just slow, it does not fit in memory at all. This
        keeps time complexity O(n_recent^2) - still expensive at very
        large recent-tier counts, and reported honestly as a real
        scale limit of the current exact-index-only implementation
        (`docs/results/qwen-phase6-consolidation-scale.md`), not hidden -
        but bounds peak memory to `chunk_size * n_recent` instead."""
        if merge_threshold < 0:
            raise ValueError("merge_threshold must be non-negative")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self._bank is None or self._bank.mu.shape[0] < 2:
            return {}
        recent_ids = [
            basin_id
            for basin_id in range(self._bank.mu.shape[0])
            if self.tier(basin_id) == "recent"
        ]
        if len(recent_ids) < 2:
            return {}

        recent_centers = self._bank.mu[recent_ids]
        candidate_pairs: list[tuple[float, int, int]] = []
        for start in range(0, len(recent_ids), chunk_size):
            chunk = recent_centers[start : start + chunk_size]
            squared_distance = torch.cdist(chunk, recent_centers).square()
            for local_i in range(squared_distance.shape[0]):
                i = start + local_i
                for j in range(i + 1, len(recent_ids)):
                    distance = float(squared_distance[local_i, j].item())
                    if distance <= merge_threshold:
                        candidate_pairs.append((distance, recent_ids[i], recent_ids[j]))
        candidate_pairs.sort(key=lambda item: item[0])

        all_ids = range(self._bank.mu.shape[0])
        groups: list[list[int]] = [[basin_id] for basin_id in all_ids]
        group_of = {basin_id: basin_id for basin_id in all_ids}
        consumed: set[int] = set()
        merged_any = False
        for _, first, second in candidate_pairs:
            if first in consumed or second in consumed:
                continue
            keep_group, drop_group = group_of[first], group_of[second]
            groups[keep_group].extend(groups[drop_group])
            groups[drop_group] = []
            for member in groups[keep_group]:
                group_of[member] = keep_group
            consumed.add(first)
            consumed.add(second)
            merged_any = True

        if not merged_any:
            return {}
        return self._rebuild_from_groups([group for group in groups if group])

    def consolidate(self) -> int:
        """Prune expired basins, then merge compatible recent ones -
        SS15's structural-maintenance operation. Split is not implemented
        (see module docstring). Blocks concurrent `write()` calls (which
        raise rather than wait) for its whole duration via
        `_consolidation_lock`; `read()` is untouched by the lock and keeps
        working throughout - see the module docstring for why that split is
        safe. Runs synchronously on the calling thread; for anything large
        enough that "synchronously" means seconds or minutes
        (`docs/results/qwen-phase6-consolidation-scale.md`), call
        `consolidate_in_background()` instead."""
        with self._consolidation_lock:
            self.prune_expired_basins()
            if self._bank is not None:
                assert self.merge_threshold is not None
                self.merge_compatible_recent_basins(self.merge_threshold)
            return 0 if self._bank is None else self._bank.version

    def consolidate_in_background(self) -> threading.Thread:
        """Run `consolidate()` on a daemon thread instead of the caller's -
        the "sleep state" SS9 Phase 6's own scale numbers make clear a live
        session needs, rather than a synchronous step in the read/write
        path. Returns the started thread; call `.join()` on it to wait for
        completion, or ignore it to let consolidation run unattended.
        Concurrent `read()` calls keep working (see module docstring);
        concurrent `write()` calls raise until this thread finishes."""
        thread = threading.Thread(target=self.consolidate, daemon=True)
        thread.start()
        return thread

    def payloads_for(self, basin_id: int) -> list[str]:
        """Active (current, not superseded or expired) payload texts at
        one basin, for callers routing directly through `centers`/`index`
        (e.g. `PersistentBasinContext`) rather than `read()`."""
        return [r.text for r in self._records.get(basin_id, []) if r.active]
