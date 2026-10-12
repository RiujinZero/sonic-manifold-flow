"""Correctness-first exact index used before ANN integration."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor


class HnswIndex:
    """Optional CPU HNSW candidate index with a tensor-compatible interface."""

    def __init__(
        self,
        centers: Tensor,
        *,
        ef_construction: int = 200,
        graph_degree: int = 16,
        ef_search: int = 128,
        seed: int = 17,
        threads: int = 0,
    ) -> None:
        if centers.ndim != 2 or centers.shape[0] == 0:
            raise ValueError("centers must have shape [basins, d] with basins > 0")
        if min(ef_construction, graph_degree, ef_search) <= 0 or seed < 0:
            raise ValueError("HNSW settings must be positive and seed non-negative")
        try:
            import hnswlib
        except ImportError as error:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "HNSW routing requires `pip install manifold-flow[ann]`"
            ) from error
        self._size = centers.shape[0]
        self._dimension = centers.shape[1]
        self._threads = threads
        self._index = hnswlib.Index(space="l2", dim=self.dimension)
        self._index.init_index(
            max_elements=self.size,
            ef_construction=ef_construction,
            M=graph_degree,
            random_seed=seed,
        )
        array = centers.detach().to(device="cpu", dtype=torch.float32).numpy()
        self._index.add_items(array, num_threads=threads)
        self._index.set_ef(max(ef_search, 1))

    @property
    def size(self) -> int:
        return self._size

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def index_bytes(self) -> int | None:
        """Return the serialized HNSW size when supported by hnswlib."""
        file_size = getattr(self._index, "index_file_size", None)
        return None if file_size is None else int(file_size())

    @torch.no_grad()
    def query(self, z: Tensor, k: int) -> Tensor:
        if z.ndim != 2 or z.shape[1] != self.dimension:
            raise ValueError("z must have shape [batch, index.dimension]")
        if not 1 <= k <= self.size:
            raise ValueError("k must be in [1, index.size]")
        query = z.detach().to(device="cpu", dtype=torch.float32).numpy()
        labels, _ = self._index.knn_query(query, k=k, num_threads=self._threads)
        return torch.from_numpy(labels.copy()).to(device=z.device, dtype=torch.long)


class AnnMainDeltaIndex:
    """Approximate immutable main index merged with exact current delta rows."""

    def __init__(self, main_centers: Tensor, main_index: object) -> None:
        if main_centers.ndim != 2 or main_centers.shape[0] == 0:
            raise ValueError("main_centers must have shape [basins, d]")
        if (
            getattr(main_index, "size", None) != main_centers.shape[0]
            or getattr(main_index, "dimension", None) != main_centers.shape[1]
        ):
            raise ValueError("main_index must match main_centers")
        self._main_centers = main_centers.detach().clone()
        self._main_index = main_index
        self._delta_ids = torch.empty(0, dtype=torch.long)
        self._delta_centers = self._main_centers.new_empty((0, self.dimension))

    @property
    def dimension(self) -> int:
        return self._main_centers.shape[1]

    @property
    def main_size(self) -> int:
        return self._main_centers.shape[0]

    @property
    def delta_size(self) -> int:
        return self._delta_ids.shape[0]

    @property
    def size(self) -> int:
        return self.main_size + int((self._delta_ids >= self.main_size).sum().item())

    @torch.no_grad()
    def upsert(self, ids: Tensor, centers: Tensor) -> None:
        if ids.ndim != 1 or centers.shape != (ids.shape[0], self.dimension):
            raise ValueError("ids and centers must have shapes [rows] and [rows, d]")
        if torch.any(ids < 0):
            raise ValueError("ids must be non-negative")
        ids = ids.to(device="cpu", dtype=torch.long)
        centers = centers.to(
            device=self._delta_centers.device, dtype=self._delta_centers.dtype
        )
        for item_id, center in zip(ids, centers, strict=True):
            integer_id = int(item_id.item())
            match = (self._delta_ids == item_id).nonzero(as_tuple=False).flatten()
            if match.numel():
                self._delta_centers[match[0]] = center
            else:
                if integer_id >= self.main_size and integer_id != self.size:
                    raise ValueError(
                        "new delta IDs must extend the dense bank in order"
                    )
                self._delta_ids = torch.cat((self._delta_ids, item_id[None]))
                self._delta_centers = torch.cat((self._delta_centers, center[None]))

    @torch.no_grad()
    def query(self, z: Tensor, k: int) -> Tensor:
        if z.ndim != 2 or z.shape[1] != self.dimension:
            raise ValueError("z must have shape [batch, index.dimension]")
        if not 1 <= k <= self.size:
            raise ValueError("k must be in [1, index.size]")
        overridden_main = self._delta_ids[self._delta_ids < self.main_size]
        main_k = min(self.main_size, k + overridden_main.numel())
        main_ids = self._main_index.query(z, main_k)
        main_centers = self._main_centers.to(device=z.device, dtype=z.dtype)[main_ids]
        main_distance = (z[:, None, :] - main_centers).square().sum(dim=-1)
        if overridden_main.numel():
            stale = torch.isin(main_ids, overridden_main.to(z.device))
            main_distance = main_distance.masked_fill(stale, torch.inf)
        ids = main_ids
        distance = main_distance
        if self.delta_size:
            delta_centers = self._delta_centers.to(device=z.device, dtype=z.dtype)
            delta_distance = torch.cdist(z, delta_centers).square()
            ids = torch.cat(
                (main_ids, self._delta_ids.to(z.device).expand(z.shape[0], -1)), dim=1
            )
            distance = torch.cat((main_distance, delta_distance), dim=1)
        keep = distance.topk(k=k, dim=-1, largest=False, sorted=True).indices
        return ids.gather(1, keep)


class ExactIndex:
    """Brute-force Euclidean candidate generation over immutable centers."""

    def __init__(self, centers: Tensor) -> None:
        if centers.ndim != 2 or centers.shape[0] == 0:
            raise ValueError("centers must have shape [basins, d] with basins > 0")
        self._centers = centers.detach()

    @property
    def size(self) -> int:
        return self._centers.shape[0]

    @property
    def dimension(self) -> int:
        return self._centers.shape[1]

    @torch.no_grad()
    def query(self, z: Tensor, k: int) -> Tensor:
        """Return IDs of the ``k`` nearest Euclidean centers for each latent."""
        if z.ndim != 2 or z.shape[1] != self.dimension:
            raise ValueError("z must have shape [batch, index.dimension]")
        if not 1 <= k <= self.size:
            raise ValueError("k must be in [1, index.size]")
        centers = self._centers.to(device=z.device, dtype=z.dtype)
        distance = torch.cdist(z, centers).square()
        return distance.topk(k=k, dim=-1, largest=False, sorted=True).indices


class ExactMainDeltaIndex:
    """Immutable exact main index with current-center overrides in a delta index."""

    def __init__(self, main_centers: Tensor) -> None:
        if main_centers.ndim != 2 or main_centers.shape[0] == 0:
            raise ValueError("main_centers must have shape [basins, d]")
        self._main_centers = main_centers.detach().clone()
        self._delta_ids = torch.empty(
            0, dtype=torch.long, device=self._main_centers.device
        )
        self._delta_centers = self._main_centers.new_empty((0, self.dimension))

    @property
    def dimension(self) -> int:
        return self._main_centers.shape[1]

    @property
    def main_size(self) -> int:
        return self._main_centers.shape[0]

    @property
    def delta_size(self) -> int:
        return self._delta_ids.shape[0]

    @property
    def size(self) -> int:
        new_ids = (self._delta_ids >= self.main_size).sum().item()
        return self.main_size + int(new_ids)

    @torch.no_grad()
    def upsert(self, ids: Tensor, centers: Tensor) -> None:
        """Insert new IDs or override moved main/delta centers."""
        if ids.ndim != 1 or centers.shape != (ids.shape[0], self.dimension):
            raise ValueError("ids and centers must have shapes [rows] and [rows, d]")
        if torch.any(ids < 0):
            raise ValueError("ids must be non-negative")
        ids = ids.to(device=self._delta_ids.device, dtype=torch.long)
        centers = centers.to(
            device=self._delta_centers.device, dtype=self._delta_centers.dtype
        )
        for basin_id, center in zip(ids, centers, strict=True):
            integer_id = int(basin_id.item())
            match = (self._delta_ids == basin_id).nonzero(as_tuple=False).flatten()
            if match.numel():
                self._delta_centers[match[0]] = center
            else:
                if integer_id >= self.main_size and integer_id != self.size:
                    raise ValueError(
                        "new delta IDs must extend the dense bank in order"
                    )
                self._delta_ids = torch.cat((self._delta_ids, basin_id[None]))
                self._delta_centers = torch.cat((self._delta_centers, center[None, :]))

    @torch.no_grad()
    def query(self, z: Tensor, k: int) -> Tensor:
        """Query main and delta, suppressing stale main rows overridden by delta."""
        if z.ndim != 2 or z.shape[1] != self.dimension:
            raise ValueError("z must have shape [batch, index.dimension]")
        if not 1 <= k <= self.size:
            raise ValueError("k must be in [1, index.size]")
        main_ids = torch.arange(self.main_size, device=z.device)
        main_centers = self._main_centers.to(device=z.device, dtype=z.dtype)
        if self.delta_size:
            delta_ids = self._delta_ids.to(z.device)
            keep_main = ~torch.isin(main_ids, delta_ids)
            ids = torch.cat((main_ids[keep_main], delta_ids))
            centers = torch.cat(
                (
                    main_centers[keep_main],
                    self._delta_centers.to(device=z.device, dtype=z.dtype),
                )
            )
        else:
            ids = main_ids
            centers = main_centers
        distance = torch.cdist(z, centers).square()
        local = distance.topk(k=k, dim=-1, largest=False, sorted=True).indices
        return ids[local]

    @torch.no_grad()
    def rebuild(self, centers: Tensor) -> None:
        """Publish current centers as a new immutable main snapshot."""
        if (
            centers.ndim != 2
            or centers.shape[0] == 0
            or centers.shape[1] != self.dimension
        ):
            raise ValueError("centers must have shape [basins, index.dimension]")
        self._main_centers = (
            centers.detach()
            .to(
                device=self._main_centers.device,
                dtype=self._main_centers.dtype,
            )
            .clone()
        )
        self._delta_ids = self._delta_ids.new_empty(0)
        self._delta_centers = self._delta_centers.new_empty((0, self.dimension))

    def serializable(self) -> dict[str, Tensor]:
        """Return CPU state sufficient to reconstruct this index exactly."""
        return {
            "main_centers": self._main_centers.detach().cpu(),
            "delta_ids": self._delta_ids.detach().cpu(),
            "delta_centers": self._delta_centers.detach().cpu(),
        }

    @classmethod
    def from_serialized(cls, payload: Mapping[str, Tensor]) -> ExactMainDeltaIndex:
        """Reconstruct an index exactly as `serializable()` captured it."""
        index = cls(payload["main_centers"])
        delta_ids = payload["delta_ids"]
        delta_centers = payload["delta_centers"]
        if delta_ids.numel():
            index.upsert(delta_ids, delta_centers)
        return index
