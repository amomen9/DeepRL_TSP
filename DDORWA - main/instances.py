"""Instance registry and random-capacity generators.

Built-in and random instances use the same three-matrix design:
  distance_matrix   : base deterministic travel times;
  max_delay_matrix  : maximum possible delay for every directed edge;
  delay_mask        : 1 only on directed edges where delay is active.

For built-ins, uncertain_routes is kept for readability.  For random instances,
delay_mask is generated directly and is not forced to be symmetric.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence
import json
import numpy as np


@dataclass
class TSPInstance:
    id: str
    name: str
    distance_matrix: np.ndarray
    max_delay_matrix: np.ndarray | None = None
    uncertain_routes: list[tuple[int, int]] | None = None  # 1-indexed directed edges
    delay_mask: np.ndarray | None = None                   # directed 0/1 matrix
    delay_distribution: str = "uniform"
    aliases: tuple[str, ...] = ()

    @property
    def n(self) -> int:
        return int(self.distance_matrix.shape[0])

    def env_kwargs(self, delay_distribution: str | None = None) -> dict[str, Any]:
        return {
            "distance_matrix": self.distance_matrix,
            "max_delay_matrix": self.max_delay_matrix,
            "uncertain_routes": self.uncertain_routes,
            "delay_mask": self.delay_mask,
            "delay_distribution": delay_distribution or self.delay_distribution,
        }


def all_directed_routes(n: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(1, n + 1) for j in range(1, n + 1) if i != j]


def directed_routes_between(nodes_1indexed: Sequence[int]) -> list[tuple[int, int]]:
    return [(i, j) for i in nodes_1indexed for j in nodes_1indexed if i != j]


def routes_to_mask(n: int, routes: Iterable[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((n, n), dtype=float)
    for i, j in routes:
        if i == j:
            continue
        if not (1 <= i <= n and 1 <= j <= n):
            raise ValueError(f"route {(i, j)} is outside 1..{n}")
        mask[i - 1, j - 1] = 1.0
    return mask


def mask_to_routes(mask: np.ndarray) -> list[tuple[int, int]]:
    ii, jj = np.where(np.asarray(mask) > 0)
    return [(int(i) + 1, int(j) + 1) for i, j in zip(ii, jj) if i != j]


def generate_random_matrices(
    n: int,
    delay_routes: Iterable[tuple[int, int]] | None = None,
    *,
    seed: int = 42,
    min_distance: int = 5,
    max_distance: int = 60,
    symmetric_distance: bool = True,
    random_delay_ratio: tuple[float, float] = (0.10, 0.15),
    delay_min_factor: float = 0.35,
    delay_max_factor: float = 1.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate (distance, max_delay, delay_mask) for capacity tests.

    delay_routes are directed and 1-indexed.  If omitted, a random directed set
    containing roughly 10--15% of all non-diagonal edges is selected.  The delay
    matrix is full, and each entry scales with its base duration; the mask is
    the third matrix that decides where delay is actually active.
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    rng = np.random.default_rng(seed)

    base = np.zeros((n, n), dtype=float)
    if symmetric_distance:
        upper = rng.integers(min_distance, max_distance + 1, size=(n, n))
        for i in range(n):
            for j in range(i + 1, n):
                base[i, j] = base[j, i] = float(upper[i, j])
    else:
        base = rng.integers(min_distance, max_distance + 1, size=(n, n)).astype(float)
        np.fill_diagonal(base, 0.0)

    factors = rng.uniform(delay_min_factor, delay_max_factor, size=(n, n))
    max_delay = np.maximum(1.0, np.round(base * factors, 1))
    np.fill_diagonal(max_delay, 0.0)

    if delay_routes is None:
        candidates = [(i, j) for i in range(1, n + 1) for j in range(1, n + 1) if i != j]
        ratio = float(rng.uniform(random_delay_ratio[0], random_delay_ratio[1]))
        k = max(1, int(round(ratio * n * n)))
        idx = rng.choice(len(candidates), size=min(k, len(candidates)), replace=False)
        delay_routes = [candidates[int(t)] for t in idx]

    delay_mask = routes_to_mask(n, delay_routes)
    return base, max_delay, delay_mask


def generate_random_instance(
    n: int,
    delay_routes: Iterable[tuple[int, int]] | None = None,
    *,
    seed: int = 42,
    instance_id: str | None = None,
    name: str | None = None,
    delay_distribution: str = "uniform",
) -> TSPInstance:
    base, max_delay, mask = generate_random_matrices(n, delay_routes, seed=seed)
    routes = mask_to_routes(mask)
    return TSPInstance(
        id=instance_id or f"random{n}",
        name=name or f"Random_{n}_nodes",
        distance_matrix=base,
        max_delay_matrix=max_delay,
        delay_mask=mask,
        uncertain_routes=routes,
        delay_distribution=delay_distribution,
        aliases=(f"random{n}",),
    )


def builtin_instances() -> list[TSPInstance]:
    d4 = np.asarray([
        [0, 8.0, 13, 16.0],
        [8.0, 0, 4.0, 7.0],
        [13, 4.0, 0, 8.0],
        [16.0, 7.0, 8.0, 0],
    ], dtype=float)
    unc4 = np.asarray([
        [0, 9.1, 1.8, 8.0],
        [2.7, 0, 8.4, 4.3],
        [9.0, 1.7, 0, 2.2],
        [7.4, 6.2, 4.0, 0],
    ], dtype=float)

    d5 = np.asarray([
        [0, 10, 12, 19, 8],
        [10, 0, 5, 7, 11],
        [12, 5, 0, 9, 7],
        [19, 7, 9, 0, 3],
        [8, 11, 7, 3, 0],
    ], dtype=float)
    unc5 = np.asarray([
        [0, 2.0, 4.6, 9.0, 2.7],
        [3.1, 0, 6.0, 1.3, 6.0],
        [2.5, 4.8, 0, 3.2, 1.9],
        [9.0, 6.4, 2.1, 0, 4.3],
        [5.1, 3.0, 1.8, 8.4, 0],
    ], dtype=float)

    d6 = np.asarray([
        [0, 10, 14, 18, 11, 16],
        [10, 0, 5, 9, 8, 13],
        [14, 5, 0, 6, 10, 9],
        [18, 9, 6, 0, 7, 11],
        [11, 8, 10, 7, 0, 6],
        [16, 13, 9, 11, 6, 0],
    ], dtype=float)
    unc6 = np.asarray([
        [0, 3.0, 4.5, 5.5, 3.2, 4.0],
        [3.3, 0, 24.0, 22.0, 5.0, 4.6],
        [4.8, 23.0, 0, 25.0, 5.5, 4.2],
        [5.2, 21.0, 24.0, 0, 4.8, 5.3],
        [3.4, 5.1, 5.7, 4.6, 0, 3.5],
        [4.1, 4.7, 4.4, 5.0, 3.6, 0],
    ], dtype=float)

    d10 = np.asarray([
        [0, 9, 15, 18, 20, 24, 22, 27, 30, 26],
        [9, 0, 7, 12, 14, 18, 19, 21, 25, 20],
        [15, 7, 0, 11, 8, 13, 15, 17, 19, 16],
        [18, 12, 11, 0, 9, 16, 8, 14, 20, 22],
        [20, 14, 8, 9, 0, 10, 12, 9, 14, 17],
        [24, 18, 13, 16, 10, 0, 18, 11, 9, 7],
        [22, 19, 15, 8, 12, 18, 0, 13, 16, 23],
        [27, 21, 17, 14, 9, 11, 13, 0, 6, 15],
        [30, 25, 19, 20, 14, 9, 16, 6, 0, 12],
        [26, 20, 16, 22, 17, 7, 23, 15, 12, 0],
    ], dtype=float)
    unc10 = np.asarray([
        [0, 3.0, 5.0, 4.2, 5.8, 4.5, 6.0, 4.0, 5.5, 4.4],
        [3.2, 0, 5.6, 4.6, 5.3, 4.1, 6.2, 4.4, 5.9, 4.7],
        [5.1, 5.8, 0, 5.0, 30.0, 5.4, 28.0, 5.7, 32.0, 5.2],
        [4.4, 4.8, 5.2, 0, 5.9, 4.8, 5.5, 4.9, 6.1, 4.6],
        [5.7, 5.1, 29.0, 5.6, 0, 5.3, 31.0, 5.0, 27.0, 5.8],
        [4.8, 4.5, 5.6, 4.9, 5.5, 0, 6.0, 5.2, 6.4, 4.1],
        [6.1, 5.9, 31.0, 5.2, 28.0, 5.8, 0, 5.4, 30.0, 6.0],
        [4.5, 4.7, 5.8, 5.1, 5.3, 5.5, 5.9, 0, 6.2, 4.8],
        [5.9, 6.0, 33.0, 5.7, 29.0, 6.1, 32.0, 5.6, 0, 5.3],
        [4.6, 4.9, 5.4, 4.8, 6.0, 4.3, 6.3, 5.0, 5.7, 0],
    ], dtype=float)

    return [
        TSPInstance("1", "Example_4_nodes", d4, unc4, uncertain_routes=[(1, 2), (1, 3)], aliases=("assignment4",)),
        TSPInstance("2", "Example_5_nodes", d5, unc5, uncertain_routes=all_directed_routes(5), aliases=("lecture5",)),
        TSPInstance("3", "Capacity_6_delay_between_3_nodes", d6, unc6, uncertain_routes=directed_routes_between([2, 3, 4]), aliases=("capacity6",)),
        TSPInstance("4", "Capacity_10_delay_between_4_nodes", d10, unc10, uncertain_routes=directed_routes_between([3, 5, 7, 9]), aliases=("capacity10",)),
    ]


def _as_instance(raw: dict[str, Any]) -> TSPInstance:
    matrix = np.asarray(raw["distance_matrix"], dtype=float)
    max_delay = raw.get("max_delay_matrix")
    delay_mask = raw.get("delay_mask")
    routes = raw.get("uncertain_routes")
    if routes is not None:
        routes = [tuple(map(int, r)) for r in routes]
    return TSPInstance(
        id=str(raw.get("id", raw.get("name", f"n{len(matrix)}"))),
        name=str(raw.get("name", raw.get("id", f"n{len(matrix)}"))),
        distance_matrix=matrix,
        max_delay_matrix=None if max_delay is None else np.asarray(max_delay, dtype=float),
        uncertain_routes=routes,
        delay_mask=None if delay_mask is None else np.asarray(delay_mask, dtype=float),
        delay_distribution=str(raw.get("delay_distribution", "uniform")),
        aliases=tuple(map(str, raw.get("aliases", ()))),
    )


def load_instances(json_path: str | None = None) -> list[TSPInstance]:
    instances = builtin_instances()
    if not json_path:
        return instances
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    raw_instances = payload.get("instances", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw_instances, list):
        raise ValueError("instance JSON must be a list or {'instances': [...]} object")
    return instances + [_as_instance(x) for x in raw_instances]


def resolve_instance_ids(instances: list[TSPInstance], requested: Iterable[str] | None) -> list[TSPInstance]:
    if not requested:
        return instances[:4]
    by_key: dict[str, TSPInstance] = {}
    for pos, inst in enumerate(instances, start=1):
        by_key[str(pos)] = inst
        by_key[inst.id] = inst
        for alias in inst.aliases:
            by_key[alias] = inst
    out: list[TSPInstance] = []
    for token in requested:
        token = str(token)
        if token not in by_key:
            known = ", ".join(f"{inst.id}:{inst.name}" for inst in instances)
            raise ValueError(f"Unknown instance id '{token}'. Known ids: {known}")
        out.append(by_key[token])
    return out


def list_instance_text(instances: list[TSPInstance]) -> str:
    rows = []
    for inst in instances:
        n_routes = 0 if inst.uncertain_routes is None else len(inst.uncertain_routes)
        alias = f" aliases={','.join(inst.aliases)}" if inst.aliases else ""
        rows.append(f"  {inst.id}: name={inst.name:<38} n={inst.n:<3} uncertain_routes={n_routes:<4}{alias}")
    return "\n".join(rows)
