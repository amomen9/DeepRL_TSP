"""Compact stochastic TSP environment.

State: current city + visited non-depot cities.  Actions 0..n-2 map to the
cities other than the current depot.  The depot is chosen randomly by default
when an environment instance is created, and then advances cyclically on every
reset/reseed unless an explicit depot_city is provided.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal
import numpy as np

DelayDistribution = Literal["uniform", "fixed", "none"]
VALID_DELAY_DISTRIBUTIONS = {"uniform", "fixed", "none"}


@dataclass(frozen=True)
class Transition:
    next_state: tuple[int, int]
    reward: float
    done: bool


class StochasticTSPEnv:
    def __init__(
        self,
        distance_matrix,
        max_delay_matrix=None,
        delay_mask=None,
        uncertain_routes: Iterable[tuple[int, int]] | None = None,  # 1-indexed
        delay_distribution: str = "uniform",
        seed: int | None = None,
        depot_city: int | None = None,
    ):
        self.base = np.asarray(distance_matrix, dtype=np.float32)
        if self.base.ndim != 2 or self.base.shape[0] != self.base.shape[1]:
            raise ValueError("distance_matrix must be square")
        self.n = int(self.base.shape[0])
        self.n_actions = max(0, self.n - 1)
        self.max_delay = np.zeros_like(self.base) if max_delay_matrix is None else np.asarray(max_delay_matrix, dtype=np.float32)
        if delay_mask is None:
            self.delay_mask = np.zeros_like(self.base, dtype=np.float32)
            if uncertain_routes is None:
                self.delay_mask[:] = (self.max_delay > 0).astype(np.float32)
            else:
                for i, j in uncertain_routes:
                    self.delay_mask[i - 1, j - 1] = 1.0
        else:
            self.delay_mask = np.asarray(delay_mask, dtype=np.float32)
        np.fill_diagonal(self.delay_mask, 0.0)
        if delay_distribution not in VALID_DELAY_DISTRIBUTIONS:
            raise ValueError(f"unknown delay_distribution={delay_distribution}")
        self.delay_distribution = delay_distribution
        self.rng = np.random.default_rng(seed)
        self._fixed_depot_city = None if depot_city is None else int(depot_city)
        self.depot_city = 0
        self._action_cities: list[int] = []
        self._city_to_action: dict[int, int] = {}
        self.realized = self.base.copy()
        self._has_reset = False
        self._set_depot(self._choose_initial_depot())
        self.realized = self._sample_realized_matrix()
        self._set_state()

    def _choose_initial_depot(self) -> int:
        if self._fixed_depot_city is not None:
            return self._fixed_depot_city
        return int(self.rng.integers(0, self.n))

    def _advance_depot(self) -> int:
        if self._fixed_depot_city is not None:
            return self._fixed_depot_city
        return (self.depot_city + 1) % self.n

    def _set_depot(self, depot_city: int) -> None:
        self.depot_city = int(depot_city)
        self._action_cities = [city for city in range(self.n) if city != self.depot_city]
        self._city_to_action = {city: idx for idx, city in enumerate(self._action_cities)}

    def _set_state(self) -> None:
        self.current = self.depot_city
        self.visited_mask = 0
        self.total_cost = 0.0

    def clone_with_seed(self, seed: int | None = None) -> "StochasticTSPEnv":
        depot_city = self._fixed_depot_city
        return StochasticTSPEnv(
            self.base,
            self.max_delay,
            self.delay_mask,
            None,
            self.delay_distribution,
            seed,
            depot_city=depot_city,
        )

    @property
    def start_state(self) -> tuple[int, int]:
        return (self.depot_city, 0)

    @property
    def full_mask(self) -> int:
        return (1 << (self.n - 1)) - 1

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self._has_reset:
            self._set_depot(self._advance_depot())
        self.realized = self._sample_realized_matrix()
        self._set_state()
        self._has_reset = True
        return self.observation(), {}

    def _sample_realized_matrix(self) -> np.ndarray:
        if self.delay_distribution == "none":
            delay = np.zeros_like(self.base)
        elif self.delay_distribution == "fixed":
            delay = self.max_delay * self.delay_mask
        elif self.delay_distribution == "uniform":
            delay = self.rng.uniform(0.0, 1.0, self.base.shape).astype(np.float32) * self.max_delay * self.delay_mask
        else:
            raise ValueError(f"unknown delay_distribution={self.delay_distribution}")
        out = self.base + delay
        np.fill_diagonal(out, 0.0)
        return out.astype(np.float32)

    @property
    def action_cities(self) -> tuple[int, ...]:
        return tuple(self._action_cities)

    def _action_bit(self, city: int) -> int:
        idx = self._city_to_action.get(int(city))
        return 0 if idx is None else (1 << idx)

    def city_to_action(self, city: int) -> int:
        return self._city_to_action[int(city)]

    def action_to_city(self, action: int) -> int:
        return self._action_cities[int(action)]

    def observation(self) -> np.ndarray:
        denom = max(1, self.n - 1)
        visited_count = int(self.visited_mask.bit_count())
        return np.array([
            self.current / denom,
            visited_count / denom,
            (denom - visited_count) / denom,
            min(self.total_cost / max(1.0, float(self.base.max()) * self.n), 10.0),
        ], dtype=np.float32)

    def valid_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.n_actions, dtype=bool)
        for a, city in enumerate(self._action_cities):
            mask[a] = (self.visited_mask & self._action_bit(city)) == 0
        return mask

    def expected_edge_cost(self, i: int, j: int) -> float:
        factor = 0.5 if self.delay_distribution == "uniform" else 1.0 if self.delay_distribution == "fixed" else 0.0
        return float(self.base[i, j] + factor * self.delay_mask[i, j] * self.max_delay[i, j])

    def transition(self, state: tuple[int, int], action: int) -> Transition:
        """Expected-model transition used by value iteration."""
        current, mask = state
        if action < 0 or action >= self.n_actions:
            return Transition(state, -1e6, True)
        next_city = self.action_to_city(action)
        bit = self._action_bit(next_city)
        if bit == 0 or (mask & bit):
            return Transition(state, -1e6, True)
        new_mask = mask | bit
        cost = self.expected_edge_cost(current, next_city)
        done = new_mask == self.full_mask
        if done:
            cost += self.expected_edge_cost(next_city, self.depot_city)
            next_city = self.depot_city
        return Transition((next_city, new_mask), -cost, done)

    def step(self, action: int):
        mask = self.valid_action_mask()
        if action < 0 or action >= self.n_actions or not mask[action]:
            return self.observation(), -1e6, True, False, {"invalid": True}
        next_city = self.action_to_city(action)
        cost = float(self.realized[self.current, next_city])
        self.visited_mask |= self._action_bit(next_city)
        done = self.visited_mask == self.full_mask
        if done:
            cost += float(self.realized[next_city, self.depot_city])
            self.current = self.depot_city
        else:
            self.current = next_city
        self.total_cost += cost
        return self.observation(), -cost, done, False, {"cost": cost, "next_city": next_city, "current": self.current}

    def run_actions(self, actions: list[int], seed: int | None = None) -> dict:
        obs, _ = self.reset(seed)
        route, ret, done = [self.depot_city], 0.0, False
        for a in actions:
            obs, r, done, _, info = self.step(a)
            ret += float(r)
            route.append(int(info.get("next_city", self.action_to_city(a))))
            if done:
                route.append(self.depot_city)
                break
        return {"route": route, "cost": -ret, "return": ret, "done": done}


def actions_from_route(route: list[int], depot_city: int = 0) -> list[int]:
    route = [int(x) for x in route]
    return [city if city < depot_city else city - 1 for city in route[1:-1] if city != depot_city]
