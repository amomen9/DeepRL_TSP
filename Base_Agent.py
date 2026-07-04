"""
Base_Agent.py - Base class for TSP solver agents.
"""

from typing import Any


class Base_Agent:
    """Base class for TSP agents. Subclasses must implement ``solve``."""

    def __init__(self, env):
        self.env = env
        self.n = env.get_num_cities()
        self.duration_matrix: Any = getattr(env, "stochastic_duration_matrix", None)
        if self.duration_matrix is None:
            self.duration_matrix = getattr(env, "duration_matrix", getattr(env, "dist", None))
        if self.duration_matrix is None:
            raise ValueError("Environment must expose a duration_matrix.")
        self.dist: Any = self.duration_matrix
        self.optimal_cost = None
        self.optimal_tours = []

    def solve(self):
        raise NotImplementedError("Subclasses must implement solve().")
