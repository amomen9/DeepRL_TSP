"""
Environment.py - TSP problem instance (duration matrix, uncertainty matrix, and validation).
"""

import itertools
import gc
import numpy as np
from Library.Library_env_elements import (
    generate_random_duration_matrix,
    generate_dynamic_noise_matrix,
    generate_random_potential_uncertainty_matrix,
    tour_cost,
)
from Library.Library_dp import should_skip_dp_due_to_memory


class StochasticTSPEnvironment:
    """A stochastic TSP problem instance.

    Matrices held by the environment
    --------------------------------
    duration_matrix (D)
        Deterministic base duration. Provided by the user, or randomly
        generated via 'generate_random_duration_matrix' when omitted.

    potential_uncertainty_matrix (U)
        Per-edge upper bound for the uniform noise distribution. Provided by
        the user, or randomly generated via
        'generate_random_potential_uncertainty_matrix' when omitted.

    noise_matrix (N)
        Sampled once per environment instance via
        'generate_dynamic_noise_matrix(U)'; each entry 'N[i,j]' is drawn
        i.i.d. from 'Uniform(0, U[i,j])'.

    uncertainty_inclusion_matrix (I)
        Built from 'uncertain_routes': 'I[i,j] = 1.0' for edges listed
        as uncertain (1-indexed), '0.0' otherwise. Defaults to the all-ones
        matrix when 'uncertain_routes' is 'None' (every edge uncertain).

    stochastic_duration_matrix
        'D + N * I' (element-wise). Used by the training approaches.
        The agent never reads this matrix directly; it observes one entry at
        a time as the reward returned by 'step()'. Sampled noise is unknown to
        the agent but constant for the lifetime of the environment instance.

    expected_stochastic_duration_matrix
        'D + U * I * 0.5' (element-wise). Since 'N[i,j] ~ Uniform(0,
        U[i,j])' has expectation 'U[i,j]/2', this is the *known*
        deterministic matrix handed to the classic DP solver.
    """

    def __init__(
        self,
        n_cities=None,
        duration_matrix=None,
        potential_uncertainty_matrix=None,
        uncertain_routes=None,
        uncertainty_scale=0.3,
        uncertainty_symmetric=False,
        initialize_dp_table=False,
        initialize_noise=False,
        seed=None,
        depot_city=None,
    ):
        # Backward-compatible argument handling: many call sites pass the
        # duration matrix as the first positional argument ('cls(matrix)').
        if (
            duration_matrix is None
            and n_cities is not None
            and isinstance(n_cities, (list, np.ndarray))
        ):
            duration_matrix = n_cities
            n_cities = None

        self.potential_uncertainty_matrix = potential_uncertainty_matrix
        self.uncertainty_scale = uncertainty_scale
        self.uncertainty_symmetric = uncertainty_symmetric
        self.initialize_dp_table = initialize_dp_table
        self.initialize_noise = initialize_noise
        self.stochastic_duration_matrix = None
        self.effective_noise_matrix = None
        self.seed = seed

        # Depot cycles across resets / reseeds.
        #   current_depot_city = depot used for the active episode
        #   depot_city         = next depot to use on the next reset
        # Both are 0-indexed internally. The actual starting depot is resolved
        # below, once self.n is known (random when 'depot_city' is None, or the
        # explicitly requested 1-indexed city otherwise). 'self._depot_arg'
        # holds the user request so it can be applied after the matrix is sized.
        self._depot_arg = depot_city
        self.depot_city = 0
        self.current_depot_city = 0

        # Resolve duration_matrix: use the one provided, or generate one
        # randomly when only 'n_cities' was given.
        if duration_matrix is None:
            if n_cities is None:
                raise ValueError("Either n_cities or duration_matrix must be provided.")
            self.duration_matrix = np.asarray(
                generate_random_duration_matrix(n_cities, seed=seed),
                dtype=float,
            )
        else:
            self.duration_matrix = np.asarray(duration_matrix, dtype=float)

        if (
            self.duration_matrix.ndim != 2
            or self.duration_matrix.shape[0] == 0
            or self.duration_matrix.shape[0] != self.duration_matrix.shape[1]
        ):
            raise ValueError("duration_matrix must be a non-empty square 2D array (NxN).")

        self.n = int(self.duration_matrix.shape[0])
        self.n_cities = self.n

        # Resolve the starting depot now that the number of cities is known.
        # depot_city=None  -> random depot in 0..n-1 (seeded by 'seed' when given
        #                     so instances stay reproducible).
        # depot_city=k     -> 1-indexed city name (matching uncertain_routes'
        #                     convention); stored 0-indexed as k-1.
        start_depot = self._resolve_start_depot(self._depot_arg, seed)
        self.depot_city = start_depot
        self.current_depot_city = start_depot

        if initialize_dp_table not in {True, False}:
            raise ValueError("initialize_dp_table must be either True or False.")
        if initialize_noise not in {True, False}:
            raise ValueError("initialize_noise must be either True or False.")

        # Validate uncertain_routes (city ids are 1-indexed and must lie in 1..n).
        if uncertain_routes is not None:
            if not isinstance(uncertain_routes, list):
                raise TypeError("uncertain_routes must be a list of (city_i, city_j) tuples.")
            for route in uncertain_routes:
                if not isinstance(route, tuple) or len(route) != 2:
                    raise TypeError("Each uncertain route must be a tuple of two integers.")
                city_i, city_j = route
                if not isinstance(city_i, int) or not isinstance(city_j, int):
                    raise TypeError("Each uncertain route must contain integer city indices.")
        self.uncertain_routes = uncertain_routes

        # State-encoding DP_Table (only built when 'initialize_model=True').
        # Used to populate the 'current_state' slot of the observation.
        self.DP_Table: np.ndarray | None = None
        self.subset_row_by_city: dict[int, dict[tuple[int, ...], int]] | None = None
        self.current_subset: frozenset[int] | None = None

        self._construct_model()

        # Many call sites expect stochastic_duration_matrix to be ready.
        # If noise was not initialized, sample it once here.
        if self.stochastic_duration_matrix is None:
            self.reseed_noise(self.seed, advance_depot=False)

        # Initialize a valid default episode state so the environment can be
        # inspected immediately after construction without requiring reset().
        self.current_location = self.current_depot_city
        self.current_subset = frozenset()
        self.current_state = 0
        self.current_visited_cities = {self.current_depot_city}
        self.current_cost = 0.0
        self.current_step_count = 0
        self.done = False
        self.terminated = False
        self.truncated = False
        self.obs = self._build_observation()

    # for backward compatibility with old call sites that passed the duration matrix as the first positional argument
    @classmethod
    def from_matrix(cls, matrix):
        """Create a TSP environment from an explicit duration matrix."""
        return cls(matrix)

    @classmethod
    def random_instance(cls, n, min_durn=1, max_durn=100, symmetric=True, seed=None):
        """Create a random TSP instance with a random duration matrix."""
        matrix = generate_random_duration_matrix(n, min_durn, max_durn, symmetric, seed)
        return cls(matrix)

    # ------------------------------------------------------------------
    # DP_Table state-encoding helpers (used only when initialize_dp_table=True)
    # ------------------------------------------------------------------

    def _state_to_location(self, state):
        dp_table = self.DP_Table
        if dp_table is None:
            raise ValueError("State table has not been initialized.")
        return np.array(np.unravel_index(state, dp_table.shape))

    def _location_to_state(self, location):
        dp_table = self.DP_Table
        if dp_table is None:
            raise ValueError("State table has not been initialized.")
        return int(np.ravel_multi_index(location, dp_table.shape))

    def _subset_list_for_city(self, current_city):
        other_cities = [
            city
            for city in range(self.n)
            if city != current_city and city != self.current_depot_city
        ]
        subsets: list[tuple[int, ...]] = [tuple()]
        for size in range(1, len(other_cities) + 1):
            subsets.extend([tuple(combo) for combo in itertools.combinations(other_cities, size)])
        return subsets

    def construct_state_table(self):
        self.DP_Table = np.empty((2 ** (self.n - 2), self.n - 1), dtype=object)
        self.subset_row_by_city = {}
        for current_city in range(self.n):
            if current_city == self.current_depot_city:
                continue
            subsets = self._subset_list_for_city(current_city)
            if len(subsets) != 2 ** (self.n - 2):
                # Keep the old table shape contract; only the non-depot cities matter.
                pass
            self.subset_row_by_city[current_city] = {subset: row for row, subset in enumerate(subsets)}
            for row, subset in enumerate(subsets):
                if current_city < self.current_depot_city:
                    col = current_city
                else:
                    col = current_city - 1
                if 0 <= col < self.n - 1:
                    self.DP_Table[row, col] = (subset, current_city)

    # ------------------------------------------------------------------
    # Core: build the matrix triple (duration / stochastic / expected)
    # ------------------------------------------------------------------

    def _construct_model(self):
        """Resolve all matrices on the env per the formulas in the class docstring."""
        self.n_states = (self.n - 1) * (2 ** (self.n - 2))
        self.n_actions = self.n - 1
        self.start_location = self.current_depot_city
        self.goal_location = self.current_depot_city
        self.reward_per_step = 0.0

        # Inclusion matrix from uncertain_routes (default = all 1.0).
        if self.uncertain_routes:
            self.uncertainty_inclusion_matrix = np.zeros((self.n, self.n), dtype=float)
            for city_i, city_j in self.uncertain_routes:
                if city_i < 1 or city_i > self.n or city_j < 1 or city_j > self.n:
                    raise ValueError(
                        "uncertain_routes entries must use 1-indexed city names within matrix bounds."
                    )
                self.uncertainty_inclusion_matrix[city_i - 1, city_j - 1] = 1.0
        else:
            self.uncertainty_inclusion_matrix = np.ones((self.n, self.n), dtype=float)

        # Potential uncertainty matrix: provided or randomly generated.
        if self.potential_uncertainty_matrix is None:
            self.potential_uncertainty_matrix = generate_random_potential_uncertainty_matrix(
                duration_matrix=self.duration_matrix,
                min_uncertainty=0,
                max_uncertainty=10,
                uncertainty_scale=self.uncertainty_scale,
                uncertainty_symmetric=self.uncertainty_symmetric,
                seed=self.seed,
            )
        else:
            self.potential_uncertainty_matrix = np.asarray(self.potential_uncertainty_matrix, dtype=float)

        if self.uncertainty_inclusion_matrix.shape != self.duration_matrix.shape:
            raise ValueError("Duration and inclusion matrices must have the same dimensions.")
        if self.potential_uncertainty_matrix.shape != self.duration_matrix.shape:
            raise ValueError("Duration and potential uncertainty matrices must have the same dimensions.")

        # DP-side matrix: noise[i,j] ~ Uniform(0, U[i,j]) ⇒ E[noise] = U/2.
        self.expected_stochastic_duration_matrix = (
            self.duration_matrix
            + self.potential_uncertainty_matrix * self.uncertainty_inclusion_matrix * 0.5
        )

        if self.initialize_dp_table:
            skip, needed_GB, free_GB = self.memory_requirements_eval(n=self.n, total_dp_runs=1)
            if skip:
                print(
                    f"!!!Memory not enough to run the DP solution!!! Needed memory: {needed_GB:.2f}GB <-> Free OS memory: {free_GB:.2f}GB."
                )
                self.initialize_dp_table = None
            # NOTE: construct_state_table() is intentionally NOT called eagerly.
            # The tabular (city, subset) -> state-index table is by far the
            # largest allocation in the DP path (~(n-1)*2**(n-2) Python tuples +
            # dict entries), yet the Held-Karp solver never reads it and the
            # deep-RL agents run with current_state == 0 (it is destroyed before
            # training begins). Building it here only spiked RAM and pushed the
            # DP node ceiling far below what the solver actually needs. Call
            # construct_state_table() explicitly if/when a tabular agent that
            # indexes current_state is added.

        if self.initialize_noise:
            self.reseed_noise(self.seed, advance_depot=False)

    # ------------------------------------------------------------------
    # Depot cycle / reseed helpers
    # ------------------------------------------------------------------
    def _resolve_start_depot(self, depot_city, seed):
        """Resolve the 0-indexed starting depot for a fresh environment instance.

        'depot_city is None' selects a random depot in '0..n-1' (every new
        instance starts at a random depot, not necessarily city 1). When 'seed'
        is provided the selection is reproducible. An explicit 'depot_city' is a
        1-indexed city name (1..n) and is converted to the 0-indexed depot.
        """
        if depot_city is None:
            return int(np.random.default_rng(seed).integers(self.n))
        if isinstance(depot_city, bool) or not isinstance(depot_city, (int, np.integer)):
            raise TypeError("depot_city must be a 1-indexed integer city name or None.")
        if depot_city < 1 or depot_city > self.n:
            raise ValueError(
                f"depot_city must be a 1-indexed city in 1..{self.n}, got {depot_city}."
            )
        return int(depot_city) - 1

    def _advance_depot_city(self):
        """Advance the depot city cyclically through the available city indices."""
        self.depot_city = (self.depot_city + 1) % self.n
        return self.depot_city

    def _available_cities(self):
        """Return all non-depot cities for the active episode."""
        return [city for city in range(self.n) if city != self.current_depot_city]

    def _action_to_city(self, action):
        """Map a 0-based action to the corresponding non-depot city."""
        available_cities = self._available_cities()
        return available_cities[action]

    def reseed_noise(self, new_seed=None, advance_depot=True):
        """|Immutable| Resample the noise matrix with a new seed, keeping the same potential uncertainty.

        Mirrors ``reset`` for the depot cycle: when ``advance_depot`` is True the
        pending depot becomes the active one and the pending pointer advances by
        one (cyclically), so the depot increments on every reseed just as it does
        on every reset. ``advance_depot=False`` (used during construction) leaves
        the depot untouched.
        """
        self.seed = new_seed
        if advance_depot:
            self.current_depot_city = self.depot_city
            self._advance_depot_city()
        potential_uncertainty_matrix = self.potential_uncertainty_matrix
        if potential_uncertainty_matrix is None:
            raise ValueError("potential_uncertainty_matrix must be initialized before reseeding noise.")
        self.noise_matrix = generate_dynamic_noise_matrix(
            potential_uncertainty_matrix=potential_uncertainty_matrix,
            min_uncertainty=0.0,
            max_uncertainty=float(np.max(potential_uncertainty_matrix)),
            uncertainty_symmetric=self.uncertainty_symmetric,
            seed=new_seed,
        )
        self.effective_noise_matrix = self.noise_matrix * self.uncertainty_inclusion_matrix
        self.stochastic_duration_matrix = self.duration_matrix + self.effective_noise_matrix

    # ------------------------------------------------------------------
    # Gym-like API
    # ------------------------------------------------------------------

    def _build_observation(self):
        """Build a 4D observation vector for the policy network."""
        max_city_index = max(1, self.n - 1)
        max_state_index = max(1, self.n_states - 1)
        max_cost = float(max(1.0, np.max(self.duration_matrix) * max(1, self.n)))
        return np.array(
            [
                float(self.current_location) / max_city_index,
                float(self.current_state) / max_state_index,
                float(len(self.current_visited_cities)) / max(1, self.n),
                float(self.current_cost) / max_cost,
            ],
            dtype=np.float32,
        )

    def reset(self):
        """Set the agent back to the current depot city and return the initial state."""
        self.current_depot_city = self.depot_city
        self.current_location = self.current_depot_city
        self.current_subset = frozenset()
        self.current_state = 0
        self.current_visited_cities = {self.current_depot_city}
        self.current_cost = 0.0
        self.current_step_count = 0
        self.done = False
        self.terminated = False
        self.truncated = False
        self.obs = self._build_observation()
        self._advance_depot_city()
        return self.obs, {}

    def step(self, action):
        """Advance one step using a city-selection action.

        The reward is '-stochastic_duration_matrix[current_location, next_city]',
        i.e. the actual (sampled-noise) cost of the chosen edge. The agent
        only ever observes this scalar; it does not see the matrix as a whole.
        """
        if not hasattr(self, "obs"):
            self.reset()

        if self.done:
            return self.obs, 0.0, self.terminated, self.truncated, {}

        action = int(action)

        stochastic_duration_matrix = self.stochastic_duration_matrix
        if stochastic_duration_matrix is None:
            raise ValueError("stochastic_duration_matrix has not been initialized.")

        invalid_action = action < 0 or action >= self.n - 1
        if invalid_action:
            reward = -1e9
            self.current_step_count += 1
            self.done = True
            self.terminated = True
            self.truncated = True
            self.obs = self._build_observation()
            info = {
                "current_location": self.current_location,
                "current_state": self.current_state,
                "visited_cities": sorted(self.current_visited_cities),
                "invalid_action": True,
            }
            return self.obs, reward, self.terminated, self.truncated, info

        next_city = self._action_to_city(action)

        invalid_action = (
            next_city in self.current_visited_cities
            or next_city < 0
            or next_city >= self.n
        )

        if invalid_action:
            reward = -1e9
            self.current_step_count += 1
            self.done = True
            self.terminated = True
            self.truncated = True
            self.obs = self._build_observation()
            info = {
                "current_location": self.current_location,
                "current_state": self.current_state,
                "visited_cities": sorted(self.current_visited_cities),
                "invalid_action": True,
            }
            return self.obs, reward, self.terminated, self.truncated, info

        reward = -float(stochastic_duration_matrix[self.current_location][next_city])
        self.current_cost += -reward
        self.current_location = next_city
        self.current_visited_cities.add(next_city)
        self.current_subset = frozenset(
            city for city in self.current_visited_cities if city not in {self.current_depot_city, self.current_location}
        )

        self.current_step_count += 1
        self.terminated = False
        self.truncated = False

        if len(self.current_visited_cities) == self.n:
            reward -= float(stochastic_duration_matrix[self.current_location][self.current_depot_city])
            self.current_cost += float(stochastic_duration_matrix[self.current_location][self.current_depot_city])
            self.current_location = self.current_depot_city
            self.current_visited_cities.add(self.current_depot_city)
            self.current_subset = frozenset(city for city in self.current_visited_cities if city != self.current_depot_city)
            self.done = True
            self.terminated = True
            self.truncated = False
            self.current_state = 0
        elif self.current_step_count >= (self.n + 1):
            self.done = True
            self.terminated = False
            self.truncated = True
        elif self.DP_Table is not None and self.subset_row_by_city is not None and self.current_subset is not None:
            next_subset = tuple(sorted(self.current_subset))
            row_map = self.subset_row_by_city[self.current_location]
            next_row = row_map[next_subset]
            next_col = self.current_location - 1 if self.current_location > self.current_depot_city else self.current_location
            if next_col >= self.n - 1:
                next_col = self.n - 2
            self.current_state = self._location_to_state((next_row, next_col))
        else:
            self.current_state = 0

        self.obs = self._build_observation()
        info = {
            "current_location": self.current_location,
            "current_state": self.current_state,
            "visited_cities": sorted(self.current_visited_cities),
        }
        return self.obs, reward, self.terminated, self.truncated, info

    def render(self):
        """Return a lightweight RGB visualization of the current episode state."""
        if not hasattr(self, "obs"):
            self.reset()

        canvas = np.zeros((64, 64, 3), dtype=np.uint8)
        location_scale = max(1, self.n - 1)
        visited_scale = max(1, self.n)

        x_pos = int(round((self.current_location / location_scale) * 60))
        x_pos = max(0, min(60, x_pos))
        canvas[:, x_pos : x_pos + 4, 0] = 220
        canvas[:, x_pos : x_pos + 4, 1] = 180

        progress = int(round((len(self.current_visited_cities) / visited_scale) * 64))
        canvas[56:64, :progress, 1] = 220
        canvas[56:64, :progress, 2] = 80

        cost_cap = float(max(1.0, np.max(self.duration_matrix) * max(1, self.n)))
        cost_ratio = min(1.0, self.current_cost / cost_cap)
        cost_width = int(round(cost_ratio * 64))
        canvas[48:54, :cost_width, 2] = 220

        return canvas

    def close(self):
        """Compatibility no-op."""
        return None

    def destroy_state_table(self):
        """Free DP state-encoding memory without destroying the environment.

        This releases:
          - self.DP_Table
          - self.subset_row_by_city

        After calling this, 'step()' will fall back to setting 'current_state = 0'
        (because the state table is unavailable), but the environment instance
        (duration/noise matrices etc.) remains usable.

        Notes
        -----
        Any direct calls to internal helpers like '_state_to_location()' or
        '_location_to_state()' will raise a ValueError after the table is
        destroyed.
        """
        self.DP_Table = None
        self.subset_row_by_city = None
        self.current_subset = None
        self.initialize_model = False

        # Best-effort cleanup (especially helpful for large object-dtype arrays)
        gc.collect()

    def get_depot(self):
        """Return the current depot city for the active episode."""
        return self.current_depot_city

    def get_duration(self, i, j):
        return self.duration_matrix[i][j]

    def get_num_cities(self):
        return self.n

    def validate_tour(self, tour):
        """Check Hamiltonian-tour validity (Ref: slide 15 TSP constraints intuition)."""
        depot = getattr(self, "current_depot_city", self.depot_city)
        if len(tour) != self.n + 1:
            return False
        if tour[0] != depot or tour[-1] != depot:
            return False
        visited = set(tour[1:-1])
        return len(visited) == self.n - 1 and visited == (set(range(self.n)) - {depot})

    def tour_cost(self, tour):
        """Total cost of a tour on the stochastic matrix the agent traverses."""
        return tour_cost(self.stochastic_duration_matrix, tour)

    def __str__(self):
        lines = [f"TSP Instance with {self.n} cities", "duration matrix:"]
        header = "     " + "".join(f"{j + 1:>6}" for j in range(self.n))
        lines.append(header)
        lines.append("     " + "-" * (6 * self.n))
        for i in range(self.n):
            row = f"{i + 1:>3} |" + "".join(f"{self.duration_matrix[i][j]:>6}" for j in range(self.n))
            lines.append(row)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # DP memory estimation helpers (moved from DP_Agent.py)
    # ------------------------------------------------------------------

    @staticmethod
    def memory_requirements_eval(n: int, total_dp_runs: int) -> tuple[bool, float, float]:
        """|!!!Immutable!!!| If insufficient memory, print the requested warning and return True."""
        skip, needed_bytes, free_bytes = should_skip_dp_due_to_memory(
            n=n, total_dp_runs=total_dp_runs
        )
        needed_GB = needed_bytes / (1024.0**3)
        free_GB = free_bytes / (1024.0**3)
        return skip, needed_GB, free_GB


TSPEnvironment = StochasticTSPEnvironment
