"""
DP_Agent.py - TSP solver using Dynamic Programming (Held-Karp algorithm).

Implements both bottom-up (Section 5.3, Ref: slides 16-17) and
top-down (Section 5.4, Ref: slides 19-21) approaches.
Finds ALL optimal tours (up to a configurable limit).
"""

from itertools import combinations

import numpy as np

from Library.Library_env_elements import subsets_of_size, format_tour
from Base_Agent import Base_Agent

_EPS = 1e-9  # tolerance for floating-point cost comparison


class TSP_DP_Agent(Base_Agent):
    """Solves TSP exactly using the Held-Karp DP algorithm."""

    def __init__(self, env, scenario_choice="expected"):
        super().__init__(env)
        # Held-Karp starts and ends at the environment's active depot, which
        # may be any city (it is randomised per instance and cycles on reset).
        self.depot = int(getattr(env, "current_depot_city", 0))
        # The classic DP solver receives the *known* expected stochastic
        # matrix D + U * I * 0.5 (provided by the Environment), not the
        # random per-instance sample used by the training agents.

        if scenario_choice != "deterministic" and scenario_choice != "expected"  and scenario_choice != "noise":
            raise ValueError(f"Invalid scenario_choice: {scenario_choice}. Expected 'expected' or 'noise' or 'deterministic'.")
        
        if scenario_choice == "deterministic":
            self.duration_matrix = getattr(env, "duration_matrix", None)
            if self.duration_matrix is None:
                raise ValueError(
                    "Environment must expose duration_matrix for the deterministic DP solver."
                )
        elif scenario_choice == "expected":
            self.duration_matrix = getattr(env, "expected_stochastic_duration_matrix", None)
            if self.duration_matrix is None:    
                raise ValueError(
                    "Environment must expose expected_stochastic_duration_matrix for the expected DP solver."
                )
        elif scenario_choice == "noise":
            self.duration_matrix = getattr(env, "stochastic_duration_matrix", None)
            if self.duration_matrix is None:
                raise ValueError(
                    "Environment must expose stochastic_duration_matrix for the noise DP solver."
                )
    
    def solve(self, method="bottom_up", max_optimal_tours=1000, objective="min", reuse_dp=None):
        """Solve the TSP instance.

        Parameters
        ----------
        method : str
            "bottom_up" (Section 5.3, Ref: slides 16-17) or
            "top_down" (Section 5.4, Ref: slides 19-21).
        max_optimal_tours : int
            Maximum number of optimal tours to enumerate.
        objective : str
            "min" to minimize tour cost, "max" to maximize tour cost.
        reuse_dp : np.ndarray or None
            Optional preallocated ``(2**(n-1), n-1)`` float64 buffer for the
            bottom-up cost table. Supplying the same buffer across successive
            solves (e.g. the best/worst/expected trio) avoids reallocating the
            large table each time. Ignored by the top-down method.

        Returns
        -------
        (optimal_cost, list_of_tours)
        """
        if objective not in {"min", "max"}:
            raise ValueError(f"Unknown objective: {objective}. Expected 'min' or 'max'.")
        maximize = objective == "max"
        if method == "bottom_up":
            return self._solve_bottom_up(max_optimal_tours, maximize=maximize, reuse_dp=reuse_dp)
        elif method == "top_down":
            return self._solve_top_down(max_optimal_tours, maximize=maximize)
        else:
            raise ValueError(f"Unknown method: {method}")

    # ------------------------------------------------------------------
    # Bottom-up DP  (Section 5.3, Ref: slides 16-17)
    # ------------------------------------------------------------------

    def _city_of_slot(self, slot):
        """Map a compact slot (0..n-2) back to its city index.

        The depot is excluded from the compact index space; slots enumerate the
        non-depot cities in increasing order, matching the environment's own
        column convention (``col = city - 1 if city > depot else city``).
        """
        return slot if slot < self.depot else slot + 1

    def _solve_bottom_up(self, max_optimal_tours=1000, maximize=False, reuse_dp=None):
        """
        Bottom-up (iterative) Held-Karp algorithm.

        Reference: Section 5.3, slides 16-17.

        State  : (S, l)  where  S ⊆ (cities \\ {depot}),  l ∈ S
        Meaning: minimum-cost path from the depot visiting every city in S,
                 ending at city l. When ``maximize`` is True, the same state
                 stores the maximum-cost path instead.

        Base case   : C({l}, l) = dist[depot][l]
        Recurrence  : C(S, l)  = min_{m ∈ S\\{l}} { C(S\\{l}, m) + dist[m][l] }
        Final answer: min_{l}  { C(cities\\{depot}, l) + dist[l][depot] }

        Memory: the cost table is indexed over the ``n-1`` non-depot cities
        (compact "slots"), giving a ``(2**(n-1), n-1)`` float64 array rather than
        ``(2**n, n)`` -- the always-unset depot bit and the unused depot column
        are dropped (~2x fewer cells). ``float64`` preserves the 1e-9 tie
        tolerance used to enumerate all optimal tours. A caller may pass
        ``reuse_dp`` to recycle one buffer across successive solves.
        """
        n = self.n
        dist = self.duration_matrix
        depot = self.depot
        INF = float("-inf") if maximize else float("inf")

        # Handle trivial cases
        if n <= 1:
            self.optimal_cost = 0
            self.optimal_tours = [[depot, depot]] if n == 1 else []
            return self.optimal_cost, self.optimal_tours
        if n == 2:
            other = next(c for c in range(n) if c != depot)
            cost = dist[depot][other] + dist[other][depot]
            self.optimal_cost = cost
            self.optimal_tours = [[depot, other, depot]]
            return self.optimal_cost, self.optimal_tours

        m = n - 1  # number of non-depot cities == number of compact slots
        city_of = self._city_of_slot
        shape = (1 << m, m)

        # Compact cost table. Optionally recycle a caller-provided buffer.
        if reuse_dp is not None and getattr(reuse_dp, "shape", None) == shape:
            dp = reuse_dp
            dp.fill(INF)
        else:
            dp = np.full(shape, INF, dtype=np.float64)

        # Base cases: singleton subsets (Ref: slide 16)
        for sl in range(m):
            dp[1 << sl, sl] = dist[depot][city_of(sl)]

        # Fill DP table for subsets of increasing size (Ref: slide 17 example)
        for size in range(2, m + 1):
            for combo in combinations(range(m), size):
                S = 0
                for b in combo:
                    S |= 1 << b
                for sl in combo:
                    l = city_of(sl)
                    S_no_l = S ^ (1 << sl)
                    best = INF
                    for sm in combo:
                        if sm == sl:
                            continue
                        val = dp[S_no_l, sm] + dist[city_of(sm)][l]
                        if (val > best) if maximize else (val < best):
                            best = val
                    dp[S, sl] = best

        # Close the tour by returning to the depot (Ref: slide 16 final step)
        full = (1 << m) - 1  # all non-depot cities included
        best_cost = INF
        for sl in range(m):
            val = dp[full, sl] + dist[city_of(sl)][depot]
            if (val > best_cost) if maximize else (val < best_cost):
                best_cost = val

        self.optimal_cost = best_cost

        # Find all last cities that achieve the optimum
        last_slots = [
            sl for sl in range(m)
            if abs(dp[full, sl] + dist[city_of(sl)][depot] - best_cost) < _EPS
        ]

        # Reconstruct all optimal tours via backtracking
        self.optimal_tours = []
        for last in last_slots:
            for path in self._backtrack_bu(
                dp,
                full,
                last,
                max_optimal_tours - len(self.optimal_tours),
                maximize=maximize,
            ):
                self.optimal_tours.append([depot] + [city_of(s) for s in path] + [depot])
                if len(self.optimal_tours) >= max_optimal_tours:
                    break
            if len(self.optimal_tours) >= max_optimal_tours:
                break

        return self.optimal_cost, self.optimal_tours

    def _backtrack_bu(self, dp, S, sl, remaining, maximize=False):
        """Backtrack through the bottom-up DP table to recover optimal paths.

        Works in compact slot space; the returned lists hold slots (the caller
        maps them back to cities via ``_city_of_slot``).
        """
        if remaining <= 0:
            return []
        if S == (1 << sl):
            return [[sl]]

        S_no_l = S ^ (1 << sl)
        l = self._city_of_slot(sl)
        target = dp[S, sl]
        paths = []
        rem = S_no_l
        while rem:
            low = rem & -rem
            sm = low.bit_length() - 1
            rem ^= low
            candidate = dp[S_no_l, sm] + self.duration_matrix[self._city_of_slot(sm)][l]
            if abs(candidate - target) < _EPS:
                for sub in self._backtrack_bu(dp, S_no_l, sm, remaining - len(paths), maximize=maximize):
                    paths.append(sub + [sl])
                    if len(paths) >= remaining:
                        return paths
        return paths

    # ------------------------------------------------------------------
    # Top-down DP  (Section 5.4, Ref: slides 19-21)
    # ------------------------------------------------------------------

    def _solve_top_down(self, max_optimal_tours=1000, maximize=False):
        """
        Top-down (recursive + memoization) Held-Karp algorithm.

        Reference: Section 5.4, slides 19-21.

        State  : (i, S)  where i = current city,
                 S ⊆ (cities \\ {depot}) \\ {i} = cities still to visit
        Meaning: minimum cost to start from city i, visit all cities in S,
                 then return to the depot. When ``maximize`` is True, the same
                 state stores the maximum-cost path instead.

        Base case   : C(i, ∅) = dist[i][depot]
        Recurrence  : C(i, S)  = min_{j ∈ S} { dist[i][j] + C(j, S\\{j}) }
        Final answer: C(depot, cities\\{depot})
        """
        n = self.n
        dist = self.duration_matrix
        depot = self.depot

        if n <= 1:
            self.optimal_cost = 0
            self.optimal_tours = [[depot, depot]] if n == 1 else []
            return self.optimal_cost, self.optimal_tours
        if n == 2:
            other = next(c for c in range(n) if c != depot)
            cost = dist[depot][other] + dist[other][depot]
            self.optimal_cost = cost
            self.optimal_tours = [[depot, other, depot]]
            return self.optimal_cost, self.optimal_tours

        memo = {}
        choices = {}  # (i, S) -> list of optimal next cities

        def dp(i, S):
            if S == 0:
                return dist[i][depot]
            key = (i, S)
            if key in memo:
                return memo[key]

            best = float("-inf") if maximize else float("inf")
            best_js = []
            for j in range(n):
                if j == depot or not (S // 2**j) % 2:
                    continue
                val = dist[i][j] + dp(j, S - 2**j)
                if (val > best + _EPS) if maximize else (val < best - _EPS):
                    best = val
                    best_js = [j]
                elif abs(val - best) < _EPS:
                    best_js.append(j)

            memo[key] = best
            choices[key] = best_js
            return best

        full = (2**n - 1) - 2**depot
        self.optimal_cost = dp(depot, full)

        # Reconstruct all optimal tours (same recurrence structure as slides 20-21)
        self.optimal_tours = []
        self._backtrack_td(choices, depot, full, [depot], max_optimal_tours)

        return self.optimal_cost, self.optimal_tours

    def _backtrack_td(self, choices, i, S, path, max_optimal_tours):
        """Backtrack through the top-down choices to recover optimal tours."""
        if len(self.optimal_tours) >= max_optimal_tours:
            return
        if S == 0:
            self.optimal_tours.append(path + [self.depot])
            return
        for j in choices.get((i, S), []):
            self._backtrack_td(choices, j, S - 2**j, path + [j], max_optimal_tours)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_results_summary(self, max_display=20):
        """Return a formatted summary of results."""
        if self.optimal_cost is None:
            return "No solution computed yet."
        lines = [
            f"Optimal cost: {self.optimal_cost}",
            f"Number of optimal tours found: {len(self.optimal_tours)}",
            "Optimal tour(s):"
        ]
        display = min(len(self.optimal_tours), max_display)
        for i in range(display):
            lines.append(f"  {format_tour(self.optimal_tours[i])}")
        if len(self.optimal_tours) > max_display:
            lines.append(f"  ... and {len(self.optimal_tours) - max_display} more")
        return "\n".join(lines)



if __name__ == "__main__":
    print("This module defines the TSP_DP_Agent class for solving TSP using DP.")
    print("Run Experiment.py to execute scaling experiments and see results.")
