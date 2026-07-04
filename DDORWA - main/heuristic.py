"""Traffic-aware greedy heuristic for random-feedback evaluation."""
from __future__ import annotations
import numpy as np
from env import StochasticTSPEnv


class HeuristicAgent:
    def __init__(self, env: StochasticTSPEnv, risk_weight: float = 0.5, lookahead_weight: float = 0.2, return_weight: float = 0.1, binary_penalty: float = 5.0):
        self.env = env
        self.env._fixed_depot_city = 0
        self.env._set_depot(0)
        self.env._has_reset = True
        self.base = np.asarray(env.base, dtype=float)
        self.delay_mask = np.asarray(env.delay_mask, dtype=float)
        # Sees only whether there may be delay, not max-delay values/distribution.
        self.risk = self.delay_mask * float(binary_penalty)
        self.risk_weight, self.lookahead_weight, self.return_weight = risk_weight, lookahead_weight, return_weight

    def act(self, obs=None, greedy: bool = True) -> int:
        cur, visited = self.env.current, self.env.visited_mask
        valid = np.where(self.env.valid_action_mask())[0]
        best_a, best_score = int(valid[0]), float("inf")
        for a in valid:
            j = int(a) + 1
            remaining = [k for k in range(1, self.env.n) if k != j and not (visited & (1 << (k - 1)))]
            look = min([self.base[j, k] for k in remaining], default=0.0)
            score = self.base[cur, j] + self.risk_weight * self.risk[cur, j] + self.lookahead_weight * look
            if not remaining:
                score += self.return_weight * self.base[j, 0]
            if score < best_score:
                best_score, best_a = score, int(a)
        return best_a

    def run_episode(self, seed: int | None = None) -> dict:
        obs, _ = self.env.reset(seed)
        ret, done, route = 0.0, False, [self.env.depot_city]
        while not done:
            a = self.act(obs)
            obs, r, done, _, info = self.env.step(a)
            ret += float(r)
            route.append(int(info.get("next_city", self.env.action_to_city(a))))
        route.append(self.env.depot_city)
        return {"route": route, "cost": -ret, "return": ret, "done": done}
