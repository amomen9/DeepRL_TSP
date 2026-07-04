"""Exact finite-state value iteration for the stochastic-feedback TSP model.

The env exposes expected rewards through env.transition(state, action).  Delays
are continuous but affect only edge reward, not the next city/visited set, so the
Bellman optimum is exact under the specified expected-delay model.
"""
from __future__ import annotations
import numpy as np
from env import StochasticTSPEnv


def value_iteration(env: StochasticTSPEnv, gamma: float = 1.0, theta: float = 1e-10, max_iter: int = 10_000) -> dict:
    depot = int(env.depot_city)
    states = [(cur, mask) for mask in range(env.full_mask + 1) for cur in range(env.n) if cur == depot or env._action_bit(cur)]
    V = {s: 0.0 for s in states}
    policy: dict[tuple[int, int], int] = {}
    it = 0
    for it in range(max_iter):
        delta = 0.0
        for s in states:
            cur, mask = s
            if mask == env.full_mask:
                continue
            valid = [a for a in range(env.n_actions) if not (mask & (1 << a))]
            q = []
            for a in valid:
                tr = env.transition(s, a)
                q.append(tr.reward + (0.0 if tr.done else gamma * V[tr.next_state]))
            new_v = max(q)
            delta = max(delta, abs(new_v - V[s]))
            V[s] = new_v
            policy[s] = valid[int(np.argmax(q))]
        if delta < theta:
            break
    route, s, actions = [env.depot_city], env.start_state, []
    while s[1] != env.full_mask:
        a = policy[s]
        actions.append(a)
        city = env.action_to_city(a)
        route.append(city)
        s = env.transition(s, a).next_state
    route.append(env.depot_city)
    return {"value": float(V[env.start_state]), "expected_cost": float(-V[env.start_state]), "policy": policy, "route": route, "actions": actions, "iterations": it + 1, "V": V}
