"""Held-Karp DP for a deterministic TSP matrix."""
from __future__ import annotations
import numpy as np


def solve_classic_dp(matrix, max_tours: int = 1, depot_city: int = 0) -> dict:
    dist = np.asarray(matrix, dtype=np.float64)
    n = dist.shape[0]
    if n == 1:
        return {"cost": 0.0, "routes": [[int(depot_city), int(depot_city)]]}

    depot_city = int(depot_city)
    if not (0 <= depot_city < n):
        raise ValueError(f"depot_city must be in [0, {n - 1}]")

    order = [depot_city] + [i for i in range(n) if i != depot_city]
    inv_order = {new: old for new, old in enumerate(order)}
    dist = dist[np.ix_(order, order)]

    full = (1 << n) - 1
    dp = {(1, 0): 0.0}
    parent: dict[tuple[int, int], list[int]] = {}
    for mask in range(1, full + 1):
        if not (mask & 1):
            continue
        for j in range(n):
            key = (mask, j)
            if key not in dp:
                continue
            for k in range(1, n):
                if mask & (1 << k):
                    continue
                nk = (mask | (1 << k), k)
                val = dp[key] + dist[j, k]
                old = dp.get(nk, np.inf)
                if val < old - 1e-12:
                    dp[nk], parent[nk] = val, [j]
                elif abs(val - old) <= 1e-12:
                    parent.setdefault(nk, []).append(j)

    best, lasts = np.inf, []
    for j in range(1, n):
        val = dp.get((full, j), np.inf) + dist[j, 0]
        if val < best - 1e-12:
            best, lasts = val, [j]
        elif abs(val - best) <= 1e-12:
            lasts.append(j)

    def backtrack(mask: int, j: int) -> list[list[int]]:
        if mask == 1 and j == 0:
            return [[0]]
        out = []
        for p in parent.get((mask, j), []):
            for r in backtrack(mask ^ (1 << j), p):
                out.append(r + [j])
                if len(out) >= max_tours:
                    return out
        return out

    routes = []
    for j in lasts:
        for r in backtrack(full, j):
            routes.append([int(inv_order[idx]) for idx in (r + [0])])
            if len(routes) >= max_tours:
                break
        if len(routes) >= max_tours:
            break
    return {"cost": float(best), "routes": routes, "dp": dp}
