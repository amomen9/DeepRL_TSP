"""Resource-safe wrappers for exact TSP solvers.

The exact methods are exponential.  This module avoids hand-crafted node limits:
1) estimate memory from the number of Held-Karp / VI states;
2) skip before running if the estimate is larger than the memory budget;
3) run the solver in a child process and terminate it on timeout.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import multiprocessing as mp
import os
import queue
import time
from typing import Any


@dataclass
class ExactLimits:
    timeout_min: float = 2.0
    memory_gb: float | None = None       # explicit budget; None means use available-memory fraction
    memory_frac: float = 0.70            # fraction of currently available memory usable by one exact job


def _available_memory_bytes() -> int | None:
    """Best-effort available-memory query without requiring psutil."""
    try:
        import psutil  # type: ignore
        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    sysconf = getattr(os, "sysconf", None)
    if sysconf is not None:
        try:
            pages = sysconf("SC_AVPHYS_PAGES")
            page_size = sysconf("SC_PAGE_SIZE")
            return int(pages * page_size)
        except Exception:
            return None
    return None


def exact_state_count(n: int) -> int:
    """Approximate number of subset-current states for exact TSP methods."""
    if n <= 1:
        return 1
    return int(n * (1 << max(0, n - 1)))


def estimate_exact_memory_bytes(n: int, method: str) -> int:
    """Conservative Python-object memory estimate for the exact solvers.

    Both current implementations use dictionaries / Python tuples, not dense C
    arrays, so per-state overhead is intentionally conservative.
    """
    states = exact_state_count(n)
    method = method.upper()
    bytes_per_state = 420 if method == "DP" else 360
    overhead = 64 * 1024 * 1024
    return int(states * bytes_per_state + overhead)


def memory_budget_bytes(limits: ExactLimits) -> int | None:
    if limits.memory_gb is not None and limits.memory_gb > 0:
        return int(limits.memory_gb * 1024**3)
    avail = _available_memory_bytes()
    if avail is None:
        return None
    return int(max(0.05, min(1.0, limits.memory_frac)) * avail)


def format_bytes(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "unknown"
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TB"


def _dp_job(matrix, depot_city: int = 0) -> dict[str, Any]:
    from classic_DP import solve_classic_dp
    from env import actions_from_route
    out = solve_classic_dp(matrix, max_tours=1, depot_city=depot_city)
    # Do not send the exponential DP table back to the parent process.
    out.pop("dp", None)
    route = out["routes"][0] if out.get("routes") else []
    out["actions"] = actions_from_route(route, depot_city=depot_city) if route else []
    return out


def _vi_job(env_kwargs: dict[str, Any], seed: int | None) -> dict[str, Any]:
    from env import StochasticTSPEnv
    from value_iteration import value_iteration
    env = StochasticTSPEnv(**env_kwargs, seed=seed)
    env.reset(seed)
    out = value_iteration(env)
    # Do not send the exponential value table / policy dictionary back.
    out.pop("V", None)
    out.pop("policy", None)
    return out


def _worker(kind: str, args: tuple[Any, ...], q) -> None:
    try:
        if kind == "DP":
            result = _dp_job(*args)
        elif kind == "VI":
            result = _vi_job(*args)
        else:
            raise ValueError(f"unknown exact method kind={kind}")
        q.put({"ok": True, **result})
    except BaseException as exc:
        q.put({"ok": False, "reason": f"failed: {type(exc).__name__}: {exc}"})


def run_exact_with_guard(kind: str, n: int, args: tuple[Any, ...], limits: ExactLimits) -> dict[str, Any]:
    """Run an exact method if memory estimate and timeout allow it.

    Returns a dict with ``ok``.  On skip/failure/timeout, ``reason`` explains why.
    """
    kind = kind.upper()
    est = estimate_exact_memory_bytes(n, kind)
    budget = memory_budget_bytes(limits)
    if budget is not None and est > budget:
        return {
            "ok": False,
            "reason": f"skipped: estimated {kind} memory {format_bytes(est)} exceeds budget {format_bytes(budget)}",
            "estimated_memory_bytes": est,
        }

    timeout_sec = max(1.0, float(limits.timeout_min) * 60.0)
    ctx = mp.get_context("spawn")
    q = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_worker, args=(kind, args, q), daemon=True)
    t0 = time.perf_counter()
    proc.start()
    proc.join(timeout_sec)

    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
            proc.join(5)
        return {
            "ok": False,
            "reason": f"timeout: {kind} exceeded {limits.timeout_min:g} minute(s)",
            "estimated_memory_bytes": est,
            "elapsed_sec": time.perf_counter() - t0,
        }

    try:
        out = q.get_nowait()
    except queue.Empty:
        code = proc.exitcode
        return {
            "ok": False,
            "reason": f"failed: {kind} worker exited without result (exitcode={code})",
            "estimated_memory_bytes": est,
            "elapsed_sec": time.perf_counter() - t0,
        }
    out["estimated_memory_bytes"] = est
    out["elapsed_sec"] = time.perf_counter() - t0
    return out
