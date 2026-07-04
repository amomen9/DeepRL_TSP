"""
Helper_parallel_run.py - Unified cross-algorithm parallel execution engine.

Contents
--------
run_parallel_task_groups - Run a heterogeneous set of (group x rep) training
    tasks in ONE ProcessPoolExecutor, rendering the project's shared per-rep
    tqdm step-bars driven by per-task shared step counters. Every algorithm
    family (A2C/SAC/PPO policy-gradient reps, the BELLO pointer-network baseline
    and the TSP_TEST POMO attention model) feeds tasks into the same pool, so the
    worker budget (dop) is shared and any surplus beyond it queues automatically.
"""
import os
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager


def run_parallel_task_groups(*, task_groups, max_workers, poll_interval=0.25):
    """Run all (group, rep) tasks in a single flat ``ProcessPoolExecutor``.

    Parameters
    ----------
    task_groups : list[dict]
        One entry per (algorithm, setting). Each dict has:
          ``key``    - hashable, unique; identifies the group for result routing.
          ``desc``   - str; method label shown on the bar, e.g. "PPO", "BELLO".
          ``suffix`` - str; bar-desc suffix, e.g. " (S1)"; "" for none.
          ``total``  - int; tqdm total for this group's bars (= ``n_timesteps``).
          ``tasks``  - list[dict]; one per repetition, in display order. Each has
                       ``fn`` (a pickle-safe callable) and ``kwargs`` (a dict).
                       The engine injects a fresh ``shared_step_counter`` proxy
                       into ``kwargs``; callers must not set it themselves.
    max_workers : int
        Maximum simultaneous worker processes (the shared dop). Tasks beyond this
        queue and start as workers free up - this is how the surplus of
        ``total_tasks - dop`` is recursively picked up by the pool.
    poll_interval : float
        Seconds between shared-counter polls that advance the bars.

    Returns
    -------
    dict
        ``{(group_key, local_rep_index): task_return_value}`` for every task,
        where ``local_rep_index`` is the task's 0-based position within its group.
    """
    from tqdm import tqdm

    all_tasks = []  # (group_index, local_rep_index, task_dict)
    for g, group in enumerate(task_groups):
        for r, task in enumerate(group["tasks"]):
            all_tasks.append((g, r, task))
    total_tasks = len(all_tasks)
    if total_tasks == 0:
        return {}

    with Manager() as mgr:
        step_counters = {(g, r): mgr.Value("i", 0) for (g, r, _) in all_tasks}

        with ProcessPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
            futures = {}
            for (g, r, task) in all_tasks:
                kw = dict(task["kwargs"])
                kw["shared_step_counter"] = step_counters[(g, r)]
                future = executor.submit(task["fn"], **kw)
                futures[future] = (g, r)

            # ── Physical tqdm bars ──────────────────────────────────────────
            # Reps of a group may share one bar when the total number of reps
            # would exceed the render cap; progress on a shared bar is the mean
            # advance of its reps (identical to the original PG behaviour).
            max_physical_bars = int(os.environ.get("TQDM_MAX_BARS", 11))
            n_groups = max(1, len(task_groups))
            slots_per_group = max(1, max_physical_bars // n_groups)

            pbars = {}
            rep_to_pbar = {}
            rep_group_size = {}
            group_progress = {}
            last_seen = {}

            physical_position = 0
            for g, group in enumerate(task_groups):
                n_reps = len(group["tasks"])
                if n_reps == 0:
                    continue
                desc = str(group["desc"])
                suffix = str(group.get("suffix", ""))
                total = int(group["total"])
                chunk_size = max(1, (n_reps + slots_per_group - 1) // slots_per_group)

                n_bar_groups = (n_reps + chunk_size - 1) // chunk_size
                for bar_idx in range(n_bar_groups):
                    start_r = bar_idx * chunk_size
                    if start_r >= n_reps:
                        break
                    end_r = min(n_reps, start_r + chunk_size)
                    reps = list(range(start_r, end_r))
                    group_size = len(reps)

                    reps_text = ",".join(str(i + 1) for i in reps)
                    if group_size > 6:
                        reps_text = f"{reps[0] + 1},{reps[1] + 1},...,{reps[-1] + 1}"

                    pb = tqdm(
                        total=total,
                        desc=f"{desc} Rep {reps_text}/{n_reps}{suffix}",
                        unit="step",
                        position=physical_position,
                        leave=True,
                        dynamic_ncols=True,
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                    )
                    pbars[(g, bar_idx)] = pb
                    rep_group_size[(g, bar_idx)] = group_size
                    group_progress[(g, bar_idx)] = 0.0

                    for r in reps:
                        rep_to_pbar[(g, r)] = (g, bar_idx)
                        last_seen[(g, r)] = 0

                    physical_position += 1

            rep_results = {}
            done = set()
            try:
                while len(done) < total_tasks:
                    for (g, r), sc in step_counters.items():
                        cur = sc.value
                        prev = last_seen.get((g, r), 0)
                        delta = cur - prev
                        if delta > 0:
                            pkey = rep_to_pbar[(g, r)]
                            group_size = rep_group_size[pkey]
                            group_progress[pkey] += delta / float(group_size)
                            whole_steps = int(group_progress[pkey])
                            if whole_steps > 0:
                                pbars[pkey].update(whole_steps)
                                group_progress[pkey] -= whole_steps
                            last_seen[(g, r)] = cur

                    for f in list(futures):
                        if f not in done and f.done():
                            done.add(f)
                            g, r = futures[f]
                            rep_results[(g, r)] = f.result()

                            pkey = rep_to_pbar[(g, r)]
                            group_size = rep_group_size[pkey]
                            total = int(task_groups[g]["total"])
                            prev = last_seen.get((g, r), 0)
                            remaining = total - prev
                            if remaining > 0:
                                group_progress[pkey] += remaining / float(group_size)
                                whole_steps = int(group_progress[pkey])
                                if whole_steps > 0:
                                    pbars[pkey].update(whole_steps)
                                    group_progress[pkey] -= whole_steps
                                last_seen[(g, r)] = total

                    time.sleep(poll_interval)
            finally:
                for pb in pbars.values():
                    pb.close()
                print()

    # Route results back to the caller keyed by (group key, local rep index).
    results = {}
    for (g, r), value in rep_results.items():
        results[(task_groups[g]["key"], r)] = value
    return results
