"""
Library_experiment_orchestrator.py - Main experiment orchestrator and per-repetition driver.

Contents
--------
run_selected_experiments - Top-level driver: builds jobs, runs / loads results,
                           saves workbooks, and plots learning curves.
_run_single_repetition   - One TSP training repetition (pickle-safe), dispatched
                           by the parallel pool.
"""
import os
import time
import copy
from datetime import datetime
from typing import Any, cast
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import torch

from .Library_plotting import LearningCurvePlot
from .Library_aggregation import _expand_k_order_configs


def _open_in_file_explorer(path: str) -> None:
    """Open ``path`` in the OS file explorer (best-effort, never raises)."""
    import sys
    import subprocess

    try:
        abspath = os.path.abspath(path)
        if sys.platform.startswith("win"):
            os.startfile(abspath)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", abspath])
        else:
            subprocess.Popen(["xdg-open", abspath])
    except Exception as exc:
        print(f"[explorer] Could not open '{path}' in file explorer: {exc}")


def _build_pg_setting_tasks(
    *, env, job, sp, n_repetitions, n_timesteps, eval_interval, max_episode_length,
    base_seed, use_saved_disk_networks_checkpoints, skip_selection_hyperparameter_match,
    match_training_matrices, n_eval_episodes=1,
):
    """Build the per-repetition parallel tasks for one A2C/SAC/PPO setting.

    Each task is a ``{"fn", "kwargs"}`` dict consumed by the unified pool engine
    (``run_parallel_task_groups``), which injects the ``shared_step_counter``.
    Mirrors the kwargs the old ``_run_pending_parallel`` built inline.
    """
    kw = dict(job["kwargs"])
    method = str(job["method"]).lower()
    tasks = []
    for r in range(n_repetitions):
        tasks.append({
            "fn": _run_single_repetition,
            "kwargs": dict(
                env=env,
                policy_based_method=method,
                actor_hidden_nn=kw["actor_hidden_nn"],
                critic_hidden_nn=kw.get("critic_hidden_nn", np.array([64, 64])),
                actor_lr=kw["actor_lr"],
                critic_lr=kw.get("critic_lr", 0.001),
                gamma=kw["gamma"],
                max_episode_length=max_episode_length,
                n_timesteps=n_timesteps,
                eval_interval=eval_interval,
                n_eval_episodes=n_eval_episodes,
                run_seed=base_seed + r,
                rep_index=r,
                n_repetitions=n_repetitions,
                enable_progress_bar=False,
                TN_step=int(kw.get("TN_step", 10)),
                alpha=float(kw.get("alpha", 0.2)),
                alpha_lr=float(kw.get("alpha_lr", 0.001)),
                auto_tune_alpha=bool(kw.get("auto_tune_alpha", True)),
                target_entropy_ratio=float(kw.get("target_entropy_ratio", 0.98)),
                tau=float(kw.get("tau", 0.005)),
                gae_lambda=float(kw.get("gae_lambda", 0.95)),
                clip_epsilon=float(kw.get("clip_epsilon", 0.2)),
                n_epochs=int(kw.get("n_epochs", 4)),
                entropy_coef=float(kw.get("entropy_coef", 0.01)),
                value_coef=float(kw.get("value_coef", 0.5)),
                full_episode_updates=bool(kw.get("full_episode_updates", True)),
                rollout_steps=int(kw.get("rollout_steps", 2048)),
                checkpoint_suffix=f"S{sp + 1}",
                use_saved_disk_networks_checkpoints=use_saved_disk_networks_checkpoints,
                skip_selection_hyperparameter_match=skip_selection_hyperparameter_match,
                match_training_matrices=match_training_matrices,
            ),
        })
    return tasks


# ── Main orchestrator ─────────────────────────────────────────────────────────

def run_selected_experiments(
    experiments,    # list of algorithm names to run, e.g. ["A2C", "SAC", "PPO", "TSP_TEST"]. Case-insensitive.
    *,
    global_config=None,
    a2c_config=None,
    sac_config=None,
    ppo_config=None,
    tsp_test_config=None,
):
    """Orchestrate training, data loading, and plotting for all selected experiments.

    Parameters
    ----------
    experiments : list[str]
        Algorithm names to run. Supported: "A2C", "SAC", "PPO", "TSP_TEST".
    global_config : dict
        Global/shared parameters (benchmark, plotting, environment, seed).
    a2c_config : dict or None
        A2C-specific hyperparameters. Required when "A2C" in experiments.
    sac_config : dict or None
        SAC-specific hyperparameters. Required when "SAC" in experiments.
    ppo_config : dict or None
        PPO-specific hyperparameters. Required when "PPO" in experiments.
    tsp_test_config : dict or None
        TSP-DRL-Test (POMO Attention Model) hyperparameters, defined in
        Experiment.py. Required when "TSP_TEST" in experiments. TSP_TEST is
        trained on the fixed instance by a dedicated runner
        (Library_tsp_test_experiment), not the A2C/SAC/PPO per-repetition
        parallel path.
    """
    # ── Unpack global config ──
    gc = global_config or {}
    env = gc.get("Environment", None)
    benchmark_curve = gc.get("benchmark_curve", 1)
    benchmark_name = gc.get("benchmark_name", "Baseline")
    n_repetitions = int(gc.get("n_repetitions", 5))
    plot_smoothing_window = gc.get("plot_smoothing_window", np.array([1]))
    curve_confidence_interval = gc.get("curve_confidence_interval", 0.6)
    curve_shaded_area_opacity = gc.get("curve_shaded_area_opacity", 0.05)
    use_existing_disk_data = gc.get("use_existing_disk_data", True)
    # Checkpoint reuse / continuation (imported from the CartPole fork). When
    # 'use_saved_disk_networks_checkpoints' is True each repetition loads a
    # matching saved actor/critic from disk and *continues* training from it,
    # accumulating the timestep counter in the sidecar. 'skip_selection_
    # hyperparameter_match' relaxes the exact-metadata match to "largest
    # n_timesteps among architecture/n_actions-compatible candidates", which is
    # the mode to use for repeated continuation runs.
    #
    # Configured via a nested 'checkpoints' dict in global_config, e.g.
    #   "checkpoints": {
    #       "use_saved_disk_networks_checkpoints": bool,
    #       "skip_selection_hyperparameter_match": bool,
    #   }
    # The legacy flat keys are still accepted as a fallback.
    # 'match_training_matrices' additionally keys checkpoint matching on the
    # instance-defining matrices (duration / inclusion / potential-uncertainty):
    # when True a saved actor/critic is only reused if it was trained on the same
    # instance. Excel-results matching always keys on the matrices; this flag
    # governs the checkpoint path only. Default True (correctness); set False to
    # restore the legacy n_actions/architecture-only checkpoint matching.
    _ckpt_cfg = gc.get("checkpoints")
    if isinstance(_ckpt_cfg, dict):
        use_saved_disk_networks_checkpoints = bool(_ckpt_cfg.get("use_saved_disk_networks_checkpoints", False))
        skip_selection_hyperparameter_match = bool(_ckpt_cfg.get("skip_selection_hyperparameter_match", False))
        match_training_matrices = bool(_ckpt_cfg.get("match_training_matrices", gc.get("match_training_matrices", True)))
    else:
        use_saved_disk_networks_checkpoints = bool(gc.get("use_saved_disk_networks_checkpoints", False))
        skip_selection_hyperparameter_match = bool(gc.get("skip_selection_hyperparameter_match", False))
        match_training_matrices = bool(gc.get("match_training_matrices", True))
    # Set True once we confirm at least one worker will actually continue from a
    # saved checkpoint. Stays False when reuse is off, or reuse is on but no saved
    # network matched (a fresh run), which keeps results out of the continuation dir.
    any_checkpoint_loaded = False
    format_sheets = bool(gc.get("format_sheets", False))
    formatted_sheets = bool(gc.get("formatted_sheets", False))
    n_timesteps = int(gc.get("n_timesteps", 100000))
    eval_interval = int(gc.get("eval_interval", 250))
    # Training-time greedy-eval episodes per eval point. The depot cycles on
    # every env.reset(), so a single eval episode measures a different depot at
    # each eval point and the learning curve zig-zags for measurement reasons
    # alone; averaging over several consecutive depots removes that artefact.
    n_eval_episodes = int(gc.get("n_eval_episodes", 1))
    max_episode_length = int(gc.get("max_episode_length", 500))
    base_seed = int(gc.get("base_seed", 42))
    curve_plot = gc.get("curve_plot", False)
    animation_plot = gc.get("animation_plot", False)
    show_curve_plots = bool(gc.get("show_curve_plots", curve_plot))
    raw_optimal_plot_value = gc.get("TSP_Optimal_Cost", 0.0)
    optimal_plot_value = None if raw_optimal_plot_value is None else float(raw_optimal_plot_value)
    k_order_aggregation_methods = gc.get("k_order_aggregation_methods", None)

    if env is None:
        print("Env is None")
        raise ValueError("Environment must be provided either as a parameter or via global_config['env'] and cannot be None.")

    assert env is not None, "Environment must be provided either as a parameter or via global_config['env']"

    from .Helper_math import _apply_optional_smoothing
    from .Helper_excel import _load_benchmark_curve

    optimum_return = None if optimal_plot_value is None else -float(optimal_plot_value)

    baseline_model = gc.get("baseline_model", None)
    baseline_algo_upper = str(baseline_model).upper() if baseline_model else None
    # Ensure the selected baseline is among the experiments so its curve is
    # produced and can be promoted to the benchmark. A2C/SAC/PPO run through the
    # policy-gradient job path; TSP_TEST and BELLO are dedicated-runner methods
    # that skip that path (handled by their own prepare/finalize below) but are
    # still listed as experiments.
    if baseline_algo_upper is not None:
        already_listed = {str(e).upper() for e in experiments}
        if baseline_algo_upper not in already_listed:
            experiments = list(experiments) + [baseline_algo_upper]

    start_time = time.perf_counter()
    start_human = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Policy-based experiment started at: {start_human}\n")
    # Disabled (kept for reference): output.log generation
    # with open("output.log", "w", encoding="utf-8") as f:
    #     f.write(f"Start the process at: {start_human}\n")
    print(f"Included experiments: {', '.join(experiments)}\n")

    stochastic_duration_matrix_mean = None
    policy_algos = {str(e).upper() for e in experiments if str(e).upper() in {"A2C", "SAC", "PPO"}}
    if policy_algos and hasattr(env, "reseed_noise") and hasattr(env, "stochastic_duration_matrix"):
        acc: np.ndarray | None = None
        for rep in range(n_repetitions):
            run_seed = base_seed + rep
            env.reseed_noise(run_seed)
            mat = np.asarray(env.stochastic_duration_matrix, dtype=np.float32)
            if acc is None:
                acc = np.zeros_like(mat, dtype=np.float64)
            acc += mat.astype(np.float64)
        if acc is not None:
            stochastic_duration_matrix_mean = (acc / float(n_repetitions)).astype(np.float32)

    from .Helper_jobs_and_hp_sweeps import (
        _build_a2c_jobs,
        _build_algo_filename,
        _build_ppo_jobs,
        _build_sac_jobs,
    )
    from .Helper_excel import (
        _load_all_excel_curves,
        _load_results_from_excel,
        save_algorithm_workbook,
    )

    curve_confidence_interval = float(curve_confidence_interval)
    if curve_confidence_interval < 0.0 or curve_confidence_interval >= 1.0:
        raise ValueError("curve_confidence_interval must be in [0, 1).")
    shade_ci = curve_confidence_interval > 0.0
    curve_ci_alpha = 1.0 - curve_confidence_interval if shade_ci else None
    curve_shaded_area_opacity = float(curve_shaded_area_opacity)

    plot_smoothing_windows = np.atleast_1d(np.asarray(plot_smoothing_window, dtype=np.int32))
    if plot_smoothing_windows.size < 1:
        raise ValueError("plot_smoothing_window must contain at least one value.")

    algo_jobs = {}
    algo_configs_map = {
        "A2C": a2c_config,
        "SAC": sac_config,
        "PPO": ppo_config,
    }
    all_setting_jobs = []
    for algo in experiments:
        algo_upper = algo.upper()
        jobs = []
        if algo_upper == "A2C":
            if a2c_config is None:
                raise ValueError("a2c_config dict is required when A2C is included.")
            jobs = _build_a2c_jobs(
                algo_config=a2c_config,
                n_repetitions=n_repetitions,
                n_timesteps=n_timesteps,
                eval_interval=eval_interval,
                max_episode_length=max_episode_length,
                base_seed=base_seed,
            )
        elif algo_upper == "SAC":
            if sac_config is None:
                raise ValueError("sac_config dict is required when SAC is included.")
            jobs = _build_sac_jobs(
                algo_config=sac_config,
                n_repetitions=n_repetitions,
                n_timesteps=n_timesteps,
                eval_interval=eval_interval,
                max_episode_length=max_episode_length,
                base_seed=base_seed,
            )
        elif algo_upper == "PPO":
            if ppo_config is None:
                raise ValueError("ppo_config dict is required when PPO is included.")
            jobs = _build_ppo_jobs(
                algo_config=ppo_config,
                n_repetitions=n_repetitions,
                n_timesteps=n_timesteps,
                eval_interval=eval_interval,
                max_episode_length=max_episode_length,
                base_seed=base_seed,
            )
        elif algo_upper == "TSP_TEST":
            # TSP-DRL-Test (POMO Attention Model): trained on the fixed
            # instance by a dedicated runner below - it must not enter the
            # A2C/SAC/PPO job-building / parallel-run path.
            if tsp_test_config is None:
                raise ValueError("tsp_test_config dict is required when TSP_TEST is included.")
            continue
        elif algo_upper == "BELLO":
            # Bello Pointer-Network baseline: trained on the fixed instance by a
            # dedicated runner below (immutable hyperparameters, no config dict) -
            # it must not enter the A2C/SAC/PPO job-building / parallel-run path.
            continue
        else:
            raise ValueError(f"Unknown algorithm: {algo}")
        algo_jobs[algo_upper] = jobs
        all_setting_jobs.extend(jobs)

    if baseline_algo_upper is None:
        benchmark_steps, benchmark_returns_raw = _load_benchmark_curve(
            benchmark_curve=benchmark_curve,
            project_eval_interval=eval_interval,
            project_n_timesteps=n_timesteps,
            episode_return_column="Episode_Return",
        )
    else:
        benchmark_steps, benchmark_returns_raw = None, None
        if "benchmark_name" not in gc:
            benchmark_name = f"{baseline_algo_upper} baseline"

    title_tag = " + ".join(experiments)

    n_cities = getattr(env, "n_cities", None)
    if n_cities is None:
        n_cities = getattr(env, "n", None)
    if n_cities is None:
        raise AttributeError("Environment must expose 'n_cities' (or 'n') for plot titles.")
    n_cities = int(n_cities)

    best_cost = gc.get("TSP_Best_Cost", None)
    worst_cost = gc.get("TSP_Worst_Cost", None)
    summary_text = None
    if best_cost is not None and worst_cost is not None:
        from .Helper_legend import _format_number_normal

        best_text = _format_number_normal(best_cost)
        worst_text = _format_number_normal(worst_cost)
        mean_cost = (float(cast(Any, worst_cost)) + float(cast(Any, best_cost))) / 2.0
        mean_text = _format_number_normal(mean_cost)

        summary_text = f"Best: {best_text} - Worst: {worst_text} - Mean: {mean_text}"

    plot_configs = []
    for window in plot_smoothing_windows:
        window = int(window)
        is_not_smoothed = window <= 1
        if benchmark_returns_raw is not None:
            benchmark_values = cast(np.ndarray, np.asarray(benchmark_returns_raw, dtype=np.float32))
            benchmark_returns = cast(np.ndarray, _apply_optional_smoothing(benchmark_values, window))
            if optimum_return is not None:
                benchmark_returns = cast(np.ndarray, np.minimum(benchmark_returns, float(optimum_return)))
            pc_benchmark_steps = cast(np.ndarray, benchmark_steps)
        else:
            benchmark_returns = None
            pc_benchmark_steps = None
        if is_not_smoothed:
            plot_obj = LearningCurvePlot(title=f"{title_tag} for {n_cities} cities - not smoothed plot", summary_text=summary_text, summary_fontsize=10)
        else:
            plot_obj = LearningCurvePlot(title=f"{title_tag} for {n_cities} cities - smoothed plot", summary_text=summary_text, summary_fontsize=10)
        plot_configs.append({
            "window": window,
            "is_not_smoothed": is_not_smoothed,
            "plot": plot_obj,
            "benchmark_steps": pc_benchmark_steps,
            "benchmark_returns": benchmark_returns,
            "y_max": float("-inf"),
            "y_min": float("inf"),
        })

    data_sheets_dir = "data sheets"
    os.makedirs(data_sheets_dir, exist_ok=True)

    setting_results: list[
        tuple[np.ndarray, np.ndarray, np.ndarray]
        | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        | None
    ] = [None] * len(all_setting_jobs)
    algo_filenames = {}
    algo_job_offsets = {}
    pending_settings = []
    algos_needing_save = set()

    offset = 0
    for algo_upper, jobs in algo_jobs.items():
        algo_job_offsets[algo_upper] = offset
        n_jobs = len(jobs)
        cfg = algo_configs_map[algo_upper]
        base_filename = _build_algo_filename(algo_upper)
        algo_filenames[algo_upper] = base_filename
        excel_path = os.path.join(data_sheets_dir, f"{base_filename}.xlsx")

        if use_existing_disk_data and os.path.isfile(excel_path):
            try:
                algo_results, _mismatches = _load_results_from_excel(
                    excel_path,
                    cfg,
                    global_config=gc,
                    formatted_sheets=formatted_sheets,
                )
            except Exception as exc:
                print(f"[{algo_upper}] Existing Excel data is incompatible. Re-running from scratch. Reason: {exc}")
                algo_results = []
                _mismatches = {}

            if algo_results:
                print(f"[{algo_upper}] Loaded {len(algo_results)} matching setting(s) from: {excel_path}")
                loaded_count = min(len(algo_results), n_jobs)
                for i, entry in enumerate(algo_results[:loaded_count]):
                    raw_returns = entry.get("raw_returns")
                    if raw_returns is not None:
                        setting_results[offset + i] = (
                            entry["learning_curve"],
                            entry["learning_curve_std"],
                            entry["timesteps"],
                            raw_returns,
                        )
                    else:
                        setting_results[offset + i] = (
                            entry["learning_curve"],
                            entry["learning_curve_std"],
                            entry["timesteps"],
                        )
                    jobs[i]["curve_label"] = entry["curve_label"]
                if len(algo_results) > n_jobs:
                    print(
                        f"[{algo_upper}] Ignoring {len(algo_results) - n_jobs} extra matching sheet(s) "
                        f"because only {n_jobs} job(s) are configured."
                    )
                if len(algo_results) < n_jobs:
                    print(f"[{algo_upper}] Only {len(algo_results)}/{n_jobs} sheets matched. "
                          f"Running remaining {n_jobs - len(algo_results)} from scratch.")
                    for i in range(len(algo_results), n_jobs):
                        pending_settings.append((offset + i, jobs[i]))
                    algos_needing_save.add(algo_upper)
            else:
                mismatch_str = ""
                if _mismatches:
                    parts = []
                    for param, (sheet_val, cfg_val) in _mismatches.items():
                        cfg_display = (
                            str(list(cfg_val)) if isinstance(cfg_val, (list, np.ndarray))
                            else str(cfg_val)
                        )
                        parts.append(f"{param} (Disk data: {sheet_val}, Config: {cfg_display})")
                    mismatch_str = "  Mismatch reason(s): " + "; ".join(parts)
                print(f"[{algo_upper}] No matching sheets found in Excel. Re-running from scratch.{mismatch_str}")
                for i, job in enumerate(jobs):
                    pending_settings.append((offset + i, job))
                algos_needing_save.add(algo_upper)
        else:
            print(f"[{algo_upper}] Running experiments from scratch.\n")
            for i, job in enumerate(jobs):
                pending_settings.append((offset + i, job))
            algos_needing_save.add(algo_upper)

        offset += n_jobs

    # ── Build the unified parallel task list ─────────────────────────────────
    # Every algorithm family feeds (setting/rep) tasks into ONE process pool so
    # the worker budget (dop = cpu_count - MIN_UNUSED_CPU_CORES) is shared across
    # the A2C/SAC/PPO settings, the BELLO baseline and the TSP_TEST experiment.
    # Any surplus beyond dop queues inside the pool automatically. BELLO and
    # TSP_TEST do their disk-load / top-up decision here (main process); only the
    # reps that still need training become pool tasks, and their results are
    # finalized after the pool (curve assembly, workbook, route plot, promotion).
    from .Helper_parallel_run import run_parallel_task_groups

    # BELLO baseline: decide how many reps still need training (disk top-up).
    bello_plan = None
    if any(str(e).upper() == "BELLO" for e in experiments):
        from .Library_bello_baseline import prepare_bello_baseline
        bello_plan = prepare_bello_baseline(
            env=env, global_config=gc, n_timesteps=n_timesteps, eval_interval=eval_interval,
            max_episode_length=max_episode_length, base_seed=base_seed,
            data_sheets_dir=data_sheets_dir, formatted_sheets=formatted_sheets,
            match_training_matrices=match_training_matrices,
        )

    # TSP_TEST experiment: decide how many reps still need training (disk top-up).
    tsp_test_plan = None
    if any(str(e).upper() == "TSP_TEST" for e in experiments):
        from .Library_tsp_test_experiment import prepare_tsp_test_experiment
        tsp_test_plan = prepare_tsp_test_experiment(
            env=env, global_config=gc, algo_config=tsp_test_config,
            n_timesteps=n_timesteps, eval_interval=eval_interval,
            max_episode_length=max_episode_length, base_seed=base_seed,
            data_sheets_dir=data_sheets_dir, formatted_sheets=formatted_sheets,
            match_training_matrices=match_training_matrices,
        )

    # Assemble the task groups (PG settings + BELLO reps + TSP_TEST reps).
    task_groups = []
    multi_pg = len(pending_settings) > 1
    for sp, (global_idx, job) in enumerate(pending_settings):
        task_groups.append({
            "key": f"PG:{sp}",
            "desc": str(job["method"]).upper(),
            "suffix": f" (S{sp + 1})" if multi_pg else "",
            "total": n_timesteps,
            "tasks": _build_pg_setting_tasks(
                env=env, job=job, sp=sp, n_repetitions=n_repetitions,
                n_timesteps=n_timesteps, eval_interval=eval_interval,
                max_episode_length=max_episode_length, base_seed=base_seed,
                use_saved_disk_networks_checkpoints=use_saved_disk_networks_checkpoints,
                skip_selection_hyperparameter_match=skip_selection_hyperparameter_match,
                match_training_matrices=match_training_matrices,
                n_eval_episodes=n_eval_episodes,
            ),
        })

    bello_n_to_train = 0
    if bello_plan is not None and bello_plan["status"] == "train":
        from .Library_bello_baseline import _run_single_bello_rep
        bello_n_to_train = int(bello_plan["n_to_train"])
        if bello_n_to_train > 0:
            bello_tasks = []
            for i in range(bello_n_to_train):
                rep_index = int(bello_plan["disk_reps"]) + i
                bello_tasks.append({
                    "fn": _run_single_bello_rep,
                    "kwargs": dict(
                        duration_matrix=bello_plan["duration_matrix"],
                        n_cities=bello_plan["n_cities"],
                        n_timesteps=bello_plan["n_timesteps"],
                        timesteps_grid=bello_plan["timesteps_grid"],
                        seed=int(base_seed) + rep_index,
                        rep_index=rep_index,
                        n_repetitions=int(bello_plan["target_reps"]),
                        metadata_base=bello_plan["metadata_base"],
                        instance_hash=bello_plan["instance_hash"],
                        instance_text=bello_plan["instance_text"],
                    ),
                })
            task_groups.append({
                "key": "BELLO", "desc": "BELLO", "suffix": "",
                "total": n_timesteps, "tasks": bello_tasks,
            })

    tsp_test_n_to_train = 0
    if tsp_test_plan is not None and tsp_test_plan["status"] == "train":
        from .Library_tsp_test_experiment import _run_single_tsp_test_rep
        tsp_test_n_to_train = int(tsp_test_plan["n_to_train"])
        if tsp_test_n_to_train > 0:
            tsp_test_tasks = []
            for i in range(tsp_test_n_to_train):
                rep_index = int(tsp_test_plan["disk_reps"]) + i
                tsp_test_tasks.append({
                    "fn": _run_single_tsp_test_rep,
                    "kwargs": dict(
                        duration_matrix=tsp_test_plan["duration_matrix"],
                        n_cities=tsp_test_plan["n_cities"],
                        hp=tsp_test_plan["hp"],
                        n_timesteps=tsp_test_plan["n_timesteps"],
                        timesteps_grid=tsp_test_plan["timesteps_grid"],
                        seed=int(base_seed) + rep_index,
                        rep_index=rep_index,
                        n_repetitions=int(tsp_test_plan["target_reps"]),
                        metadata_base=tsp_test_plan["metadata_base"],
                        instance_hash=tsp_test_plan["instance_hash"],
                        instance_text=tsp_test_plan["instance_text"],
                    ),
                })
            task_groups.append({
                "key": "TSP_TEST", "desc": "TSP_TEST", "suffix": "",
                "total": n_timesteps, "tasks": tsp_test_tasks,
            })

    # ── Run the unified pool ─────────────────────────────────────────────────
    pool_results = {}
    if task_groups:
        cpu_count = os.cpu_count() or 1
        dop = max(1, cpu_count - max(0, int(os.environ.get("MIN_UNUSED_CPU_CORES", 2))))
        total_tasks = sum(len(g["tasks"]) for g in task_groups)
        max_workers = min(total_tasks, dop)

        breakdown = []
        if pending_settings:
            breakdown.append(f"{len(pending_settings)} PG setting(s) × {n_repetitions} rep(s)")
        if bello_n_to_train > 0:
            breakdown.append(f"BELLO {bello_n_to_train} rep(s)")
        if tsp_test_n_to_train > 0:
            breakdown.append(f"TSP_TEST {tsp_test_n_to_train} rep(s)")
        print(f"CPU cores available: {cpu_count}. "
              f"Total tasks: {total_tasks} ({'; '.join(breakdown)}). "
              f"Parallel workers: {max_workers}.\n")
        for global_idx, job in pending_settings:
            print(f"Setting {global_idx + 1}/{len(all_setting_jobs)}: {job['curve_label']}")
        if bello_n_to_train > 0:
            _bello_role = "Baseline" if baseline_algo_upper == "BELLO" else "Experiment"
            print(f"{_bello_role}: BELLO ({bello_n_to_train} new rep(s))")
        if tsp_test_n_to_train > 0:
            print(f"Experiment: TSP_TEST ({tsp_test_n_to_train} new rep(s))")
        print()

        # Up-front report of which network checkpoints the workers will load
        # (emitted before the pool starts so the lines are not interleaved with
        # the child processes' tqdm bars). Checkpoint continuation is a
        # policy-gradient-only feature, so the report covers PG settings only.
        if use_saved_disk_networks_checkpoints and pending_settings:
            any_checkpoint_loaded = _report_checkpoint_loads(
                pending_settings=pending_settings,
                env=env,
                n_repetitions=n_repetitions,
                n_timesteps=n_timesteps,
                eval_interval=eval_interval,
                max_episode_length=max_episode_length,
                skip_selection_hyperparameter_match=skip_selection_hyperparameter_match,
                match_training_matrices=match_training_matrices,
            ) > 0

        pool_results = run_parallel_task_groups(
            task_groups=task_groups, max_workers=max_workers,
        )

    # ── Route policy-gradient results back into setting_results ──────────────
    for sp, (global_idx, _job) in enumerate(pending_settings):
        returns_list = [
            np.asarray(pool_results[(f"PG:{sp}", r)][0], dtype=np.float32)
            for r in range(n_repetitions)
        ]
        ts = np.asarray(pool_results[(f"PG:{sp}", 0)][1], dtype=np.int32)
        raw_returns = np.asarray(returns_list, dtype=np.float32)
        lc_mean = np.mean(raw_returns, axis=0)
        lc_std = (
            np.std(raw_returns, axis=0, ddof=1)
            if len(returns_list) > 1 else np.zeros_like(lc_mean)
        )
        setting_results[global_idx] = (lc_mean, lc_std, ts, raw_returns)

    for algo_upper in algos_needing_save:
        jobs = algo_jobs[algo_upper]
        off = algo_job_offsets[algo_upper]
        cfg = algo_configs_map[algo_upper]
        base_filename = algo_filenames[algo_upper]
        algo_results_to_save = [setting_results[off + i] for i in range(len(jobs))]
        if any(r is not None for r in algo_results_to_save):
            save_algorithm_workbook(
                data_sheets_dir,
                base_filename,
                algo_upper,
                jobs,
                algo_results_to_save,
                global_config=gc,
                algo_config=cfg,
                format_sheets=format_sheets,
            )

    if baseline_algo_upper is not None and baseline_algo_upper in algo_jobs:
        off = algo_job_offsets.get(baseline_algo_upper, 0)
        baseline_result = setting_results[off] if off < len(setting_results) else None
        if baseline_result is not None:
            lc_mean_baseline, _lc_std_baseline, ts_baseline = baseline_result[:3]
            benchmark_steps = np.asarray(ts_baseline, dtype=np.int32)
            benchmark_returns_raw = np.asarray(lc_mean_baseline, dtype=np.float32)
            for pc in plot_configs:
                smoothed = cast(np.ndarray, _apply_optional_smoothing(benchmark_returns_raw, int(pc["window"])))
                if optimum_return is not None:
                    smoothed = cast(np.ndarray, np.minimum(smoothed, float(optimum_return)))
                pc["benchmark_steps"] = benchmark_steps
                pc["benchmark_returns"] = smoothed

    # ── Bello Pointer-Network baseline (finalize) ─────────────────────────────
    # BELLO trained its shortfall reps in the shared pool above; assemble the
    # combined curve here (or reuse the fully-on-disk curve produced by prepare).
    # Its curve is drawn as a regular experiment curve next to A2C/SAC/PPO -
    # unless it was selected as the baseline_model, in which case it becomes the
    # benchmark curve instead (mirroring the TSP_TEST handling below). Disk reuse
    # uses the same matching rules as the other algorithms, but a matching
    # BELLO.xlsx always short-circuits retraining regardless of the global
    # 'use_existing_disk_data'.
    bello_result = None
    bello_label = None
    if bello_plan is not None:
        if bello_plan["status"] == "ready":
            bello_result = bello_plan["result"]
            bello_label = bello_plan["curve_label"]
        elif bello_plan["status"] == "train" and bello_n_to_train > 0:
            from .Library_bello_baseline import finalize_bello_baseline
            bello_new_reps = [pool_results[("BELLO", i)] for i in range(bello_n_to_train)]
            bello_result, bello_label = finalize_bello_baseline(
                bello_plan, bello_new_reps,
                base_seed=base_seed, data_sheets_dir=data_sheets_dir, plots_dir="plots",
                global_config=gc, format_sheets=format_sheets,
            )
        if bello_result is not None and baseline_algo_upper == "BELLO":
            lc_mean_bello, _lc_std_bello, ts_bello = bello_result[:3]
            benchmark_steps = np.asarray(ts_bello, dtype=np.int32)
            benchmark_returns_raw = np.asarray(lc_mean_bello, dtype=np.float32)
            for pc in plot_configs:
                smoothed = cast(np.ndarray, _apply_optional_smoothing(benchmark_returns_raw, int(pc["window"])))
                if optimum_return is not None:
                    smoothed = cast(np.ndarray, np.minimum(smoothed, float(optimum_return)))
                pc["benchmark_steps"] = benchmark_steps
                pc["benchmark_returns"] = smoothed

    # ── TSP-DRL-Test (POMO Attention Model) experiment (finalize) ─────────────
    # TSP_TEST trained its shortfall reps in the shared pool above; assemble the
    # combined curve here (or reuse the fully-on-disk curve produced by prepare).
    # Its curve is drawn as a regular experiment curve next to A2C/SAC/PPO -
    # unless it was selected as the baseline_model, in which case it becomes the
    # benchmark curve instead (mirroring the Bello handling above).
    tsp_test_result = None
    tsp_test_label = None
    if tsp_test_plan is not None:
        if tsp_test_plan["status"] == "ready":
            tsp_test_result = tsp_test_plan["result"]
            tsp_test_label = tsp_test_plan["curve_label"]
        elif tsp_test_plan["status"] == "train" and tsp_test_n_to_train > 0:
            from .Library_tsp_test_experiment import finalize_tsp_test_experiment
            tsp_test_new_reps = [pool_results[("TSP_TEST", i)] for i in range(tsp_test_n_to_train)]
            tsp_test_result, tsp_test_label = finalize_tsp_test_experiment(
                tsp_test_plan, tsp_test_new_reps,
                base_seed=base_seed, data_sheets_dir=data_sheets_dir, plots_dir="plots",
                global_config=gc, format_sheets=format_sheets,
            )
        if tsp_test_result is not None and baseline_algo_upper == "TSP_TEST":
            lc_mean_tt, _lc_std_tt, ts_tt = tsp_test_result[:3]
            benchmark_steps = np.asarray(ts_tt, dtype=np.int32)
            benchmark_returns_raw = np.asarray(lc_mean_tt, dtype=np.float32)
            for pc in plot_configs:
                smoothed = cast(np.ndarray, _apply_optional_smoothing(benchmark_returns_raw, int(pc["window"])))
                if optimum_return is not None:
                    smoothed = cast(np.ndarray, np.minimum(smoothed, float(optimum_return)))
                pc["benchmark_steps"] = benchmark_steps
                pc["benchmark_returns"] = smoothed

    current_basenames = set(f"{fn}.xlsx" for fn in algo_filenames.values())

    extra_curves = []
    if use_existing_disk_data:
        algo_configs = {k: v for k, v in algo_configs_map.items() if v is not None}
        all_disk_curves = _load_all_excel_curves(
            data_sheets_dir,
            algo_configs,
            global_config=gc,
            formatted_sheets=formatted_sheets,
        )
        for curve_info in all_disk_curves:
            if curve_info["source_file"] in current_basenames:
                continue
            source_algo = os.path.splitext(curve_info["source_file"])[0].upper()
            if source_algo not in algo_jobs:
                continue
            extra_curves.append(curve_info)
        if extra_curves:
            counts = Counter(c["source_file"] for c in extra_curves)
            for fname, n in sorted(counts.items()):
                print(f"Loaded {n} additional curve(s) from '{fname}' Excel file in '{data_sheets_dir}'.")

    def _emit_curves_for_setting(
        *,
        lc_mean_fallback,
        lc_std_fallback,
        raw_returns,
        timesteps,
        base_label,
        n_for_ci_fallback,
        curve_ls="solid",
        n_reps_for_agg=None,
    ):
        # n_reps_for_agg lets curves whose repetition count differs from the
        # project-wide n_repetitions (e.g. TSP_TEST) still get their k-order
        # aggregation expansion; None keeps the historical behaviour.
        configs = _expand_k_order_configs(
            k_order_aggregation_methods, raw_returns, timesteps,
            n_repetitions if n_reps_for_agg is None else int(n_reps_for_agg),
        )
        raw_arr = np.asarray(raw_returns, dtype=np.float32) if raw_returns is not None else None
        for suffix, rep_idx in configs:
            if suffix == "":
                lc_raw = np.asarray(lc_mean_fallback, dtype=np.float32)
                lc_std_raw = np.asarray(lc_std_fallback, dtype=np.float32)
                n_for_ci = n_for_ci_fallback
                label = base_label
            else:
                assert raw_arr is not None
                sel = raw_arr[np.asarray(rep_idx, dtype=np.int64), :]
                lc_raw = np.mean(sel, axis=0).astype(np.float32)
                if sel.shape[0] > 1:
                    lc_std_raw = np.std(sel, axis=0, ddof=1).astype(np.float32)
                else:
                    lc_std_raw = np.zeros_like(lc_raw, dtype=np.float32)
                n_for_ci = int(sel.shape[0])
                label = f"{base_label} | {suffix}"
            for pc in plot_configs:
                window = int(pc["window"])
                timesteps_arr = cast(np.ndarray, np.array(timesteps, dtype=np.int32, copy=False))
                lc_w = cast(np.ndarray, np.array(_apply_optional_smoothing(lc_raw, window), dtype=np.float32, copy=False))
                lc_std_w = cast(np.ndarray, np.array(_apply_optional_smoothing(lc_std_raw, window), dtype=np.float32, copy=False))
                if optimum_return is None:
                    plot_values = lc_w
                else:
                    plot_values = np.minimum(lc_w, float(optimum_return))
                plot_obj = pc["plot"]
                plot_obj.add_curve(timesteps_arr.tolist(), plot_values.tolist(), label=label, ls=curve_ls)
                if plot_values.size:
                    pc["y_max"] = max(pc["y_max"], float(np.max(plot_values)))
                    pc["y_min"] = min(pc["y_min"], float(np.min(plot_values)))
                if shade_ci and n_for_ci is not None:
                    band = plot_obj.add_shaded_ci(
                        timesteps_arr.tolist(), plot_values.tolist(), lc_std_w.tolist(), n=int(n_for_ci),
                        alpha=curve_ci_alpha, fill_opacity=curve_shaded_area_opacity,
                        y_lower_cap=None,
                        y_upper_cap=(
                            None
                            if optimum_return is None
                            else float(optimum_return)
                        ),
                    )
                    # Fold the shaded band's extent into the y-limits so the top
                    # of the band (mean + CI margin) is never hidden under the
                    # legend on smoothed plots.
                    if band is not None:
                        band_lower, band_upper = band
                        if band_upper.size:
                            pc["y_max"] = max(pc["y_max"], float(np.max(band_upper)))
                            pc["y_min"] = min(pc["y_min"], float(np.min(band_lower)))

    for idx, job in enumerate(all_setting_jobs):
        res = setting_results[idx]
        if res is None:
            continue
        lc_raw, lc_std_raw, timesteps = res[:3]
        raw_returns_for_idx = res[3] if isinstance(res, tuple) and len(res) >= 4 else None
        _emit_curves_for_setting(
            lc_mean_fallback=lc_raw,
            lc_std_fallback=lc_std_raw,
            raw_returns=raw_returns_for_idx,
            timesteps=timesteps,
            base_label=job["curve_label"],
            n_for_ci_fallback=n_repetitions,
            curve_ls="solid",
        )

    for curve_info in extra_curves:
        _emit_curves_for_setting(
            lc_mean_fallback=curve_info["learning_curve"],
            lc_std_fallback=curve_info["learning_curve_std"],
            raw_returns=curve_info.get("raw_returns"),
            timesteps=curve_info["timesteps"],
            base_label=curve_info["curve_label"],
            n_for_ci_fallback=int(curve_info["n_repetitions"]) if "n_repetitions" in curve_info else None,
            curve_ls="solid",
        )

    # TSP-DRL-Test curve (regular experiment curve; skipped when it was
    # promoted to the benchmark via baseline_model == "TSP_TEST" above).
    if tsp_test_result is not None and baseline_algo_upper != "TSP_TEST":
        lc_tt, lc_std_tt, ts_tt = tsp_test_result[:3]
        raw_tt = tsp_test_result[3] if len(tsp_test_result) >= 4 else None
        n_reps_tt = int(raw_tt.shape[0]) if raw_tt is not None else None
        _emit_curves_for_setting(
            lc_mean_fallback=lc_tt,
            lc_std_fallback=lc_std_tt,
            raw_returns=raw_tt,
            timesteps=ts_tt,
            base_label=tsp_test_label or "TSP-Test (POMO-AM)",
            n_for_ci_fallback=n_reps_tt,
            curve_ls="solid",
            n_reps_for_agg=n_reps_tt,
        )

    # Bello curve (regular experiment curve; skipped when it was promoted to the
    # benchmark via baseline_model == "BELLO" above).
    if bello_result is not None and baseline_algo_upper != "BELLO":
        lc_b, lc_std_b, ts_b = bello_result[:3]
        raw_b = bello_result[3] if len(bello_result) >= 4 else None
        n_reps_b = int(raw_b.shape[0]) if raw_b is not None else None
        _emit_curves_for_setting(
            lc_mean_fallback=lc_b,
            lc_std_fallback=lc_std_b,
            raw_returns=raw_b,
            timesteps=ts_b,
            base_label=bello_label or "Bello baseline",
            n_for_ci_fallback=n_reps_b,
            curve_ls="solid",
            n_reps_for_agg=n_reps_b,
        )

    for pc in plot_configs:
        plot_obj = pc["plot"]
        if pc.get("benchmark_steps") is not None and pc.get("benchmark_returns") is not None:
            benchmark_steps_arr = cast(np.ndarray, np.array(pc["benchmark_steps"], dtype=np.int32, copy=False))
            benchmark_returns_arr = cast(np.ndarray, np.array(pc["benchmark_returns"], dtype=np.float32, copy=False))
            plot_obj.ax.plot(
                benchmark_steps_arr.tolist(), benchmark_returns_arr.tolist(),
                label=benchmark_name, ls=":", c="gray",
            )
            # The benchmark is a curve too — fold it into the y-limits so it
            # cannot hide under the legend when it is the highest line.
            if benchmark_returns_arr.size:
                pc["y_max"] = max(float(pc["y_max"]), float(np.max(benchmark_returns_arr)))
                pc["y_min"] = min(float(pc["y_min"]), float(np.min(benchmark_returns_arr)))
        if optimum_return is not None:
            plot_obj.add_hline(optimum_return, label="TSP optimum")
            x_left = float(plot_obj.ax.get_xlim()[0])
            plot_obj.ax.text(
                x_left,
                optimum_return,
                f"TSP optimum: {optimum_return:g}",
                va="top",
                ha="left",
                fontsize=8,
                color="k",
            )
        else:
            plot_obj.ax.plot([], [], label="Memory Insufficient", ls="--", c="k")

    opt = optimum_return

    # Set the y-limits to the EXACT tracked data extent (mean curves + CI bands +
    # benchmark; see y_max/y_min accumulation above). The display margins are
    # applied later, in LearningCurvePlot.save():
    #   * lower margin = 1/4 of the tracked range, below y_min;
    #   * upper gap    = 1/6 of the tracked range, between y_max and the legend
    #                    box bottom (so the highest curve/band sits just below
    #                    the legend and nothing hides under it).
    # save() needs the legend's real height for the upper gap, which is why the
    # margins are not baked in here.
    for pc in plot_configs:
        y_min_curves = float(pc.get("y_min", float("inf")))
        y_max_curves = float(pc.get("y_max", float("-inf")))

        if np.isfinite(y_min_curves) and np.isfinite(y_max_curves) and y_max_curves >= y_min_curves:
            if opt is not None:
                # DP optimum available: it is the ceiling (curves are capped at it).
                least_value = min(opt, y_min_curves)
                most_value = max(opt, y_max_curves)
            else:
                # DP skipped (memory insufficient): the biggest value across all
                # curves, CI bands and timesteps becomes the ceiling instead.
                least_value = y_min_curves
                most_value = y_max_curves

            if most_value <= least_value:  # degenerate range guard
                most_value = least_value + 1.0

            y_lower = least_value
            y_upper_final = most_value
        else:
            if opt is not None:
                y_lower = opt - 1.0
                y_upper_final = opt + 1.0
            else:
                y_lower = 0.0
                y_upper_final = 1.0

        pc["plot"].set_ylim(y_lower, y_upper_final)

    plot_filename_tag = "-".join(e.upper() for e in experiments)

    # In checkpoint-reuse / continuation mode, learning-curve figures and the
    # returns-summary artifacts go under "Trial Continuation Analysis" instead
    # of "plots" (ported from the CartPole fork) so continued-run diagnostics
    # are kept separate from from-scratch results. This only applies when a saved
    # network actually matched and was continued from; if reuse was enabled but
    # nothing loaded (a fresh run), results stay under "plots".
    plots_dir = (
        "Trial Continuation Analysis"
        if use_saved_disk_networks_checkpoints and any_checkpoint_loaded
        else "plots"
    )
    os.makedirs(plots_dir, exist_ok=True)
    existing_plot_files = {
        f for f in os.listdir(plots_dir)
        if os.path.isfile(os.path.join(plots_dir, f))
    }

    plot_entries: list[tuple[dict, str]] = []
    for pc in plot_configs:
        window = int(pc["window"])
        suffix = f"w{window}-not-smoothed" if window <= 1 else f"w{window}-smoothed"
        filename = f"{plot_filename_tag}_{suffix}.png"
        plot_entries.append((pc, filename))

    new_plot_count = len(plot_entries)

    saved_window_to_filename: dict[int, str] = {}
    for pc, filename in plot_entries:
        actual_path = pc["plot"].save(filename, out_dir=plots_dir)
        saved_window_to_filename[int(pc["window"])] = os.path.basename(actual_path)
        try:
            plt.close(pc["plot"].fig)
        except Exception:
            pass

    combined_created = False
    window_to_filename: dict[int, str] = saved_window_to_filename
    combined_windows = [101, 201]
    if all(w in window_to_filename for w in combined_windows):
        combined_images = []
        combined_missing = False
        for w in combined_windows:
            fpath = os.path.join(plots_dir, window_to_filename[w])
            if not os.path.isfile(fpath):
                combined_missing = True
                break
            combined_images.append(plt.imread(fpath))

        if not combined_missing:
            fig_combined, axes_combined = plt.subplots(1, 2, figsize=(14, 5))
            for ax, w, img in zip(axes_combined, combined_windows, combined_images):
                ax.imshow(img)
                ax.axis("off")
                ax.set_title(f"Smoothing window: {w}", fontsize=10)

            fig_combined.tight_layout()
            combined_filename = f"Twin_{plot_filename_tag}_w101-w201-combined.png"
            combined_path = os.path.join(plots_dir, combined_filename)
            from .Helper_progress_bar import get_unique_filepath
            combined_path = get_unique_filepath(combined_path)
            fig_combined.savefig(combined_path, dpi=300)
            combined_created = True
            print(f"Saved combined 101/201 plot to {combined_path}")

    print(f"Saved {new_plot_count} new plot(s) to {plots_dir}/")

    if animation_plot:
        print("[animation] Animation is unavailable in the trimmed TSP configuration.")
    if curve_plot or animation_plot or combined_created:
        _open_in_file_explorer(plots_dir)
        plt.show(block=show_curve_plots)

    # ── Trial Continuation Analysis: returns summary table (ported feature) ──
    try:
        from .Library_continuation_analysis import build_returns_summary_table

        build_returns_summary_table(
            algo_jobs=algo_jobs,
            algo_job_offsets=algo_job_offsets,
            setting_results=setting_results,
            n_repetitions=n_repetitions,
            last_fraction=0.1,
            output_dir="Trial Continuation Analysis",
            use_saved_disk_networks_checkpoints=use_saved_disk_networks_checkpoints,
        )
    except Exception as exc:
        print(f"[summary] Failed to build returns summary table: {exc}")

    total_time = (time.perf_counter() - start_time) / 60.0
    # Disabled (kept for reference): output.log generation
    # with open("output.log", "w", encoding="utf-8") as f:
    #     f.write(f"Total execution time: {total_time:.3f} minutes\n")
    print(f"\nExperiment finished in {total_time:.3f} minutes.")
    return stochastic_duration_matrix_mean


_CKPT_ALGO_DIR = {"a2c": "A2C", "sac": "SAC", "ppo": "PPO"}


def _build_pg_checkpoint_metadata(
    *,
    component,
    policy_based_method,
    actor_hidden_nn,
    critic_hidden_nn,
    actor_lr,
    critic_lr,
    gamma,
    n_actions,
    max_episode_length,
    eval_interval,
    n_timesteps,
    use_saved_disk_networks_checkpoints,
    gae_lambda=None,
    clip_epsilon=None,
    n_epochs=None,
    entropy_coef=None,
    value_coef=None,
    rollout_steps=None,
    TN_step=None,
    instance_matrices_text=None,
    instance_matrix_hash=None,
):
    """Build the sidecar metadata dict for an actor/critic checkpoint.

    Shared by the worker (which writes it) and the up-front load report (which
    matches against it) so both stay in lock-step. ``n_actions`` is the strict
    field: it encodes the TSP instance size and is what makes a checkpoint
    weight-compatible with the current environment.

    When ``match_training_matrices`` is enabled the caller passes
    ``instance_matrices_text`` / ``instance_matrix_hash`` (computed from the
    environment's duration / inclusion / potential-uncertainty matrices); the
    hash then joins the strict-field gate so a checkpoint trained on a different
    instance is never reused. When matrix matching is off both are ``None`` and
    the metadata is identical to before.
    """
    md = {
        "algo_type": str(policy_based_method).upper(),
        "component": component,
        "actor_hidden_nn": np.asarray(actor_hidden_nn, dtype=np.int32).tolist(),
        "actor_lr": float(actor_lr),
        "gamma": float(gamma),
        "n_actions": int(n_actions),
        "max_episode_length": int(max_episode_length),
        "eval_interval": int(eval_interval),
        "n_timesteps": int(n_timesteps),
        "use_saved_disk_networks_checkpoints": bool(use_saved_disk_networks_checkpoints),
    }
    if instance_matrix_hash is not None:
        md["instance_matrix_hash"] = str(instance_matrix_hash)
        if instance_matrices_text is not None:
            md["instance_matrices_text"] = str(instance_matrices_text)
    if component == "Critic" or str(policy_based_method).lower() in ("a2c", "sac", "ppo"):
        md["critic_hidden_nn"] = np.asarray(critic_hidden_nn, dtype=np.int32).tolist()
        md["critic_lr"] = float(critic_lr)
    if str(policy_based_method).lower() == "ppo":
        md.update({
            "gae_lambda": float(gae_lambda),
            "clip_epsilon": float(clip_epsilon),
            "n_epochs": int(n_epochs),
            "entropy_coef": float(entropy_coef),
            "value_coef": float(value_coef),
            "rollout_steps": int(rollout_steps) if rollout_steps is not None else None,
        })
    else:
        md["TN_step"] = int(TN_step) if TN_step is not None else None
    return md


def _report_checkpoint_loads(
    *,
    pending_settings,
    env,
    n_repetitions,
    n_timesteps,
    eval_interval,
    max_episode_length,
    skip_selection_hyperparameter_match,
    match_training_matrices=True,
):
    """Print which actor checkpoints each (setting, rep) worker will load.

    Returns the number of (setting, rep) workers that resolved to a saved
    checkpoint and will therefore continue from it (0 means every worker trains
    fresh despite reuse being enabled).
    """
    from .Library_checkpointing import (
        tsp_actor_checkpoint_path,
        resolve_continuation_source,
    )
    from .Library_env_elements import env_matrices_text_and_hash

    n_actions = int(getattr(env, "n_actions"))
    if match_training_matrices:
        _instance_text, _instance_hash = env_matrices_text_and_hash(env)
    else:
        _instance_text, _instance_hash = None, None
    resolved_count = 0  # number of (setting, rep) workers that will actually continue from a saved checkpoint
    print("Checkpoint reuse enabled - resolving saved networks to continue from:")
    for sp, (_global_idx, job) in enumerate(pending_settings):
        method = str(job["method"]).lower()
        algo_dir = _CKPT_ALGO_DIR.get(method)
        if algo_dir is None:
            continue
        kw = dict(job["kwargs"])
        suffix = f"S{sp + 1}"
        for r in range(n_repetitions):
            actor_ck = tsp_actor_checkpoint_path(
                algo_type=algo_dir, rep_index=r, checkpoint_suffix=suffix
            )
            actor_md = _build_pg_checkpoint_metadata(
                component="Actor",
                policy_based_method=method,
                actor_hidden_nn=kw["actor_hidden_nn"],
                critic_hidden_nn=kw.get("critic_hidden_nn", np.array([64, 64])),
                actor_lr=kw["actor_lr"],
                critic_lr=kw.get("critic_lr", 0.001),
                gamma=kw["gamma"],
                n_actions=n_actions,
                max_episode_length=max_episode_length,
                eval_interval=eval_interval,
                n_timesteps=n_timesteps,
                use_saved_disk_networks_checkpoints=True,
                gae_lambda=kw.get("gae_lambda"),
                clip_epsilon=kw.get("clip_epsilon"),
                n_epochs=kw.get("n_epochs"),
                entropy_coef=kw.get("entropy_coef"),
                value_coef=kw.get("value_coef"),
                rollout_steps=kw.get("rollout_steps"),
                TN_step=kw.get("TN_step"),
                instance_matrices_text=_instance_text,
                instance_matrix_hash=_instance_hash,
            )
            source = resolve_continuation_source(
                checkpoint_path=actor_ck.file_path,
                metadata=actor_md,
                skip_selection_hyperparameter_match=skip_selection_hyperparameter_match,
            )
            tag = f"{algo_dir} S{sp + 1} rep{r}"
            if source is None:
                print(f"  [{tag}] no match - training fresh")
            elif not source.needs_conversion:
                resolved_count += 1
                print(f"  [{tag}] continue from {os.path.relpath(source.path)}")
            else:
                resolved_count += 1
                print(
                    f"  [{tag}] down-convert from {os.path.relpath(source.path)} "
                    f"{list(source.source_hidden or [])} -> {list(source.target_hidden or [])}"
                )
    print()
    return resolved_count


def _run_single_repetition(
    env,
    policy_based_method,
    actor_hidden_nn,
    critic_hidden_nn=np.array([64, 64]),
    actor_lr=0.001,
    critic_lr=0.001,
    gamma=0.99,
    max_episode_length=500,
    n_timesteps=1000000,
    eval_interval=250,
    n_eval_episodes=1,
    run_seed=42,
    rep_index=0,
    n_repetitions=1,
    enable_progress_bar=True,
    shared_step_counter=None,
    TN_step=10,
    alpha=0.2,
    alpha_lr=0.001,
    auto_tune_alpha=True,
    target_entropy_ratio=0.98,
    tau=0.005,
    gae_lambda=0.95,
    clip_epsilon=0.2,
    n_epochs=4,
    entropy_coef=0.01,
    value_coef=0.5,
    full_episode_updates=True,
    rollout_steps=None,
    checkpoint_suffix=None,
    use_saved_disk_networks_checkpoints=False,
    skip_selection_hyperparameter_match=False,
    match_training_matrices=True,
):
    """Run one TSP training repetition (pickle-safe for ProcessPoolExecutor)."""

    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    torch.manual_seed(run_seed)

    # ── Checkpoint reuse / continuation setup (imported from CartPole fork) ──
    # Resolve the actor/critic checkpoint paths and build their sidecar metadata
    # up front; the nested loader is invoked right after each agent is created so
    # training can *continue* from a matching saved network.
    _ckpt_algo_dir = _CKPT_ALGO_DIR.get(policy_based_method)
    _n_actions = int(getattr(env, "n_actions"))
    _actor_ck = _critic_ck = None
    _actor_md = _critic_md = None
    actor_loaded_path = actor_loaded_timesteps = None
    critic_loaded_path = critic_loaded_timesteps = None
    if _ckpt_algo_dir is not None:
        from .Library_checkpointing import tsp_actor_checkpoint_path, tsp_critic_checkpoint_path

        _actor_ck = tsp_actor_checkpoint_path(
            algo_type=_ckpt_algo_dir, rep_index=rep_index, checkpoint_suffix=checkpoint_suffix
        )
        _critic_ck = tsp_critic_checkpoint_path(
            algo_type=_ckpt_algo_dir, rep_index=rep_index, checkpoint_suffix=checkpoint_suffix
        )
        if match_training_matrices:
            from .Library_env_elements import env_matrices_text_and_hash
            _instance_text, _instance_hash = env_matrices_text_and_hash(env)
        else:
            _instance_text, _instance_hash = None, None
        _md_common = dict(
            policy_based_method=policy_based_method,
            actor_hidden_nn=actor_hidden_nn,
            critic_hidden_nn=critic_hidden_nn,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            gamma=gamma,
            n_actions=_n_actions,
            max_episode_length=max_episode_length,
            eval_interval=eval_interval,
            n_timesteps=n_timesteps,
            use_saved_disk_networks_checkpoints=use_saved_disk_networks_checkpoints,
            gae_lambda=gae_lambda,
            clip_epsilon=clip_epsilon,
            n_epochs=n_epochs,
            entropy_coef=entropy_coef,
            value_coef=value_coef,
            rollout_steps=rollout_steps,
            TN_step=TN_step,
            instance_matrices_text=_instance_text,
            instance_matrix_hash=_instance_hash,
        )
        _actor_md = _build_pg_checkpoint_metadata(component="Actor", **_md_common)
        _critic_md = _build_pg_checkpoint_metadata(component="Critic", **_md_common)

    def _load_into_agent(agent):
        """When reuse is on, load matching saved actor/critic weights into the
        freshly built agent so the training run continues from them."""
        nonlocal actor_loaded_path, actor_loaded_timesteps
        nonlocal critic_loaded_path, critic_loaded_timesteps
        if not use_saved_disk_networks_checkpoints or _ckpt_algo_dir is None:
            return
        from .Library_checkpointing import load_payload_for_continuation

        actor_loaded_path, actor_loaded_timesteps = load_payload_for_continuation(
            model=agent.actor,
            checkpoint_path=_actor_ck.file_path,
            metadata=_actor_md,
            skip_selection_hyperparameter_match=skip_selection_hyperparameter_match,
        )
        critic = getattr(agent, "critic", None)
        if isinstance(critic, torch.nn.Module):
            critic_loaded_path, critic_loaded_timesteps = load_payload_for_continuation(
                model=critic,
                checkpoint_path=_critic_ck.file_path,
                metadata=_critic_md,
                skip_selection_hyperparameter_match=skip_selection_hyperparameter_match,
            )

    if policy_based_method == "a2c":
        from A2C_Agent import A2C_Agent
        from A2C import run_TSP_a2c

        env_rep_A2C = copy.deepcopy(env)
        agent = A2C_Agent(
            env_rep_A2C,
            actor_hidden_nn=actor_hidden_nn,
            critic_hidden_nn=critic_hidden_nn,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            gamma=gamma,
            TN_step=TN_step,
        )
        _load_into_agent(agent)
        rep_returns, rep_timesteps = run_TSP_a2c(
            agent,
            env_rep_A2C,
            n_timesteps=n_timesteps,
            eval_interval=eval_interval,
            truncation_step=max_episode_length,
            enable_progress_bar=enable_progress_bar,
            progress_bar_desc=f"A2C Rep {rep_index + 1}/{n_repetitions}",
            progress_bar_position=rep_index if enable_progress_bar else None,
            shared_step_counter=shared_step_counter,
            eval_n_episodes=n_eval_episodes,
            reseed_noise_seed=run_seed,
            full_episode_updates=full_episode_updates,
        )
    elif policy_based_method == "ppo":
        from PPO_Agent import TSP_PPO_Agent
        from PPO import run_TSP_ppo

        env_rep_PPO = copy.deepcopy(env)
        agent = TSP_PPO_Agent(
            env_rep_PPO,
            actor_hidden_nn=actor_hidden_nn,
            critic_hidden_nn=critic_hidden_nn,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_epsilon=clip_epsilon,
            n_epochs=n_epochs,
            entropy_coef=entropy_coef,
            value_coef=value_coef,
        )
        _load_into_agent(agent)
        rep_returns, rep_timesteps = run_TSP_ppo(
            agent,
            env_rep_PPO,
            n_timesteps=n_timesteps,
            eval_interval=eval_interval,
            truncation_step=max_episode_length,
            enable_progress_bar=enable_progress_bar,
            progress_bar_desc=f"PPO Rep {rep_index + 1}/{n_repetitions}",
            progress_bar_position=rep_index if enable_progress_bar else None,
            shared_step_counter=shared_step_counter,
            eval_n_episodes=n_eval_episodes,
            reseed_noise_seed=run_seed,
            full_episode_updates=full_episode_updates,
            rollout_steps=rollout_steps,
        )
    elif policy_based_method == "sac":
        from SAC_Agent import TSP_SAC_Agent
        from SAC import run_TSP_sac

        env_rep_SAC = copy.deepcopy(env)
        agent = TSP_SAC_Agent(
            env_rep_SAC,
            actor_hidden_nn=actor_hidden_nn,
            critic_hidden_nn=critic_hidden_nn,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            gamma=gamma,
            TN_step=TN_step,
            alpha=alpha,
            alpha_lr=alpha_lr,
            auto_tune_alpha=auto_tune_alpha,
            target_entropy_ratio=target_entropy_ratio,
            tau=tau,
        )
        _load_into_agent(agent)
        rep_returns, rep_timesteps = run_TSP_sac(
            agent,
            env_rep_SAC,
            n_timesteps=n_timesteps,
            eval_interval=eval_interval,
            truncation_step=max_episode_length,
            enable_progress_bar=enable_progress_bar,
            progress_bar_desc=f"SAC Rep {rep_index + 1}/{n_repetitions}",
            progress_bar_position=rep_index if enable_progress_bar else None,
            shared_step_counter=shared_step_counter,
            eval_n_episodes=n_eval_episodes,
            reseed_noise_seed=run_seed,
            full_episode_updates=full_episode_updates,
        )
    else:
        raise ValueError(f"Unknown policy_based_method: {policy_based_method}")

    # ── Persist the trained network(s) ──────────────────────────────────────
    # The payload keeps the project's historical dict shape (state_dict +
    # n_actions + actor_hidden_nn) so Use_Trained_Model can still load it;
    # a JSON metadata sidecar is written alongside. When reuse is enabled and a
    # checkpoint was loaded, training continued from it, so we overwrite that
    # same file and accumulate its timestep counter; otherwise we save in place
    # (reuse off) or as a fresh non-overwriting file (reuse on, no prior match).
    if _ckpt_algo_dir is not None:
        from .Library_checkpointing import save_continuation_or_new, save_payload_in_place

        _n_actions_agent = int(getattr(agent, "n_actions", _n_actions))

        def _payload_for(model, hidden_nn):
            return {
                "state_dict": model.state_dict(),
                "actor_hidden_nn": np.asarray(hidden_nn, dtype=np.int32).tolist(),
                "n_actions": _n_actions_agent,
                "run_seed": int(run_seed),
                "rep_index": int(rep_index),
                "checkpoint_suffix": checkpoint_suffix,
            }

        actor_payload = _payload_for(agent.actor, actor_hidden_nn)
        if use_saved_disk_networks_checkpoints:
            save_continuation_or_new(
                payload=actor_payload,
                checkpoint_path=_actor_ck.file_path,
                metadata=_actor_md,
                loaded_path=actor_loaded_path,
                loaded_timesteps=actor_loaded_timesteps,
                n_timesteps=n_timesteps,
            )
        else:
            save_payload_in_place(
                payload=actor_payload,
                checkpoint_path=_actor_ck.file_path,
                metadata=_actor_md,
            )

        critic = getattr(agent, "critic", None)
        if isinstance(critic, torch.nn.Module):
            critic_payload = _payload_for(critic, critic_hidden_nn)
            critic_payload["critic_hidden_nn"] = np.asarray(critic_hidden_nn, dtype=np.int32).tolist()
            if use_saved_disk_networks_checkpoints:
                save_continuation_or_new(
                    payload=critic_payload,
                    checkpoint_path=_critic_ck.file_path,
                    metadata=_critic_md,
                    loaded_path=critic_loaded_path,
                    loaded_timesteps=critic_loaded_timesteps,
                    n_timesteps=n_timesteps,
                )
            else:
                save_payload_in_place(
                    payload=critic_payload,
                    checkpoint_path=_critic_ck.file_path,
                    metadata=_critic_md,
                )

    return rep_returns, rep_timesteps


################[ Main Execution Block             ]################
if __name__ == "__main__":
    pass
####################################################################
