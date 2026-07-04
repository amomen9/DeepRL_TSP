"""
Library_tsp_test_experiment.py - run the TSP-DRL-Test POMO Attention Model as
a policy-based experiment of the main repo.

When ``included_algorithms['TSP_TEST']`` is enabled (or
``global_config['baseline_model'] == 'TSP_TEST'``) the orchestrator drives the
TSP-DRL-Test Attention Model through the shared parallel pool via three
entrypoints - :func:`prepare_tsp_test_experiment` (disk reuse / top-up
decision), :func:`_run_single_tsp_test_rep` (the pickle-safe per-rep pool
worker) and :func:`finalize_tsp_test_experiment` (curve assembly + save + plot).
It trains the submodule's Attention Model with POMO shared-baseline REINFORCE on
the *same* fixed ``duration_matrix`` instance the RL agents see, under the
main-repo standards (``eval_interval``, ``n_timesteps``, ``max_episode_length``,
``base_seed``), and returns a learning curve in the ``(lc_mean, lc_std,
timesteps, raw_returns)`` shape so it is drawn next to the A2C/PPO/SAC curves and
the Bello benchmark.

Design (mirrors Library_bello_baseline, with one deliberate difference)
-----------------------------------------------------------------------
* **Hyperparameters come from Experiment.py.** Unlike Bello (whose
  hyperparameters are immutable, baked into its submodule), every TSP_TEST
  hyperparameter is read from the ``tsp_test_config`` section defined inside
  ``Experiment.py`` (:data:`TSP_TEST_DEFAULT_HP` only fills omitted keys).
* **Matrix learning.** Each repetition trains on the fixed instance through
  the submodule's matrix API (``set_duration_matrix`` / ``stack_matrix_nodes``
  / ``stack_l_matrix``): node features are the outgoing-row concat
  incoming-column of the normalized matrix, and tours are scored by the
  directed matrix cost - identical data interface to the Bello baseline, so
  the two methods compete on equal information.
* **Curve.** The method's own budget (``steps`` gradient updates) is mapped
  onto the shared env-step x-grid ``[eval_interval, .., n_timesteps]``: at
  grid point ``p`` the model has done ``round(steps * x_p / n_timesteps)``
  updates. The curve value is the (negated) mean sampled tour cost over that
  segment's updates - the same "average distance" convention as the Bello
  baseline, in the RL return convention.
* **Disk reuse.** Results are saved to / loaded from
  ``data sheets/TSP_TEST.xlsx`` with the same rules as the other algorithms
  (project-config + instance-matrix hash + per-sheet hyperparameters), always
  consulted regardless of ``use_existing_disk_data``. The repetition count
  never gates matching: on-disk repetitions are loaded and only the shortfall
  against ``n_repetitions`` is trained (top-up). Per-rep actors are
  checkpointed to ``Checkpoints/TSP/TSP_TEST/`` with an instance-hash sidecar.
* **Isolation.** The submodule is imported as the package ``tsp_drl_test``
  via ``importlib`` (never through ``sys.path``), so its ``config``/``env``/
  ``train`` module names cannot collide with the TSP-DRL_Bello submodule's
  identically named flat modules loaded by Library_bello_baseline.
"""

import importlib.util
import os
import sys

import numpy as np
import torch


TSP_TEST_ALGO = "TSP_TEST"
_TSP_TEST_PKG = "tsp_drl_test"

# Fallbacks for keys omitted from Experiment.py's ``tsp_test_config``. The
# section defined in Experiment.py is the source of truth; these mirror it.
TSP_TEST_DEFAULT_HP = {
    "n_repetitions": 5,
    "steps": 2000,
    "batch": 64,
    "pomo_size": None,       # None -> n_cities (resolved before matching)
    "embed": 128,
    "n_heads": 8,
    "n_layers": 3,
    "ff_hidden": 512,
    "clip_logits": 10.0,
    "softmax_T": 1.0,
    "lr": 3e-4,
    "weight_decay": 1e-6,
    "grad_norm_clip": 1.0,
    "is_lr_decay": False,
    "lr_decay": 0.96,
    "lr_decay_step": 500,
    "curve_label": "TSP-Test (POMO-AM)",
}

# Cosmetic / bookkeeping keys that must not participate in disk matching.
_NON_MATCHING_KEYS = {"curve_label", "n_repetitions"}


def _tsp_test_repo_dir() -> str:
    """Absolute path to the TSP-DRL-Test submodule (sibling of ``Library/``)."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "TSP-DRL-Test")


def _import_tsp_test():
    """Load the TSP-DRL-Test repo as the package ``tsp_drl_test``.

    Uses an explicit ``importlib`` spec (no ``sys.path`` mutation): the Bello
    submodule occupies the flat module names ``config``/``env``/``train`` when
    it runs in the same process, so this submodule must live under a package.
    """
    pkg = sys.modules.get(_TSP_TEST_PKG)
    if pkg is not None:
        return pkg
    repo = _tsp_test_repo_dir()
    init_path = os.path.join(repo, "__init__.py")
    spec = importlib.util.spec_from_file_location(
        _TSP_TEST_PKG, init_path, submodule_search_locations=[repo]
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build an import spec for '{init_path}'.")
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[_TSP_TEST_PKG] = pkg
    try:
        spec.loader.exec_module(pkg)
    except Exception:
        sys.modules.pop(_TSP_TEST_PKG, None)
        raise
    return pkg


def _resolve_hp(algo_config: dict | None, n_cities: int) -> tuple[dict, str, int]:
    """Merge Experiment.py's ``tsp_test_config`` over the defaults.

    Returns ``(hyperparams, curve_label, n_repetitions)`` with ``pomo_size``
    resolved to a concrete int (matching must never key on ``None``) and the
    non-matching keys stripped out of ``hyperparams``.
    """
    merged = dict(TSP_TEST_DEFAULT_HP)
    for key, value in (algo_config or {}).items():
        if key not in TSP_TEST_DEFAULT_HP:
            raise KeyError(
                f"Unknown tsp_test_config key '{key}'. "
                f"Valid keys: {sorted(TSP_TEST_DEFAULT_HP)}."
            )
        merged[key] = value

    curve_label = str(merged["curve_label"])
    n_repetitions = int(merged["n_repetitions"])

    pomo_size = merged["pomo_size"]
    pomo_size = int(n_cities) if pomo_size is None else min(int(pomo_size), int(n_cities))

    hp = {
        "steps": int(merged["steps"]),
        "batch": int(merged["batch"]),
        "pomo_size": pomo_size,
        "embed": int(merged["embed"]),
        "n_heads": int(merged["n_heads"]),
        "n_layers": int(merged["n_layers"]),
        "ff_hidden": int(merged["ff_hidden"]),
        "clip_logits": float(merged["clip_logits"]),
        "softmax_T": float(merged["softmax_T"]),
        "lr": float(merged["lr"]),
        "weight_decay": float(merged["weight_decay"]),
        "grad_norm_clip": float(merged["grad_norm_clip"]),
        "is_lr_decay": bool(merged["is_lr_decay"]),
        "lr_decay": float(merged["lr_decay"]),
        "lr_decay_step": int(merged["lr_decay_step"]),
    }
    return hp, curve_label, n_repetitions


def _build_cfg(pkg, hp: dict, n_cities: int):
    """Construct a TSP-DRL-Test ``Config`` for the fixed instance."""
    return pkg.config.Config(
        mode="train",
        city_t=int(n_cities),
        input_dim=2 * int(n_cities),  # outgoing row + incoming column per city
        batch=hp["batch"],
        steps=hp["steps"],
        embed=hp["embed"],
        n_heads=hp["n_heads"],
        n_layers=hp["n_layers"],
        ff_hidden=hp["ff_hidden"],
        clip_logits=hp["clip_logits"],
        softmax_T=hp["softmax_T"],
        pomo_size=hp["pomo_size"],
        decode_type="sampling",
        lr=hp["lr"],
        weight_decay=hp["weight_decay"],
        grad_norm_clip=hp["grad_norm_clip"],
        is_lr_decay=hp["is_lr_decay"],
        lr_decay=hp["lr_decay"],
        lr_decay_step=hp["lr_decay_step"],
        issaver=False,  # checkpointing is handled by the main repo's machinery
    )


def _hyperparams_for_disk(*, hp, n_timesteps, eval_interval, max_episode_length):
    """The hyperparameter columns used to *match* an existing TSP_TEST.xlsx:
    the main-repo standards plus every Experiment.py hyperparameter. The
    repetition count is deliberately absent (matching never keys on it)."""
    match_hp = {
        "n_timesteps": int(n_timesteps),
        "eval_interval": int(eval_interval),
        "max_episode_length": int(max_episode_length),
    }
    match_hp.update(hp)
    return match_hp


def _grid_timesteps(n_timesteps: int, eval_interval: int) -> np.ndarray:
    """Shared env-step x-grid used by every learning curve in the project."""
    P = max(1, int(n_timesteps) // int(eval_interval))
    return (np.arange(1, P + 1, dtype=np.int64) * int(eval_interval)).astype(np.int32)


def _train_one_rep(*, pkg, cfg, hp, duration_matrix, timesteps_grid, n_timesteps, seed,
                   device, rep_index, n_repetitions, shared_step_counter=None):
    """Train one POMO Attention Model on the fixed instance; return its curve.

    Returns ``(curve, best_tour, best_cost, model)`` where ``curve`` is the
    per-grid-point return (negated mean sampled tour cost) and ``best_tour``
    is the lowest directed-cost tour sampled during training."""
    import torch.optim as optim

    torch.manual_seed(int(seed))

    tenv = pkg.env.TspEnv(cfg)
    tenv.set_duration_matrix(duration_matrix)

    model = pkg.model.AttentionModel(cfg).to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay)
    )
    scheduler = None
    if cfg.is_lr_decay:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=int(cfg.lr_decay_step), gamma=float(cfg.lr_decay)
        )

    # The fixed instance, repeated into a training batch; POMO multistart plus
    # sampling decode supply the per-update tour diversity. The shared baseline
    # is each copy's own rollout-group mean (no critic network).
    inputs = tenv.stack_matrix_nodes(int(cfg.batch), device=device)

    P = int(len(timesteps_grid))
    total_steps = int(cfg.steps)
    targets = [int(round(total_steps * (p + 1) / P)) for p in range(P)]

    # Drive the shared step counter on the env-step scale so the parent renders
    # the same PPO-style bar for this method (grad steps -> [0, n_timesteps]).
    n_ts = int(n_timesteps)
    _progress_stride = max(1, n_ts // 1000)
    _last_pushed = 0

    def _push_progress(done_steps, grid_p=None):
        nonlocal _last_pushed
        if shared_step_counter is None:
            return
        if total_steps > 0:
            val = int(round(done_steps / total_steps * n_ts))
        elif grid_p is not None:
            val = int(round(grid_p / P * n_ts))
        else:
            val = 0
        if val > n_ts:
            val = n_ts
        if val - _last_pushed >= _progress_stride or (val >= n_ts and _last_pushed < n_ts):
            shared_step_counter.value = val
            _last_pushed = val

    curve = np.empty(P, dtype=np.float32)
    best_cost = float("inf")
    best_tour = None

    def _track_best(costs, tours):
        nonlocal best_cost, best_tour
        flat_costs = costs.reshape(-1)
        min_idx = int(flat_costs.argmin().item())
        min_cost = float(flat_costs[min_idx].item())
        if min_cost < best_cost:
            best_cost = min_cost
            best_tour = [int(c) for c in tours.reshape(-1, tours.size(-1))[min_idx].tolist()]

    done = 0
    for p in range(P):
        seg_costs = []
        while done < targets[p]:
            costs, tours = pkg.train.pomo_step(
                model, optimizer, inputs, tenv.stack_l_matrix,
                pomo_size=int(cfg.pomo_size) if cfg.pomo_size is not None else None,
                grad_norm_clip=cfg.grad_norm_clip,
                scheduler=scheduler,
            )
            seg_costs.append(float(costs.mean().item()))
            _track_best(costs, tours)
            done += 1
            _push_progress(done)

        if seg_costs:
            mean_cost = float(np.mean(seg_costs))
        else:
            # total_steps < P: no training this segment, sample the current model.
            with torch.no_grad():
                tours, _ = model.rollout(inputs, decode_type="sampling")
                costs = tenv.stack_l_matrix(tours)
                mean_cost = float(costs.mean().item())
                _track_best(costs, tours)
        curve[p] = -mean_cost  # RL return convention: higher (less negative) is better
        _push_progress(done, grid_p=p + 1)

    if shared_step_counter is not None:
        shared_step_counter.value = n_ts
    return curve, best_tour, best_cost, model


def _save_tsp_test_checkpoint(*, model, hp, n_cities, rep_index, metadata_base,
                              instance_hash, instance_text):
    """Persist one rep's actor with an instance-hash sidecar (same matching
    rules as the other algorithms)."""
    from .Library_checkpointing import tsp_actor_checkpoint_path, save_payload_in_place

    ck = tsp_actor_checkpoint_path(algo_type=TSP_TEST_ALGO, rep_index=rep_index)
    ck.ensure_dir()
    payload = {
        "state_dict": model.state_dict(),
        "n_actions": int(n_cities),
        "input_dim": int(2 * n_cities),
        "rep_index": int(rep_index),
        "tsp_test_hp": dict(hp),
    }
    metadata = dict(metadata_base)
    if instance_hash is not None:
        metadata["instance_matrix_hash"] = instance_hash
        if instance_text is not None:
            metadata["instance_matrices_text"] = instance_text
    save_payload_in_place(payload=payload, checkpoint_path=ck.file_path, metadata=metadata)
    return ck.file_path


def prepare_tsp_test_experiment(
    *,
    env,
    global_config,
    algo_config=None,
    n_timesteps,
    eval_interval,
    max_episode_length,
    base_seed,
    data_sheets_dir,
    formatted_sheets=False,
    match_training_matrices=True,
):
    """Main-process disk-load + top-up decision for the TSP-DRL-Test experiment.

    The old monolithic ``run_tsp_test_experiment`` is split into three so TSP_TEST
    can run inside the shared parallel pool: this ``prepare`` step (disk reuse,
    main process), the pickle-safe per-rep worker
    :func:`_run_single_tsp_test_rep` (pool), and
    :func:`finalize_tsp_test_experiment` (curve assembly + save + plot, main
    process).

    The target repetition count comes from ``tsp_test_config['n_repetitions']``
    (decoupled from the project-wide ``n_repetitions``, like the Bello benchmark).
    Disk reuse never keys on the repetition count: every repetition already in
    TSP_TEST.xlsx is loaded, and only the shortfall ``target - on_disk`` is
    trained. A matching workbook is always consulted regardless of the global
    ``use_existing_disk_data`` flag.

    Returns a plan dict:
      ``{"status": "ready", "result": (...), "curve_label": str}``
          nothing to train (fully covered by disk).
      ``{"status": "train", "n_to_train": int, "curve_label": str, ...}``
          train ``n_to_train`` reps in the pool, then call
          :func:`finalize_tsp_test_experiment`.
    """
    from .Helper_excel import _load_results_from_excel
    from .Library_env_elements import env_matrices_text_and_hash

    duration_matrix = np.asarray(getattr(env, "duration_matrix"), dtype=float)
    n_cities = int(duration_matrix.shape[0])
    hp, curve_label, target_reps = _resolve_hp(algo_config, n_cities)
    timesteps_grid = _grid_timesteps(n_timesteps, eval_interval)
    grid_len = int(len(timesteps_grid))

    match_hp = _hyperparams_for_disk(
        hp=hp,
        n_timesteps=n_timesteps,
        eval_interval=eval_interval,
        max_episode_length=max_episode_length,
    )
    base_filename = TSP_TEST_ALGO
    excel_path = os.path.join(data_sheets_dir, f"{base_filename}.xlsx")

    # ── Load whatever repetitions already exist on disk ──────────────────────
    disk_raw = None  # (D, grid_len) per-repetition curves recovered from disk
    if os.path.isfile(excel_path):
        try:
            results, _mismatches = _load_results_from_excel(
                excel_path, match_hp, global_config=global_config, formatted_sheets=formatted_sheets
            )
        except Exception as exc:
            print(f"[{TSP_TEST_ALGO}] Existing workbook unreadable; retraining. Reason: {exc}")
            results = []
        if results:
            entry = results[0]
            raw = entry.get("raw_returns")
            raw_arr = np.asarray(raw, dtype=np.float32) if raw is not None else None
            if raw_arr is not None and raw_arr.ndim == 2 and raw_arr.shape[1] == grid_len:
                disk_raw = raw_arr
            else:
                print(
                    f"[{TSP_TEST_ALGO}] Disk per-rep grid differs from the current "
                    "standards; retraining from scratch."
                )
        else:
            print(f"[{TSP_TEST_ALGO}] No matching results on disk (different instance/"
                  "standards/hyperparameters). Training TSP-DRL-Test.")
    else:
        print(f"[{TSP_TEST_ALGO}] No saved workbook found. Training TSP-DRL-Test.")

    disk_reps = 0 if disk_raw is None else int(disk_raw.shape[0])
    n_to_train = max(0, target_reps - disk_reps)

    # ── Disk already has at least the target: load all of them, train nothing ─
    if n_to_train == 0 and disk_reps > 0:
        raw_returns = disk_raw
        lc_mean = raw_returns.mean(axis=0).astype(np.float32)
        lc_std = (
            raw_returns.std(axis=0, ddof=1).astype(np.float32)
            if raw_returns.shape[0] > 1 else np.zeros_like(lc_mean)
        )
        print(
            f"[{TSP_TEST_ALGO}] Loaded all {disk_reps} repetition(s) from {excel_path} "
            f"(>= target {target_reps}); no training needed."
        )
        return {
            "status": "ready",
            "result": (lc_mean, lc_std, timesteps_grid, raw_returns),
            "curve_label": curve_label,
        }

    if disk_reps > 0:
        print(
            f"[{TSP_TEST_ALGO}] Found {disk_reps} repetition(s) on disk; topping up to "
            f"the target of {target_reps} by training {n_to_train} more."
        )
    print(
        f"[{TSP_TEST_ALGO}] POMO Attention Model on the {n_cities}-city instance: "
        f"{n_to_train} new rep(s) queued to the shared pool "
        f"({hp['steps']} grad steps/rep, {hp['batch']}x{hp['pomo_size']} trajectories/step)."
    )

    instance_text, instance_hash = (None, None)
    if match_training_matrices:
        instance_text, instance_hash = env_matrices_text_and_hash(env)
    metadata_base = {
        "algo_type": TSP_TEST_ALGO,
        "component": "Actor",
        "n_actions": n_cities,
        "input_dim": 2 * n_cities,
        "n_timesteps": int(n_timesteps),
        "eval_interval": int(eval_interval),
        **{k: hp[k] for k in ("steps", "batch", "pomo_size", "embed", "n_heads",
                              "n_layers", "lr", "clip_logits")},
    }

    # Pre-create the checkpoint dir once in the main process so concurrent workers
    # don't race on it.
    try:
        from .Library_checkpointing import tsp_actor_checkpoint_path
        tsp_actor_checkpoint_path(algo_type=TSP_TEST_ALGO, rep_index=0).ensure_dir()
    except Exception:
        pass

    return {
        "status": "train",
        "n_to_train": int(n_to_train),
        "disk_reps": int(disk_reps),
        "target_reps": int(target_reps),
        "disk_raw": disk_raw,
        "duration_matrix": duration_matrix,
        "n_cities": int(n_cities),
        "hp": hp,
        "curve_label": curve_label,
        "timesteps_grid": timesteps_grid,
        "n_timesteps": int(n_timesteps),
        "eval_interval": int(eval_interval),
        "metadata_base": metadata_base,
        "instance_hash": instance_hash,
        "instance_text": instance_text,
        "match_hp": match_hp,
        "excel_path": excel_path,
        "base_filename": base_filename,
    }


def _run_single_tsp_test_rep(
    *,
    duration_matrix,
    n_cities,
    hp,
    n_timesteps,
    timesteps_grid,
    seed,
    rep_index,
    n_repetitions,
    metadata_base,
    instance_hash,
    instance_text,
    shared_step_counter=None,
):
    """Train one TSP-DRL-Test repetition in a worker process (pickle-safe).

    Mirrors the A2C/SAC/PPO per-rep worker: pins to a single CPU thread, trains on
    the fixed instance, drives ``shared_step_counter`` so the parent renders a
    PPO-style bar, saves its own checkpoint, and returns
    ``(curve, best_tour, best_cost)``.
    """
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    pkg = _import_tsp_test()
    cfg = _build_cfg(pkg, hp, int(n_cities))
    device = torch.device("cpu")

    curve, best_tour, best_cost, model = _train_one_rep(
        pkg=pkg,
        cfg=cfg,
        hp=hp,
        duration_matrix=np.asarray(duration_matrix, dtype=float),
        timesteps_grid=np.asarray(timesteps_grid),
        n_timesteps=int(n_timesteps),
        seed=int(seed),
        device=device,
        rep_index=int(rep_index),
        n_repetitions=int(n_repetitions),
        shared_step_counter=shared_step_counter,
    )

    try:
        _save_tsp_test_checkpoint(
            model=model, hp=hp, n_cities=int(n_cities), rep_index=int(rep_index),
            metadata_base=metadata_base, instance_hash=instance_hash,
            instance_text=instance_text,
        )
    except Exception as exc:
        print(f"[{TSP_TEST_ALGO}] Could not save rep {rep_index} checkpoint: {exc}")

    if shared_step_counter is not None:
        shared_step_counter.value = int(n_timesteps)
    return curve, best_tour, best_cost


def finalize_tsp_test_experiment(
    plan,
    new_rep_results,
    *,
    base_seed,
    data_sheets_dir,
    plots_dir,
    global_config,
    format_sheets=False,
):
    """Combine disk + freshly trained TSP-DRL-Test reps into the method's curve.

    ``new_rep_results`` is the list of ``(curve, best_tour, best_cost)`` tuples
    returned by :func:`_run_single_tsp_test_rep` for this run's reps, in rep
    order. Assembles the mean/std curve, saves the combined workbook, draws the
    reconstructed-route plot, and returns ``(result_tuple, curve_label)``.
    """
    from .Helper_excel import save_algorithm_workbook

    disk_raw = plan["disk_raw"]
    disk_reps = int(plan["disk_reps"])
    target_reps = int(plan["target_reps"])
    timesteps_grid = plan["timesteps_grid"]
    duration_matrix = plan["duration_matrix"]
    excel_path = plan["excel_path"]
    base_filename = plan["base_filename"]
    match_hp = plan["match_hp"]
    curve_label = plan["curve_label"]

    new_curves = []
    overall_best_cost = float("inf")
    overall_best_tour = None
    for i, (curve, best_tour, best_cost) in enumerate(new_rep_results):
        new_curves.append(np.asarray(curve, dtype=np.float32))
        rep_index = disk_reps + i
        print(f"  [{TSP_TEST_ALGO}] rep {rep_index + 1}/{target_reps}: best tour cost {best_cost:.4f}")
        if best_tour is not None and best_cost < overall_best_cost:
            overall_best_cost = best_cost
            overall_best_tour = best_tour

    new_raw = np.asarray(new_curves, dtype=np.float32)                # (n_to_train, grid_len)
    if disk_raw is not None:
        raw_returns = np.concatenate([disk_raw, new_raw], axis=0)     # (target, grid_len)
    else:
        raw_returns = new_raw
    lc_mean = raw_returns.mean(axis=0).astype(np.float32)
    lc_std = (
        raw_returns.std(axis=0, ddof=1).astype(np.float32)
        if raw_returns.shape[0] > 1
        else np.zeros_like(lc_mean)
    )
    result_tuple = (lc_mean, lc_std, timesteps_grid, raw_returns)

    # ── Save the combined (disk + new) repetitions to disk ────────────────────
    # The disk repetitions were folded into ``raw_returns`` above, so the fresh
    # workbook is a strict superset; remove the stale single-setting file first
    # so the rep-count update replaces it instead of appending a duplicate.
    if os.path.isfile(excel_path):
        try:
            os.remove(excel_path)
        except OSError as exc:
            print(f"[{TSP_TEST_ALGO}] Could not remove stale workbook before re-saving: {exc}")
    save_hp = dict(match_hp)
    save_hp["n_repetitions"] = int(raw_returns.shape[0])
    tsp_test_job = {
        "curve_label": curve_label,
        "method": "tsp_test",
        "hyperparams": save_hp,
    }
    try:
        save_algorithm_workbook(
            data_sheets_dir, base_filename, TSP_TEST_ALGO, [tsp_test_job], [result_tuple],
            global_config=global_config, algo_config=save_hp, format_sheets=format_sheets,
        )
    except Exception as exc:
        print(f"[{TSP_TEST_ALGO}] Failed to save workbook: {exc}")

    # ── Post-training plot on reconstructed coordinates ───────────────────────
    # ``overall_best_tour`` covers the newly trained repetitions only (the disk
    # repetitions store curves, not tours), so the plot is drawn only when we
    # actually trained at least one repetition.
    if overall_best_tour is not None:
        try:
            from .Library_bello_plot import plot_route_on_reconstructed_coords

            out_path = plot_route_on_reconstructed_coords(
                duration_matrix=duration_matrix,
                tour=overall_best_tour,
                out_dir=plots_dir,
                filename=f"{TSP_TEST_ALGO}_route_reconstructed.png",
                seed=int(base_seed),
                title=(
                    f"TSP-Test (POMO-AM) best tour (cost {overall_best_cost:.3f}) "
                    "on reconstructed coordinates"
                ),
            )
            print(f"[{TSP_TEST_ALGO}] Saved reconstructed-coordinate route plot to {out_path}")
        except Exception as exc:
            print(f"[{TSP_TEST_ALGO}] Could not draw reconstructed-coordinate plot: {exc}")

    return result_tuple, curve_label
