"""
Library_bello_baseline.py - run the TSP-DRL_Bello Pointer Network as a baseline.

When ``global_config['baseline_model'] == 'Bello'`` the orchestrator drives the
(re-architected) Bello actor/critic through the shared parallel pool via three
entrypoints - :func:`prepare_bello_baseline` (disk reuse / top-up decision),
:func:`_run_single_bello_rep` (the pickle-safe per-rep pool worker) and
:func:`finalize_bello_baseline` (curve assembly + save + plot). It trains on the
*same* fixed ``duration_matrix`` instance the RL agents see, under the main-repo
standards (``eval_interval``, ``n_timesteps``, ``n_repetitions``,
``max_episode_length``, ``base_seed``), and returns a learning curve in the same
``(lc_mean, lc_std, timesteps, raw_returns)`` shape so it can be drawn as the
benchmark curve.

Design (matches the agreed plan)
--------------------------------
* **Immutable hyperparameters.** Bello's hyperparameters are exactly the
  defaults baked into the submodule's ``config.py`` (mirrored in
  :data:`BELLO_IMMUTABLE_HP`); only the *data* (the instance matrix) and the
  per-city ``input_dim`` (``2*n``) change.
* **Matrix learning.** Each repetition trains on the fixed instance using the
  submodule's matrix API (``set_duration_matrix`` / ``stack_matrix_nodes`` /
  ``stack_l_matrix``): a directed tour cost on ``duration_matrix``. For a
  symmetric matrix this reduces to the original Euclidean objective.
* **Curve.** Bello's own training budget (``steps`` gradient updates) is mapped
  onto the shared env-step x-grid ``[eval_interval, .., n_timesteps]``: at grid
  point ``p`` the model has done ``round(steps * x_p / n_timesteps)`` updates.
  The curve value is the (negated) mean tour cost over that segment's batches -
  i.e. Bello's reported "average distance", in the RL return convention.
* **Disk reuse.** Results are saved to / loaded from ``data sheets/BELLO.xlsx``
  with the *same* rules as the other algorithms (project-config + instance-matrix
  hash + per-sheet hyperparameters). ``use_existing_disk_data`` is *forced True*
  for Bello: a matching workbook short-circuits retraining. Per-rep actors are
  checkpointed to ``Checkpoints/TSP/BELLO/`` with an instance-hash sidecar.
* **Plot.** After training, the best tour found is plotted on 2-D coordinates
  reconstructed from the instance via the route-preserving symmetrisation +
  classical MDS embedding in :mod:`Library_bello_plot`.
"""

import os
import sys

import numpy as np
import torch


# Bello hyperparameters, mirroring the submodule ``config.py`` argparser
# defaults. These are immutable: the baseline must run with the standards set
# inside the Bello repo, not the main project's RL hyperparameters.
BELLO_IMMUTABLE_HP = {
    "batch": 512,
    "steps": 15000,
    "embed": 128,
    "hidden": 128,
    "clip_logits": 10,
    "softmax_T": 1.0,
    "optim": "Adam",
    "init_min": -0.08,
    "init_max": 0.08,
    "n_glimpse": 1,
    "n_process": 3,
    "decode_type": "sampling",
    "lr": 1e-3,
    "is_lr_decay": True,
    "lr_decay": 0.96,
    "lr_decay_step": 5000,
}

_BELLO_ALGO = "BELLO"

# Fixed benchmark repetition count. The Bello baseline (the benchmark curve) is
# always produced from this many repetitions, independent of the project-wide
# ``global_config['n_repetitions']`` that drives the RL algorithms. Disk matching
# never keys on the repetition count: every repetition stored in BELLO.xlsx is
# loaded, and only the shortfall ``target - disk`` is trained to top it up.
BELLO_N_REPETITIONS = 5


def _bello_repo_dir() -> str:
    """Absolute path to the TSP-DRL_Bello submodule (sibling of ``Library/``)."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "TSP-DRL_Bello")


def _import_bello():
    """Import the submodule's ``config`` / ``actor`` / ``critic`` / ``env``.

    The submodule uses bare intra-package imports (``from config import ...``),
    so its directory must be on ``sys.path``. The module names don't collide
    with the main project's modules.
    """
    repo = _bello_repo_dir()
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import config as bello_config  # type: ignore
    import actor as bello_actor    # type: ignore
    import critic as bello_critic  # type: ignore
    import env as bello_env        # type: ignore
    return bello_config, bello_actor, bello_critic, bello_env


def _build_bello_cfg(bello_config, n_cities: int):
    """Construct a Bello ``Config`` for the fixed instance with immutable HP."""
    model_dir = os.path.join("Checkpoints", "TSP", _BELLO_ALGO) + os.sep
    kwargs = dict(BELLO_IMMUTABLE_HP)
    kwargs.update(
        mode="train",
        city_t=int(n_cities),
        input_dim=2 * int(n_cities),  # outgoing row + incoming column per city
        # dirs are created by Config.__init__; point the unused logger/pkl dirs
        # at the checkpoint dir so no stray ./Csv ./Pkl folders are created.
        log_dir=model_dir,
        model_dir=model_dir,
        pkl_dir=model_dir,
        cuda_dv="0",
        islogger=False,
        issaver=False,
        log_step=10,
        seed=1,
        alpha=0.99,
        act_model_path=None,
    )
    return bello_config.Config(**kwargs)


def _bello_hyperparams_for_disk(*, n_timesteps, eval_interval, max_episode_length):
    """The hyperparameter columns used to *match* an existing BELLO.xlsx.
    Combines the main-repo standards with Bello's immutable hyperparameters.

    The repetition count is deliberately absent: matching never keys on how many
    repetitions are stored. The actual stored count is written separately as an
    ``n_repetitions`` column at save time (informational, and used only to build
    the ``rep_*`` columns)."""
    hp = {
        "n_timesteps": int(n_timesteps),
        "eval_interval": int(eval_interval),
        "max_episode_length": int(max_episode_length),
    }
    for key, value in BELLO_IMMUTABLE_HP.items():
        hp[key] = value
    return hp


def _grid_timesteps(n_timesteps: int, eval_interval: int) -> np.ndarray:
    """Shared env-step x-grid used by every learning curve in the project."""
    P = max(1, int(n_timesteps) // int(eval_interval))
    return (np.arange(1, P + 1, dtype=np.int64) * int(eval_interval)).astype(np.int32)


def _train_one_rep(*, bello_actor, bello_critic, bello_env, cfg, duration_matrix,
                   timesteps_grid, n_timesteps, seed, device, rep_index, n_repetitions,
                   shared_step_counter=None):
    """Train one Bello actor/critic on the fixed instance and return its curve.

    Returns ``(curve, best_tour, best_cost, actor)`` where ``curve`` is the
    per-grid-point return (negated mean tour cost), ``best_tour`` is the lowest
    directed-cost tour seen during training (on the un-normalised matrix)."""
    import torch.nn as nn
    import torch.optim as optim

    torch.manual_seed(int(seed))

    benv = bello_env.Env_tsp(cfg)
    benv.set_duration_matrix(duration_matrix)

    act = bello_actor.PtrNet1(cfg).to(device)
    cri = bello_critic.PtrNet2(cfg).to(device)
    act_opt = optim.Adam(act.parameters(), lr=cfg.lr)
    cri_opt = optim.Adam(cri.parameters(), lr=cfg.lr)
    act_sched = cri_sched = None
    if cfg.is_lr_decay:
        act_sched = optim.lr_scheduler.StepLR(act_opt, step_size=int(cfg.lr_decay_step), gamma=cfg.lr_decay)
        cri_sched = optim.lr_scheduler.StepLR(cri_opt, step_size=int(cfg.lr_decay_step), gamma=cfg.lr_decay)
    mse = nn.MSELoss()

    # The fixed instance, repeated into a training batch. Sampling (decode_type)
    # supplies the per-step tour diversity that drives the policy gradient.
    inputs = benv.stack_matrix_nodes(int(cfg.batch)).to(device)

    P = int(len(timesteps_grid))
    total_steps = int(cfg.steps)
    targets = [int(round(total_steps * (p + 1) / P)) for p in range(P)]

    # Drive the shared step counter on the env-step scale so the parent process
    # renders the same PPO-style bar for this method: map completed grad steps
    # (done/total_steps) onto [0, n_timesteps], throttled like the PG loops.
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
    done = 0
    for p in range(P):
        seg_costs = []
        while done < targets[p]:
            pred_tour, ll = act(inputs, device)
            real_l = benv.stack_l_matrix(pred_tour)            # (batch,) directed cost on D
            pred_l = cri(inputs, device)                       # (batch,) state-value
            cri_loss = mse(pred_l, real_l.detach())
            cri_opt.zero_grad(); cri_loss.backward()
            nn.utils.clip_grad_norm_(cri.parameters(), max_norm=1.0, norm_type=2)
            cri_opt.step()
            adv = real_l.detach() - pred_l.detach()
            act_loss = (adv * ll).mean()
            act_opt.zero_grad(); act_loss.backward()
            nn.utils.clip_grad_norm_(act.parameters(), max_norm=1.0, norm_type=2)
            act_opt.step()
            if cfg.is_lr_decay:
                act_sched.step(); cri_sched.step()

            seg_costs.append(float(real_l.mean().item()))
            step_min_idx = int(real_l.argmin().item())
            step_min = float(real_l[step_min_idx].item())
            if step_min < best_cost:
                best_cost = step_min
                best_tour = [int(c) for c in pred_tour[step_min_idx].tolist()]
            done += 1
            _push_progress(done)

        if seg_costs:
            mean_cost = float(np.mean(seg_costs))
        else:
            # total_steps < P: no training this segment, sample the current model.
            with torch.no_grad():
                pt, _ = act(inputs, device)
                rl = benv.stack_l_matrix(pt)
                mean_cost = float(rl.mean().item())
                mi = int(rl.argmin().item())
                if float(rl[mi].item()) < best_cost:
                    best_cost = float(rl[mi].item())
                    best_tour = [int(c) for c in pt[mi].tolist()]
        curve[p] = -mean_cost  # RL return convention: higher (less negative) is better
        _push_progress(done, grid_p=p + 1)

    if shared_step_counter is not None:
        shared_step_counter.value = n_ts
    return curve, best_tour, best_cost, act


def _save_bello_checkpoint(*, actor, cfg, n_cities, rep_index, metadata_base, instance_hash, instance_text):
    """Persist one rep's actor with an instance-hash sidecar (same matching
    rules as the other algorithms)."""
    from .Library_checkpointing import tsp_actor_checkpoint_path, save_payload_in_place

    ck = tsp_actor_checkpoint_path(algo_type=_BELLO_ALGO, rep_index=rep_index)
    ck.ensure_dir()
    payload = {
        "state_dict": actor.state_dict(),
        "n_actions": int(n_cities),
        "input_dim": int(2 * n_cities),
        "rep_index": int(rep_index),
        "bello_cfg": {k: v for k, v in vars(cfg).items() if isinstance(v, (int, float, str, bool))},
    }
    metadata = dict(metadata_base)
    if instance_hash is not None:
        metadata["instance_matrix_hash"] = instance_hash
        if instance_text is not None:
            metadata["instance_matrices_text"] = instance_text
    save_payload_in_place(payload=payload, checkpoint_path=ck.file_path, metadata=metadata)
    return ck.file_path


def prepare_bello_baseline(
    *,
    env,
    global_config,
    n_timesteps,
    eval_interval,
    max_episode_length,
    base_seed,
    data_sheets_dir,
    formatted_sheets=False,
    match_training_matrices=True,
):
    """Main-process disk-load + top-up decision for the Bello benchmark.

    The old monolithic ``run_bello_baseline`` is split into three so Bello can run
    inside the shared parallel pool: this ``prepare`` step (disk reuse, main
    process), the pickle-safe per-rep worker :func:`_run_single_bello_rep` (pool),
    and :func:`finalize_bello_baseline` (curve assembly + save + plot, main
    process).

    The benchmark always targets :data:`BELLO_N_REPETITIONS` repetitions,
    independent of the project-wide ``n_repetitions``. Disk reuse never keys on the
    repetition count: every repetition already in BELLO.xlsx is loaded, and only
    the shortfall ``target - on_disk`` is trained. ``use_existing_disk_data`` is
    forced True for Bello (a matching workbook is always consulted).

    Returns a plan dict:
      ``{"status": "ready", "result": (lc_mean, lc_std, timesteps, raw_or_None)}``
          nothing to train (fully covered by disk / legacy workbook).
      ``{"status": "train", "n_to_train": int, ...}``
          train ``n_to_train`` reps in the pool, then call
          :func:`finalize_bello_baseline`.
    """
    from .Helper_excel import _load_results_from_excel
    from .Library_env_elements import env_matrices_text_and_hash

    duration_matrix = np.asarray(getattr(env, "duration_matrix"), dtype=float)
    n_cities = int(duration_matrix.shape[0])
    timesteps_grid = _grid_timesteps(n_timesteps, eval_interval)
    grid_len = int(len(timesteps_grid))
    target_reps = int(BELLO_N_REPETITIONS)

    # Matching hyperparameters deliberately omit the repetition count.
    match_hp = _bello_hyperparams_for_disk(
        n_timesteps=n_timesteps,
        eval_interval=eval_interval,
        max_episode_length=max_episode_length,
    )
    base_filename = _BELLO_ALGO
    excel_path = os.path.join(data_sheets_dir, f"{base_filename}.xlsx")

    # ── Load whatever repetitions already exist on disk ─────────────────────────
    disk_raw = None  # (D, grid_len) per-repetition curves recovered from disk
    if os.path.isfile(excel_path):
        try:
            results, _mismatches = _load_results_from_excel(
                excel_path, match_hp, global_config=global_config, formatted_sheets=formatted_sheets
            )
        except Exception as exc:
            print(f"[BELLO] Existing workbook unreadable; retraining. Reason: {exc}")
            results = []
        if results:
            entry = results[0]
            raw = entry.get("raw_returns")
            raw_arr = np.asarray(raw, dtype=np.float32) if raw is not None else None
            if raw_arr is not None and raw_arr.ndim == 2 and raw_arr.shape[1] == grid_len:
                disk_raw = raw_arr
            elif raw_arr is None:
                # Legacy workbook without per-rep columns: can't top up, so reuse
                # its mean curve as-is (the historical behaviour).
                lc = np.asarray(entry["learning_curve"], dtype=np.float32)
                lc_std = np.asarray(entry["learning_curve_std"], dtype=np.float32)
                ts = np.asarray(entry["timesteps"], dtype=np.int32)
                print(f"[BELLO] Loaded baseline curve from {excel_path} (no per-rep data; reused as-is).")
                return {"status": "ready", "result": (lc, lc_std, ts, None)}
            else:
                print(
                    "[BELLO] Disk per-rep grid differs from the current standards; "
                    "retraining from scratch."
                )
        else:
            print("[BELLO] No matching baseline on disk (different instance/standards). Training Bello.")
    else:
        print("[BELLO] No saved baseline workbook found. Training Bello.")

    disk_reps = 0 if disk_raw is None else int(disk_raw.shape[0])
    n_to_train = max(0, target_reps - disk_reps)

    # ── Disk already has at least the target: load all of them, train nothing ───
    if n_to_train == 0 and disk_reps > 0:
        raw_returns = disk_raw
        lc_mean = raw_returns.mean(axis=0).astype(np.float32)
        lc_std = (
            raw_returns.std(axis=0, ddof=1).astype(np.float32)
            if raw_returns.shape[0] > 1 else np.zeros_like(lc_mean)
        )
        print(
            f"[BELLO] Loaded all {disk_reps} repetition(s) from {excel_path} "
            f"(>= target {target_reps}); no training needed."
        )
        return {"status": "ready", "result": (lc_mean, lc_std, timesteps_grid, raw_returns)}

    if disk_reps > 0:
        print(
            f"[BELLO] Found {disk_reps} repetition(s) on disk; topping up to the "
            f"target of {target_reps} by training {n_to_train} more."
        )
    print(
        f"[BELLO] Pointer Network baseline on the {n_cities}-city instance: "
        f"{n_to_train} new rep(s) queued to the shared pool."
    )

    instance_text, instance_hash = (None, None)
    if match_training_matrices:
        instance_text, instance_hash = env_matrices_text_and_hash(env)
    metadata_base = {
        "algo_type": _BELLO_ALGO,
        "component": "Actor",
        "n_actions": n_cities,
        "input_dim": 2 * n_cities,
        "n_timesteps": int(n_timesteps),
        "eval_interval": int(eval_interval),
        **{k: BELLO_IMMUTABLE_HP[k] for k in ("lr", "embed", "hidden", "clip_logits", "decode_type")},
    }

    # Pre-create the checkpoint dir once in the main process so concurrent workers
    # don't race on it (Config also uses exist_ok, so this is belt-and-braces).
    try:
        from .Library_checkpointing import tsp_actor_checkpoint_path
        tsp_actor_checkpoint_path(algo_type=_BELLO_ALGO, rep_index=0).ensure_dir()
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


def _run_single_bello_rep(
    *,
    duration_matrix,
    n_cities,
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
    """Train one Bello repetition in a worker process (pickle-safe).

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

    bello_config, bello_actor, bello_critic, bello_env = _import_bello()
    cfg = _build_bello_cfg(bello_config, int(n_cities))
    device = torch.device("cpu")

    curve, best_tour, best_cost, actor = _train_one_rep(
        bello_actor=bello_actor,
        bello_critic=bello_critic,
        bello_env=bello_env,
        cfg=cfg,
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
        _save_bello_checkpoint(
            actor=actor, cfg=cfg, n_cities=int(n_cities), rep_index=int(rep_index),
            metadata_base=metadata_base, instance_hash=instance_hash, instance_text=instance_text,
        )
    except Exception as exc:
        print(f"[BELLO] Could not save rep {rep_index} checkpoint: {exc}")

    if shared_step_counter is not None:
        shared_step_counter.value = int(n_timesteps)
    return curve, best_tour, best_cost


def finalize_bello_baseline(
    plan,
    new_rep_results,
    *,
    base_seed,
    data_sheets_dir,
    plots_dir,
    global_config,
    format_sheets=False,
):
    """Combine disk + freshly trained Bello reps and produce the benchmark curve.

    ``new_rep_results`` is the list of ``(curve, best_tour, best_cost)`` tuples
    returned by :func:`_run_single_bello_rep` for this run's reps, in rep order.
    Assembles the mean/std curve, saves the combined workbook, draws the
    reconstructed-route plot, and returns ``(lc_mean, lc_std, timesteps,
    raw_returns)``.
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

    new_curves = []
    overall_best_cost = float("inf")
    overall_best_tour = None
    for i, (curve, best_tour, best_cost) in enumerate(new_rep_results):
        new_curves.append(np.asarray(curve, dtype=np.float32))
        rep_index = disk_reps + i
        print(f"  [BELLO] rep {rep_index + 1}/{target_reps}: best tour cost {best_cost:.4f}")
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
    # The disk repetitions were folded into ``raw_returns`` above, so the freshly
    # written workbook is a strict superset. Remove the old single-setting file
    # first so the rep count update replaces it instead of appending a duplicate
    # setting (which would then be ambiguous to match on the next run).
    if os.path.isfile(excel_path):
        try:
            os.remove(excel_path)
        except OSError as exc:
            print(f"[BELLO] Could not remove stale workbook before re-saving: {exc}")
    save_hp = dict(match_hp)
    save_hp["n_repetitions"] = int(raw_returns.shape[0])
    bello_job = {
        "curve_label": "Bello baseline",
        "method": "bello",
        "hyperparams": save_hp,
    }
    try:
        save_algorithm_workbook(
            data_sheets_dir, base_filename, _BELLO_ALGO, [bello_job], [result_tuple],
            global_config=global_config, algo_config=save_hp, format_sheets=format_sheets,
        )
    except Exception as exc:
        print(f"[BELLO] Failed to save baseline workbook: {exc}")

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
                filename=f"{_BELLO_ALGO}_route_reconstructed.png",
                seed=int(base_seed),
                title=f"Bello best tour (cost {overall_best_cost:.3f}) on reconstructed coordinates",
            )
            print(f"[BELLO] Saved reconstructed-coordinate route plot to {out_path}")
        except Exception as exc:
            print(f"[BELLO] Could not draw reconstructed-coordinate plot: {exc}")

    return result_tuple
