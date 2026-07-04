"""
Library_checkpoint_eval.py - exhaustive evaluation of saved TSP actor checkpoints.

TSP adaptation of the CartPole fork's
``run_actor_checkpoint_evaluation_exhaustive`` (``Library.py``). The CartPole
version evaluates fixed-environment CartPole policies (DQN / REINFORCE / AC /
A2C / PPO) over N episodes for one or more policy-evaluation methods. This TSP
version keeps the same shape - iterate every saved checkpoint of each enabled
algorithm and roll it out for N episodes under each requested action-selection
method - but reuses the project's own TSP rollout machinery
(:mod:`Use_Trained_Model`) and reports *tour cost* statistics (lower is
better) instead of CartPole episode returns.

Each checkpoint is evaluated *individually* (not round-robin), so the spread of
performance across repetitions / continuation snapshots is visible.
"""

import copy
import glob
import os

import numpy as np
import torch

from .Library_networks import Policy_NN


def _open_in_file_explorer(path):
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


def _list_checkpoints(algo_name):
    ckpt_dir = os.path.join("Checkpoints", "TSP", str(algo_name).upper())
    return sorted(glob.glob(os.path.join(ckpt_dir, "actor_rep*.pt")))


def _load_actor(checkpoint_path, expected_n_actions, fallback_hidden_nn):
    """Load one actor checkpoint, returning ``(actor, n_actions)`` or
    ``(None, ckpt_n_actions)`` when the action space does not match."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_n_actions = int(payload.get("n_actions", expected_n_actions))
    if ckpt_n_actions != expected_n_actions:
        return None, ckpt_n_actions
    hidden_nn = payload.get("actor_hidden_nn", fallback_hidden_nn)
    actor = Policy_NN(
        nn_hidden_layer_widths=np.asarray(hidden_nn, dtype=np.int32),
        output_size=expected_n_actions,
    )
    actor.load_state_dict(payload["state_dict"])
    actor.eval()
    return actor, ckpt_n_actions


def _resolve_target_matrix(dp_env, target):
    if target == "deterministic":
        return np.asarray(dp_env.duration_matrix, dtype=float)
    if target == "stochastic":
        return np.asarray(dp_env.stochastic_duration_matrix, dtype=float)
    return np.asarray(dp_env.expected_stochastic_duration_matrix, dtype=float)


def run_actor_checkpoint_evaluation_exhaustive(
    *,
    dp_env,
    included_algo_checkpoint_eval,
    n_episodes=100,
    action_selection_methods=("sample", "argmax"),
    target_matrix="expected",
    reseed_seed=None,
    show_curve_plots=False,
    output_dir="Checkpoint Evaluation Trials",
    optimal_cost=None,
):
    """Evaluate every saved checkpoint of each enabled algorithm.

    Parameters
    ----------
    dp_env : TSP environment with an initialised DP/state table.
    included_algo_checkpoint_eval : {ALGO: {"enabled": bool,
                                            "actor_hidden_nn": array}}.
    n_episodes : episodes rolled out per (checkpoint, method).
    action_selection_methods : iterable of "sample" / "argmax".
    target_matrix : "expected" | "deterministic" | "stochastic".
    """
    # Imported lazily so this module stays import-light for the worker pool.
    from Use_Trained_Model import _run_loop_on_matrix

    os.makedirs(output_dir, exist_ok=True)

    # Roll out on a copy whose DP state table is dropped: step() only consults
    # the DP subset map while a table is present, and the trained-model rollout
    # path (like evaluate_trained_model_on_matrices) expects it absent. The DP
    # table, if any, is only needed for the optimal-cost reference computed by
    # the caller before this routine runs.
    dp_env = copy.deepcopy(dp_env)
    if getattr(dp_env, "DP_Table", None) is not None and hasattr(dp_env, "destroy_state_table"):
        dp_env.destroy_state_table()

    if isinstance(action_selection_methods, str):
        action_selection_methods = [action_selection_methods]
    action_selection_methods = list(action_selection_methods)

    if reseed_seed is not None:
        dp_env.reseed_noise(reseed_seed)

    expected_n_actions = int(dp_env.n_actions)
    target_mat = _resolve_target_matrix(dp_env, target_matrix)
    n_full_tour = int(dp_env.n) + 1

    cfg = included_algo_checkpoint_eval or {}
    enabled_algos = [
        algo for algo in ("A2C", "PPO", "SAC")
        if bool((cfg.get(algo) or {}).get("enabled", False))
    ]
    if not enabled_algos:
        print("[checkpoint-eval] No algorithms enabled; nothing to evaluate.")
        return {}

    print(f"\n{'=' * 78}")
    print(
        f"  Exhaustive checkpoint evaluation - {n_episodes} episode(s) per checkpoint, "
        f"target='{target_matrix}', methods={action_selection_methods}"
    )
    if optimal_cost is not None:
        print(f"  DP optimal cost (reference): {optimal_cost:.4f}")
    print(f"{'=' * 78}")

    results: dict = {}
    plot_series: list[tuple[str, list[str], list[float]]] = []

    for algo in enabled_algos:
        algo_cfg = cfg.get(algo) or {}
        fallback_hidden_nn = algo_cfg.get("actor_hidden_nn", np.array([64, 64], dtype=np.int32))
        ckpt_paths = _list_checkpoints(algo)
        if not ckpt_paths:
            print(f"\n[{algo}] No checkpoints found in Checkpoints/TSP/{algo}/; skipping.")
            continue

        print(f"\n[{algo}] {len(ckpt_paths)} checkpoint file(s) found.")
        results[algo] = {}
        for method in action_selection_methods:
            labels: list[str] = []
            means: list[float] = []
            print(f"  Action-selection method: '{method}'")
            for path in ckpt_paths:
                actor, ckpt_n_actions = _load_actor(path, expected_n_actions, fallback_hidden_nn)
                name = os.path.basename(path)
                if actor is None:
                    print(f"    {name}: skipped (n_actions={ckpt_n_actions} != {expected_n_actions})")
                    continue

                tours, costs = _run_loop_on_matrix(
                    dp_env, [actor], target_mat, n_episodes, method
                )
                valid = [
                    c for t, c in zip(tours, costs)
                    if len(t) == n_full_tour and np.isfinite(c)
                ]
                n_valid = len(valid)
                if n_valid > 0:
                    arr = np.asarray(valid, dtype=float)
                    mean = float(np.mean(arr))
                    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
                    best = float(np.min(arr))
                else:
                    mean = std = best = float("inf")

                results[algo].setdefault(method, {})[name] = {
                    "mean": mean, "std": std, "best": best,
                    "n_valid": n_valid, "n_episodes": n_episodes,
                }
                labels.append(name)
                means.append(mean)
                print(
                    f"    {name}: mean={mean:.4f}, std={std:.4f}, best={best:.4f}, "
                    f"valid={n_valid}/{n_episodes}"
                )
            if labels:
                plot_series.append((f"{algo} ({method})", labels, means))

    if show_curve_plots and plot_series:
        _plot_checkpoint_means(plot_series, output_dir, target_matrix, optimal_cost)

    return results


def _plot_checkpoint_means(plot_series, output_dir, target_matrix, optimal_cost):
    """Bar chart of per-checkpoint mean tour cost for each (algo, method)."""
    import matplotlib.pyplot as plt

    n = len(plot_series)
    fig, axes = plt.subplots(n, 1, figsize=(10, max(3, 2.4 * n)), squeeze=False)
    for ax, (series_label, labels, means) in zip(axes[:, 0], plot_series):
        finite = [m if np.isfinite(m) else np.nan for m in means]
        x = np.arange(len(labels))
        ax.bar(x, finite, color="#3a7ca5")
        if optimal_cost is not None:
            ax.axhline(optimal_cost, color="crimson", linestyle="--", linewidth=1,
                       label=f"DP optimal = {optimal_cost:.2f}")
            ax.legend(fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Mean tour cost")
        ax.set_title(series_label, fontsize=10)
    fig.suptitle(f"Checkpoint evaluation - mean tour cost (target='{target_matrix}')")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_path = os.path.join(output_dir, "checkpoint_evaluation_means.png")
    fig.savefig(out_path, dpi=200)
    print(f"\nSaved checkpoint-evaluation plot to {out_path}")
    _open_in_file_explorer(output_dir)
    plt.show(block=True)
