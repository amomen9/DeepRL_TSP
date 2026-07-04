"""
Use_Trained_Model.py - evaluate a trained TSP actor against the three
matrices held by ``StochasticTSPEnvironment``:

  1. ``duration_matrix`` (deterministic baseline).
  2. ``expected_stochastic_duration_matrix`` (matrix the DP solver sees).
  3. ``stochastic_duration_matrix`` after a fresh ``reseed_noise()`` call.

Each matrix gets its own loop of ``n_use_trained_model`` episodes. The 3 loops
are independent: no matrix overrides another. Mean/Std for each loop are
computed across the loop's ``n_use_trained_model`` repetitions.
"""

import copy
import glob
import os

import numpy as np
import torch

from Library.Library_networks import Policy_NN
from Library.Library_env_elements import format_tour
from Classic_TSP_DP import (
    _format_route_edge_costs,
    _format_scalar_value,
    _tour_cost_value,
)


def _checkpoint_paths(algo_name):
    ckpt_dir = os.path.join("Checkpoints", "TSP", str(algo_name).upper())
    return sorted(glob.glob(os.path.join(ckpt_dir, "actor_rep*.pt")))


def _invalid_action_mask(env, n_actions):
    mask = torch.zeros(n_actions, dtype=torch.bool)
    visited = getattr(env, "current_visited_cities", None)
    if not visited:
        return mask
    current = getattr(env, "current_location", None)
    for action in range(n_actions):
        # Honour the env's active depot instead of assuming actions map to
        # cities 1..n-1 (the depot cycles and may be any city).
        next_city = env._action_to_city(action)
        if next_city in visited or next_city == current:
            mask[action] = True
    return mask


def _select_action(actor, obs, n_actions, env, action_selection_method):
    with torch.no_grad():
        state = torch.as_tensor(obs, dtype=torch.float32)
        logits = actor(state)
        mask = _invalid_action_mask(env, n_actions)
        if bool(torch.all(mask)):
            return int(torch.argmax(logits).item())
        masked_logits = logits.masked_fill(mask, -1e9)
        if action_selection_method == "argmax":
            return int(torch.argmax(masked_logits).item())
        if action_selection_method == "sample":
            dist = torch.distributions.Categorical(logits=masked_logits)
            return int(dist.sample().item())
        raise ValueError(
            f"Unknown action_selection_method: '{action_selection_method}'. "
            "Use 'argmax' or 'sample'."
        )


def _rollout_on_env(env, actor, action_selection_method):
    obs, _ = env.reset()
    n_actions = env.n_actions
    depot = env.current_depot_city
    tour = [env.current_location]
    done, truncated = False, False
    while not (done or truncated):
        action = _select_action(actor, obs, n_actions, env, action_selection_method)
        chosen_city = env._action_to_city(action)  # depot-aware action -> city
        obs, _, done, truncated, info = env.step(action)
        if info.get("invalid_action"):
            break  # leave tour as-is; validate_tour() will mark it invalid
        # When all cities have been visited, env.step() auto-appends the
        # closing edge back to the depot, leaving current_location at the depot.
        # Record both the chosen city and the depot in that case so the tour
        # contains every visit explicitly.
        if env.current_location != chosen_city and chosen_city != depot:
            tour.append(chosen_city)
            tour.append(env.current_location)
        else:
            tour.append(env.current_location)
    return tour


def _run_loop_on_matrix(dp_env, actors, target_matrix, n_use_trained_model, action_selection_method):
    """Run ``n_use_trained_model`` episodes with the env's step() reading from
    ``target_matrix``. Actors are used round-robin so every saved repetition
    contributes to the loop's statistics.
    """
    eval_env = copy.deepcopy(dp_env)
    eval_env.stochastic_duration_matrix = np.asarray(target_matrix, dtype=float).copy()
    target = eval_env.stochastic_duration_matrix

    tours, costs = [], []
    for i in range(n_use_trained_model):
        actor = actors[i % len(actors)]
        tour = _rollout_on_env(eval_env, actor, action_selection_method)
        if eval_env.validate_tour(tour):
            cost = _tour_cost_value(target, tour)
        else:
            cost = float("inf")
        tours.append(tour)
        costs.append(cost)
    return tours, costs


def evaluate_trained_model_on_matrices(
    dp_env,
    algo_name,
    actor_hidden_nn,
    n_use_trained_model=10,
    action_selection_method="argmax",
    reseed_seed=None,
):
    """Evaluate the saved baseline actors against the three reference matrices.

    Returns
    -------
    (model_rows, model_best_tour, eval_summary)
        ``model_rows`` is a list of 3 dicts (one per matrix) ready to be
        appended via :func:`Classic_TSP_DP._format_route_equivalency_table`.
        ``model_best_tour`` is the tour with lowest cost on
        ``expected_stochastic_duration_matrix`` across the 3 loops (used to
        recompute the *Extra cost* column).
        ``eval_summary`` is a short list of (label, mean, std, n_valid)
        tuples for logging.
    """
    ckpt_paths = _checkpoint_paths(algo_name)
    if not ckpt_paths:
        raise FileNotFoundError(
            f"No checkpoints found in Checkpoints/TSP/{str(algo_name).upper()}/. "
            "Run training first to produce actor_rep*.pt files."
        )

    expected_n_actions = int(dp_env.n_actions)
    actors: list[torch.nn.Module] = []
    skipped_ckpts: list[tuple[str, int]] = []

    # Load only actors whose action space matches the current dp_env.
    # This prevents a common failure mode: evaluating a checkpoint trained
    # for a different TSP size than the env used for evaluation.
    for p in ckpt_paths:
        payload = torch.load(p, map_location="cpu", weights_only=False)
        ckpt_n_actions = int(payload.get("n_actions", expected_n_actions))
        if ckpt_n_actions != expected_n_actions:
            skipped_ckpts.append((p, ckpt_n_actions))
            continue

        ckpt_actor_hidden_nn = payload.get("actor_hidden_nn", actor_hidden_nn)
        actor = Policy_NN(
            nn_hidden_layer_widths=np.asarray(ckpt_actor_hidden_nn, dtype=np.int32),
            output_size=expected_n_actions,
        )
        actor.load_state_dict(payload["state_dict"])
        actor.eval()
        actors.append(actor)

    if not actors:
        distinct = sorted({n for _, n in skipped_ckpts})
        skipped_preview = ", ".join(f"{os.path.basename(p)}(n_actions={n})" for p, n in skipped_ckpts[:5])
        raise ValueError(
            "No actor checkpoints match the current environment action space. "
            f"dp_env.n={dp_env.n}, dp_env.n_actions={expected_n_actions}. "
            f"Checkpoint action-space sizes found: {distinct}. "
            f"Skipped (preview): {skipped_preview}"
        )

    if skipped_ckpts:
        skipped_preview = ", ".join(f"{os.path.basename(p)}(n_actions={n})" for p, n in skipped_ckpts[:5])
        print(
            f"[evaluate] Skipping {len(skipped_ckpts)} checkpoint(s) due to action-space mismatch. "
            f"dp_env.n_actions={expected_n_actions}. Skipped preview: {skipped_preview}"
        )

    duration_matrix = np.asarray(dp_env.duration_matrix, dtype=float)
    expected_matrix = np.asarray(dp_env.expected_stochastic_duration_matrix, dtype=float)

    if reseed_seed is not None:
        dp_env.reseed_noise(reseed_seed)
    else:
        dp_env.reseed_noise()  # reseed with a random seed to get a different stochastic matrix
    
    print(f"Matrix means dur, exp, sto: {np.mean(dp_env.duration_matrix.flatten())}, {np.mean(dp_env.expected_stochastic_duration_matrix.flatten())}, {np.mean(dp_env.stochastic_duration_matrix.flatten())}")
    stochastic_matrix = np.asarray(dp_env.stochastic_duration_matrix, dtype=float).copy()

    matrices = [
        ("M·det", "duration_matrix", duration_matrix),
        ("M·exp", "expected_stochastic_duration_matrix", expected_matrix),
        ("M·sto", "stochastic_duration_matrix (reseeded)", stochastic_matrix),
    ]

    per_matrix = []
    all_valid_tours = []
    for label, name, mat in matrices:
        tours, costs = _run_loop_on_matrix(
            dp_env, actors, mat, n_use_trained_model, action_selection_method
        )
        per_matrix.append((label, name, mat, tours, costs))
        for t in tours:
            if len(t) == dp_env.n + 1:
                all_valid_tours.append(t)

    if all_valid_tours:
        best_idx = int(np.argmin([
            _tour_cost_value(expected_matrix, t) for t in all_valid_tours
        ]))
        model_best_tour = list(all_valid_tours[best_idx])
    else:
        model_best_tour = None

    model_rows = []
    eval_summary = []
    for label, name, mat, tours, costs in per_matrix:
        # Preserve *all* repetition costs as rep_1..rep_n in the output matrix.
        # Invalid/unfinished tours are treated as +inf for statistics.
        rep_costs: list[float] = []
        valid_mask: list[bool] = []
        for t, c in zip(tours, costs):
            is_valid = (len(t) == dp_env.n + 1)
            valid_mask.append(is_valid)
            rep_costs.append(float(c) if is_valid and np.isfinite(c) else float("inf"))

        n_valid = int(sum(1 for v in valid_mask if v))

        if n_valid > 0:
            costs_arr = np.asarray([c for c in rep_costs if np.isfinite(c)], dtype=float)
            mean = float(np.mean(costs_arr))
            std = float(np.std(costs_arr, ddof=1)) if costs_arr.size > 1 else 0.0

            rep_idx = int(np.argmin(rep_costs))
            rep_tour = list(tours[rep_idx]) if valid_mask[rep_idx] else None
        else:
            mean = float("inf")
            std = 0.0
            rep_tour = None

        eval_summary.append((name, mean, std, n_valid))

        # Prepare repetition columns in model_rows.
        rep_fields = {f"rep_{i + 1}": _format_scalar_value(rep_costs[i]) for i in range(len(rep_costs))}

        if rep_tour is not None:
            model_rows.append({
                "row_label": label,
                "tour": rep_tour,
                "model_route": format_tour(rep_tour),
                "model_cost": _format_scalar_value(_tour_cost_value(mat, rep_tour)),
                "model_mean": _format_scalar_value(mean),
                "model_std": _format_scalar_value(std),
                "deterministic_route": "",
                "uncertain_route": "",
                "deterministic_cost": _format_scalar_value(_tour_cost_value(duration_matrix, rep_tour)),
                "uncertain_cost": _format_scalar_value(_tour_cost_value(expected_matrix, rep_tour)),
                "extra_cost": "",
                "deterministic_individual_route_costs": _format_route_edge_costs(duration_matrix, rep_tour),
                "uncertain_individual_route_costs": _format_route_edge_costs(expected_matrix, rep_tour),
                "mod_route_costs": _format_route_edge_costs(mat, rep_tour),
                **rep_fields,
            })
        else:
            model_rows.append({
                "row_label": label,
                "tour": None,
                "model_route": "Not found",
                "model_cost": "Not found",
                "model_mean": _format_scalar_value(mean),
                "model_std": _format_scalar_value(std),
                "deterministic_route": "Not found",
                "uncertain_route": "Not found",
                "deterministic_cost": "Not found",
                "uncertain_cost": "Not found",
                "extra_cost": "Not found",
                "deterministic_individual_route_costs": "Not found",
                "uncertain_individual_route_costs": "Not found",
                "mod_route_costs": "Not found",
                **rep_fields,
            })

    return model_rows, model_best_tour, eval_summary


def fill_dp_route_match(model_rows, det_tours, avg_tours):
    """Set Det/Unc Route columns of each model row to ``format_tour(model tour)``
    if that tour also appears among the DP det/unc tours, else ``"Not found"``.
    """
    det_set = {tuple(t) for t in det_tours}
    avg_set = {tuple(t) for t in avg_tours}
    for row in model_rows:
        tour = row.get("tour")
        if tour is None:
            row["deterministic_route"] = "Not found"
            row["uncertain_route"] = "Not found"
            continue
        tup = tuple(tour)
        row["deterministic_route"] = format_tour(tour) if tup in det_set else "Not found"
        row["uncertain_route"] = format_tour(tour) if tup in avg_set else "Not found"
    return model_rows
