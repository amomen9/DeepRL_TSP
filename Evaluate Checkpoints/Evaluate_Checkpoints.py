"""
Evaluate_Checkpoints.py - standalone exhaustive evaluation of saved TSP actor
checkpoints.

Imported (and adapted) from the CartPole fork's ``Evaluate Checkpoints.py``
(https://github.com/amomen9/CartPole-v1-PolicyBased-pytorch). It builds a TSP
environment exactly the way ``Experiment.py`` does (loading a saved matrix trio
from disk, or the inline example), optionally solves it with the Held-Karp DP to
get a reference optimal cost, then rolls out *every* saved checkpoint of each
enabled algorithm for ``n_episodes`` episodes under each requested
action-selection method and reports tour-cost statistics.

Run from the project root with:  python "Evaluate Checkpoints/Evaluate_Checkpoints.py"
"""

import os
import sys

# Allow the project-root imports below to resolve when this script is run from
# its "Evaluate Checkpoints/" subdirectory.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from Environment import TSPEnvironment
from Library.Library_env_elements import inclusion_matrix_to_uncertain_routes
from Library.Helper_excel import load_sample_matrices
from Library.Library_checkpoint_eval import run_actor_checkpoint_evaluation_exhaustive
from Classic_TSP_DP import run_single_DP_experiment


_reconfigure = getattr(sys.stdout, "reconfigure", None)
if _reconfigure is not None:
    _reconfigure(encoding="utf-8")


def experiment(input=None):
    # ── Global parameters ────────────────────────────────────────────────────
    global_config = {
        "n_episodes": 100,                      # episodes rolled out per (checkpoint, method)
        "action_selection_methods": ["sample", "argmax"],  # TSP analogue of softmax/argmax
        "target_matrix": "expected",            # "expected" | "deterministic" | "stochastic"
        "reseed_seed": None,                    # reseed env noise before evaluation (None = leave as-is)
        "show_curve_plots": True,               # save + show a per-checkpoint mean-cost bar chart
        "solve_dp_reference": True,             # solve Held-Karp DP to draw the optimal-cost reference line
        # Trio selection when input is None (matrices loaded from disk):
        "sample_data_id": None,                 # 3-digit trio id; None -> latest created trio
        "sample_data_dimension": 10,            # matrix size to load; mandatory when input is None
    }

    # Which algorithms / architectures to evaluate.
    included_algo_checkpoint_eval = {
        "A2C": {"enabled": True, "actor_hidden_nn": [64, 64]},
        "PPO": {"enabled": True, "actor_hidden_nn": [128, 128]},
        "SAC": {"enabled": False, "actor_hidden_nn": [64, 64]},
    }

    # ── Build the TSP environment (mirrors Experiment.Test_TSP) ──────────────
    if input is None:
        if global_config["sample_data_dimension"] is None:
            raise ValueError("sample_data_dimension is mandatory when input is None.")
        input = load_sample_matrices(
            global_config["sample_data_dimension"], file_id=global_config["sample_data_id"]
        )

    env = TSPEnvironment(
        duration_matrix=input["duration_matrix"],
        potential_uncertainty_matrix=input["potential_uncertainty_matrix"],
        uncertain_routes=inclusion_matrix_to_uncertain_routes(input["uncertainty_inclusion_matrix"]),
        uncertainty_scale=0.0,
        uncertainty_symmetric=True,
        initialize_dp_table=True,
        initialize_noise=False,
        seed=None,
    )

    optimal_cost = None
    if global_config["solve_dp_reference"]:
        result = run_single_DP_experiment(
            env=env,
            method="bottom_up",
            max_optimal_tours=50,
            max_equivalency_table_rows=100,
            seed=None,
            print_equivalency_table=False,
            # DP-solution caching keys on the on-disk trio; absent for inline inputs.
            sample_data_id=(input.get("file_id") if isinstance(input, dict) else None),
            sample_data_timestamp=(input.get("timestamp") if isinstance(input, dict) else None),
            sample_data_dimension=global_config["sample_data_dimension"],
        )
        optimal_cost = None if result is None else result["deterministic"]["cost"]

    run_actor_checkpoint_evaluation_exhaustive(
        dp_env=env,
        included_algo_checkpoint_eval=included_algo_checkpoint_eval,
        n_episodes=global_config["n_episodes"],
        action_selection_methods=global_config["action_selection_methods"],
        target_matrix=global_config["target_matrix"],
        reseed_seed=global_config["reseed_seed"],
        show_curve_plots=global_config["show_curve_plots"],
        # Save trial results next to this script (Evaluate Checkpoints/Checkpoint
        # Evaluation Trials), independent of the current working directory.
        output_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "Checkpoint Evaluation Trials"),
        optimal_cost=optimal_cost,
    )


if __name__ == "__main__":
    experiment()
