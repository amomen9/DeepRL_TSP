"""
Experiment.py - Main entry point for TSP experiments.

Runs the Held-Karp solver, Reinforcement Learning trials, and heuristic agent trials on 

example instances and random instances of

increasing size, reporting optimal solutions and computing times.

References: Section 5.3 (slides 16-17), Section 5.4 (slides 19-21),
and method comparison note on slide 18.
"""

import os
import sys
import copy
import ast
import argparse
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np

from Library.Library_experiment_orchestrator import run_selected_experiments
from Environment import TSPEnvironment
from Library.Library_env_elements import inclusion_matrix_to_uncertain_routes
from Library.Helper_excel import load_sample_matrices
from Classic_TSP_DP import (
    _build_route_equivalency_rows,
    _format_route_equivalency_table,
    run_single_DP_experiment,
)
from Use_Trained_Model import evaluate_trained_model_on_matrices, fill_dp_route_match


_reconfigure = getattr(sys.stdout, "reconfigure", None)
if _reconfigure is not None:
    _reconfigure(encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Configuration schemas
#
# These ``TypedDict``s document the exact set of keys each config section
# accepts and double as the single source of truth for key-validation below.
# They are *typing only*: annotating the dict literals with them adds editor
# autocomplete and static type-checking but changes nothing at runtime (the
# configs stay plain dicts, and PEP 526 local annotations are never evaluated).
# ``total=False`` lets partial dicts (e.g. overrides) type-check too.
# ─────────────────────────────────────────────────────────────────────────────
class IncludedAlgorithms(TypedDict, total=False):
    A2C: bool
    PPO: bool
    SAC: bool
    TSP_TEST: bool


class A2CConfig(TypedDict, total=False):
    gamma: List[float]
    actor_lr: List[float]
    actor_hidden_nn: List[List[int]]
    critic_lr: List[float]
    critic_hidden_nn: List[List[int]]
    FULL_EPISODE_UPDATES: List[bool]
    TN_step: List[int]
    legend_parameters: Dict[str, list]


class PPOConfig(TypedDict, total=False):
    gamma: List[float]
    actor_lr: List[float]
    actor_hidden_nn: List[List[int]]
    critic_lr: List[float]
    critic_hidden_nn: List[List[int]]
    FULL_EPISODE_UPDATES: List[bool]
    gae_lambda: List[float]
    clip_epsilon: List[float]
    n_epochs: List[int]
    entropy_coef: List[float]
    value_coef: List[float]
    rollout_steps: List[int]
    legend_parameters: Dict[str, list]


class SACConfig(TypedDict, total=False):
    gamma: List[float]
    actor_lr: List[float]
    actor_hidden_nn: List[List[int]]
    critic_lr: List[float]
    critic_hidden_nn: List[List[int]]
    FULL_EPISODE_UPDATES: List[bool]
    TN_step: List[int]
    alpha: List[float]
    alpha_lr: List[float]
    auto_tune_alpha: List[bool]
    target_entropy_ratio: List[float]
    tau: List[float]
    legend_parameters: Dict[str, list]


class TSPTestConfig(TypedDict, total=False):
    n_repetitions: int
    steps: int
    batch: int
    pomo_size: Optional[int]
    embed: int
    n_heads: int
    n_layers: int
    ff_hidden: int
    clip_logits: float
    softmax_T: float
    lr: float
    weight_decay: float
    grad_norm_clip: float
    is_lr_decay: bool
    lr_decay: float
    lr_decay_step: int
    curve_label: str


class GlobalConfig(TypedDict, total=False):
    MIN_UNUSED_CPU_CORES: int
    n_repetitions: int
    k_order_aggregation_methods: Dict[str, Any]
    benchmark_curve: int
    benchmark_name: str
    plot_smoothing_window: List[int]
    curve_confidence_interval: float
    curve_shaded_area_opacity: float
    curve_plot: bool
    animation_plot: bool
    TSP_Optimal_Cost: Optional[float]
    TSP_Best_Cost: Optional[float]
    TSP_Worst_Cost: Optional[float]
    use_existing_disk_data: bool
    checkpoints: Dict[str, bool]
    match_training_matrices: bool
    Environment: Any
    n_timesteps: float
    max_episode_length: int
    base_seed: int
    eval_interval: int
    n_eval_episodes: int
    baseline_model: Optional[str]
    n_use_trained_model: int
    action_selection_method: str
    trained_model_reseed_seed: Optional[int]


# Section name -> schema; the schema's annotations are the allowed keys.
_SECTION_SCHEMAS: Dict[str, Any] = {
    "global_config": GlobalConfig,
    "a2c_config": A2CConfig,
    "ppo_config": PPOConfig,
    "sac_config": SACConfig,
    "tsp_test_config": TSPTestConfig,
    "included_algorithms": IncludedAlgorithms,
}

# Short aliases accepted on the command line (and in ``--set``) -> section name.
_CLI_SECTION_ALIASES: Dict[str, str] = {
    "global": "global_config",
    "a2c": "a2c_config",
    "ppo": "ppo_config",
    "sac": "sac_config",
    "tsp_test": "tsp_test_config",
    "included": "included_algorithms",
}


def _validate_config_keys(section_name: str, values: dict) -> None:
    """Raise if *values* holds a key the section's schema doesn't declare.

    Catches typos in the in-script config, in programmatic ``overrides`` and in
    command-line overrides up front, instead of letting them silently no-op deep
    inside a run. Only top-level keys are checked (nested dicts are free-form).
    """
    schema = _SECTION_SCHEMAS.get(section_name)
    if schema is None:
        raise KeyError(
            f"Unknown configuration section '{section_name}'. "
            f"Valid sections: {sorted(_SECTION_SCHEMAS)}."
        )
    allowed = set(schema.__annotations__)
    unknown = sorted(k for k in values if k not in allowed)
    if unknown:
        raise KeyError(
            f"Unknown key(s) {unknown} for section '{section_name}'. "
            f"Valid keys: {sorted(allowed)}."
        )


def _str2bool(value: str) -> bool:
    """argparse helper so boolean flags accept true/false/yes/no/1/0."""
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "t", "yes", "y", "1"):
        return True
    if value.lower() in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got '{value}'")


def _apply_set_item(targets: Dict[str, Any], item: str) -> None:
    """Apply one generic ``SECTION.KEY[.SUBKEY...]=VALUE`` CLI override in place.

    Walks into the live section dict so nested overrides (e.g. a single key
    inside ``checkpoints``) preserve their sibling keys. ``VALUE`` is parsed with
    ``ast.literal_eval`` so numbers, bools and lists work (``[0.9,0.99]``,
    ``[[256,256]]``); anything that won't parse is kept as a raw string.
    """
    if "=" not in item:
        raise ValueError(
            f"--set expects SECTION.KEY=VALUE (e.g. ppo.gamma=[0.9,0.99]), got '{item}'."
        )
    path, raw = item.split("=", 1)
    parts = [p for p in path.strip().split(".") if p]
    if len(parts) < 2:
        raise ValueError(
            f"--set path '{path}' must be SECTION.KEY (optionally nested), e.g. "
            "global.checkpoints.use_saved_disk_networks_checkpoints=True."
        )
    section_name = _CLI_SECTION_ALIASES.get(parts[0])
    if section_name is None:
        raise KeyError(
            f"Unknown --set section '{parts[0]}'. Valid: {sorted(_CLI_SECTION_ALIASES)}."
        )
    keys = parts[1:]
    _validate_config_keys(section_name, {keys[0]: None})  # top-level key check
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        value = raw  # fall back to the raw string
    node: dict = targets[section_name]
    for key in keys[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            child = {}
            node[key] = child
        node = child
    node[keys[-1]] = value


def Test_TSP(input=None, *, overrides=None):
    """A2C / SAC / PPO training on some TSP matrices from ``Test_TSP``.

    ``overrides`` lets the per-ablation ``Experiment_*.py`` scripts reproduce a
    single sweep with one command (the TSP analogue of the CartPole fork's
    dedicated ``Experiment_*.py`` files). It is a dict with any of the keys
    ``global_config`` / ``a2c_config`` / ``ppo_config`` / ``sac_config`` /
    ``included_algorithms``; each maps to a dict that is shallow-merged into the
    corresponding configuration just before the run, e.g.
    ``overrides={"ppo_config": {"gamma": [0.9, 0.99, 1.0]}}``.
    """
    # Trio selection used when ``input`` is None (matrices are loaded from disk).
    #   sample_data_id        -> 3-digit trio id; None picks the latest created trio.
    #   sample_data_dimension -> matrix size to load; mandatory when input is None.
    sample_data_id = None
    sample_data_dimension = 80

    # Resolve the matrices: a provided ``input`` overrides the on-disk trio.
    if input is None:
        if sample_data_dimension is None:
            raise ValueError(
                "sample_data_dimension is mandatory when input is None."
            )
        input = load_sample_matrices(sample_data_dimension, file_id=sample_data_id)

    duration_matrix = input["duration_matrix"]
    potential_uncertainty_matrix = input["potential_uncertainty_matrix"]
    uncertain_routes = inclusion_matrix_to_uncertain_routes(
        input["uncertainty_inclusion_matrix"]
    )

    env = TSPEnvironment(
        duration_matrix=duration_matrix,
        potential_uncertainty_matrix=potential_uncertainty_matrix,
        uncertain_routes=uncertain_routes,
        uncertainty_scale=0.0,
        uncertainty_symmetric=True,
        initialize_dp_table=True,
        initialize_noise=False,
        seed=None,
    )

    print("\nTSP duration matrix for policy-based training:\n", env.duration_matrix, "\n\n")
    print("duration_matrix:\n", env.duration_matrix, "\n\n")
    print("potential_uncertainty_matrix:\n", env.potential_uncertainty_matrix, "\n\n")
    if env.stochastic_duration_matrix is not None:
        print("stochastic_duration_matrix:\n", env.stochastic_duration_matrix, "\n\n")
    print("expected_stochastic_duration_matrix:\n", env.expected_stochastic_duration_matrix, "\n\n")
    print(f"Effective noise matrix for policy-based training:\n{env.effective_noise_matrix}\n")

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
        sample_data_dimension=sample_data_dimension,
    )
    env.destroy_state_table()
    dp_env = copy.deepcopy(env)
    env = None

    best_cost = None if result is None else result["deterministic"]["cost"]
    worst_cost = None if result is None else result["worst"]["cost"]

    global_config: GlobalConfig = {
        "MIN_UNUSED_CPU_CORES": 3,
        "n_repetitions": 6,
        "k_order_aggregation_methods": {                    # methods to aggregate the k-order equivalency results into a single value for plotting and comparison, with their parameters (e.g., k value for k-order methods)
            "enabled": [True],      # ✓/✗
            "inclusion_or_exclusion_mode": ["inclusion"],   # "inclusion" or "exclusion" mode for k-order aggregation which controls whether we consider the top-k best tours (inclusion) or the tours ranked from k to the end (exclusion) when aggregating the equivalency results. Default: "inclusion".
            "ordering_timesteps": ["n_timesteps"],          # "n_timesteps" or "final_timestep" which controls whether we order the tours based on their costs at each timestep (n_timesteps) or only at the final timestep (final_timestep) when performing k-order aggregation. Default: "n_timesteps".
            "mean": [True],
            "k_order_max": ([True], [1]),
            "k_order_min": ([False], [1]),
            "k_order_median": ([False], [3]),
            "k_order_cluster": ([True], [([2], "max")]),
            "legend_parameters": {
                "enabled": [r"k-agg: ", True],
                "inclusion_or_exclusion_mode": [r"mode: ", True],
                "ordering_timesteps": [r"$t_{\mathrm{ord}}$: ", True],
                "mean": [r"mean", False],
                "k_order_max": [r"$k_{\max}$: ", True],
                "k_order_min": [r"$k_{\min}$: ", False],
                "k_order_median": [r"$k_{\mathrm{med}}$: ", True],
                "k_order_cluster": [r"clt: ", True],
            },
        },
        "benchmark_curve": 1,
        "benchmark_name": "Baseline",
        "baseline_model": "TSP_TEST",      # Bello | TSP_TEST | A2C | PPO | SAC | None (or None) to disable; if set, the model is evaluated on the TSP matrices and its performance is included in the equivalency table and the learning curves plot for comparison with the DP optimal solution and the RL training curves. Bello and TSP_TEST are dedicated-runner baselines (their mean curve becomes the benchmark curve); TSP_TEST normally runs as a regular experiment via included_algorithms instead.
        "plot_smoothing_window": [1, 101, 201, 251, 351],
        "curve_confidence_interval": 0.95,
        "curve_shaded_area_opacity": 0.05,
        "curve_plot": True,
        "animation_plot": False,
        "TSP_Optimal_Cost": best_cost,
        "TSP_Best_Cost": best_cost,
        "TSP_Worst_Cost": worst_cost,
        "use_existing_disk_data": False,
        # Checkpoint reuse / continuation (imported from the CartPole fork).
        #   use_saved_disk_networks_checkpoints -> when True, each repetition loads
        #       a matching saved actor/critic from Checkpoints/TSP/<ALGO>/ and
        #       *continues* training from it, accumulating the timestep counter in
        #       the .txt sidecar. Learning-curve plots and the returns-summary
        #       table are then written under "Trial Continuation Analysis/".
        #   skip_selection_hyperparameter_match -> relax the exact-metadata match
        #       to "largest n_timesteps among n_actions-compatible candidates";
        #       use this for repeated continuation runs.
        #   match_training_matrices -> also key checkpoint matching on the
        #       instance-defining matrices (duration / inclusion / potential-
        #       uncertainty): a saved actor/critic is reused only if it was
        #       trained on the same instance. Excel-results matching always keys
        #       on these matrices regardless of this flag. Set False to restore
        #       the legacy n_actions/architecture-only checkpoint matching.
        "checkpoints": {
            "use_saved_disk_networks_checkpoints": True,
            "skip_selection_hyperparameter_match": True,
            "match_training_matrices": True,
        },
        "Environment": dp_env,
        "n_timesteps": 5e5,
        "max_episode_length": dp_env.n_cities + 1,
        "base_seed": 42,
        "eval_interval": 250,
        "n_eval_episodes": 5,                   # number of episodes to run for evaluation at each eval_interval during training; this controls how many times we evaluate the trained model on the TSP environment at each evaluation point, and can affect the stability and reliability of the evaluation results.
        "n_use_trained_model": 10,              # number of evaluation episodes to run when evaluating the trained model on the TSP matrices; this controls how many times we run the trained model on the TSP environment to compute an average cost for comparison with the DP optimal solution and the baseline model, and can affect the stability and reliability of the evaluation results.
        "action_selection_method": "sample",    # "sample"/"argmax". "sample" for sampling from the policy's action distribution, "argmax" for always choosing the action with the highest probability; this controls how actions are selected during evaluation of the trained model on the TSP matrices, and can affect the performance of the trained model compared to the DP optimal solution and the baseline model.
        "trained_model_reseed_seed": None,      # Optional seed for reseeding the environment when evaluating the trained model on the TSP matrices. If set, this ensures that the evaluation is reproducible and not affected by stochasticity in the environment; if None, no reseeding is done and the evaluation may yield different results across runs due to environment randomness.
    }

    included_algorithms: IncludedAlgorithms = {
        "A2C": False,
        "PPO": True,
        "SAC": False,
        "TSP_TEST": True,   # TSP-DRL-Test submodule: POMO-trained Attention Model (Kool 2019 + Kwon 2020), the policy-based contender to the Bello baseline. Configured via tsp_test_config below; run by a dedicated runner (Library_tsp_test_experiment) on the same fixed instance, its curve is drawn next to A2C/PPO/SAC and the Bello benchmark.
    
    }

    a2c_config: A2CConfig = {
        "gamma": [0.999],                    # list of discount factors to sweep
        "actor_lr": [3.5e-4],
        "actor_hidden_nn": [[64, 64]],
        "critic_lr": [0.01],
        "critic_hidden_nn": [[128, 128]],
        "FULL_EPISODE_UPDATES": [True],
        "TN_step": [10],
        "legend_parameters": {
            "gamma": [r"γ:", True],
            "actor_lr": [r"A-α:", True],
            "critic_lr": [r"C-β:", True],
            "actor_hidden_nn": [r"A-NN:", True],
            "critic_hidden_nn": [r"C-NN:", False],
            "FULL_EPISODE_UPDATES": [r"Full-Ep:", True],
            "TN_step": [r"TN:", False],
        },
    }


    ppo_config: PPOConfig = {
        "gamma": [0.99],                    # list of discount factors to sweep
        "actor_lr": [3e-4],                 # actor learning rate(s) to sweep
        "actor_hidden_nn": [[128, 128]],    # actor NN architectures to sweep
        "critic_lr": [1e-3],                # 1e-2 made the critic oscillate (noisy GAE advantages -> zig-zag curves); keep within ~3x of actor_lr
        "critic_hidden_nn": [[128, 128]],   # 512/256 # critic NN architectures to sweep (shrunk: 4-D input needs little critic capacity, far cheaper fwd/bwd)
        "FULL_EPISODE_UPDATES": [True, False],    # True updated on one ~n-step episode at a time (tiny batch -> per-episode overfitting); False uses the rollout_steps buffer below
        "gae_lambda": [0.96],               # 0.96 # GAE lambda parameter which controls the bias-variance trade-off of the Generalized Advantage Estimation (GAE). Default: 0.95. Set to 1.0 to disable GAE and use regular advantage estimation.
        "clip_epsilon": [0.1],              # 0.1  # PPO clipping epsilon which controls the clipping range for the probability ratio in the PPO surrogate objective. Default: 0.2.
        "n_epochs": [10],                   # 16   # of optimisation epochs per rollout which controls how many times we reuse each collected rollout batch of data to update the policy. Default: 10. Set to 1 to skip PPO epoch trials and only do one epoch per rollout.
        "entropy_coef": [0.01],      
        "value_coef": [0.5],                # 0.5 # coefficient for the value loss term in the PPO objective which controls the relative importance of the value function loss compared to the policy loss. Default: 0.5.
        "rollout_steps": [512],             # 1024 # of env steps per rollout (PPO buffer size) which controls how many steps of data we collect in each rollout before we perform policy updates. Default: 2048. Set to a large number (e.g., 1e6) to skip rollout length trials and effectively use the entire episode as one rollout.
        "legend_parameters": {
            "gamma": [r"γ:", True],
            "actor_lr": [r"A-α:", True],
            "critic_lr": [r"C-β:", False],
            "actor_hidden_nn": [r"A-NN:", True],
            "critic_hidden_nn": [r"C-NN", False],
            "FULL_EPISODE_UPDATES": [r"Full-Ep:", True],
            "gae_lambda": [r"$λ_{GAE}$:", True],
            "clip_epsilon": [r"ε:", False],
            "n_epochs": [r"epc:", False],
            "entropy_coef": [r"ent:", False],
            "value_coef": [r"val:", False],
            "rollout_steps": [r"roll:", False],
        },
    }


    sac_config: SACConfig = {
        "gamma": [0.99],
        "actor_lr": [3e-4],
        "actor_hidden_nn": [[64, 64]],
        "critic_lr": [3e-4],
        "critic_hidden_nn": [[64, 64]],
        "FULL_EPISODE_UPDATES": [True],
        "TN_step": [1],
        "alpha": [0.2],
        "alpha_lr": [3e-3],
        "auto_tune_alpha": [True],
        "target_entropy_ratio": [0.01],
        "tau": [0.005],
        "legend_parameters": {
            "gamma": [r"γ:", False],
            "actor_lr": [r"A-α:", True],
            "critic_lr": [r"C-β:", False],
            "actor_hidden_nn": [r"A-NN:", True],
            "critic_hidden_nn": [r"C-NN", False],
            "FULL_EPISODE_UPDATES": [r"Full-Ep:", True],
            "TN_step": [r"TN:", False],
            "alpha": [r"$α_{ent}$:", False],
            "alpha_lr": [r"$α_{ent}$-lr:", False],
            "auto_tune_alpha": [r"At-α:", False],
            "target_entropy_ratio": [r"$Hr$:", False],
            "tau": [r"$τ$:", False],
        },
    }


    # TSP-DRL-Test (submodule): the Attention Model (Kool et al. 2019) trained
    # with POMO shared-baseline REINFORCE (Kwon et al. 2020) - the strongest
    # policy-based recipe for the TSP and the direct contender to the Bello
    # Pointer-Network baseline. It learns the same fixed duration-matrix
    # instance through the same data interface as Bello (outgoing-row concat
    # incoming-column node features, directed matrix tour cost), so the two
    # compete on equal information. These are the method's global config
    # params - the single source of truth, passed to the dedicated runner in
    # Library_tsp_test_experiment.py (scalars, not sweep lists: one setting,
    # like the Bello benchmark). Override on the CLI via --set tsp_test.KEY=VALUE.
    tsp_test_config: TSPTestConfig = {
        "n_repetitions": 5,                     # repetitions for this method's mean curve; decoupled from the global n_repetitions (like the Bello benchmark, and matching its 5 reps so the comparison is like-for-like). Disk reuse tops up only the shortfall.
        "steps": 2000,                          # gradient updates per repetition (Bello uses 15000: POMO needs a fraction of the budget to dominate the curve)
        "batch": 64,                            # instance copies per update; trajectories per update = batch * pomo_size (64*12 = 768 vs Bello's 512, but with no critic and no LSTM recurrence each one is much cheaper)
        "pomo_size": None,                      # POMO rollouts per instance copy, each forced to start from a different city; None -> n_cities (full multistart). The rollout-group mean is the REINFORCE baseline - no critic network at all.
        "embed": 128,                           # embedding / model dimension (same capacity class as Bello's 128)
        "n_heads": 8,                           # attention heads (must divide embed)
        "n_layers": 3,                          # encoder self-attention layers (POMO paper uses 6 for TSP100; 3 is ample for small instances and faster)
        "ff_hidden": 512,                       # encoder feed-forward hidden size
        "clip_logits": 10.0,                    # C: pointer logits clipped to [-C, C] via C*tanh(.) - Bello's exploration trick, retained by Kool/POMO
        "softmax_T": 1.0,                       # pointer softmax temperature
        "lr": 3e-4,                             # Adam learning rate (canonical POMO is 1e-4 for TSP100 from scratch; 3e-4 converges faster on small fixed instances and stays stable thanks to the shared-baseline advantage)
        "weight_decay": 1e-6,                   # Adam weight decay (canonical POMO value)
        "grad_norm_clip": 1.0,                  # L2 gradient-norm clip; 0 disables
        "is_lr_decay": False,                   # POMO trains at constant lr; enable + tune the two knobs below for very long runs
        "lr_decay": 0.96,
        "lr_decay_step": 500,
        "curve_label": "TSP-Test (POMO-AM)",    # legend label for this method's curve (cosmetic; never participates in disk matching)
    }


    # ── Configuration overrides ──────────────────────────────────────────────
    # One registry of the editable sections, shared by the validation pass, the
    # programmatic ``overrides`` and the command-line overrides below.
    _config_targets: Dict[str, Any] = {
        "global_config": global_config,
        "a2c_config": a2c_config,
        "ppo_config": ppo_config,
        "sac_config": sac_config,
        "tsp_test_config": tsp_test_config,
        "included_algorithms": included_algorithms,
    }

    # Validate the in-script config first, so a mistyped key fails fast right
    # here rather than silently doing nothing deep inside the run.
    for _section, _values in _config_targets.items():
        _validate_config_keys(_section, _values)

    # Programmatic per-ablation overrides (used by the Experiment_*.py forks).
    if overrides:
        for _section, _values in overrides.items():
            if _section not in _config_targets:
                raise KeyError(
                    f"Unknown overrides section '{_section}'. "
                    f"Valid sections: {sorted(_config_targets)}."
                )
            if not isinstance(_values, dict):
                raise TypeError(f"overrides['{_section}'] must be a dict.")
            _validate_config_keys(_section, _values)
            _config_targets[_section].update(_values)

    # ── Command-line overrides ───────────────────────────────────────────────
    # Built AFTER the config dicts above so anything passed on the command line
    # WINS over the values hard-coded in this script (and over the programmatic
    # ``overrides`` applied just above). Every flag defaults to None, so running
    # with no flags reproduces the in-script behaviour exactly. parse_known_args
    # keeps this harmless when Test_TSP is driven by the Ablation Scripts forks
    # (their argv carries no flags; any unrelated args are ignored, not errored).
    # List-valued flags define a sweep axis, e.g. --ppo-gamma 0.9 0.99 1.0.
    cli = argparse.ArgumentParser(
        prog="Experiment.py",
        description=(
            "Run the TSP A2C/PPO/SAC experiment. Each flag overrides the matching "
            "value defined inside Experiment.py; omit it to keep the in-script "
            "value. List-valued flags define a sweep axis."
        ),
    )

    grp_g = cli.add_argument_group("global_config overrides")
    grp_g.add_argument("--n-timesteps", type=float, dest="global__n_timesteps",
                       help="training steps per repetition (e.g. 5e5)")
    grp_g.add_argument("--n-repetitions", type=int, dest="global__n_repetitions")
    grp_g.add_argument("--eval-interval", type=int, dest="global__eval_interval")
    grp_g.add_argument("--n-eval-episodes", type=int, dest="global__n_eval_episodes")
    grp_g.add_argument("--base-seed", type=int, dest="global__base_seed")
    grp_g.add_argument("--baseline-model", type=str, dest="global__baseline_model",
                       help="A2C | PPO | SAC | None")
    grp_g.add_argument("--n-use-trained-model", type=int, dest="global__n_use_trained_model")
    grp_g.add_argument("--action-selection-method", type=str,
                       dest="global__action_selection_method", help="e.g. sample | argmax")
    grp_g.add_argument("--min-unused-cpu-cores", type=int, dest="global__MIN_UNUSED_CPU_CORES")
    grp_g.add_argument("--curve-plot", type=_str2bool, nargs="?", const=True,
                       dest="global__curve_plot")
    grp_g.add_argument("--animation-plot", type=_str2bool, nargs="?", const=True,
                       dest="global__animation_plot")
    grp_g.add_argument("--use-existing-disk-data", type=_str2bool, nargs="?", const=True,
                       dest="global__use_existing_disk_data")

    cli.add_argument("--algos", nargs="+", choices=["A2C", "PPO", "SAC", "TSP_TEST"],
                     dest="included_algos",
                     help="enable exactly these algorithms (the rest are disabled)")

    # Common per-algorithm sweep knobs (one group each).
    for _algo in ("a2c", "ppo", "sac"):
        grp = cli.add_argument_group(f"{_algo}_config overrides")
        grp.add_argument(f"--{_algo}-gamma", nargs="+", type=float, dest=f"{_algo}__gamma")
        grp.add_argument(f"--{_algo}-actor-lr", nargs="+", type=float, dest=f"{_algo}__actor_lr")
        grp.add_argument(f"--{_algo}-critic-lr", nargs="+", type=float, dest=f"{_algo}__critic_lr")

    # Algorithm-specific scalar sweep knobs.
    cli.add_argument("--a2c-tn-step", nargs="+", type=int, dest="a2c__TN_step")
    cli.add_argument("--ppo-gae-lambda", nargs="+", type=float, dest="ppo__gae_lambda")
    cli.add_argument("--ppo-clip-epsilon", nargs="+", type=float, dest="ppo__clip_epsilon")
    cli.add_argument("--ppo-n-epochs", nargs="+", type=int, dest="ppo__n_epochs")
    cli.add_argument("--ppo-entropy-coef", nargs="+", type=float, dest="ppo__entropy_coef")
    cli.add_argument("--ppo-value-coef", nargs="+", type=float, dest="ppo__value_coef")
    cli.add_argument("--ppo-rollout-steps", nargs="+", type=int, dest="ppo__rollout_steps")
    cli.add_argument("--sac-tn-step", nargs="+", type=int, dest="sac__TN_step")
    cli.add_argument("--sac-alpha", nargs="+", type=float, dest="sac__alpha")
    cli.add_argument("--sac-alpha-lr", nargs="+", type=float, dest="sac__alpha_lr")
    cli.add_argument("--sac-tau", nargs="+", type=float, dest="sac__tau")
    cli.add_argument("--sac-target-entropy-ratio", nargs="+", type=float,
                     dest="sac__target_entropy_ratio")

    # Generic escape hatch for any key not exposed above, including nested keys
    # and structured values, e.g.:
    #   --set ppo.actor_hidden_nn="[[256,256]]"
    #   --set global.checkpoints.use_saved_disk_networks_checkpoints=True
    cli.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                     dest="cli_set_items",
                     help="override an arbitrary (optionally nested) config value")

    cli_args, _ = cli.parse_known_args()

    # Fold the populated (non-None) flags into the section dicts. dest names use
    # the "<section>__<key>" convention so the target section is unambiguous.
    cli_overrides: Dict[str, dict] = {}
    for _dest, _value in vars(cli_args).items():
        if _value is None or _dest in ("included_algos", "cli_set_items"):
            continue
        _prefix, _key = _dest.split("__", 1)
        cli_overrides.setdefault(_CLI_SECTION_ALIASES[_prefix], {})[_key] = _value

    if cli_args.included_algos is not None:
        cli_overrides.setdefault("included_algorithms", {}).update(
            {name: (name in cli_args.included_algos)
             for name in ("A2C", "PPO", "SAC", "TSP_TEST")}
        )

    for _section, _values in cli_overrides.items():
        _validate_config_keys(_section, _values)
        _config_targets[_section].update(_values)

    # Generic / nested --set overrides applied last (highest precedence).
    for _item in cli_args.cli_set_items:
        _apply_set_item(_config_targets, _item)

    # Final safety re-validation after every override layer.
    for _section, _values in _config_targets.items():
        _validate_config_keys(_section, _values)

    os.environ["MIN_UNUSED_CPU_CORES"] = str(int(global_config.get("MIN_UNUSED_CPU_CORES", 2)))

    _baseline_model = global_config.get("baseline_model")
    if _baseline_model:
        _baseline_upper = str(_baseline_model).upper()
        if _baseline_upper == "NONE":
            pass
        elif _baseline_upper == "BELLO":
            # Pointer-Network baseline; handled as a dedicated path inside
            # run_selected_experiments, not as an included A2C/SAC/PPO experiment.
            pass
        elif _baseline_upper in included_algorithms:
            included_algorithms[_baseline_upper] = True
        else:
            raise ValueError(
                f"Unknown baseline_model '{_baseline_model}'. "
                "Use one of: A2C, SAC, PPO, TSP_TEST, Bello, NONE (or None)."
            )

    algo_order = ["A2C", "SAC", "PPO", "TSP_TEST"]
    experiments = [name for name in algo_order if included_algorithms.get(name, False)]

    stochastic_duration_matrix_mean = run_selected_experiments(
        experiments,
        global_config=global_config,
        a2c_config=a2c_config,
        sac_config=sac_config,
        ppo_config=ppo_config,
        tsp_test_config=tsp_test_config,
    )
    # None when no A2C/SAC/PPO experiment ran (e.g. --algos TSP_TEST only).
    if stochastic_duration_matrix_mean is not None:
        print("expectation is a zero matrix:\n", stochastic_duration_matrix_mean - dp_env.expected_stochastic_duration_matrix, "\n\n")

    _baseline_for_eval = global_config.get("baseline_model")
    _baseline_upper_for_eval = str(_baseline_for_eval).upper() if _baseline_for_eval else None
    # The equivalency-table append below evaluates a saved Policy_NN actor; the
    # Bello baseline uses a Pointer Network and TSP_TEST an Attention Model
    # (their own curves + reconstructed-route plots are produced inside
    # run_selected_experiments), so skip them here.
    if result is not None and _baseline_upper_for_eval and _baseline_upper_for_eval not in {"NONE", "BELLO", "TSP_TEST"}:
        algo_cfg_lookup = {
            "A2C": a2c_config,
            "SAC": sac_config,
            "PPO": ppo_config,
        }
        algo_cfg_for_eval = algo_cfg_lookup.get(_baseline_upper_for_eval) or {}
        actor_hidden_nn_cfg = algo_cfg_for_eval.get("actor_hidden_nn", [[64, 64]])
        if isinstance(actor_hidden_nn_cfg, (list, tuple)) and len(actor_hidden_nn_cfg) > 0 and isinstance(actor_hidden_nn_cfg[0], (list, tuple, np.ndarray)):
            actor_hidden_nn_eval = actor_hidden_nn_cfg[0]
        else:
            actor_hidden_nn_eval = actor_hidden_nn_cfg

        n_use_trained_model = int(global_config.get("n_use_trained_model", 10))
        action_selection_method = str(global_config.get("action_selection_method", "argmax"))
        reseed_seed = global_config.get("trained_model_reseed_seed", None)

        model_rows, model_best_tour, eval_summary = evaluate_trained_model_on_matrices(
            dp_env=dp_env,
            algo_name=_baseline_upper_for_eval,
            actor_hidden_nn=actor_hidden_nn_eval,
            n_use_trained_model=n_use_trained_model,
            action_selection_method=action_selection_method,
            reseed_seed=reseed_seed,
        )

        ed = result["equivalency_data"]
        fill_dp_route_match(model_rows, ed["det_optimal_tours"], ed["avg_tours"])

        dp_rows = _build_route_equivalency_rows(
            ed["det_optimal_tours"],
            ed["avg_tours"],
            ed["det_matrix"],
            ed["avg_matrix"],
            ed["avg_opt_cost"],
        )

        print(f"\n{'=' * 80}")
        print(
            f"  Equivalency table with appended {_baseline_upper_for_eval} model evaluations "
            f"(n_use_trained_model={n_use_trained_model}, action_selection_method='{action_selection_method}')"
        )
        print(f"{'=' * 80}")
        for label, mean, std, n_valid in eval_summary:
            print(f"  {label}: mean={mean:.4f}, std={std:.4f}, valid_episodes={n_valid}/{n_use_trained_model}")
        print()
        for line in _format_route_equivalency_table(
            dp_rows,
            max_equivalency_table_rows=ed.get("max_equivalency_table_rows", 100),
            show_cost_chain=True,
            model_rows=model_rows,
            model_best_tour=model_best_tour,
            expected_matrix=ed["avg_matrix"],
        ):
            print(line)


if __name__ == "__main__":
    input = {
        "duration_matrix": [
            [0, 10, 12, 19],
            [10, 0, 5, 7],
            [12, 5, 0, 9],
            [19, 7, 9, 0],
        ],
        "potential_uncertainty_matrix": [
            [0, 2.0, 4.6, 9.0],
            [3.1, 0, 6.0, 1.3],
            [2.5, 4.8, 0, 3.2],
            [9.0, 6.4, 2.1, 0],
        ],
        "uncertainty_inclusion_matrix": [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ],
    }
    # print_uncertainty_comparison("Assignment Example (4 cities)", result)

    Test_TSP()  # give input of the 3 matrices above to solve the problem with this input,
                     # or pass None to load the trio (by id/dimension) from disk in Test_TSP()
