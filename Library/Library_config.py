"""
Library_config.py - Configuration schemas, validation, and CLI-override helpers.

Extracted from ``Experiment.py`` so the experiment entry point keeps only the
``Test_TSP`` driver. Contents:

* The ``TypedDict`` schemas for every config section (``IncludedAlgorithms`` /
  ``A2CConfig`` / ``PPOConfig`` / ``SACConfig`` / ``TSPTestConfig`` /
  ``GlobalConfig``) plus the ``_SECTION_SCHEMAS`` / ``_CLI_SECTION_ALIASES``
  registries built from them.
* ``_validate_config_keys`` - fail-fast key validation against the schemas.
* ``_str2bool`` - argparse helper for boolean flags.
* ``_apply_set_item`` - generic ``SECTION.KEY[.SUBKEY...]=VALUE`` CLI override.
"""

import ast
import argparse
from typing import Any, Dict, List, Optional, TypedDict


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
    BELLO: bool


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
    cleanup_output_files: bool
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
