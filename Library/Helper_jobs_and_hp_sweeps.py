"""
Helper_jobs_and_hp_sweeps.py - Algorithm-specific setting-job builders, hyperparameter sweeps, and filename helpers.

Contents
--------
build_algorithm_filename - Workbook filename stem (uppercased algo name).
_build_algo_filename     - Per-algorithm filename builder.
_parse_pg_config         - Parse policy-gradient config into sweepable arrays.
_build_a2c_jobs          - Build A2C setting jobs.
_build_ppo_jobs          - Build PPO setting jobs.
_build_sac_jobs          - Build SAC setting jobs.
"""
import itertools

import numpy as np

from .Helper_legend import _build_legend_parts, _resolve_legend_flags


# PPO knobs added for the representation / baseline work. A setting that leaves
# one of these at its legacy value is behaviourally identical to the pre-change
# PPO, so the key is omitted from the disk-matching hyperparameters below and
# such runs keep reusing the workbooks and checkpoints they already have. Any
# non-default value is recorded and therefore gets its own row.
_PPO_LEGACY_DEFAULTS = {
    "observation_mode": "legacy",
    "actor_arch": "mlp",
    "advantage_mode": "gae",
    "group_size": 8,
    "eval_n_starts": 1,
    "pointer_embed": 128,
    "pointer_n_heads": 8,
    "pointer_n_layers": 2,
    "pointer_ff_hidden": 256,
}


def _sweep_axis(cfg, key, default, cast=None):
    """Normalize one config entry into a list of values to sweep.

    Accepts a bare scalar, a list/tuple, or a numpy array, so a config may write
    either ``"actor_arch": "pointer"`` or ``"actor_arch": ["mlp", "pointer"]``.
    """
    raw = cfg.get(key, default)
    if isinstance(raw, (str, bytes)):
        values = [raw]
    elif isinstance(raw, np.ndarray):
        values = np.atleast_1d(raw).tolist()
    elif isinstance(raw, (list, tuple)):
        values = list(raw)
    else:
        values = [raw]
    if not values:
        values = default if isinstance(default, (list, tuple)) else [default]
    return [cast(v) for v in values] if cast is not None else values


def _parse_hidden_nn(raw):
    """Normalize a hidden-layer config into a list of int32 width arrays."""
    if isinstance(raw, np.ndarray) and raw.ndim == 1:
        return [raw]
    if isinstance(raw, list) and len(raw) > 0 and not isinstance(raw[0], (list, np.ndarray)):
        return [np.asarray(raw, dtype=np.int32)]
    return [np.asarray(arch, dtype=np.int32) for arch in raw]


def build_algorithm_filename(algo_name: str) -> str:
    """Return the workbook filename stem for an algorithm."""
    return str(algo_name).upper()


def _build_algo_filename(algo_name):
    """Build the workbook filename stem for a single algorithm."""
    return build_algorithm_filename(algo_name)


def _parse_pg_config(cfg):
    """Parse policy-gradient config into sweepable arrays (with backward compat)."""
    gammas = np.atleast_1d(np.asarray(cfg.get("gamma", np.array([0.99])), dtype=np.float32))
    learning_rates = np.atleast_1d(np.asarray(cfg.get("actor_lr", np.array([0.001])), dtype=np.float32))

    raw_nn = cfg.get("actor_hidden_nn", [[32, 32]])
    if isinstance(raw_nn, np.ndarray) and raw_nn.ndim == 1:
        nn_architectures = [raw_nn]
    elif isinstance(raw_nn, list) and len(raw_nn) > 0 and not isinstance(raw_nn[0], (list, np.ndarray)):
        nn_architectures = [np.asarray(raw_nn, dtype=np.int32)]
    else:
        nn_architectures = [np.asarray(arch, dtype=np.int32) for arch in raw_nn]

    legend = _resolve_legend_flags(cfg)
    return gammas, learning_rates, nn_architectures, legend


def _build_a2c_jobs(*, algo_config, n_repetitions, n_timesteps, eval_interval,
                    max_episode_length, base_seed):
    cfg = algo_config
    gammas, actor_learning_rates, actor_architectures, legend = _parse_pg_config(cfg)
    critic_learning_rates = np.atleast_1d(np.asarray(cfg.get("critic_lr", np.array([0.001])), dtype=np.float32))
    raw_critic_nn = cfg.get("critic_hidden_nn", [[64, 64]])
    if isinstance(raw_critic_nn, np.ndarray) and raw_critic_nn.ndim == 1:
        critic_architectures = [raw_critic_nn]
    elif isinstance(raw_critic_nn, list) and len(raw_critic_nn) > 0 and not isinstance(raw_critic_nn[0], (list, np.ndarray)):
        critic_architectures = [np.asarray(raw_critic_nn, dtype=np.int32)]
    else:
        critic_architectures = [np.asarray(arch, dtype=np.int32) for arch in raw_critic_nn]
    TN_steps = np.atleast_1d(np.asarray(cfg.get("TN_step", np.array([10])), dtype=np.int32))
    full_episode_updates_sweep = np.atleast_1d(np.asarray(cfg.get("FULL_EPISODE_UPDATES", np.array([True]))))

    setting_jobs = []
    for gamma_val in gammas:
        gamma_val = float(gamma_val)
        for actor_nn in actor_architectures:
            actor_nn = np.asarray(actor_nn, dtype=np.int32)
            for actor_lr_val in actor_learning_rates:
                actor_lr_val = float(actor_lr_val)
                for critic_nn in critic_architectures:
                    critic_nn = np.asarray(critic_nn, dtype=np.int32)
                    for critic_lr_val in critic_learning_rates:
                        critic_lr_val = float(critic_lr_val)
                        for tn_step in TN_steps:
                            tn_step = int(tn_step)
                            for full_ep_val in full_episode_updates_sweep:
                                full_ep_bool = bool(full_ep_val)
                                iter_cfg = {
                                    **cfg,
                                    "gamma": gamma_val,
                                    "actor_lr": actor_lr_val,
                                    "actor_hidden_nn": actor_nn,
                                    "critic_lr": critic_lr_val,
                                    "critic_hidden_nn": critic_nn,
                                    "TN_step": tn_step,
                                    "FULL_EPISODE_UPDATES": full_ep_bool,
                                }
                                label_parts = ["A2C"] + _build_legend_parts(legend, iter_cfg)
                                curve_label = ", ".join(label_parts)
                                setting_jobs.append({
                                    "curve_label": curve_label,
                                    "method": "a2c",
                                    "kwargs": dict(
                                        method="a2c",
                                        n_repetitions=n_repetitions,
                                        n_timesteps=n_timesteps,
                                        eval_interval=eval_interval,
                                        max_episode_length=max_episode_length,
                                        actor_lr=actor_lr_val,
                                        critic_lr=critic_lr_val,
                                        gamma=gamma_val,
                                        actor_hidden_nn=actor_nn,
                                        critic_hidden_nn=critic_nn,
                                        TN_step=tn_step,
                                        full_episode_updates=full_ep_bool,
                                        base_seed=base_seed,
                                        plot_smoothing_window=1,
                                    ),
                                    "hyperparams": {
                                        "n_repetitions": n_repetitions,
                                        "n_timesteps": n_timesteps,
                                        "eval_interval": eval_interval,
                                        "max_episode_length": max_episode_length,
                                        "actor_lr": actor_lr_val,
                                        "critic_lr": critic_lr_val,
                                        "gamma": gamma_val,
                                        "actor_hidden_nn": str(actor_nn.tolist()),
                                        "critic_hidden_nn": str(critic_nn.tolist()),
                                        "TN_step": tn_step,
                                        "FULL_EPISODE_UPDATES": full_ep_bool,
                                    },
                                })
    return setting_jobs


def _build_ppo_jobs(*, algo_config, n_repetitions, n_timesteps, eval_interval,
                    max_episode_length, base_seed):
    cfg = algo_config
    _gammas, _actor_lrs, _actor_nns, legend = _parse_pg_config(cfg)

    # Sweep axes, in the order the nested loops used to visit them (the last
    # axis varies fastest, exactly as before) so an unchanged config still
    # produces its settings in the same order -- checkpoint suffixes S1/S2/...
    # and previously saved workbooks stay aligned. The representation/baseline
    # axes are appended at the end for the same reason.
    axes = {
        "gamma": [float(v) for v in _gammas],
        "actor_hidden_nn": [np.asarray(a, dtype=np.int32) for a in _actor_nns],
        "actor_lr": [float(v) for v in _actor_lrs],
        "critic_hidden_nn": _parse_hidden_nn(cfg.get("critic_hidden_nn", [[64, 64]])),
        "critic_lr": _sweep_axis(cfg, "critic_lr", [0.001], float),
        "gae_lambda": _sweep_axis(cfg, "gae_lambda", [0.95], float),
        "clip_epsilon": _sweep_axis(cfg, "clip_epsilon", [0.2], float),
        "n_epochs": _sweep_axis(cfg, "n_epochs", [4], int),
        "entropy_coef": _sweep_axis(cfg, "entropy_coef", [0.01], float),
        "value_coef": _sweep_axis(cfg, "value_coef", [0.5], float),
        "FULL_EPISODE_UPDATES": _sweep_axis(cfg, "FULL_EPISODE_UPDATES", [True], bool),
        "rollout_steps": _sweep_axis(cfg, "rollout_steps", [2048], int),
        "observation_mode": _sweep_axis(cfg, "observation_mode", ["legacy"], str),
        "actor_arch": _sweep_axis(cfg, "actor_arch", ["mlp"], str),
        "advantage_mode": _sweep_axis(cfg, "advantage_mode", ["gae"], str),
        "group_size": _sweep_axis(cfg, "group_size", [8], int),
        "eval_n_starts": _sweep_axis(cfg, "eval_n_starts", [1], int),
        "pointer_embed": _sweep_axis(cfg, "pointer_embed", [128], int),
        "pointer_n_heads": _sweep_axis(cfg, "pointer_n_heads", [8], int),
        "pointer_n_layers": _sweep_axis(cfg, "pointer_n_layers", [2], int),
        "pointer_ff_hidden": _sweep_axis(cfg, "pointer_ff_hidden", [256], int),
    }

    setting_jobs = []
    skipped_pointer_legacy = 0
    seen_effective = set()
    for combo in itertools.product(*axes.values()):
        v = dict(zip(axes.keys(), combo))

        # The pointer actor reads the per-city feature block, which only the
        # enriched observation has. Sweeping both axes is the natural way to ask
        # for "legacy+mlp vs enriched+mlp vs enriched+pointer", so the impossible
        # corner is dropped rather than raised on.
        if v["actor_arch"] == "pointer" and v["observation_mode"] != "enriched":
            skipped_pointer_legacy += 1
            continue
        if v["actor_arch"] == "pointer" and v["pointer_embed"] % v["pointer_n_heads"]:
            raise ValueError(
                f"ppo_config: pointer_embed ({v['pointer_embed']}) must be divisible by "
                f"pointer_n_heads ({v['pointer_n_heads']})."
            )

        # Several axes are inert for a given combo (the group loop reads neither
        # the rollout buffer nor GAE; the MLP actor ignores the attention dims).
        # Collapsing them keeps a broad sweep from emitting duplicate curves that
        # would train identically and plot on top of each other.
        inert = set()
        if v["advantage_mode"] == "group":
            inert |= {"rollout_steps", "FULL_EPISODE_UPDATES", "gae_lambda"}
        else:
            inert.add("group_size")
        if v["actor_arch"] != "pointer":
            inert |= {"pointer_embed", "pointer_n_heads", "pointer_n_layers",
                      "pointer_ff_hidden"}
        effective = tuple(
            (k, str(val.tolist()) if isinstance(val, np.ndarray) else val)
            for k, val in v.items() if k not in inert
        )
        if effective in seen_effective:
            continue
        seen_effective.add(effective)

        iter_cfg = {**cfg, **v}
        curve_label = ", ".join(["PPO"] + _build_legend_parts(legend, iter_cfg))

        hyperparams = {
            "n_repetitions": n_repetitions,
            "n_timesteps": n_timesteps,
            "eval_interval": eval_interval,
            "max_episode_length": max_episode_length,
            "actor_lr": v["actor_lr"],
            "critic_lr": v["critic_lr"],
            "gamma": v["gamma"],
            "actor_hidden_nn": str(v["actor_hidden_nn"].tolist()),
            "critic_hidden_nn": str(v["critic_hidden_nn"].tolist()),
            "gae_lambda": v["gae_lambda"],
            "clip_epsilon": v["clip_epsilon"],
            "n_epochs": v["n_epochs"],
            "entropy_coef": v["entropy_coef"],
            "value_coef": v["value_coef"],
            "rollout_steps": v["rollout_steps"],
            "FULL_EPISODE_UPDATES": v["FULL_EPISODE_UPDATES"],
        }
        # Only non-legacy values of the new knobs join the disk-matching key.
        hyperparams.update({
            key: v[key] for key, legacy in _PPO_LEGACY_DEFAULTS.items()
            if v[key] != legacy
        })

        setting_jobs.append({
            "curve_label": curve_label,
            "method": "ppo",
            "kwargs": dict(
                method="ppo",
                n_repetitions=n_repetitions,
                n_timesteps=n_timesteps,
                eval_interval=eval_interval,
                max_episode_length=max_episode_length,
                actor_lr=v["actor_lr"],
                critic_lr=v["critic_lr"],
                gamma=v["gamma"],
                actor_hidden_nn=v["actor_hidden_nn"],
                critic_hidden_nn=v["critic_hidden_nn"],
                gae_lambda=v["gae_lambda"],
                clip_epsilon=v["clip_epsilon"],
                n_epochs=v["n_epochs"],
                entropy_coef=v["entropy_coef"],
                value_coef=v["value_coef"],
                full_episode_updates=v["FULL_EPISODE_UPDATES"],
                rollout_steps=v["rollout_steps"],
                observation_mode=v["observation_mode"],
                actor_arch=v["actor_arch"],
                advantage_mode=v["advantage_mode"],
                group_size=v["group_size"],
                eval_n_starts=v["eval_n_starts"],
                pointer_embed=v["pointer_embed"],
                pointer_n_heads=v["pointer_n_heads"],
                pointer_n_layers=v["pointer_n_layers"],
                pointer_ff_hidden=v["pointer_ff_hidden"],
                base_seed=base_seed,
                plot_smoothing_window=1,
            ),
            "hyperparams": hyperparams,
        })

    if skipped_pointer_legacy:
        print(
            f"[PPO] Skipped {skipped_pointer_legacy} setting(s) combining "
            "actor_arch='pointer' with observation_mode='legacy': the pointer "
            "actor needs the enriched observation's per-city features."
        )
    return setting_jobs


def _build_sac_jobs(*, algo_config, n_repetitions, n_timesteps, eval_interval,
                    max_episode_length, base_seed):
    cfg = algo_config
    gammas, actor_learning_rates, actor_architectures, legend = _parse_pg_config(cfg)
    critic_learning_rates = np.atleast_1d(np.asarray(cfg.get("critic_lr", np.array([0.001])), dtype=np.float32))
    raw_critic_nn = cfg.get("critic_hidden_nn", [[64, 64]])
    if isinstance(raw_critic_nn, np.ndarray) and raw_critic_nn.ndim == 1:
        critic_architectures = [raw_critic_nn]
    elif isinstance(raw_critic_nn, list) and len(raw_critic_nn) > 0 and not isinstance(raw_critic_nn[0], (list, np.ndarray)):
        critic_architectures = [np.asarray(raw_critic_nn, dtype=np.int32)]
    else:
        critic_architectures = [np.asarray(arch, dtype=np.int32) for arch in raw_critic_nn]
    TN_steps = np.atleast_1d(np.asarray(cfg.get("TN_step", np.array([10])), dtype=np.int32))
    alphas = np.atleast_1d(np.asarray(cfg.get("alpha", np.array([0.2])), dtype=np.float32))
    alpha_lrs = np.atleast_1d(np.asarray(cfg.get("alpha_lr", np.array([0.001])), dtype=np.float32))
    auto_tune_alphas = np.atleast_1d(np.asarray(cfg.get("auto_tune_alpha", np.array([True]))))
    target_entropy_ratios = np.atleast_1d(np.asarray(cfg.get("target_entropy_ratio", np.array([0.98])), dtype=np.float32))
    taus = np.atleast_1d(np.asarray(cfg.get("tau", np.array([0.005])), dtype=np.float32))
    full_episode_updates_sweep = np.atleast_1d(np.asarray(cfg.get("FULL_EPISODE_UPDATES", np.array([True]))))

    setting_jobs = []
    for gamma_val in gammas:
        gamma_val = float(gamma_val)
        for actor_nn in actor_architectures:
            actor_nn = np.asarray(actor_nn, dtype=np.int32)
            for actor_lr_val in actor_learning_rates:
                actor_lr_val = float(actor_lr_val)
                for critic_nn in critic_architectures:
                    critic_nn = np.asarray(critic_nn, dtype=np.int32)
                    for critic_lr_val in critic_learning_rates:
                        critic_lr_val = float(critic_lr_val)
                        for tn_step in TN_steps:
                            tn_step = int(tn_step)
                            for alpha_val in alphas:
                                alpha_val = float(alpha_val)
                                for alpha_lr_val in alpha_lrs:
                                    alpha_lr_val = float(alpha_lr_val)
                                    for auto_tune_val in auto_tune_alphas:
                                        auto_tune_bool = bool(auto_tune_val)
                                        for target_entropy_ratio_val in target_entropy_ratios:
                                            target_entropy_ratio_val = float(target_entropy_ratio_val)
                                            for tau_val in taus:
                                                tau_val = float(tau_val)
                                                for full_ep_val in full_episode_updates_sweep:
                                                    full_ep_bool = bool(full_ep_val)
                                                    iter_cfg = {
                                                        **cfg,
                                                        "gamma": gamma_val,
                                                        "actor_lr": actor_lr_val,
                                                        "actor_hidden_nn": actor_nn,
                                                        "critic_lr": critic_lr_val,
                                                        "critic_hidden_nn": critic_nn,
                                                        "TN_step": tn_step,
                                                        "alpha": alpha_val,
                                                        "alpha_lr": alpha_lr_val,
                                                        "auto_tune_alpha": auto_tune_bool,
                                                        "target_entropy_ratio": target_entropy_ratio_val,
                                                        "tau": tau_val,
                                                        "FULL_EPISODE_UPDATES": full_ep_bool,
                                                    }
                                                    label_parts = ["SAC"] + _build_legend_parts(legend, iter_cfg)
                                                    curve_label = ", ".join(label_parts)
                                                    setting_jobs.append({
                                                        "curve_label": curve_label,
                                                        "method": "sac",
                                                        "kwargs": dict(
                                                            method="sac",
                                                            n_repetitions=n_repetitions,
                                                            n_timesteps=n_timesteps,
                                                            eval_interval=eval_interval,
                                                            max_episode_length=max_episode_length,
                                                            actor_lr=actor_lr_val,
                                                            critic_lr=critic_lr_val,
                                                            gamma=gamma_val,
                                                            actor_hidden_nn=actor_nn,
                                                            critic_hidden_nn=critic_nn,
                                                            TN_step=tn_step,
                                                            alpha=alpha_val,
                                                            alpha_lr=alpha_lr_val,
                                                            auto_tune_alpha=auto_tune_bool,
                                                            target_entropy_ratio=target_entropy_ratio_val,
                                                            tau=tau_val,
                                                            full_episode_updates=full_ep_bool,
                                                            base_seed=base_seed,
                                                            plot_smoothing_window=1,
                                                        ),
                                                        "hyperparams": {
                                                            "n_repetitions": n_repetitions,
                                                            "n_timesteps": n_timesteps,
                                                            "eval_interval": eval_interval,
                                                            "max_episode_length": max_episode_length,
                                                            "actor_lr": actor_lr_val,
                                                            "critic_lr": critic_lr_val,
                                                            "gamma": gamma_val,
                                                            "actor_hidden_nn": str(actor_nn.tolist()),
                                                            "critic_hidden_nn": str(critic_nn.tolist()),
                                                            "TN_step": tn_step,
                                                            "alpha": alpha_val,
                                                            "alpha_lr": alpha_lr_val,
                                                            "auto_tune_alpha": auto_tune_bool,
                                                            "target_entropy_ratio": target_entropy_ratio_val,
                                                            "tau": tau_val,
                                                            "FULL_EPISODE_UPDATES": full_ep_bool,
                                                        },
                                                    })
    return setting_jobs
