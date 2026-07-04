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
import numpy as np

from .Helper_legend import _build_legend_parts, _resolve_legend_flags


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
    gammas, actor_learning_rates, actor_architectures, legend = _parse_pg_config(cfg)
    critic_learning_rates = np.atleast_1d(np.asarray(cfg.get("critic_lr", np.array([0.001])), dtype=np.float32))
    raw_critic_nn = cfg.get("critic_hidden_nn", [[64, 64]])
    if isinstance(raw_critic_nn, np.ndarray) and raw_critic_nn.ndim == 1:
        critic_architectures = [raw_critic_nn]
    elif isinstance(raw_critic_nn, list) and len(raw_critic_nn) > 0 and not isinstance(raw_critic_nn[0], (list, np.ndarray)):
        critic_architectures = [np.asarray(raw_critic_nn, dtype=np.int32)]
    else:
        critic_architectures = [np.asarray(arch, dtype=np.int32) for arch in raw_critic_nn]
    gae_lambdas = np.atleast_1d(np.asarray(cfg.get("gae_lambda", np.array([0.95])), dtype=np.float32))
    clip_epsilons = np.atleast_1d(np.asarray(cfg.get("clip_epsilon", np.array([0.2])), dtype=np.float32))
    n_epochs_sweep = np.atleast_1d(np.asarray(cfg.get("n_epochs", np.array([4])), dtype=np.int32))
    entropy_coefs = np.atleast_1d(np.asarray(cfg.get("entropy_coef", np.array([0.01])), dtype=np.float32))
    value_coefs = np.atleast_1d(np.asarray(cfg.get("value_coef", np.array([0.5])), dtype=np.float32))
    rollout_steps_sweep = np.atleast_1d(np.asarray(cfg.get("rollout_steps", np.array([2048])), dtype=np.int64))
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
                        for gae_lambda_val in gae_lambdas:
                            gae_lambda_val = float(gae_lambda_val)
                            for clip_eps_val in clip_epsilons:
                                clip_eps_val = float(clip_eps_val)
                                for n_epochs_val in n_epochs_sweep:
                                    n_epochs_val = int(n_epochs_val)
                                    for ent_coef_val in entropy_coefs:
                                        ent_coef_val = float(ent_coef_val)
                                        for val_coef_val in value_coefs:
                                            val_coef_val = float(val_coef_val)
                                            for full_ep_val in full_episode_updates_sweep:
                                                full_ep_bool = bool(full_ep_val)
                                                for rollout_steps_val in rollout_steps_sweep:
                                                    rollout_steps_val = int(rollout_steps_val)
                                                    iter_cfg = {
                                                        **cfg,
                                                        "gamma": gamma_val,
                                                        "actor_lr": actor_lr_val,
                                                        "actor_hidden_nn": actor_nn,
                                                        "critic_lr": critic_lr_val,
                                                        "critic_hidden_nn": critic_nn,
                                                        "gae_lambda": gae_lambda_val,
                                                        "clip_epsilon": clip_eps_val,
                                                        "n_epochs": n_epochs_val,
                                                        "entropy_coef": ent_coef_val,
                                                        "value_coef": val_coef_val,
                                                        "rollout_steps": rollout_steps_val,
                                                        "FULL_EPISODE_UPDATES": full_ep_bool,
                                                    }
                                                    label_parts = ["PPO"] + _build_legend_parts(legend, iter_cfg)
                                                    curve_label = ", ".join(label_parts)
                                                    setting_jobs.append({
                                                        "curve_label": curve_label,
                                                        "method": "ppo",
                                                        "kwargs": dict(
                                                            method="ppo",
                                                            n_repetitions=n_repetitions,
                                                            n_timesteps=n_timesteps,
                                                            eval_interval=eval_interval,
                                                            max_episode_length=max_episode_length,
                                                            actor_lr=actor_lr_val,
                                                            critic_lr=critic_lr_val,
                                                            gamma=gamma_val,
                                                            actor_hidden_nn=actor_nn,
                                                            critic_hidden_nn=critic_nn,
                                                            gae_lambda=gae_lambda_val,
                                                            clip_epsilon=clip_eps_val,
                                                            n_epochs=n_epochs_val,
                                                            entropy_coef=ent_coef_val,
                                                            value_coef=val_coef_val,
                                                            full_episode_updates=full_ep_bool,
                                                            rollout_steps=rollout_steps_val,
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
                                                            "gae_lambda": gae_lambda_val,
                                                            "clip_epsilon": clip_eps_val,
                                                            "n_epochs": n_epochs_val,
                                                            "entropy_coef": ent_coef_val,
                                                            "value_coef": val_coef_val,
                                                            "rollout_steps": rollout_steps_val,
                                                            "FULL_EPISODE_UPDATES": full_ep_bool,
                                                        },
                                                    })
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
