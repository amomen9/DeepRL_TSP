"""
Helper_math.py - Numerical helpers: smoothing, annealing, Boltzmann sampling,
                 cross-repetition averaging.

Contents
--------
smooth                     - Savitzky-Golay smoothing of a 1-D curve.
linear_anneal              - Linear annealing scheduler.
boltzmann_action           - Sample an action under softmax(probs / temp).
_apply_optional_smoothing  - Apply Savitzky-Golay smoothing only when window is valid.
average_over_repetitions   - Run n repetitions of a policy method and average curves.
"""
import os
import time as _time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager

import numpy as np
from scipy.signal import savgol_filter


def smooth(y, window, poly=2):
    '''
    y: vector to be smoothed
    window: size of the smoothing window '''
    # Lazy import so multiprocessing workers don't import SciPy DLLs at startup
    return savgol_filter(y, window, poly)


def _apply_optional_smoothing(learning_curve, plot_smoothing_window):
    if plot_smoothing_window is None:
        return learning_curve
    max_window = len(learning_curve) if len(learning_curve) % 2 == 1 else len(learning_curve) - 1
    window = min(int(plot_smoothing_window), max_window)
    if window >= 3:
        return smooth(learning_curve, window)
    return learning_curve


def average_over_repetitions(
    env,
    method,
    n_repetitions,
    n_timesteps,
    eval_interval,
    max_episode_length,
    actor_lr,
    gamma,
    actor_hidden_nn=np.array([16, 16]),
    critic_hidden_nn=np.array([64, 64]),
    critic_lr=0.001,
    base_seed=42,
    plot_smoothing_window=None,
    return_raw=False,
    TN_step=10,
):
    """Run ``n_repetitions`` of the given method and return (mean, std, timesteps)."""
    if env is None:
        raise ValueError("env must be provided")
    from .Helper_progress_bar import _create_step_progress_bar
    from .Library_experiment_orchestrator import _run_single_repetition

    returns_over_repetitions = []
    timesteps = None

    cpu_count = os.cpu_count() or 1
    unused_cores = max(0, int(os.environ.get("MIN_UNUSED_CPU_CORES", 2)))
    available_workers = max(1, cpu_count - unused_cores)
    parallel_workers = max(1, min(n_repetitions, available_workers))
    use_parallel = parallel_workers > 1 and n_repetitions > 1

    if use_parallel:
        manager = Manager()
        step_counters = [manager.Value("i", 0) for _ in range(n_repetitions)]
        try:
            with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
                future_to_rep = {}
                for rep in range(n_repetitions):
                    run_seed = base_seed + rep
                    future = executor.submit(
                        _run_single_repetition,
                        env=env,
                        policy_based_method=method,
                        actor_hidden_nn=actor_hidden_nn,
                        critic_hidden_nn=critic_hidden_nn,
                        actor_lr=actor_lr,
                        critic_lr=critic_lr,
                        gamma=gamma,
                        max_episode_length=max_episode_length,
                        n_timesteps=n_timesteps,
                        eval_interval=eval_interval,
                        run_seed=run_seed,
                        rep_index=rep,
                        n_repetitions=n_repetitions,
                        enable_progress_bar=False,
                        shared_step_counter=step_counters[rep],
                        TN_step=TN_step,
                    )
                    future_to_rep[future] = rep

                pbars = [
                    _create_step_progress_bar(
                        total=n_timesteps,
                        desc=f"{method.upper()} Rep {rep + 1}/{n_repetitions}",
                        position=rep,
                        leave=False,
                    )
                    for rep in range(n_repetitions)
                ]

                done_futures = set()
                try:
                    while len(done_futures) < n_repetitions:
                        for rep in range(n_repetitions):
                            current = step_counters[rep].value
                            delta = current - pbars[rep].n
                            if delta > 0:
                                pbars[rep].update(delta)

                        for future in list(future_to_rep):
                            if future not in done_futures and future.done():
                                done_futures.add(future)
                                rep = future_to_rep[future]
                                rep_returns, rep_timesteps = future.result()
                                returns_over_repetitions.append(np.asarray(rep_returns, dtype=np.float32))
                                if timesteps is None:
                                    timesteps = np.asarray(rep_timesteps, dtype=np.int32)
                                remaining = n_timesteps - pbars[rep].n
                                if remaining > 0:
                                    pbars[rep].update(remaining)

                        _time.sleep(0.25)
                finally:
                    for pb in pbars:
                        pb.close()
                    print()
        finally:
            manager.shutdown()
    else:
        for rep in range(n_repetitions):
            run_seed = base_seed + rep
            rep_returns, rep_timesteps = _run_single_repetition(
                env=env,
                policy_based_method=method,
                actor_hidden_nn=actor_hidden_nn,
                critic_hidden_nn=critic_hidden_nn,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                gamma=gamma,
                max_episode_length=max_episode_length,
                n_timesteps=n_timesteps,
                eval_interval=eval_interval,
                run_seed=run_seed,
                rep_index=rep,
                n_repetitions=n_repetitions,
                enable_progress_bar=True,
                TN_step=TN_step,
            )
            returns_over_repetitions.append(np.asarray(rep_returns, dtype=np.float32))
            if timesteps is None:
                timesteps = np.asarray(rep_timesteps, dtype=np.int32)

    min_length = min(len(rep_returns) for rep_returns in returns_over_repetitions)
    returns_over_repetitions = [
        np.asarray(rep_returns[:min_length], dtype=np.float32)
        for rep_returns in returns_over_repetitions
    ]
    if timesteps is not None:
        timesteps = np.asarray(timesteps[:min_length], dtype=np.int32)

    all_returns = np.array(returns_over_repetitions)
    learning_curve = np.mean(all_returns, axis=0)
    learning_curve_std = (
        np.std(all_returns, axis=0, ddof=1)
        if all_returns.shape[0] > 1
        else np.zeros_like(learning_curve)
    )
    learning_curve = _apply_optional_smoothing(learning_curve, plot_smoothing_window)
    learning_curve_std = _apply_optional_smoothing(learning_curve_std, plot_smoothing_window)

    if return_raw:
        raw_returns = np.asarray(returns_over_repetitions, dtype=np.float32)
        return learning_curve, learning_curve_std, timesteps, raw_returns
    return learning_curve, learning_curve_std, timesteps


if __name__ == '__main__':
    import matplotlib.pyplot as visplt

    from .Library_plotting import LearningCurvePlot

    x = np.arange(100)
    y = 0.01 * x + np.random.rand(100) - 0.4
    plot = LearningCurvePlot(title="Test Learning Curve")
    plot.add_curve(x, y, label="method 1")
    plot.add_curve(x, smooth(y, window=35), label="method 1 smoothed")
    plot.save(name="learning_curve_test.png")
    visplt.show()
