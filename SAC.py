import math
import random
from collections import deque

import numpy as np
from tqdm import tqdm


def run_TSP_sac(agent, env, n_timesteps=200000, eval_interval=250,
                truncation_step=None, enable_progress_bar=True,
                progress_bar_desc="TSP SAC Steps",
                progress_bar_position=None,
                shared_step_counter=None,
                eval_n_episodes=1,
                reseed_noise_seed=None,
                full_episode_updates=True,
                replay_capacity=100000,
                batch_size=64,
                learning_starts=None,
                updates_per_step=1):
    """SAC step-based training loop for the TSP DP_Table environment.

    Episodes terminate when all cities are visited.

    full_episode_updates=True: the agent performs one discrete-SAC update
    (twin Q-critic + entropy-regularized actor) on the full trajectory at the
    end of each episode. This is the original (pre-FULL_EPISODE_UPDATES)
    behaviour.
    full_episode_updates=False: the conventional off-policy SAC loop. Every
    transition is pushed into a replay buffer of capacity ``replay_capacity``;
    once at least ``learning_starts`` transitions have been collected, each env
    step draws ``updates_per_step`` random minibatches of ``batch_size`` and
    runs a discrete-SAC gradient update on each (the agent's update is already
    off-policy: soft Bellman targets via the twin target critics).

    eval_returns are recorded as GREEEDY evaluation returns (argmax policy),
    not the last sampled training episode return (which can be much noisier
    and can bias algorithm comparisons).
    """
    if reseed_noise_seed is not None:
        env.reseed_noise(reseed_noise_seed)  # Deterministic reseed when caller provides a seed

    if truncation_step is None:
        truncation_step = int(env.n + 1)

    if not full_episode_updates:
        return _run_conventional_sac(
            agent, env,
            n_timesteps=n_timesteps,
            eval_interval=eval_interval,
            truncation_step=truncation_step,
            enable_progress_bar=enable_progress_bar,
            progress_bar_desc=progress_bar_desc,
            progress_bar_position=progress_bar_position,
            shared_step_counter=shared_step_counter,
            eval_n_episodes=eval_n_episodes,
            reseed_noise_seed=reseed_noise_seed,
            replay_capacity=int(replay_capacity),
            batch_size=int(batch_size),
            learning_starts=(int(batch_size) if learning_starts is None else int(learning_starts)),
            updates_per_step=int(updates_per_step),
        )

    data_count = math.ceil(n_timesteps / eval_interval)
    eval_returns = np.empty(data_count, dtype=np.float32)
    eval_timesteps = np.empty(data_count, dtype=np.int32)
    eval_write_idx = 0
    next_eval_step = eval_interval

    global_step = 0
    last_episode_return = 0.0

    pbar = None
    if enable_progress_bar:
        tqdm_kwargs = {
            "total": n_timesteps,
            "desc": progress_bar_desc,
            "unit": "step",
            "bar_format": "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            "dynamic_ncols": True,
            "leave": True,
        }
        if progress_bar_position is not None:
            tqdm_kwargs["position"] = int(progress_bar_position)
        pbar = tqdm(**tqdm_kwargs)
    last_progress_update = 0

    try:
        while global_step < n_timesteps:
            states, actions, rewards, next_states, dones = [], [], [], [], []
            state, _ = env.reset()
            episode_done = False

            for _ in range(truncation_step):
                action, _ = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                episode_done = bool(terminated) or bool(truncated)

                states.append(state)
                actions.append(action)
                rewards.append(float(reward))
                next_states.append(next_state)
                dones.append(episode_done)

                state = next_state
                global_step += 1

                if (global_step - last_progress_update) >= 512 or global_step >= n_timesteps:
                    if pbar is not None:
                        pbar.update(global_step - last_progress_update)
                    if shared_step_counter is not None:
                        shared_step_counter.value = min(global_step, n_timesteps)
                    last_progress_update = global_step

                if episode_done or global_step >= n_timesteps:
                    break

            if states:
                agent.update(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    dones=dones,
                )
            if states:
                last_episode_return = float(sum(rewards))
                if pbar is not None:
                    pbar.set_postfix_str(f"episode_return={last_episode_return:.2f}", refresh=False)

            # Record GREEDY evaluation at episode boundary times
            while eval_write_idx < data_count and global_step >= next_eval_step:
                eval_returns[eval_write_idx] = float(agent.evaluate(n_eval_episodes=eval_n_episodes))
                eval_timesteps[eval_write_idx] = next_eval_step
                eval_write_idx += 1
                next_eval_step += eval_interval

    finally:
        if shared_step_counter is not None:
            shared_step_counter.value = min(global_step, n_timesteps)
        if pbar is not None:
            pbar.close()
        env.close()

    if global_step % eval_interval != 0 and eval_write_idx < data_count:
        eval_returns[eval_write_idx] = float(agent.evaluate(n_eval_episodes=eval_n_episodes))
        eval_timesteps[eval_write_idx] = global_step
        eval_write_idx += 1

    return eval_returns[:eval_write_idx], eval_timesteps[:eval_write_idx]


def _run_conventional_sac(agent, env, *, n_timesteps, eval_interval, truncation_step,
                          enable_progress_bar, progress_bar_desc, progress_bar_position,
                          shared_step_counter, eval_n_episodes, reseed_noise_seed,
                          replay_capacity, batch_size, learning_starts, updates_per_step):
    """Conventional off-policy SAC training loop with a replay buffer.

    A single continuous stream of environment steps (the env is reset in place
    at each episode boundary) stores every transition in a FIFO replay buffer.
    Once ``learning_starts`` transitions have accumulated, each env step samples
    ``updates_per_step`` random minibatches of size ``batch_size`` from the
    buffer and performs a discrete-SAC gradient step on each. This is the
    textbook off-policy SAC update cadence; the agent's ``update`` already uses
    soft Bellman targets with the twin target critics, so a randomly sampled
    minibatch is a valid off-policy batch.
    """
    data_count = math.ceil(n_timesteps / eval_interval)
    eval_returns = np.empty(data_count, dtype=np.float32)
    eval_timesteps = np.empty(data_count, dtype=np.int32)
    eval_write_idx = 0
    next_eval_step = eval_interval

    global_step = 0

    # Reproducible minibatch sampling, independent of global RNG state.
    rng = random.Random(0 if reseed_noise_seed is None else int(reseed_noise_seed))
    replay = deque(maxlen=replay_capacity)

    pbar = None
    if enable_progress_bar:
        tqdm_kwargs = {
            "total": n_timesteps,
            "desc": progress_bar_desc,
            "unit": "step",
            "bar_format": "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            "dynamic_ncols": True,
            "leave": True,
        }
        if progress_bar_position is not None:
            tqdm_kwargs["position"] = int(progress_bar_position)
        pbar = tqdm(**tqdm_kwargs)
    last_progress_update = 0

    def _learn():
        batch = rng.sample(replay, batch_size)
        b_states, b_actions, b_rewards, b_next_states, b_dones = zip(*batch)
        agent.update(
            states=list(b_states),
            actions=list(b_actions),
            rewards=list(b_rewards),
            next_states=list(b_next_states),
            dones=list(b_dones),
        )

    try:
        state, _ = env.reset()
        episode_step = 0
        while global_step < n_timesteps:
            action, _ = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_step += 1
            episode_done = bool(terminated) or bool(truncated) or episode_step >= truncation_step

            replay.append((state, action, float(reward), next_state, episode_done))

            state = next_state
            global_step += 1

            if (global_step - last_progress_update) >= 512 or global_step >= n_timesteps:
                if pbar is not None:
                    pbar.update(global_step - last_progress_update)
                if shared_step_counter is not None:
                    shared_step_counter.value = min(global_step, n_timesteps)
                last_progress_update = global_step

            # Off-policy gradient steps once the buffer has warmed up.
            if len(replay) >= learning_starts:
                for _ in range(updates_per_step):
                    _learn()

            # Record GREEDY evaluation on the eval_interval grid.
            while eval_write_idx < data_count and global_step >= next_eval_step:
                eval_returns[eval_write_idx] = float(agent.evaluate(n_eval_episodes=eval_n_episodes))
                eval_timesteps[eval_write_idx] = next_eval_step
                eval_write_idx += 1
                next_eval_step += eval_interval

            if episode_done:
                state, _ = env.reset()
                episode_step = 0

    finally:
        if shared_step_counter is not None:
            shared_step_counter.value = min(global_step, n_timesteps)
        if pbar is not None:
            pbar.close()
        env.close()

    if global_step % eval_interval != 0 and eval_write_idx < data_count:
        eval_returns[eval_write_idx] = float(agent.evaluate(n_eval_episodes=eval_n_episodes))
        eval_timesteps[eval_write_idx] = global_step
        eval_write_idx += 1

    return eval_returns[:eval_write_idx], eval_timesteps[:eval_write_idx]
