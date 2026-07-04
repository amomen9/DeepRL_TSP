import math

import numpy as np
from tqdm import tqdm


def run_TSP_ppo(agent, env, n_timesteps=200000, eval_interval=250,
                truncation_step=None, enable_progress_bar=True,
                progress_bar_desc="TSP PPO Steps",
                progress_bar_position=None,
                shared_step_counter=None,
                eval_n_episodes=1,
                reseed_noise_seed=None,
                full_episode_updates=True,
                rollout_steps=None):
    """PPO step-based training loop for the TSP DP_Table environment.

    Episodes terminate when all cities are visited.

    full_episode_updates=True: one PPO update (clipped surrogate + GAE) is
    performed at the end of each episode on the full trajectory; ``rollout_steps``
    is ignored (the whole episode is a single rollout). This is the original
    (pre-FULL_EPISODE_UPDATES) behaviour.
    full_episode_updates=False: the conventional PPO loop. A continuous stream
    of environment steps (the env is reset in place at each episode boundary) is
    collected into a fixed-length rollout buffer of ``rollout_steps`` steps that
    spans episode boundaries. When the buffer is full a PPO update runs and the
    buffer is cleared. GAE uses each transition's ``done`` flag, so terminals
    inside the rollout are handled correctly and the rollout no longer collapses
    to a single whole-episode update.

    eval_returns are recorded as GREEEDY evaluation returns (argmax policy),
    not the last sampled training episode return (which can be much noisier
    and can bias algorithm comparisons).
    """
    if reseed_noise_seed is not None:
        env.reseed_noise(reseed_noise_seed)  # Deterministic reseed when caller provides a seed

    if truncation_step is None:
        truncation_step = int(env.n + 1)

    if not full_episode_updates:
        # Conventional PPO: a fixed-length rollout buffer that persists across
        # episode boundaries. Default to the standard PPO horizon when unset.
        if rollout_steps is None or int(rollout_steps) <= 0:
            rollout_steps = 2048
        else:
            rollout_steps = int(rollout_steps)
        return _run_conventional_ppo(
            agent, env,
            n_timesteps=n_timesteps,
            eval_interval=eval_interval,
            truncation_step=truncation_step,
            rollout_steps=rollout_steps,
            enable_progress_bar=enable_progress_bar,
            progress_bar_desc=progress_bar_desc,
            progress_bar_position=progress_bar_position,
            shared_step_counter=shared_step_counter,
            eval_n_episodes=eval_n_episodes,
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
            states, actions, rewards, next_states, dones, masks = [], [], [], [], [], []
            state, _ = env.reset()
            episode_done = False
            episode_return = 0.0

            for _ in range(truncation_step):
                # Capture the invalid-action mask for THIS state (env is still at
                # `state` until env.step below) so the PPO update can mask the
                # policy distribution exactly as select_action does. Without this
                # the importance ratio explodes to inf -> NaN actor weights.
                mask = agent._get_invalid_action_mask()
                action, _ = agent.select_action(state, mask=mask)
                next_state, reward, terminated, truncated, _ = env.step(action)
                episode_done = bool(terminated) or bool(truncated)

                states.append(state)
                actions.append(action)
                rewards.append(float(reward))
                next_states.append(next_state)
                dones.append(episode_done)
                masks.append(mask)

                state = next_state
                global_step += 1
                episode_return += float(reward)

                if (global_step - last_progress_update) >= 512 or global_step >= n_timesteps:
                    if pbar is not None:
                        pbar.update(global_step - last_progress_update)
                    if shared_step_counter is not None:
                        shared_step_counter.value = min(global_step, n_timesteps)
                    last_progress_update = global_step

                if episode_done or global_step >= n_timesteps:
                    break

            # One PPO update per episode on the whole-episode trajectory.
            if states:
                agent.update(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    dones=dones,
                    masks=masks,
                )

            last_episode_return = episode_return
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


def _run_conventional_ppo(agent, env, *, n_timesteps, eval_interval, truncation_step,
                          rollout_steps, enable_progress_bar, progress_bar_desc,
                          progress_bar_position, shared_step_counter, eval_n_episodes):
    """Conventional PPO training loop with a persistent fixed-length rollout.

    A single continuous stream of environment steps (the env is reset in place
    at each episode boundary) fills a rollout buffer of ``rollout_steps``
    transitions that spans episode boundaries. Each full buffer triggers one
    PPO update (clipped surrogate + GAE over ``n_epochs``) and is then cleared.
    Per-transition ``dones`` let GAE reset the advantage recursion at terminals
    inside the rollout, so the rollout never collapses to a whole-episode update.
    """
    data_count = math.ceil(n_timesteps / eval_interval)
    eval_returns = np.empty(data_count, dtype=np.float32)
    eval_timesteps = np.empty(data_count, dtype=np.int32)
    eval_write_idx = 0
    next_eval_step = eval_interval

    global_step = 0

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

    states, actions, rewards, next_states, dones, masks = [], [], [], [], [], []

    def _flush_rollout():
        if states:
            agent.update(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                dones=dones,
                masks=masks,
            )
            states.clear(); actions.clear(); rewards.clear()
            next_states.clear(); dones.clear(); masks.clear()

    try:
        state, _ = env.reset()
        episode_step = 0
        while global_step < n_timesteps:
            # Capture the invalid-action mask for THIS state before stepping so
            # the PPO update masks the policy exactly as select_action did.
            mask = agent._get_invalid_action_mask()
            action, _ = agent.select_action(state, mask=mask)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_step += 1
            episode_done = bool(terminated) or bool(truncated) or episode_step >= truncation_step

            states.append(state)
            actions.append(action)
            rewards.append(float(reward))
            next_states.append(next_state)
            dones.append(episode_done)
            masks.append(mask)

            state = next_state
            global_step += 1

            if (global_step - last_progress_update) >= 512 or global_step >= n_timesteps:
                if pbar is not None:
                    pbar.update(global_step - last_progress_update)
                if shared_step_counter is not None:
                    shared_step_counter.value = min(global_step, n_timesteps)
                last_progress_update = global_step

            # One PPO update per completed fixed-length rollout, then clear.
            if len(states) >= rollout_steps:
                _flush_rollout()

            # Record GREEDY evaluation on the eval_interval grid.
            while eval_write_idx < data_count and global_step >= next_eval_step:
                eval_returns[eval_write_idx] = float(agent.evaluate(n_eval_episodes=eval_n_episodes))
                eval_timesteps[eval_write_idx] = next_eval_step
                eval_write_idx += 1
                next_eval_step += eval_interval

            if episode_done:
                state, _ = env.reset()
                episode_step = 0

        # Final partial rollout.
        _flush_rollout()

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
