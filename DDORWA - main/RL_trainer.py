"""Generic RL training loop and learning-curve plotting.

Training is step-budgeted.  Several synchronous environments can collect
independent complete tours before one batched on-policy update.  Evaluation uses
a fixed seed set across checkpoints, so learning curves reflect policy changes
rather than changing test traffic.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import csv, os
import numpy as np
import matplotlib.pyplot as plt
import torch
try:
    torch.set_num_threads(1)
except Exception:
    pass
from env import StochasticTSPEnv
from RL_policy_based import PolicyBasedAgent, AgentConfig


@dataclass
class TrainConfig:
    timesteps: int = 2_000
    eval_interval: int = 250
    eval_episodes: int = 10
    seed: int = 0
    log_path: str | None = None
    model_path: str | None = None
    n_envs: int = 1
    minibatch_size: int = 64


def set_seed(seed: int):
    np.random.seed(seed); torch.manual_seed(seed)


def evaluate(agent: PolicyBasedAgent, n_episodes: int = 10, seed: int = 0) -> dict:
    vals = [agent.run_episode(greedy=True, seed=seed + i)["return"] for i in range(n_episodes)]
    return {"mean_return": float(np.mean(vals)), "std_return": float(np.std(vals)), "mean_cost": float(-np.mean(vals))}


def _collect_parallel_episodes(agent: PolicyBasedAgent, envs: list[StochasticTSPEnv], seed0: int, batch_idx: int) -> list[dict]:
    obs = []
    trajectories = []
    for i, env in enumerate(envs):
        o, _ = env.reset(seed0 + 10_000 * batch_idx + i)
        obs.append(o)
        trajectories.append({"states": [], "actions": [], "rewards": [], "masks": [], "next_states": [], "dones": [], "route": [env.depot_city], "return": 0.0})

    done = np.zeros(len(envs), dtype=bool)
    while not bool(done.all()):
        active = np.where(~done)[0]
        obs_batch = np.asarray([obs[i] for i in active], dtype=np.float32)
        masks = np.asarray([envs[i].valid_action_mask() for i in active], dtype=bool)
        action_cities_batch = [envs[i].action_cities for i in active]
        actions = agent.act_batch(obs_batch, masks, greedy=False, action_cities_batch=action_cities_batch)
        for local_i, env_i in enumerate(active):
            env = envs[env_i]
            tr = trajectories[env_i]
            mask = masks[local_i].copy()
            action = int(actions[local_i])
            next_obs, reward, d, _, info = env.step(action)
            tr["states"].append(obs[env_i])
            tr["actions"].append(action)
            tr["rewards"].append(float(reward))
            tr["masks"].append(mask)
            tr["next_states"].append(next_obs)
            tr["dones"].append(bool(d))
            tr["route"].append(int(info.get("next_city", env.action_to_city(action))))
            tr["return"] += float(reward)
            obs[env_i] = next_obs
            done[env_i] = bool(d)
            if d:
                tr["route"].append(env.depot_city)
                tr["cost"] = -float(tr["return"])
                tr["done"] = True
    return trajectories


def train_agent(env: StochasticTSPEnv, agent_cfg: AgentConfig | None = None, train_cfg: TrainConfig | None = None) -> tuple[PolicyBasedAgent, list[dict]]:
    cfg = train_cfg or TrainConfig()
    set_seed(cfg.seed)
    n_envs = max(1, int(cfg.n_envs))
    envs = [env] + [env.clone_with_seed(cfg.seed + i) for i in range(1, n_envs)]
    agent = PolicyBasedAgent(envs[0], agent_cfg)
    logs, steps, batch_idx = [], 0, 0

    while steps < cfg.timesteps:
        trajectories = _collect_parallel_episodes(agent, envs, cfg.seed, batch_idx)
        loss = agent.update_batch(trajectories, minibatch_size=cfg.minibatch_size)
        steps += sum(len(t["rewards"]) for t in trajectories)
        batch_idx += 1

        should_eval = cfg.eval_episodes > 0 and (steps >= cfg.eval_interval * (len(logs) + 1) or steps >= cfg.timesteps)
        if should_eval:
            ev = evaluate(agent, cfg.eval_episodes, seed=10_000 + cfg.seed)
            logs.append({"step": int(steps), **ev, **loss, "episode_return": float(np.mean([t["return"] for t in trajectories]))})

    if cfg.log_path and logs:
        save_logs(logs, cfg.log_path, {**asdict(cfg), **asdict(agent.cfg)})
    if cfg.model_path:
        os.makedirs(os.path.dirname(cfg.model_path) or ".", exist_ok=True)
        agent.save(cfg.model_path)
    return agent, logs


def save_logs(logs: list[dict], path: str, meta: dict | None = None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = sorted(set().union(*[r.keys() for r in logs])) if logs else ["step"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step"] + [k for k in keys if k != "step"])
        w.writeheader(); w.writerows(logs)


def plot_learning_curve(logs: list[dict], benchmark: float | None = None, out_path: str | None = None):
    x = [r["step"] for r in logs]; y = [r["mean_return"] for r in logs]
    plt.figure(figsize=(6, 4)); plt.plot(x, y, label="RL eval mean return")
    if benchmark is not None:
        plt.axhline(benchmark, linestyle="--", label="VI optimal expected return")
    plt.xlabel("Environment steps"); plt.ylabel("Return"); plt.legend(); plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True); plt.savefig(out_path, dpi=160)
    return plt.gcf()
