"""Large-scale test: pointer network vs A2C on n=50 and n=100.

These sizes are where Bello et al. showed the pointer network really shines.
Training budget is higher because larger instances need more steps.

Expected runtime: 20-40 minutes depending on your machine.
Run:  python test_pointer_large.py
"""
import numpy as np
import torch
from env import StochasticTSPEnv
from instances import generate_random_instance
from RL_pointer_network import PointerNetAgent, PointerNetConfig
from RL_policy_based import PolicyBasedAgent, AgentConfig

# ========================= helpers =========================

def train_pointer(env, steps, seed=42):
    np.random.seed(seed); torch.manual_seed(seed)
    cfg = PointerNetConfig(
        embed_dim    = 64,
        n_glimpses   = 1,
        entropy_coef = 0.05,
        actor_lr     = 3e-4,
        critic_lr    = 1e-3,
    )
    agent = PointerNetAgent(env, cfg)
    s = 0
    while s < steps:
        traj = agent.run_episode(greedy=False, seed=seed + s)
        agent.update(traj)
        s += len(traj["rewards"])
        if s % 10_000 == 0:
            print(f"  PointerNet step {s}/{steps}...")
    return agent

def train_a2c(env, steps, seed=42):
    np.random.seed(seed); torch.manual_seed(seed)
    cfg = AgentConfig(
        method        = "A2C",
        actor_hidden  = (64, 64),
        critic_hidden = (128, 128),
        actor_lr      = 3e-4,
        entropy_coef  = 0.01,
        gamma         = 1.0,
    )
    agent = PolicyBasedAgent(env, cfg)
    s = 0
    while s < steps:
        traj = agent.run_episode(greedy=False, seed=seed + s)
        agent.update(traj)
        s += len(traj["rewards"])
        if s % 10_000 == 0:
            print(f"  A2C step {s}/{steps}...")
    return agent

def evaluate(agent, n_episodes=20, seed=10_042):
    costs = []
    for k in range(n_episodes):
        ep = agent.run_episode(greedy=True, seed=seed + k)
        costs.append(ep["cost"])
    return float(np.mean(costs)), float(np.min(costs)), float(np.std(costs))

# ========================= test cases =========================
# steps scale with n because tours are longer
test_cases = [
    ("50-node  (random)", generate_random_instance(50,  seed=42), 100_000),
    ("100-node (random)", generate_random_instance(100, seed=42), 200_000),
]

# ========================= run =========================
print(f"\n{'Instance':<22} {'Method':<12} {'Mean cost':>10} {'Min cost':>10} {'Std':>8}")
print("-" * 68)

for label, inst, train_steps in test_cases:
    env = StochasticTSPEnv(**inst.env_kwargs())
    print(f"\nTraining {label}  ({train_steps:,} steps each)...")

    print("  Training PointerNet...")
    ptr_agent = train_pointer(env, steps=train_steps)
    ptr_mean, ptr_min, ptr_std = evaluate(ptr_agent)

    print("  Training A2C...")
    a2c_agent = train_a2c(env, steps=train_steps)
    a2c_mean, a2c_min, a2c_std = evaluate(a2c_agent)

    print(f"\n{label:<22} {'PointerNet':<12} {ptr_mean:>10.2f} {ptr_min:>10.2f} {ptr_std:>8.2f}")
    print(f"{label:<22} {'A2C':<12} {a2c_mean:>10.2f} {a2c_min:>10.2f} {a2c_std:>8.2f}")

print("\nDone.")