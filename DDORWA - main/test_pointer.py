"""Scale test: pointer network vs A2C baseline across instance sizes.

Tests n = 6, 10, 15, 20 nodes.
For n=6 and n=10 we use the built-in instances so results are comparable
to the paper's Table 3.  For n=15 and n=20 we use random instances.

Run:  python test_pointer_scale.py
"""
import numpy as np
import torch
from env import StochasticTSPEnv
from instances import builtin_instances, generate_random_instance
from RL_pointer_network import PointerNetAgent, PointerNetConfig
from RL_policy_based import PolicyBasedAgent, AgentConfig

# ========================= helpers =========================

def train_pointer(env, steps=30_000, seed=42):
    np.random.seed(seed); torch.manual_seed(seed)
    cfg   = PointerNetConfig(embed_dim=64, n_glimpses=1,
                             entropy_coef=0.05, actor_lr=3e-4, critic_lr=1e-3)
    agent = PointerNetAgent(env, cfg)
    s = 0
    while s < steps:
        traj = agent.run_episode(greedy=False, seed=seed + s)
        agent.update(traj)
        s += len(traj["rewards"])
    return agent

def train_a2c(env, steps=30_000, seed=42):
    np.random.seed(seed); torch.manual_seed(seed)
    cfg   = AgentConfig(method="A2C", actor_hidden=(32,32),
                        critic_hidden=(64,64), actor_lr=3e-4,
                        entropy_coef=0.01, gamma=1.0)
    agent = PolicyBasedAgent(env, cfg)
    s = 0
    while s < steps:
        traj = agent.run_episode(greedy=False, seed=seed + s)
        agent.update(traj)
        s += len(traj["rewards"])
    return agent

def evaluate(agent, env, n_episodes=20, seed=10_042):
    costs = []
    for k in range(n_episodes):
        ep = agent.run_episode(greedy=True, seed=seed + k)
        costs.append(ep["cost"])
    return float(np.mean(costs)), float(np.min(costs)), float(np.std(costs))

# ========================= instances =========================

builtin = builtin_instances()

test_cases = [
    ("6-node  (builtin)", builtin[2]),           # Capacity_6
    ("10-node (builtin)", builtin[3]),            # Capacity_10
    ("15-node (random)", generate_random_instance(15, seed=42)),
    ("20-node (random)", generate_random_instance(20, seed=42)),
]

TRAIN_STEPS = 30_000   # increase to 100k for better results but slower

# ========================= run =========================

print(f"\n{'Instance':<22} {'Method':<12} {'Mean cost':>10} {'Min cost':>10} {'Std':>8}")
print("-" * 68)

for label, inst in test_cases:
    env = StochasticTSPEnv(**inst.env_kwargs())

    # Pointer network
    ptr_agent  = train_pointer(env, steps=TRAIN_STEPS)
    ptr_mean, ptr_min, ptr_std = evaluate(ptr_agent, env)

    # A2C baseline
    a2c_agent  = train_a2c(env, steps=TRAIN_STEPS)
    a2c_mean, a2c_min, a2c_std = evaluate(a2c_agent, env)

    print(f"{label:<22} {'PointerNet':<12} {ptr_mean:>10.2f} {ptr_min:>10.2f} {ptr_std:>8.2f}")
    print(f"{label:<22} {'A2C':<12} {a2c_mean:>10.2f} {a2c_min:>10.2f} {a2c_std:>8.2f}")
    print()

print("Done.")