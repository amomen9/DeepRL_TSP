"""Fair comparison experiment for stochastic TSP.

Instance selection:
  --instances 1 3        use selected built-in / JSON instances
  --random 20            use one generated 20-node random instance; overrides --instances
  no instance argument    use built-in ids 1..4

Exact VI / DP are resource-guarded instead of hard node-bounded.  Before each
exact run, memory is estimated; during the run, a time limit is enforced.  If an
exact method is skipped or times out, RL and the heuristic still continue.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse, csv, os, random
from collections import Counter
import numpy as np
import torch

from env import StochasticTSPEnv
from RL_policy_based import AgentConfig
from RL_trainer import TrainConfig, train_agent
from RL_pointer_network import PointerNetAgent, PointerNetConfig
from heuristic import HeuristicAgent
from instances import TSPInstance, generate_random_instance, load_instances, resolve_instance_ids, list_instance_text
from exact_safety import ExactLimits, run_exact_with_guard, format_bytes

try:
    torch.set_num_threads(1)
except Exception:
    pass


@dataclass
class Config:
    seed: int = 42
    runs: int = 20
    eval_seed_offset: int = 10_000
    output_dir: str = "Results_all"
    instance_file: str | None = None
    instance_ids: tuple[str, ...] = ()
    random_n: int | None = None
    directions: tuple[tuple[int, int], ...] | None = None
    delay_distribution: str | None = None
    print_random_matrices: bool = True

    exact_time_min: float = 1.0
    exact_memory_gb: float | None = None
    exact_memory_frac: float = 0.70

    methods: tuple[str, ...] = ("A2C",)
    use_pointer_net: bool = False
    pointer_embed_dim: int = 64
    pointer_n_glimpses: int = 1
    timesteps: int = 200000
    save_models: bool = True
    actor_lr: float = 1e-3
    critic_lr: float = 3e-3
    gamma: float = 0.99
    entropy_coef: float = 0.0
    n_step: int = 5
    actor_hidden: tuple[int, ...] = (32, 32)
    critic_hidden: tuple[int, ...] = (64, 64)
    n_envs: int = 8
    minibatch_size: int = 64


def set_global_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def parse_hidden(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.replace(";", ",").split(",") if x.strip())


def parse_directions(items: list[str] | None) -> tuple[tuple[int, int], ...] | None:
    if not items:
        return None
    out: list[tuple[int, int]] = []
    for item in items:
        for chunk in item.replace(";", " ").split():
            parts = chunk.replace("-", ",").replace(":", ",").split(",")
            if len(parts) != 2:
                raise ValueError("directions must look like: --directions 1,2 3,4")
            out.append((int(parts[0]), int(parts[1])))
    return tuple(out)


def eval_seed(cfg: Config, run_idx: int) -> int:
    return int(cfg.seed + cfg.eval_seed_offset + run_idx)


def exact_limits(cfg: Config) -> ExactLimits:
    return ExactLimits(cfg.exact_time_min, cfg.exact_memory_gb, cfg.exact_memory_frac)


def make_env(inst: TSPInstance, seed: int | None, cfg: Config) -> StochasticTSPEnv:
    return StochasticTSPEnv(**inst.env_kwargs(cfg.delay_distribution), seed=seed)


def select_instances(cfg: Config) -> list[TSPInstance]:
    if cfg.random_n is not None:
        return [generate_random_instance(cfg.random_n, cfg.directions, seed=cfg.seed, delay_distribution=cfg.delay_distribution or "uniform")]
    all_instances = load_instances(cfg.instance_file)
    if not cfg.instance_ids:
        return resolve_instance_ids(all_instances, ("1", "2", "3", "4"))
    return resolve_instance_ids(all_instances, cfg.instance_ids)


def print_matrices(inst: TSPInstance) -> None:
    if not inst.id.startswith("random"):
        return
    print("\nGenerated random instance matrices")
    print("-" * 92)
    print(f"id={inst.id}, name={inst.name}, n={inst.n}")
    print("distance_matrix =")
    print(np.array2string(inst.distance_matrix, precision=1, suppress_small=True))
    print("max_delay_matrix =")
    print(np.array2string(np.asarray(inst.max_delay_matrix), precision=1, suppress_small=True))
    print("delay_mask =")
    print(np.array2string(np.asarray(inst.delay_mask), precision=0, suppress_small=True))
    print(f"uncertain_routes = {inst.uncertain_routes}")


def safe_value_iteration(inst: TSPInstance, cfg: Config) -> dict:
    return run_exact_with_guard("VI", inst.n, (inst.env_kwargs(cfg.delay_distribution), cfg.seed), exact_limits(cfg))


def safe_classic_dp(inst: TSPInstance, cfg: Config, depot_city: int) -> dict:
    return run_exact_with_guard("DP", inst.n, (inst.distance_matrix, depot_city), exact_limits(cfg))


def train_pointer_net_for_instance(inst: TSPInstance, cfg: Config):
    train_seed = cfg.seed + 2_000 + inst.n
    env = make_env(inst, train_seed, cfg)
    agent_cfg = PointerNetConfig(
        embed_dim=cfg.pointer_embed_dim,
        n_glimpses=cfg.pointer_n_glimpses,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        gamma=cfg.gamma,
        entropy_coef=cfg.entropy_coef,
        n_step=cfg.n_step,
    )
    agent = PointerNetAgent(env, agent_cfg)
    np.random.seed(train_seed); torch.manual_seed(train_seed)
    steps, ep = 0, 0
    while steps < cfg.timesteps:
        traj = agent.run_episode(seed=train_seed + ep)
        agent.update(traj)
        steps += len(traj["rewards"])
        ep += 1
    model_path = os.path.join(cfg.output_dir, f"{inst.id}_PointerNet.pt") if cfg.save_models else None
    if model_path:
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        agent.save(model_path)
    return agent, model_path


def train_rl_for_instance(inst: TSPInstance, cfg: Config, method: str):
    train_seed = cfg.seed + 1_000 + inst.n + 37 * sum(ord(c) for c in method)
    env = make_env(inst, train_seed, cfg)
    agent_cfg = AgentConfig(
        method=method,
        actor_hidden=cfg.actor_hidden,
        critic_hidden=cfg.critic_hidden,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        gamma=cfg.gamma,
        entropy_coef=cfg.entropy_coef,
        n_step=cfg.n_step,
    )
    model_path = os.path.join(cfg.output_dir, f"{inst.id}_{method}.pt") if cfg.save_models else None
    train_cfg = TrainConfig(timesteps=cfg.timesteps, eval_interval=10**12, eval_episodes=0, seed=train_seed, log_path=None, model_path=model_path, n_envs=cfg.n_envs, minibatch_size=cfg.minibatch_size)
    agent, _ = train_agent(env, agent_cfg, train_cfg)
    return agent, model_path


def summarize(records: list[dict]) -> list[dict]:
    rows = []
    for inst in sorted({r["instance"] for r in records}):
        for agent in sorted({r["agent"] for r in records if r["instance"] == inst}):
            vals = np.asarray([r["cost"] for r in records if r["instance"] == inst and r["agent"] == agent], dtype=float)
            rows.append({"instance": inst, "agent": agent, "mean_cost": float(vals.mean()), "std_cost": float(vals.std(ddof=0)), "min_cost": float(vals.min()), "max_cost": float(vals.max())})
    return rows


def print_table(rows: list[dict]) -> None:
    print("\n" + "=" * 92)
    print("Fair comparison results: available methods evaluated on the same realized seeds per run")
    print("=" * 92)
    if not rows:
        print("No evaluation records were produced.")
        return
    print(f"{'Instance':<22} | {'Agent':<16} | {'Mean cost':>10} | {'Std':>8} | {'Min':>8} | {'Max':>8}")
    print("-" * 92)
    for r in rows:
        print(f"{r['instance']:<22} | {r['agent']:<16} | {r['mean_cost']:>10.3f} | {r['std_cost']:>8.3f} | {r['min_cost']:>8.3f} | {r['max_cost']:>8.3f}")


def print_solution_summary(records: list[dict], planned: dict[str, dict[str, str | list[int]]], statuses: list[str]) -> None:
    print("\n" + "=" * 92)
    print("Solutions / routes used during fair evaluation")
    print("=" * 92)
    if statuses:
        print("\nMethod status:")
        for s in statuses:
            print("  " + s)
    for inst in sorted(set(planned) | {r["instance"] for r in records}):
        print(f"\n{inst}")
        for agent, route in planned.get(inst, {}).items():
            print(f"  planned {agent:<16}: {route}")
        for agent in sorted({r["agent"] for r in records if r["instance"] == inst}):
            routes = [tuple(r["route"]) for r in records if r["instance"] == inst and r["agent"] == agent]
            common = Counter(routes).most_common(3)
            text = "; ".join(f"{list(route)} x{count}" for route, count in common)
            print(f"  evaluated {agent:<14}: {text}")


def save_records(cfg: Config, rows: list[dict], records: list[dict]) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "comparison_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["instance", "agent", "mean_cost", "std_cost", "min_cost", "max_cost"])
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(cfg.output_dir, "comparison_records.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["instance", "agent", "run", "seed", "cost", "return", "done", "route"])
        w.writeheader(); w.writerows(records)
    print(f"\nSaved fair-comparison files to: {os.path.abspath(cfg.output_dir)}")


def run_all(cfg: Config):
    set_global_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)
    selected = select_instances(cfg)

    print("\nSelected instances")
    print("-" * 92)
    print(list_instance_text(selected))
    for inst in selected:
        if cfg.print_random_matrices:
            print_matrices(inst)
    mem_text = f"{cfg.exact_memory_gb:g} GB" if cfg.exact_memory_gb else f"{cfg.exact_memory_frac:g} of available RAM"
    ptr_str = f", PointerNet(embed={cfg.pointer_embed_dim}, glimpses={cfg.pointer_n_glimpses})" if cfg.use_pointer_net else ""
    print(f"\nRL methods={list(cfg.methods)}{ptr_str}, train steps={cfg.timesteps}, parallel envs={cfg.n_envs}, minibatch={cfg.minibatch_size}, eval runs={cfg.runs}, base seed={cfg.seed}")
    print(f"Exact-method guard: timeout={cfg.exact_time_min:g} minute(s), memory budget={mem_text}")

    records: list[dict] = []
    planned: dict[str, dict[str, str | list[int]]] = {}
    statuses: list[str] = []

    for inst in selected:
        print(f"\n--- {inst.name} (n={inst.n}) ---")
        planning_env = make_env(inst, cfg.seed, cfg)
        vi = safe_value_iteration(inst, cfg)
        dp = safe_classic_dp(inst, cfg, planning_env.depot_city)

        rl_agents = {}
        for method in cfg.methods:
            try:
                rl_agent, model_path = train_rl_for_instance(inst, cfg, method)
                rl_agents[method] = rl_agent
                print(f"  trained {method}" + (f"; model saved to {model_path}" if model_path else ""))
            except Exception as exc:
                statuses.append(f"{inst.name} / {method}: failed: {type(exc).__name__}: {exc}")

        ptr_agent = None
        if cfg.use_pointer_net:
            try:
                ptr_agent, ptr_path = train_pointer_net_for_instance(inst, cfg)
                print(f"  trained PointerNet" + (f"; model saved to {ptr_path}" if ptr_path else ""))
            except Exception as exc:
                statuses.append(f"{inst.name} / PointerNet: failed: {type(exc).__name__}: {exc}")

        planned[inst.name] = {}
        if vi.get("ok"):
            planned[inst.name]["VI_optimal"] = vi["route"]
            print(f"  VI_optimal: ok, est_mem={format_bytes(vi.get('estimated_memory_bytes'))}, time={vi.get('elapsed_sec', 0):.2f}s")
        else:
            planned[inst.name]["VI_optimal"] = str(vi["reason"])
            statuses.append(f"{inst.name} / VI_optimal: {vi['reason']}")
            print(f"  VI_optimal: {vi['reason']}")
        if dp.get("ok"):
            planned[inst.name]["Classic_DP_base"] = dp["routes"][0]
            print(f"  Classic_DP_base: ok, est_mem={format_bytes(dp.get('estimated_memory_bytes'))}, time={dp.get('elapsed_sec', 0):.2f}s")
        else:
            planned[inst.name]["Classic_DP_base"] = str(dp["reason"])
            statuses.append(f"{inst.name} / Classic_DP_base: {dp['reason']}")
            print(f"  Classic_DP_base: {dp['reason']}")

        for run_idx in range(cfg.runs):
            seed = eval_seed(cfg, run_idx)
            if vi.get("ok"):
                res = make_env(inst, seed, cfg).run_actions(vi["actions"], seed=seed)
                records.append({"instance": inst.name, "agent": "VI_optimal", "run": run_idx + 1, "seed": seed, **res})
            if dp.get("ok"):
                res = make_env(inst, seed, cfg).run_actions(dp["actions"], seed=seed)
                records.append({"instance": inst.name, "agent": "Classic_DP_base", "run": run_idx + 1, "seed": seed, **res})
            for method, rl_agent in rl_agents.items():
                env_rl = make_env(inst, seed, cfg); rl_agent.env = env_rl
                res = rl_agent.run_episode(greedy=True, seed=seed)
                records.append({"instance": inst.name, "agent": method, "run": run_idx + 1, "seed": seed, "route": res["route"], "cost": res["cost"], "return": res["return"], "done": res["done"]})
            if ptr_agent is not None:
                ptr_agent.env = make_env(inst, seed, cfg)
                res = ptr_agent.run_episode(greedy=True, seed=seed)
                records.append({"instance": inst.name, "agent": "PointerNet", "run": run_idx + 1, "seed": seed, "route": res["route"], "cost": res["cost"], "return": res["return"], "done": res["done"]})
            try:
                res = HeuristicAgent(make_env(inst, seed, cfg)).run_episode(seed=seed)
                records.append({"instance": inst.name, "agent": "Heuristic", "run": run_idx + 1, "seed": seed, **res})
            except Exception as exc:
                if run_idx == 0:
                    statuses.append(f"{inst.name} / Heuristic: failed: {type(exc).__name__}: {exc}")

    rows = summarize(records) if records else []
    print_table(rows)
    print_solution_summary(records, planned, statuses)
    print(f"\nRepeated-run control: run k uses evaluation seed = {cfg.seed} + {cfg.eval_seed_offset} + k. Every method receives the same seed for that run.")
    save_records(cfg, rows, records)
    return rows, records


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Fair stochastic TSP comparison.")
    p.add_argument("--quick", action="store_true", help="Fast smoke test.")
    p.add_argument("--list-instances", action="store_true")
    p.add_argument("--instance-file", type=str, default=None)
    p.add_argument("--instances", nargs="*", default=None, help="Instance ids. If omitted, built-in ids 1..4 run.")
    p.add_argument("--random", type=int, default=None, metavar="N", help="Generate one random N-node instance; overrides --instances.")
    p.add_argument("--directions", nargs="*", default=None, help="Directed delayed edges for --random, e.g. 1,2 3,4. Non-symmetric.")
    p.add_argument("--exact-time-min", type=float, default=None, help="Terminate each VI/DP run after this many minutes.")
    p.add_argument("--exact-memory-gb", type=float, default=None, help="Explicit memory budget for one VI/DP run. Default uses a fraction of available RAM.")
    p.add_argument("--exact-memory-frac", type=float, default=None, help="Fraction of currently available RAM allowed for one VI/DP run.")
    p.add_argument("--runs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--distribution", choices=["uniform", "fixed", "none"], default=None)
    p.add_argument("--method", choices=["REINFORCE", "AC", "A2C", "PPO"], nargs="+", default=["A2C"], help="One or more RL methods to train/evaluate.")
    p.add_argument("--pointer-net", action="store_true", help="Also train and evaluate the Pointer Network agent.")
    p.add_argument("--pointer-embed-dim", type=int, default=None, help="Embedding dimension for the Pointer Network (default 64).")
    p.add_argument("--pointer-glimpses", type=int, default=None, help="Number of glimpse steps for the Pointer Network (default 1).")
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--n-envs", type=int, default=None, help="Number of synchronous training environments for RL.")
    p.add_argument("--minibatch-size", type=int, default=None, help="Mini-batch size used inside RL updates.")
    p.add_argument("--actor-lr", type=float, default=None)
    p.add_argument("--critic-lr", type=float, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--entropy", type=float, default=None)
    p.add_argument("--n-step", type=int, default=None)
    p.add_argument("--actor-hidden", type=str, default=None)
    p.add_argument("--critic-hidden", type=str, default=None)
    p.add_argument("--no-save-models", action="store_true")
    args = p.parse_args()

    cfg = Config(methods=tuple(args.method), instance_file=args.instance_file)
    if args.quick:
        cfg.runs = 3; cfg.timesteps = 120; cfg.output_dir = "Results_all_quick"; cfg.exact_time_min = 0.25
    if args.instances is not None: cfg.instance_ids = tuple(args.instances)
    if args.random is not None: cfg.random_n = args.random
    cfg.directions = parse_directions(args.directions)
    if args.exact_time_min is not None: cfg.exact_time_min = args.exact_time_min
    if args.exact_memory_gb is not None: cfg.exact_memory_gb = args.exact_memory_gb
    if args.exact_memory_frac is not None: cfg.exact_memory_frac = args.exact_memory_frac
    if args.runs is not None: cfg.runs = args.runs
    if args.seed is not None: cfg.seed = args.seed
    if args.out is not None: cfg.output_dir = args.out
    if args.distribution is not None: cfg.delay_distribution = args.distribution
    if args.timesteps is not None: cfg.timesteps = args.timesteps
    if args.n_envs is not None: cfg.n_envs = max(1, args.n_envs)
    if args.minibatch_size is not None: cfg.minibatch_size = max(1, args.minibatch_size)
    if args.actor_lr is not None: cfg.actor_lr = args.actor_lr
    if args.critic_lr is not None: cfg.critic_lr = args.critic_lr
    if args.gamma is not None: cfg.gamma = args.gamma
    if args.entropy is not None: cfg.entropy_coef = args.entropy
    if args.n_step is not None: cfg.n_step = args.n_step
    if args.actor_hidden is not None: cfg.actor_hidden = parse_hidden(args.actor_hidden)
    if args.critic_hidden is not None: cfg.critic_hidden = parse_hidden(args.critic_hidden)
    if args.no_save_models: cfg.save_models = False
    if args.pointer_net: cfg.use_pointer_net = True
    if args.pointer_embed_dim is not None: cfg.pointer_embed_dim = args.pointer_embed_dim
    if args.pointer_glimpses is not None: cfg.pointer_n_glimpses = args.pointer_glimpses
    if args.list_instances:
        print(list_instance_text(load_instances(cfg.instance_file))); raise SystemExit(0)
    return cfg


if __name__ == "__main__":
    run_all(parse_args())
