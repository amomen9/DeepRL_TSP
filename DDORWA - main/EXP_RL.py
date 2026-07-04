"""RL learning-curve / HPO experiment for stochastic TSP.

Instance selection matches EXP_ALL:
  --instances 1 3        use selected built-in / JSON instances
  --random 20            use one generated random instance; overrides --instances
  no instance argument    use built-in ids 1..4

One command can either run a single config, sweep one hyperparameter, or compare
RL methods under the same base config.  VI is used only as an optional benchmark;
if exact solving is too slow or too memory-heavy, RL still runs and the plot is
saved without the VI line.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import argparse, os
import numpy as np

from instances import TSPInstance, generate_random_instance, load_instances, resolve_instance_ids, list_instance_text
from RL_policy_based import AgentConfig
from RL_trainer import TrainConfig
from helper import train_rl_experiment, load_and_plot_curves
from exact_safety import ExactLimits, run_exact_with_guard, format_bytes

METHODS = ("REINFORCE", "AC", "A2C", "PPO")


@dataclass
class RLExpConfig:
    out_dir: str = "Results_RL"
    instance_file: str | None = None
    instance_ids: tuple[str, ...] = ()
    random_n: int | None = None
    directions: tuple[tuple[int, int], ...] | None = None
    delay_distribution: str | None = None
    seed: int = 42

    exact_time_min: float = 1.0
    exact_memory_gb: float | None = None
    exact_memory_frac: float = 0.70

    methods: tuple[str, ...] = ("A2C",)
    sweep: str | None = None
    seeds: tuple[int, ...] = (0, 1, 2)
    timesteps: int = 200000
    eval_interval: int = 100
    eval_episodes: int = 10
    n_envs: int = 8
    minibatch_size: int = 64

    actor_lr: float = 1e-3
    critic_lr: float = 3e-3
    gamma: float = 0.99
    entropy_coef: float = 0.0
    n_step: int = 5
    actor_hidden: tuple[int, ...] = (32, 32)
    critic_hidden: tuple[int, ...] = (64, 64)

    actor_lrs: tuple[float, ...] = (1e-4, 1e-3, 1e-2)
    critic_lrs: tuple[float, ...] = (3e-4, 1e-3, 3e-3)
    gammas: tuple[float, ...] = (0.95, 0.99, 1.0)
    entropies: tuple[float, ...] = (0.0, 0.01, 0.05)
    n_steps: tuple[int, ...] = (1, 3, 5, 10)
    actor_hidden_values: tuple[tuple[int, ...], ...] = ((32,), (32, 32), (64, 64))
    critic_hidden_values: tuple[tuple[int, ...], ...] = ((64,), (64, 64), (128, 128))


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


def exact_limits(cfg: RLExpConfig) -> ExactLimits:
    return ExactLimits(cfg.exact_time_min, cfg.exact_memory_gb, cfg.exact_memory_frac)


def select_instances(cfg: RLExpConfig) -> list[TSPInstance]:
    if cfg.random_n is not None:
        return [generate_random_instance(cfg.random_n, cfg.directions, seed=cfg.seed, delay_distribution=cfg.delay_distribution or "uniform")]
    all_instances = load_instances(cfg.instance_file)
    if not cfg.instance_ids:
        return resolve_instance_ids(all_instances, ("1", "2", "3", "4"))
    return resolve_instance_ids(all_instances, cfg.instance_ids)


def print_random_matrices(inst: TSPInstance) -> None:
    if not inst.id.startswith("random"):
        return
    print("\nGenerated random instance matrices")
    print("-" * 92)
    print(f"id={inst.id}, name={inst.name}, n={inst.n}")
    print("distance_matrix =")
    print(np.array2string(inst.distance_matrix, precision=1, suppress_small=True))
    print("max_delay_matrix =")
    print(np.array2string(inst.max_delay_matrix, precision=1, suppress_small=True))
    print("delay_mask =")
    print(np.array2string(inst.delay_mask, precision=0, suppress_small=True))
    print(f"uncertain_routes = {inst.uncertain_routes}")


def safe_vi_return(inst: TSPInstance, cfg: RLExpConfig) -> tuple[float | None, str]:
    out = run_exact_with_guard("VI", inst.n, (inst.env_kwargs(cfg.delay_distribution), cfg.seed), exact_limits(cfg))
    if out.get("ok"):
        return float(out["value"]), (
            f"VI benchmark return={out['value']:.3f}; route={out['route']}; "
            f"est_mem={format_bytes(out.get('estimated_memory_bytes'))}; time={out.get('elapsed_sec', 0):.2f}s"
        )
    return None, f"VI benchmark unavailable: {out['reason']}"


def base_agent_cfg(cfg: RLExpConfig, method: str) -> AgentConfig:
    return AgentConfig(
        method=method,
        actor_hidden=cfg.actor_hidden,
        critic_hidden=cfg.critic_hidden,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        gamma=cfg.gamma,
        entropy_coef=cfg.entropy_coef,
        n_step=cfg.n_step,
    )


def sweep_configs(cfg: RLExpConfig) -> list[tuple[str, AgentConfig]]:
    """Return (legend_label, agent_config) curves.

    If --sweep method is used, each curve is one RL algorithm with the same
    base config.  Otherwise each selected method receives the requested HP
    sweep; the label always starts with the algorithm name.
    """
    if cfg.sweep == "method" or (cfg.sweep is None and len(cfg.methods) > 1):
        return [(method, base_agent_cfg(cfg, method)) for method in cfg.methods]

    out: list[tuple[str, AgentConfig]] = []
    for method in cfg.methods:
        base = base_agent_cfg(cfg, method)
        s = cfg.sweep
        if s is None:
            out.append((f"{method}: actor_lr={base.actor_lr:g}, actor_hidden={list(base.actor_hidden)}", base))
        elif s == "actor_lr":
            out += [(f"{method}: actor_lr={v:g}", replace(base, actor_lr=float(v))) for v in cfg.actor_lrs]
        elif s == "critic_lr":
            if method == "REINFORCE":
                raise ValueError("critic_lr sweep is not valid for REINFORCE")
            out += [(f"{method}: critic_lr={v:g}", replace(base, critic_lr=float(v))) for v in cfg.critic_lrs]
        elif s == "actor_hidden":
            out += [(f"{method}: actor_hidden={list(v)}", replace(base, actor_hidden=tuple(v))) for v in cfg.actor_hidden_values]
        elif s == "critic_hidden":
            if method == "REINFORCE":
                raise ValueError("critic_hidden sweep is not valid for REINFORCE")
            out += [(f"{method}: critic_hidden={list(v)}", replace(base, critic_hidden=tuple(v))) for v in cfg.critic_hidden_values]
        elif s == "entropy":
            out += [(f"{method}: entropy={v:g}", replace(base, entropy_coef=float(v))) for v in cfg.entropies]
        elif s == "gamma":
            out += [(f"{method}: gamma={v:g}", replace(base, gamma=float(v))) for v in cfg.gammas]
        elif s == "n_step":
            if method != "A2C":
                raise ValueError("n_step sweep is only valid for A2C")
            out += [(f"{method}: n_step={v}", replace(base, n_step=int(v))) for v in cfg.n_steps]
        else:
            raise ValueError(f"unknown sweep target: {s}")
    return out


def safe_name(text: str) -> str:
    bad = {"=": "_", " ": "", "[": "", "]": "", ",": "-", ".": "p", ":": "_"}
    for a, b in bad.items():
        text = text.replace(a, b)
    return text


def plot_title(inst: TSPInstance, cfg: RLExpConfig) -> str:
    if cfg.sweep == "method" or (cfg.sweep is None and len(cfg.methods) > 1):
        return f"[problem size: {inst.n}] Comparison: Different methods of RL"
    if cfg.sweep is None:
        return f"[problem size: {inst.n}] Comparison: Single config of {cfg.methods[0]}"
    algo = "+".join(cfg.methods)
    return f"[problem size: {inst.n}] Comparison: {cfg.sweep} of {algo}"


def run_for_instance(inst: TSPInstance, cfg: RLExpConfig):
    print_random_matrices(inst)
    vi_return, vi_msg = safe_vi_return(inst, cfg)
    print(f"\nInstance: id={inst.id}, name={inst.name}, n={inst.n}")
    print(vi_msg)

    env_kwargs = inst.env_kwargs(cfg.delay_distribution)
    train_cfg = TrainConfig(timesteps=cfg.timesteps, eval_interval=cfg.eval_interval, eval_episodes=cfg.eval_episodes, seed=cfg.seed, n_envs=cfg.n_envs, minibatch_size=cfg.minibatch_size)
    configs = sweep_configs(cfg)
    csv_files, labels = [], []
    for label, agent_cfg in configs:
        csv_path = os.path.join(cfg.out_dir, f"{inst.id}_{safe_name(label)}.csv")
        print(f"Running {label}: seed_offsets={list(cfg.seeds)}, timesteps={cfg.timesteps}")
        train_rl_experiment(env_kwargs, agent_cfg, train_cfg, list(cfg.seeds), csv_path)
        csv_files.append(csv_path); labels.append(label)

    suffix = safe_name(cfg.sweep or ("method" if len(cfg.methods) > 1 else "single_config"))
    method_tag = safe_name("+".join(cfg.methods))
    plot_path = os.path.join(cfg.out_dir, f"{inst.id}_{method_tag}_{suffix}.png")
    load_and_plot_curves(csv_files, labels=labels, vi_return=vi_return, out_path=plot_path, title=plot_title(inst, cfg))
    print(f"Saved CSV files: {csv_files}")
    print(f"Saved comparison plot: {os.path.abspath(plot_path)}")
    return csv_files, plot_path


def run_exp(cfg: RLExpConfig):
    os.makedirs(cfg.out_dir, exist_ok=True)
    selected = select_instances(cfg)
    print("\nSelected instances")
    print("-" * 92)
    print(list_instance_text(selected))
    mem_text = f"{cfg.exact_memory_gb:g} GB" if cfg.exact_memory_gb else f"{cfg.exact_memory_frac:g} of available RAM"
    print(f"\nMethods={list(cfg.methods)}, sweep={cfg.sweep or 'none'}, steps={cfg.timesteps}, parallel envs={cfg.n_envs}, minibatch={cfg.minibatch_size}, seed_offsets={list(cfg.seeds)}, base seed={cfg.seed}")
    print(f"VI benchmark guard: timeout={cfg.exact_time_min:g} minute(s), memory budget={mem_text}")
    return [run_for_instance(inst, cfg) for inst in selected]


def parse_args() -> RLExpConfig:
    p = argparse.ArgumentParser(description="RL training / HPO curves for stochastic TSP.")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--list-instances", action="store_true")
    p.add_argument("--instance-file", type=str, default=None)
    p.add_argument("--instances", nargs="*", default=None, help="Instance ids. If omitted, built-in ids 1..4 run.")
    p.add_argument("--instance", type=str, default=None, help="Backward-compatible single instance id.")
    p.add_argument("--random", type=int, default=None, metavar="N", help="Generate one random N-node instance; overrides --instances.")
    p.add_argument("--directions", nargs="*", default=None, help="Directed delayed edges for --random, e.g. 1,2 3,4.")
    p.add_argument("--exact-time-min", type=float, default=None, help="Terminate VI benchmark after this many minutes.")
    p.add_argument("--exact-memory-gb", type=float, default=None, help="Explicit memory budget for VI. Default uses a fraction of available RAM.")
    p.add_argument("--exact-memory-frac", type=float, default=None, help="Fraction of currently available RAM allowed for VI.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--distribution", choices=["uniform", "fixed", "none"], default=None)
    p.add_argument("--method", choices=METHODS, nargs="+", default=["A2C"], help="One or more RL methods.")
    p.add_argument("--sweep", choices=["method", "actor_lr", "critic_lr", "actor_hidden", "critic_hidden", "entropy", "gamma", "n_step"], default=None)
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--eval-interval", type=int, default=None)
    p.add_argument("--eval-episodes", type=int, default=None)
    p.add_argument("--n-envs", type=int, default=None, help="Number of synchronous training environments for RL.")
    p.add_argument("--minibatch-size", type=int, default=None, help="Mini-batch size used inside PPO updates.")
    p.add_argument("--seeds", type=int, nargs="*", default=None)

    p.add_argument("--actor-lr", "--base-actor-lr", type=float, default=None)
    p.add_argument("--critic-lr", "--base-critic-lr", type=float, default=None)
    p.add_argument("--gamma", "--base-gamma", type=float, default=None)
    p.add_argument("--entropy", "--base-entropy", type=float, default=None)
    p.add_argument("--n-step", "--base-n-step", type=int, default=None)
    p.add_argument("--actor-hidden", "--base-actor-hidden", type=str, default=None)
    p.add_argument("--critic-hidden", "--base-critic-hidden", type=str, default=None)

    p.add_argument("--actor-lrs", "--lrs", type=float, nargs="*", default=None)
    p.add_argument("--critic-lrs", type=float, nargs="*", default=None)
    p.add_argument("--gamma-values", type=float, nargs="*", default=None)
    p.add_argument("--entropy-values", type=float, nargs="*", default=None)
    p.add_argument("--n-step-values", type=int, nargs="*", default=None)
    p.add_argument("--actor-hidden-values", type=str, nargs="*", default=None, help="Example: 32 32,32 64,64")
    p.add_argument("--critic-hidden-values", type=str, nargs="*", default=None)
    args = p.parse_args()

    cfg = RLExpConfig(methods=tuple(args.method), instance_file=args.instance_file)
    if args.quick:
        cfg.out_dir = "Results_RL_quick"; cfg.timesteps = 40; cfg.eval_interval = 20; cfg.eval_episodes = 3; cfg.seeds = (0, 1); cfg.exact_time_min = 0.25; cfg.n_envs = 2; cfg.minibatch_size = 16
    if args.instances is not None: cfg.instance_ids = tuple(args.instances)
    if args.instance is not None: cfg.instance_ids = (args.instance,)
    if args.random is not None: cfg.random_n = args.random
    cfg.directions = parse_directions(args.directions)
    if args.exact_time_min is not None: cfg.exact_time_min = args.exact_time_min
    if args.exact_memory_gb is not None: cfg.exact_memory_gb = args.exact_memory_gb
    if args.exact_memory_frac is not None: cfg.exact_memory_frac = args.exact_memory_frac
    if args.seed is not None: cfg.seed = args.seed
    if args.out is not None: cfg.out_dir = args.out
    if args.distribution is not None: cfg.delay_distribution = args.distribution
    if args.sweep is not None: cfg.sweep = args.sweep
    if args.timesteps is not None: cfg.timesteps = args.timesteps
    if args.eval_interval is not None: cfg.eval_interval = args.eval_interval
    if args.eval_episodes is not None: cfg.eval_episodes = args.eval_episodes
    if args.n_envs is not None: cfg.n_envs = max(1, args.n_envs)
    if args.minibatch_size is not None: cfg.minibatch_size = max(1, args.minibatch_size)
    if args.seeds is not None and len(args.seeds) > 0: cfg.seeds = tuple(args.seeds)

    if args.actor_lr is not None: cfg.actor_lr = args.actor_lr
    if args.critic_lr is not None: cfg.critic_lr = args.critic_lr
    if args.gamma is not None: cfg.gamma = args.gamma
    if args.entropy is not None: cfg.entropy_coef = args.entropy
    if args.n_step is not None: cfg.n_step = args.n_step
    if args.actor_hidden is not None: cfg.actor_hidden = parse_hidden(args.actor_hidden)
    if args.critic_hidden is not None: cfg.critic_hidden = parse_hidden(args.critic_hidden)

    if args.actor_lrs: cfg.actor_lrs = tuple(args.actor_lrs)
    if args.critic_lrs: cfg.critic_lrs = tuple(args.critic_lrs)
    if args.gamma_values: cfg.gammas = tuple(args.gamma_values)
    if args.entropy_values: cfg.entropies = tuple(args.entropy_values)
    if args.n_step_values: cfg.n_steps = tuple(args.n_step_values)
    if args.actor_hidden_values: cfg.actor_hidden_values = tuple(parse_hidden(x) for x in args.actor_hidden_values)
    if args.critic_hidden_values: cfg.critic_hidden_values = tuple(parse_hidden(x) for x in args.critic_hidden_values)

    if args.list_instances:
        print(list_instance_text(load_instances(cfg.instance_file))); raise SystemExit(0)
    return cfg


if __name__ == "__main__":
    run_exp(parse_args())
