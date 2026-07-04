"""Experiment helpers: multi-seed RL runs and CSV learning-curve plots."""
from __future__ import annotations
from dataclasses import asdict
import csv, os
import numpy as np
import matplotlib.pyplot as plt
from env import StochasticTSPEnv
from RL_policy_based import AgentConfig
from RL_trainer import TrainConfig, train_agent
from value_iteration import value_iteration

# Small plotting knobs: set smoothing to 1 to disable it.
PLOT_SMOOTH_WINDOW = 7
PLOT_BAND_SCALE = 0.35  # shrink variance band for readability; 1.0 = full one-std band


def _smooth_curve(y, window=PLOT_SMOOTH_WINDOW):
    """Light moving-average smoothing; keeps array length unchanged."""
    y = np.asarray(y, dtype=float)
    w = int(window)
    if w <= 1 or y.size < 3:
        return y
    w = min(w, y.size if y.size % 2 == 1 else y.size - 1)
    if w <= 1:
        return y
    pad_left = w // 2
    pad_right = w - 1 - pad_left
    padded = np.pad(y, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, np.ones(w, dtype=float) / w, mode="valid")


def train_rl_experiment(env_kwargs: dict, agent_cfg: AgentConfig, train_cfg: TrainConfig, seeds: list[int], csv_path: str) -> str:
    """Run one RL config across independent seeds; save all logs in one CSV."""
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    rows = []
    base_seed = int(train_cfg.seed)
    for seed_offset in seeds:
        seed = base_seed + int(seed_offset)
        env = StochasticTSPEnv(**env_kwargs, seed=seed)
        cfg = TrainConfig(**{**asdict(train_cfg), "seed": seed, "log_path": None, "model_path": None})
        _, logs = train_agent(env, agent_cfg, cfg)
        for r in logs:
            rows.append({"seed": seed, **asdict(agent_cfg), **r})
    keys = ["seed", "step"] + [k for k in sorted(set().union(*[r.keys() for r in rows])) if k not in {"seed", "step"}]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    return csv_path


def load_and_plot_curves(csv_files: list[str], labels: list[str] | None = None, vi_return: float | None = None, out_path: str | None = None, title: str | None = None):
    """Each CSV is one curve. Plot mean across seeds with one-std variance band."""
    import pandas as pd

    plt.figure(figsize=(7, 4.5))
    labels = labels or [os.path.splitext(os.path.basename(p))[0] for p in csv_files]
    for path, label in zip(csv_files, labels):
        df = pd.read_csv(path)
        if df.empty:
            continue
        grouped = df.groupby("step")["mean_return"]
        x = grouped.mean().index.to_numpy(dtype=float)
        m = grouped.mean().to_numpy(dtype=float)
        sd = grouped.std(ddof=0).fillna(0.0).to_numpy(dtype=float)
        m_s = _smooth_curve(m)
        sd_s = _smooth_curve(sd) * float(PLOT_BAND_SCALE)
        plt.plot(x, m_s, label=label)
        plt.fill_between(x, m_s - sd_s, m_s + sd_s, alpha=0.14)
    if vi_return is not None:
        plt.axhline(vi_return, linestyle="--", color="0.25", linewidth=1.4, label="VI optimal expected return")
    if title:
        plt.title(title)
    plt.xlabel("Environment steps"); plt.ylabel("Evaluation return"); plt.legend(); plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True); plt.savefig(out_path, dpi=160)
    return plt.gcf()


def vi_benchmark(env_kwargs: dict) -> dict:
    env = StochasticTSPEnv(**env_kwargs, seed=0)
    return value_iteration(env)
