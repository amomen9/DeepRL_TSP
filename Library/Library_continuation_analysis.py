"""
Library_continuation_analysis.py - "Trial Continuation Analysis" reporting.

TSP-native adaptation of the CartPole fork's ``build_returns_summary_table``
(``Helper.py``). The CartPole version depends on that project's legend/exclusion
helpers and job schema; this version consumes the TSP orchestrator's own
structures directly:

  * ``algo_jobs``        : {ALGO_UPPER: [job, ...]} where each job has a
                           "curve_label" (the setting label) and "method".
  * ``algo_job_offsets`` : {ALGO_UPPER: int} offset into ``setting_results``.
  * ``setting_results``  : list of (lc_mean, lc_std, timesteps[, raw_returns]).

It pools every repetition and every evaluation point into an overall mean/std,
and the last ``last_fraction`` of eval points into a "last N%" mean/std (the
converged-performance estimate). The table is always printed; CSV/Markdown
artifacts under ``output_dir`` are written only when checkpoint reuse is on.
"""

import os
from typing import Any

import numpy as np


def _fmt(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "-"
    return f"{x:,.3f}"


def _render_aligned_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    sep = "  "
    line_fmt = sep.join("{:<" + str(w) + "}" for w in widths)
    out = [line_fmt.format(*headers), line_fmt.format(*["-" * w for w in widths])]
    for row in rows:
        out.append(line_fmt.format(*row))
    return "\n".join(out)


def _aggregate_returns(entry, n_repetitions: int, last_fraction: float):
    """Return ((mean_all, std_all, n_all), (mean_last, std_last, n_last)) for one
    setting_results entry."""
    raw = entry[3] if len(entry) >= 4 and entry[3] is not None else None
    if raw is not None and getattr(raw, "size", 0) > 0:
        arr = np.asarray(raw, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
    else:
        # Fallback: only the per-step mean curve is available (e.g. reloaded
        # from Excel without per-rep data); treat it as a single pseudo-rep.
        arr = np.asarray(entry[0], dtype=np.float64).reshape(1, -1)

    flat_all = arr.reshape(-1)
    n_pts = arr.shape[1]
    n_last = max(1, int(np.ceil(last_fraction * n_pts))) if last_fraction > 0 and n_pts > 0 else n_pts
    flat_last = arr[:, -n_last:].reshape(-1) if n_pts > 0 else np.array([])

    def stats(v):
        if v.size == 0:
            return float("nan"), float("nan"), 0
        return float(np.mean(v)), float(np.std(v, ddof=0)), int(v.size)

    return stats(flat_all), stats(flat_last)


def build_returns_summary_table(
    *,
    algo_jobs: dict[str, list[dict[str, Any]]],
    algo_job_offsets: dict[str, int],
    setting_results: list,
    n_repetitions: int,
    last_fraction: float = 0.1,
    output_dir: str = "Trial Continuation Analysis",
    print_to_stdout: bool = True,
    use_saved_disk_networks_checkpoints: bool = False,
) -> dict[str, Any]:
    """Build and emit the per-(algorithm, setting) returns summary table."""
    last_fraction = min(max(float(last_fraction), 0.0), 1.0)

    rows_data: list[dict[str, Any]] = []
    for algo_upper, jobs in algo_jobs.items():
        offset = int(algo_job_offsets.get(algo_upper, 0))
        for i, job in enumerate(jobs):
            idx = offset + i
            if idx >= len(setting_results):
                continue
            entry = setting_results[idx]
            if entry is None:
                continue
            (mean_all, std_all, n_all), (mean_last, std_last, n_last) = _aggregate_returns(
                entry, n_repetitions, last_fraction
            )
            rows_data.append({
                "algorithm": algo_upper,
                "setting": job.get("curve_label", algo_upper),
                "mean_all": mean_all,
                "std_all": std_all,
                "n_all": n_all,
                "mean_last": mean_last,
                "std_last": std_last,
                "n_last": n_last,
            })

    last_pct = int(round(last_fraction * 100))
    headers = [
        "Algorithm", "Setting",
        "Mean (all)", "Std (all)",
        f"Mean (last {last_pct}%)", f"Std (last {last_pct}%)",
        "N (all)", "N (last)",
    ]
    text_rows = [
        [
            r["algorithm"], r["setting"],
            _fmt(r["mean_all"]), _fmt(r["std_all"]),
            _fmt(r["mean_last"]), _fmt(r["std_last"]),
            f"{r['n_all']:,}", f"{r['n_last']:,}",
        ]
        for r in rows_data
    ]
    rendered = _render_aligned_table(headers, text_rows) if text_rows else "(no completed settings to summarize)"

    title = f"Returns summary - mean/std across all repetitions and eval points (n_repetitions={n_repetitions})"
    if print_to_stdout:
        def _safe_print(s: str) -> None:
            try:
                print(s)
            except UnicodeEncodeError:
                print(s.encode("ascii", errors="replace").decode("ascii"))

        _safe_print("\n" + "=" * len(title))
        _safe_print(title)
        _safe_print("=" * len(title))
        _safe_print(rendered)
        _safe_print("")

    result = {
        "rows": rows_data,
        "headers": headers,
        "text": rendered,
        "csv_path": None,
        "markdown_path": None,
    }

    # Disk artifacts only when checkpoint reuse is enabled (continuation runs).
    if not use_saved_disk_networks_checkpoints or not rows_data:
        return result

    os.makedirs(output_dir, exist_ok=True)

    # Tag the artifacts with the run id so each run's summary is kept distinct
    # and the end-of-run cleanup can group/prune them like the plots.
    from .Library_output_cleanup import stamp_run_id

    csv_path = stamp_run_id(os.path.join(output_dir, "results_summary.csv"))
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in rows_data:
            f.write(",".join([
                r["algorithm"], f"\"{r['setting']}\"",
                f"{r['mean_all']:.6f}" if np.isfinite(r["mean_all"]) else "",
                f"{r['std_all']:.6f}" if np.isfinite(r["std_all"]) else "",
                f"{r['mean_last']:.6f}" if np.isfinite(r["mean_last"]) else "",
                f"{r['std_last']:.6f}" if np.isfinite(r["std_last"]) else "",
                str(r["n_all"]), str(r["n_last"]),
            ]) + "\n")
    result["csv_path"] = csv_path

    md_path = stamp_run_id(os.path.join(output_dir, "results_summary.md"))
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join("---" for _ in headers) + " |\n")
        for row in text_rows:
            f.write("| " + " | ".join(row) + " |\n")
    result["markdown_path"] = md_path

    print(f"Saved returns summary to {csv_path} and {md_path}")
    return result
