"""
Library_aggregation.py - Cross-repetition aggregation helpers.

Contents
--------
data_cluster             - 1-D natural-breaks clustering of per-rep values.
_expand_k_order_configs  - Expands ``k_order_aggregation_methods`` into curves.
"""
import numpy as np


# ── 1-D proximity clustering for k_order_aggregation_methods ─────────────────

def data_cluster(input_data, number_of_clusters, cluster_choice):
    """Cluster a 1-D set of n_repetitions values by proximity and return the
    rep_numbers (indices into ``input_data``) of the chosen cluster.

    The clustering algorithm is the natural-breaks split: sort the values, then
    cut the sorted sequence at the (number_of_clusters - 1) largest consecutive
    gaps. The resulting clusters are contiguous in value and already sorted by
    value (smallest cluster first, largest last).

    Parameters
    ----------
    input_data : sequence of float, length n_repetitions
        Ordered (by rep_number) per-repetition values to cluster.
    number_of_clusters : int
        Number of clusters. Must satisfy 1 <= number_of_clusters <= n_repetitions.
    cluster_choice : {"min", "max", "median"}
        "min" returns the smallest-values cluster, "max" the largest, "median"
        the middle (index ``len(clusters)//2``) cluster.

    Returns
    -------
    np.ndarray
        Sorted array of rep_numbers belonging to the chosen cluster.
    """
    arr = np.asarray(input_data, dtype=np.float64)
    n = int(arr.size)
    if not (1 <= int(number_of_clusters) <= n):
        raise ValueError(
            f"number_of_clusters must satisfy 1 <= k <= n_repetitions={n}, got {number_of_clusters}"
        )
    if cluster_choice not in ("min", "max", "median"):
        raise ValueError(
            f"cluster_choice must be one of 'min', 'max', 'median'; got {cluster_choice!r}"
        )
    k = int(number_of_clusters)

    sort_idx = np.argsort(arr, kind="stable")
    sorted_vals = arr[sort_idx]

    if k == 1:
        groups_sorted = [sort_idx]
    elif k == n:
        groups_sorted = [np.array([i], dtype=np.int64) for i in sort_idx]
    else:
        gaps = np.diff(sorted_vals)
        # (k-1) largest gaps mark the split positions; sort ascending so np.split slices left→right.
        split_positions = np.sort(np.argpartition(gaps, -(k - 1))[-(k - 1):]) + 1
        groups_sorted = np.split(sort_idx, split_positions)

    if cluster_choice == "min":
        chosen = groups_sorted[0]
    elif cluster_choice == "max":
        chosen = groups_sorted[-1]
    else:  # "median"
        chosen = groups_sorted[len(groups_sorted) // 2]
    return np.sort(np.asarray(chosen, dtype=np.int64))


def _expand_k_order_configs(k_order_cfg, raw_returns, timesteps, n_repetitions):
    """Expand ``k_order_aggregation_methods`` into a list of plot-curve configs.

    Returns
    -------
    list of (label_suffix: str, rep_indices: np.ndarray)
        Each entry produces one curve: mean over ``raw_returns[rep_indices, :]``.
        ``label_suffix`` is "" when this is the default plain-mean fallback so
        callers can preserve the original (no-suffix) curve label.

    Falls back to a single (",", arange(n)) entry when:
      - k_order_cfg is not a dict, OR
      - raw_returns is None/empty (no per-rep data available), OR
      - the cartesian expansion produces no valid curves.
    """
    default = [("", np.arange(int(n_repetitions), dtype=np.int64))]

    if not isinstance(k_order_cfg, dict):
        return default
    if raw_returns is None:
        return default
    raw = np.asarray(raw_returns, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[0] != n_repetitions or raw.shape[1] == 0:
        return default

    def _as_list(v):
        if isinstance(v, (list, tuple, np.ndarray)):
            return list(v)
        return [v]

    # legend_parameters mirrors the algo-config legend pattern:
    # {key: [label_prefix, show_flag]}. Missing keys fall back to hardcoded
    # defaults so older configs without a legend block still produce labels.
    raw_legend = k_order_cfg.get("legend_parameters")
    legend = raw_legend if isinstance(raw_legend, dict) else {}
    _DEFAULT_LEGEND = {
        "inclusion_or_exclusion_mode": ("", True),
        "ordering_timesteps": ("@", True),
        "mean": ("mean", True),
        "k_order_max": ("k_order_max=", True),
        "k_order_min": ("k_order_min=", True),
        "k_order_median": ("k_order_median=", True),
        "k_order_cluster": ("cluster k=", True),
    }

    def _legend_for(key):
        entry = legend.get(key)
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            return str(entry[0]), bool(entry[1])
        return _DEFAULT_LEGEND.get(key, ("", False))

    def _fmt_val(v):
        from .Helper_legend import _fmt_legend

        if isinstance(v, bool):
            return str(v)
        return _fmt_legend(v)

    def _build_suffix(method_key, *, k=None, cluster_choice=None, ot=None, mode=None):
        parts: list[str] = []
        m_label, m_show = _legend_for(method_key)
        if m_show:
            if method_key == "mean":
                parts.append(m_label)
            elif method_key == "k_order_cluster":
                val = f"{_fmt_val(k)}-{cluster_choice}" if cluster_choice is not None else _fmt_val(k)
                parts.append(f"{m_label}{val}")
            else:
                parts.append(f"{m_label}{_fmt_val(k)}")
        if ot is not None:
            ot_label, ot_show = _legend_for("ordering_timesteps")
            if ot_show:
                parts.append(f"{ot_label}{_fmt_val(float(ot))}")
        if mode is not None:
            mode_label, mode_show = _legend_for("inclusion_or_exclusion_mode")
            if mode_show:
                parts.append(f"{mode_label}{mode[:3]}")
        return ", ".join(parts) if parts else method_key

    enabled_vals = _as_list(k_order_cfg.get("enabled", [True]))
    # Cartesian over enabled: False entries skip; we just need at least one True to do any work.
    if not any(bool(v) for v in enabled_vals):
        return default
    # Number of True entries in `enabled` (cartesian dimension multiplier).
    n_enabled_true = sum(1 for v in enabled_vals if bool(v))

    inc_exc_modes = [m for m in _as_list(k_order_cfg.get("inclusion_or_exclusion_mode", ["inclusion"]))
                     if m in ("inclusion", "exclusion")]
    if not inc_exc_modes:
        inc_exc_modes = ["inclusion"]

    ts_arr = np.asarray(timesteps, dtype=np.float64)

    # ``ordering_timesteps`` accepts the sentinel string "n_timesteps" (or an
    # empty list, which defaults to ["n_timesteps"]) to mean the final eval
    # point - i.e. rank reps by their value at the end of training.
    ordering_ts_raw = _as_list(k_order_cfg.get("ordering_timesteps", []))
    if not ordering_ts_raw:
        ordering_ts_raw = ["n_timesteps"]

    def _resolve_ordering_ts(v):
        if isinstance(v, str) and v.strip().lower() == "n_timesteps":
            return float(ts_arr[-1]) if ts_arr.size > 0 else 0.0
        return float(v)

    ordering_ts = [_resolve_ordering_ts(t) for t in ordering_ts_raw]

    def _find_ts_idx(ot):
        if ts_arr.size == 0:
            return 0
        return int(np.argmin(np.abs(ts_arr - float(ot))))

    def _apply_inc_exc(rep_idx, mode):
        if mode == "exclusion":
            keep = np.setdiff1d(np.arange(n_repetitions, dtype=np.int64), np.asarray(rep_idx, dtype=np.int64))
            return keep
        return np.asarray(rep_idx, dtype=np.int64)

    results = []

    # `mean` sub-method: plain mean over all reps; ignores ordering_timesteps and inclusion/exclusion.
    mean_enabled = any(bool(v) for v in _as_list(k_order_cfg.get("mean", [False])))
    if mean_enabled:
        mean_suffix = _build_suffix("mean") or "mean"
        for _ in range(n_enabled_true):
            results.append((mean_suffix, np.arange(n_repetitions, dtype=np.int64)))

    # k_order_max / k_order_min / k_order_median
    rank_methods = [
        ("k_order_max", "max"),
        ("k_order_min", "min"),
        ("k_order_median", "median"),
    ]
    for cfg_key, op in rank_methods:
        method_cfg = k_order_cfg.get(cfg_key)
        if not (isinstance(method_cfg, (list, tuple)) and len(method_cfg) == 2):
            continue
        m_enabled_list, k_list = method_cfg
        m_enabled_list = _as_list(m_enabled_list)
        k_list = _as_list(k_list)
        n_m_true = sum(1 for v in m_enabled_list if bool(v))
        if n_m_true == 0:
            continue
        for ot in ordering_ts:
            t_idx = _find_ts_idx(ot)
            values_at_t = raw[:, t_idx]
            order_asc = np.argsort(values_at_t, kind="stable")
            for k in k_list:
                k_int = int(k)
                if k_int < 1 or k_int > n_repetitions:
                    continue
                if op == "max":
                    selected = order_asc[-k_int:]
                elif op == "min":
                    selected = order_asc[:k_int]
                else:  # median: k middle reps centered on the median rank
                    mid = n_repetitions // 2
                    half = k_int // 2
                    lo = max(0, mid - half)
                    hi = lo + k_int
                    if hi > n_repetitions:
                        hi = n_repetitions
                        lo = hi - k_int
                    selected = order_asc[lo:hi]
                selected = np.sort(np.asarray(selected, dtype=np.int64))
                for mode in inc_exc_modes:
                    final_idx = _apply_inc_exc(selected, mode)
                    if final_idx.size == 0:
                        continue
                    label = _build_suffix(cfg_key, k=k_int, ot=ot, mode=mode)
                    for _ in range(n_enabled_true * n_m_true):
                        results.append((label, final_idx))

    # k_order_cluster: (enabled_list, [(num_clusters_list, cluster_choice_or_list), ...])
    cluster_cfg = k_order_cfg.get("k_order_cluster")
    if isinstance(cluster_cfg, (list, tuple)) and len(cluster_cfg) == 2:
        c_enabled_list, pairs_list = cluster_cfg
        c_enabled_list = _as_list(c_enabled_list)
        n_c_true = sum(1 for v in c_enabled_list if bool(v))
        if n_c_true > 0 and isinstance(pairs_list, (list, tuple)):
            for ot in ordering_ts:
                t_idx = _find_ts_idx(ot)
                values_at_t = raw[:, t_idx]
                for pair in pairs_list:
                    if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                        continue
                    k_list, choice_list = pair
                    k_list = _as_list(k_list)
                    choice_list = _as_list(choice_list)
                    for k in k_list:
                        try:
                            k_int = int(k)
                        except (TypeError, ValueError):
                            continue
                        for choice in choice_list:
                            try:
                                selected = data_cluster(values_at_t, k_int, str(choice))
                            except Exception as exc:
                                print(f"[k_order_cluster] skipping (k={k_int}, choice={choice!r}) @{ot:g}: {exc}")
                                continue
                            for mode in inc_exc_modes:
                                final_idx = _apply_inc_exc(selected, mode)
                                if final_idx.size == 0:
                                    continue
                                label = _build_suffix(
                                    "k_order_cluster",
                                    k=k_int,
                                    cluster_choice=str(choice),
                                    ot=ot,
                                    mode=mode,
                                )
                                for _ in range(n_enabled_true * n_c_true):
                                    results.append((label, final_idx))

    return results if results else default
