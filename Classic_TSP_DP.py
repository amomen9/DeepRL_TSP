"""
Classic_TSP_DP.py - classic TSP Dynamic Programming experiment helpers.

This file originally had two layers:
- a single-instance runner
- a DP experiment runner with repetitions and averaging

Per requirements, the repetitions feature has been removed.
The functionality is now merged into a single function named
run_single_DP_experiment (no n_repetitions argument).
"""

import time
import io
import contextlib

import numpy as np

from DP_Agent import TSP_DP_Agent
from Environment import TSPEnvironment
from Library.Library_env_elements import format_tour, subsets_of_size
from Library.Helper_excel import load_dp_solution, save_dp_solution

_EPS = 1e-9


def _compute_held_karp_dp_table(duration_matrix: np.ndarray, reuse_table: np.ndarray | None = None, depot: int = 0) -> np.ndarray:
    """Build the Held-Karp bottom-up DP table for a duration matrix.

    ``depot`` is the 0-indexed start/end city (defaults to 0); the table is
    built over subsets of the non-depot cities.
    """
    dist = np.asarray(duration_matrix, dtype=float)
    n = len(dist)

    expected_shape = (1 << n, n)
    if reuse_table is not None and getattr(reuse_table, "shape", None) == expected_shape:
        dp = reuse_table
        dp.fill(np.nan)
    else:
        dp = np.full(expected_shape, np.nan, dtype=float)

    if n <= 1:
        return dp

    if n == 2:
        other = next(c for c in range(n) if c != depot)
        dp[1 << other, other] = dist[depot, other]
        return dp

    for l in range(n):
        if l == depot:
            continue
        dp[1 << l, l] = dist[depot, l]

    for size in range(2, n):
        for S in subsets_of_size(n, size, depot=depot):
            for l in range(n):
                if l == depot or not (S & (1 << l)):
                    continue

                S_no_l = S ^ (1 << l)
                best = np.inf

                for m in range(n):
                    if m == depot or not (S_no_l & (1 << m)):
                        continue
                    prev = dp[S_no_l, m]
                    if not np.isfinite(prev):
                        continue
                    val = prev + dist[m, l]
                    if val < best:
                        best = val

                if np.isfinite(best):
                    dp[S, l] = best

    return dp


def _infer_preview_matrix_name(label: str) -> str:
    """Infer a friendly variable-like matrix name from the experiment label."""
    lowered = label.lower()
    if "assignment" in lowered:
        return "d_example"
    if "lecture" in lowered:
        return "d_lecture"
    return "combined_matrix"


def format_duration_uncertainty_preview(
    env
) -> str:
    """Format the duration and uncertainty additions as a wide table.

    Preview display intentionally uses only:
      - duration_rows (base durations)
      - uncertainty_rows (potential uncertainty values)
    The inclusion matrix is accepted for backward compatibility but not used for
    determining whether to show the "+ unc" in the preview.
    """
    duration_rows = np.asarray(env.duration_matrix, dtype=object)
    uncertainty_rows = np.asarray(env.expected_stochastic_duration_matrix, dtype=float)

    n = duration_rows.shape[0]
    values = []
    for i in range(n):
        row_values = []
        for j in range(n):
            base = str(duration_rows[i, j])
            unc = float(uncertainty_rows[i, j])

            # Only append "+ unc" when unc is non-zero.
            if np.isfinite(unc) and not np.isclose(unc, 0.0):
                row_values.append(f"{base} + {unc:g}")
            else:
                row_values.append(base)
        values.append(row_values)

    col_labels = [str(idx + 1) for idx in range(n)]
    col_widths = [
        max(len(col_labels[j]), max(len(values[i][j]) for i in range(n)))
        for j in range(n)
    ]
    row_label_width = max(3, len(str(n)))
    table_width = row_label_width + 3 + sum(col_widths) + 3 * (n - 1)

    lines = [f"  {'=' * table_width}"]
    header_cells = " | ".join(f"{col_labels[j]:^{col_widths[j]}}" for j in range(n))
    lines.append(f"  {' ' * row_label_width} | {header_cells}")
    lines.append(
        "  "
        + "-" * row_label_width
        + "-+-"
        + "-+-".join("-" * col_widths[j] for j in range(n))
    )

    for i in range(n):
        row_label = f"{i + 1:>{row_label_width}}"
        row_cells = " | ".join(f"{values[i][j]:^{col_widths[j]}}" for j in range(n))
        lines.append(f"  {row_label} | {row_cells}")

    lines.append(f"  {'=' * table_width}")
    return "\n".join(lines)


def _format_scalar_value(value):
    """Format numeric values without trailing zeros."""
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{float(value):g}"


def _format_route_edge_costs(duration_matrix, tour):
    """Format per-edge costs for a tour.

    Environment.step assigns reward for edges chosen by the agent (n-1 actions),
    and then adds the final return-to-depot cost implicitly after the last
    city is visited.

    Therefore, for reporting "route sequential costs", we exclude the final
    implicit edge (tour[-2] -> tour[-1]).
    """
    dist = np.asarray(duration_matrix, dtype=float)
    # For a valid tour, len(tour) == n + 1 and tour[-1] == 0.
    # We exclude the last edge back to depot to produce n-1 costs.
    last_edge_index_exclusive = max(0, len(tour) - 2)
    return "→".join(
        f"{float(dist[tour[i], tour[i + 1]]):g}" for i in range(last_edge_index_exclusive)
    )


def _tour_cost_value(duration_matrix, tour):
    """Compute the total cost of a tour."""
    dist = np.asarray(duration_matrix, dtype=float)
    return float(sum(dist[tour[i], tour[i + 1]] for i in range(len(tour) - 1)))


def _build_route_equivalency_rows(det_optimal_tours, avg_tours, det_matrix, avg_matrix, avg_opt_cost):
    """Build the comparison rows for the deterministic and uncertain routes."""
    det_keys = {tuple(tour) for tour in det_optimal_tours}
    avg_keys = {tuple(tour) for tour in avg_tours}

    common_routes = [tuple(tour) for tour in det_optimal_tours if tuple(tour) in avg_keys]
    det_only_routes = [tuple(tour) for tour in det_optimal_tours if tuple(tour) not in avg_keys]
    avg_only_routes = [tuple(tour) for tour in avg_tours if tuple(tour) not in det_keys]

    common_routes.sort(
        key=lambda route: (_tour_cost_value(det_matrix, list(route)), format_tour(list(route)))
    )
    det_only_routes.sort(
        key=lambda route: (_tour_cost_value(det_matrix, list(route)), format_tour(list(route)))
    )
    avg_only_routes.sort(
        key=lambda route: (_tour_cost_value(avg_matrix, list(route)), format_tour(list(route)))
    )

    rows = []

    def add_det_row(route_tuple):
        route = list(route_tuple)
        in_avg = route_tuple in avg_keys
        det_optimal_cost = _tour_cost_value(det_matrix, route)
        avg_cost = _tour_cost_value(avg_matrix, route)

        rows.append(
            {
                "tour": route,
                "deterministic_route": format_tour(route),
                "uncertain_route": format_tour(route) if in_avg else "Not found",
                "deterministic_cost": _format_scalar_value(det_optimal_cost),
                "uncertain_cost": _format_scalar_value(avg_cost),
                "extra_cost": _format_scalar_value(avg_cost - avg_opt_cost) if np.isfinite(avg_opt_cost) else "Not found",
                "deterministic_individual_route_costs": _format_route_edge_costs(det_matrix, route),
                "uncertain_individual_route_costs": _format_route_edge_costs(avg_matrix, route),
            }
        )

    def add_avg_only_row(route_tuple):
        route = list(route_tuple)
        det_cost = _tour_cost_value(det_matrix, route)
        avg_cost = _tour_cost_value(avg_matrix, route)

        rows.append(
            {
                "tour": route,
                "deterministic_route": "Not found",
                "uncertain_route": format_tour(route),
                "deterministic_cost": _format_scalar_value(det_cost),
                "uncertain_cost": _format_scalar_value(avg_cost),
                "extra_cost": _format_scalar_value(avg_cost - avg_opt_cost) if np.isfinite(avg_opt_cost) else "Not found",
                "deterministic_individual_route_costs": _format_route_edge_costs(det_matrix, route),
                "uncertain_individual_route_costs": _format_route_edge_costs(avg_matrix, route),
            }
        )

    for route_tuple in common_routes:
        add_det_row(route_tuple)

    for route_tuple in det_only_routes:
        add_det_row(route_tuple)

    for route_tuple in avg_only_routes:
        add_avg_only_row(route_tuple)

    return rows


def _format_route_equivalency_table(
    rows,
    max_equivalency_table_rows=10,
    show_cost_chain: bool = False,
    model_rows=None,
    model_best_tour=None,
    expected_matrix=None,
):
    """Render the route comparison table as a polished aligned block.

    Parameters
    ----------
    rows : list[dict]
        DP equivalency rows produced by ``_build_route_equivalency_rows``.
    model_rows : list[dict] or None
        Optional rows describing trained-model evaluations to append at the
        bottom of the table. Each dict may contain: ``row_label``,
        ``deterministic_route``, ``uncertain_route``, ``model_route``,
        ``deterministic_cost``, ``uncertain_cost``, ``model_cost``,
        ``extra_cost``, ``model_mean``, ``model_std``,
        ``deterministic_individual_route_costs``,
        ``uncertain_individual_route_costs``, ``mod_route_costs``.
    model_best_tour : list[int] or None
        Best tour the trained model produced (lowest cost on
        ``expected_matrix``). When supplied together with ``expected_matrix``,
        every row's ``Extra cost`` cell is recomputed as
        ``cost(row_tour, expected_matrix) - cost(model_best_tour, expected_matrix)``.
    expected_matrix : np.ndarray or None
        Matrix used as the reference for the recomputed ``Extra cost``.
    """
    # If model_rows are appended, allow the caller to store all repetition costs
    # as rep_1..rep_n (instead of Model Mean/Std).
    appended_model_rows = list(model_rows) if model_rows else []
    rep_nums: list[int] = []
    for m_row in appended_model_rows:
        for k in m_row.keys():
            if not isinstance(k, str) or not k.startswith("rep_"):
                continue
            suffix = k[len("rep_"):]
            if suffix.isdigit():
                rep_nums.append(int(suffix))
    rep_nums = sorted(set(rep_nums))
    rep_cols = [f"rep_{n}" for n in rep_nums]

    headers = [
        "row",
        "Det Route",
        "Unc Route",
        "Model Route",
        "Det Cost",
        "Unc Cost",
        "Model Cost",
        "Extra cost if followed the det route",
        *rep_cols,
    ]
    if show_cost_chain:
        headers.extend(
            [
                "Det Route Costs",
                "Unc Route Costs",
                "Mod Route Costs",
            ]
        )

    display_rows = rows[:max_equivalency_table_rows]

    extra_baseline = None
    if model_best_tour is not None and expected_matrix is not None:
        extra_baseline = _tour_cost_value(expected_matrix, model_best_tour)

    def _row_extra_cost(row, fallback):
        if extra_baseline is None:
            return fallback
        tour = row.get("tour")
        if tour is None:
            return fallback
        try:
            value = _tour_cost_value(expected_matrix, tour) - extra_baseline
        except Exception:
            return fallback
        return _format_scalar_value(value)

    def _wrap_header_text(text, max_width):
        """Wrap header text without changing its characters."""
        text = str(text)
        if max_width <= 0:
            return [text]
        if len(text) <= max_width:
            return [text]

        words = text.split()
        lines = []
        current = words[0]
        for word in words[1:]:
            candidate = current + " " + word
            if len(candidate) <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines

    table_rows = []
    for idx, row in enumerate(display_rows, start=1):
        extra_str = _row_extra_cost(row, row["extra_cost"])
        row_cells = [
            str(idx),
            row["deterministic_route"],
            row["uncertain_route"],
            "",  # Model Route (filled only for appended model rows)
            row["deterministic_cost"],
            row["uncertain_cost"],
            "",  # Model Cost
            extra_str,
            *["" for _ in rep_cols],  # per-rep model costs (model-only columns)
        ]
        if show_cost_chain:
            row_cells.extend(
                [
                    row["deterministic_individual_route_costs"],
                    row["uncertain_individual_route_costs"],
                    "",  # Mod Route Costs (filled only for appended model rows)
                ]
            )
        table_rows.append(row_cells)

    for m_idx, m_row in enumerate(appended_model_rows):
        row_label = m_row.get("row_label", f"M{m_idx + 1}")
        extra_default = m_row.get("extra_cost", "")
        extra_str = _row_extra_cost(m_row, extra_default)
        row_cells = [
            str(row_label),
            m_row.get("deterministic_route", "Not found"),
            m_row.get("uncertain_route", "Not found"),
            m_row.get("model_route", ""),
            m_row.get("deterministic_cost", ""),
            m_row.get("uncertain_cost", ""),
            m_row.get("model_cost", ""),
            extra_str,
            *[m_row.get(rep_col, "") for rep_col in rep_cols],
        ]
        if show_cost_chain:
            row_cells.extend(
                [
                    m_row.get("deterministic_individual_route_costs", ""),
                    m_row.get("uncertain_individual_route_costs", ""),
                    m_row.get("mod_route_costs", ""),
                ]
            )
        table_rows.append(row_cells)

    widths = []
    for col_idx in range(len(headers)):
        data_width = max((len(row[col_idx]) for row in table_rows), default=0)
        widths.append(max(1, data_width))

    # Right-align numeric columns:
    # 0 row index, 4..7 costs and extra cost, and per-rep columns immediately after 7.
    base_right = {0, 4, 5, 6, 7}
    rep_start_idx = 8  # after Extra cost column
    align_right = base_right.union(set(range(rep_start_idx, rep_start_idx + len(rep_cols))))

    def _render_cell(text, col_idx):
        if col_idx in align_right:
            return f"{text:>{widths[col_idx]}}"
        return f"{text:<{widths[col_idx]}}"

    def _header_needed_width(header_text, base_width):
        words = header_text.split()
        width = max(base_width, max((len(word) for word in words), default=len(header_text)))

        def _header_line_count(test_width):
            line_count = 1
            current_len = 0
            for word in words:
                word_len = len(word)
                if current_len == 0:
                    current_len = word_len
                elif current_len + 1 + word_len <= test_width:
                    current_len += 1 + word_len
                else:
                    line_count += 1
                    current_len = word_len
            return line_count

        while _header_line_count(width) > 4:
            width += 1
        return width

    for col_idx, header in enumerate(headers):
        widths[col_idx] = _header_needed_width(header, widths[col_idx])

    header_blocks = [_wrap_header_text(headers[i], widths[i]) for i in range(len(headers))]
    max_header_lines = max(len(block) for block in header_blocks)

    lines = []
    for header_line_idx in range(max_header_lines):
        rendered_headers = []
        for col_idx, block in enumerate(header_blocks):
            text = block[header_line_idx] if header_line_idx < len(block) else ""
            rendered_headers.append(f"{text:^{widths[col_idx]}}")
        lines.append(" | ".join(rendered_headers))

    separator_line = "-+-".join("-" * widths[i] for i in range(len(headers)))
    lines.append(separator_line)

    dp_row_count = len(display_rows)
    for row_idx, row in enumerate(table_rows):
        if appended_model_rows and row_idx == dp_row_count:
            lines.append(separator_line)
        lines.append(" | ".join(_render_cell(row[i], i) for i in range(len(headers))))

    if len(rows) > max_equivalency_table_rows:
        lines.append(
            f"... truncated ({len(rows) - max_equivalency_table_rows} more row(s); increase max_equivalency_table_rows to show them)"
        )

    return lines


def _print_trial_summary(
    method,
    n_repetitions,
    det_optimal_cost,
    avg_cost,
    worst_cost,
    det_optimal_tours,
    avg_tours,
    det_matrix,
    avg_matrix,
    avg_opt_cost,
    max_equivalency_table_rows=10,
    print_equivalency_table=True,
):
    """Print the compact post-preview summary requested by the experiment output."""
    print(f"#Trial repetitions = {n_repetitions}")
    print(f"Method = {method}")
    print()
    print(f"{'-' * 72}")
    print("  Optimal solutions:")
    print(f"  Optimal Deterministic tour cost: {det_optimal_cost:.2f}")
    print(f"  Optimal Uncertain     tour cost: {avg_cost:.2f}")
    print(f"  {det_optimal_cost:.2f} <= Solution Boundary <= {worst_cost:.2f}")
    print()
    if not print_equivalency_table:
        return
    print("  Deterministic and Uncertain routes equivalency table:")
    print()
    for line in _format_route_equivalency_table(
        _build_route_equivalency_rows(
            det_optimal_tours,
            avg_tours,
            det_matrix,
            avg_matrix,
            avg_opt_cost,
        ),
        max_equivalency_table_rows=max_equivalency_table_rows,
        show_cost_chain=True,
    ):
        print(line)


def _solve_dp_cases(env, method, max_optimal_tours, return_tables):
    """Solve the best (deterministic), best-expected, and worst cases via Held-Karp.

    Returns a dict of costs, tours, elapsed times, the two reference matrices,
    and optional DP-table snapshots (only when ``return_tables``). Validation
    asserts mirror the previous inline behavior. This helper prints nothing.
    """
    # One reusable Held-Karp buffer shared by all three solves (best / expected
    # / worst). The trio runs sequentially, so a single compact (2**(n-1), n-1)
    # float64 table is allocated once and refilled per instance instead of being
    # reallocated three times; only the small (cost, tours) results are retained.
    reuse_dp = None
    if method == "bottom_up" and env.n_cities > 2:
        _m = env.n_cities - 1
        reuse_dp = np.empty((1 << _m, _m), dtype=float)

    #--------------------- Best case on duration_matrix
    det_agent = TSP_DP_Agent(env, scenario_choice="deterministic")

    start = time.perf_counter()
    det_optimal_cost, det_optimal_tours = det_agent.solve(
        method=method, max_optimal_tours=max_optimal_tours, reuse_dp=reuse_dp
    )
    det_elapsed = time.perf_counter() - start

    det_matrix = env.duration_matrix
    assert det_matrix is not None

    env.destroy_state_table()
    working_dp_table = None
    best_dp_table_snapshot = None
    expected_dp_table_snapshot = None
    worst_dp_table_snapshot = None

    if return_tables:
        working_dp_table = _compute_held_karp_dp_table(det_matrix, reuse_table=working_dp_table, depot=env.current_depot_city)
        best_dp_table_snapshot = np.array(working_dp_table, copy=True)

    for t in det_optimal_tours:
        assert env.validate_tour(t), f"Invalid tour: {t}"
        recomputed = _tour_cost_value(det_matrix, t)
        assert abs(recomputed - det_optimal_cost) < _EPS, (
            f"Cost mismatch: tour_cost={recomputed}, expected={det_optimal_cost}"
        )

    #--------------------- Best case on expected_stochastic_duration_matrix
    expected_agent = TSP_DP_Agent(env, scenario_choice="expected")

    start = time.perf_counter()
    expected_cost, expected_tours = expected_agent.solve(
        method=method, max_optimal_tours=max_optimal_tours, reuse_dp=reuse_dp
    )
    expected_elapsed = time.perf_counter() - start

    expected_matrix = env.expected_stochastic_duration_matrix
    assert expected_matrix is not None

    env.destroy_state_table()
    if return_tables:
        working_dp_table = _compute_held_karp_dp_table(expected_matrix, reuse_table=working_dp_table, depot=env.current_depot_city)
        expected_dp_table_snapshot = np.array(working_dp_table, copy=True)

    for t in expected_tours:
        assert env.validate_tour(t), f"Invalid tour: {t}"
        recomputed = _tour_cost_value(expected_matrix, t)
        assert abs(recomputed - expected_cost) < _EPS, (
            f"Cost mismatch: tour_cost={recomputed}, expected={expected_cost}"
        )

    #--------------------- Worst case on duration_matrix (maximize cost)
    worst_matrix = env.duration_matrix
    assert worst_matrix is not None

    worst_agent = TSP_DP_Agent(env, scenario_choice="deterministic")

    start = time.perf_counter()
    worst_cost, worst_tours = worst_agent.solve(
        method=method,
        max_optimal_tours=max_optimal_tours,
        objective="max",
        reuse_dp=reuse_dp,
    )
    worst_elapsed = time.perf_counter() - start

    env.destroy_state_table()
    if return_tables:
        working_dp_table = _compute_held_karp_dp_table(worst_matrix, reuse_table=working_dp_table, depot=env.current_depot_city)
        worst_dp_table_snapshot = np.array(working_dp_table, copy=True)

    for t in worst_tours:
        assert env.validate_tour(t), f"Invalid tour: {t}"
        recomputed = _tour_cost_value(worst_matrix, t)
        assert abs(recomputed - worst_cost) < _EPS, (
            f"Cost mismatch: tour_cost={recomputed}, expected={worst_cost}"
        )

    return {
        "det_optimal_cost": det_optimal_cost,
        "det_optimal_tours": det_optimal_tours,
        "det_elapsed": det_elapsed,
        "best_dp_table_snapshot": best_dp_table_snapshot,
        "expected_cost": expected_cost,
        "expected_tours": expected_tours,
        "expected_elapsed": expected_elapsed,
        "expected_dp_table_snapshot": expected_dp_table_snapshot,
        "worst_cost": worst_cost,
        "worst_tours": worst_tours,
        "worst_elapsed": worst_elapsed,
        "worst_dp_table_snapshot": worst_dp_table_snapshot,
        "det_matrix": det_matrix,
        "expected_matrix": expected_matrix,
    }


def _validate_cached_solution(env, cached, det_matrix, expected_matrix, tol=1e-6):
    """Return True iff every cached tour is valid for ``env`` and its recomputed
    cost matches the cached cost. A False result means the cache is stale (e.g.
    the trio was regenerated) and must be recomputed.
    """
    checks = [
        (cached.get("best_tours"), cached.get("best_cost"), det_matrix),
        (cached.get("worst_tours"), cached.get("worst_cost"), det_matrix),
        (cached.get("expected_tours"), cached.get("expected_cost"), expected_matrix),
    ]
    for tours, cost, matrix in checks:
        if not tours or cost is None:
            return False
        for tour in tours:
            if not env.validate_tour(tour):
                return False
            if abs(_tour_cost_value(matrix, tour) - float(cost)) > tol:
                return False
    return True


def run_single_DP_experiment(
    env,
    method="bottom_up",
    max_optimal_tours=100,
    max_equivalency_table_rows=10,
    seed=None,
    return_tables=False,
    print_report=True,
    print_equivalency_table=True,
    sample_data_id=None,
    sample_data_timestamp=None,
    sample_data_dimension=None,
):
    """|Immutable| Run DP on one best-case baseline, one best expected case, and one global worst-case instance.

    Repetitions and averaging have been removed.

    Returns a dict compatible with the previous "deterministic/average" shape,
    plus an additional top-level key:
      - report: a string containing everything this function printed
                (including prints from nested helper calls).

    DP-solution caching
    -------------------
    When ``sample_data_id`` and ``sample_data_dimension`` are supplied (the
    on-disk trio the environment was built from), the best / worst / expected
    solutions are cached in a sibling ``duration_matrix_DP-Solution_*`` workbook.
    On a later run for the same trio + dimension the cached tours are read back
    and the (2**n) DP solve is skipped entirely. Caching is disabled when
    ``return_tables`` is True (the DP tables must be materialized).
    """
    report_buffer = io.StringIO()
    result = None

    # DP-solution caching is keyed on the on-disk trio (id + dimension). It is
    # disabled when the DP tables must be materialized (return_tables).
    cache_enabled = (
        sample_data_id is not None
        and sample_data_dimension is not None
        and not return_tables
    )

    with contextlib.redirect_stdout(report_buffer):
        n_cities = env.n_cities
        label = getattr(env, "label", None) or f"Example ({n_cities} cities)"

        effective_seed = getattr(env, "seed", None)
        if seed is not None:
            effective_seed = seed

        print(f"\n{'=' * 80}")
        print("  TSP Dynamic Programming Solver")
        print(f"{'=' * 80}")
        print("  DP experiment for {0} cities (best duration, best expected, and global worst duration runs)".format(n_cities))
        print(f"{'=' * 80}")

        matrix_name = _infer_preview_matrix_name(label)
        print("expected_stochastic_duration_matrix:\n")
        print(
            format_duration_uncertainty_preview(
                env=env
            )
        )
        print()
        print("worst_duration_matrix:\n")
        worst_matrix_preview = np.asarray(env.duration_matrix, dtype=float)
        print(worst_matrix_preview)
        print()

        det_matrix = env.duration_matrix
        expected_matrix = env.expected_stochastic_duration_matrix
        assert det_matrix is not None and expected_matrix is not None

        best_dp_table_snapshot = None
        expected_dp_table_snapshot = None
        worst_dp_table_snapshot = None

        cached = (
            load_dp_solution(sample_data_id, sample_data_dimension)
            if cache_enabled
            else None
        )
        cache_hit = cached is not None and _validate_cached_solution(
            env, cached, det_matrix, expected_matrix
        )

        if cache_hit:
            det_optimal_cost = cached["best_cost"]
            det_optimal_tours = [list(t) for t in cached["best_tours"]]
            expected_cost = cached["expected_cost"]
            expected_tours = [list(t) for t in cached["expected_tours"]]
            worst_cost = cached["worst_cost"]
            worst_tours = [list(t) for t in cached["worst_tours"]]
            det_elapsed = expected_elapsed = worst_elapsed = 0.0
            print(
                f"[DP cache] Loaded best/worst/expected solutions for dimension "
                f"{sample_data_dimension} (id {sample_data_id}) from cache; "
                f"skipping DP computation."
            )
            env.destroy_state_table()
        else:
            # Without a DP table there is nothing to compute (only reachable on a
            # cache miss; a cache hit above needs no table).
            if env.initialize_dp_table is None:
                print("Memory not enough to initialize DP table. DP experiment skipped.")
                return None
            elif env.initialize_dp_table is False:
                print("DP table initialization skipped by configuration. DP experiment skipped.")
                return None

            solved = _solve_dp_cases(env, method, max_optimal_tours, return_tables)
            det_optimal_cost = solved["det_optimal_cost"]
            det_optimal_tours = solved["det_optimal_tours"]
            det_elapsed = solved["det_elapsed"]
            best_dp_table_snapshot = solved["best_dp_table_snapshot"]
            expected_cost = solved["expected_cost"]
            expected_tours = solved["expected_tours"]
            expected_elapsed = solved["expected_elapsed"]
            expected_dp_table_snapshot = solved["expected_dp_table_snapshot"]
            worst_cost = solved["worst_cost"]
            worst_tours = solved["worst_tours"]
            worst_elapsed = solved["worst_elapsed"]
            worst_dp_table_snapshot = solved["worst_dp_table_snapshot"]
            det_matrix = solved["det_matrix"]
            expected_matrix = solved["expected_matrix"]

            if cache_enabled:
                try:
                    saved_path = save_dp_solution(
                        sample_data_id,
                        sample_data_dimension,
                        {
                            "best_cost": det_optimal_cost,
                            "best_tours": det_optimal_tours,
                            "worst_cost": worst_cost,
                            "worst_tours": worst_tours,
                            "expected_cost": expected_cost,
                            "expected_tours": expected_tours,
                        },
                        timestamp=sample_data_timestamp,
                    )
                    print(
                        f"[DP cache] Saved best/worst/expected solutions for dimension "
                        f"{sample_data_dimension} (id {sample_data_id}) to '{saved_path}'."
                    )
                except Exception as exc:
                    print(f"[DP cache] Failed to save DP solution: {exc}")

        det_matrix_for_report = det_matrix
        avg_matrix_for_report = expected_matrix

        _print_trial_summary(
            method=method,
            n_repetitions=1,
            det_optimal_cost=det_optimal_cost,
            avg_cost=expected_cost,
            worst_cost=worst_cost,
            det_optimal_tours=det_optimal_tours,
            avg_tours=expected_tours,
            det_matrix=det_matrix_for_report,
            avg_matrix=avg_matrix_for_report,
            avg_opt_cost=expected_cost,
            max_equivalency_table_rows=max_equivalency_table_rows,
            print_equivalency_table=print_equivalency_table,
        )

        result = {
            "deterministic": {
                "cost": det_optimal_cost,
                "tours": det_optimal_tours,
                "elapsed": det_elapsed,
                "dp_table": best_dp_table_snapshot if best_dp_table_snapshot is not None else np.array([]),
            },
            "average": {
                "cost": expected_cost,
                "tours": expected_tours,
                "elapsed": expected_elapsed,
                "dp_table_mean": expected_dp_table_snapshot if expected_dp_table_snapshot is not None else np.array([]),
                "dp_table_var": np.zeros_like(expected_dp_table_snapshot) if expected_dp_table_snapshot is not None else np.array([]),
                "repetitions": [{"seed": effective_seed}],
            },
            "worst": {
                "cost": worst_cost,
                "tours": worst_tours,
                "elapsed": worst_elapsed,
                "dp_table": worst_dp_table_snapshot if worst_dp_table_snapshot is not None else np.array([]),
            },
            "equivalency_data": {
                "det_optimal_tours": [list(t) for t in det_optimal_tours],
                "avg_tours": [list(t) for t in expected_tours],
                "det_matrix": np.asarray(det_matrix_for_report, dtype=float),
                "avg_matrix": np.asarray(avg_matrix_for_report, dtype=float),
                "avg_opt_cost": float(expected_cost) if expected_cost is not None else float("inf"),
                "max_equivalency_table_rows": int(max_equivalency_table_rows),
            },
            "report": report_buffer.getvalue(),
        }

    report = report_buffer.getvalue()
    if print_report:
        print(report, end="")

    if result is not None:
        result["report"] = report
    return result


if __name__ == "__main__":
    # Simple smoke test
    duration_mat = [
        [0, 10, 12, 19, 8],
        [10, 0, 5, 7, 11],
        [12, 5, 0, 9, 7],
        [19, 7, 9, 0, 3],
        [8, 11, 7, 3, 0],
    ]
    duration_mat = np.array(duration_mat, dtype=float)

    env = TSPEnvironment(
        duration_matrix=duration_mat,
        potential_uncertainty_matrix=np.zeros_like(duration_mat, dtype=float),
        uncertain_routes=None,
        initialize_dp_table=True,
        seed=0,
    )
    setattr(env, "label", "Example (5 cities)")

    res = run_single_DP_experiment(
        env,
        method="bottom_up",
        max_optimal_tours=10,
        max_equivalency_table_rows=10,
        return_tables=False,
    )
    assert res is not None
    print("OK", res["deterministic"]["cost"], res["average"]["cost"])
