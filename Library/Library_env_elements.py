"""
Library_problem.py - TSP problem instance generation and tour helpers.

Contents
--------
Random-instance generators:
    stochasticity_distribution_sampler
    generate_random_duration_matrix
    generate_random_potential_uncertainty_matrix
    generate_dynamic_noise_matrix

Tour and subset utilities:
    format_tour
    tour_cost
    subsets_of_size
    print_duration_matrix
"""
import numpy as np
import random
import itertools
import json
import hashlib


# ── Instance-identity signature (duration / inclusion / potential-uncertainty) ──
#
# The three matrices below fully define the *deterministic* structure of a TSP
# instance (the per-episode noise is reseeded separately and is not part of the
# instance identity). Results and network checkpoints saved to disk are only
# valid for the instance they were produced on, so every disk-load matching rule
# keys on these three matrices. We keep the full matrices as human-readable text
# (for provenance) but compare a compact hash of them (robust to float
# formatting / row ordering, cheap to store and equate).

_INSTANCE_MATRIX_KEYS = (
    "duration_matrix",
    "uncertainty_inclusion_matrix",
    "potential_uncertainty_matrix",
)


def matrices_text_and_hash(
    duration_matrix,
    uncertainty_inclusion_matrix,
    potential_uncertainty_matrix,
    *,
    decimals: int = 6,
):
    """Return ``(text, hash)`` capturing the three instance-defining matrices.

    ``text`` is a stable, human-readable JSON dump of the full (rounded) matrices;
    ``hash`` is a SHA1 over that same canonical text. Values are rounded to
    ``decimals`` places first so that cosmetic float noise does not change the
    signature. Returns ``(None, None)`` if any matrix is missing.
    """
    if (
        duration_matrix is None
        or uncertainty_inclusion_matrix is None
        or potential_uncertainty_matrix is None
    ):
        return None, None

    def _rounded_list(matrix):
        return np.round(np.asarray(matrix, dtype=float), decimals).tolist()

    payload = {
        "duration_matrix": _rounded_list(duration_matrix),
        "uncertainty_inclusion_matrix": _rounded_list(uncertainty_inclusion_matrix),
        "potential_uncertainty_matrix": _rounded_list(potential_uncertainty_matrix),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return text, digest


def env_matrices_text_and_hash(env, *, decimals: int = 6):
    """Convenience wrapper: compute :func:`matrices_text_and_hash` from an
    environment exposing ``duration_matrix``, ``uncertainty_inclusion_matrix``
    and ``potential_uncertainty_matrix``. Returns ``(None, None)`` when ``env``
    is missing any of them."""
    if env is None:
        return None, None
    return matrices_text_and_hash(
        getattr(env, "duration_matrix", None),
        getattr(env, "uncertainty_inclusion_matrix", None),
        getattr(env, "potential_uncertainty_matrix", None),
        decimals=decimals,
    )


def stochasticity_distribution_sampler(lower_bound=0, upper_bound=1, distribution_type="uniform", seed=None):
    """Sample a value based on the given stochasticity distribution."""

    if seed is not None:
        random.seed(seed)
    else:
        random.seed()

    if distribution_type == "uniform":
        return random.uniform(lower_bound, upper_bound)
    elif distribution_type == "normal":
        mean = (lower_bound + upper_bound) / 2
        stddev = (upper_bound - lower_bound) / 4  # 95% of values within bounds
        sample = random.gauss(mean, stddev)
        return max(lower_bound, min(sample, upper_bound))  # Clamp to bounds
    else:
        raise ValueError(f"Unsupported distribution type: {distribution_type}")


def generate_random_duration_matrix(n, min_durn=1, max_durn=100, symmetric=True, seed=None):
    """Generate a random duration matrix for n cities."""
    if seed is not None:
        random.seed(seed)
    else:
        random.seed()

    duration_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = int(
                stochasticity_distribution_sampler(
                    lower_bound=min_durn,
                    upper_bound=max_durn + 1,
                    distribution_type="uniform",
                )
            )
            duration_matrix[i][j] = d
            if symmetric:
                duration_matrix[j][i] = d
            else:
                duration_matrix[j][i] = int(
                    stochasticity_distribution_sampler(
                        lower_bound=min_durn,
                        upper_bound=max_durn + 1,
                        distribution_type="uniform",
                    )
                )
    return duration_matrix


def generate_random_potential_uncertainty_matrix(
    duration_matrix,
    min_uncertainty=0.0,
    max_uncertainty=10.0,
    uncertainty_scale=1.0,
    uncertainty_symmetric=False,
    seed=None,
):
    """Generate a random uncertainty matrix for n cities."""
    n = len(duration_matrix)
    min_element = np.min(duration_matrix)
    max_element = np.max(duration_matrix)
    if seed is not None:
        random.seed(seed)
    else:
        random.seed()

    unc_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            u = stochasticity_distribution_sampler(
                lower_bound=min_element,
                upper_bound=max_element,
                distribution_type="uniform",
            ) * uncertainty_scale
            u = round(u, 2)  # Round to 2 decimal places for cleaner output
            if u < min_uncertainty:
                u = min_uncertainty
            elif u > max_uncertainty:
                u = max_uncertainty
            unc_matrix[i][j] = u
            if uncertainty_symmetric:
                unc_matrix[j][i] = u
            else:
                u = stochasticity_distribution_sampler(
                    lower_bound=min_element,
                    upper_bound=max_element,
                    distribution_type="uniform",
                ) * uncertainty_scale
                u = round(u, 2)  # Round to 2 decimal places for cleaner output
                if u < min_uncertainty:
                    u = min_uncertainty
                elif u > max_uncertainty:
                    u = max_uncertainty
                unc_matrix[j][i] = u

    return unc_matrix


def generate_random_inclusion_matrix(n, n_uncertain_routes, symmetric=True, seed=None):
    """Generate a uniform random n x n uncertainty inclusion matrix of 0s and 1s.

    The diagonal is always 0. If symmetric, exactly n_uncertain_routes 1s appear
    above the diagonal and the same count below (mirrored). If not symmetric,
    exactly n_uncertain_routes 1s appear across all off-diagonal cells.
    """
    if seed is not None:
        random.seed(seed)
    else:
        random.seed()

    uncertainty_inclusion_matrix = [[0] * n for _ in range(n)]

    if symmetric:
        # Cells strictly above the diagonal; mirror each chosen cell below.
        upper_cells = [(i, j) for i in range(n) for j in range(i + 1, n)]
        if n_uncertain_routes > len(upper_cells):
            raise ValueError(
                f"n_uncertain_routes={n_uncertain_routes} exceeds the number of "
                f"available upper-triangular routes ({len(upper_cells)}) for n={n}."
            )
        for i, j in random.sample(upper_cells, n_uncertain_routes):
            uncertainty_inclusion_matrix[i][j] = 1
            uncertainty_inclusion_matrix[j][i] = 1
    else:
        # All off-diagonal cells (diagonal stays 0).
        off_diagonal_cells = [
            (i, j) for i in range(n) for j in range(n) if i != j
        ]
        if n_uncertain_routes > len(off_diagonal_cells):
            raise ValueError(
                f"n_uncertain_routes={n_uncertain_routes} exceeds the number of "
                f"available off-diagonal routes ({len(off_diagonal_cells)}) for n={n}."
            )
        for i, j in random.sample(off_diagonal_cells, n_uncertain_routes):
            uncertainty_inclusion_matrix[i][j] = 1

    return uncertainty_inclusion_matrix


def inclusion_matrix_to_uncertain_routes(uncertainty_inclusion_matrix):
    """Convert an inclusion matrix into a 1-indexed ``uncertain_routes`` list.

    Each cell (i, j) holding a truthy value yields the route (i + 1, j + 1),
    matching the 1-indexed city convention used by the environment.
    """
    routes = []
    for i, row in enumerate(uncertainty_inclusion_matrix):
        for j, value in enumerate(row):
            if value:
                routes.append((i + 1, j + 1))
    return routes


def generate_dynamic_noise_matrix(
    potential_uncertainty_matrix,
    min_uncertainty=0.0,
    max_uncertainty=10.0,
    uncertainty_symmetric=False,
    seed=None,
):
    """Generate a random noise matrix for n cities.

    Noise for element (i,j) is sampled uniformly from [0, potential_uncertainty_matrix[i][j]]
    and then clamped to [min_uncertainty, max_uncertainty].
    """
    n = len(potential_uncertainty_matrix)
    if seed is not None:
        random.seed(seed)
    else:
        random.seed()

    unc_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            u = stochasticity_distribution_sampler(
                lower_bound=0,
                upper_bound=potential_uncertainty_matrix[i][j],
                distribution_type="uniform",
            )
            u = round(u, 2)  # Round to 2 decimal places for cleaner output
            if u < min_uncertainty:
                u = min_uncertainty
            elif u > max_uncertainty:
                u = max_uncertainty
            unc_matrix[i][j] = u

            if uncertainty_symmetric:
                unc_matrix[j][i] = u
            else:
                u = stochasticity_distribution_sampler(
                    lower_bound=0,
                    upper_bound=potential_uncertainty_matrix[j][i],
                    distribution_type="uniform",
                )
                u = round(u, 2)  # Round to 2 decimal places for cleaner output
                if u < min_uncertainty:
                    u = min_uncertainty
                elif u > max_uncertainty:
                    u = max_uncertainty
                unc_matrix[j][i] = u

    return unc_matrix


def format_tour(tour):
    """Format a tour as '1→2→3→...→1' (1-indexed display)."""
    return "→".join(str(city + 1) for city in tour)


def tour_cost(duration_matrix, tour):
    """Calculate total cost of a tour."""
    return sum(duration_matrix[tour[i]][tour[i + 1]] for i in range(len(tour) - 1))


def subsets_of_size(n, size, depot=0):
    """Generate all subset codes for subsets of the non-depot cities of a given size.

    Each subset is encoded as an integer where 2**j is added for each city j in the subset.
    The depot city (0-indexed, default 0) is never included.
    """
    cities = [c for c in range(n) if c != depot]
    for combo in itertools.combinations(cities, size):
        subset_cities_code = 0
        for c in combo:
            subset_cities_code += 2**c
        yield subset_cities_code
