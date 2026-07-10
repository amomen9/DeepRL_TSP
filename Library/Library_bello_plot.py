"""
Library_bello_plot.py - reconstruct 2-D coordinates from a (possibly asymmetric)
duration matrix *after* a tour has been found, and plot that tour.

The re-architected Bello learns on a duration matrix, not on coordinates, so it
has no coordinates to draw. Per the agreed procedure, coordinates are
reconstructed only *after* the efficient route is known (by Bello or any other
algorithm):

  1. Copy the duration matrix into ``temp_duration_matrix``.
  2. Make it symmetric while preserving the found route:
     - for every edge on the found tour, keep its traversed cost and mirror it
       across the diagonal (``temp[a, b] = temp[b, a] = D[a, b]``);
     - for every other off-diagonal pair, pick one side at random and copy it to
       the other (random symmetrisation of the unused entries).
  3. Embed the now-symmetric ``temp_duration_matrix`` into 2-D Euclidean
     coordinates with classical (Torgerson) multidimensional scaling.

These coordinates are for visualisation only; they are never used for learning.
"""

import os

import numpy as np


def _route_directed_edges(tour):
    """Directed edges of a cyclic tour given as a permutation of city indices."""
    n = len(tour)
    return [(int(tour[i]), int(tour[(i + 1) % n])) for i in range(n)]


def reconstruct_coordinates_from_route(duration_matrix, tour, *, seed=None):
    """Return ``(coords, temp_duration_matrix)``.

    ``coords`` is an ``(n, 2)`` array of Euclidean coordinates obtained by
    classical MDS of the route-preserving symmetrised matrix described in the
    module docstring. Works for any algorithm's found ``tour``.
    """
    D = np.asarray(duration_matrix, dtype=float)
    n = D.shape[0]
    temp = D.copy()
    rng = np.random.default_rng(seed)

    route_pairs = set()
    for a, b in _route_directed_edges(tour):
        if a == b:
            continue
        temp[a, b] = D[a, b]
        temp[b, a] = D[a, b]  # mirror the traversed (directed) route cost
        route_pairs.add(frozenset((a, b)))

    for i in range(n):
        for j in range(i + 1, n):
            if frozenset((i, j)) in route_pairs:
                continue
            if rng.random() < 0.5:
                temp[j, i] = temp[i, j]   # above-diagonal value wins
            else:
                temp[i, j] = temp[j, i]   # below-diagonal value wins

    np.fill_diagonal(temp, 0.0)
    coords = _classical_mds(temp, n_components=2)
    return coords, temp


def _classical_mds(distance_matrix, n_components=2):
    """Classical (Torgerson) MDS: embed a symmetric distance matrix in
    ``n_components`` Euclidean dimensions."""
    D = np.asarray(distance_matrix, dtype=float)
    n = D.shape[0]
    D2 = D ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * (J @ D2 @ J)
    B = (B + B.T) / 2.0  # enforce symmetry against round-off
    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(eigvals)[::-1][:n_components]
    top_vals = np.clip(eigvals[order], 0.0, None)
    coords = eigvecs[:, order] * np.sqrt(top_vals)
    return coords


def plot_route_on_reconstructed_coords(
    *, duration_matrix, tour, out_dir, filename, seed=None, title=None
):
    """Reconstruct coordinates from ``tour`` and save a route plot. Returns the
    written file path."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    coords, _temp = reconstruct_coordinates_from_route(duration_matrix, tour, seed=seed)
    os.makedirs(out_dir, exist_ok=True)
    # Tag with the run id so the end-of-run cleanup groups it with this run.
    from .Library_output_cleanup import stamp_run_id
    out_path = stamp_run_id(os.path.join(out_dir, filename))

    closed = list(tour) + [tour[0]]
    xs = coords[closed, 0]
    ys = coords[closed, 1]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(xs, ys, "k-", linewidth=0.8, zorder=1)
    ax.plot(coords[:, 0], coords[:, 1], "o", color="tab:orange", markersize=12, zorder=2)
    for i in range(coords.shape[0]):
        ax.text(coords[i, 0], coords[i, 1], str(i), ha="center", va="center", fontsize=8, zorder=3)
    if title:
        ax.set_title(title, fontsize=10)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path
