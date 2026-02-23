from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from math import ceil
from collections import defaultdict
import torch
from scipy.spatial.distance import cdist


def make_cluster(rng, center, n=100, scale=0.6):
    """Generate a 2D Gaussian cluster."""
    center = np.asarray(center, dtype=float)
    return rng.normal(loc=center, scale=scale, size=(n, 2))


def scale_sizes(
    p: np.ndarray,
    min_size: float = 8,
    max_size: float = 200.0,
    gamma: float = 0.8,
    default_size: float = 20,
):
    """Map probabilities to scatter marker areas."""
    p = np.asarray(p, dtype=float)
    if np.allclose(p.max(), p.min()):
        return np.full_like(p, default_size)

    p_norm = (p - p.min()) / (p.max() - p.min())
    p_norm = np.clip(p_norm, 0.0, 1.0) ** gamma
    return min_size + (max_size - min_size) * p_norm


def sampling_probs(points: np.ndarray) -> np.ndarray:
    """Compute the sampling probability P for each point (matches coreset sampling logic)."""
    num_points = points.shape[0]
    mu = np.mean(points, axis=0)

    D = pairwise_distances(points, mu.reshape(1, -1))
    D2 = D * D

    uniform = 1.0 / (2.0 * num_points)
    D2_sum = float(D2.sum())

    if D2_sum == 0.0:
        P = np.full((num_points,), 1.0 / num_points, dtype=np.float64)
    else:
        proportional = (D2 / D2_sum).flatten()
        P = uniform + proportional
        P = (P / P.sum()).astype(np.float64)

    return P


def generate_coreset_idx(points: np.ndarray, n: int) -> List[int]:
    """Proportional sampling without replacement. Returns 0-based indices."""
    rng = np.random.default_rng(43)

    num_points = points.shape[0]
    k = min(max(1, n), num_points)
    P = sampling_probs(points)
    return rng.choice(np.arange(num_points), size=k, p=P, replace=False).tolist()


def compute_distance(
    v: torch.Tensor, w: torch.Tensor, metric="euclidean"
) -> np.ndarray:
    # Euclidean distance
    if metric in ("euclidean", "cosine"):
        return cdist(np.atleast_2d(v), np.atleast_2d(w), metric=metric)

    raise ValueError(f"Metric has to be one of [euclidean, cosine] and got {metric}")


def fast_pp_dists(data, dists, selected):
    dists[selected] = 0.0

    # Update nearest-selected distances
    norm = 0
    for i in range(len(data)):
        if dists[i] != 0:
            di = compute_distance(data[i, :], data[selected, :], metric="euclidean")[0][
                0
            ]
            if di < dists[i]:
                dists[i] = di
        norm += dists[i] * dists[i]

    P = np.array([])
    for i in range(len(data)):
        d2 = dists[i] * dists[i]
        P = np.append(P, d2 / norm)

    return dists, P


def plot_mean(ax, mu: np.ndarray):
    ax.scatter([mu[0]], [mu[1]], s=150, c="tab:red", marker="o", label="Mean")


def main():
    rng = np.random.default_rng(42)

    clusters: List[Tuple[float, float, float]] = [
        (0.0, 0.0, 0.5),
        (-2.0, -1.0, 0.5),
        (-2.5, 2.0, 0.5),
        (2.0, 0.5, 0.5),
    ]

    DETERMINANT = False
    targets = np.array(
        [
            [0.8, 0],
            [-0.8, 0],
            [0, 0.8],
            [0, -1],
        ]
    )

    points_per_cluster = 150

    final_clusters = [
        make_cluster(rng, center=(cx, cy), n=points_per_cluster, scale=scale)
        for cx, cy, scale in clusters
    ]
    data = np.vstack(final_clusters)

    # ---- Plot 1: before selection ----
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        data[:, 0],
        data[:, 1],
        s=20,
        alpha=1,
        linewidths=0.0,
        label="Point",
        color="black",
    )

    ax.set_axis_off()

    plt.tight_layout()
    plt.savefig("pp-cluster-scatter-before.png", dpi=200)
    plt.show()
    plt.close(fig)

    # ---- Plot 2: iterative selection ----
    selection = np.array([], dtype=int)
    dists = defaultdict(lambda: float("Inf"))
    P = np.full(shape=(len(data),), fill_value=1 / len(data))
    for i in range(4):
        if DETERMINANT:
            # compute squared distances
            diff = data[:, [0, 1]] - targets[i]
            dist_sq = np.sqrt(np.sum(diff**2, axis=1))

            # index of closest point
            idx = np.argmin(dist_sq)
        else:
            idx = np.random.choice(range(len(data)), size=1, p=P)[0]

        print(idx)
        selection = np.append(selection, idx)
        print(selection)

        sizes = scale_sizes(P)

        mask = np.ones(len(data), dtype=bool)
        mask[selection] = False
        rest = data[mask]
        sizes = sizes[mask]

        reduced = data[selection]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(
            rest[:, 0],
            rest[:, 1],
            s=sizes,
            alpha=1,
            linewidths=0.0,
            label="Point",
            color="black",
        )

        ax.scatter(
            reduced[:, 0],
            reduced[:, 1],
            s=150,
            alpha=1,
            linewidths=0.8,
            label="Point",
            color="lime",
            edgecolor="black",
        )

        ax.set_axis_off()

        plt.tight_layout()
        plt.savefig(f"pp-cluster-scatter-step-{i}-before.png", dpi=200)
        plt.show()
        plt.close(fig)

        dists, P = fast_pp_dists(data, dists, selection[-1])

        sizes = scale_sizes(P)

        mask = np.ones(len(data), dtype=bool)
        mask[selection] = False
        rest = data[mask]
        sizes = sizes[mask]

        reduced = data[selection]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(
            rest[:, 0],
            rest[:, 1],
            s=sizes,
            alpha=1,
            linewidths=0.0,
            label="Point",
            color="black",
        )

        ax.scatter(
            reduced[:, 0],
            reduced[:, 1],
            s=150,
            alpha=1,
            linewidths=0.8,
            label="Point",
            edgecolor="black",
            color="lime",
        )

        ax.set_axis_off()

        plt.tight_layout()
        plt.savefig(f"pp-cluster-scatter-step-{i}-after.png", dpi=200)
        plt.show()
        plt.close(fig)

        print("next step")
        print(f"Selection = {selection}")

    # ---- Plot 3: coreset selection ----
    P = sampling_probs(data)
    sizes = scale_sizes(P)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        data[:, 0],
        data[:, 1],
        s=sizes,
        alpha=1,
        linewidths=0.0,
        label="Point",
        color="black",
    )
    plot_mean(ax, mu=np.mean(data, axis=0))

    ax.set_axis_off()

    plt.tight_layout()
    plt.savefig("cs-cluster-scatter-before.png", dpi=200)
    plt.show()
    plt.close(fig)

    n = ceil(0.1 * len(data))
    selection = generate_coreset_idx(data, n)

    mask = np.ones(len(data), dtype=bool)
    mask[selection] = False
    rest = data[mask]

    reduced = data[selection]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        rest[:, 0],
        rest[:, 1],
        s=sizes[mask],
        alpha=1,
        linewidths=0.0,
        label="Point",
        color="black",
    )
    ax.scatter(
        reduced[:, 0],
        reduced[:, 1],
        s=sizes[selection],
        alpha=1,
        linewidths=0.8,
        label="Point",
        color="lime",
        edgecolor="black",
    )
    plot_mean(ax, mu=np.mean(data, axis=0))

    ax.set_axis_off()

    plt.tight_layout()
    plt.savefig("cs-cluster-scatter-after.png", dpi=200)
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
