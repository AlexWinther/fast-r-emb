from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from math import ceil


def make_cluster(rng, center, n=100, scale=0.6):
    """Generate a 2D Gaussian cluster."""
    center = np.asarray(center, dtype=float)
    return rng.normal(loc=center, scale=scale, size=(n, 2))


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


def generate_coreset_idx(
    points: np.ndarray, n: int, rng: np.random.Generator
) -> List[int]:
    """Proportional sampling without replacement. Returns 0-based indices."""
    num_points = points.shape[0]
    k = min(max(1, n), num_points)
    P = sampling_probs(points)
    return rng.choice(np.arange(num_points), size=k, p=P, replace=False).tolist()


def scale_sizes(
    p: np.ndarray, min_size: float = 8.0, max_size: float = 250.0, gamma: float = 0.45
):
    """Map probabilities to scatter marker areas."""
    p = np.asarray(p, dtype=float)
    if np.allclose(p.max(), p.min()):
        return np.full_like(p, (min_size + max_size) / 2.0)

    p_norm = (p - p.min()) / (p.max() - p.min())
    p_norm = np.clip(p_norm, 0.0, 1.0) ** gamma
    return min_size + (max_size - min_size) * p_norm


def plot_mean(ax, mu: np.ndarray):
    ax.scatter([mu[0]], [mu[1]], s=80, c="tab:red", marker="o", label="Mean")


def main():
    rng = np.random.default_rng(42)

    clusters: List[Tuple[float, float, float]] = [
        (0.8, 0, 0.3),
        (0, 0.8, 0.3),
        (-0.8, 0, 0.3),
        (0, -1, 0.3),
    ]

    points_per_cluster = 100
    budget = 0.2
    coreset_size = ceil(points_per_cluster * len(clusters) * budget)

    final_clusters = [
        make_cluster(rng, center=(cx, cy), n=points_per_cluster, scale=scale)
        for cx, cy, scale in clusters
    ]
    data = np.vstack(final_clusters)
    mu = np.mean(data, axis=0)

    # Probabilities + sizes for ALL points
    P = sampling_probs(data)
    sizes = scale_sizes(P, min_size=8, max_size=150, gamma=0.45)

    # ---- Plot 1: before selection ----
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        data[:, 0],
        data[:, 1],
        s=sizes,
        alpha=0.65,
        linewidths=0.0,
        label="Point",
        color="black",
    )
    plot_mean(ax, mu)

    ax.set_axis_off()

    plt.tight_layout()
    plt.savefig("cluster-scatter-before.png", dpi=200)
    plt.close(fig)

    # Sample coreset (use same RNG for determinism)
    coreset_idx = generate_coreset_idx(data, coreset_size, rng=rng)
    coreset = data[coreset_idx]

    mask = np.ones(len(data), dtype=bool)
    mask[coreset_idx] = False
    rest = data[mask]

    # ---- Plot 2: after selection ----
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
        coreset[:, 0],
        coreset[:, 1],
        s=sizes[coreset_idx],
        alpha=1,
        edgecolors="black",
        color="lime",
        linewidths=0.8,
        label="Coreset",
    )
    plot_mean(ax, mu)

    ax.set_axis_off()

    plt.tight_layout()
    plt.savefig("cluster-scatter-after.png", dpi=200)

    plt.show()


if __name__ == "__main__":
    main()
