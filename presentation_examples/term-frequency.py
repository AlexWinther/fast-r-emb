import re
import math
from collections import Counter
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np


TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def build_vocab(docs: List[str]) -> List[str]:
    return sorted(set(term for doc in docs for term in tokenize(doc)))


def term_frequency_vector(text: str, vocab: List[str]) -> List[int]:
    counts = Counter(tokenize(text))
    return [int(counts.get(term, 0)) for term in vocab]


def euclidean_distance(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(a: List[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def cosine_similarity(a: List[float], b: List[float]) -> float:
    na, nb = norm(a), norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot(a, b) / (na * nb)


def pairwise_metrics(
    vectors: List[List[float]],
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Returns:
      distances[i][j] = Euclidean distance between doc i and doc j
      cosines[i][j]   = Cosine similarity between doc i and doc j
    """
    n = len(vectors)
    distances = [[0.0] * n for _ in range(n)]
    cosines = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i, n):
            d = euclidean_distance(vectors[i], vectors[j])
            c = cosine_similarity(vectors[i], vectors[j])
            distances[i][j] = distances[j][i] = d
            cosines[i][j] = cosines[j][i] = c

    return distances, cosines


def print_matrix(
    matrix: List[List[float]], title: str, doc_labels: Optional[List[str]] = None
) -> None:
    n = len(matrix)
    labels = doc_labels if doc_labels else [f"D{i}" for i in range(n)]

    print(f"\n{title}")
    header = "     " + "  ".join(f"{lab:>8}" for lab in labels)
    print(header)
    for i, row in enumerate(matrix):
        line = f"{labels[i]:>4} " + "  ".join(f"{val:8.4f}" for val in row)
        print(line)


def plot_matrix_heatmap(matrix, labels, title, filename):
    mat = np.array(matrix)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, cmap="Greens", vmin=0, vmax=1)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    ax.set_title(title)
    plt.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_matrix_table(matrix, labels, title, filename):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis("off")

    table = ax.table(
        cellText=[[f"{v:.3f}" for v in row] for row in matrix],
        rowLabels=labels,
        colLabels=labels,
        loc="center",
    )

    table.scale(1, 1.5)
    ax.set_title(title, pad=20)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def main():
    doc1 = """def sum(a: int, b: int) -> int:
        return a + b"""

    doc2 = """def sum(first: float, second: float) -> float:
        return first + second"""

    doc3 = """def multiply(first: float, second: float) -> float:
        return first * second"""

    docs = [doc1, doc2, doc3]

    vocab = build_vocab(docs)
    vectors = [term_frequency_vector(doc, vocab) for doc in docs]

    distances, cosines = pairwise_metrics(vectors)

    print("Vocabulary:", vocab)

    # Optional: show one vector per doc (can be long for big vocab)
    for i, v in enumerate(vectors):
        print(f"\nTF vector doc {i}:")
        print(v)

    print_matrix(distances, "Pairwise Euclidean distances")
    print_matrix(cosines, "Pairwise cosine similarities")

    labels = ["sum int", "sum float", "multiply"]
    plot_matrix_heatmap(distances, labels, "Euclidean Distance", "euclidean.png")
    plot_matrix_heatmap(cosines, labels, "Cosine Similarity", "cosine.png")

    plot_matrix_table(distances, labels, "Euclidean Distance", "euclidean_table.png")
    plot_matrix_table(cosines, labels, "Cosine Similarity", "cosine_table.png")


if __name__ == "__main__":
    main()
