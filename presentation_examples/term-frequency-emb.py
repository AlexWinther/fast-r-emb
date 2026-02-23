import re
import math
from collections import Counter
from typing import List, Tuple, Optional, cast
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

try:
    from embedding.codeT5 import CodeT5PlusEmbedder
except Exception as e:
    raise ImportError(
        "CodeT5PlusEmbedder could not be imported. Ensure the module at src/embedding/codeT5.py is available."
    ) from e


TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def tokenize(text: str) -> List[str]:
    """Tokenize input text into a list of lowercase tokens.

    Uses the compiled TOKEN_RE to extract tokens from the given text and
    normalizes them to lowercase.
    """
    return [t.lower() for t in TOKEN_RE.findall(text)]


def build_vocab(docs: List[str]) -> List[str]:
    """Build a sorted vocabulary from a list of documents.

    The vocabulary is composed of unique tokens produced by tokenizing
    each document, with terms sorted in lexical order.
    """
    return sorted(set(term for doc in docs for term in tokenize(doc)))


def term_frequency_vector(text: str, vocab: List[str]) -> List[float]:
    """Compute a term-frequency vector for 'text' with respect to 'vocab'.

    Each element corresponds to the count of the term (from vocab) in the
    tokenized text.
    """
    counts = Counter(tokenize(text))
    return [float(counts.get(term, 0)) for term in vocab]


def euclidean_distance(a: List[float], b: List[float]) -> float:
    """Compute Euclidean distance between two vectors.

    Supports Python lists or torch tensors. If tensors are provided, uses
    PyTorch operations and returns a Python float.
    """
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def dot(a: List[float], b: List[float]) -> float:
    """Compute the dot product of two vectors."""
    return sum(x * y for x, y in zip(a, b))


def norm(a: List[float]) -> float:
    """Compute the L2 norm (Euclidean length) of vector a."""
    return math.sqrt(sum(x * x for x in a))


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors.

    Returns 0.0 if either vector has zero length to avoid division by zero.
    """
    na, nb = norm(a), norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot(a, b) / (na * nb)


def pairwise_metrics(
    vectors: torch.Tensor,
) -> Tuple[List[List[float]], List[List[float]]]:
    """Compute pairwise distances and cosine similarities for vectors.

    Returns two matrices:
      - distances[i][j] is the Euclidean distance between vector i and j
      - cosines[i][j]   is the cosine similarity between vector i and j
    """
    n = len(vectors)
    distances = [[0.0] * n for _ in range(n)]
    cosines = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i, n):
            vi = vectors[i].tolist()
            vj = vectors[j].tolist()
            d = euclidean_distance(vi, vj)
            c = cosine_similarity(vi, vj)
            distances[i][j] = distances[j][i] = d
            cosines[i][j] = cosines[j][i] = c

    return distances, cosines


def print_matrix(
    matrix: List[List[float]], title: str, doc_labels: Optional[List[str]] = None
) -> None:
    """Pretty-print a matrix with optional row/col labels."""
    n = len(matrix)
    labels = doc_labels if doc_labels else [f"D{i}" for i in range(n)]

    print(f"\n{title}")
    header = "     " + "  ".join(f"{lab:>8}" for lab in labels)
    print(header)
    for i, row in enumerate(matrix):
        line = f"{labels[i]:>4} " + "  ".join(f"{val:8.4f}" for val in row)
        print(line)


def plot_matrix_heatmap(matrix, labels, title, filename):
    """Plot and save a heatmap of matrix with given labels."""
    mat = np.array(matrix)

    _, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, cmap="Greens", vmin=0, vmax=1)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_matrix_table(matrix, labels, title, filename):
    """Plot and save a table representation of the matrix."""
    _, ax = plt.subplots(figsize=(7, 4))
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


def embed_string_with_codet5(
    text: str,
    model_dir: str = "Salesforce/codet5p-110m-embedding",
    batch_size: int = 4,
    device: Optional[str] = None,
    seed: int = 0,
    caching: bool = False,
):
    """Embed a string using the CodeT5+ model.

    This helper wraps the CodeT5PlusEmbedder from src/embedding/codeT5.py
    and returns the embeddings for the input text. The return value is a
    2D tensor of shape (1, embedding_dim).
    """

    embedder = CodeT5PlusEmbedder(
        model_dir=model_dir, batch_size=batch_size, device=device
    )
    return embedder.embed_test_cases([text], seed=seed, caching=caching)


def main():
    doc1 = """def sum(a: int, b: int) -> int:
        return a + b"""

    doc2 = """def sum(first: float, second: float) -> float:
        return first + second"""

    doc3 = """def multiply(first: float, second: float) -> float:
        return first * second"""

    docs = [doc1, doc2, doc3]

    vectors = torch.cat([embed_string_with_codet5(doc) for doc in docs], 0)
    vectors = F.normalize(vectors, dim=1)

    print(vectors.shape)

    distances, cosines = pairwise_metrics(vectors)

    # Optional: show one vector per doc (can be long for big vocab)
    # for i, v in enumerate(vectors):
    #     print(f"\nTF vector doc {i}:")
    #     print(v)

    print_matrix(distances, "Pairwise Euclidean distances")
    print_matrix(cosines, "Pairwise cosine similarities")

    labels = ["sum int", "sum float", "multiply"]
    plot_matrix_heatmap(distances, labels, "Euclidean Distance", "euclidean-emb.png")
    plot_matrix_heatmap(cosines, labels, "Cosine Similarity", "cosine-emb.png")

    plot_matrix_table(
        distances, labels, "Euclidean Distance", "euclidean_table-emb.png"
    )
    plot_matrix_table(cosines, labels, "Cosine Similarity", "cosine_table-emb.png")


if __name__ == "__main__":
    main()
