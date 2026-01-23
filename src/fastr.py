import random
from collections import defaultdict
from pathlib import Path
from time import perf_counter_ns
from typing import Callable, List

# import faiss
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.random_projection import (
    SparseRandomProjection,
    johnson_lindenstrauss_min_dim,
)

from embedding.codeT5 import CodeT5PlusEmbedder
from hashing import min_hashing
from metric import compute_metrics
from utils import ReductionResult, TestSuite, load_test_suite, set_all_random_seeds

embedder = CodeT5PlusEmbedder()


# --------------------  UTILS  --------------------


def generate_signature(
    test_suite: TestSuite, dims: int = 10, k: int = 5
) -> List[List[str]]:
    shingles = test_suite.coverage_ids or []

    # Generate k-shingles when no coverage_ids are given
    if shingles == []:
        shingle = set()
        for test_case in test_suite.test_cases:
            for i in range(len(test_case) - k + 1):
                shingle.add(hash(test_case[i : i + k]))
        shingles.append(shingle)

    signatures = [min_hashing(shingle, dims) for shingle in shingles]

    return signatures


def store_signatures(signatures: List[List[str]], outfile: Path) -> None:
    with open(outfile, "w") as f:
        for signature in signatures:
            for _hash in signature:
                f.write(_hash)
                f.write(" ")
            f.write("\n")


def load_signatures(input_file: Path) -> List[List[str]]:
    signatures = []

    with open(input_file, "r") as fin:
        for tc in fin:
            signatures.append(tc.strip().split())

    return signatures


def compute_distance(
    v: torch.Tensor, w: torch.Tensor, metric="euclidean"
) -> np.ndarray:
    # Euclidean distance
    if metric in ("euclidean", "cosine"):
        return cdist(np.atleast_2d(v), np.atleast_2d(w), metric=metric)

    raise ValueError(f"Metric has to be one of [euclidean, cosine] and got {metric}")


def ns_to_sec(ns: float) -> float:
    return ns / 10**9


# --------------- PREPERATION PHASE ---------------


def preparation_FAST_R(
    test_suite: TestSuite, dimensions: int = 0, seed: int = 0
) -> torch.Tensor:
    vectorizer = HashingVectorizer()  # compute "TF"
    test_cases = vectorizer.fit_transform(test_suite.test_cases)

    # dimensionality reduction
    if dimensions <= 0:
        e = 0.5  # epsilon in jl lemma
        dimensions = int(
            johnson_lindenstrauss_min_dim(len(test_suite.test_cases), eps=e)
        )
    srp = SparseRandomProjection(n_components=dimensions)  # type: ignore[arg-type]
    projectedTestSuite = srp.fit_transform(test_cases)

    return torch.tensor(projectedTestSuite.toarray())


def preparation_codet5p(
    test_suite: TestSuite, dimensions: int = 0, seed: int = 0
) -> torch.Tensor:
    return embedder.embed_test_cases(tuple(test_suite.test_cases), seed=seed)


# ---------------  REDUCTION PHASE  ---------------


# FAST++ reduction phase
def reduction_plus_plus(
    test_suite: torch.Tensor, budget: int, dist_metric: str = "euclidean"
) -> List[int]:
    n = test_suite.shape[0]
    reduced_test_suite = []

    # Dict to keep track of closest selection
    D = defaultdict(lambda: float("Inf"))

    # first center
    selected = random.randrange(n)
    reduced_test_suite.append(selected + 1)
    D[selected] = 0.0

    while len(reduced_test_suite) < min(budget, n):
        # Update nearest-selected distances
        norm = 0
        for i in range(len(test_suite)):
            if D[i] != 0:
                di = compute_distance(
                    test_suite[i, :], test_suite[selected, :], metric=dist_metric
                )[0][0]
                if di < D[i]:
                    D[i] = di
            norm += D[i] * D[i]

        # safe exit point (if all distances are 0)
        # (but not all test cases have been selected)
        if norm == 0:
            remaining = list(set(range(1, n + 1)) - set(reduced_test_suite))
            random.shuffle(remaining)
            reduced_test_suite.extend(remaining[: budget - len(reduced_test_suite)])
            break

        c = 0
        coin = random.random() * norm
        for i, di in D.items():
            w = di * di
            if coin < c + w:
                reduced_test_suite.append(i + 1)
                D[i] = 0
                selected = i
                break
            c += w

    return reduced_test_suite


# FAST-CS Reduction phase
def reduction_cs(
    test_suite: torch.Tensor, budget: int, dist_metric: str = "cosine"
) -> List[int]:
    n = test_suite.shape[0]
    k = min(budget, n)

    X = test_suite

    mu = X.mean(dim=0)

    if dist_metric == "cosine":
        X = F.normalize(X, p=2, dim=1)
        mu = F.normalize(X.mean(dim=0), p=2, dim=0)
    elif dist_metric == "euclidean":
        mu = X.mean(dim=0)
    else:
        raise ValueError(f"Undefined metric for CS sampling: {dist_metric}")

    # compute distances to com
    D = compute_distance(test_suite, mu, metric=dist_metric)
    D2 = D * D

    # compute probabilities of being sampled
    uniform = 1 / (2 * n)
    D2_sum = float(D2.sum())

    if D2_sum == 0.0:
        P = torch.full((n,), 1.0 / n, dtype=torch.float64)
    else:
        proportional = D2 / D2_sum

        # Normalize to sum to 1
        P = uniform + proportional
        P = (P / P.sum()).flatten()

    # proportional sampling
    reduced = np.random.choice(np.arange(n), size=k, p=P, replace=False) + 1

    return reduced.tolist()


# def reduction_faiss(
#     test_suite: torch.Tensor, budget: int = 0, dist_metric: str = "euclidean"
# ) -> List[int]:
#     arr = test_suite.float()
#     faiss.normalize_L2(arr)
#
#     n, dim = arr.shape
#     k = n if budget == 0 else budget
#
#     selected_indices = [random.randrange(n)]
#     selected_set = set(selected_indices)
#     remaining_set = set(range(n)) - selected_set
#     remaining_indices = list(remaining_set)
#
#     while len(selected_indices) < k:
#         # Compute mean of selected
#         mean_vec = np.mean(arr[selected_indices], axis=0, keepdims=True)
#         faiss.normalize_L2(mean_vec)
#         neg_mean = -mean_vec.astype("float32")
#
#         # Create a new index with remaining only
#         remaining_arr = arr[remaining_indices]
#         index = faiss.IndexFlatIP(dim)
#         index.add(remaining_arr)
#
#         # Search top diverse candidates
#         num_to_select = min(1, k - len(selected_indices))
#         _, I = index.search(neg_mean, num_to_select)
#         print(I)
#         exit()
#
#         newly_selected = []
#         for i in I[0]:
#             idx = remaining_indices[i]  # Map back to global index
#             if idx not in selected_set:
#                 selected_indices.append(idx)
#                 selected_set.add(idx)
#                 newly_selected.append(idx)
#
#         remaining_set -= set(newly_selected)
#         remaining_indices = list(remaining_set)
#
#     return selected_indices


# -----------  Full Algorithm Template  -----------


def preperation_reduction_algorithm(
    prep_phase,
    reduction_phase,
    algo_name: str,
    dist_metric: str = "euclidean",
) -> Callable[[TestSuite, int, int, int, bool], ReductionResult]:
    def algo(
        raw_test_suite: TestSuite,
        dimensions: int = 0,
        budget: int = 0,
        random_seed: int = 0,
        verbose: bool = False,
        # cache: bool = False,
    ) -> ReductionResult:
        set_all_random_seeds(random_seed)

        prep_start = perf_counter_ns()
        test_suite = prep_phase(raw_test_suite, dimensions=dimensions, seed=random_seed)
        prep_end = perf_counter_ns()
        prep_time_ns = prep_end - prep_start

        if verbose:
            print(
                f"Preparation phase of {algo_name} algorithm took {prep_time_ns} ns ({ns_to_sec(prep_time_ns):.2f} sec)"
            )

        if budget <= 0:
            budget = len(test_suite)

        reduction_start = perf_counter_ns()
        reduced_test_suite = reduction_phase(
            test_suite, budget, dist_metric=dist_metric
        )
        reduction_end = perf_counter_ns()
        reduction_time_ns = reduction_end - reduction_start

        if verbose:
            print(
                f"Reduction phase of {algo_name} algorithm took {reduction_time_ns} ns ({ns_to_sec(reduction_time_ns):.2f} sec)"
            )

        return ReductionResult(prep_time_ns, reduction_time_ns, reduced_test_suite)

    return algo


# ----------  Algorithm implementations  ----------


# Full FAST++ implementation
def fast_pp(
    raw_test_suite: TestSuite,
    dimensions: int = 0,
    budget: int = 0,
    random_seed: int = 0,
    verbose: bool = False,
    # cache: bool = False,
) -> ReductionResult:
    algo = preperation_reduction_algorithm(
        prep_phase=preparation_FAST_R,
        reduction_phase=reduction_plus_plus,
        algo_name="FAST++",
    )

    return algo(raw_test_suite, dimensions, budget, random_seed, verbose)


# Full FAST++ implementation
def fast_pp_emb(
    raw_test_suite: TestSuite,
    dimensions: int = 0,
    budget: int = 0,
    random_seed: int = 0,
    verbose: bool = False,
    # cache: bool = False,
) -> ReductionResult:
    algo = preperation_reduction_algorithm(
        prep_phase=preparation_codet5p,
        reduction_phase=reduction_plus_plus,
        algo_name="embedding FAST++",
    )

    return algo(raw_test_suite, dimensions, budget, random_seed, verbose)


# Full FAST-CS implementation
def fast_cs(
    raw_test_suite: TestSuite,
    dimensions: int = 0,
    budget: int = 0,
    random_seed: int = 0,
    verbose: bool = False,
    # cache: bool = False,
) -> ReductionResult:
    algo = preperation_reduction_algorithm(
        prep_phase=preparation_FAST_R, reduction_phase=reduction_cs, algo_name="FAST-CS"
    )

    return algo(raw_test_suite, dimensions, budget, random_seed, verbose)


# Full FAST-CS implementation
def fast_cs_emb(
    raw_test_suite: TestSuite,
    dimensions: int = 0,
    budget: int = 0,
    random_seed: int = 0,
    verbose: bool = False,
    # cache: bool = False,
) -> ReductionResult:
    algo = preperation_reduction_algorithm(
        prep_phase=preparation_codet5p,
        reduction_phase=reduction_cs,
        algo_name="embedding FAST-CS",
    )

    return algo(raw_test_suite, dimensions, budget, random_seed, verbose)


def random_baseline(
    raw_test_suite: TestSuite,
    dimensions: int = 0,
    budget: int = 0,
    random_seed: int = 0,
    verbose: bool = False,
    # cache: bool = False,
) -> ReductionResult:
    algo = preperation_reduction_algorithm(
        prep_phase=lambda test_suite, dimensions, seed: len(test_suite.test_cases),
        reduction_phase=lambda test_suite, budget, dist_metric: np.random.choice(
            np.arange(test_suite), size=budget, replace=False
        )
        + 1,
        algo_name="random_baseline",
    )

    return algo(raw_test_suite, dimensions, budget, random_seed, verbose)


# # FAISS implementation
# def faiss_emb(
#     raw_test_suite: TestSuite,
#     dimensions: int = 0,
#     budget: int = 0,
#     random_seed: int = 0,
#     verbose: bool = False,
#     # cache: bool = False,
# ) -> ReductionResult:
#     algo = preperation_reduction_algorithm(
#         prep_phase=preparation_codet5p,
#         reduction_phase=reduction_faiss,
#         algo_name="embedding FAISS",
#     )
#
#     return algo(raw_test_suite, dimensions, budget, random_seed, verbose)


def main():
    test_suite = load_test_suite("grep", "v3", "line")
    budget = 30

    # Fast++
    reduction_result = fast_pp(test_suite, budget=budget)
    result = compute_metrics(test_suite, reduction_result)
    print(result)

    # Fast++ with embeddings
    reduction_result = fast_pp_emb(test_suite, budget=budget)
    result = compute_metrics(test_suite, reduction_result)
    print(result)
