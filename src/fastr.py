import random
from collections import defaultdict
from pathlib import Path
from time import perf_counter_ns
from typing import Callable, List

import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.random_projection import (
    SparseRandomProjection,
    johnson_lindenstrauss_min_dim,
)

from embedding.codeT5 import CodeT5PlusEmbedder
from hashing import min_hashing
from metric_new import compute_metrics
from utils import ReductionResult, TestSuite, load_test_suite

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


def euclidean_dist(v: torch.Tensor, w: torch.Tensor) -> np.ndarray:
    result = cdist(np.atleast_2d(v), np.atleast_2d(w))

    return result


def ns_to_sec(ns: float) -> float:
    return ns / 10**9


# --------------- PREPERATION PHASE ---------------


def preparation_FAST_R(test_suite: TestSuite, dimensions: int = 0) -> torch.Tensor:
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


def preparation_codet5p(test_suite: TestSuite, dimensions: int = 0) -> torch.Tensor:
    embedder = CodeT5PlusEmbedder(dataset=f"{test_suite.program}_{test_suite.version}")

    return embedder.embed_test_cases(list(test_suite.test_cases))


# ---------------  REDUCTION PHASE  ---------------


# FAST++ reduction phase
def reduction_plus_plus(test_suite: torch.Tensor, budget: int) -> List[int]:
    reduced_test_suite = []

    # distance to closest center
    D = defaultdict(lambda: float("Inf"))

    # select first center randomly
    selectedTC = random.randint(0, len(test_suite) - 1)
    reduced_test_suite.append(selectedTC + 1)
    D[selectedTC] = 0

    while len(reduced_test_suite) < budget:
        # k-means++ tc reductionCS
        norm = 0
        for tc in range(len(test_suite)):
            if D[tc] != 0:
                dist = euclidean_dist(test_suite[tc, :], test_suite[selectedTC, :])
                if dist < D[tc]:
                    D[tc] = dist[0][0]
            norm += D[tc]

        # safe exit point (if all distances are 0)
        # (but not all test cases have been selected)
        if norm == 0:
            extraTCS = list(
                set(range(1, len(test_suite) + 1)) - set(reduced_test_suite)
            )
            random.shuffle(extraTCS)
            reduced_test_suite.extend(extraTCS[: budget - len(reduced_test_suite)])
            break

        c = 0
        coinToss = random.random() * norm
        for tc, dist in D.items():
            if coinToss < c + dist:
                reduced_test_suite.append(tc + 1)
                D[tc] = 0
                selectedTC = tc
                break
            c += dist

    return reduced_test_suite


# FAST-CS Reduction phase
def reduction_cs(test_suite: torch.Tensor, budget: int) -> List[int]:
    # compute center of mass (com)
    mu = torch.mean(test_suite, dim=0)
    print(mu.shape)

    # compute distances to com
    D = euclidean_dist(test_suite, mu)
    D_squared = D**2

    # compute probabilities of being sampled
    uniform = 1 / (2 * len(test_suite))
    proportional_squared_distance = D_squared / np.sum(D_squared)

    # Normalize to sum to 1
    P = uniform + proportional_squared_distance
    P = (P / P.sum()).flatten()

    # proportional sampling
    reducedTS = list(
        np.random.choice(
            list(range(1, len(test_suite) + 1)),
            size=min(budget, len(test_suite)),
            p=P,
            replace=False,
        ).tolist()
    )

    return reducedTS


# -----------  Full Algorithm Template  -----------


def preperation_reduction_algorithm(
    prep_phase, reduction_phase, algo_name: str
) -> Callable[[TestSuite, int, int, bool], ReductionResult]:
    def algo(
        raw_test_suite: TestSuite,
        dimensions: int = 0,
        budget: int = 0,
        verbose: bool = False,
        # cache: bool = False,
    ) -> ReductionResult:
        prep_start = perf_counter_ns()
        test_suite = prep_phase(raw_test_suite, dimensions=dimensions)
        prep_end = perf_counter_ns()
        prep_time_ns = prep_end - prep_start

        if verbose:
            print(
                f"Preparation phase of {algo_name} algorithm took {prep_time_ns} ns ({ns_to_sec(prep_time_ns):.2f} sec)"
            )

        if budget <= 0:
            budget = len(test_suite)

        reduction_start = perf_counter_ns()
        reduced_test_suite = reduction_phase(test_suite, budget)
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
    verbose: bool = False,
    # cache: bool = False,
) -> ReductionResult:
    algo = preperation_reduction_algorithm(
        prep_phase=preparation_FAST_R,
        reduction_phase=reduction_plus_plus,
        algo_name="FAST++",
    )

    return algo(raw_test_suite, dimensions, budget, verbose)


# Full FAST++ implementation
def fast_pp_emb(
    raw_test_suite: TestSuite,
    dimensions: int = 0,
    budget: int = 0,
    verbose: bool = False,
    # cache: bool = False,
) -> ReductionResult:
    algo = preperation_reduction_algorithm(
        prep_phase=preparation_codet5p,
        reduction_phase=reduction_plus_plus,
        algo_name="embedding FAST++",
    )

    return algo(raw_test_suite, dimensions, budget, verbose)


# Full FAST-CS implementation
def fast_cs(
    raw_test_suite: TestSuite,
    dimensions: int = 0,
    budget: int = 0,
    verbose: bool = False,
    # cache: bool = False,
) -> ReductionResult:
    algo = preperation_reduction_algorithm(
        prep_phase=preparation_FAST_R, reduction_phase=reduction_cs, algo_name="FAST-CS"
    )

    return algo(raw_test_suite, dimensions, budget, verbose)


# Full FAST-CS implementation
def fast_cs_emb(
    raw_test_suite: TestSuite,
    dimensions: int = 0,
    budget: int = 0,
    verbose: bool = False,
    # cache: bool = False,
) -> ReductionResult:
    algo = preperation_reduction_algorithm(
        prep_phase=preparation_codet5p,
        reduction_phase=reduction_cs,
        algo_name="embedding FAST-CS",
    )

    return algo(raw_test_suite, dimensions, budget, verbose)


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


if __name__ == "__main__":
    main()
