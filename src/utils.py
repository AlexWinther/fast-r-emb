import pickle
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Protocol, Sequence, Set

import numpy as np
import torch

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

SIR = [
    ("flex", "v3"),
    ("grep", "v3"),
    ("gzip", "v1"),
    ("sed", "v6"),
    ("make", "v1"),
]
D4J = [
    ("math", "v1"),
    ("closure", "v1"),
    ("time", "v1"),
    ("lang", "v1"),
    ("chart", "v1"),
]

Coverage_type = Literal["line", "function", "branch"] | None


@dataclass
class TestSuite:
    """Structured metadata for a test suite."""

    program: str
    version: str
    java_flag: bool
    test_cases: Sequence[str]

    # Optional coverage info (white box)
    coverage_type: Coverage_type
    coverage_ids: List[Set[int]] | None

    fault_matrix: Dict[int, List[int]]


@dataclass
class ReductionResult:
    """Structured metadata for a reduction result."""

    prep_time_ns: int
    red_time_ns: int
    test_case_selection: List[int]


class ReductionAlgorithm(Protocol):
    def __call__(
        self,
        raw_test_suite: TestSuite,
        dimensions: int = 0,
        budget: int = 0,
        random_seed: int = 42,
        verbose: bool = False,
        cache: bool = False,
    ) -> ReductionResult: ...


def _load_java_fault_matrix(path: Path) -> Dict[int, List[int]]:
    fault_matrix = defaultdict(list)

    with open(path) as f:
        fault_matrix.update({int(tc.strip()): [1] for tc in f.readlines()})

    return fault_matrix


def _load_C_fault_matrix(path: Path) -> Dict[int, List[int]]:
    fault_matrix = defaultdict(list)
    with open(path, "rb") as f:
        raw_fault_matrix = pickle.load(f)

    fault_matrix.update({int(key): value for key, value in raw_fault_matrix.items()})

    return fault_matrix


# utility function to get which faults are detected
def _load_fault_matrix(
    program: str, version: str, java_flag: bool
) -> Dict[int, List[int]]:
    """INPUT:
    (str)fault_matrix: path of fault_matrix (pickle file)

    OUTPUT:
    (dict)faults_dict: key=tcID, val=[detected faults]
    """
    base_path = DATA_DIR / f"{program}_{version}"
    fault_matrix_path = base_path / (
        "fault_matrix.txt" if java_flag else "fault_matrix_key_tc.pickle"
    )

    if java_flag:
        fault_matrix = _load_java_fault_matrix(fault_matrix_path)
    else:
        fault_matrix = _load_C_fault_matrix(fault_matrix_path)

    return fault_matrix


def _filter_out_license(test_case: str) -> str:
    if test_case.startswith("/*"):
        return "*/".join(test_case.split("*/")[1:])

    return test_case


# utility function to load test suite
def load_test_suite(
    program: str, version: str, coverage_type: Coverage_type = None
) -> TestSuite:
    java_flag = (program, version) in D4J
    base_path = DATA_DIR / f"{program}_{version}"
    bbox_file = base_path / f"{program}-bbox.txt"

    with open(bbox_file) as f:
        test_cases = list(map(lambda x: x.strip(), f.readlines()))

        # if java_flag:
        #     test_cases = list(map(_filter_out_license, test_cases))

    coverage_ids = None
    if coverage_type:
        wbox_file = base_path / f"{program}-{coverage_type}.txt"
        coverage_ids = list(
            map(lambda x: set(map(int, x.split())), open(wbox_file).readlines())
        )

    fault_matrix = _load_fault_matrix(program, version, java_flag)

    return TestSuite(
        program,
        version,
        java_flag,
        test_cases,
        coverage_type,
        coverage_ids,
        fault_matrix,
    )


def set_all_random_seeds(random_seed: int) -> None:
    random.seed(random_seed)
    np.random.seed(random_seed)
    np.random.default_rng(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)


def main():
    test_suite = load_test_suite("grep", "v3")

    print(test_suite.fault_matrix)


if __name__ == "__main__":
    main()
