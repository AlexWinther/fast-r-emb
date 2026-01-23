from dataclasses import dataclass
from datetime import datetime
from math import ceil
from typing import List, Sequence, Tuple
import os
import platform
import json
from pathlib import Path
import hashlib


from tqdm import tqdm
import numpy as np
import pandas as pd
import rich.traceback
from rich import print
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from fastr import fast_cs_emb, fast_pp_emb, fast_cs, fast_pp, random_baseline
from metric import compute_metrics
from utils import D4J, SIR, ReductionAlgorithm, TestSuite, load_test_suite

rich.traceback.install()


# -----------------------------
# Method metadata (separate factors for analysis)
# -----------------------------


@dataclass(frozen=True)
class MethodSpec:
    name: str  # unique label (e.g., "cs_tf_srp")
    algorithm_family: str  # "cs" or "pp"
    representation: str  # "tf_srp" or "emb"
    fn: ReductionAlgorithm


METHODS: List[MethodSpec] = [
    # MethodSpec(
    #     name="random_baseline",
    #     algorithm_family="random",
    #     representation="SuiteLength",
    #     fn=random_baseline,
    # ),
    MethodSpec(
        name="cs_tf_srp", algorithm_family="cs", representation="tf_srp", fn=fast_cs
    ),
    MethodSpec(
        name="cs_emb", algorithm_family="cs", representation="emb", fn=fast_cs_emb
    ),
    MethodSpec(
        name="pp_tf_srp", algorithm_family="pp", representation="tf_srp", fn=fast_pp
    ),
    MethodSpec(
        name="pp_emb", algorithm_family="pp", representation="emb", fn=fast_pp_emb
    ),
]


# -----------------------------
# Deterministic seeding (do NOT use Python's hash(); it varies per process unless PYTHONHASHSEED is fixed)
# -----------------------------


def stable_u32_seed(*parts: str, namespace: str = "tsr-exp-v1") -> int:
    """Create a deterministic 32-bit seed from string parts."""
    h = hashlib.blake2b(digest_size=8, person=namespace.encode("utf-8"))
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\0")
    return int.from_bytes(h.digest()[:4], "little", signed=False)


# -----------------------------
# Budgeting
# -----------------------------


def num_test_cases_from_budget(test_suite: TestSuite, budget_prop: float) -> int:
    """Compute exact number of tests to select at a proportional budget."""
    n_total = len(test_suite.test_cases)
    return max(1, ceil(n_total * float(budget_prop)))


def achieved_budget_ratio(test_suite: TestSuite, n_selected: int) -> float:
    return float(n_selected) / float(len(test_suite.test_cases))


# -----------------------------
# Suite metadata
# -----------------------------


def infer_language(program: str, version: str) -> str:
    """Optional helper: label language (Java/C) for analysis.

    Adjust mapping to your suite naming conventions.
    """
    if (program.lower(), version.lower()) in D4J:
        return "Java"
    return "C"  # default fallback


# -----------------------------
# Core runner (paired design)
# -----------------------------


def run_experiments(
    suites: Sequence[Tuple[str, str]],
    methods: Sequence[MethodSpec],
    budgets: Sequence[float],
    runs: int,
    base_seed: int = 20260108,
) -> pd.DataFrame:
    """Runs a fully paired experiment.

    For each (suite, run_id), a single seed is generated and used for all methods and budgets.
    """
    console = Console()
    results = []

    budgets = [float(b) for b in budgets]
    if any(b <= 0 for b in budgets):
        raise ValueError("All budgets must be > 0.")

    budgets = sorted(set(budgets))

    progress = Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )

    total = len(suites) * runs * len(methods) * len(budgets)
    with progress:
        overall_task = progress.add_task("Overall", total=total)

        for program, version in suites:
            suite_id = f"{program}_{version}"
            language = infer_language(program, version)

            test_suite = load_test_suite(program, version)
            n_total = len(test_suite.test_cases)

            for run_id in range(runs):
                seed = stable_u32_seed(str(base_seed), suite_id, str(run_id))

                for m in methods:
                    for budget_prop in budgets:
                        n_selected = num_test_cases_from_budget(test_suite, budget_prop)

                        reduction_result = m.fn(
                            test_suite,
                            dimensions=0,
                            budget=n_selected,
                            random_seed=seed,
                            verbose=False,
                        )

                        row = compute_metrics(test_suite, reduction_result)

                        row.update(
                            {
                                "timestamp_utc": datetime.now().isoformat(
                                    timespec="seconds"
                                ),
                                "test_suite": suite_id,
                                "program": program,
                                "version": version,
                                "language": language,
                                "n_total_tests": n_total,
                                "budget_prop_requested": float(budget_prop),
                                "n_selected": int(n_selected),
                                "budget_prop_achieved": achieved_budget_ratio(
                                    test_suite, n_selected
                                ),
                                "run_id": int(run_id),
                                "random_seed": int(seed),
                                "method": m.name,
                                "algorithm_family": m.algorithm_family,
                                "representation": m.representation,
                            }
                        )

                        results.append(row)
                        progress.update(overall_task, advance=1)

    return pd.DataFrame(results)


def run_experiments_tqdm(
    suites: Sequence[Tuple[str, str]],
    methods: Sequence[MethodSpec],
    budgets: Sequence[float],
    runs: int,
    base_seed: int = 20260108,
) -> pd.DataFrame:
    """Runs a fully paired experiment.

    For each (suite, run_id), a single seed is generated and used for all methods and budgets.
    """
    results = []

    budgets = [float(b) for b in budgets]
    if any(b <= 0 for b in budgets):
        raise ValueError("All budgets must be > 0.")

    budgets = sorted(set(budgets))

    total = len(suites) * runs * len(methods) * len(budgets)

    with tqdm(
        total=total, desc="Overall", unit="run", dynamic_ncols=True, smoothing=0.05
    ) as pbar:
        for program, version in suites:
            suite_id = f"{program}_{version}"
            language = infer_language(program, version)

            test_suite = load_test_suite(program, version)
            n_total = len(test_suite.test_cases)

            for run_id in range(runs):
                seed = stable_u32_seed(str(base_seed), suite_id, str(run_id))

                for m in methods:
                    for budget_prop in budgets:
                        n_selected = num_test_cases_from_budget(test_suite, budget_prop)

                        reduction_result = m.fn(
                            test_suite,
                            dimensions=0,
                            budget=n_selected,
                            random_seed=seed,
                            verbose=False,
                        )

                        row = compute_metrics(test_suite, reduction_result)

                        row.update(
                            {
                                "timestamp_utc": datetime.now().isoformat(
                                    timespec="seconds"
                                ),
                                "test_suite": suite_id,
                                "program": program,
                                "version": version,
                                "language": language,
                                "n_total_tests": n_total,
                                "budget_prop_requested": float(budget_prop),
                                "n_selected": int(n_selected),
                                "budget_prop_achieved": achieved_budget_ratio(
                                    test_suite, n_selected
                                ),
                                "run_id": int(run_id),
                                "random_seed": int(seed),
                                "method": m.name,
                                "algorithm_family": m.algorithm_family,
                                "representation": m.representation,
                            }
                        )

                        results.append(row)
                        pbar.set_description(f"Processing {m.name} ({run_id})")
                        pbar.update(1)

    return pd.DataFrame(results)


def build_default_budgets() -> List[float]:
    budgets = np.concatenate(
        [
            np.arange(0.1, 1.01, 0.1),
        ]
    )
    return [float(f"{b:.4f}") for b in budgets.tolist()]


def methods_to_metadata(methods: Sequence[MethodSpec]) -> list[dict]:
    return [
        {
            "name": m.name,
            "algorithm_family": m.algorithm_family,
            "representation": m.representation,
        }
        for m in methods
    ]


def write_run_metadata(out_dir: Path, **kwargs) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "created_utc": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        **kwargs,
    }

    (out_dir / "run-metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


def main() -> None:
    suites = D4J + SIR
    budgets = build_default_budgets()
    runs = 10

    out_dir = Path("data") / f"results-{datetime.today().strftime('%Y-%m-%d-%H%M%S')}"
    write_run_metadata(
        out_dir,
        suites=suites,
        budgets=budgets,
        runs=runs,
        methods=methods_to_metadata(METHODS),
        base_seed=20260108,
    )

    df = run_experiments_tqdm(
        suites=suites, methods=METHODS, budgets=budgets, runs=runs, base_seed=20260108
    )
    df.to_csv(out_dir / "results.csv", index=False)
    print(df.head())
    print(f"Wrote: {out_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
