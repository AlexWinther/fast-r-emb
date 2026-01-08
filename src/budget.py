from math import ceil
import pandas as pd
from typing import Sequence, Tuple
from datetime import datetime

from tqdm.auto import tqdm
from fastr import fast_cs, fast_cs_emb, fast_pp, fast_pp_emb
from metric import compute_metrics
from utils import (
    D4J,
    SIR,
    ReductionAlgorithm,
    TestSuite,
    load_test_suite,
)

import rich.traceback
from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich import print
import numpy as np

rich.traceback.install()


def num_test_cases_from_budget(test_suite: TestSuite, budget: float) -> int:
    """Compute the number of test cases to select during reduction.

    Args:
        test_suite (TestSuite): The test suite that will be reduced.
        budget (float): The percentage of the test suite that should be selected.

    Return:
        int: The number of tests to be selected.
    """
    N = max(1, ceil(len(test_suite.test_cases) * budget))

    return N


def run_single_budget_experiment(
    algo: ReductionAlgorithm,
    test_suite: TestSuite,
    budget: float,
    random_seed: int,
) -> pd.DataFrame:
    """
    Args:
        algo (ReductionAlgorithm): The reduction algorithm to use.
        test_suite (TestSuite): The test suite to run the experiment on.
        budget (float): Proportion of test_suite to reduce to.
        random_seed (int): Random seed to pass to the reduction algorithm for reproducibility

    Returns:
        pd.DataFrame: The results of the experiment run (1 row)
    """
    N = num_test_cases_from_budget(test_suite, budget)

    reduction_result = algo(
        test_suite, dimensions=0, budget=N, random_seed=random_seed, verbose=False
    )
    row = compute_metrics(test_suite, reduction_result)

    return pd.DataFrame(row)


def run_multiple_budget_experiments(
    algos: Sequence[ReductionAlgorithm],
    test_suites: Sequence[TestSuite],
    test_suite_names: Sequence[str],
    budgets: Sequence[float],
    iteration: int,
):
    results = []
    for test_suite_name, test_suite in zip(test_suite_names, test_suites):
        random_seed = hash(f"{test_suite_name}-{iteration}") % 2**32
        for algo in algos:
            for budget in budgets:
                result = run_single_budget_experiment(
                    algo=algo,
                    test_suite=test_suite,
                    budget=budget,
                    random_seed=random_seed,
                )
                result["budget"] = budget
                result["algo"] = algo.__class__
                results.append(result)

    return pd.concat(results)


def run_experiments_tqdm(
    suites: Sequence[Tuple[str, str]],
    algos: Sequence[ReductionAlgorithm],
    budgets: Sequence[float],
    runs: int = 1,
):
    results = []
    pbar = tqdm(
        total=len(suites) * runs * len(algos) * len(budgets), position=0, leave=True
    )

    for program, version in tqdm(suites, desc="Test suites", position=1, leave=False):
        test_suite = load_test_suite(program, version)
        Ns = [num_test_cases_from_budget(test_suite, budget) for budget in budgets]

        for run in tqdm(range(runs), desc="Runs", position=2, leave=False):
            random_seed = abs(hash(f"{program}_{version}-{run}")) % (2**32)

            for algo in tqdm(algos, desc="Algos", position=3, leave=False):
                for N in tqdm(Ns, desc="Budgets (N)", position=4, leave=False):
                    reduction_result = algo(
                        test_suite,
                        dimensions=0,
                        budget=N,
                        random_seed=random_seed,
                    )

                    test = 1 / 0

                    result = compute_metrics(test_suite, reduction_result)
                    result["test_cases_selected"] = N
                    result["algo"] = algo.__qualname__
                    result["test_suite"] = f"{program}_{version}"
                    result["random_seed"] = random_seed

                    results.append(result)
                    pbar.update(1)

    return pd.concat(results).reset_index(drop=True)


def run_experiments_rich(
    suites: Sequence[Tuple[str, str]],
    algos: Sequence[ReductionAlgorithm],
    budgets: Sequence[float],
    runs: int = 1,
):
    console = Console()
    results = []

    progress = Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,  # keep final bars; set True to clear on finish
    )

    with progress:
        overall_task = progress.add_task(
            "Overall experiment", total=len(suites) * runs * len(algos) * len(budgets)
        )
        test_suite_task = progress.add_task("Test Suite", total=len(suites))
        runs_task = progress.add_task("Runs (current test suite)", total=runs)
        algos_task = progress.add_task("Algos (current run)", total=len(algos))
        ns_task = progress.add_task("Budgets (current algo)", total=len(budgets))

        for program, version in suites:
            progress.reset(runs_task, total=runs, completed=0)
            progress.update(
                test_suite_task, description=f"Test suite (suite={program}_{version})"
            )

            test_suite = load_test_suite(program, version)
            Ns = [num_test_cases_from_budget(test_suite, budget) for budget in budgets]

            for run in range(runs):
                random_seed = abs(hash(f"{program}_{version}-{run}")) % (2**32)

                progress.reset(algos_task, total=len(algos), completed=0)
                progress.update(runs_task, description=f"Runs (run={run})")

                for algo in algos:
                    progress.reset(ns_task, total=len(Ns), completed=0)
                    progress.update(
                        algos_task,
                        advance=1,
                        description=f"Algos (run={run}, algo={algo.__qualname__})",
                    )

                    for N in Ns:
                        progress.update(
                            ns_task,
                            advance=1,
                            description=f"Budgets (N={N})",
                        )

                        reduction_result = algo(
                            test_suite,
                            dimensions=0,
                            budget=N,
                            random_seed=random_seed,
                        )

                        result = compute_metrics(test_suite, reduction_result)

                        result["test_cases_selected"] = N
                        result["algo"] = algo.__qualname__
                        result["test_suite"] = f"{program}_{version}"
                        result["random_seed"] = random_seed

                        results.append(result)
                        progress.update(overall_task, advance=1)

                progress.update(runs_task, advance=1)
            progress.update(test_suite_task, advance=1)
    return pd.concat(results).reset_index(drop=True)


def main():
    test_suites = SIR + D4J
    print(test_suites)
    algos = [fast_cs, fast_cs_emb, fast_pp, fast_pp_emb]
    budgets = np.concat(
        [
            np.arange(0.01, 0.26, 0.01),
            np.arange(0.3, 0.5, 0.05),
            np.arange(0.5, 1.1, 0.25),
        ]
    ).tolist()
    runs = 40

    results = run_experiments_rich(test_suites, algos, budgets, runs)

    print(results)
    results.to_csv(f"data/results-{datetime.today().strftime('%Y-%m-%d-%H%M%S')}.csv")


if __name__ == "__main__":
    main()
