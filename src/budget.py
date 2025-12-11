from math import floor
import pandas as pd
from tqdm import tqdm
from typing import Sequence
from matplotlib import pyplot as plt
import seaborn as sns

from fastr_new import fast_pp, fast_pp_emb
from metric_new import compute_metrics
from utils import D4J, SIR, Coverage_type, ReductionAlgorithm, load_test_suite


def run_budget_experiment_duplicates(
    reduction_algo: ReductionAlgorithm,
    program: str,
    version: str,
    budget: float = 1.0,
    runs: int = 10,
) -> pd.DataFrame:
    test_suite = load_test_suite(program, version)

    N = floor(len(test_suite.test_cases) * budget)

    results = []

    for run_id in tqdm(range(runs)):
        reduction_result_pp = reduction_algo(
            test_suite, dimensions=0, budget=N, verbose=False
        )
        row = compute_metrics(test_suite, reduction_result_pp)

        row["run_id"] = run_id
        results.append(row)

    return pd.DataFrame(results)


def full_budget_experiment(
    algo: ReductionAlgorithm,
    program: str,
    version: str,
    budgets: Sequence[float],
    runs_per_budget: int = 1,
) -> pd.DataFrame:
    results = pd.DataFrame()

    for budget in tqdm(budgets):
        result = run_budget_experiment_duplicates(
            algo, program, version, budget, runs_per_budget
        )
        result["budget"] = budget

        results = pd.concat([results, result])

    return results


def main():
    budgets = [percentile / 100 for percentile in range(1, 20, 1)]
    print(budgets)
    result_pp = full_budget_experiment(fast_pp, "grep", "v3", budgets, 10)
    result_pp['algo'] = "fast_pp"

    result_pp_emb = full_budget_experiment(fast_pp_emb, "grep", "v3", budgets, 10)
    result_pp_emb['algo'] = "fast_pp_emb"

    result = pd.concat([result_pp, result_pp_emb])

    result.to_csv('data/results.csv')

    # fig, ax = plt.subplots(2, 2)
    #
    # sns.scatterplot(data=result, x="budget", y='fft', ax=ax[0][0], hue='algo')
    # sns.scatterplot(data=result, x="budget", y='tsr', ax=ax[0][1], hue='algo')
    # sns.scatterplot(data=result, x="budget", y='fdl', ax=ax[1][0], hue='algo')
    # sns.scatterplot(data=result, x="budget", y='apfd', ax=ax[1][1], hue='algo')
    #
    # ax[0][0].set_title('fft')
    # ax[0][1].set_title('tsr')
    # ax[1][0].set_title('fdl')
    # ax[1][1].set_title('apfd')
    #
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()
