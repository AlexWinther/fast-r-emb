import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns

from scipy.stats import wilcoxon, friedmanchisquare, rankdata
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
import bambi as bmb
import arviz as az


DEFAULT_PATH = "data/results-2026-01-17-165719/results.csv"


def fit_glmm(data: pd.DataFrame):
    data = data.copy()

    model = bmb.Model(
        formula="p(faults_detected, total_faults) ~ algorithm_family * representation + budget + (1 | language) + (1 | representation)",
        data=data,
        family="binomial",
    )

    print(model)
    idata = model.fit(draws=2000, tune=2000, chains=4, cores=1, target_accept=0.9)

    az.plot_trace(
        idata,
        compact=False,
        backend_kwargs={"layout": "constrained"},
    )

    plt.show()
    exit()


def kendalls_w_from_pivot(pivot):
    # Rank within each block (row)
    ranked = pivot.apply(lambda row: rankdata(row), axis=1, result_type="expand")

    N, k = ranked.shape

    # Sum ranks per condition
    Rj = ranked.sum(axis=0)
    Rbar = N * (k + 1) / 2

    W = (12 * ((Rj - Rbar) ** 2).sum()) / (N**2 * (k**3 - k))
    return W


def plot_by_algorithm_family_and_language(
    save_path, df, agg_fn: str = "mean", include_random: bool = False
):
    df = df.copy()

    # if not include_random:
    #     df = df[df["algorithm_family"] != "random"]

    fig, ax = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=False)

    sns.lineplot(
        data=df[(df["algorithm_family"] != "pp") & (df["language"] == "C")],
        x="budget",
        y="fdl",
        hue="method",
        style="method",
        estimator=agg_fn,
        dashes={
            "Random reduction": (2, 2),
            "FAST-CS TF": (1, 0),
            "FAST-CS EMB": (1, 0),
        },
        errorbar=("ci", 95),  # bootstrap CI by default
        n_boot=1000,  # increase for smoother CI
        ax=ax[0, 0],
    )

    sns.lineplot(
        data=df[(df["algorithm_family"] != "cs") & (df["language"] == "C")],
        x="budget",
        y="fdl",
        hue="method",
        style="method",
        estimator=agg_fn,
        dashes={"Random reduction": (2, 2), "FAST++ TF": (1, 0), "FAST++ EMB": (1, 0)},
        errorbar=("ci", 95),  # bootstrap CI by default
        n_boot=1000,  # increase for smoother CI
        ax=ax[0, 1],
    )

    sns.lineplot(
        data=df[(df["algorithm_family"] != "pp") & (df["language"] == "Java")],
        x="budget",
        y="fdl",
        hue="method",
        style="method",
        estimator=agg_fn,
        dashes={
            "Random reduction": (2, 2),
            "FAST-CS TF": (1, 0),
            "FAST-CS EMB": (1, 0),
        },
        errorbar=("ci", 95),  # bootstrap CI by default
        n_boot=1000,  # increase for smoother CI
        ax=ax[1, 0],
    )

    sns.lineplot(
        data=df[(df["algorithm_family"] != "cs") & (df["language"] == "Java")],
        x="budget",
        y="fdl",
        hue="method",
        style="method",
        estimator=agg_fn,
        dashes={"Random reduction": (2, 2), "FAST++ TF": (1, 0), "FAST++ EMB": (1, 0)},
        errorbar=("ci", 95),  # bootstrap CI by default
        n_boot=1000,  # increase for smoother CI
        ax=ax[1, 1],
    )

    ax[0, 0].set_title("Coresets - C")
    ax[0, 1].set_title("K-means++ - C")
    ax[1, 0].set_title("Coresets - Java")
    ax[1, 1].set_title("K-means++ - Java")

    plt.legend()
    plt.tight_layout()

    plt.savefig(save_path, bbox_inches="tight")


def load_and_preprocess_data(csv_path):
    """Load and preprocess the results CSV."""
    df_raw = pd.read_csv(csv_path, index_col=None)

    cols_of_interest = [
        "total_faults",
        "faults_detected",
        "fdl",
        "test_suite",
        "language",
        "budget_prop_requested",
        "random_seed",
        "method",
        "algorithm_family",
        "representation",
    ]

    df = df_raw[cols_of_interest].copy()
    df = df.rename(columns={"budget_prop_requested": "budget"})

    # Rename methods for paper-ready graphs
    method_mapping = {
        "random_baseline": "Random reduction",
        "cs_tf_srp": "FAST-CS TF",
        "cs_emb": "FAST-CS EMB",
        "pp_tf_srp": "FAST++ TF",
        "pp_emb": "FAST++ EMB",
    }
    df["method"] = df["method"].map(method_mapping)

    return df


def filter_by_language(df, language):
    """Filter dataframe by language."""
    return df[df["language"] == language]


def create_pivot(df):
    """Create pivot table for Friedman test."""
    return df.pivot(
        columns="method", index=["test_suite", "budget", "random_seed"], values="fdl"
    )


def run_friedman_test(pivot):
    """Run Friedman test on the pivot data."""
    result = friedmanchisquare(
        pivot["FAST-CS TF"],
        pivot["FAST-CS EMB"],
        pivot["FAST++ TF"],
        pivot["FAST++ EMB"],
        pivot["Random reduction"],
    )
    n = len(pivot)
    k = 5
    kendall_w = result.statistic / (n * (k - 1))
    return result, kendall_w


def run_pairwise_wilcoxon(pivot, correction_method):
    """Run pairwise Wilcoxon tests with optional multiple test correction."""
    methods = [
        "FAST-CS TF",
        "FAST-CS EMB",
        "FAST++ TF",
        "FAST++ EMB",
        "Random reduction",
    ]
    results = []
    for method1 in methods:
        for method2 in methods:
            if method1 == method2:
                continue

            test_result = wilcoxon(
                pivot[method1], pivot[method2], alternative="less", method="asymptotic"
            )

            diff = pivot[method2] - pivot[method1]
            mean_diff = np.mean(diff)
            pct_diff = (diff / (pivot[method1] + 0.001)) * 100
            mean_pct_diff = np.mean(pct_diff)

            # Rank-biserial correlation approximation
            n = len(diff)
            r_biserial = 1 - (2 * test_result.statistic) / (n * (n + 1))
            r_biserial_ = test_result.zstatistic / np.sqrt(len(pivot))

            result = {
                "method1": method1,
                "method2": method2,
                "pval": test_result.pvalue,
                "mean_diff": mean_diff,
                "mean_pct_diff": mean_pct_diff,
                "rank_biserial": r_biserial,
                "rank_biserial_": r_biserial_,
            }

            results.append(result)

    results_df = pd.DataFrame(results)
    if correction_method == "none":
        results_df["reject"] = results_df["pval"] < 0.05
        results_df["p_adj"] = results_df["pval"]
    else:
        reject, p_adj, _, _ = multipletests(
            results_df["pval"], alpha=0.05, method=correction_method
        )
        results_df["reject"] = reject
        results_df["p_adj"] = p_adj

    return results_df


def plot_rejection_matrix(results, language, correction_method, save_path):
    """Plot the rejection matrix for Wilcoxon tests."""
    rejection_mat = results.pivot_table(
        values="reject", index="method1", columns="method2"
    ).fillna(0)

    plt.figure(figsize=(8, 6))

    # make a color map of fixed colors
    cmap = colors.ListedColormap(["lightcoral", "palegreen"])
    bounds = [0, 1]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    plt.imshow(rejection_mat.to_numpy(), cmap=cmap, norm=norm)

    # Binary legend
    plt.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                label="Do not reject H0",
                markerfacecolor="lightcoral",
                markersize=10,
            ),
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                label="Reject H0",
                markerfacecolor="palegreen",
                markersize=10,
            ),
        ],
        loc="lower right",
    )

    plt.xticks(
        ticks=np.arange(len(rejection_mat.columns)),
        labels=rejection_mat.columns,
        rotation=45,
    )
    plt.yticks(
        ticks=np.arange(len(rejection_mat.index)),
        labels=rejection_mat.index,
        rotation=0,
        va="center",
    )

    correction_label = (
        f"{correction_method.capitalize()} adjusted p-values"
        if correction_method != "none"
        else "no correction"
    )
    plt.title(
        f"Wilcoxon signed-rank test rejection matrix for {language}\n(alpha=0.05, {correction_label})"
    )

    plt.savefig(save_path, bbox_inches="tight")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze test suite reduction methods using statistical tests."
    )
    parser.add_argument(
        "--csv",
        default=DEFAULT_PATH,
        help=f"Path to results.csv (default: {DEFAULT_PATH})",
    )
    parser.add_argument(
        "--correction",
        choices=["holm", "bonferroni", "sidak", "none"],
        default="holm",
        help="Multiple test correction method for Wilcoxon tests (default: holm). Use 'none' to disable correction.",
    )
    args = parser.parse_args()

    df = load_and_preprocess_data(args.csv)

    # fit_glmm(df)

    # Plot performance across budgets by algorithm family and language
    plot_by_algorithm_family_and_language("outputs/fdl_across_budget.png", df)

    for language in ["C", "Java"]:
        print(f"\n=== Analysis for {language} ===\n")
        df_lang = filter_by_language(df, language)
        pivot = create_pivot(df_lang)

        friedman_results, kendall_w = run_friedman_test(pivot)
        print(
            f"Friedman test statistic: {friedman_results.statistic}, p-value: {friedman_results.pvalue}, Kendall's W: {kendall_w}"
        )

        if friedman_results.pvalue < 0.05:
            print(
                "Significant differences found between methods, proceeding to pairwise Wilcoxon tests."
            )
            results = run_pairwise_wilcoxon(pivot, args.correction)
            print(results.sort_values(by="p_adj"))
            plot_rejection_matrix(
                results,
                language,
                args.correction,
                f"outputs/rejection_matrix_{language.lower()}.png",
            )
        else:
            print("No significant differences found.")


if __name__ == "__main__":
    main()
