import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import numpy as np


base_path = Path(__file__).parent.parent

DATA_PATH = base_path / "data/results-2026-01-23-115413/results.csv"
SAVE_PATH = base_path / "presentation_examples" / "output"
SAVE_PATH.mkdir(exist_ok=True)


def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)

    data["budget"] = data["budget_prop_requested"]

    method_map = {
        "random_baseline": "Random reduction",
        "cs_tf_srp": "FAST-CS TF",
        "cs_emb": "FAST-CS EMB",
        "pp_tf_srp": "FAST++ TF",
        "pp_emb": "FAST++ EMB",
    }

    data["method"] = data["method"].map(method_map)

    return data


def create_data() -> pd.DataFrame:
    data = dict()
    data["variable"] = [
        "FAST++ TF",
        "FAST-CS EMB",
        "FAST++ EMB",
        "Budget",
    ]

    data["OR"] = [
        0.888,
        0.974,
        1.134,
        1.052,
    ]

    data["OR_upper"] = [
        0.925,
        1.016,
        1.183,
        1.052,
    ]

    data["OR_lower"] = [
        0.852,
        0.934,
        1.088,
        1.051,
    ]

    return pd.DataFrame(data)


def compute_probabilities(or_data: pd.DataFrame, initial_odds: float) -> pd.DataFrame:
    result = or_data.copy()
    log_odds = result["OR"] * initial_odds
    log_odds_lower = result["OR_lower"] * initial_odds
    log_odds_upper = result["OR_upper"] * initial_odds

    result["prob"] = np.exp(log_odds) / (1 + np.exp(log_odds))
    result["prob_upper"] = np.exp(log_odds_upper) / (1 + np.exp(log_odds_upper))
    result["prob_lower"] = np.exp(log_odds_lower) / (1 + np.exp(log_odds_lower))

    return result.drop(columns=["OR", "OR_upper", "OR_lower"])


def plot_curves(df: pd.DataFrame, language: str, savepath: Path) -> None:
    _, ax = plt.subplots(1, 1, figsize=(10, 4), sharex=True, sharey=True)

    sns.lineplot(
        data=df,
        x="budget",
        y="fdl",
        hue="method",
        style="method",
        # estimator="mean",
        dashes={
            "Random reduction": (2, 2),
            "FAST-CS TF": (1, 0),
            "FAST-CS EMB": (1, 0),
            "FAST++ TF": (1, 0),
            "FAST++ EMB": (1, 0),
        },
        errorbar=None,  # bootstrap CI by default
        # n_boot=1000,  # increase for smoother CI
        ax=ax,
    )

    ax.set_xlabel("Budget")
    ax.set_ylabel("FDL")
    ax.set_title(f"Fault Detection Loss (FDL) across budgets for {language}")

    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath)
    plt.show()


def plot_bars_auc(df: pd.DataFrame, language: str, savepath: Path):
    _, ax = plt.subplots(1, 1, figsize=(10, 4), sharex=True, sharey=True)

    df = df.sort_values("budget")

    df = (
        df.groupby(["method", "test_suite", "run_id", "language", "algorithm_family"])[
            [
                "method",
                "fdl",
                "budget",
            ]
        ]
        .apply(lambda data: np.trapezoid(data["fdl"], data["budget"]))
        .rename("auc")
        .reset_index()
    )

    df = df.sort_values("method", ascending=False)

    palette = ["lightgrey", "tab:blue", "tab:blue", "tab:blue", "tab:blue"]
    # palette = None

    bar = sns.barplot(df, x="method", y="auc", estimator="mean", palette=palette, ax=ax)

    ax.set_xlabel("Method")
    ax.set_ylabel("FDL")

    ax.set_title(f"Average Fault Detection Loss (FDL) for each method for {language}")

    plt.tight_layout()
    plt.savefig(savepath)
    plt.show()


def plot_bars_budget(df: pd.DataFrame, language: str, savepath: Path):
    _, ax = plt.subplots(1, 1, figsize=(10, 4), sharex=True, sharey=True)

    df = df[df["budget"].isin([0.1, 0.2, 0.5])]

    sns.barplot(df, x="budget", y="fdl", hue="method", estimator="mean")

    ax.set_xlabel("Budget")
    ax.set_ylabel("FDL")

    ax.set_title(
        f"Fault Detection Loss (FDL) for each method across budgets for {language}"
    )

    plt.tight_layout()
    plt.savefig(savepath)
    plt.show()


def create_language_plots(data, language):
    plot_curves(
        data[data["language"] == language],
        language,
        SAVE_PATH / f"curve_{language}.png",
    )
    plot_bars_auc(
        data[data["language"] == language],
        language,
        SAVE_PATH / f"bar_auc_{language}.png",
    )
    plot_bars_budget(
        data[data["language"] == language],
        language,
        SAVE_PATH / f"bar_budget_{language}.png",
    )


def _errorbars(data):
    print(data)
    if data[0] == 1.101:
        return (1.071, 1.132)
    if data[0] == 1.032:
        return (1.004, 1.061)
    if data[0] == 0.076:
        return (0.023, 0.252)
    if data[0] == 2.660:
        return (2.628, 2.692)

    return (data.min(), data.max())


def plot_bars_regression(df: pd.DataFrame, save_path: Path) -> None:
    # df = df.sort_values("OR")
    _, ax = plt.subplots(1, 1, figsize=(10, 4))

    # Point estimates
    sns.pointplot(data=df, x="OR", y="variable", linestyle="none", color="black", ax=ax)
    # Asymmetric confidence intervals
    xerr = np.vstack([df["OR"] - df["OR_lower"], df["OR_upper"] - df["OR"]])

    ax.errorbar(
        df["OR"],
        np.arange(len(df)),
        xerr=xerr,
        fmt="none",
        ecolor="black",
        capsize=3,
        linewidth=1.5,
    )

    # Reference line at OR = 1
    ax.axvline(1, linestyle="--", color="gray")

    ax.set_xlabel("Odds Ratio (95% CI)")
    ax.set_ylabel("")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_bars_regression_prob(
    df: pd.DataFrame, initial_odds: float, save_path: Path
) -> None:
    df = compute_probabilities(df, initial_odds)
    _, ax = plt.subplots(1, 1, figsize=(10, 4))

    # Point estimates
    sns.pointplot(
        data=df, x="prob", y="variable", linestyle="none", color="black", ax=ax
    )
    # Asymmetric confidence intervals
    xerr = np.vstack([df["prob"] - df["prob_lower"], df["prob_upper"] - df["prob"]])

    ax.errorbar(
        df["prob"],
        np.arange(len(df)),
        xerr=xerr,
        fmt="none",
        ecolor="black",
        capsize=3,
        linewidth=1.5,
    )

    # Reference line at OR = 1
    baseline_prop = np.exp(initial_odds) / (1 + np.exp(initial_odds))
    ax.axvline(baseline_prop, linestyle="--", color="gray")

    ax.set_xlabel("Probability after change (95% CI)")
    ax.set_ylabel("")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def main():
    data = load_data()

    create_language_plots(data, "C")
    create_language_plots(data, "Java")

    data = create_data()

    plot_bars_regression(data, SAVE_PATH / "regression_coef.png")
    plot_bars_regression_prob(data, 1.013, SAVE_PATH / "regression_coef_prob.png")


if __name__ == "__main__":
    main()
