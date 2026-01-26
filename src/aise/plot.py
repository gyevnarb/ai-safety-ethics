import re
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from adjustText import adjust_text
from matplotlib import cm, rcParams
from matplotlib.colors import Normalize

ABBREVIATIONS = {
    "rl": "reinforcement learning",
    "ais": "AI safety",
    "sae": "sparse autoencoder",
    "ssl": "self-supervised learning",
    "cot": "chain of thought",
    "ood": "out of distribution",
    "rai": "responsible AI",
    "dnn": "deep neural network",
}

HATCHES = ["///", "\\\\", "...", "xxx", "+++", "***"]

get_color = lambda z: cm.get_cmap("seismic")(Normalize(vmin=z.min(), vmax=z.max())(z))  # noqa: E731
get_tops = lambda df, n: pd.concat([df.head(n), df.tail(n)]).sort_values("z_score")  # noqa: E731

rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42


def _ensure_plots_dir(output_dir: Path) -> Path:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def load_inputs(
    data_path: Path = Path("output/processed_data.csv"),
    categories_path: Path = Path("data/categories.csv"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(data_path, index_col=0)
    categories = pd.read_csv(categories_path)
    return data, categories


def compute_log_odds_df(data: pd.DataFrame, alpha: float = 0.01) -> pd.DataFrame:
    corpus_ethics = data[data["corpus"] == "Ethics"]["clean_text"].tolist()
    corpus_safety = data[data["corpus"] == "Safety"]["clean_text"].tolist()

    tokens_ethics = [tok for doc in corpus_ethics for tok in doc.split()]
    tokens_safety = [tok for doc in corpus_safety for tok in doc.split()]

    counts_ethics = Counter(tokens_ethics)
    counts_safety = Counter(tokens_safety)

    vocab = list(set(counts_ethics.keys()) | set(counts_safety.keys()))
    N_A = sum(counts_ethics.values())
    N_B = sum(counts_safety.values())
    alpha_0 = alpha * len(vocab)

    freq_data: list[tuple[str, float, int, int, int, float, float]] = []
    for word in vocab:
        cA = counts_ethics[word]
        cB = counts_safety[word]

        freq_A = (cA + alpha) / (N_A + alpha_0)
        freq_B = (cB + alpha) / (N_B + alpha_0)

        log_odds_A = np.log(freq_A / (1 - freq_A))
        log_odds_B = np.log(freq_B / (1 - freq_B))
        var = (1 / (cA + alpha)) + (1 / (cB + alpha))
        z = (log_odds_A - log_odds_B) / np.sqrt(var)

        freq_data.append((word, z, cA + cB, cA, cB, freq_A, freq_B))

    return pd.DataFrame(
        freq_data,
        columns=[
            "word",
            "z_score",
            "total_count",
            "ethics_count",
            "safety_count",
            "ethics_freq",
            "safety_freq",
        ],
    ).sort_values("z_score", ascending=False)


def plot_category_overlap(
    data: pd.DataFrame,
    categories: pd.DataFrame,
    output_dir: Path = Path("output"),
    n_labels: int = 10,
    ordering: str = "diff",  # or "sim"
    show: bool = False,
) -> None:
    for cat_type in ["risk", "mitigation"]:
        cat_col = f"mixed_{cat_type}_categories"
        all_cats = data[cat_col].dropna().str.split(";", expand=True).stack()
        error_cat = all_cats[~all_cats.isin(categories["low"])]
        if not error_cat.empty:
            typer.secho(f"Unknown categories in {cat_col}:")
            typer.secho(error_cat.to_string())
        all_cats = all_cats.reset_index().rename(columns={"level_1": "cat_n", 0: "low"})
        all_cats[f"{cat_type}_document_corpus"] = data.loc[
            all_cats["custom_id"], "corpus"
        ].to_numpy()
        all_cats[f"{cat_type}_category_corpus"] = (
            categories.set_index("low").loc[all_cats["low"], "field"].to_numpy()
        )
        all_cats["high"] = (
            categories.set_index("low").loc[all_cats["low"], "high"].to_numpy()
        )
        all_cats["is_overlap"] = (
            all_cats[f"{cat_type}_document_corpus"]
            != all_cats[f"{cat_type}_category_corpus"]
        )

        for field in categories["field"].unique():
            sub_cats = all_cats[all_cats[f"{cat_type}_category_corpus"] == field]
            plots_dir = _ensure_plots_dir(output_dir)
            palette = ["C3", "C0"]
            if ordering == "sim":
                order = (
                    sub_cats[sub_cats["is_overlap"]]["low"]
                    .value_counts()
                    .sort_values(ascending=False)
                    .index
                )
            else:
                order = (
                    sub_cats.value_counts(["low", "is_overlap"])
                    .unstack(fill_value=0)
                    .pipe(lambda x: (x[False] - x[True]) / (x[False]))
                    .sort_values(ascending=False)
                    .index
                )
            plt.figure(figsize=(6, 4))
            sns.countplot(
                data=sub_cats,
                y="low",
                hue="is_overlap",
                order=order[:n_labels],
                palette=palette,
            )
            ax = plt.gca()
            # Find the maximum width among all bars
            max_width_bar = max((bar.get_width() for bar in ax.patches), default=0)
            for bar in ax.patches:
                bar.set_edgecolor("black")
                # Add count labels to the right of bars (or inside for the longest bar)
                width = bar.get_width()
                if width > 0:
                    if width == max_width_bar:
                        # Longest bar: label inside to avoid border overlap
                        ax.text(
                            width * 0.98,
                            bar.get_y() + bar.get_height() / 2,
                            f"{int(width)}",
                            ha="right",
                            va="center",
                            color="white",
                            fontsize=9,
                        )
                    else:
                        # Other bars: label to the right
                        ax.text(
                            width,
                            bar.get_y() + bar.get_height() / 2,
                            f" {int(width)}",
                            ha="left",
                            va="center",
                            fontsize=9,
                        )
            max_width = 30
            new_labels = []
            for label in ax.get_yticklabels():
                text = label.get_text()
                # label.set_color("red" if field == "Safety" else "blue")
                text = re.sub(r"\(.*\)$", "", text).strip()
                wrapped = "\n".join(textwrap.wrap(text[:90], max_width))
                new_labels.append(wrapped)
            ax.set_yticklabels(new_labels)

            handles, labels = ax.get_legend_handles_labels()
            labels = ["Safety", "Ethics"] if field == "Safety" else ["Ethics", "Safety"]
            ax.legend(handles, labels, title="Annotated as")
            # plt.xlabel("Number of Documents")
            plt.xlabel(f"Number of {field} {cat_type.capitalize()} Categories")
            plt.ylabel("")
            plt.grid(axis="x", linestyle=":", alpha=0.7)
            plt.tight_layout()
            plt.savefig(plots_dir / f"fig_overlap_{ordering}_{field}_{cat_type}.pdf")
            if show:
                plt.show()
            plt.close()


def plot_categories(
    data: pd.DataFrame,
    categories: pd.DataFrame,
    output_dir: Path = Path("output"),
    level: str = "high",
    n_labels: int = 10,
    show: bool = False,
) -> None:
    cat_df = []
    for cat_type in ["risk", "mitigation"]:
        cat_col = f"{cat_type}_categories"
        all_cats = data[cat_col].dropna().str.split(";", expand=True).stack()
        error_cat = all_cats[~all_cats.isin(categories["low"])]
        if not error_cat.empty:
            typer.secho(f"Unknown categories in {cat_col}:")
            typer.secho(error_cat.to_string())
        cat_counts = all_cats.value_counts()
        cat_df.append(cat_counts.rename("count"))
    cat_df = pd.concat(cat_df).fillna(0).astype(int)
    (output_dir / "category_counts.csv").write_text(cat_df.to_csv())
    cat_df = categories.join(cat_df, how="left", on="low")
    cat_df["count"] = cat_df["count"].fillna(0)

    counts = cat_df.groupby(["field", "type", level]).sum()["count"].reset_index()
    plots_dir = _ensure_plots_dir(output_dir)
    for field, cat_type in counts[["field", "type"]].drop_duplicates().to_numpy():
        sub_counts = counts[(counts["field"] == field) & (counts["type"] == cat_type)]
        # sub_counts = sub_counts.head(n_labels)
        plt.figure(figsize=(6, 4))
        sns.barplot(
            data=sub_counts,
            y=level,
            x="count",
            palette="Reds" if field == "Safety" else "Blues",
            order=sub_counts.sort_values("count", ascending=False)[level],
        )
        ax = plt.gca()
        for i, bar in enumerate(ax.patches):
            # bar.set_hatch(HATCHES[i % len(HATCHES)])
            bar.set_edgecolor("black")
            # Add count labels next to bars (or inside for the first bar)
            width = bar.get_width()
            if i == 0:
                # First label inside the bar to avoid border overlap
                ax.text(
                    width * 0.98,
                    bar.get_y() + bar.get_height() / 2,
                    f"{int(width)}",
                    ha="right",
                    va="center",
                    fontsize=9,
                )
            else:
                # Other labels next to bars
                ax.text(
                    width,
                    bar.get_y() + bar.get_height() / 2,
                    f" {int(width)}",
                    ha="left",
                    va="center",
                    fontsize=9,
                )
        max_width = 30
        new_labels = []
        for label in ax.get_yticklabels():
            text = label.get_text()
            wrapped = "\n".join(textwrap.wrap(text[:100], max_width))
            new_labels.append(wrapped)

        ax.set_yticklabels(new_labels)
        ax.set_axisbelow(True)
        xlabel_str = "Risk Types" if cat_type == "Risk" else "Mitigation Strategies"
        plt.xlabel(f"Category Counts for {field} {xlabel_str}")
        plt.ylabel("")
        plt.grid(axis="x", linestyle=":", alpha=0.7)
        plt.tight_layout()
        plt.savefig(
            plots_dir / f"fig_category_counts_{level}_{field}_{cat_type}.pdf",
            bbox_inches="tight",
        )
        if show:
            plt.show()
        plt.close()


def plot_log_odds(
    df: pd.DataFrame,
    output_dir: Path = Path("output"),
    labels_top_n: int = 10,
    tops_n: int = 20,
    show: bool = False,
) -> None:
    plot_df = get_tops(df, tops_n)
    score_type = "z_score"

    plots_dir = _ensure_plots_dir(output_dir)
    plt.figure(figsize=(7, 6))
    plt.scatter(
        df["total_count"], df[score_type], s=1, c=get_color(df[score_type]), alpha=0.5
    )

    # Collect text objects for label adjustment
    texts = []
    for _, plot_row in get_tops(df, labels_top_n).iterrows():
        text = plt.text(
            plot_row["total_count"],
            plot_row[score_type],
            plot_row["word"],
            fontsize=10,
            ha="center",
            va="bottom",
        )
        texts.append(text)

    # Adjust overlapping labels with connecting arrows
    font_sizes = np.interp(
        np.abs(plot_df[score_type]), (0, max(np.abs(plot_df[score_type]))), (2, 12)
    )
    for i, (_, plot_row) in enumerate(plot_df.iterrows()):
        y_pos = plot_df[score_type].min() + i / len(plot_df) * (
            plot_df[score_type].max() - plot_df[score_type].min()
        )
        word = plot_row["word"]
        plt.text(
            1,
            y_pos,
            ABBREVIATIONS.get(word, word),
            fontsize=font_sizes[i],
            ha="left",
            va="bottom",
        )
    plt.axhline(0, color="black", linewidth=1, ls=":", alpha=0.4)
    plt.xscale("log")
    plt.xlabel("Total Count (log scale)", fontsize=14)
    plt.ylabel("Normalized log-odds ratio (z-score)", fontsize=14)
    plt.grid(axis="both", linestyle=":", alpha=0.7)
    plt.tight_layout()
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.6))
    plt.savefig(plots_dir / "fig_log_odds.pdf")
    if show:
        plt.show()
    plt.close()


def plot_total_freq(
    df: pd.DataFrame,
    output_dir: Path = Path("output"),
    show: bool = False,
) -> None:
    pA = df.sort_values("ethics_count", ascending=False)[df["safety_count"] < 5].head(20)
    pB = df.sort_values("safety_count", ascending=False)[df["ethics_count"] < 5].head(20)
    plot_df = pd.concat([pA, pB])

    plots_dir = _ensure_plots_dir(output_dir)
    plt.figure(figsize=(7, 6))
    bars_ethics = plt.barh(
        plot_df["word"], plot_df["ethics_freq"], color="blue", alpha=0.6, label="Ethics"
    )
    plt.barh(
        plot_df["word"], -plot_df["safety_freq"], color="red", alpha=0.6, label="Safety"
    )

    plt.legend()
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("Corpus Relative Frequency", fontsize=14)
    plt.ylabel("Most Distinctive Words", fontsize=14, labelpad=20)
    ax = plt.gca()
    ax.set_yticks([])
    for bar, label in zip(bars_ethics, plot_df["word"], strict=False):
        y = bar.get_y() + bar.get_height() / 2
        sign = -1 if plot_df[plot_df["word"] == label]["z_score"].to_numpy()[0] < 0 else 1
        plt.text(
            sign * -0.02 * ax.get_xlim()[1],
            y,
            ABBREVIATIONS.get(label, label),
            va="center",
            ha="right" if sign > 0 else "left",
            fontsize=10,
        )
    # Format x-axis with scientific notation and add vertical grid lines
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    plt.grid(axis="x", linestyle=":", alpha=0.7)
    plt.xlim(-plot_df[["safety_freq"]].max().to_numpy()[0] * 1.01,
             plot_df[["ethics_freq"]].max().to_numpy()[0] * 1.01)
    plt.tight_layout()
    plt.savefig(plots_dir / "fig_total_freq.pdf")
    if show:
        plt.show()
    plt.close()


app = typer.Typer(rich_markup_mode="rich")


@app.command()
def overlap(
    data_path: Path = typer.Option(
        Path("output/processed_data.csv"),
        "--data",
        "-d",
        help="Path to processed data CSV",
    ),
    categories_path: Path = typer.Option(
        Path("data/categories.csv"), "--categories", "-c", help="Path to categories CSV"
    ),
    output_dir: Path = typer.Option(
        Path("output"), "--output", "-o", help="Output directory"
    ),
    n_labels: int = typer.Option(
        10, "--n-labels", help="Number of category labels to show in each plot"
    ),
    ordering: str = typer.Option(
        "diff", help="Ordering method for overlap plots: 'diff' or 'sim'"
    ),
    show: bool = typer.Option(False, "--show", help="Show plots interactively"),
) -> None:
    """Plot category overlap between Safety and Ethics corpora."""
    data, cats = load_inputs(data_path, categories_path)
    plot_category_overlap(
        data, cats, output_dir=output_dir, show=show, n_labels=n_labels, ordering=ordering
    )
    typer.secho("Generated category overlap plots.")


@app.command()
def categories(
    data_path: Path = typer.Option(
        Path("output/processed_data.csv"),
        "--data",
        "-d",
        help="Path to processed data CSV",
    ),
    categories_path: Path = typer.Option(
        Path("data/categories.csv"), "--categories", "-c", help="Path to categories CSV"
    ),
    output_dir: Path = typer.Option(
        Path("output"), "--output", "-o", help="Output directory"
    ),
    level: str = typer.Option("high", help="The taxonomic level to use for plotting."),
    n_labels: int = typer.Option(
        10, "--n-labels", help="Number of category labels to show in each plot"
    ),
    show: bool = typer.Option(False, "--show", help="Show plots interactively"),
) -> None:
    """Plot category counts for Risk Types and Mitigation Strategies."""
    data, cats = load_inputs(data_path, categories_path)
    plot_categories(
        data, cats, output_dir=output_dir, level=level, n_labels=n_labels, show=show
    )
    typer.secho("Generated category counts plot.")


@app.command()
def log_odds(
    data_path: Path = typer.Option(
        Path("output/processed_data.csv"),
        "--data",
        "-d",
        help="Path to processed data CSV",
    ),
    output_dir: Path = typer.Option(
        Path("output"), "--output", "-o", help="Output directory"
    ),
    alpha: float = typer.Option(0.01, "--alpha", help="Dirichlet prior strength"),
    labels_top_n: int = typer.Option(10, "--labels-top-n", help="Top labels annotated"),
    tops_n: int = typer.Option(20, "--tops-n", help="Top/Bottom words listed"),
    show: bool = typer.Option(False, "--show", help="Show plots interactively"),
) -> None:
    """Plot log-odds ratios of words between Ethics and Safety corpora."""
    data, _ = load_inputs(data_path)
    df = compute_log_odds_df(data, alpha=alpha)
    plot_log_odds(
        df, output_dir=output_dir, labels_top_n=labels_top_n, tops_n=tops_n, show=show
    )
    typer.secho("Generated log-odds plot.")


@app.command()
def freqs(
    data_path: Path = typer.Option(
        Path("output/processed_data.csv"),
        "--data",
        "-d",
        help="Path to processed data CSV",
    ),
    output_dir: Path = typer.Option(
        Path("output"), "--output", "-o", help="Output directory"
    ),
    alpha: float = typer.Option(0.01, "--alpha", help="Dirichlet prior strength"),
    show: bool = typer.Option(False, "--show", help="Show plots interactively"),
) -> None:
    """Plot total frequencies of most distinctive words."""
    data, _ = load_inputs(data_path)
    df = compute_log_odds_df(data, alpha=alpha)
    plot_total_freq(df, output_dir=output_dir, show=show)
    typer.secho("Generated total frequencies plot.")


@app.command()
def all(
    data_path: Path = typer.Option(
        Path("output/processed_data.csv"),
        "--data",
        "-d",
        help="Path to processed data CSV",
    ),
    categories_path: Path = typer.Option(
        Path("data/categories.csv"), "--categories", "-c", help="Path to categories CSV"
    ),
    output_dir: Path = typer.Option(
        Path("output"), "--output", "-o", help="Output directory"
    ),
    alpha: float = typer.Option(0.01, "--alpha", help="Dirichlet prior strength"),
    n_labels: int = typer.Option(
        10, "--n-labels", help="Number of category labels to show in overlap plots"
    ),
    ordering: str = typer.Option(
        "diff", help="Ordering method for overlap plots: 'diff' or 'sim'"
    ),
    level: str = typer.Option(
        "high", help="The taxonomic level to use for tacetogory plotting."
    ),
    show: bool = typer.Option(False, "--show", help="Show plots interactively"),
) -> None:
    """Run all plotting commands."""
    data, cats = load_inputs(data_path, categories_path)
    df = compute_log_odds_df(data, alpha=alpha)
    plot_category_overlap(
        data, cats, output_dir=output_dir, show=show, n_labels=n_labels, ordering=ordering
    )
    plot_categories(
        data, cats, output_dir=output_dir, level=level, n_labels=n_labels, show=show
    )
    plot_log_odds(df, output_dir=output_dir, show=show)
    plot_total_freq(df, output_dir=output_dir, show=show)


def cli() -> None:
    app()


if __name__ == "__main__":
    app()
