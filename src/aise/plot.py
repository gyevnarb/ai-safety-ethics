"""Figures comparing the AI ethics and AI safety corpora: words and annotated categories.

Reads the preprocessed papers that ``aise analyze`` writes (output/processed_data.csv)
and the taxonomies in data/categories.csv; figures go to <output>/plots/.

    uv run aise plot all -d output/categories/processed_data.csv
"""

import re
import textwrap
from collections import Counter
from pathlib import Path
from typing import Annotated

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from adjustText import adjust_text
from matplotlib.colors import Normalize
from matplotlib.offsetbox import HPacker
from matplotlib.transforms import blended_transform_factory, offset_copy

from aise import config
from aise.style import FIELD_COLORS, GRID, INK, INK_2

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



def get_color(z: pd.Series) -> np.ndarray:
    """Diverging colours for z-scores, from the lowest (blue) to the highest (red)."""
    return mpl.colormaps["seismic"](Normalize(vmin=z.min(), vmax=z.max())(z))


def get_tops(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """The n first and n last rows, sorted by z-score."""
    return pd.concat([df.head(n), df.tail(n)]).sort_values("z_score")


def _ensure_plots_dir(output_dir: Path) -> Path:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def load_inputs(
    data_path: Path = config.PROCESSED_DATA,
    categories_path: Path = config.CATEGORIES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The preprocessed papers and the category taxonomies."""
    data = pd.read_csv(data_path, index_col=0)
    categories = pd.read_csv(categories_path)
    return data, categories


def compute_log_odds_df(data: pd.DataFrame, alpha: float = 0.01) -> pd.DataFrame:
    """Per word, the z-scored log odds ratio of Ethics vs Safety, with prior alpha."""
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


# hatching tells the corpora apart in greyscale print
FIELD_HATCHES = {"Ethics": None, "Safety": "//////"}

# Overlap figures are sized to sit two abreast (one per field) on a ~7in text width.
OVERLAP_WIDTH = 3.4
OVERLAP_ROW = 0.27  # height per category
OVERLAP_MARGIN = 0.65  # height for the legend and axis labels
OVERLAP_LEGEND = 0.16  # height of the legend strip at the top
OVERLAP_XLABEL_OFFSET = 15  # points from the x axis down to its label
OVERLAP_RC = {
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.edgecolor": INK_2,
    "xtick.color": INK_2,
    "ytick.color": INK,
    "hatch.linewidth": 0.5,
}


def _inline_legend_title(leg, fig, ax):
    """Put a legend's title on the same line as its entries.

    Matplotlib always stacks the title above them, so this swaps in a horizontal box
    (it relies on private Legend attributes).
    """
    box = leg._legend_box  # noqa: SLF001
    leg._legend_box = HPacker(  # noqa: SLF001
        pad=box.pad, sep=leg.columnspacing * leg._fontsize, align="center",  # noqa: SLF001
        children=[leg._legend_title_box, leg._legend_handle_box],  # noqa: SLF001
    )
    leg._legend_box.set_figure(fig)  # noqa: SLF001
    leg._legend_box.axes = ax  # noqa: SLF001
    leg._legend_box.set_offset(leg._findoffset)  # noqa: SLF001


def _overlap_label(text: str, width: int = 28, max_lines: int = 2) -> str:
    text = re.sub(r"\(.*\)$", "", text).strip()
    lines = textwrap.wrap(text, width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][: width - 1].rstrip() + "\u2026"
    return "\n".join(lines)


def plot_category_overlap(
    data: pd.DataFrame,
    categories: pd.DataFrame,
    output_dir: Path = config.OUTPUT_DIR,
    n_labels: int = 10,
    ordering: str = "diff",  # or "sim"
    show: bool = False,
    png: bool = False,
) -> None:
    """For each field's categories, count papers from each corpus annotated with them.

    One figure per field and category type, each sized to half the text width, with
    the same size, x-scale and colours so that the Ethics and Safety figures
    can be placed side by side.
    """
    plots_dir = _ensure_plots_dir(output_dir)
    fields = list(FIELD_COLORS)
    field_of = categories.set_index("low")["field"]
    for cat_type in ["risk", "mitigation"]:
        cat_col = f"mixed_{cat_type}_categories"
        all_cats = data[cat_col].dropna().str.split(";", expand=True).stack()
        error_cat = all_cats[~all_cats.isin(categories["low"])]
        if not error_cat.empty:
            typer.secho(f"Unknown categories in {cat_col}:")
            typer.secho(error_cat.to_string())
        all_cats = all_cats[all_cats.isin(categories["low"])]
        all_cats = all_cats.reset_index().rename(columns={"level_1": "cat_n", 0: "low"})
        all_cats["corpus"] = data.loc[all_cats["custom_id"], "corpus"].to_numpy()
        all_cats["field"] = field_of.loc[all_cats["low"]].to_numpy()

        # papers per category and corpus, restricted to each field's own categories
        tables = {}
        for field in fields:
            counts = pd.crosstab(
                all_cats.loc[all_cats["field"] == field, "low"],
                all_cats.loc[all_cats["field"] == field, "corpus"],
            ).reindex(columns=fields, fill_value=0)
            other = fields[1 - fields.index(field)]
            if ordering == "sim":
                key = counts[other]
            else:
                key = (counts[field] - counts[other]) / counts[field].clip(lower=1)
            tables[field] = counts.loc[key.sort_values(ascending=False).index[:n_labels]]
        # a common x-scale so bar lengths compare across the two panels
        xmax = max(t.to_numpy().max() for t in tables.values()) * 1.12

        for field, counts in tables.items():
            n = len(counts)
            height = OVERLAP_MARGIN + n * OVERLAP_ROW
            with plt.rc_context(OVERLAP_RC):
                fig, ax = plt.subplots(figsize=(OVERLAP_WIDTH, height))
                y = np.arange(n)
                bar_h = 0.38
                for j, corpus in enumerate(fields):
                    offset = (j - 0.5) * (bar_h + 0.04)
                    values = counts[corpus].to_numpy()
                    ax.barh(
                        y + offset, values, height=bar_h, facecolor=FIELD_COLORS[corpus],
                        hatch=FIELD_HATCHES[corpus], edgecolor="white", linewidth=0,
                        label=corpus, zorder=2,
                    )
                    for yi, v in zip(y + offset, values, strict=True):
                        ax.text(
                            v + xmax * 0.01, yi, f"{v}", ha="left", va="center",
                            fontsize=5.5, color=INK_2,
                        )
                ax.set_yticks(y, [_overlap_label(c) for c in counts.index])
                ax.set_ylim(n - 0.5, -0.5)
                ax.set_xlim(0, xmax)
                ax.tick_params(axis="y", length=0, pad=3)
                ax.tick_params(axis="x", length=2.5, width=0.6)
                ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
                for side in ["top", "right", "left"]:
                    ax.spines[side].set_visible(False)
                ax.spines["bottom"].set_linewidth(0.6)
                kind = cat_type.capitalize()
                ax.set_xlabel(rf"Number of $\bf{{{field}}}$ $\bf{{{kind}}}$ Categories",
                              color=INK_2)
                # centred on the canvas like the legend, so a label wider than the bar
                # area is not pushed off the right edge
                below_ticks = offset_copy(
                    blended_transform_factory(fig.transFigure, ax.transAxes),
                    fig=fig, y=-OVERLAP_XLABEL_OFFSET, units="points",
                )
                ax.xaxis.set_label_coords(0.5, 0, transform=below_ticks)
                # the field's own corpus first, as in the legend's original order
                handles = dict(zip(*ax.get_legend_handles_labels()[::-1], strict=True))
                order = sorted(handles, key=lambda c: c != field)
                leg = ax.legend(
                    [handles[c] for c in order], order, title="Annotated as",
                    loc="upper center", bbox_to_anchor=(0.5, 1.0),
                    bbox_transform=fig.transFigure, ncol=2, frameon=False,
                    borderaxespad=0.3, handlelength=1.2, handleheight=0.8,
                    columnspacing=1.2, title_fontproperties={"size": 6.5},
                )
                _inline_legend_title(leg, fig, ax)
                # the legend spans the canvas, so lay out the axes below its strip
                leg.set_in_layout(False)
                fig.tight_layout(rect=(0, 0, 1, 1 - OVERLAP_LEGEND / height))
                stem = f"fig_overlap_{ordering}_{field}_{cat_type}"
                fig.savefig(plots_dir / f"{stem}.pdf")
                if png:
                    fig.savefig(plots_dir / f"{stem}.png", dpi=300)
                if show:
                    plt.show()
                plt.close(fig)


def plot_categories(
    data: pd.DataFrame,
    categories: pd.DataFrame,
    output_dir: Path = config.OUTPUT_DIR,
    level: str = "high",
    show: bool = False,
) -> None:
    """Papers per risk and mitigation category at the given taxonomic level, by field."""
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
    output_dir: Path = config.OUTPUT_DIR,
    labels_top_n: int = 10,
    tops_n: int = 20,
    show: bool = False,
) -> None:
    """Log odds ratio of each word against its frequency, labelling the extremes."""
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
    adjust_text(texts, arrowprops={"arrowstyle": "-", "color": "gray", "lw": 0.5,
                                   "alpha": 0.6})
    plt.savefig(plots_dir / "fig_log_odds.pdf")
    if show:
        plt.show()
    plt.close()


def plot_total_freq(
    df: pd.DataFrame,
    output_dir: Path = config.OUTPUT_DIR,
    show: bool = False,
) -> None:
    """Relative frequencies of the words most distinctive of each corpus."""
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


# ---- commands -------------------------------------------------------------------------
DataPath = Annotated[
    Path, typer.Option("--data", "-d", help="Path to processed data CSV")]
CategoriesPath = Annotated[
    Path, typer.Option("--categories", "-c", help="Path to categories CSV")]
OutputDir = Annotated[Path, typer.Option("--output", "-o", help="Output directory")]
Alpha = Annotated[float, typer.Option("--alpha", help="Dirichlet prior strength")]
NLabels = Annotated[
    int, typer.Option("--n-labels", help="Number of categories in each overlap plot")]
Ordering = Annotated[
    str, typer.Option(help="Ordering method for overlap plots: 'diff' or 'sim'")]
Level = Annotated[str, typer.Option(help="The taxonomic level of the category plots.")]
Show = Annotated[bool, typer.Option("--show", help="Show plots interactively")]

app = typer.Typer(rich_markup_mode="rich", help=__doc__.split("\n\n")[0])


@app.command()
def overlap(
    data_path: DataPath = config.PROCESSED_DATA,
    categories_path: CategoriesPath = config.CATEGORIES,
    output_dir: OutputDir = config.OUTPUT_DIR,
    n_labels: NLabels = 10,
    ordering: Ordering = "diff",
    show: Show = False,
    png: Annotated[bool, typer.Option("--png", help="Also save PNG copies")] = False,
) -> None:
    """Plot category overlap between Safety and Ethics corpora."""
    data, cats = load_inputs(data_path, categories_path)
    plot_category_overlap(
        data, cats, output_dir=output_dir, show=show, n_labels=n_labels,
        ordering=ordering, png=png,
    )
    typer.secho("Generated category overlap plots.")


@app.command()
def categories(
    data_path: DataPath = config.PROCESSED_DATA,
    categories_path: CategoriesPath = config.CATEGORIES,
    output_dir: OutputDir = config.OUTPUT_DIR,
    level: Level = "high",
    show: Show = False,
) -> None:
    """Plot category counts for Risk Types and Mitigation Strategies."""
    data, cats = load_inputs(data_path, categories_path)
    plot_categories(data, cats, output_dir=output_dir, level=level, show=show)
    typer.secho("Generated category counts plot.")


@app.command()
def log_odds(
    data_path: DataPath = config.PROCESSED_DATA,
    output_dir: OutputDir = config.OUTPUT_DIR,
    alpha: Alpha = 0.01,
    labels_top_n: Annotated[
        int, typer.Option("--labels-top-n", help="Top labels annotated")] = 10,
    tops_n: Annotated[int, typer.Option("--tops-n", help="Top/Bottom words listed")] = 20,
    show: Show = False,
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
    data_path: DataPath = config.PROCESSED_DATA,
    output_dir: OutputDir = config.OUTPUT_DIR,
    alpha: Alpha = 0.01,
    show: Show = False,
) -> None:
    """Plot total frequencies of most distinctive words."""
    data, _ = load_inputs(data_path)
    df = compute_log_odds_df(data, alpha=alpha)
    plot_total_freq(df, output_dir=output_dir, show=show)
    typer.secho("Generated total frequencies plot.")


@app.command("all")
def all_plots(
    data_path: DataPath = config.PROCESSED_DATA,
    categories_path: CategoriesPath = config.CATEGORIES,
    output_dir: OutputDir = config.OUTPUT_DIR,
    alpha: Alpha = 0.01,
    n_labels: NLabels = 10,
    ordering: Ordering = "diff",
    level: Level = "high",
    show: Show = False,
) -> None:
    """Run all plotting commands."""
    data, cats = load_inputs(data_path, categories_path)
    df = compute_log_odds_df(data, alpha=alpha)
    plot_category_overlap(
        data, cats, output_dir=output_dir, show=show, n_labels=n_labels, ordering=ordering
    )
    plot_categories(data, cats, output_dir=output_dir, level=level, show=show)
    plot_log_odds(df, output_dir=output_dir, show=show)
    plot_total_freq(df, output_dir=output_dir, show=show)


if __name__ == "__main__":
    app()
