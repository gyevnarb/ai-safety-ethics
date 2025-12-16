import re
import textwrap
from collections import Counter

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize

ABBREVIATIONS = {
    "rl": "reinforcement learning",
    "ais": "AI safety",
    "sae": "sparse autoencoder",
    "ssl": "self-supervised learning",
    "cot": "chain of thought",
    "ood": "out of distribution",
    "rai": "responsible AI",
}

get_color = lambda z: cm.get_cmap("seismic")(Normalize(vmin=z.min(), vmax=z.max())(z))
get_tops = lambda df, n: pd.concat([df.head(n), df.tail(n)]).sort_values("z_score")

data = pd.read_csv("output/processed_data.csv", index_col=0)
categories = pd.read_csv("data/categories.csv")
corpus_ethics = data[data["corpus"] == "Ethics"]["clean_text"].tolist()
corpus_safety = data[data["corpus"] == "Safety"]["clean_text"].tolist()

tokens_ethics = [tok for doc in corpus_ethics for tok in doc.split()]
tokens_safety = [tok for doc in corpus_safety for tok in doc.split()]

counts_ethics = Counter(tokens_ethics)
counts_safety = Counter(tokens_safety)

vocab = list(set(counts_ethics.keys()) | set(counts_safety.keys()))

N_A = sum(counts_ethics.values())
N_B = sum(counts_safety.values())

# -----------------------------------------------------------
# 4. Log-odds ratio with informative Dirichlet prior
#    (Monroe et al., 2008)
# -----------------------------------------------------------
# Use background frequencies as priors
alpha = 0.01
alpha_0 = alpha * len(vocab)

freq_data = []
for word in vocab:
    cA = counts_ethics[word]
    cB = counts_safety[word]

    # Smoothed frequencies
    freq_A = (cA + alpha) / (N_A + alpha_0)
    freq_B = (cB + alpha) / (N_B + alpha_0)

    # Log odds for each corpus
    log_odds_A = np.log(freq_A / (1 - freq_A))
    log_odds_B = np.log(freq_B / (1 - freq_B))

    # Variance
    var = (1 / (cA + alpha)) + (1 / (cB + alpha))

    # z-score
    z = (log_odds_A - log_odds_B) / np.sqrt(var)

    freq_data.append((word, z, cA + cB, cA, cB, freq_A, freq_B))

df = pd.DataFrame(
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
)
df = df.sort_values("z_score", ascending=False)

# -----------------------------------------------------------
# 5. Category Visualization
# -----------------------------------------------------------
cat_df = []
for cat_type in ["risk", "mitigation"]:
    cat_col = f"{cat_type}_categories"
    all_cats = data[cat_col].dropna().str.split(";", expand=True).stack()
    error_cat = all_cats[~all_cats.isin(categories["low"])]
    if not error_cat.empty:
        print(f"Unknown categories in {cat_col}:")
        print(error_cat)
    cat_counts = all_cats.value_counts()
    cat_df.append(cat_counts.rename("count"))
cat_df = pd.concat(cat_df).fillna(0).astype(int)
cat_df.to_csv("output/category_counts.csv")
cat_df = categories.join(cat_df, how="left", on="low")
cat_df["count"] = cat_df["count"].fillna(0)
counts = cat_df.groupby(["field", "type", "high"]).sum()["count"].reset_index()
print(counts)

hatches = ["///", "\\\\", "...", "xxx", "+++", "***"]
for field, cat_type in counts[["field", "type"]].drop_duplicates().values:
    sub_counts = counts[(counts["field"] == field) & (counts["type"] == cat_type)]
    plt.figure(figsize=(8, 4))
    sns.barplot(
        data=sub_counts,
        y="high",
        x="count",
        palette="Reds" if field == "Safety" else "Blues",
        order=sub_counts.sort_values("count", ascending=False)["high"],
    )
    ax = plt.gca()
    for i, bar in enumerate(ax.patches):
        bar.set_hatch(hatches[i % len(hatches)])
        bar.set_edgecolor("black")
    max_width = 30  # character width before wrapping; adjust as needed
    new_labels = []
    for label in ax.get_yticklabels():
        text = label.get_text()
        wrapped = "\n".join(textwrap.wrap(text, max_width))
        new_labels.append(wrapped)

    ax.set_yticklabels(new_labels)
    plt.xlabel(
        f"Category Counts for {field} {'Risk Types' if cat_type == 'Risk' else 'Mitigation Strategies'}"
    )
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(f"output/plots/fig_category_counts_{field}_{cat_type}.png", dpi=300)
    plt.show()


# -----------------------------------------------------------
# 6. Frequency Visualization
# -----------------------------------------------------------
plot_df = get_tops(df, 20)
score_type = "z_score"

plt.figure(figsize=(7, 6))
plt.scatter(
    df["total_count"], df[score_type], s=1, c=get_color(df[score_type]), alpha=0.5
)
for _, plot_row in get_tops(df, 10).iterrows():
    plt.text(
        plot_row["total_count"],
        plot_row[score_type],
        plot_row["word"],
        fontsize=10,
        ha="center",
        va="bottom",
    )
font_sizes = np.interp(
    np.abs(plot_df[score_type]), (0, max(np.abs(plot_df[score_type]))), (2, 12)
)
for i, (_, plot_row) in enumerate(plot_df.iterrows()):
    y_pos = plot_df[score_type].min() + i / len(plot_df) * (
        plot_df[score_type].max() - plot_df[score_type].min()
    )
    plt.text(1, y_pos, plot_row["word"], fontsize=font_sizes[i], ha="left", va="bottom")
plt.axhline(0, color="black", linewidth=1, ls=":", alpha=0.4)
plt.xscale("log")
plt.xlabel("Total Count (log scale)")
plt.ylabel("z-score (positive → Ethics, negative → Safety)")
plt.tight_layout()
plt.savefig("output/plots/fig_log_odds.png", dpi=300)
plt.show()

pA = df.sort_values("ethics_count", ascending=False)[df["safety_count"] < 5].head(20)
pB = df.sort_values("safety_count", ascending=False)[df["ethics_count"] < 5].head(20)
plot_df = pd.concat([pA, pB])

plt.figure(figsize=(7, 6))
bars_ethics = plt.barh(
    plot_df["word"], plot_df["ethics_freq"], color="blue", alpha=0.6, label="Ethics"
)
bars_safety = plt.barh(
    plot_df["word"], -plot_df["safety_freq"], color="red", alpha=0.6, label="Safety"
)

plt.legend()
plt.axvline(0, color="black", linewidth=1)
plt.xlabel("Safety ← Corpus Frequency → Ethics")
plt.ylabel("Most Distinctive Words")
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
xticks = ax.get_xticks()
ax.set_xticklabels([f"{abs(tick)}" for tick in xticks])
plt.tight_layout()
plt.savefig("output/plots/fig_total_freq.png", dpi=300)
plt.show()
