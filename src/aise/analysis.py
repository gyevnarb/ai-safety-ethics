"""Analyze paper metadata for topic modeling and network analysis.

This script will read a CSV file containing paper metadata, preprocess the text data,
perform topic modeling using BERTopic, and analyze term co-occurrence networks.
It also includes visualization of topic distributions and network graphs.

The input CSV file is expected to have at least the following columns:
- title: The title of the paper.
- abstract: The abstract of the paper.
- corpus: The corpus/category the paper belongs to (e.g., 'Ethics', 'Safety').
- year: The publication year of the paper (optional).
- authors: The authors of the paper (optional).
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
import typer
import umap
from bertopic import BERTopic
from hdbscan import HDBSCAN
from networkx.drawing.nx_pydot import graphviz_layout
from rich.console import Console
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from aise.util import pandas_to_rich

app = typer.Typer(rich_markup_mode="rich")
console = Console()

RANDOM_SEED = 42
CUSTOM_STOPWORDS = {
    "et",
    "al",
    "using",
    "however",
    "also",
    "one",
    "two",
    "new",
    "based",
    "within",
    "may",
    "many",
    "different",
    "use",
    "approach",
    "result",
    "study",
    "paper",
    "show",
    "propose",
    "system",
    "model",
    "method",
    "post",
    "pre",
    "ai",
    "ieee",
}


def read_data(file_path: Path) -> pd.DataFrame:
    """Read the CSV data of paper metadata and ensure required columns exist."""
    # Parameters - update paths if needed
    df = pd.read_csv(file_path, index_col=0)
    typer.secho(f"Total rows: {len(df)}")

    # Ensure required columns exist
    for col in ["title", "abstract", "year", "authors"]:
        if col not in df.columns:
            typer.secho(
                f'WARNING: column "{col}" not found. Consider adding it.',
                fg=typer.colors.YELLOW,
            )
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the text data in the dataframe.

    Includes tokenization, lemmatization, and stopword removal.
    """
    nltk.download("stopwords")

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    stop_words = set(nltk.corpus.stopwords.words("english")).union(CUSTOM_STOPWORDS)

    def preprocess(text: str) -> str:
        if pd.isna(text):
            return ""
        text = re.sub(r"http\S+|www\S+|doi\.org\S+", "", str(text))  # remove URLs
        text = re.sub(r"©.*", "", text, flags=re.IGNORECASE)  # remove copyright
        text = text.lower()

        doc = nlp(text)
        tokens = [
            t.lemma_
            for t in doc
            if t.is_alpha and t.lemma_ not in stop_words and len(t.lemma_) > 1
        ]
        return " ".join(tokens)

    # Apply preprocessing to abstracts (progress-safe)
    console.print("Preprocessing text data...")
    if "clean_text" not in df.columns:
        full_text = df["title"].fillna("").str.cat(df["abstract"].fillna(""), sep=" <> ")
        df["clean_text"] = full_text.map(preprocess)

    console.print("Sample cleaned text:")
    console.print(df["clean_text"].iloc[0][:300])

    return df


def get_statistics(df: pd.DataFrame) -> None:
    """Compute and display basic statistics about the dataset."""
    console.print("Dataset statistics:")

    # Basic descriptive statistics
    console.print(f"Years range: {df['year'].min()} - {df['year'].max()}")
    _df_word_counts = df["clean_text"].str.split().apply(len)
    df["word_count"] = _df_word_counts
    console.print(
        pandas_to_rich(df.groupby("corpus")["word_count"].median(), "Median word count:")
    )

    # Keyness: log-likelihood ratio to find distinctive words
    def word_freqs(series: pd.Series) -> Counter:
        words = " ".join(series).split()
        return Counter(words)

    freq_ethics = word_freqs(df[df.corpus == "Ethics"]["clean_text"])
    freq_safety = word_freqs(df[df.corpus == "Safety"]["clean_text"])

    results = []
    eps = 1e-9
    N1, N2 = sum(freq_ethics.values()), sum(freq_safety.values())
    r_total_ethics = N1 / (N1 + N2)
    r_total_safety = 1.0 - r_total_ethics
    all_words = set(list(freq_ethics.keys()) + list(freq_safety.keys()))
    for w in all_words:
        O1, O2 = freq_ethics.get(w, 0), freq_safety.get(w, 0)
        if O1 + O2 == 0:
            LL = 0.0
        else:
            E1 = r_total_ethics * (O1 + O2) + eps
            E2 = r_total_safety * (O1 + O2) + eps
            LL = 2 * ((O1 * np.log((O1 / E1) + eps)) + (O2 * np.log((O2 / E2) + eps)))
        results.append((w, LL, O1, O2))

    keyness_df = pd.DataFrame(
        results,
        columns=["word", "LL", "ethics_count", "safety_count"],
    )
    keyness_df = keyness_df.sort_values("LL", ascending=False).reset_index(drop=True)
    console.print(
        pandas_to_rich(
            keyness_df.head(30), "Top 30 distinctive words by log-likelihood ratio:"
        ),
    )


def model_topic(
    df: pd.DataFrame,
    min_cluster_size: int = 10,
    plot: bool = True,
    output_dir: Path = Path("output"),
) -> BERTopic:
    """Perform topic modeling using BERTopic and visualize results."""
    # Initialize embedding model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = df["clean_text"].tolist()

    console.print("Computing embeddings (this may take time)")
    embeddings = embedding_model.encode(texts, show_progress_bar=True)

    # Fit BERTopic
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    topic_model = BERTopic(
        embedding_model=embedding_model,
        calculate_probabilities=True,
        hdbscan_model=hdbscan_model,
    )

    topics, _ = topic_model.fit_transform(
        texts,
        embeddings,
    )
    df["topic"] = topics

    # Topic summary
    topic_info = topic_model.get_topic_info()
    topic_info["Representative_Docs"] = topic_info["Representative_Docs"].apply(
        lambda x: [x[:100] + "..." for x in x]
    )
    console.print(
        pandas_to_rich(
            topic_info.head(15),
            f"Total topics found: {len(topic_info) - 1}; Top 15 topics:",
        )
    )

    # Compare topic distributions across corpora
    if any(df["corpus"] == "Safety") and any(df["corpus"] == "Ethics"):
        topic_info = topic_info[["Topic", "Name"]].set_index("Topic")
        df["topic_name"] = df.topic.apply(lambda x: topic_info.loc[x, "Name"])

        topic_counts = df.pivot_table(
            index="corpus",
            columns=["topic", "topic_name"],
            aggfunc="size",
            fill_value=0,
        ).T

        topic_counts["ethics_prop"] = (
            topic_counts["Ethics"] / topic_counts["Ethics"].sum()
        )
        topic_counts["safety_prop"] = (
            topic_counts["Safety"] / topic_counts["Safety"].sum()
        )
        topic_counts = topic_counts[["ethics_prop", "safety_prop"]].fillna(0)
        diff = (
            (topic_counts["ethics_prop"] - topic_counts["safety_prop"])
            .abs()
            .sort_values(ascending=False)
        )
        console.print(
            pandas_to_rich(diff.head(15), "Top differing topics (by absolute prop diff):")
        )
    else:
        console.print("Topic counts could not be computed: only one corpus selected.")

    # Visualize top 10 topics by combined frequency (BERTopic visualization)
    if plot:
        plot_topic_analysis(df, topic_model, embeddings, embedding_model, output_dir)

    return topic_model


def analyze_network(
    df: pd.DataFrame,
    plot: bool = True,
    topic_model: BERTopic = None,
    output_dir: Path = Path("output"),
) -> None:
    """Analyze term co-occurrence networks and visualize."""

    def build_cooccurrence_graph(
        series: pd.Series,
        top_n: int | None = None,
    ) -> nx.Graph:
        vectorizer = TfidfVectorizer(sublinear_tf=True)
        X = vectorizer.fit_transform(series)
        tfidf_scores = X.max(0).toarray().ravel()  # Can also use max()
        words = vectorizer.get_feature_names_out()

        df = pd.DataFrame({"word": words, "tfidf": tfidf_scores})
        if top_n is not None:
            vocab = (
                df.sort_values(by="tfidf", ascending=False).head(top_n)["word"].tolist()
            )
        else:
            vocab = df["word"].tolist()
        cooc = defaultdict(int)
        for text in series:
            tokens = [t for t in text.split() if t in vocab]
            for a, b in combinations(set(tokens), 2):
                cooc[tuple(sorted((a, b)))] += 1
        G = nx.Graph()
        for (a, b), w in cooc.items():
            G.add_edge(a, b, weight=w)
        return G

    G_eth = None
    if any(df["corpus"] == "Ethics"):
        G_eth = build_cooccurrence_graph(
            df[df.corpus == "Ethics"]["clean_text"],
            top_n=1000,
        )
        console.print(
        f"Ethics graph nodes: {G_eth.number_of_nodes()} edges: {G_eth.number_of_edges()}"
        )

    G_saf = None
    if any(df["corpus"] == "Safety"):
        G_saf = build_cooccurrence_graph(
            df[df.corpus == "Safety"]["clean_text"],
            top_n=1000,
        )
        console.print(
            f"Safety graph nodes: {G_saf.number_of_nodes()} edges: {G_saf.number_of_edges()}"
        )

    if plot:
        if G_eth is not None:
            plot_top_subgraph(
                G_eth,
                top_k=50,
                title="Ethics co-word subgraph (top tf-idf nodes)",
                topic_model=topic_model,
                output_dir=output_dir,
            )
        if G_saf is not None:
            plot_top_subgraph(
                G_saf,
                top_k=50,
                title="Safety co-word subgraph (top tf-idf nodes)",
                topic_model=topic_model,
                output_dir=output_dir,
            )


def plot_topic_analysis(
    df: pd.DataFrame,
    topic_model: BERTopic,
    embeddings: np.ndarray,
    embedding_model: SentenceTransformer,
    output_dir: Path = Path("output"),
) -> None:
    """Plot UMAP projections and temporal semantic drift analysis."""
    fig = topic_model.visualize_topics(use_ctfidf=True)
    fig.write_html(output_dir / "bertopic_topics.html")

    fig = topic_model.visualize_heatmap(use_ctfidf=True)
    fig.write_html(output_dir / "bertopic_heatmap.html")

    topics_per_class = topic_model.topics_per_class(
        df["clean_text"], classes=df["corpus"]
    )
    fig = topic_model.visualize_topics_per_class(topics_per_class)
    fig.write_html(output_dir / "bertopic_topics_per_class.html")

    fig = topic_model.visualize_barchart(top_n_topics=12)
    fig.write_html(output_dir / "bertopic_barchart.html")

    # UMAP projection of embeddings and corpus scatter
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=RANDOM_SEED,
    )
    umap_emb = reducer.fit_transform(embeddings)
    df["umap_x"] = umap_emb[:, 0]
    df["umap_y"] = umap_emb[:, 1]

    plt.figure(figsize=(11, 7))

    # Separate outlier and non-outlier documents
    df_main = df[df["topic"] != -1]
    df_outliers = df[df["topic"] == -1]

    # Plot non-outlier documents (colored by topic, styled by corpus)
    sns.scatterplot(
        data=df_main,
        x="umap_x",
        y="umap_y",
        hue="topic",
        style="corpus",
        palette="tab20",
        s=25,
        alpha=0.7,
        legend=False,
    )

    # Plot outliers with very low opacity (gray, faint)
    plt.scatter(
        df_outliers["umap_x"],
        df_outliers["umap_y"],
        color="gray",
        alpha=0.1,
        s=10,
        label="Outliers (-1)",
    )

    # Compute topic centroids in UMAP space
    topic_centroids = df.groupby("topic")[["umap_x", "umap_y"]].mean().reset_index()

    # Annotate each topic with its label
    for _, row in topic_centroids.iterrows():
        topic_label = topic_model.get_topic(row["topic"])
        if topic_label:
            label_text = "_".join(label[0] for label in topic_label[:3])
            plt.text(
                row["umap_x"],
                row["umap_y"],
                label_text,
                fontsize=9,
                fontweight="bold",
                color="black",
                alpha=0.8,
                ha="center",
            )

    plt.title("UMAP projection: documents by topic and corpus (with topic labels)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.savefig(output_dir / "umap_topics.png", dpi=300)
    plt.show()

    # Temporal semantic drift: average embedding by year &
    # cosine similarity between corpora per year
    if "Ethics" in df["corpus"] and "Safety" in df["corpus"]:
        min_docs_per_year = 2
        if "year" in df.columns:
            years = sorted(df["year"].dropna().unique())
            year_sims = []
            for y in years:
                sub = df[df["year"] == y]
                if (
                    len(sub[sub.corpus == "Ethics"]) < min_docs_per_year
                    or len(sub[sub.corpus == "Safety"]) < min_docs_per_year
                ):
                    year_sims.append(np.nan)
                    continue
                e_vec = np.mean(
                    embedding_model.encode(
                        sub[sub.corpus == "Ethics"]["clean_text"].tolist(),
                    ),
                    axis=0,
                )
                s_vec = np.mean(
                    embedding_model.encode(
                        sub[sub.corpus == "Safety"]["clean_text"].tolist(),
                    ),
                    axis=0,
                )
                sim = cosine_similarity([e_vec], [s_vec])[0][0]
                year_sims.append(sim)
            plt.figure(figsize=(10, 4))
            plt.plot(years, year_sims, marker="o")
            plt.xlabel("Year")
            plt.ylabel("Cosine similarity (Ethics vs Safety)")
            plt.title("Temporal semantic convergence")
            plt.savefig(output_dir / "temporal_semantic_drift.png", dpi=300)
            plt.show()
        else:
            console.print("No year column found; skipping temporal drift analysis")

        # Corpus-level centroid similarity
        ethics_vec = np.mean(
            [
                emb
                for emb, c in zip(embeddings, df["corpus"], strict=True)
                if c == "Ethics"
            ],
            axis=0,
        )
        safety_vec = np.mean(
            [
                emb
                for emb, c in zip(embeddings, df["corpus"], strict=True)
                if c == "Safety"
            ],
            axis=0,
        )
        corp_sim = cosine_similarity([ethics_vec], [safety_vec])[0][0]
        console.print(f"Corpus-level cosine similarity: {corp_sim}")


def plot_top_subgraph(
    G: nx.Graph,
    top_k: int = 30,
    title: str = "Co-word subgraph",
    topic_model: BERTopic = None,
    output_dir: Path = Path("output"),
) -> None:
    """Visualize a subgraph of the co-word network (top-degree nodes)."""
    deg = dict(G.degree(weight="weight"))
    top_nodes = sorted(deg.items(), key=lambda x: x[1], reverse=True)[:top_k]
    nodes = [n for n, _ in top_nodes]
    H = G.subgraph(nodes)

    # --- Map words to topics if topic model provided ---
    if topic_model is not None:
        topic_words = topic_model.get_topics()
        word_to_topic = {}
        for topic_id, words in topic_words.items():
            for w, _ in words:
                word_to_topic[w] = topic_id

        topics = [
            word_to_topic.get(n, -1) for n in H.nodes()
        ]  # -1 if word not in any topic
        unique_topics = sorted({t for t in topics if t != -1})
        color_map = plt.get_cmap("tab10", len(unique_topics))
        node_colors = [
            color_map(unique_topics.index(t))
            if t in unique_topics
            else (0.8, 0.8, 0.8, 0.5)
            for t in topics
        ]
    else:
        node_colors = "lightblue"

    # --- Plot ---
    plt.figure(figsize=(10, 8))
    pos = graphviz_layout(H, prog="dot")
    nx.draw_networkx_nodes(
        H,
        pos,
        node_size=[deg[n] for n in H.nodes()],
        node_color=node_colors,
    )
    weights = np.array([H[u][v]["weight"] for u, v in H.edges()])
    nx.draw_networkx_edges(
        H,
        pos,
        width=np.power(weights / weights.max(), 1.2),
    )
    nx.draw_networkx_labels(H, pos, font_size=11, font_weight="bold")
    plt.title(title)
    plt.axis("off")

    # Optional legend for topics
    if topic_model is not None:
        for topic_id in unique_topics:
            topic_label = topic_model.get_topic(topic_id)
            label_text = "_".join(label[0] for label in topic_label[:2])
            plt.scatter(
                [],
                [],
                color=color_map(unique_topics.index(topic_id)),
                label=label_text,
            )
        plt.legend(loc="best", frameon=False)

    plt.savefig(output_dir / "coword_graph.png", dpi=300)
    plt.show()


@app.command()
def analyze(
    file_path: Path = typer.Option(
        Path("data/papers.csv"),
        "-p",
        "--path",
        help="Path to the CSV file containing paper metadata.",
    ),
    output_path: Path = typer.Option(
        Path("output/"),
        "-o",
        "--output",
        help="Directory to save analysis results and visualizations.",
    ),
    force: bool = typer.Option(
        False,
        "-f",
        "--force",
        help="Whether to force re-preprocessing of data even if cached.",
    ),
    corpus: str | None = typer.Option(
        None,
        "-c",
        "--corpus",
        help=(
            "Corpus to perform topic modeling on ('ethics', 'safety'). "
            "If not given, then both corpora are used."
        ),
    ),
    min_cluster_size: int = typer.Option(
        default=10,
        help="Minimum cluster size for HDBSCAN in topic modeling.",
    ),
    topic_modeling: bool = typer.Option(
        default=True,
        help="Whether to perform topic modeling analysis.",
    ),
    network_analysis: bool = typer.Option(
        default=True,
        help="Whether to perform term co-occurrence network analysis.",
    ),
    plot: bool = typer.Option(
        default=True,
        help="Whether to plot visualizations.",
    ),
) -> int:
    """AI safety and ethics paper analysis tool.

    This script reads a CSV file containing paper metadata, preprocesses the text data,
    performs topic modeling using BERTopic, and analyzes term co-occurrence networks.
    It also includes visualization of topic distributions and network graphs.

    The input CSV file is expected to have at least the following columns:
    - key: Unique identifier for each paper.
    - title: The title of the paper.
    - abstract: The abstract of the paper.
    - corpus: The corpus/category the paper belongs to ('Ethics', 'Safety').
    - year: The publication year of the paper.
    - authors: The authors of the paper (optional).

    All preprocessed data and visualizations are saved to the specified output directory.
    """
    if corpus is not None and corpus not in {"ethics", "safety"}:
        console.print(
            "Invalid topic_corpus. Choose from 'ethics', 'safety', or None.",
        )
        raise typer.Exit(1)

    # Read data
    output_path.mkdir(parents=True, exist_ok=True)
    output_df_file = output_path / "processed_data.csv"
    if output_df_file.exists() and not force:
        console.print("Using existing processed data.")
        df = pd.read_csv(output_df_file)
    else:
        console.print("Reading and preprocessing data...")
        df = read_data(file_path)
        df = preprocess_data(df)
        df.to_csv(output_df_file, index=True)

    if corpus is not None:
        corpus = corpus.lower().capitalize()
        df = df[df["corpus"] == corpus]

    console.print(f"Using only corpus: {corpus}" if corpus else "Using both corpora")

    get_statistics(df)

    # Topic modeling with BERTopic
    topic_model = None
    if topic_modeling:
        topic_model = model_topic(
            df,
            min_cluster_size=min_cluster_size,
            plot=plot,
            output_dir=output_path,
        )

    # Co-word network: top-N vocabulary co-occurrence
    if network_analysis:
        analyze_network(df, plot=plot, topic_model=topic_model, output_dir=output_path)

    raise typer.Exit(0)


def cli() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    app()
