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
import typer
import umap
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = typer.Typer()

RANDOM_SEED = 42


def read_data(file_path: Path) -> pd.DataFrame:
    """Read the CSV data of paper metadata and ensure required columns exist."""
    # Parameters - update paths if needed
    df = pd.read_csv(file_path, index_col=0)
    typer.secho(f"Total rows: {len(df)}")

    # Ensure required columns exist
    for col in ["title", "abstract", "year", "authors"]:
        if col not in df.columns:
            typer.secho(
                f'WARNING: column "{col}" not found. Consider adding it.',  # noqa: Q003
                fg=typer.colors.YELLOW,
            )
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the text data in the dataframe.

    Includes tokenization, lemmatization, and stopword removal.
    """
    import spacy

    nltk.download("stopwords")

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    stop_words = set(nltk.corpus.stopwords.words("english"))

    def preprocess(text: str) -> str:
        if pd.isna(text):
            return ""
        text = re.sub(r"http\S+|www\S+|doi\.org\S+", "", str(text))
        text = text.lower()

        doc = nlp(text)
        tokens = [
            t.lemma_
            for t in doc
            if t.is_alpha and t.lemma_ not in stop_words and len(t.lemma_) > 1
        ]
        return " ".join(tokens)

    # Apply preprocessing to abstracts (progress-safe)
    typer.echo("Preprocessing text data...")
    if "clean_text" not in df.columns:
        full_text = (
            df["title"].fillna("").str.cat(df["abstract"].fillna(""), sep=" <> ")
        )
        df["clean_text"] = full_text.map(preprocess)

    typer.echo("Sample cleaned text:")
    typer.echo(df["clean_text"].iloc[0][:300])

    return df


def model_topic(
    df: pd.DataFrame, min_cluster_size: int = 10, plot: bool = True,
) -> None:
    """Perform topic modeling using BERTopic and visualize results."""
    # Initialize embedding model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Small optimization: if corpus is large, sample for initial modeling
    texts = df["clean_text"].tolist()

    typer.echo("Computing embeddings (this may take time)")
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
    topics, _ = topic_model.fit_transform(texts, embeddings)
    df["topic"] = topics

    # Topic summary
    topic_info = topic_model.get_topic_info()
    typer.echo(f"Total topics found: {len(topic_info) - 1}")  # exclude -1 (outliers)
    typer.echo(topic_info.head(15))

    # Compare topic distributions across corpora
    topic_counts = df.pivot_table(
        index="corpus",
        columns="topic",
        aggfunc="size",
        fill_value=0,
    )
    # ensure both indexes present
    if "Ethics" in topic_counts.index and "Safety" in topic_counts.index:
        topic_counts["ethics_prop"] = (
            topic_counts.loc["Ethics"] / topic_counts.loc["Ethics"].sum()
        )
        topic_counts["safety_prop"] = (
            topic_counts.loc["Safety"] / topic_counts.loc["Safety"].sum()
        )
        topic_counts = topic_counts[["ethics_prop", "safety_prop"]].fillna(0)
        diff = (
            (topic_counts["ethics_prop"] - topic_counts["safety_prop"])
            .abs()
            .sort_values(ascending=False)
        )
        typer.echo("Top differing topics (by absolute prop diff):")
        typer.echo(diff.head(15))
    else:
        typer.echo("Topic counts could not be computed: check topics and corpus labels")

    # Visualize top 10 topics by combined frequency (BERTopic visualization)
    try:
        topic_model.visualize_topics(top_n_topics=10)
    except Exception as e:
        typer.echo(f"Visualization failed (reason): {e}")

    if plot:
        plot_topic_analysis(df, topic_model, embeddings, embedding_model)


def plot_topic_analysis(
    df: pd.DataFrame,
    topic_model: BERTopic,
    embeddings: np.ndarray,
    embedding_model: SentenceTransformer,
) -> None:
    """Plot UMAP projections and temporal semantic drift analysis."""
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
    plt.show()

    # Temporal semantic drift: average embedding by year &
    # cosine similarity between corpora per year
    if "year" in df.columns:
        years = sorted(df["year"].dropna().unique())
        year_sims = []
        for y in years:
            sub = df[df["year"] == y]
            if (
                len(sub[sub.corpus == "Ethics"]) < 2
                or len(sub[sub.corpus == "Safety"]) < 2
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
        plt.show()
    else:
        typer.echo("No year column found; skipping temporal drift analysis")

    # Corpus-level centroid similarity
    ethics_vec = np.mean(
        [emb for emb, c in zip(embeddings, df["corpus"], strict=True) if c == "Ethics"],
        axis=0,
    )
    safety_vec = np.mean(
        [emb for emb, c in zip(embeddings, df["corpus"], strict=True) if c == "Safety"],
        axis=0,
    )
    corp_sim = cosine_similarity([ethics_vec], [safety_vec])[0][0]
    typer.echo(f"Corpus-level cosine similarity: {corp_sim}")


def analyze_network(df: pd.DataFrame, plot: bool = True) -> None:
    """Analyze term co-occurrence networks and visualize."""

    def build_cooccurrence_graph(series: pd.Series, top_n: int = 100) -> nx.Graph:
        words = " ".join(series).split()
        freq = Counter(words).most_common(top_n)
        vocab = {w for w, _ in freq}
        cooc = defaultdict(int)
        for text in series:
            tokens = [t for t in text.split() if t in vocab]
            for a, b in combinations(set(tokens), 2):
                cooc[tuple(sorted((a, b)))] += 1
        G = nx.Graph()
        for (a, b), w in cooc.items():
            G.add_edge(a, b, weight=w)
        return G

    G_eth = build_cooccurrence_graph(df[df.corpus == "Ethics"]["clean_text"], top_n=120)
    G_saf = build_cooccurrence_graph(df[df.corpus == "Safety"]["clean_text"], top_n=120)

    typer.echo(
        f"Ethics graph nodes: {G_eth.number_of_nodes()} edges: {G_eth.number_of_edges()}"
    )
    typer.echo(
        f"Safety graph nodes: {G_saf.number_of_nodes()} edges: {G_saf.number_of_edges()}"
    )

    # Visualize a subgraph of the co-word network (top-degree nodes)
    def plot_top_subgraph(
        G: nx.Graph,
        top_k: int = 30,
        title: str = "Co-word subgraph",
    ) -> None:
        deg = dict(G.degree(weight="weight"))
        top_nodes = sorted(deg.items(), key=lambda x: x[1], reverse=True)[:top_k]
        nodes = [n for n, _ in top_nodes]
        H = G.subgraph(nodes)
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(H, k=0.5, seed=42)
        nx.draw_networkx_nodes(H, pos, node_size=[deg[n] / 20 for n in H.nodes()])
        nx.draw_networkx_edges(H, pos, alpha=0.4)
        nx.draw_networkx_labels(H, pos, font_size=9)
        plt.title(title)
        plt.axis("off")
        plt.show()

    if plot:
        plot_top_subgraph(G_eth, top_k=30, title="Ethics co-word subgraph (top nodes)")
        plot_top_subgraph(G_saf, top_k=30, title="Safety co-word subgraph (top nodes)")


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
    topic_modeling: bool = typer.Option(
        default=True,
        help="Whether to perform topic modeling on the dataset.",
    ),
    min_cluster_size: int = typer.Option(
        default=10,
        help="Minimum cluster size for HDBSCAN in topic modeling.",
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
    """Analyze the paper metadata and perform topic modeling."""
    # Read data
    output_path.mkdir(parents=True, exist_ok=True)
    output_df_file = output_path / "processed_data.csv"
    if output_df_file.exists():
        df = pd.read_csv(output_df_file)
    else:
        df = read_data(file_path)
        df = preprocess_data(df)
        df.to_csv(output_df_file, index=False)

    # Basic descriptive statistics
    typer.echo(f"Years range: {df['year'].min()} - {df['year'].max()}")

    # Word counts and lexical richness per document
    _df_word_counts = df["clean_text"].str.split().apply(len)
    df["word_count"] = _df_word_counts
    typer.echo("\nWord count (median) per corpus:")
    typer.echo(df.groupby("corpus")["word_count"].median())

    # Keyness: log-likelihood ratio to find distinctive words
    def word_freqs(series: pd.Series) -> Counter:
        words = " ".join(series).split()
        return Counter(words)

    freq_ethics = word_freqs(df[df.corpus == "Ethics"]["clean_text"])
    freq_safety = word_freqs(df[df.corpus == "Safety"]["clean_text"])

    N1, N2 = sum(freq_ethics.values()), sum(freq_safety.values())
    all_words = set(list(freq_ethics.keys()) + list(freq_safety.keys()))
    results = []
    for w in all_words:
        O1, O2 = freq_ethics.get(w, 0), freq_safety.get(w, 0)
        E1 = N1 * (O1 + O2) / (N1 + N2) if (N1 + N2) > 0 else 0
        E2 = N2 * (O1 + O2) / (N1 + N2) if (N1 + N2) > 0 else 0
        LL = 2 * (
            (O1 * np.log((O1 / (E1 + 1e-9)) + 1e-9))
            + (O2 * np.log((O2 / (E2 + 1e-9)) + 1e-9))
        )
        results.append((w, LL, O1, O2))

    keyness_df = pd.DataFrame(
        results,
        columns=["word", "LL", "ethics_count", "safety_count"],
    )
    keyness_df = keyness_df.sort_values("LL", ascending=False).reset_index(drop=True)
    keyness_df.head(30)

    # Topic modeling with BERTopic
    if topic_modeling:
        model_topic(df, min_cluster_size=min_cluster_size, plot=plot)

    # Co-word network: top-N vocabulary co-occurrence
    if network_analysis:
        analyze_network(df, plot=plot)

    raise typer.Exit(0)


if __name__ == "__main__":
    app()
