import csv
import enum
import pickle
import re
from pathlib import Path

import pandas as pd
import typer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tqdm import tqdm

app = typer.Typer()


def remove_urls(text: str) -> str:
    # Matches http/https URLs and www URLs
    url_pattern = r"http\S+|www\.\S+"
    return re.sub(url_pattern, "", text)


class PreprocessType(enum.Enum):
    VOSViewer = "vosviewer"
    Keywords = "keywords"


class RetrievalCategory(enum.Enum):
    Safety = "safety"
    Ethics = "ethics"
    Both = "both"


@app.command()
def preprocess(
    output: PreprocessType = typer.Option(
        PreprocessType.VOSViewer,
        "--type",
        "-t",
        help="The type of preprocessing to perform.",
    ),
    retrieval_category: RetrievalCategory = typer.Option(
        "both",
        "--retrieval-cat",
        "-r",
        help="The retrieval category to filter by.",
    ),
    input_file: str = typer.Option(
        "data/papers.csv", "--input-path", "-i", help="Path to the input data file."
    ),
    output_dir: str = typer.Option(
        "output/", "--output-path", "-o", help="Path to the output data directory."
    ),
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    """Process the data corpus for the given application type."""
    if output == PreprocessType.VOSViewer:
        df = pd.read_csv(input_file, index_col=0)
        if retrieval_category != RetrievalCategory.Both:
            df = df[df["Retrieval"] == retrieval_category.value.capitalize()]
        text = df["Title"].str.cat(df["Abstract Note"], sep=" ")
        text = text.apply(remove_urls)
        text.to_csv(
            output_dir + retrieval_category.value + ".txt",
            index=False,
            header=False,
            quoting=csv.QUOTE_NONE,
            sep="\t",
        )

        # Write binary score of which category each paper belongs to.
        if retrieval_category == RetrievalCategory.Both:
            score = df["Retrieval"] == "Safety"  # 1: Safety, 0: Ethics
            score.astype(int).to_csv(
                output_dir + "scores.txt",
                index=False,
                header=False,
                quoting=csv.QUOTE_NONE,
                sep="\t",
            )

    elif output == PreprocessType.Keywords:
        # Load JSON input into dataframe. Expecting a list/dict with a 'keywords' field per record
        # Try to load as JSON file first
        df = pd.read_json(input_file, orient="index")
        df = pd.read_csv("data/papers.csv", index_col=0).join(df, how="inner")

        # Expect a column named 'keywords' (case-insensitive fallback)
        keywords_col = None
        for col in df.columns:
            if col.lower() == "keywords":
                keywords_col = col
                break
        if keywords_col is None:
            raise ValueError("No 'keywords' column found in input JSON")

        # Build stopword set
        STOPWORDS = set(stopwords.words("english"))

        # Initialize stemmer
        stemmer = PorterStemmer()

        def stem_token(t: str) -> str:
            return stemmer.stem(t)

        def clean_keyword_phrase(phrases: list) -> list[str]:
            if not isinstance(phrases, list):
                return []
            tokens_out: list[str] = []
            for phrase in phrases:
                # remove text in parentheses
                phrase = re.sub(r"\([^)]*\)", "", phrase)
                part = phrase.strip().lower()
                if not part:
                    continue
                # tokenize on whitespace and punctuation
                toks = re.findall(r"[a-zA-Z0-9]+(?:'[a-z]+)?", part)
                toks = [t for t in toks if t not in STOPWORDS]
                if toks:
                    tokens_out.append(" ".join(toks))
            return tokens_out

        # Apply cleaning to the keywords column and write out one keyword phrase per line
        df[keywords_col] = df[keywords_col].apply(clean_keyword_phrase)

        # Flatten and write to output file
        df = df.loc[~df.index.duplicated(keep="first"), :]
        df[[keywords_col, "Retrieval"]].dropna(how="all", axis=1).to_json(
            output_dir + "keywords.json", orient="index"
        )


@app.command()
def compare(
    input_file: str = typer.Option(
        "output/keywords.json",
        "--input-path",
        "-i",
        help="Path to the input keywords file.",
    ),
    min_common: int = typer.Option(
        2, "--min-common", "-m", help="Minimum number of common keywords to report."
    ),
    output_dir: str = typer.Option(
        "output/", "--output-path", "-o", help="Directory of outputs."
    ),
):
    df = pd.read_json(input_file, orient="index")
    papers = pd.read_csv("data/papers.csv", index_col=0)

    ethics = df[df["Retrieval"] == "Ethics"]
    safety = df[df["Retrieval"] == "Safety"]
    if (Path(output_dir) / "shared_keywords.pkl").exists():
        shared_keywords = pickle.load(
            (Path(output_dir) / "shared_keywords.pkl").open("rb")
        )
    else:
        shared_keywords = {}
        for idx, erow in tqdm(ethics.iterrows(), total=ethics.shape[0]):
            for idx2, srow in safety.iterrows():
                common = set(erow["keywords"]).intersection(set(srow["keywords"]))
                if common:
                    shared_keywords[(idx, idx2)] = (common, len(common))
        pickle.dump(
            shared_keywords,
            (Path(output_dir) / "shared_keywords.pkl").open("wb"),
        )

    # Prepare rows for output
    rows = []
    for (idx, idx2), (keywords, count) in sorted(
        filter(lambda x: x[1][1] >= min_common, shared_keywords.items()),
        key=lambda x: x[1],
        reverse=True,
    ):
        rows.append(
            {
                "ethics_id": idx,
                "safety_id": idx2,
                "ethics_title": papers.loc[idx, "Title"],
                "safety_title": papers.loc[idx2, "Title"],
                "shared_keywords": ", ".join(sorted(keywords)),
                "count": count,
            }
        )

    from rich.table import Table
    from rich.console import Console

    console = Console()
    table = Table(title="Shared keywords across Ethics and Safety")
    table.add_column("#", style="cyan", justify="right")
    table.add_column("Ethics ID")
    table.add_column("Safety ID")
    table.add_column("Ethics title")
    table.add_column("Safety title")
    table.add_column("Shared keywords")
    table.add_column("Count", justify="right")

    for i, r in enumerate(rows, start=1):
        table.add_row(
            str(i),
            str(r["ethics_id"]),
            str(r["safety_id"]),
            r["ethics_title"],
            r["safety_title"],
            r["shared_keywords"],
            str(r["count"]),
        )

    console.print(table)


if __name__ == "__main__":
    app()
