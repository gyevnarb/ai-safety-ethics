import json
import logging
import pandas as pd
import time
import typer
from enum import Enum
from openai import OpenAI
from pathlib import Path
from rich.logging import RichHandler


class Domain(str, Enum):
    base = "base"
    safety = "safety"
    ethics = "ethics"


app = typer.Typer()

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
# Get current time for log file naming
current_time = time.strftime("%Y%m%d-%H%M%S")
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(f"main_{current_time}.log", mode="a")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
logger.addHandler(file_handler)
logging.getLogger("httpx").setLevel(logging.WARNING)

SYSTEM_PROMPTS = {
    "base": "You are an expert researcher in AI.",
    "safety": "You are an expert researcher in AI safety.",
    "ethics": "You are an expert researcher in AI ethics.",
}

SUMMARIZE_PROMPT = """You are given a scientific paper with the following content.
TITLE: {title}
ABSTRACT: {abstract}

Extract the 10 most important and relevant key phrases from the paper. Give your answer as a list.
Extract the research question addressed in the paper. Answer in one concise sentence.

Give your answer in a JSON object with the following keys: key_phrases, research_question."""


@app.command()
def process(
    n_papers: int = typer.Option(
        1,
        "-n",
        "--num-papers",
        help="Number of papers to process. If -1, process all papers.",
    ),
    domain: Domain = typer.Option(
        Domain.base, "-d", "--domain", help="System prompt to use."
    ),
    try_count: int = typer.Option(
        3, "-t", "--try-count", help="Number of retries for JSON parsing."
    ),
):
    """Process papers to extract key phrases and research questions."""
    logger.info("Starting the paper processing script.")
    logger.info(
        f"Domain: {domain}, Number of papers: {n_papers}, Try count: {try_count}"
    )

    # Setup API client
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="")
    models = client.models.list()
    model_ids = [model.id for model in models.data]

    # Load papers.csv into a dataframe
    papers_df = pd.read_csv("papers.csv")

    # Process the first n_papers
    result = (
        json.load(open("results.json", "r")) if Path("results.json").exists() else {}
    )
    for i in range(n_papers) if n_papers > 0 else range(len(papers_df)):
        paper = papers_df.iloc[i]
        title = paper["Title"]
        abstract = paper["Abstract"]
        key = paper["Key"]
        logger.info(f"{i + 1}/{len(papers_df)}. Title: {title}")
        if key in result:
            logger.info("    Already processed. Skipping.")
            continue

        messages = [
            {"role": "system", "content": SYSTEM_PROMPTS[domain]},
            {
                "role": "user",
                "content": SUMMARIZE_PROMPT.format(title=title, abstract=abstract),
            },
        ]
        for t in range(try_count):
            try:
                completion = client.chat.completions.create(
                    model=model_ids[0],
                    messages=messages,
                    temperature=0.7,
                    top_p=0.8,
                    max_tokens=16384,
                )
                content = json.loads(completion.choices[0].message.content)
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON.")
                messages.append(
                    {
                        "role": "assistant",
                        "content": completion.choices[0].message.content,
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": "The previous response was not a valid JSON. Please try again.",
                    }
                )
                continue
            break
        else:
            logger.error(
                "Exceeded maximum retries for JSON parsing for: %s(%s).",
                paper["Key"],
                title,
            )
            continue

        logger.info("    Example phrases: %s", content["key_phrases"][:3])
        result[paper["Key"]] = content
        with open("results.json", "w") as f:
            json.dump(result, f, indent=2)


@app.command()
def analyze():
    """Analyze the results to find patterns and common key phrases."""
    with open("results.json", "r") as f:
        results = json.load(f)

    phrase_count = {}
    for paper_key, content in results.items():
        for phrase in content["key_phrases"]:
            phrase_count[phrase] = phrase_count.get(phrase, 0) + 1

    sorted_phrases = sorted(phrase_count.items(), key=lambda x: x[1], reverse=True)
    logger.info("Most common key phrases:")
    for phrase, count in sorted_phrases[:20]:
        logger.info(f"{phrase}: {count}")


if __name__ == "__main__":
    app()
