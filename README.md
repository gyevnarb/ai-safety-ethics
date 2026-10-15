# Bridging the Gap in the Responsible AI Divides
Bridging research problems of the fields AI safety and AI ethics.

Supplementary materials for the titular paper are found in [supplementary.pdf](supplementary.pdf).
Supporting figures are found in [output.zip](output.zip).

The software package name is shortened to `aise` (short for AI Safety and Ethics).

## Installation

The project uses the Python project manager [uv](https://docs.astral.sh/uv/getting-started/installation/).

1. Clone the repo: `git clone https://github.com/gyevnarb/ai-safety-ethics.git`
2. Install dependencies from the repository root: `uv sync`
3. Install the spaCy model used for preprocessing: `uv pip install "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"` (`uv sync` removes it again, so rerun this after syncing)

You will also need:
- [Graphviz](https://www.graphviz.org/download/) for the co-word network plots;
- `pdftotext` ([Poppler](https://poppler.freedesktop.org/)) to extract the papers' full text;
- `OPENAI_API_KEY` in the environment for the LLM classification and coding stages.

spaCy and [BERTopic](https://maartengr.github.io/BERTopic/index.html) can take more than 1 GiB of storage.

## Usage

Run every command from the repository root. All stages are available through a single CLI:

```bash
uv run aise --help
```

| Command | Purpose |
| --- | --- |
| `aise analyze` | Preprocess papers, topic modelling and co-word networks |
| `aise plot` | Word and category figures comparing the two corpora |
| `aise engagement` | Classify papers into the four engagement modes (OpenAI Batch API) |
| `aise engagement-plots` | Figures of the classification |
| `aise qualitative` | Qualitative coding and thematic analysis of the rare-mode papers |
| `aise qualitative-plots` | Figures and LaTeX table of the qualitative analysis |

Add `--help` to any command for its subcommands and options. Each group can also be run as a module, e.g. `uv run python -m aise.engagement`.

A typical end-to-end run:

```bash
uv run aise engagement extract            # PDFs in pdf/ -> text in data/txt/
uv run aise engagement prepare            # write batch requests
uv run aise engagement submit
uv run aise engagement fetch              # repeat until all batches complete
uv run aise engagement parse
uv run aise engagement-plots figures

uv run aise qualitative code --stage 1    # then: code --stage 2, freeze, themes,
                                          # spans, representatives, report
uv run aise qualitative-plots figures
```

### Inputs, outputs and runs

Inputs are read from `data/` and results are written to `output/`; the paths are defined in [`src/aise/config.py`](src/aise/config.py). Human corrections of the engagement classification are kept in `data/engagement_human_labels.csv`; `aise engagement parse` applies them when the file exists.

`data/full_data.csv` combines everything per paper: the risk and mitigation annotations of `data/annotated_papers.csv` plus its engagement classification (levels, mode, rationale and evidence). Regenerate it with `aise engagement-plots merge`.

Each LLM run gets its own folder, selected with environment variables:

- `AISE_PROMPT_VERSION` (default `v1`): engagement classification, in `output/engagement/<version>/`;
- `AISE_QUAL_VERSION` (default `v1`): qualitative coding, in `output/qualitative/<version>/`;
- `AISE_QUAL_TEST_N`: if set, codes only this many papers per mode (for testing the pipeline).
