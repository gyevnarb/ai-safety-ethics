# Bridging the Gap in the Responsible AI Divides
Bridging research problems of the fields AI safety and AI ethics.

The package name is shortened to `aise` (short for AI Safety and Ethics).

# Installation

The setup process uses the super-fast and lightweight Python project manager `uv`, although you can use other solutions too (e.g. pip).

- Make sure that `uv` is installed from [here](https://docs.astral.sh/uv/getting-started/installation/).
- For plotting graphs, you will also need to install [Graphviz](https://www.graphviz.org/download/) separately.
- This codebase uses [Spacy](https://spacy.io/) and [BERTopic](https://maartengr.github.io/BERTopic/index.html) for preprocessing and topic analysis. The prerequisities for these can take more than 1 GiB of storage.

### Follow the steps below to get started with the project:
1. Clone the repo using `git clone https://github.com/gyevnarb/ai-safety-ethics.git` (or using an SSH key).

2. Install dependencies with `uv` by running: `uv sync` from the root folder of the repository, you have just cloned.

Here is a short script to run everything above:
```bash
# Run all conditions in interventions.json for DATA_ID 6jmfx with k=7 iterations per condition,
# each iteration running sequentially (-s), the Docker image rebuilt (--build), using Codex (-a),
# in dry-run mode (-n).
bash batch_run.sh -n -s --build 6jmfx -a codex -k 7

# Run all dataset experiments for all conditions in interventions.json with k=3 iterations
# per condition, in sequential mode.
bash all_experiments.sh -s claude -- -k 3
```

### 4. Evaluate the experiments

Perform the following steps to run the LLM-as-a-judge evaluation setup:

1. Run `bash copy_results.sh [negative|positive]`.
    - The positional argument corresponds to the adversary goal you selected earlier.
    - This script creates a new folder, `results`, and copies data from the `workspace` and `logs` folders into it.
2. For each DATA_ID, separately run `bash evaluate_batch.sh <DATA_ID> <EVAL_CONFIG_PATH>` to evaluate the experiments for that DATA_ID. The `EVAL_CONFIG_PATH` argument refers to the evaluation schema:
    - For the reject (negative) adversary goal, use `eval_schema_negative.json`.
    - For the exaggerate (positive) adversary goal, use `eval_schema_positive.json`.
3. Collect the results into a CSV by running:
    ```bash
    python3 annotation_app/export_eval_csv.py -o full.csv
    ```
    This produces a CSV file called `full.csv` containing all evaluation results in tabular format.

### 5. Plot the experiments

You can use the scripts in the `analysis` folder to reproduce all figures from the paper using your experimental results.
The script currently reproduces figures for our experimental data.
To point it at your own data file, replace the following line in `plots.R` (line 39):

```r
in_path <- here("..", "results", "full.csv") # <-- Replace this to point to your path
```

---

## Dataset Wrapper

This repository includes a small helper package that wraps several sources for dataset loading.
The currently supported providers are GitHub, Open Science Framework (OSF), Kaggle, HuggingFace (HF), and OpenML (not used in the experiments; hidden from the AI).

### Usage

To use the package locally, I recommend [uv](https://docs.astral.sh/uv/getting-started/installation/).
Run the following command from the repo root to start using the package:

```bash
uv run aise --help
```

The `--help` option will output a helpful description of how to use the command, and list all possible command line options that can be passed to the CLI.
