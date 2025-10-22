# AI Safety and Ethics
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
git clone https://github.com/gyevnarb/ai-safety-ethics.git
cd ai-safety-ethics
uv sync
```

# Usage

The pacakage comes with a command line interface (CLI) that exposes a single command `aise`, which can be called with the following command once installation is done:

```bash
uv run aise --help
```

The `--help` option will output a helpful description of how to use the command, and list all possible command line options that can be passed to the CLI.