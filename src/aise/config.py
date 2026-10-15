"""Where the inputs and outputs live, and which run each command reads and writes.

Paths are relative to the repository root, so commands are run from there. Two
environment variables select earlier runs:

- ``AISE_PROMPT_VERSION`` (default v1): the engagement classification run, in
  output/engagement/<version>/;
- ``AISE_QUAL_VERSION`` (default v1): the qualitative coding run, in
  output/qualitative/<version>/; with ``AISE_QUAL_TEST_N`` > 0, it codes only that
  many papers per mode (for testing the pipeline end to end).
"""

import os
from pathlib import Path

# ---- inputs (only read, except data/txt, which ``aise engagement extract`` fills) ----
DATA_DIR = Path("data")
PAPERS = DATA_DIR / "papers_all_fields.csv"  # metadata of all retrieved papers
ANNOTATIONS = DATA_DIR / "annotated_papers.csv"  # risk and mitigation annotations
CATEGORIES = DATA_DIR / "categories.csv"  # the risk and mitigation taxonomies
# human corrections of the engagement classification (levels on the 1 to 4 scale)
HUMAN_LABELS = DATA_DIR / "engagement_human_labels.csv"
# annotations joined with the classification (``aise engagement-plots merge``)
FULL_DATA = DATA_DIR / "full_data.csv"
TXT_DIR = DATA_DIR / "txt"  # full text extracted from the PDFs, one file per paper
PDF_DIR = Path("pdf")

# ---- outputs ---------------------------------------------------------------------------
OUTPUT_DIR = Path("output")
PROCESSED_DATA = OUTPUT_DIR / "processed_data.csv"  # written by ``aise analyze``

PROMPT_VERSION = os.environ.get("AISE_PROMPT_VERSION", "v1")
ENGAGEMENT_DIR = OUTPUT_DIR / "engagement" / PROMPT_VERSION
ENGAGEMENT_PLOTS_DIR = OUTPUT_DIR / "plots" / "modes"

QUAL_VERSION = os.environ.get("AISE_QUAL_VERSION", "v1")
QUAL_TEST_N = int(os.environ.get("AISE_QUAL_TEST_N", "0"))
QUALITATIVE_DIR = OUTPUT_DIR / "qualitative" / QUAL_VERSION
