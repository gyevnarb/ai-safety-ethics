"""The look shared by the figures: colours of the fields and modes, ink, and saving.

Importing this module embeds fonts in PDFs as TrueType (Type 42), which journals require.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import typer
from matplotlib import rcParams

from aise.engagement import CATEGORIES

rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

# text and lines, darkest first, and the page
INK, INK_2, INK_3, SURFACE = "#0b0b0b", "#52514e", "#8a8983", "#ffffff"
GRID = "#e6e5e1"

# the paper's corpora (the "Retrieval" column)
FIELD_COLORS = {"Ethics": "#1f77b4", "Safety": "#d62728"}
FIELD_NAMES = {"Ethics": "AI ethics", "Safety": "AI safety"}
FIELD_ABBREV = {"Ethics": "AIE", "Safety": "AIS"}

# the four modes of the paper's Figure 1; all but Disengagement are rare
MODES = CATEGORIES
RARE_MODES = MODES[1:]
# The colours of Figure 1 (TikZ): marks use each mode box's outline, draw=<c>!70!black,
# and backgrounds its fill, fill=<c>!12, with c = gray, blue, orange and green!60!black.
MODE_COLORS = dict(zip(MODES, ["#595959", "#0000b3", "#b35900", "#006b00"], strict=True))
MODE_FILLS = dict(zip(MODES, ["#f0f0f0", "#e0e0ff", "#fff0e0", "#e0f3e0"], strict=True))


def save_figure(fig, name: str, out_dir: Path):
    """Save the figure as out_dir/<name>.pdf and .png, cropped to its contents."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(out_dir / f"{name}.png", dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    typer.secho(f"  wrote {out_dir / name}.pdf")
