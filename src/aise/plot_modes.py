"""Figures for the engagement/integration classification (see aise.engagement).

    uv run python -m aise.plot_modes join                 # classifications + annotations
    uv run python -m aise.plot_modes figures              # all figures
    uv run python -m aise.plot_modes figures --pilot      # from the pilot CSV
    uv run python -m aise.plot_modes figures --results a.jsonl --results b.jsonl

Figures (written to output/plots/modes/ as PDF and PNG):

1. fig_modes_grid        grid of engagement x integration levels, one panel per field
   fig_modes_grid_combined   the same grid with both fields pooled into one panel
2. fig_modes_shares      share of papers per mode by field, with a zoom on the rarer modes
   fig_modes_shares_rare     the zoom on its own, to sit beneath fig_modes_grid_combined
3. fig_modes_time        share of papers in each non-disengaged mode over time, by field
   fig_modes_time_combined   the same in one panel, both fields pooled
   fig_modes_time_combined_yearly   the same by single year (before 2019 pooled)
4. fig_modes_problems    which risk / mitigation categories are over-represented among
                         papers that integrate both fields' concerns
   fig_modes_problems_adjusted_integration   the same from one Firth logistic regression of all
                         categories with field, period and paper length as controls
   fig_modes_problems_adjusted_engagement   the same model for engaging papers (high
                         engagement: Radical confrontation or Critical bridging)
5. fig_modes_levels      distribution of engagement and integration levels by field
6. fig_modes_examples    one quoted example paper per mode, laid out like Figure 1
7. fig_modes_taxonomy    share of each paper's annotated categories that come from the
                         other field's taxonomy, by mode and field (a check on the
                         classification that does not depend on it)
8. fig_modes_reference   which papers refer to the other field, by direction (field),
                         and the modes of those that do
9. fig_modes_topics      share of each mode's papers in each high-level risk and
                         mitigation category, grouped by the taxonomy it comes from

The existing annotations in data/ are only read, never modified.
"""

import math
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from matplotlib import rcParams
from matplotlib.path import Path as MPath

from aise import engagement as E
from aise.firth import firth_logit, holm

rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

ANNOTATIONS = Path("data/annotated_papers.csv")
CATEGORIES = Path("data/categories.csv")
OUT = Path("output/plots/modes")

MODES = E.CATEGORIES  # Disengagement, Compartmentalized coexistence, Radical ..., Critical ...
RARE_MODES = MODES[1:]
# The colours of the paper's Figure 1 (TikZ): marks use each mode box's outline,
# draw=<c>!70!black, and backgrounds its fill, fill=<c>!12, with c = gray, blue,
# orange and green!60!black.
MODE_COLORS = dict(zip(MODES, ["#595959", "#0000b3", "#b35900", "#006b00"], strict=True))
MODE_FILLS = dict(zip(MODES, ["#f0f0f0", "#e0e0ff", "#fff0e0", "#e0f3e0"], strict=True))
# the fields keep the colours of the paper's existing figures (plot.py)
FIELD_COLORS = {"Ethics": "#1f77b4", "Safety": "#d62728"}
FIELD_NAMES = {"Ethics": "AI ethics", "Safety": "AI safety"}
FIELD_ABBREV = {"Ethics": "AIE", "Safety": "AIS"}
# single-hue sequential ramp for the ordinal levels (light -> dark); a 4-level scale
# skips the middle step so neighbouring levels stay well apart
LEVEL_RAMP = ["#b7d3f6", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"]
LEVEL_COLORS = dict(zip(E.LEVELS, LEVEL_RAMP if len(E.LEVELS) == 5
                        else [LEVEL_RAMP[i] for i in (0, 1, 3, 4)], strict=True))
INK, INK_2, INK_3, SURFACE = "#0b0b0b", "#52514e", "#8a8983", "#ffffff"
# boundaries between low and high levels, for drawing the quadrant lines
X_CUT, Y_CUT = E.ENGAGEMENT_HIGH - 0.5, E.INTEGRATION_HIGH - 0.5
TOP = max(E.LEVELS) + 0.5  # upper edge of the level grid
# On the 1-5 scale (v6) the grids draw the quadrant boundaries through the middle
# level 3, so nodes with a 3 sit on a boundary, split between the modes either side.
MID = 3 if E.FIVE_LEVELS else None
GRID_X, GRID_Y = (MID, MID) if MID else (X_CUT, Y_CUT)

app = typer.Typer(help=__doc__.split("\n\n")[0])


# ---- data ----------------------------------------------------------------------
def load_results(paths: list[Path]) -> pd.DataFrame:
    """Classified papers from parsed CSVs or raw result JSONL files, with metadata."""
    frames = []
    for path in paths:
        if path.suffix == ".jsonl":
            rows = [E.parse_line(line) for line in path.read_text().splitlines()]
            frames.append(pd.DataFrame(rows).set_index("key"))
        else:
            frames.append(pd.read_csv(path, index_col=0))
    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="last")]
    if "error" in df:
        df = df[df["error"].isna()]
    df["engagement"] = df["engagement"].astype(int)
    df["integration"] = df["integration"].astype(int)
    df["category"] = [E.quadrant(e, i) for e, i in zip(df["engagement"], df["integration"],
                                                        strict=True)]
    meta = E.load_papers()[["Retrieval", "Publication_Year", "Title", "Abstract"]]
    df = df.drop(columns=[c for c in meta.columns if c in df]).join(meta, how="left")
    df["input"] = ["full text" if (E.TXT_DIR / f"{k}.txt").exists() else "abstract only"
                   for k in df.index]
    return df


def join_annotations(df: pd.DataFrame) -> pd.DataFrame:
    """Add the existing risk and mitigation annotations (read-only) to classified papers.

    Uses the ``mixed_*`` columns of data/annotated_papers.csv, as the paper's figures do,
    and adds their high-level categories from data/categories.csv.
    """
    ann = pd.read_csv(ANNOTATIONS, index_col=0)
    cols = {"mixed_risk_categories": "risk_categories",
            "mixed_mitigation_categories": "mitigation_categories"}
    joined = df.join(ann[list(cols)].rename(columns=cols), how="left")
    low_to_high = (pd.read_csv(CATEGORIES).drop_duplicates("low")
                   .set_index("low")["high"])
    for kind in ("risk", "mitigation"):
        joined[f"{kind}_categories_high"] = [
            ";".join(dict.fromkeys(low_to_high.get(c.strip(), c.strip())
                                   for c in str(v).split(";") if c.strip()))
            if isinstance(v, str) else np.nan
            for v in joined[f"{kind}_categories"]
        ]
    return joined


def explode(joined: pd.DataFrame, kind: str, level: str = "high") -> pd.DataFrame:
    """One row per (paper, category) for risk or mitigation categories."""
    col = f"{kind}_categories_high" if level == "high" else f"{kind}_categories"
    long = joined[col].dropna().str.split(";").explode().str.strip()
    long = long[long != ""]
    return long.rename("cat").to_frame().join(joined.drop(columns=[col]))


def wilson(k: np.ndarray, n: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    """95% Wilson score interval for k successes out of n."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / n
        centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)


# ---- quotes (figure 6) ------------------------------------------------------------
def _normalize(text: str) -> str:
    text = text.lower().replace("’", "'").replace("‘", "'")
    text = re.sub(r"[“”\"]", "", text)
    text = re.sub(r"-\s*\n\s*", "", text)
    return re.sub(r"[^\w']+", " ", text).strip()


def verified_quotes(key: str, evidence) -> list[str]:
    """Evidence quotes that occur verbatim in the paper's text (the model may paraphrase)."""
    txt = E.TXT_DIR / f"{key}.txt"
    if not txt.exists() or not isinstance(evidence, str) or not evidence.strip():
        return []
    body = _normalize(txt.read_text())
    quotes = [q.strip().strip("\"“”") for q in evidence.split(" | ")]
    return [q for q in quotes if q and _normalize(q) in body]


def pick_quote(key: str, row: pd.Series, max_words: int = 45,
               use_evidence: bool = True) -> tuple[str, str]:
    """(quote, source): a verified evidence quote, else the abstract's first sentence."""
    quotes = verified_quotes(key, row.get("evidence")) if use_evidence else []
    if quotes:
        quote, source = min(quotes, key=lambda q: abs(len(q.split()) - 25)), "evidence"
    elif isinstance(row.get("Abstract"), str):
        quote = re.split(r"(?<=[.!?])\s+", row["Abstract"].strip())[0]
        source = "abstract"
    else:
        return "", "none"
    words = quote.split()
    return (" ".join(words[:max_words]) + " …" if len(words) > max_words else quote,
            source)


def pick_examples(df: pd.DataFrame, overrides: dict[str, str]) -> dict:
    """One representative paper per mode.

    Preference: full text; a quote verifiable against the text; levels furthest from
    the quadrant boundaries (most typical of the mode).
    """
    examples = {}
    for mode in MODES:
        if mode in overrides:
            key = overrides[mode]
        else:
            cands = df[df["category"] == mode].copy()
            if cands.empty:
                examples[mode] = None
                continue
            cands["full_text"] = cands["input"] == "full text"
            # a disengaged paper is best shown by its own framing (the abstract), so
            # evidence quotes are not needed for it
            cands["has_quote"] = [mode == MODES[0] or bool(verified_quotes(k, r.get("evidence")))
                                  for k, r in cands.iterrows()]
            cands["depth"] = np.hypot(cands["engagement"] - X_CUT,
                                      cands["integration"] - Y_CUT)
            cands = cands.sort_values(["full_text", "has_quote", "depth"],
                                      ascending=False, kind="stable")
            key = cands.index[0]
        quote, source = pick_quote(key, df.loc[key], use_evidence=mode != MODES[0])
        examples[mode] = (key, quote, source)
    return examples


# ---- figure helpers ------------------------------------------------------------------
def _style(ax, grid_axis: str | None = None):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_3)
    ax.tick_params(colors=INK_2, labelsize=9, length=3)
    if grid_axis:
        ax.grid(axis=grid_axis, color="#e6e5e1", lw=0.8, zorder=0)
        ax.set_axisbelow(True)


def _save(fig, name: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(out_dir / f"{name}.png", dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    typer.secho(f"  wrote {out_dir / name}.pdf")


def _fields(df: pd.DataFrame) -> list[str]:
    return [f for f in FIELD_COLORS if f in set(df["Retrieval"])]


# ---- figure 1: level grid ---------------------------------------------------------------
def _quadrants(ax, bottom: float = 0.5):
    """Quadrant tint and boundaries, so the Figure 1 structure reads at a glance."""
    for mode, (x0, x1, y0, y1) in {
        "Disengagement": (0.5, GRID_X, bottom, GRID_Y),
        "Compartmentalized coexistence": (0.5, GRID_X, GRID_Y, TOP),
        "Radical confrontation": (GRID_X, TOP, bottom, GRID_Y),
        "Critical bridging": (GRID_X, TOP, GRID_Y, TOP),
    }.items():
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, color=MODE_FILLS[mode],
                                   lw=0, zorder=0))
    ax.axvline(GRID_X, color=INK_3, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.axhline(GRID_Y, color=INK_3, lw=1, ls=(0, (4, 3)), zorder=1)


def _wedges(e: int, i: int) -> list[tuple[float, float, str]]:
    """(start angle, end angle, mode) pieces of the node at (e, i): one whole circle,
    or, on a boundary through the middle level, one piece per adjacent quadrant."""
    on_x, on_y = e == MID, i == MID
    if on_x and on_y:
        return [(0, 90, "Critical bridging"), (90, 180, "Compartmentalized coexistence"),
                (180, 270, "Disengagement"), (270, 360, "Radical confrontation")]
    if on_x:  # left half: low engagement, right half: high
        return [(90, 270, E.quadrant(1, i)), (-90, 90, E.quadrant(max(E.LEVELS), i))]
    if on_y:  # bottom half: low integration, top half: high
        return [(180, 360, E.quadrant(e, 1)), (0, 180, E.quadrant(e, max(E.LEVELS)))]
    return [(0, 360, E.quadrant(e, i))]


def _node(ax, e: int, i: int, size: float):
    """A grid node of the given marker area; split into coloured wedges on a boundary."""
    pieces = _wedges(e, i)
    if len(pieces) == 1:
        ax.scatter(e, i, s=size, color=MODE_COLORS[pieces[0][2]], linewidths=0, zorder=3)
        return
    for start, end, mode in pieces:
        # a wedge path spans the unit circle, so it scales like a full circle marker
        # no seam between the pieces: it would cut through the count on top
        ax.scatter(e, i, s=size, marker=MPath.wedge(start, end), color=MODE_COLORS[mode],
                   linewidths=0, zorder=3)


def _mode_handles() -> list:
    handles = [plt.Line2D([], [], marker="o", ls="", ms=8, mfc=MODE_COLORS[m], mew=0,
                          label=m) for m in MODES]
    if MID:
        handles.append(plt.Line2D([], [], marker="o", ls="", ms=8, fillstyle="left",
                                  mfc=MODE_COLORS[MODES[0]], mfcalt=MODE_COLORS[MODES[1]],
                                  mew=0, label=f"Score {MID}: between modes"))
    return handles


def fig_grid(df: pd.DataFrame, out_dir: Path):
    fields = _fields(df)
    fig, axes = plt.subplots(1, len(fields), figsize=(5.2 * len(fields), 5.0),
                             sharey=True, facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    counts = df.groupby(["Retrieval", "engagement", "integration"]).size()
    max_share = (counts / counts.groupby(level=0).transform("sum")).max()
    for ax, field in zip(axes, fields, strict=True):
        sub = df[df["Retrieval"] == field]
        n = len(sub)
        _quadrants(ax)
        for (e, i), k in sub.groupby(["engagement", "integration"]).size().items():
            share = k / n
            # circle area proportional to the share of the field's papers
            _node(ax, e, i, max(260, 2600 * share / max_share))
            ax.text(e, i, f"{k}", ha="center", va="center", fontsize=8.5, zorder=4,
                    color=SURFACE)
        ax.set_xlim(0.5, TOP)
        ax.set_ylim(0.5, TOP)
        ax.set_xticks(E.LEVELS)
        ax.set_yticks(E.LEVELS)
        ax.set_aspect("equal")
        ax.set_title(f"{FIELD_NAMES[field]} papers (n = {n})", fontsize=11, color=INK,
                     loc="left")
        ax.set_xlabel("Engagement", fontsize=11, color=INK, fontweight="bold")
        _style(ax)
    axes[0].set_ylabel("Integration", fontsize=11, color=INK, fontweight="bold")
    handles = _mode_handles()
    for h in handles:
        h.set_markersize(10)
    # two rows, so the larger text fits within the width of the panels
    fig.legend(handles=handles, loc="lower center", ncol=math.ceil(len(handles) / 2),
               frameon=False, fontsize=12, labelcolor=INK_2, bbox_to_anchor=(0.5, -0.13))
    _save(fig, "fig_modes_grid", out_dir)


def fig_grid_combined(df: pd.DataFrame, out_dir: Path):
    """One grid for both fields: pooled counts, with each field's count beneath."""
    fields = _fields(df)
    fig, ax = plt.subplots(figsize=(5.4, 5.6), facecolor=SURFACE)
    bottom = 0.1  # room for the breakdown under the bottom row
    _quadrants(ax, bottom)
    counts = df.groupby(["engagement", "integration", "Retrieval"]).size().unstack(
        fill_value=0).reindex(columns=fields, fill_value=0)
    totals = counts.sum(axis=1)
    for (e, i), k in totals.items():
        # circle area proportional to the share of all papers
        size = max(420, 3000 * k / totals.max())
        _node(ax, e, i, size)
        ax.text(e, i, f"{k}", ha="center", va="center", fontsize=11, zorder=4,
                color=SURFACE)
        split = "\n".join(f"{FIELD_ABBREV[f]} {counts.loc[(e, i), f]}" for f in fields)
        # just below the circle's edge (marker size is an area in points^2)
        ax.annotate(split, (e, i), xytext=(0, -(math.sqrt(size / math.pi) + 3)),
                    textcoords="offset points", ha="center", va="top", fontsize=9, linespacing=1.1,
                    zorder=4, color=INK_2,
                    # on the vertical boundary, mask the dashed line behind the text
                    bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.85)
                    if e == MID else None)
    ax.set_xlim(0.5, TOP)
    ax.set_ylim(bottom, TOP)
    ax.set_xticks(E.LEVELS)
    ax.set_yticks(E.LEVELS)
    ax.set_xlabel("Engagement", fontsize=13, color=INK, fontweight="bold")
    ax.set_ylabel("Integration", fontsize=13, color=INK, fontweight="bold")
    _style(ax)
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    _save(fig, "fig_modes_grid_combined", out_dir)


# ---- figure 2: shares by field ---------------------------------------------------------------
def fig_shares(df: pd.DataFrame, out_dir: Path):
    fields = _fields(df)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.4), facecolor=SURFACE,
                                   gridspec_kw={"width_ratios": [1.1, 1]})
    # left: 100% stacked bars
    for y, field in enumerate(fields):
        sub = df[df["Retrieval"] == field]
        left = 0.0
        for mode in MODES:
            share = (sub["category"] == mode).mean() * 100
            ax1.barh(y, share, left=left, color=MODE_COLORS[mode], height=0.6,
                     edgecolor=SURFACE, linewidth=2, zorder=2)
            if share >= 6:
                ax1.text(left + share / 2, y, f"{share:.0f}%", ha="center", va="center",
                         fontsize=11, color=SURFACE)
            left += share
    ax1.set_yticks(range(len(fields)), [f"{FIELD_NAMES[f]}\n(n = "
                                        f"{(df['Retrieval'] == f).sum()})" for f in fields])
    ax1.invert_yaxis()
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("Share of papers (%)", fontsize=13, color=INK)
    _style(ax1, "x")
    ax1.tick_params(labelsize=11)
    # right: zoom on the rarer modes, with 95% Wilson intervals
    offsets = np.linspace(-0.15, 0.15, len(fields))
    for off, field in zip(offsets, fields, strict=True):
        sub = df[df["Retrieval"] == field]
        k = np.array([(sub["category"] == m).sum() for m in RARE_MODES])
        n = np.full(len(RARE_MODES), len(sub))
        lo, hi = wilson(k, n)
        ys = np.arange(len(RARE_MODES)) + off
        ax2.errorbar(k / n * 100, ys, xerr=[(k / n - lo) * 100, (hi - k / n) * 100], fmt="o",
                     color=FIELD_COLORS[field], ms=6, capsize=0, lw=1.6,
                     label=FIELD_NAMES[field], zorder=3)
    ax2.set_yticks(range(len(RARE_MODES)), RARE_MODES)
    ax2.invert_yaxis()
    ax2.set_xlim(left=0)
    ax2.set_xlabel("Share of papers (%), 95% CI", fontsize=13, color=INK)
    ax2.legend(frameon=False, fontsize=11, labelcolor=INK_2, loc="lower right")
    _style(ax2, "x")
    ax2.tick_params(labelsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, color=MODE_COLORS[m], label=m) for m in MODES]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=11,
               labelcolor=INK_2, bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    _save(fig, "fig_modes_shares", out_dir)


def fig_shares_rare(df: pd.DataFrame, out_dir: Path):
    """The right panel of fig_shares on its own, as wide as fig_grid_combined (5.4in)
    and with its font sizes, to sit beneath it at the same scale."""
    fields = _fields(df)
    fig, ax = plt.subplots(figsize=(5.4, 2.0), facecolor=SURFACE)
    offsets = np.linspace(-0.17, 0.17, len(fields))
    for off, field in zip(offsets, fields, strict=True):
        sub = df[df["Retrieval"] == field]
        k = np.array([(sub["category"] == m).sum() for m in RARE_MODES])
        n = np.full(len(RARE_MODES), len(sub))
        lo, hi = wilson(k, n)
        ys = np.arange(len(RARE_MODES)) + off
        ax.errorbar(k / n * 100, ys, xerr=[(k / n - lo) * 100, (hi - k / n) * 100], fmt="o",
                    color=FIELD_COLORS[field], ms=7, capsize=0, lw=1.8,
                    label=FIELD_ABBREV[field], zorder=3)
    labels = [textwrap.fill(m, 16, break_long_words=False) for m in RARE_MODES]
    ax.set_yticks(range(len(RARE_MODES)), labels)
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.set_xlabel("Share of papers (%), 95% CI", fontsize=13, color=INK, fontweight="bold")
    ax.legend(frameon=False, fontsize=11, labelcolor=INK_2, loc="lower right")
    _style(ax, "x")
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    _save(fig, "fig_modes_shares_rare", out_dir)


# ---- figure 3: over time --------------------------------------------------------------------
def _periods(df: pd.DataFrame, period_years: int, start_year: int):
    """Add a ``period`` column of period_years-long bins; papers before start_year are
    pooled into the first period (the AI ethics venues start in 2018). Returns the
    frame, the sorted periods and their tick labels."""
    year = df["Publication_Year"].astype(int).clip(lower=start_year)
    df = df.assign(period=start_year + (year - start_year) // period_years * period_years)
    periods = sorted(df["period"].unique())
    labels = [str(p) if period_years == 1 else f"{p}–{str(p + period_years - 1)[-2:]}"
              for p in periods]
    if (df["Publication_Year"].astype(int) < start_year).any():
        labels[0] = f"≤{periods[0] + period_years - 1}"
    return df, periods, labels


def fig_time(df: pd.DataFrame, out_dir: Path, period_years: int, start_year: int,
             min_n: int):
    """Periods with fewer than min_n papers in a field are not plotted for that field."""
    fields = _fields(df)
    df, periods, labels = _periods(df, period_years, start_year)
    fig, axes = plt.subplots(1, len(RARE_MODES), figsize=(4.2 * len(RARE_MODES), 3.4),
                             sharey=True, facecolor=SURFACE)
    for ax, mode in zip(axes, RARE_MODES, strict=True):
        for field in fields:
            sub = df[df["Retrieval"] == field]
            n = sub.groupby("period").size().reindex(periods, fill_value=0).to_numpy()
            k = (sub[sub["category"] == mode].groupby("period").size()
                 .reindex(periods, fill_value=0).to_numpy())
            n = np.where(n >= min_n, n, 0)  # too few papers: leave a gap
            lo, hi = wilson(k, n)
            x = np.arange(len(periods))
            with np.errstate(invalid="ignore", divide="ignore"):
                ax.plot(x, np.where(n > 0, k / n * 100, np.nan), "-o",
                        color=FIELD_COLORS[field], lw=2, ms=5, label=FIELD_NAMES[field],
                        zorder=3)
            ax.fill_between(x, lo * 100, hi * 100, color=FIELD_COLORS[field], alpha=0.12,
                            lw=0, zorder=2)
        ax.set_xticks(range(len(periods)), labels, rotation=0)
        ax.set_ylim(bottom=0)
        _style(ax, "y")
    axes[0].set_ylabel("Share of the field's papers (%)", fontsize=10, color=INK)
    axes[-1].legend(frameon=False, fontsize=9, labelcolor=INK_2)
    fig.tight_layout()
    _save(fig, "fig_modes_time", out_dir)


# one marker per mode, so the lines stay apart without colour (print, colour blindness)
MODE_MARKERS = dict(zip(RARE_MODES, ["s", "^", "D"], strict=True))


def fig_time_combined(df: pd.DataFrame, out_dir: Path, period_years: int,
                      start_year: int, min_n: int, name: str = "fig_modes_time_combined"):
    """The rarer modes over time in one panel, both fields pooled: the share of all
    papers in each period, with 95% Wilson bands. Periods with fewer than min_n
    papers are left out. As wide as fig_grid_combined, with its font sizes."""
    df, periods, labels = _periods(df, period_years, start_year)
    n = df.groupby("period").size().reindex(periods, fill_value=0).to_numpy()
    n = np.where(n >= min_n, n, 0)  # too few papers: leave a gap
    x = np.arange(len(periods))
    fig, ax = plt.subplots(figsize=(5.4, 3.2), facecolor=SURFACE)
    top = 0.0
    for mode in RARE_MODES:
        k = (df[df["category"] == mode].groupby("period").size()
             .reindex(periods, fill_value=0).to_numpy())
        lo, hi = wilson(k, n)
        with np.errstate(invalid="ignore", divide="ignore"):
            ax.plot(x, np.where(n > 0, k / n * 100, np.nan), "-",
                    marker=MODE_MARKERS[mode], color=MODE_COLORS[mode], lw=2, ms=8,
                    mec=SURFACE, mew=1.5, label=mode, zorder=3)
        ax.fill_between(x, lo * 100, hi * 100, color=MODE_COLORS[mode], alpha=0.10, lw=0,
                        zorder=2)
        top = max(top, np.nanmax(hi) * 100)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, top * 1.4)  # headroom so the legend clears the bands
    ax.set_ylabel("Share of papers (%)", fontsize=13, color=INK, fontweight="bold")
    ax.legend(frameon=False, fontsize=11, labelcolor=INK_2, loc="upper left")
    _style(ax, "y")
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    _save(fig, name, out_dir)


# ---- figure 4: bridging problems --------------------------------------------------------------
# the outcome of the problems models: a paper is high on the axis (integration: it
# integrates both fields' concerns; engagement: it engages the other field or the divide)
AXIS_HIGH = {"integration": E.INTEGRATION_HIGH, "engagement": E.ENGAGEMENT_HIGH}
AXIS_PAPERS = {"integration": "integrating", "engagement": "engaging"}


def problem_table(joined: pd.DataFrame, kind: str, level: str, min_papers: int,
                  axis: str = "integration"):
    """Log odds ratio of being high on axis, in vs outside each category.

    A paper "integrates" when its integration level is high (Compartmentalized
    coexistence or Critical bridging) and "engages" when its engagement level is high
    (Radical confrontation or Critical bridging). Haldane-Anscombe correction (+0.5)
    and a Wald 95% interval.
    """
    joined = joined.assign(high=joined[axis] >= AXIS_HIGH[axis])
    long = explode(joined, kind, level)
    total_k, total_n = int(joined["high"].sum()), len(joined)
    rows = []
    for cat, g in long.groupby("cat"):
        papers = g[~g.index.duplicated()]
        n, k = len(papers), int(papers["high"].sum())
        if n < min_papers:
            continue
        a, b = k + 0.5, n - k + 0.5  # in category: high / not
        c, d = total_k - k + 0.5, total_n - n - (total_k - k) + 0.5  # outside
        lor = math.log(a * d / (b * c))
        se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        rows.append({"category": cat, "n": n, "k": k, "share": k / n,
                     "log_or": lor, "lo": lor - 1.96 * se, "hi": lor + 1.96 * se})
    return pd.DataFrame(rows).sort_values("log_or") if rows else pd.DataFrame()


def _trim(t: pd.DataFrame, keep: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The keep categories at each end of a table sorted by log odds ratio, and the
    middle ones left out (the categories closest to no association)."""
    if len(t) <= 2 * keep + 1:
        return t, t.iloc[0:0]
    return pd.concat([t.iloc[:keep], t.iloc[-keep:]]), t.iloc[keep:-keep]


def fig_problems(joined: pd.DataFrame, out_dir: Path, level: str, min_papers: int,
                 keep: int):
    tables, gaps = {}, {}
    for kind in ("risk", "mitigation"):
        t, cut = _trim(problem_table(joined, kind, level, min_papers), keep)
        tables[kind], gaps[kind] = t, len(cut)
        if len(cut):
            sig = ((cut["lo"] > 0) | (cut["hi"] < 0)).sum()
            typer.secho(f"  {kind}: left out {len(cut)} middle categories ({sig} with a CI "
                        f"excluding 0): {', '.join(cut['category'])}")
    heights = [len(t) + (gaps[k] > 0) or 1 for k, t in tables.items()]
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.28 * max(heights) + 1.6),
                             facecolor=SURFACE)
    for ax, (kind, t) in zip(axes, tables.items(), strict=True):
        _style(ax, "x")
        ax.set_title(f"{kind.capitalize()} categories", fontsize=10.5, color=INK, loc="left")
        if t.empty:
            ax.text(0.5, 0.5, f"No category with >= {min_papers} papers", ha="center",
                    transform=ax.transAxes, color=INK_2)
            continue
        # rows bottom to top; a gap row between the two ends marks the left-out middle
        gap = gaps[kind] > 0
        y = np.arange(len(t)) + (gap & (np.arange(len(t)) >= keep))
        sig = (t["lo"] > 0) | (t["hi"] < 0)
        color = np.where(t["log_or"] > 0, MODE_COLORS["Critical bridging"], INK_3)
        ax.hlines(y, t["lo"], t["hi"], color=color, lw=1.6, zorder=2)
        ax.scatter(t["log_or"], y, s=36, zorder=3, color=np.where(sig, color, SURFACE),
                   edgecolors=color, linewidths=1.6)
        ax.axvline(0, color=INK_3, lw=1, zorder=1)
        labels = [f"{textwrap.shorten(c, 48, placeholder='…')} ({i}/{n})"
                  for c, i, n in zip(t["category"], t["k"], t["n"], strict=True)]
        if gap:
            y = np.append(y, keep)
            labels.append(f"⋯ {gaps[kind]} more categories")
        ax.set_yticks(y, labels, fontsize=8.5)
        if gap:
            ax.get_yticklabels()[-1].set_color(INK_3)
            ax.get_yticklabels()[-1].set_fontstyle("italic")
        ax.set_xlabel("Log odds ratio, integrating papers (95% CI)", fontsize=9.5,
                      color=INK)
    fig.tight_layout()
    _save(fig, "fig_modes_problems", out_dir)


YEAR_BINS = [(None, 2018), (2019, 2020), (2021, 2022), (2023, 2023), (2024, 2024),
             (2025, None)]  # the first bin is the reference


def problem_design(joined: pd.DataFrame, level: str, min_papers: int,
                   axis: str = "integration"):
    """Design matrix for the adjusted problems model, and the outcome (high on axis).

    One indicator per risk and per mitigation category with at least min_papers papers
    (all in one model, so each is adjusted for the others), plus controls: home field,
    publication period and log full-text length in words (longer papers have more room
    to take up both fields' concerns).
    """
    X = pd.DataFrame({"const": 1.0}, index=joined.index)
    terms = []
    for kind in ("risk", "mitigation"):
        long = explode(joined, kind, level)
        for cat, g in long.groupby("cat"):
            keys = g.index.unique()
            if len(keys) >= min_papers:
                X[f"{kind}: {cat}"] = joined.index.isin(keys).astype(float)
                terms.append((kind, cat))
    X["Field: AI safety"] = (joined["Retrieval"] == "Safety").astype(float)
    year = joined["Publication_Year"].astype(int)
    for lo, hi in YEAR_BINS[1:]:
        name = f"Year: {lo}" if lo == hi else f"Year: {lo}+" if hi is None else \
            f"Year: {lo}-{hi}"
        X[name] = ((year >= lo) & (year <= (hi or year.max()))).astype(float)
    words = pd.Series({k: len((E.TXT_DIR / f"{k}.txt").read_text(errors="ignore").split())
                       for k in joined.index})
    log_words = np.log(words.clip(lower=1))
    X["Log words (centred)"] = log_words - log_words.mean()
    y = (joined[axis] >= AXIS_HIGH[axis]).astype(float)
    return X, y, terms


def problem_regression(joined: pd.DataFrame, level: str, min_papers: int,
                       axis: str = "integration") -> pd.DataFrame:
    """Adjusted log odds ratios of being high on axis, from one Firth logistic regression.

    Profile-likelihood 95% intervals; Holm-adjusted p-values across the category terms.
    The unadjusted log odds ratio of problem_table is kept alongside for comparison.
    """
    X, y, terms = problem_design(joined, level, min_papers, axis)
    res = firth_logit(X, y)
    cats = [f"{k}: {c}" for k, c in terms]
    res["p_holm"] = np.nan
    res.loc[cats, "p_holm"] = holm(res.loc[cats, "p"])
    res["kind"] = [t.split(": ", 1)[0] if t in cats else "control" for t in res.index]
    res["category"] = [t.split(": ", 1)[1] if t in cats else t for t in res.index]
    res["n"] = X.sum().astype(int)
    res["k"] = X.mul(y, axis=0).sum().astype(int)
    for kind in ("risk", "mitigation"):
        raw = problem_table(joined, kind, level, min_papers, axis).set_index("category")
        idx = res.index[res["kind"] == kind]
        res.loc[idx, "log_or_unadjusted"] = raw.loc[res.loc[idx, "category"],
                                                    "log_or"].to_numpy()
    res.attrs["papers"], res.attrs["events"] = len(y), int(y.sum())
    return res


def fig_problems_adjusted(joined: pd.DataFrame, out_dir: Path, level: str,
                          min_papers: int, keep: int, axis: str = "integration"):
    suffix = f"_{axis}"
    res = problem_regression(joined, level, min_papers, axis)
    res.to_csv(out_dir / f"problems_regression{suffix}.csv")
    n, events = res.attrs["papers"], res.attrs["events"]
    typer.secho(f"  adjusted {axis} model: {n} papers, {events} {AXIS_PAPERS[axis]}, "
                f"{len(res)} parameters ({events / (len(res) - 1):.1f} events per term)")
    ctrl = res[res["kind"] == "control"]
    for term, r in ctrl.iterrows():
        typer.secho(f"    {term:24s} {r['coef']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] "
                    f"p={r['p']:.3f}")
    tables, gaps = {}, {}
    for kind in ("risk", "mitigation"):
        t = res[res["kind"] == kind].rename(columns={"coef": "log_or"}).sort_values("log_or")
        tables[kind], cut = _trim(t, keep)
        gaps[kind] = len(cut)
        if len(cut):
            sig = ((cut["lo"] > 0) | (cut["hi"] < 0)).sum()
            typer.secho(f"  {kind} (adjusted): left out {len(cut)} middle categories "
                        f"({sig} with a CI excluding 0): {', '.join(cut['category'])}")
    heights = [len(t) + (gaps[k] > 0) for k, t in tables.items()]
    fig, axes = plt.subplots(1, 2, figsize=(16, 0.42 * max(heights) + 2.4),
                             facecolor=SURFACE)
    for ax, (kind, t) in zip(axes, tables.items(), strict=True):
        _style(ax, "x")
        ax.tick_params(labelsize=12)
        ax.set_title(f"{kind.capitalize()} categories", fontsize=14, color=INK, loc="left",
                     fontweight="bold")
        gap = gaps[kind] > 0
        y = np.arange(len(t)) + (gap & (np.arange(len(t)) >= keep))
        sig = (t["lo"] > 0) | (t["hi"] < 0)
        color = np.where(t["log_or"] > 0, MODE_COLORS["Critical bridging"], INK_3)
        # unbounded profile intervals (no high paper) run to the panel edge
        ax.hlines(y, t["lo"].clip(-8, 8), t["hi"].clip(-8, 8), color=color, lw=2,
                  zorder=2)
        ax.scatter(t["log_or_unadjusted"], y, marker="|", s=110, color=INK_3, lw=1.5,
                   zorder=2.5)
        ax.scatter(t["log_or"], y, s=60, zorder=3, color=np.where(sig, color, SURFACE),
                   edgecolors=color, linewidths=2)
        ax.axvline(0, color=INK_3, lw=1, zorder=1)
        labels = [f"{textwrap.shorten(c, 48, placeholder='…')} ({i}/{m})"
                  + (" *" if p < 0.05 else "")
                  for c, i, m, p in zip(t["category"], t["k"], t["n"],
                                        t["p_holm"], strict=True)]
        if gap:
            y = np.append(y, keep)
            labels.append(f"⋯ {gaps[kind]} more categories")
        ax.set_yticks(y, labels, fontsize=12)
        if gap:
            ax.get_yticklabels()[-1].set_color(INK_3)
            ax.get_yticklabels()[-1].set_fontstyle("italic")
        lo, hi = t["lo"].replace(-np.inf, np.nan).min(), t["hi"].replace(np.inf,
                                                                         np.nan).max()
        ax.set_xlim(min(lo, t["log_or_unadjusted"].min()) - 0.3,
                    max(hi, t["log_or_unadjusted"].max()) + 0.3)
        ax.set_xlabel("Adjusted log odds ratio (95% CI)",
                      fontsize=13, color=INK, fontweight="bold")
    handles = [
        plt.Line2D([], [], marker="o", ls="-", color=INK_2, mfc=INK_2, ms=9,
                   label="Adjusted (95% profile CI excludes 0)"),
        plt.Line2D([], [], marker="o", ls="-", color=INK_2, mfc=SURFACE, ms=9,
                   label="Adjusted (CI includes 0)"),
        plt.Line2D([], [], marker="|", ls="none", color=INK_3, ms=14, mew=1.6,
                   label="Unadjusted"),
        plt.Line2D([], [], ls="none", label="* Holm-adjusted p < 0.05"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=14,
               labelcolor=INK_2, handlelength=1.6)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    _save(fig, f"fig_modes_problems_adjusted{suffix}", out_dir)


# ---- figure 5: level distributions ---------------------------------------------------------------
def fig_levels(df: pd.DataFrame, out_dir: Path):
    fields = _fields(df)
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.85 * len(fields) + 1.8), sharey=True,
                             facecolor=SURFACE)
    for ax, axis in zip(axes, ("engagement", "integration"), strict=True):
        for y, field in enumerate(fields):
            sub = df[df["Retrieval"] == field]
            shares = sub[axis].value_counts(normalize=True).reindex(E.LEVELS, fill_value=0)
            left = 0.0
            for level, share in shares.items():
                share *= 100
                ax.barh(y, share, left=left, color=LEVEL_COLORS[level], height=0.6,
                        edgecolor=SURFACE, linewidth=2, zorder=2)
                if share >= 5:
                    ax.text(left + share / 2, y, f"{share:.0f}%", ha="center", va="center",
                            fontsize=11, color=INK if level <= 2 else SURFACE)
                left += share
        ax.set_xlim(0, 100)
        ax.set_xlabel("Share of papers (%)", fontsize=13, color=INK)
        _style(ax, "x")
        ax.tick_params(labelsize=11)
    axes[0].set_yticks(range(len(fields)), [FIELD_NAMES[f] for f in fields])
    axes[0].invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=f"Level {i}")
               for i, c in LEVEL_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=11,
               labelcolor=INK_2, bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    _save(fig, "fig_modes_levels", out_dir)


# ---- figure 6: examples ------------------------------------------------------------------------
def fig_examples(df: pd.DataFrame, examples: dict, out_dir: Path):
    # same layout as Figure 1: low engagement on the left, high integration on top
    layout = {"Compartmentalized coexistence": (0, 0), "Critical bridging": (0, 1),
              "Disengagement": (1, 0), "Radical confrontation": (1, 1)}
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.0), facecolor=SURFACE)
    for mode, (r, c) in layout.items():
        ax = axes[r, c]
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(MODE_COLORS[mode])
            spine.set_linewidth(2)
        ax.set_facecolor(MODE_FILLS[mode])
        n = int((df["category"] == mode).sum())
        ax.text(0.04, 0.93, f"{mode}", transform=ax.transAxes, fontsize=12,
                fontweight="bold", color=INK, va="top")
        ax.text(0.96, 0.93, f"n = {n}", transform=ax.transAxes, fontsize=9.5, color=INK_2,
                va="top", ha="right")
        ex = examples.get(mode)
        if ex is None:
            ax.text(0.04, 0.72, "No paper in this quadrant.", transform=ax.transAxes,
                    fontsize=9.5, color=INK_2, va="top")
            continue
        key, quote, source = ex
        row = df.loc[key]
        head = textwrap.fill(f"{row['Title']}", 56)
        meta = (f"{FIELD_NAMES[row['Retrieval']]}, {row['Publication_Year']} · "
                f"engagement {row['engagement']}, integration {row['integration']}")
        body = textwrap.fill(f"“{quote}”", 64) if quote else ""
        note = " · quote from abstract" if source == "abstract" else ""
        ax.text(0.04, 0.78, head, transform=ax.transAxes, fontsize=9.5, color=INK, va="top",
                fontweight="semibold", linespacing=1.3)
        ax.text(0.04, 0.78 - 0.09 * (head.count("\n") + 1), meta + note,
                transform=ax.transAxes, fontsize=8.5, color=INK_2, va="top")
        ax.text(0.04, 0.78 - 0.09 * (head.count("\n") + 1) - 0.14, body,
                transform=ax.transAxes, fontsize=9, color=INK, va="top", style="italic",
                linespacing=1.35)
    fig.text(0.5, 0.005, "Engagement →", ha="center", fontsize=10, color=INK_2)
    fig.text(0.005, 0.5, "Integration →", va="center", rotation=90, fontsize=10,
             color=INK_2)
    fig.tight_layout(rect=(0.02, 0.02, 1, 1))
    _save(fig, "fig_modes_examples", out_dir)


# ---- figure 7: taxonomy mixing --------------------------------------------------------------
def taxonomy_mixing(joined: pd.DataFrame) -> pd.DataFrame:
    """Per paper, how many of its low-level risk and mitigation categories come from the
    other field's taxonomy in data/categories.csv (independent of the classifier)."""
    low_field = pd.read_csv(CATEGORIES).drop_duplicates("low").set_index("low")["field"]
    long = pd.concat([explode(joined, kind, "low") for kind in ("risk", "mitigation")])
    long["other"] = long["cat"].map(low_field) != long["Retrieval"]
    per = long.groupby(level=0).agg(n=("other", "size"), k=("other", "sum"))
    per["share"] = per["k"] / per["n"]
    return per.join(joined[["Retrieval", "category"]])


def _bootstrap_mean(x: np.ndarray, rng, reps: int = 2000) -> tuple[float, float]:
    """95% percentile bootstrap interval of the mean."""
    means = rng.choice(x, (reps, len(x))).mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def fig_taxonomy(joined: pd.DataFrame, out_dir: Path):
    per = taxonomy_mixing(joined)
    fields = _fields(per)
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(5.4, 2.6), facecolor=SURFACE)
    offsets = np.linspace(-0.17, 0.17, len(fields))
    for off, field in zip(offsets, fields, strict=True):
        sub = per[per["Retrieval"] == field]
        means, los, his = [], [], []
        for mode in MODES:
            x = sub.loc[sub["category"] == mode, "share"].to_numpy()
            lo, hi = _bootstrap_mean(x, rng) if len(x) else (np.nan, np.nan)
            means.append(x.mean() if len(x) else np.nan)
            los.append(lo)
            his.append(hi)
            typer.secho(f"  taxonomy {FIELD_ABBREV[field]} {mode}: {len(x)} papers, "
                        f"{np.mean(x) * 100:.0f}% of categories from the other field")
        means, los, his = (np.array(v) * 100 for v in (means, los, his))
        ax.errorbar(means, np.arange(len(MODES)) + off, xerr=[means - los, his - means],
                    fmt="o", color=FIELD_COLORS[field], ms=7, capsize=0, lw=1.8,
                    label=f"{FIELD_ABBREV[field]} papers", zorder=3)
    labels = [textwrap.fill(m, 16, break_long_words=False) for m in MODES]
    ax.set_yticks(range(len(MODES)), labels)
    for tick, mode in zip(ax.get_yticklabels(), MODES, strict=True):
        tick.set_color(MODE_COLORS[mode])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Categories from the other field's\ntaxonomy (%, mean, 95% CI)",
                  fontsize=13, color=INK, fontweight="bold")
    ax.legend(frameon=False, fontsize=11, labelcolor=INK_2, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=2)
    _style(ax, "x")
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    _save(fig, "fig_modes_taxonomy", out_dir)


# ---- figure 8: references to the other field ---------------------------------------------------
def fig_reference(df: pd.DataFrame, out_dir: Path):
    """Which papers refer to the other field at all, and how many of those that do go on
    to engage or integrate it. The direction of reference follows from the field."""
    fields = _fields(df)
    ref = df["other_field_referenced"].astype(bool)
    segments = {  # the few unreferencing papers outside Disengagement keep their mode
        "No reference to the other field": (df["category"] == MODES[0]) & ~ref,
        "References it, Disengagement": (df["category"] == MODES[0]) & ref,
        **{m: df["category"] == m for m in RARE_MODES},
    }
    colors = {"No reference to the other field": MODE_FILLS[MODES[0]],
              "References it, Disengagement": MODE_COLORS[MODES[0]], **MODE_COLORS}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.4), facecolor=SURFACE,
                                   gridspec_kw={"width_ratios": [3.2, 1]})
    names = [f"{FIELD_ABBREV[f]} → {FIELD_ABBREV[o]}"
             for f in fields for o in FIELD_ABBREV if o != f]
    # left: 100% stacked bars
    for y, field in enumerate(fields):
        in_field = df["Retrieval"] == field
        left = 0.0
        for name, mask in segments.items():
            share = (mask & in_field).sum() / in_field.sum() * 100
            # the pale no-reference segment gets an outline so it reads on white
            pale = name == "No reference to the other field"
            ax1.barh(y, share, left=left, color=colors[name], height=0.6,
                     edgecolor=INK_3 if pale else SURFACE, linewidth=0.8 if pale else 2,
                     zorder=2)
            if share >= 6:
                ax1.text(left + share / 2, y, f"{share:.0f}%", ha="center", va="center",
                         fontsize=11, color=INK if pale else SURFACE)
            left += share
    ax1.set_yticks(range(len(fields)), [f"{name}\n(n = {(df['Retrieval'] == f).sum()})"
                                        for name, f in zip(names, fields, strict=True)])
    ax1.invert_yaxis()
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("Share of papers (%)", fontsize=13, color=INK)
    _style(ax1, "x")
    ax1.tick_params(labelsize=11)
    # right: among papers that refer to the other field, the share in each rarer mode
    offsets = np.linspace(-0.15, 0.15, len(fields))
    for off, field, name in zip(offsets, fields, names, strict=True):
        sub = df[(df["Retrieval"] == field) & ref]
        k = np.array([(sub["category"] == m).sum() for m in RARE_MODES])
        n = np.full(len(RARE_MODES), len(sub))
        lo, hi = wilson(k, n)
        typer.secho(f"  reference {name}: {len(sub)} of {(df['Retrieval'] == field).sum()} "
                    f"papers refer to the other field, {k.sum()} of them not disengaged")
        ax2.errorbar(k / n * 100, np.arange(len(RARE_MODES)) + off,
                     xerr=[(k / n - lo) * 100, (hi - k / n) * 100], fmt="o",
                     color=FIELD_COLORS[field], ms=6, capsize=0, lw=1.6, label=name,
                     zorder=3)
    ax2.set_yticks(range(len(RARE_MODES)),
                   [textwrap.fill(m, 16, break_long_words=False) for m in RARE_MODES])
    ax2.invert_yaxis()
    ax2.set_xlim(left=0)
    ax2.set_xlabel("Share of papers that refer to\nthe other field (%), 95% CI",
                   fontsize=13, color=INK)
    ax2.legend(frameon=False, fontsize=11, labelcolor=INK_2, loc="lower center",
               bbox_to_anchor=(0.5, 1.0), ncol=2)
    _style(ax2, "x")
    ax2.tick_params(labelsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[s], edgecolor=INK_3, lw=0.8,
                             label=s) for s in segments]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=11,
               labelcolor=INK_2, bbox_to_anchor=(0.5, -0.3))
    fig.tight_layout()
    _save(fig, "fig_modes_reference", out_dir)


# ---- figure 9: topic profile ---------------------------------------------------------------
def topic_profile(joined: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Share of each mode's papers annotated with each high-level category (rows), with
    the field whose taxonomy the category comes from."""
    long = explode(joined, kind, "high").rename_axis("key").reset_index()
    k = (long.drop_duplicates(["key", "cat"]).groupby(["cat", "category"]).size()
         .unstack(fill_value=0).reindex(columns=MODES, fill_value=0))
    t = k / joined["category"].value_counts().reindex(MODES)
    cats = pd.read_csv(CATEGORIES)
    origin = cats[cats["type"] == kind.capitalize()].drop_duplicates("high").set_index(
        "high")["field"]
    t = t[t.index.isin(origin.index)]  # drops the few labels from the other kind's taxonomy
    t["field"] = origin.reindex(t.index)
    return t.sort_values(["field", MODES[0]], ascending=[True, False])


def fig_topics(joined: pd.DataFrame, out_dir: Path):
    tables = {kind: topic_profile(joined, kind) for kind in ("risk", "mitigation")}
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
        "share", [SURFACE, LEVEL_RAMP[-1]])
    vmax = max(t[MODES].to_numpy().max() for t in tables.values()) * 100
    counts = joined["category"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(15, 0.36 * max(map(len, tables.values())) + 4),
                             facecolor=SURFACE)
    for ax, (kind, t) in zip(axes, tables.items(), strict=True):
        v = t[MODES].to_numpy() * 100
        ax.imshow(v, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        for (r, c), x in np.ndenumerate(v):
            ax.text(c, r, f"{x:.0f}", ha="center", va="center", fontsize=10,
                    color=SURFACE if x > vmax * 0.55 else INK)
        ax.set_yticks(range(len(t)), [textwrap.shorten(c, 46, placeholder="…")
                                      for c in t.index], fontsize=11)
        for tick, field in zip(ax.get_yticklabels(), t["field"], strict=True):
            tick.set_color(FIELD_COLORS[field])
        ax.xaxis.tick_top()
        ax.set_xticks(range(len(MODES)), [f"{m} (n = {counts[m]})" for m in MODES],
                      fontsize=11, rotation=30, ha="left", rotation_mode="anchor")
        for tick, mode in zip(ax.get_xticklabels(), MODES, strict=True):
            tick.set_color(MODE_COLORS[mode])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        # separate the categories of the two taxonomies
        split = int((t["field"] == t["field"].iloc[0]).sum())
        ax.axhline(split - 0.5, color=INK_2, lw=1.2)
        ax.set_title(f"{kind.capitalize()} categories", fontsize=14, color=INK, loc="left",
                     fontweight="bold", pad=150)
    handles = [plt.Line2D([], [], ls="none", marker="s", color=FIELD_COLORS[f], ms=9,
                          label=f"{FIELD_NAMES[f]} taxonomy") for f in FIELD_COLORS]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=12,
               labelcolor=INK_2, title="Cells: share of each mode's papers (%). "
               "Labels: category from the", title_fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _save(fig, "fig_modes_topics", out_dir)


# ---- commands ---------------------------------------------------------------------------------
def _resolve(results: list[Path] | None, pilot: bool) -> list[Path]:
    if results:
        return results
    return [E.out_dir() / ("engagement_pilot.csv" if pilot else "engagement.csv")]


@app.command()
def join(
    results: list[Path] = typer.Option(None, help="Parsed CSV or .jsonl result files."),
    pilot: bool = typer.Option(False, help="Use the pilot CSV instead of the full run."),
):
    """Join classifications with the existing annotations into a new CSV."""
    joined = join_annotations(load_results(_resolve(results, pilot)))
    dest = E.out_dir() / ("engagement_joined_pilot.csv" if pilot else "engagement_joined.csv")
    joined.to_csv(dest)
    typer.secho(f"Wrote {dest} ({len(joined)} papers; data/ left unchanged)")


@app.command()
def figures(
    results: list[Path] = typer.Option(None, help="Parsed CSV or .jsonl result files."),
    pilot: bool = typer.Option(False, help="Use the pilot CSV instead of the full run."),
    include_abstract_only: bool = typer.Option(
        False, help="Include papers classified from their abstract only."),
    period_years: int = typer.Option(2, help="Years per period in the time figure."),
    start_year: int = typer.Option(2018, help="Earlier years join the first period."),
    min_period_papers: int = typer.Option(5, help="Hide periods with fewer papers."),
    level: str = typer.Option("high", help="Category level for figure 4: high or low."),
    min_papers: int = typer.Option(10, help="Minimum papers per category in figure 4."),
    problem_rows: int = typer.Option(
        5, help="Categories kept at each end of figure 4's panels; the middle is cut."),
    example: list[str] = typer.Option(
        None, help='Override an example in figure 6, e.g. "Critical bridging=PUPA48EE".'),
    out_dir: Path = typer.Option(OUT),
):
    """Draw all figures."""
    df = load_results(_resolve(results, pilot))
    if not include_abstract_only:
        dropped = int((df["input"] != "full text").sum())
        df = df[df["input"] == "full text"]
        typer.secho(f"{len(df)} papers ({dropped} abstract-only papers left out)")
    joined = join_annotations(df)
    examples = pick_examples(df, dict(e.split("=", 1) for e in example or []))
    for mode, ex in examples.items():
        if ex:
            note = "" if ex[2] == "evidence" else f"  [quote from {ex[2]}]"
            typer.secho(f"example {mode}: {ex[0]} {df.loc[ex[0], 'Title'][:55]}{note}")
    fig_grid(df, out_dir)
    fig_grid_combined(df, out_dir)
    fig_shares(df, out_dir)
    fig_shares_rare(df, out_dir)
    fig_time(df, out_dir, period_years, start_year, min_period_papers)
    fig_time_combined(df, out_dir, period_years, start_year, min_period_papers)
    # one point per year; everything before 2019 pooled into the first point
    fig_time_combined(df, out_dir, 1, 2018, min_period_papers,
                      name="fig_modes_time_combined_yearly")
    fig_problems(joined, out_dir, level, min_papers, problem_rows)
    fig_problems_adjusted(joined, out_dir, level, min_papers, problem_rows)
    # fig_problems_adjusted(joined, out_dir, level, min_papers, problem_rows, "engagement")
    fig_levels(df, out_dir)
    fig_examples(df, examples, out_dir)
    fig_taxonomy(joined, out_dir)
    fig_reference(df, out_dir)
    fig_topics(joined, out_dir)


if __name__ == "__main__":
    app()
