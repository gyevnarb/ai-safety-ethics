"""Figures for the qualitative coding and thematic analysis (see aise.qualitative).

    uv run python -m aise.plot_qualitative                 # AISE_QUAL_VERSION, default v1
    AISE_QUAL_VERSION=test4 uv run python -m aise.plot_qualitative

Needs the outputs of ``aise.qualitative`` up to ``report`` (assignments.csv, themes.json;
theme_spans.csv for --theme-source spans). Several versions of each view are drawn, to
choose from (written to output/qualitative/<QUAL_VERSION>/plots/ as PDF and PNG):

1. fig_qual_growth          codebook size as papers are coded in stages 1 and 2, and the
                            size of the fixed codebook (a saturation check)
2. Themes by mode: which themes each mode's papers engage with
   fig_qual_themes_dots     share of each mode's papers per theme, one row per theme, a
                            dot per mode with 95% Wilson intervals
   fig_qual_themes_bars     the same as small multiples, one column per mode
   fig_qual_themes_heatmap  the same as a table of shades with counts
   fig_qual_themes_fields   as fig_qual_themes_dots, by field instead of mode
3. Stance: where papers agree and disagree
   fig_qual_stance_modes    stance mix of all code applications, per mode
   fig_qual_stance_themes   stance mix per theme and mode, diverging between contesting
                            (left) and endorsing (right)
   fig_qual_stance_themes_fields   the same per theme and field (AI ethics, AI safety)
   fig_qual_stance_codes    the codes some papers contest or qualify, papers per stance,
                            one column per mode
4. fig_qual_themes_overview  for the paper: fig_qual_themes_dots (left) next to
                            fig_qual_stance_themes_fields (right), on one theme axis
5. fig_qual_codes_heatmap   share of each mode's papers per code, codes grouped by theme
   tab_qual_codes.tex       the same as a LaTeX table

A paper engages with a theme if it has a code of that theme (--theme-source codes), or if
the spans stage recorded spans of the theme in it (--theme-source spans).
"""

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from matplotlib.patches import Patch

from aise import plot_modes as P
from aise import qualitative as Q

MODES = P.RARE_MODES
MODE_ABBREV = dict(zip(MODES, ["CC", "RC", "CB"], strict=True))
# the mode colours of Figure 1 are hard to tell apart for protanopes (orange, green),
# so modes also differ in marker shape and are labelled directly
MODE_MARKERS = dict(zip(MODES, ["o", "s", "D"], strict=True))
STANCES = ["contests", "qualifies", "describes", "endorses"]
# diverging: warm for contesting, cool for endorsing, neutral gray between
STANCE_COLORS = dict(zip(STANCES, ["#9e3a24", "#e07650", "#cfcec8", "#2c6e9e"],
                         strict=True))
STANCE_INK = dict(zip(STANCES, ["#ffffff", P.INK, P.INK, "#ffffff"], strict=True))
HEAT_RAMP = "Blues"  # single-hue sequential

app = typer.Typer(help=__doc__.split("\n\n")[0])


# ---- data ----------------------------------------------------------------------------
class Data:
    """Code applications, paper-theme incidence and paper counts for one run."""

    def __init__(self, out: Path, theme_source: str):
        self.out = out
        self.assignments = pd.read_csv(out / "assignments.csv")
        self.themes = json.loads((out / "themes.json").read_text())["themes"]
        keys = [c["key"] for c in Q.read_jsonl(out / "codings_final.jsonl")]
        meta = pd.read_csv(Q.E.out_dir() / "engagement.csv", index_col=0)
        self.papers = meta.loc[keys, ["category", "Retrieval"]].rename(
            columns={"category": "mode", "Retrieval": "field"})
        self.n_mode = self.papers["mode"].value_counts().reindex(MODES, fill_value=0)
        fields = self.papers["field"].value_counts()
        self.n_field = fields.reindex([f for f in P.FIELD_COLORS if f in fields.index])

        if theme_source == "spans":
            spans = pd.read_csv(out / "theme_spans.csv")
            incidence = spans[["key", "theme"]]
        else:
            incidence = self.assignments[["key", "theme"]]
        incidence = incidence.dropna().drop_duplicates()
        self.incidence = incidence.join(self.papers, on="key")
        # themes ordered by the number of papers engaging with them, most first
        counts = self.incidence["theme"].value_counts()
        self.theme_order = [t["name"] for t in self.themes]
        self.theme_order.sort(key=lambda t: -counts.get(t, 0))

    def theme_table(self, by: str = "mode") -> tuple[pd.DataFrame, pd.Series]:
        """Papers per theme (rows, in theme order) and group, and the papers per group."""
        n = self.n_mode if by == "mode" else self.n_field
        table = pd.crosstab(self.incidence["theme"], self.incidence[by])
        return table.reindex(index=self.theme_order, columns=n.index, fill_value=0), n


def _wrap(text: str, width: int = 44) -> str:
    return "\n".join(textwrap.wrap(text, width, break_long_words=False))


def _style(ax, grid_axis: str = "x"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(P.INK_3)
    ax.tick_params(colors=P.INK_2, labelcolor=P.INK, length=0)
    ax.grid(axis=grid_axis, color="#e6e5e0", lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def _percent(ax, axis: str = "x"):
    fmt = plt.FuncFormatter(lambda v, _: f"{v:.0%}")
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)


def _mode_label(mode: str, n: int) -> str:
    return f"{mode} (n={n})"


# ---- 1: codebook growth ----------------------------------------------------------------
def fig_growth(d: Data, out_dir: Path):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    _style(ax, "y")
    ends = []
    for stage, color, ls in ((1, P.INK, "-"), (2, P.INK_2, (0, (4, 2)))):
        rows = Q.read_jsonl(d.out / f"stage{stage}" / "codings.jsonl")
        if not rows:
            continue
        x = np.arange(1, len(rows) + 1)
        y = [r["codebook_size"] for r in rows]
        ax.plot(x, y, color=color, lw=2, ls=ls, zorder=3)
        ends.append((x[-1], y[-1], f"Stage {stage}: {y[-1]} codes"))
    final = len(json.loads((d.out / "codebook_final.json").read_text())["codes"])
    ax.axhline(final, color=P.INK_3, lw=1, ls=(0, (1, 2)), zorder=2)
    for x, y, label in ends:
        ax.annotate(label, (x, y), xytext=(-2, 5), textcoords="offset points",
                    ha="right", va="bottom", fontsize=8.5, color=P.INK)
    # at the left, under its line, where the stage-1 curve is still low
    ax.annotate(f"Fixed codebook: {final} codes", (0, final),
                xycoords=("axes fraction", "data"), xytext=(4, -4),
                textcoords="offset points", ha="left", va="top", fontsize=8.5,
                color=P.INK)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.set_xlabel("Papers coded (in each stage's random order)", color=P.INK_2)
    ax.set_ylabel("Codes in the codebook", color=P.INK_2)
    ax.set_title("Codebook growth during coding", loc="left", fontsize=11, color=P.INK)
    P._save(fig, "fig_qual_growth", out_dir)


# ---- 2: themes by mode -----------------------------------------------------------------
def _dots(d: Data, out_dir: Path, by: str, name: str):
    table, n = d.theme_table(by)
    groups = list(n.index)
    colors = P.MODE_COLORS if by == "mode" else P.FIELD_COLORS
    markers = MODE_MARKERS if by == "mode" else dict(zip(groups, ["o", "s"], strict=False))
    labels = ({g: _mode_label(g, n[g]) for g in groups} if by == "mode"
              else {g: f"{P.FIELD_NAMES[g]} (n={n[g]})" for g in groups})
    rows = len(table)
    fig, ax = plt.subplots(figsize=(7.2, 0.62 * rows + 1.2))
    _style(ax)
    offsets = np.linspace(-0.22, 0.22, len(groups)) if len(groups) > 1 else [0]
    for g, off in zip(groups, offsets, strict=True):
        k = table[g].to_numpy()
        p = k / max(n[g], 1)
        lo, hi = P.wilson(k, np.full(rows, n[g]))
        y = np.arange(rows) + off
        ax.hlines(y, lo, hi, color=colors[g], lw=1.2, alpha=0.55, zorder=2)
        ax.scatter(p, y, s=34, marker=markers[g], color=colors[g], edgecolor=P.SURFACE,
                   lw=1.2, zorder=3, label=labels[g])
    ax.set_yticks(np.arange(rows), [_wrap(t) for t in table.index], fontsize=8.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_xlim(0, 1)
    _percent(ax)
    what = "mode" if by == "mode" else "field"
    ax.set_xlabel(f"Share of the {what}'s papers engaging with the theme (95% CI)",
                  color=P.INK_2)
    ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=len(groups),
              frameon=False, fontsize=8.5, handletextpad=0.3, columnspacing=1.2)
    P._save(fig, name, out_dir)


def fig_themes_bars(d: Data, out_dir: Path):
    table, n = d.theme_table()
    rows = len(table)
    fig, axes = plt.subplots(1, len(MODES), figsize=(9.6, 0.5 * rows + 1.2), sharey=True)
    for ax, mode in zip(axes, MODES, strict=True):
        _style(ax)
        k = table[mode].to_numpy()
        p = k / max(n[mode], 1)
        lo, hi = P.wilson(k, np.full(rows, n[mode]))
        y = np.arange(rows)
        ax.barh(y, p, height=0.68, color=P.MODE_COLORS[mode], zorder=2)
        ax.errorbar(p, y, xerr=[p - lo, hi - p], fmt="none", ecolor=P.INK_2, lw=0.9,
                    capsize=0, zorder=3)
        ax.set_xlim(0, 1)
        _percent(ax)
        ax.set_title(_wrap(_mode_label(mode, n[mode]), 24), fontsize=9.5, color=P.INK,
                     loc="left")
    axes[0].set_yticks(np.arange(rows), [_wrap(t) for t in table.index], fontsize=8.5)
    axes[0].set_ylim(rows - 0.5, -0.5)
    fig.supxlabel("Share of the mode's papers engaging with the theme (95% CI)",
                  fontsize=9.5, color=P.INK_2)
    fig.tight_layout()
    P._save(fig, "fig_qual_themes_bars", out_dir)


def _heatmap(ax, shares: pd.DataFrame, counts: pd.DataFrame):
    im = ax.imshow(shares.to_numpy(float), cmap=HEAT_RAMP, vmin=0, vmax=1, aspect="auto")
    for (i, j), p in np.ndenumerate(shares.to_numpy(float)):
        k = counts.iat[i, j]
        ax.text(j, i, f"{p:.0%}" if k else "–", ha="center", va="center", fontsize=8,
                color="#ffffff" if p > 0.55 else P.INK)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0, labelcolor=P.INK)
    # 2px surface gaps between cells
    ax.set_xticks(np.arange(-0.5, shares.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, shares.shape[0]), minor=True)
    ax.grid(which="minor", color=P.SURFACE, lw=2)
    ax.tick_params(which="minor", length=0)
    return im


def fig_themes_heatmap(d: Data, out_dir: Path):
    table, n = d.theme_table()
    shares = table / n.replace(0, np.nan)
    fig, ax = plt.subplots(figsize=(6.6, 0.48 * len(table) + 1.4))
    im = _heatmap(ax, shares, table)
    ax.set_xticks(range(len(MODES)), [_wrap(_mode_label(m, n[m]), 18) for m in MODES],
                  fontsize=8.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(table)), [_wrap(t) for t in table.index], fontsize=8.5)
    bar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, format=plt.FuncFormatter(
        lambda v, _: f"{v:.0%}"))
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=8, labelcolor=P.INK_2)
    bar.set_label("Share of the mode's papers", fontsize=8.5, color=P.INK_2)
    P._save(fig, "fig_qual_themes_heatmap", out_dir)


# ---- 3: stance -------------------------------------------------------------------------
def _stance_bars(ax, y, counts: pd.DataFrame, diverging: bool, height=0.7, label_min=0.07):
    """Horizontal 100% stance bars, one per row of counts (columns are STANCES).
    Diverging puts contests and qualifies left of zero, describes and endorses right."""
    totals = counts.sum(axis=1).to_numpy(float)
    shares = counts.div(np.where(totals == 0, np.nan, totals), axis=0).fillna(0)
    left = (-(shares["contests"] + shares["qualifies"])).to_numpy() if diverging \
        else np.zeros(len(shares))
    for stance in STANCES:
        w = shares[stance].to_numpy()
        ax.barh(y, w, left=left, height=height, color=STANCE_COLORS[stance],
                edgecolor=P.SURFACE, lw=1.5, zorder=2)
        for yi, li, wi in zip(y, left, w, strict=True):
            if wi >= label_min:
                ax.text(li + wi / 2, yi, f"{wi:.0%}", ha="center", va="center",
                        fontsize=7.5, color=STANCE_INK[stance], zorder=3)
        left = left + w
    return totals


def _stance_legend(fig, y=1.0, x=0.5):
    handles = [Patch(color=STANCE_COLORS[s], label=s.capitalize()) for s in STANCES]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(x, y), ncol=4,
               frameon=False, fontsize=8.5, handlelength=1.2, columnspacing=1.2)


def fig_stance_modes(d: Data, out_dir: Path):
    a = d.assignments
    counts = pd.crosstab(a["mode"], a["stance"]).reindex(index=MODES, columns=STANCES,
                                                         fill_value=0)
    counts.loc["All rare modes"] = counts.sum()
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    _style(ax)
    y = np.arange(len(counts))
    totals = _stance_bars(ax, y, counts, diverging=False)
    labels = [_mode_label(m, d.n_mode[m]) for m in MODES] + ["All rare modes"]
    ax.set_yticks(y, labels, fontsize=8.5)
    ax.set_ylim(len(counts) - 0.5, -0.6)
    for yi, t in zip(y, totals, strict=True):
        ax.text(1.01, yi, f"{int(t)}", va="center", fontsize=8, color=P.INK_2)
    ax.text(1.01, -0.75, "codes\napplied", fontsize=7.5, color=P.INK_3, va="bottom")
    ax.set_xlim(0, 1)
    _percent(ax)
    ax.set_xlabel("Share of code applications", color=P.INK_2)
    _stance_legend(fig, y=0.98)
    fig.tight_layout()
    P._save(fig, "fig_qual_stance_modes", out_dir)


def fig_stance_themes(d: Data, out_dir: Path, by: str = "mode"):
    """Stance mix per theme, one bar per mode (by="mode") or field (by="field")."""
    if by == "mode":
        groups, abbrev, colors = MODES, MODE_ABBREV, P.MODE_COLORS
        names = {m: m for m in MODES}
        name = "fig_qual_stance_themes"
    else:
        groups = list(d.n_field.index)
        abbrev, colors, names = P.FIELD_ABBREV, P.FIELD_COLORS, P.FIELD_NAMES
        name = "fig_qual_stance_themes_fields"
    a = d.assignments.dropna(subset=["theme"])
    counts = pd.crosstab([a["theme"], a[by]], a["stance"]).reindex(
        index=pd.MultiIndex.from_product([d.theme_order, groups]), columns=STANCES,
        fill_value=0)
    gap = 1.5  # extra space between themes, for their titles
    y = np.array([i * len(groups) + j + i * gap for i in range(len(d.theme_order))
                  for j in range(len(groups))], float)
    fig, ax = plt.subplots(figsize=(8.2, 0.21 * len(y) + 1.4))
    _style(ax)
    totals = _stance_bars(ax, y, counts, diverging=True, height=0.78, label_min=0.12)
    ax.axvline(0, color=P.INK_2, lw=1, zorder=3)
    ax.set_yticks(y, [abbrev[g] for _, g in counts.index], fontsize=7.5)
    for tick, (_, g) in zip(ax.get_yticklabels(), counts.index, strict=True):
        tick.set_color(colors[g])
    for yi, t in zip(y, totals, strict=True):
        ax.text(1.02, yi, f"{int(t)}" if t else "–", va="center", fontsize=7,
                color=P.INK_3)
    for i, theme in enumerate(d.theme_order):
        top = y[i * len(groups)]
        ax.text(-1.0, top - 0.75, _wrap(theme, 90), fontsize=8.5, color=P.INK,
                fontweight="bold", va="bottom")
    ax.set_xlim(-1, 1.1)
    ax.set_ylim(y[-1] + 0.6, -1.9)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1], ["100%", "50%", "0", "50%", "100%"])
    ax.set_xlabel("← contests or qualifies      Share of the theme's code applications"
                  "      describes or endorses →", color=P.INK_2, fontsize=8.5)
    key = "   ".join(f"{abbrev[g]} = {names[g]}" for g in groups)
    fig.text(0.5, 0.0, f"{key};  numbers at right: codes applied", ha="center",
             va="top", fontsize=7.5, color=P.INK_2)
    fig.tight_layout(rect=(0, 0.01, 1, 0.97))
    _stance_legend(fig, y=0.965)
    P._save(fig, name, out_dir)


def fig_themes_overview(d: Data, out_dir: Path):
    """For the paper: per theme, the share of each mode's papers engaging with it (left)
    and the stance of each field's code applications (right), on one shared theme axis."""
    table, n = d.theme_table()
    rows = len(table)
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(10.4, 0.62 * rows + 1.4), sharey=True,
        gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.1})
    y = np.arange(rows)

    _style(left)
    for mode, off in zip(MODES, (-0.25, 0, 0.25), strict=True):
        k = table[mode].to_numpy()
        lo, hi = P.wilson(k, np.full(rows, n[mode]))
        left.hlines(y + off, lo, hi, color=P.MODE_COLORS[mode], lw=1.6, alpha=0.6,
                    zorder=2)
        left.scatter(k / n[mode], y + off, s=40, marker=MODE_MARKERS[mode],
                     color=P.MODE_COLORS[mode], edgecolor=P.SURFACE, lw=1, zorder=3,
                     label=_mode_label(mode, n[mode]))
    left.set_yticks(y, [_wrap(t, 30) for t in table.index], fontsize=11.5)
    left.set_ylim(rows - 0.5, -0.5)
    left.set_xlim(0, 1)
    # no label at 100%, which would run into the right panel's first tick label
    left.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1], ["0%", "20%", "40%", "60%", "80%", ""])
    left.set_xlabel("Share of papers", color=P.INK_2, fontsize=13,
                    fontweight="bold")
    left.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False,
                fontsize=12, handletextpad=0.2, columnspacing=0.9, borderaxespad=0.2,
                markerscale=1.2)

    _style(right)
    a = d.assignments.dropna(subset=["theme"])
    fields = list(d.n_field.index)
    counts = pd.crosstab([a["theme"], a["field"]], a["stance"]).reindex(
        index=pd.MultiIndex.from_product([table.index, fields]), columns=STANCES,
        fill_value=0)
    offsets = dict(zip(fields, (-0.19, 0.19), strict=False))
    ys = np.array([i + offsets[f] for i in range(rows) for f in fields])
    totals = _stance_bars(right, ys, counts, diverging=True, height=0.34, label_min=1.1)
    right.axvline(0, color=P.INK_2, lw=1, zorder=3)
    for yi, (_, f), t in zip(ys, counts.index, totals, strict=True):
        right.text(-0.99, yi, P.FIELD_ABBREV[f], va="center", fontsize=10.5,
                   color=P.FIELD_COLORS[f])
        right.text(1.01, yi, f"{int(t)}", va="center", fontsize=10.5, color=P.INK_2)
    right.set_xlim(-1, 1.1)
    right.set_xticks([-1, -0.5, 0, 0.5, 1], ["100%", "50%", "0", "50%", "100%"])
    right.set_xlabel("Share of code applications", color=P.INK_2, fontsize=13,
                     fontweight="bold")
    handles = [Patch(color=STANCE_COLORS[s], label=s.capitalize()) for s in STANCES]
    right.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
                 frameon=False, fontsize=12, handlelength=1.2, columnspacing=1.0,
                 borderaxespad=0.2)
    for ax in (left, right):
        ax.tick_params(axis="x", labelsize=12)
    # start the mode legend where the theme labels start, so that it adds no white
    # space on the left
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    x0 = min(t.get_window_extent(renderer).x0 for t in left.get_yticklabels())
    x0 = left.transAxes.inverted().transform((x0, 0))[0]
    mode_legend = left.get_legend()
    mode_legend.set_loc("lower left")
    mode_legend.set_bbox_to_anchor((x0, 1.0), transform=left.transAxes)
    P._save(fig, "fig_qual_themes_overview", out_dir)


def fig_stance_codes(d: Data, out_dir: Path, max_codes: int):
    a = d.assignments
    papers = (a.groupby(["code", "mode", "stance"])["key"].nunique()
              .unstack("stance").reindex(columns=STANCES, fill_value=0).fillna(0))
    pooled = papers.groupby("code").sum()
    against = pooled["contests"] + pooled["qualifies"]
    codes = against[against > 0].index
    order = (pd.DataFrame({"against": against[codes], "all": pooled.loc[codes].sum(axis=1)})
             .sort_values(["against", "all"], ascending=False).index[:max_codes])
    if not len(order):
        typer.secho("  no contested or qualified codes; skipped fig_qual_stance_codes")
        return
    rows = len(order)
    fig, axes = plt.subplots(1, len(MODES), figsize=(10, 0.3 * rows + 1.4), sharey=True)
    shown = papers.loc[papers.index.get_level_values("code").isin(order)]
    lim_left = max(1, (shown["contests"] + shown["qualifies"]).max())
    lim_right = max(1, (shown["describes"] + shown["endorses"]).max())
    for ax, mode in zip(axes, MODES, strict=True):
        _style(ax)
        m = papers.xs(mode, level="mode") if mode in papers.index.get_level_values(
            "mode") else pd.DataFrame(columns=STANCES)
        m = m.reindex(index=order, fill_value=0)
        y = np.arange(rows)
        left = -(m["contests"] + m["qualifies"]).to_numpy(float)
        for stance in STANCES:
            w = m[stance].to_numpy(float)
            ax.barh(y, w, left=left, height=0.72, color=STANCE_COLORS[stance],
                    edgecolor=P.SURFACE, lw=1.2, zorder=2)
            left = left + w
        ax.axvline(0, color=P.INK_2, lw=1, zorder=3)
        ax.set_xlim(-lim_left - 0.5, lim_right + 0.5)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=5))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{abs(int(v))}"))
        ax.set_title(_wrap(_mode_label(mode, d.n_mode[mode]), 26), fontsize=9.5,
                     color=P.INK, loc="left")
    axes[0].set_yticks(np.arange(rows), [_wrap(c, 48) for c in order], fontsize=8)
    axes[0].set_ylim(rows - 0.5, -0.5)
    fig.supxlabel("Papers: ← contesting or qualifying | describing or endorsing →",
                  fontsize=9, color=P.INK_2)
    fig.tight_layout()
    _stance_legend(fig, y=0.995)
    P._save(fig, "fig_qual_stance_codes", out_dir)


# ---- 4: codes by mode ------------------------------------------------------------------
def _codes_by_mode(d: Data, min_papers: int) -> tuple[pd.DataFrame, pd.Series]:
    """Papers per code (rows, grouped by theme) and mode, and the theme of each code."""
    a = d.assignments.drop_duplicates(["key", "code"])
    counts = pd.crosstab(a["code"], a["mode"]).reindex(columns=MODES, fill_value=0)
    counts = counts[counts.sum(axis=1) >= min_papers]
    theme_of = a.drop_duplicates("code").set_index("code")["theme"]
    rank = {t: i for i, t in enumerate(d.theme_order)}
    order = sorted(counts.index, key=lambda c: (rank.get(theme_of[c], 99),
                                                -counts.loc[c].sum()))
    return counts.loc[order], theme_of


def fig_codes_heatmap(d: Data, out_dir: Path, min_papers: int):
    counts, theme_of = _codes_by_mode(d, min_papers)
    order = list(counts.index)
    if not order:
        typer.secho(f"  no code with {min_papers}+ papers; skipped fig_qual_codes_heatmap")
        return
    shares = counts / d.n_mode.replace(0, np.nan)
    fig, ax = plt.subplots(figsize=(7.6, 0.27 * len(order) + 1.6))
    im = _heatmap(ax, shares, counts)
    ax.set_xticks(range(len(MODES)), [MODE_ABBREV[m] for m in MODES], fontsize=8.5)
    for tick, m in zip(ax.get_xticklabels(), MODES, strict=True):
        tick.set_color(P.MODE_COLORS[m])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(order)), [_wrap(c, 60) for c in order], fontsize=7.5)
    # theme brackets left of the code labels
    themes = [theme_of[c] for c in order]
    start = 0
    for i in range(1, len(themes) + 1):
        if i == len(themes) or themes[i] != themes[start]:
            if i < len(themes):
                ax.axhline(i - 0.5, color=P.INK_2, lw=1.2)
            ax.annotate(_wrap(themes[start], 28), xy=(1.04, (start + i - 1) / 2),
                        xycoords=("axes fraction", "data"), fontsize=7.5, color=P.INK,
                        va="center", ha="left")
            start = i
    bar = fig.colorbar(im, ax=ax, fraction=0.6 / len(order), pad=0.01,
                       location="bottom", aspect=30,
                       format=plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=7.5, labelcolor=P.INK_2)
    bar.set_label("Share of the mode's papers", fontsize=8, color=P.INK_2)
    abbrev = "   ".join(f"{v} = {k} (n={d.n_mode[k]})" for k, v in MODE_ABBREV.items())
    ax.set_title(abbrev, fontsize=7.5, color=P.INK_2, pad=22)
    P._save(fig, "fig_qual_codes_heatmap", out_dir)


def _tex(text: str) -> str:
    for char, repl in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                       ("_", r"\_"), ("#", r"\#"), ("$", r"\$"), ("\u2013", "--"),
                       ("\u2014", "---")):
        text = text.replace(char, repl)
    return text


def tab_codes(d: Data, out_dir: Path, min_papers: int):
    r"""fig_qual_codes_heatmap as a LaTeX longtable, cells shaded by share.

    Needs \usepackage{array,booktabs,longtable} and \usepackage[table]{xcolor}.
    """
    counts, theme_of = _codes_by_mode(d, min_papers)
    if counts.empty:
        typer.secho(f"  no code with {min_papers}+ papers; skipped tab_qual_codes")
        return
    shares = counts / d.n_mode.replace(0, np.nan)
    abbrevs = [MODE_ABBREV[m] for m in MODES]
    legend = "; ".join(f"{MODE_ABBREV[m]} = {_tex(m.lower())} ($n={d.n_mode[m]}$)"
                       for m in MODES)
    caption = (
        "Share of each mode's papers to which a code was applied, for codes applied to "
        f"at least {min_papers} papers, grouped by theme (themes ordered by the number "
        "of papers engaging with them, codes within a theme by their number of papers). "
        f"{legend}. Darker cells mark larger shares; -- marks no paper.")
    mode_cols = " & ".join("\\multicolumn{1}{c}{" + a + "}" for a in abbrevs)
    header = (f"\\toprule\n\\multicolumn{{1}}{{@{{}}l}}{{Code}} & {mode_cols} \\\\\n"
              "\\midrule\n")
    ncol = 1 + len(MODES)
    num_col = ">{\\raggedleft\\arraybackslash}p{\\qualnum}"
    theme_col = "@{}>{\\raggedright\\arraybackslash}p{\\qualwidth}@{}"
    lines = [
        "% Generated by aise.plot_qualitative; do not edit by hand.",
        "% Needs \\usepackage{array,booktabs,longtable} and \\usepackage[table]{xcolor}.",
        "\\definecolor{qualheat}{HTML}{2c6e9e}",
        "\\begingroup",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\renewcommand{\\arraystretch}{1.1}",
        # code labels ragged right, indented under their theme also when they wrap
        # the table spans \\linewidth: fixed-width share columns, the code column gets
        # the rest (computed here: inside a cell, \\linewidth is the cell's width)
        "\\def\\qualnum{3em}",
        "\\edef\\qualwidth{\\the\\linewidth}",
        "\\setlength{\\LTleft}{0pt}\\setlength{\\LTright}{0pt}",
        "\\setlength{\\LTcapwidth}{\\linewidth}",
        (f"\\edef\\qualcode{{\\the\\dimexpr\\linewidth-\\qualnum*{len(MODES)}"
         f"-\\tabcolsep*{2 * len(MODES)}\\relax}}"),
        ("\\begin{longtable}{@{}>{\\raggedright\\arraybackslash\\hangindent=1em"
         f"\\hangafter=0}}p{{\\qualcode}}{num_col * len(MODES)}@{{}}}}"),
        f"\\caption{{{caption}}}",
        "\\label{tab:qual-codes}\\\\",
        header + "\\endfirsthead",
        (f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\tablename\\ \\thetable{{}} "
         "(continued)}\\\\"),
        header + "\\endhead",
        (f"\\midrule\n\\multicolumn{{{ncol}}}{{r@{{}}}}{{\\emph{{Continued on next "
         "page}}\\\\\n\\endfoot"),
        "\\bottomrule\n\\endlastfoot",
    ]
    theme = None
    for code in counts.index:
        if theme_of[code] != theme:
            if theme is not None:
                lines.append("\\addlinespace[3pt]")
            theme = theme_of[code]
            # wraps if it is too long for the line
            lines.append(f"\\multicolumn{{{ncol}}}{{{theme_col}}}"
                         f"{{\\textbf{{{_tex(theme)}}}}}\\\\*")
        cells = []
        for m in MODES:
            k, p = counts.loc[code, m], shares.loc[code, m]
            # shade at 70% strength at most, so black text stays legible
            cells.append(f"\\cellcolor{{qualheat!{round(70 * p)}}}{p:.0%}".replace(
                "%", "\\%") if k else "--")
        lines.append(f"{_tex(code)} & {' & '.join(cells)} \\\\")
    lines += ["\\end{longtable}", "\\endgroup", ""]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tab_qual_codes.tex").write_text("\n".join(lines))
    typer.secho(f"  wrote {out_dir / 'tab_qual_codes.tex'}")


@app.command()
def figures(
    theme_source: str = typer.Option(
        "codes", help="Theme incidence from the coded codes ('codes') or the spans stage "
        "('spans')."),
    max_codes: int = typer.Option(30, help="Most rows in fig_qual_stance_codes."),
    min_papers: int = typer.Option(3, help="Fewest papers for a code in "
                                   "fig_qual_codes_heatmap."),
    run_dir: Path = typer.Option(None, help="Default: output/qualitative/<v>."),
    out_dir: Path = typer.Option(None, help="Default: <run-dir>/plots."),
):
    """Draw all figures."""
    if theme_source not in ("codes", "spans"):
        raise typer.BadParameter("theme-source is codes or spans")
    run_dir = run_dir or Q.OUT
    d = Data(run_dir, theme_source)
    out_dir = out_dir or run_dir / "plots"
    typer.secho(f"{len(d.papers)} papers, {len(d.assignments)} code applications, "
                f"{len(d.theme_order)} themes")
    fig_growth(d, out_dir)
    _dots(d, out_dir, "mode", "fig_qual_themes_dots")
    fig_themes_bars(d, out_dir)
    fig_themes_heatmap(d, out_dir)
    _dots(d, out_dir, "field", "fig_qual_themes_fields")
    fig_stance_modes(d, out_dir)
    fig_stance_themes(d, out_dir)
    fig_stance_themes(d, out_dir, by="field")
    fig_stance_codes(d, out_dir, max_codes)
    fig_themes_overview(d, out_dir)
    fig_codes_heatmap(d, out_dir, min_papers)
    tab_codes(d, out_dir, min_papers)


if __name__ == "__main__":
    app()
