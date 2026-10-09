"""Human annotation of a paper subsample, for agreement with the LLM classifier.

    uv run python -m aise.annotate sample             # draw the subsample (once)
    uv run python -m aise.annotate serve              # annotate in the browser
    uv run python -m aise.annotate agreement          # print the agreement metrics

The subsample is stratified by the classifier's mode (the same number of papers from
each of the four modes) plus a small fully random subsample; all are full-text papers.
The page shows papers in a shuffled order and never reveals a paper's classification or
stratum before your judgement is saved. Your first saved judgement of a paper is kept as
the blind one, which the metrics use; edits made after seeing the classifier's answer are
recorded separately. Everything lives in output/annotation/<PROMPT_VERSION>/.
"""

import json
import re
import threading
import webbrowser
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd
import typer
from scipy import stats

from aise import engagement as E
from aise import plot_modes as P

ROOT = Path("output/annotation") / E.PROMPT_VERSION
SAMPLE = ROOT / "sample.csv"
ANNOTATIONS = ROOT / "annotations.json"
HTML = Path(__file__).with_name("annotate.html")
AXES = ("engagement", "integration")

app = typer.Typer(help=__doc__.split("\n\n")[0])


# ---- data ----------------------------------------------------------------------------
def classified() -> pd.DataFrame:
    """The classifier's full-text results for the current prompt version."""
    df = P.load_results([E.out_dir() / "engagement.csv"])
    return df[df["input"] == "full text"]


def rubric() -> dict:
    """Each axis's description and level definitions, parsed from the classifier prompt
    so the annotator works from exactly the same scheme."""
    out = {}
    for axis, (start, end) in {"engagement": ("## Axis 1", "## Axis 2"),
                               "integration": ("## Axis 2", "Score the two axes")}.items():
        block = E.PROMPT.split(start, 1)[1].split(end, 1)[0]
        levels = {int(m[1]): m[2] for m in re.finditer(r"^- (\d): (.+)$", block, re.M)}
        intro = re.sub(r"^.*\n", "", block, count=1)  # drop the heading line
        intro = re.split(r"^- \d:", intro, flags=re.M)[0].strip()
        out[axis] = {"intro": intro, "levels": levels}
    out["note"] = re.search(r"Score the two axes.*?(?=\n\n)", E.PROMPT, re.S)[0]
    return out


def load_annotations() -> dict:
    return json.loads(ANNOTATIONS.read_text()) if ANNOTATIONS.exists() else {}


_lock = threading.Lock()


def save_annotation(key: str, data: dict) -> dict:
    """Store a judgement. The first save is the blind one and is never overwritten."""
    with _lock:
        ann = load_annotations()
        now = datetime.now(UTC).isoformat(timespec="seconds")
        fields = {a: int(data[a]) for a in AXES}
        fields["other_field_referenced"] = bool(data.get("other_field_referenced"))
        entry = ann.get(key) or {"created": now}
        if not entry.get("revealed"):  # still blind: edits update the blind judgement
            entry["blind"] = fields
        elif fields != entry["blind"]:
            entry["changed_after_reveal"] = True
        entry.update(current=fields, unsure=bool(data.get("unsure")),
                     notes=str(data.get("notes", "")), updated=now)
        if data.get("reveal"):
            entry["revealed"] = True
        ann[key] = entry
        ROOT.mkdir(parents=True, exist_ok=True)
        tmp = ANNOTATIONS.with_suffix(".tmp")
        tmp.write_text(json.dumps(ann, indent=1))
        tmp.replace(ANNOTATIONS)
        return entry


# ---- agreement ------------------------------------------------------------------------
def kappa(a: np.ndarray, b: np.ndarray, levels, weights: str = "none",
          w: np.ndarray | None = None) -> float:
    """Cohen's kappa of two raters, optionally linear/quadratic weighted and with
    sample weights (for estimates reweighted to the corpus)."""
    k = len(levels)
    pos = {v: i for i, v in enumerate(levels)}
    obs = np.zeros((k, k))
    np.add.at(obs, ([pos[x] for x in a], [pos[x] for x in b]),
              np.ones(len(a)) if w is None else w)
    obs /= obs.sum()
    exp = np.outer(obs.sum(1), obs.sum(0))
    i, j = np.indices((k, k))
    dis = {"none": (i != j).astype(float), "linear": np.abs(i - j) / (k - 1),
           "quadratic": ((i - j) / (k - 1)) ** 2}[weights]
    denom = (dis * exp).sum()
    return float(1 - (dis * obs).sum() / denom) if denom > 0 else float("nan")


def _metrics(h: pd.DataFrame, w: np.ndarray | None) -> dict:
    """Point estimates for one set of paired judgements (human h_*, classifier l_*)."""
    wt = np.ones(len(h)) if w is None else w
    mean = lambda x: float(np.average(x, weights=wt))  # noqa: E731
    out = {}
    for axis in AXES:
        a, b = h[f"h_{axis}"].to_numpy(), h[f"l_{axis}"].to_numpy()
        out[f"{axis}: exact agreement"] = mean(a == b)
        out[f"{axis}: within one level"] = mean(np.abs(a - b) <= 1)
        out[f"{axis}: Cohen's kappa"] = kappa(a, b, E.LEVELS, w=w)
        out[f"{axis}: linear-weighted kappa"] = kappa(a, b, E.LEVELS, "linear", w)
        out[f"{axis}: quadratic-weighted kappa"] = kappa(a, b, E.LEVELS, "quadratic", w)
        out[f"{axis}: mean difference (classifier - human)"] = mean(b - a)
        high = {"engagement": E.ENGAGEMENT_HIGH, "integration": E.INTEGRATION_HIGH}[axis]
        ah, bh = a >= high, b >= high
        out[f"{axis} high/low: agreement"] = mean(ah == bh)
        out[f"{axis} high/low: kappa"] = kappa(ah, bh, (False, True), w=w)
    a, b = h["h_mode"].to_numpy(), h["l_mode"].to_numpy()
    out["mode: agreement"] = mean(a == b)
    out["mode: Cohen's kappa"] = kappa(a, b, E.CATEGORIES, w=w)
    a, b = h["h_ref"].to_numpy(), h["l_ref"].to_numpy()
    out["other field referenced: agreement"] = mean(a == b)
    out["other field referenced: kappa"] = kappa(a, b, (False, True), w=w)
    return out


def _with_ci(h: pd.DataFrame, weighted: bool, corpus: pd.Series, reps: int = 1000,
             seed: int = 0) -> dict:
    """Estimates with percentile bootstrap 95% intervals; resampling is within the
    classifier's modes, the strata the sample was drawn from."""
    def weights(d):
        if not weighted:
            return None
        # post-stratification: each mode counts in proportion to its corpus size
        n = d["l_mode"].map(d["l_mode"].value_counts())
        return (d["l_mode"].map(corpus) / n).to_numpy(float)

    point = _metrics(h, weights(h))
    rng = np.random.default_rng(seed)
    groups = [g.index.to_numpy() for _, g in h.groupby("l_mode")]
    boot = {k: [] for k in point}
    with np.errstate(invalid="ignore", divide="ignore"):
        for _ in range(reps):
            idx = np.concatenate([rng.choice(g, len(g)) for g in groups])
            d = h.loc[idx].reset_index(drop=True)
            for k, v in _metrics(d, weights(d)).items():
                boot[k].append(v)
    return {k: [point[k], *np.nanpercentile(boot[k], [2.5, 97.5]).tolist()]
            for k in point}


def agreement_report(reps: int = 1000) -> dict:
    sample = pd.read_csv(SAMPLE, index_col="key")
    ann = load_annotations()
    llm = classified()
    rows = []
    for key, entry in ann.items():
        if key not in sample.index:
            continue
        hb, lr = entry["blind"], llm.loc[key]
        rows.append({
            "key": key, "stratum": sample.loc[key, "stratum"],
            "h_engagement": hb["engagement"], "h_integration": hb["integration"],
            "h_mode": E.quadrant(hb["engagement"], hb["integration"]),
            "h_ref": bool(hb["other_field_referenced"]),
            "l_engagement": int(lr["engagement"]), "l_integration": int(lr["integration"]),
            "l_mode": lr["category"], "l_ref": bool(lr["other_field_referenced"]),
        })
    report = {"n_sample": len(sample), "n_annotated": len(rows),
              "changed_after_reveal": sum(bool(e.get("changed_after_reveal"))
                                          for e in ann.values()),
              "unsure": sum(bool(e.get("unsure")) for e in ann.values()),
              "levels": list(E.LEVELS), "modes": E.CATEGORIES, "sets": {}}
    if not rows:
        return report
    h = pd.DataFrame(rows)
    corpus = llm["category"].value_counts()
    sets = {
        "Annotated sample": (h, False),
        "Reweighted to the corpus": (h, True),
        "Random subsample only": (h[h["stratum"] == "random"].reset_index(drop=True),
                                  False),
    }
    for name, (d, weighted) in sets.items():
        if len(d) < 2:
            continue
        report["sets"][name] = {"n": len(d), "metrics": _with_ci(d, weighted, corpus, reps)}
    # mode confusion (rows: human, columns: classifier) and per-mode precision
    conf = pd.crosstab(h["h_mode"], h["l_mode"]).reindex(
        index=E.CATEGORIES, columns=E.CATEGORIES, fill_value=0)
    report["mode_confusion"] = conf.to_numpy().tolist()
    report["per_mode"] = {
        m: {"classifier_n": int((h["l_mode"] == m).sum()),
            "human_agrees": int(((h["l_mode"] == m) & (h["h_mode"] == m)).sum())}
        for m in E.CATEGORIES}
    for axis in AXES:
        report[f"{axis}_confusion"] = pd.crosstab(h[f"h_{axis}"], h[f"l_{axis}"]).reindex(
            index=E.LEVELS, columns=E.LEVELS, fill_value=0).to_numpy().tolist()
        rho = stats.spearmanr(h[f"h_{axis}"], h[f"l_{axis}"])
        report[f"{axis}_spearman"] = [float(rho.statistic), float(rho.pvalue)]
    return report


# ---- web server -------------------------------------------------------------------------
def _paper(key: str, sample: pd.DataFrame, meta: pd.DataFrame, llm: pd.DataFrame) -> dict:
    row = meta.loc[key]
    text = (E.TXT_DIR / f"{key}.txt").read_text(errors="ignore")
    entry = load_annotations().get(key)
    home = P.FIELD_NAMES[row["Retrieval"]]
    out = {"key": key, "title": row["Title"], "year": int(row["Publication_Year"]),
           "doi": row.get("DOI") if isinstance(row.get("DOI"), str) else None,
           "abstract": row["Abstract"] if isinstance(row["Abstract"], str) else "",
           "text": text[:E.MAX_CHARS], "truncated": len(text) > E.MAX_CHARS,
           "home": home, "other": "AI safety" if home == "AI ethics" else "AI ethics",
           "annotation": entry}
    if entry and entry.get("revealed"):  # the classifier's answer only after a judgement
        out["llm"] = _llm(key, llm)
    return out


def _llm(key: str, llm: pd.DataFrame) -> dict:
    r = llm.loc[key]
    ev = r["evidence"] if isinstance(r["evidence"], str) else ""
    quotes = [q.strip().strip("\"“”") for q in ev.split(" | ") if q.strip()]
    verified = set(P.verified_quotes(key, ev))
    return {"engagement": int(r["engagement"]), "integration": int(r["integration"]),
            "mode": r["category"], "rationale": r["rationale"],
            "other_field_referenced": bool(r["other_field_referenced"]),
            "evidence": [{"quote": q, "verbatim": q in verified} for q in quotes]}


def make_handler(sample: pd.DataFrame, meta: pd.DataFrame, llm: pd.DataFrame):
    order = sample.sort_values("order").index.tolist()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the terminal quiet
            pass

        def _send(self, body, status=200, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = unquote(self.path.split("?", 1)[0])
            if path == "/":
                self._send(HTML.read_bytes(), ctype="text/html; charset=utf-8")
            elif path == "/api/init":
                ann = load_annotations()
                self._send({"order": order, "done": [k for k in order if k in ann],
                            "titles": {k: meta.loc[k, "Title"] for k in order},
                            "rubric": rubric(), "levels": list(E.LEVELS),
                            "high": {"engagement": E.ENGAGEMENT_HIGH,
                                     "integration": E.INTEGRATION_HIGH},
                            "modes": E.CATEGORIES, "colors": P.MODE_COLORS,
                            "version": E.PROMPT_VERSION})
            elif path.startswith("/api/paper/"):
                key = path.rsplit("/", 1)[1]
                if key not in sample.index:
                    self._send({"error": "not in sample"}, 404)
                else:
                    self._send(_paper(key, sample, meta, llm))
            elif path == "/api/agreement":
                self._send(agreement_report())
            else:
                self._send({"error": "not found"}, 404)

        def do_POST(self):
            path = unquote(self.path)
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            key = path.rsplit("/", 1)[1]
            if not path.startswith("/api/annotate/") or key not in sample.index:
                self._send({"error": "not found"}, 404)
                return
            if any(data.get(a) not in E.LEVELS for a in AXES):
                self._send({"error": "both axes need a level"}, 400)
                return
            entry = save_annotation(key, data)
            self._send({"annotation": entry,
                        "llm": _llm(key, llm) if entry.get("revealed") else None})

    return Handler


# ---- commands ---------------------------------------------------------------------------
@app.command()
def sample(
    per_mode: int = typer.Option(20, help="Papers drawn from each classifier mode."),
    n_random: int = typer.Option(20, help="Further papers drawn fully at random."),
    seed: int = typer.Option(0),
    force: bool = typer.Option(False, help="Redraw even if annotations exist."),
):
    """Draw the stratified + random subsample."""
    if E.FIVE_LEVELS:
        raise typer.BadParameter("annotation uses the current 1-4 rubric; use v7 or later")
    if load_annotations() and not force:
        raise typer.BadParameter(f"{ANNOTATIONS} exists; pass --force to redraw anyway")
    llm = classified()
    rng = np.random.default_rng(seed)
    picks = []
    for mode in E.CATEGORIES:
        keys = llm.index[llm["category"] == mode].to_numpy()
        take = min(per_mode, len(keys))
        if take < per_mode:
            typer.secho(f"  only {take} {mode} papers available", fg="yellow")
        picks += [(k, mode) for k in rng.choice(keys, take, replace=False)]
    rest = llm.index.difference([k for k, _ in picks]).to_numpy()
    picks += [(k, "random") for k in rng.choice(rest, n_random, replace=False)]
    out = pd.DataFrame(picks, columns=["key", "stratum"])
    out["classifier_mode"] = llm.loc[out["key"], "category"].to_numpy()
    out["order"] = rng.permutation(len(out))  # shown shuffled, so strata stay hidden
    ROOT.mkdir(parents=True, exist_ok=True)
    out.to_csv(SAMPLE, index=False)
    typer.secho(f"Wrote {SAMPLE}: {len(out)} papers")
    typer.secho(out.groupby(["stratum", "classifier_mode"]).size().to_string())


@app.command()
def serve(port: int = typer.Option(8765), open_browser: bool = typer.Option(True)):
    """Serve the annotation page on localhost."""
    if not SAMPLE.exists():
        raise typer.BadParameter(f"no sample yet; run `python -m aise.annotate sample`")
    smp = pd.read_csv(SAMPLE, index_col="key")
    meta = E.load_papers()
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(smp, meta, classified()))
    url = f"http://127.0.0.1:{port}/"
    typer.secho(f"Annotating {len(smp)} papers at {url} (Ctrl+C to stop); "
                f"saving to {ANNOTATIONS}")
    if open_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


@app.command()
def agreement(reps: int = typer.Option(1000, help="Bootstrap resamples.")):
    """Print the agreement metrics and save them as JSON."""
    report = agreement_report(reps)
    (ROOT / "agreement.json").write_text(json.dumps(report, indent=1))
    typer.secho(f"{report['n_annotated']} of {report['n_sample']} papers annotated; "
                f"{report['changed_after_reveal']} changed after seeing the classifier")
    for name, s in report["sets"].items():
        typer.secho(f"\n{name} (n = {s['n']})", bold=True)
        for metric, (v, lo, hi) in s["metrics"].items():
            typer.secho(f"  {metric:52s} {v:6.3f}  [{lo:6.3f}, {hi:6.3f}]")


if __name__ == "__main__":
    app()
