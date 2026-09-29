"""Classify papers by how they engage the AI ethics / AI safety divide (Figure 1).

The model scores each paper on two axes, in one call:

* ``engagement``, an integer level from 1 to 5: how directly the paper engages the
  other field or the tensions between AI ethics (AIE) and AI safety (AIS);
* ``integration``, an integer level from 1 to 5: how far it brings AIE and AIS
  concerns into one frame.

The model is not told about the four modes of Figure 1. ``parse`` assigns each paper
its ``category`` (quadrant) from the two levels: high means engagement >= 4 and
integration >= 3.

Typical loop while iterating on the prompt::

    uv run python -m aise.engagement extract            # PDFs -> text (once)
    uv run python -m aise.engagement pilot --n 20       # quick synchronous run
    uv run python -m aise.engagement parse --pilot      # inspect results

Full run through the OpenAI Batch API::

    uv run python -m aise.engagement prepare
    uv run python -m aise.engagement submit
    uv run python -m aise.engagement fetch              # re-run until all complete
    uv run python -m aise.engagement parse

Needs ``OPENAI_API_KEY`` in the environment and ``pdftotext`` (poppler) on the path.
Outputs go to ``output/engagement/<PROMPT_VERSION>/`` so prompt iterations never
overwrite each other.
"""

import json
import os
import random
import re
import subprocess
from pathlib import Path

import pandas as pd
import typer
from openai import Client

PROMPT_VERSION = "v6"
MODEL = "gpt-6-luna"
REASONING_EFFORT = "medium"
MAX_COMPLETION_TOKENS = 8000  # includes reasoning tokens
MAX_CHARS = 60_000  # ~15k tokens of body text per paper
MIN_TEXT_CHARS = 1000  # less than this means extraction failed (e.g. scanned PDF)
MAX_SHARD_BYTES = 150_000_000  # OpenAI batch input files are capped at 200 MB
MAX_SHARD_REQUESTS = 1000
LEVELS = (1, 2, 3, 4, 5)  # both axes are scored on this integer scale
# lowest level that counts as "high" on each axis when computing the quadrant
ENGAGEMENT_HIGH = 4  # level 3 is one substantive passage, not a thread
INTEGRATION_HIGH = 3  # level 3 is both fields' concerns side by side

PAPERS = Path("data/papers_all_fields.csv")
PDF_DIR = Path("pdf")
TXT_DIR = Path("output/engagement/txt")
OUT_ROOT = Path("output/engagement")

CATEGORIES = [
    "Disengagement",
    "Compartmentalized coexistence",
    "Radical confrontation",
    "Critical bridging",
]
# (engagement high?, integration high?) -> Figure 1 quadrant
QUADRANTS = {
    (False, False): "Disengagement",
    (False, True): "Compartmentalized coexistence",
    (True, False): "Radical confrontation",
    (True, True): "Critical bridging",
}

SYSTEM = (
    "You are an expert annotator of research papers in AI ethics and AI safety. "
    "You apply the coding scheme exactly as written and ground every judgement in "
    "the text of the paper."
)

PROMPT = """\
We study how research papers position themselves with respect to the divide between \
two research communities.

## The two fields
- AI ethics (AIE): the broad field studying the ethical and technosocial dimensions of AI, \
represented by, but not limited to, venues such as ACM FAccT and AIES. It includes \
work in philosophy and machine ethics, law and policy, science and technology \
studies, and critical AI studies. Typical concerns: fairness and discrimination, \
accountability, transparency, privacy, labour, power and structural injustice, the \
moral status and values of AI systems, present-day and tangible harms of deployed \
systems.
- AI safety (AIS): the practices of alignment and safety research labs and researchers. \
Typical concerns: alignment, robustness, interpretability, dangerous capabilities, \
evaluation of frontier models, misuse, loss of control, catastrophic and existential \
risk from advanced AI.

This paper was retrieved from the {home} corpus, so its "other field" is {other}.

## Axis 1: engagement (1 to 5)
How directly and substantively does the paper engage with the other field as a field: \
its community, its positions, priorities and arguments, or the tensions between AIE and \
AIS (their differing priorities, values, evidence standards, risk framings)? Engagement \
can be critical or constructive; what matters is directness.

Using concepts, methods or topics that happen to belong to the other field is NOT \
engagement by itself. For example, an AI ethics paper that studies RLHF or alignment, or \
an AI safety paper that also covers fairness or privacy, has engagement 1 unless it \
addresses the other field's positions or its relationship to the paper's own field. \
Such topical overlap belongs on the integration axis instead.
- 1: the other field is never addressed as a field, and the AIE-AIS relationship is \
never mentioned (topical overlap alone still scores 1).
- 2: the other field is named or cited in passing (related work, motivation), \
without discussing its positions.
- 3: one substantive passage discusses the other field's positions, priorities or \
arguments, or how the paper relates to them, but this is not a thread of the paper.
- 4: engaging the other field's positions, or the AIE-AIS tensions, is a major \
thread of the paper.
- 5: the AIE-AIS relationship, or the other field's positions, is the central topic.

## Axis 2: integration (1 to 5)
How far does the paper bring AIE and AIS concerns, concepts, methods or problem \
framings together within one frame, rather than working within a single field or \
setting the fields against each other?

Mixing topics is not the same as integrating the fields. A technical paper that \
combines, say, a fairness metric with a robustness technique uses ingredients from \
both fields, but integrates them only if it treats them as the concerns of each field \
and connects them. Showing that a technical property (e.g. lack of robustness) leads \
to a harm (e.g. discrimination) is not enough by itself.

Taking the other field's concerns into the paper's own framework while dismissing \
that field's approaches or positions is low integration, however much of both \
fields the paper discusses.
- 1: works entirely within one field's framing, or treats the other field only as \
an opponent to be discredited.
- 2: topics or techniques from both fields appear, but as technical ingredients \
rather than as concerns of the other field; or the other field's concerns are only \
acknowledged, kept separate, or mentioned in passing (e.g. a paragraph noting safety \
risks in an otherwise fairness-focused paper).
- 3: treats concerns from both fields as concerns, side by side in the same work \
(e.g. both discrimination and loss of control), with little connection between them.
- 4: explicitly argues that the fields' concerns are connected, treating each as a \
concern of that field, e.g. shows how one problem is both an ethics problem and a \
safety problem and draws on both fields to address it.
- 5: builds a genuinely joint framing, problem definition, method or governance \
proposal that serves both fields.

Score the two axes independently: a paper can engage the other field intensely while \
integrating nothing (e.g. a polemic), or place both fields' concerns side by side \
without engaging the tensions between them.

Score each axis as an integer from 1 to 5: the level whose description best fits \
the paper as a whole.

## Output
Return JSON with:
- "evidence": up to 3 short verbatim quotes (each under 40 words) that most inform \
your scores; an empty list if the paper never touches the other field;
- "rationale": 2-4 sentences explaining the scores;
- "engagement" and "integration": integers from 1 to 5, as defined above;
- "other_field_referenced": true if the paper refers to the other field or its \
literature at all.

## Paper ({source})
### Title
{title}

### Abstract
{abstract}

### Full text
{body}
"""

SCHEMA = {
    "name": "engagement_classification",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "evidence": {"type": "array", "items": {"type": "string"}},
            "rationale": {"type": "string"},
            "engagement": {"type": "integer", "enum": list(LEVELS)},
            "integration": {"type": "integer", "enum": list(LEVELS)},
            "other_field_referenced": {"type": "boolean"},
        },
        "required": [
            "evidence",
            "rationale",
            "engagement",
            "integration",
            "other_field_referenced",
        ],
    },
}

app = typer.Typer(rich_markup_mode="rich", help=__doc__.split("\n\n")[0])


def out_dir() -> Path:
    path = OUT_ROOT / PROMPT_VERSION
    path.mkdir(parents=True, exist_ok=True)
    return path


def client() -> Client:
    if not os.environ.get("OPENAI_API_KEY"):
        raise typer.BadParameter("Set OPENAI_API_KEY in the environment.")
    return Client()


def load_papers() -> pd.DataFrame:
    papers = pd.read_csv(PAPERS, index_col="Key")
    return papers[["Retrieval", "Publication_Year", "DOI", "Title", "Abstract"]]


# ---- text extraction --------------------------------------------------------
# "R EFERENCES" / "A PPENDIX": small-caps headings come out of pdftotext letter-spaced
REFERENCES = re.compile(
    r"\n[ \t\f]*(?:\d+\.?\s*)?(r ?eferences|bibliography|works cited)[ \t]*\n",
    re.IGNORECASE,
)
MIN_CUT = 0.15  # ignore headings in the first 15% of a paper (tables of contents etc.)
# Candidate appendix headings on a line of their own: "Appendix", "APPENDIX A",
# "A Appendix", "Appendix B: Prompts", "Supplementary Material", ... Case-sensitive
# so running text ("appendix b shows") does not match; is_appendix_heading() then
# rejects wrapped sentences such as "...see\nAppendix B.4 for details."
APPENDIX = re.compile(
    r"^[ \t\f]*(?:[A-Z]\.?[ \t]+)?"
    r"(?:A ?PPENDIX|Appendix|A ?PPENDICES|Appendices"
    r"|TECHNICAL APPENDIX|Technical Appendix"
    r"|(?:SUPPLEMENTARY|SUPPLEMENTAL)[ \t]+(?:MATERIALS?|INFORMATION)"
    r"|(?:Supplementary|Supplemental)[ \t]+(?:Materials?|Information))"
    r"(?:[ \t]+(?:[A-Z]|\d{1,2}))?"
    r"(?:[ \t]*[:.–—-][ \t]*(?P<title>[^\n]{1,60}))?[ \t]*$",
    re.MULTILINE,
)


def is_appendix_heading(text: str, match: re.Match) -> bool:
    title = match.group("title")
    if title and (not title[0].isupper() or "," in title or title[-1] in ".,;)"):
        return False
    # the previous line must end a paragraph: blank, a page break or page number,
    # or a line ending in sentence-final punctuation
    before = text[: match.start()]
    if before.endswith(("\n\n", "\f")) or match.group().lstrip(" \t").startswith("\f"):
        return True
    prev = before.rstrip("\n").rsplit("\n", 1)[-1].strip()
    return not prev or prev.isdigit() or prev[-1] in ".!?:)]\"”"


# Appendices without an "Appendix" heading usually start with a lettered section,
# e.g. "A  Additional results" or "A.1 Prompt templates". Only looked for after the
# reference list, where such a line cannot be a body section.
LETTERED_SECTION = re.compile(
    r"^[ \t\f]*A(?:\.1)?\.?[ \t]+(?P<title>[A-Z][^\n,]{2,60})[ \t]*$", re.MULTILINE
)


def remove_appendices(text: str) -> str:
    """Remove appendices, keeping the body and the reference list.

    Walks the paper's headings in order: an appendix heading stops keeping text and
    a reference-list heading resumes it, so a reference list that comes after an
    appendix is kept. After a reference list, a lettered section heading ("A ...")
    also counts as the start of an appendix.
    """
    start = len(text) * MIN_CUT
    events = [(m.start(), "refs") for m in REFERENCES.finditer(text)]
    events += [
        (m.start(), "appendix")
        for m in APPENDIX.finditer(text)
        if is_appendix_heading(text, m)
    ]
    events = sorted(e for e in events if e[0] > start)
    first_refs = next((pos for pos, kind in events if kind == "refs"), None)
    if first_refs is not None:
        events += [
            (m.start(), "appendix")
            for m in LETTERED_SECTION.finditer(text, first_refs)
            # a real word in the title, so table debris ("A B C") does not count
            if re.search(r"[A-Za-z]{3,}", m.group("title"))
            and is_appendix_heading(text, m)
        ]
        events.sort()

    kept, keep_from = [], 0
    for pos, kind in events:
        if kind == "appendix" and keep_from is not None:
            kept.append(text[keep_from:pos])
            keep_from = None
        elif kind == "refs" and keep_from is None:
            keep_from = pos
    if keep_from is not None:
        kept.append(text[keep_from:])
    return "\n".join(kept)


def clean_text(text: str) -> str:
    """Drop appendices (keeping the references) and collapse whitespace."""
    text = remove_appendices(text)
    text = re.sub(r"-\n(\w)", r"\1", text)  # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


@app.command()
def extract(force: bool = typer.Option(False, help="Re-extract existing text files.")):
    """Extract text from pdf/<Key>.pdf into output/engagement/txt/<Key>.txt."""
    TXT_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    done = failed = 0
    for pdf in pdfs:
        dest = TXT_DIR / f"{pdf.stem}.txt"
        if dest.exists() and not force:
            continue
        result = subprocess.run(
            ["pdftotext", "-enc", "UTF-8", str(pdf), "-"],
            capture_output=True,
            text=True,
            check=False,
        )
        text = clean_text(result.stdout)
        if result.returncode != 0 or len(text) < MIN_TEXT_CHARS:
            typer.secho(f"{pdf.stem}: little or no text", fg=typer.colors.YELLOW)
            failed += 1
            continue
        dest.write_text(text)
        done += 1
    typer.secho(f"Extracted {done} new, {failed} failed, {len(pdfs)} PDFs in total.")


# ---- requests ---------------------------------------------------------------
def build_messages(key: str, row: pd.Series, max_chars: int) -> tuple[list, str]:
    txt = TXT_DIR / f"{key}.txt"
    if txt.exists():
        body, source = txt.read_text()[:max_chars], "full text"
    else:
        body = "(not available; judge from the title and abstract)"
        source = "abstract only"
    home = "AI ethics" if row["Retrieval"] == "Ethics" else "AI safety"
    other = "AI safety" if home == "AI ethics" else "AI ethics"
    prompt = PROMPT.format(
        home=home,
        other=other,
        source=source,
        title=row["Title"],
        abstract=row["Abstract"] if isinstance(row["Abstract"], str) else "(none)",
        body=body,
    )
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
    ]
    return messages, source


def request_body(messages: list) -> dict:
    return {
        "model": MODEL,
        "messages": messages,
        "reasoning_effort": REASONING_EFFORT,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "response_format": {"type": "json_schema", "json_schema": SCHEMA},
    }


def select(papers: pd.DataFrame, keys: str | None, n: int | None, seed: int):
    if keys:
        return papers.loc[[k.strip() for k in keys.split(",")]]
    if n:
        return papers.sample(n=n, random_state=seed)
    return papers


@app.command()
def prepare(
    keys: str = typer.Option(None, help="Comma-separated paper keys to include."),
    n: int = typer.Option(None, help="Random sample of n papers."),
    seed: int = typer.Option(0),
    max_chars: int = typer.Option(MAX_CHARS, help="Characters of body text per paper."),
):
    """Write Batch API request files (shards) for the selected papers."""
    papers = select(load_papers(), keys, n, seed)
    out = out_dir()
    for old in out.glob("requests_*.jsonl"):
        old.unlink()

    shard, size, n_shard, chars, n_abstract = [], 0, 0, 0, 0

    def flush():
        nonlocal shard, size, n_shard
        if shard:
            (out / f"requests_{n_shard}.jsonl").write_text("".join(shard))
            n_shard, shard, size = n_shard + 1, [], 0

    for key, row in papers.iterrows():
        messages, source = build_messages(key, row, max_chars)
        n_abstract += source == "abstract only"
        chars += sum(len(m["content"]) for m in messages)
        line = json.dumps(
            {
                "custom_id": key,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": request_body(messages),
            }
        ) + "\n"
        if size + len(line) > MAX_SHARD_BYTES or len(shard) >= MAX_SHARD_REQUESTS:
            flush()
        shard.append(line)
        size += len(line)
    flush()
    typer.secho(
        f"Wrote {n_shard} shard(s) for {len(papers)} papers to {out} "
        f"({n_abstract} abstract only). ~{chars / 4 / 1e6:.1f}M input tokens."
    )


# ---- synchronous pilot --------------------------------------------------------
@app.command()
def pilot(
    keys: str = typer.Option(None, help="Comma-separated paper keys."),
    n: int = typer.Option(20, help="Random sample size when --keys is not given."),
    seed: int = typer.Option(0),
    max_chars: int = typer.Option(MAX_CHARS),
):
    """Classify a few papers synchronously, for iterating on the prompt."""
    papers = select(load_papers(), keys, n, seed)
    api = client()
    dest = out_dir() / "pilot.jsonl"
    with dest.open("a") as f:
        for key, row in papers.iterrows():
            messages, _ = build_messages(key, row, max_chars)
            response = api.chat.completions.create(**request_body(messages))
            # store in the same shape as a Batch API output line
            f.write(
                json.dumps(
                    {
                        "custom_id": key,
                        "response": {"status_code": 200, "body": response.model_dump()},
                    }
                )
                + "\n"
            )
            f.flush()
            typer.secho(f"{key}: {response.choices[0].message.content[-160:]}")
    typer.secho(f"Appended {len(papers)} results to {dest}")


# ---- batch submission ---------------------------------------------------------
def load_state() -> dict:
    path = out_dir() / "batches.json"
    return json.loads(path.read_text()) if path.exists() else {}


def save_state(state: dict):
    (out_dir() / "batches.json").write_text(json.dumps(state, indent=2))


@app.command()
def submit():
    """Upload request shards and create one batch per shard (skips submitted ones)."""
    api, state = client(), load_state()
    for shard in sorted(out_dir().glob("requests_*.jsonl")):
        if shard.name in state:
            typer.secho(f"{shard.name}: already submitted ({state[shard.name]['batch']})")
            continue
        uploaded = api.files.create(file=shard.open("rb"), purpose="batch")
        batch = api.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": f"engagement {PROMPT_VERSION} {shard.name}"},
        )
        state[shard.name] = {"file": uploaded.id, "batch": batch.id}
        save_state(state)
        typer.secho(f"{shard.name}: submitted batch {batch.id}")


@app.command()
def fetch():
    """Report batch status and download finished outputs."""
    api, state = client(), load_state()
    for name, info in state.items():
        batch = api.batches.retrieve(info["batch"])
        counts = batch.request_counts
        typer.secho(
            f"{name}: {batch.status} ({counts.completed}/{counts.total} done, "
            f"{counts.failed} failed)"
        )
        dest = out_dir() / f"output_{name}"
        if batch.status == "completed" and not dest.exists():
            dest.write_text(api.files.content(batch.output_file_id).text)
            if batch.error_file_id:
                errors = api.files.content(batch.error_file_id).text
                (out_dir() / f"errors_{name}").write_text(errors)
            typer.secho(f"  saved {dest}")


# ---- parsing ----------------------------------------------------------------------
def quadrant(engagement: int, integration: int) -> str:
    """The Figure 1 quadrant for a pair of levels."""
    return QUADRANTS[(engagement >= ENGAGEMENT_HIGH, integration >= INTEGRATION_HIGH)]


def parse_line(line: str) -> dict:
    obj = json.loads(line)
    row = {"key": obj["custom_id"], "error": None}
    body = (obj.get("response") or {}).get("body") or {}
    try:
        content = body["choices"][0]["message"]["content"]
        result = json.loads(content)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as e:
        row["error"] = f"unparseable response: {e!r}"
        return row
    for axis in ("engagement", "integration"):
        value = result[axis]
        if not isinstance(value, (int, float)) or value not in LEVELS:
            row["error"] = f"{axis} is not a level from 1 to 5: {value!r}"
            return row
        row[axis] = int(value)
    row["category"] = quadrant(row["engagement"], row["integration"])
    row["other_field_referenced"] = result["other_field_referenced"]
    row["rationale"] = result["rationale"]
    row["evidence"] = " | ".join(result["evidence"])
    row["reasoning_tokens"] = (
        body.get("usage", {}).get("completion_tokens_details", {}).get("reasoning_tokens")
    )
    return row


@app.command()
def parse(pilot_only: bool = typer.Option(False, "--pilot", help="Parse pilot.jsonl.")):
    """Combine outputs into output/engagement/<version>/engagement[_pilot].csv."""
    out = out_dir()
    files = [out / "pilot.jsonl"] if pilot_only else sorted(out.glob("output_*.jsonl"))
    lines = [line for f in files if f.exists() for line in f.read_text().splitlines()]
    if not lines:
        raise typer.BadParameter(f"No results found in {out}.")
    df = pd.DataFrame([parse_line(line) for line in lines])
    # keep the latest result per paper (pilot runs may repeat papers)
    df = df.drop_duplicates("key", keep="last").set_index("key")
    papers = load_papers()
    df = papers[["Retrieval", "Publication_Year", "Title"]].join(df, how="inner")
    df["input"] = ["full text" if (TXT_DIR / f"{k}.txt").exists() else "abstract only"
                   for k in df.index]
    dest = out / ("engagement_pilot.csv" if pilot_only else "engagement.csv")
    df.to_csv(dest)

    ok = df[df["error"].isna()]
    typer.secho(f"Wrote {dest}: {len(df)} papers, {len(df) - len(ok)} with errors.")
    if len(ok):
        typer.secho("\nPapers per level:")
        levels = {
            axis: ok[axis].value_counts().reindex(LEVELS, fill_value=0)
            for axis in ("engagement", "integration")
        }
        typer.secho(pd.DataFrame(levels).to_string())
        typer.secho("\nCategory by corpus:")
        typer.secho(pd.crosstab(ok["category"], ok["Retrieval"], margins=True).to_string())
        typer.secho("\nMean scores by corpus:")
        typer.secho(ok.groupby("Retrieval")[["engagement", "integration"]]
                    .describe().round(2).to_string())
    if not pilot_only:
        missing = set(papers.index) - set(df.index)
        if missing:
            typer.secho(f"{len(missing)} papers have no result yet.", fg=typer.colors.YELLOW)


@app.command()
def sample_for_validation(
    n: int = typer.Option(10, help="Papers per category."),
    seed: int = typer.Option(0),
):
    """Draw a stratified sample of classified papers for manual checking."""
    df = pd.read_csv(out_dir() / "engagement.csv", index_col=0)
    rng = random.Random(seed)
    picks = []
    for _, group in df.groupby("category"):
        idx = list(group.index)
        picks += rng.sample(idx, min(n, len(idx)))
    sample = df.loc[picks, ["Retrieval", "Title", "engagement", "integration",
                            "category", "rationale", "evidence"]]
    sample["human_engagement"] = ""
    sample["human_integration"] = ""
    sample["human_category"] = ""
    dest = out_dir() / "validation_sample.csv"
    sample.to_csv(dest)
    typer.secho(f"Wrote {len(sample)} papers to {dest}")


if __name__ == "__main__":
    app()
