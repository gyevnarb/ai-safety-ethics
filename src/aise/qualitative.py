"""Inductive qualitative coding and thematic analysis of the rare-mode papers.

The papers in the three rare modes of Figure 1 (Compartmentalized coexistence, Radical
confrontation, Critical bridging; final labels in output/engagement/<PROMPT_VERSION>/
engagement.csv) are coded for the particular risks and mitigation strategies they address
and the positions they take on them, so that points of agreement and disagreement can be
compared across modes. The model only ever sees a paper's title, abstract and full text
and the shared codebook: never its mode, engagement scores, earlier rationale, field
corpus or taxonomy labels. Modes and fields are joined back in only by ``report``.

Every stage visits the papers in its own random order (saved in its order.json)::

    uv run aise qualitative code --stage 1   # open coding; the codebook grows
    uv run aise qualitative code --stage 2   # recode with the full codebook, adding,
                                             # merging, redefining codes
    uv run aise qualitative freeze           # drop stale codes, consolidate, fix the
                                             # codebook
    uv run aise qualitative themes           # group the fixed codes into themes
    uv run aise qualitative spans            # theme spans in every paper
    uv run aise qualitative representatives  # representative spans per theme
    uv run aise qualitative report           # CSVs joined with mode and field

(``uv run python -m aise.qualitative <command>`` works too.)

Stages 1 and 2 are sequential, since each paper is coded against the codebook as the
papers before it left it; both resume where they stopped. Outputs go to
output/qualitative/<QUAL_VERSION>/. Needs ``OPENAI_API_KEY`` in the environment.
"""

import json
import random
import re
import shutil
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import openai
import pandas as pd
import typer

from aise import config
from aise import engagement as E

MODEL = "gpt-6.1-sol"
REASONING_EFFORT = "high"
MAX_COMPLETION_TOKENS = 32_000  # includes reasoning tokens
MAX_CHARS = 250_000  # ~62k tokens; cuts only the three longest papers
SEED = 20261011
OUT = config.QUALITATIVE_DIR
SPAN_WORKERS = 8  # the spans stage has no shared state, so it runs in parallel

FOCI = ["risk", "mitigation", "risk-mitigation relation"]
STANCES = ["endorses", "contests", "qualifies", "describes"]

SYSTEM = (
    "You are an experienced qualitative researcher performing inductive coding of "
    "research papers on the risks of AI systems and the strategies for mitigating them. "
    "You ground every code in the text of the paper and quote it verbatim."
)

AIM = """\
We are analysing how research papers engage with the risks of AI systems and the strategies for mitigating them: which particular risks and mitigation strategies a paper addresses, how it frames them, and what positions it takes on them (for instance, which risks it treats as serious, neglected, overstated or misframed, and which mitigations it advocates, criticises, considers insufficient or harmful, and why). The analysis will be used to compare where papers agree and disagree."""  # noqa: E501

CODING_RULES = """\
## Coding rules
- Code inductively: derive codes from what the paper says, not from categories the paper does not support.
- A code names a particular risk, a particular mitigation strategy, or a relation or claim connecting them (e.g. a mitigation that addresses, neglects or worsens a risk). Pitch codes at the level of a recognisable risk or mitigation that many papers could address (e.g. a kind of harm, or a family of interventions), not at the level of one paper's mechanism, method or case. Avoid generic codes like "AI risk" or "regulation", but also avoid paper-specific ones: the details of how a paper treats a risk or mitigation belong in its position and spans, not in a new code.
- Do not create separate codes for opposing positions on the same risk or mitigation: one code covers them all, and the stance records whether the paper endorses, contests or qualifies it.
- Reuse an existing code whenever its definition fits, even if loosely. Create a new code only when the paper's content falls clearly outside every existing code; never create a near-duplicate, a narrower variant or a special case of an existing code.
- Code substantive engagement only: code only the risks and mitigations that are central to the paper's argument, and skip those mentioned in passing (e.g. in a list in the related work) or only as background.
- Code what the paper argues against as well as what it advances. If the paper substantively criticises a risk framing, a prioritisation of risks, or a mitigation proposed by others (e.g. it argues that some risk is overstated or a distraction, or that some approach is inadequate or harmful), code that risk or mitigation too, with the stance "contests" or "qualifies", even though it is not the paper's own proposal.
- For every code you apply, record the paper's stance towards its content:
  - "endorses": treats the risk as real and serious, or advocates or adopts the mitigation;
  - "contests": argues the risk is overstated, misframed or a distraction, or that the mitigation is inadequate, misguided or harmful;
  - "qualifies": a conditional or mixed position (e.g. a mitigation is useful only under some conditions);
  - "describes": reports the risk or mitigation without taking a position.
- For every code you apply, write the paper's position in one sentence, and quote 1 to 3 spans that support it. Spans must be copied verbatim from the title, abstract or full text, each under 50 words.
- A paper typically has between 2 and 8 codes."""  # noqa: E501

STAGE1 = """\
{aim}

We are building a shared codebook one paper at a time. Code the paper below against the current codebook, and add new codes only where the paper's central content falls outside every existing code.

{rules}

## Current codebook ({n} codes)
{codebook}

## Output
Return JSON with:
- "codes": every code you apply to this paper. For an existing code, copy its name exactly, set "new" to false, and leave "definition" empty (its "focus" is ignored). For a new code, set "new" to true, give a short name, its focus, and a one- or two-sentence definition that would let another coder apply it to other papers.
- "memo": 1-3 sentences on how the paper engages with risks and mitigations overall.

{paper}"""  # noqa: E501

STAGE2 = """\
{aim}

A first pass over the whole corpus built the codebook below, one paper at a time. This is the second pass: code the paper below from scratch against the full codebook, and revise the codebook where this paper shows it is lacking.

{rules}

## Revising the codebook
- Add a code (as in the first pass) if the paper has substantive content that no code captures.
- Merge a code into another if the two are redundant: their definitions cover the same content, one is a needless special case of the other, one is specific to a single paper, or they capture opposing positions on the same risk or mitigation. Give the definition of the merged code.
- Redefine a code if its definition is unclear, too broad or too narrow to apply consistently.
- Do not retire codes: codes that no paper uses in this pass are retired automatically.
- Most papers need no revisions; return an empty list then.

## Codebook ({n} codes)
{codebook}

## Output
Return JSON with:
- "codes": every code you apply to this paper. For an existing code, copy its name exactly, set "new" to false, and leave "definition" empty (its "focus" is ignored). For a new code, set "new" to true, give a short name, its focus, and a one- or two-sentence definition that would let another coder apply it to other papers.
- "revisions": merges and redefinitions of existing codes. For a merge, "code" is the code to remove, "into" the existing code it merges into, and "definition" the definition of the merged code (empty to keep it). For a redefinition, "code" is the code, "into" is empty, and "definition" the new definition. Give the reason in one sentence.
- "memo": 1-3 sentences on how the paper engages with risks and mitigations overall.

{paper}"""  # noqa: E501

CONSOLIDATE = """\
{aim}

Two passes over the corpus coded every paper with the codebook below. For each code you see its focus, its definition, the number of papers it was applied to, the stances papers took towards it, and positions from a few of those papers. Before the codebook is fixed, consolidate it:
- Merge codes that are redundant, overlapping, paper-specific, or needless special cases of a broader code, and codes that capture opposing positions on the same risk or mitigation (the stance recorded with each application already tells positions apart). A merge may go into one of the merged codes or into a new code; give the definition and focus of the result. Keep codes apart only when they concern substantively different risks or mitigations, since the codebook must let us compare where papers agree and disagree about the same thing.
- Redefine codes whose definitions are unclear or do not match how they were applied.
- Leave everything else as it is; return empty lists if nothing needs changing.

## Codebook ({n} codes)
{codebook}

## Output
Return JSON with:
- "merges": each with "codes" (two or more existing code names, copied exactly), "into" (the name of the resulting code), "focus", "definition" and a one-sentence "reason".
- "redefinitions": each with "code", the new "definition" and a one-sentence "reason"."""  # noqa: E501

THEMES = """\
{aim}

Inductive coding of the corpus produced the fixed codebook below. For each code you see its focus, its definition, the number of papers it was applied to, the stances papers took towards it, and positions from a few of those papers.

Perform thematic analysis: group related codes into themes. A theme is a pattern of shared meaning across codes that captures something important about how the papers engage with risks and mitigations; it is more than a topic label. Describe each theme, including the main positions papers take within it and where they agree and disagree.

Every code must be in exactly one theme. Use as many themes as the codes call for (typically between 6 and 15).

## Codebook ({n} codes)
{codebook}

## Output
Return JSON with "themes": each with a short "name", a "description" of 3-6 sentences, and "codes" (the names of its codes, copied exactly)."""  # noqa: E501

SPANS = """\
{aim}

Thematic analysis of the corpus produced the themes below, each built from codes of a fixed codebook. Read the paper below and, for every theme the paper engages with substantively, record:
- "position": the paper's position within the theme, in one or two sentences;
- "spans": the 1 to 3 text spans that best represent how the paper engages with the theme, copied verbatim from the title, abstract or full text, each under 60 words.

Skip themes the paper touches only in passing. For orientation, the codes assigned to this paper during coding are listed after the themes.

## Themes
{themes}

## Codes assigned to this paper
{codes}

## Output
Return JSON with "themes": one entry per theme the paper engages with, with the theme "name" copied exactly, "position" and "spans".

{paper}"""  # noqa: E501

REPRESENT = """\
{aim}

Thematic analysis of the corpus produced the theme below. Below it are the spans recorded for this theme across the corpus, each with an id, the paper it comes from (anonymised) and that paper's position within the theme.

Select the {k} spans that together best represent the theme: each should be clear and typical of an important position within the theme, and together they should cover the range of positions, including the points where papers agree and where they disagree. Take at most one span per paper.

## Theme: {name}
{description}

## Spans
{spans}

## Output
Return JSON with "selected": each with the span "id" and a one-sentence "why" it represents the theme."""  # noqa: E501


def _strict(name: str, properties: dict) -> dict:
    return {
        "name": name,
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": properties,
            "required": list(properties),
        },
    }


def _array(properties: dict) -> dict:
    return {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "properties": properties,
            "required": list(properties),
        },
    }


STR = {"type": "string"}
STRS = {"type": "array", "items": STR}
CODES = _array({
    "code": STR,
    "new": {"type": "boolean"},
    "focus": {"type": "string", "enum": FOCI},
    "definition": STR,
    "stance": {"type": "string", "enum": STANCES},
    "position": STR,
    "spans": STRS,
})
STAGE1_SCHEMA = _strict("coding", {"codes": CODES, "memo": STR})
STAGE2_SCHEMA = _strict("recoding", {
    "codes": CODES,
    "revisions": _array({
        "action": {"type": "string", "enum": ["merge", "redefine"]},
        "code": STR,
        "into": STR,
        "definition": STR,
        "reason": STR,
    }),
    "memo": STR,
})
CONSOLIDATE_SCHEMA = _strict("consolidation", {
    "merges": _array({
        "codes": STRS,
        "into": STR,
        "focus": {"type": "string", "enum": FOCI},
        "definition": STR,
        "reason": STR,
    }),
    "redefinitions": _array({"code": STR, "definition": STR, "reason": STR}),
})
THEMES_SCHEMA = _strict("themes", {
    "themes": _array({"name": STR, "description": STR, "codes": STRS}),
})
SPANS_SCHEMA = _strict("theme_spans", {
    "themes": _array({"name": STR, "position": STR, "spans": STRS}),
})
REPRESENT_SCHEMA = _strict("representatives", {
    "selected": _array({"id": STR, "why": STR}),
})

app = typer.Typer(help=__doc__.split("\n\n")[0])


# ---- files and state ------------------------------------------------------------------
def stage_dir(stage: int) -> Path:
    """The folder of coding stage 1 or 2 (created if missing)."""
    path = OUT / f"stage{stage}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path):
    """The JSON object in a file."""
    return json.loads(path.read_text())


def write_json(path: Path, obj):
    """Write obj as JSON, atomically, so a crash never leaves a half-written file."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    """The objects in a JSON Lines file; none if it does not exist."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def append_jsonl(path: Path, obj):
    """Append obj to a JSON Lines file."""
    with path.open("a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def order(name: str, keys: list[str]) -> list[str]:
    """The stage's random paper order, drawn once and saved so a resumed run keeps it."""
    path = OUT / name / "order.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return read_json(path)
    keys = sorted(keys)
    random.Random(f"{SEED}-{name}").shuffle(keys)
    write_json(path, keys)
    return keys


# ---- the corpus (blind) ---------------------------------------------------------------
def corpus() -> pd.DataFrame:
    """Title and abstract of the rare-mode papers, indexed by key; nothing else."""
    modes = E.load_classified()["category"]
    keys = modes.index[modes != E.CATEGORIES[0]]
    if config.QUAL_TEST_N:  # a small corpus for testing: QUAL_TEST_N papers per mode
        rare = modes.loc[keys].sort_index()
        keys = rare.groupby(rare).sample(config.QUAL_TEST_N, random_state=SEED).index
    return E.load_papers().loc[keys, ["Title", "Abstract"]]


def paper_text(key: str, row: pd.Series) -> tuple[str, str]:
    """The paper section of a prompt, and the source text spans are checked against."""
    txt = E.text_path(key)
    body = txt.read_text()[:MAX_CHARS] if txt.exists() else "(not available)"
    abstract = row["Abstract"] if isinstance(row["Abstract"], str) else "(none)"
    section = (
        f"## Paper\n### Title\n{row['Title']}\n\n### Abstract\n{abstract}\n\n"
        f"### Full text\n{body}"
    )
    return section, f"{row['Title']}\n{abstract}\n{body}"


def _norm(text: str) -> str:
    text = re.sub(r"-\s*\n\s*", "", text)  # words hyphenated across lines
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def verbatim(span: str, source: str) -> bool:
    """Whether every part of a span (split at ellipses) occurs in the source text.

    Ignores case, punctuation and whitespace.
    """
    norm_source = _norm(source)
    parts = [_norm(p) for p in re.split(r"\.\.\.|…|\[\.\.\.\]", span)]
    parts = [p for p in parts if p]
    return bool(parts) and all(p in norm_source for p in parts)


# ---- the model -------------------------------------------------------------------------
_client = None


def call(prompt: str, schema: dict) -> tuple[dict, dict]:
    """The model's JSON answer to the prompt, and the token usage; retries on errors."""
    global _client  # noqa: PLW0603
    if _client is None:
        _client = E.client()
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt}]
    for attempt in range(5):
        try:
            response = _client.chat.completions.create(
                model=MODEL,
                messages=messages,
                reasoning_effort=REASONING_EFFORT,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                response_format={"type": "json_schema", "json_schema": schema},
            )
            result = json.loads(response.choices[0].message.content)
            return result, response.usage.model_dump()
        except (openai.APIError, json.JSONDecodeError, TypeError) as e:
            wait = 15 * 2**attempt
            typer.secho(f"  attempt {attempt + 1} failed ({e!r}); retrying in {wait}s",
                        fg="yellow")
            time.sleep(wait)
    raise RuntimeError("The model call failed five times.")


# ---- the codebook ----------------------------------------------------------------------
# A codebook is {"codes": {name: {"focus", "definition", "created"}},
# "aliases": {merged name: name it was merged into}, "retired": {name: entry}}
def resolve(book: dict, name: str) -> str | None:
    """The current name of a code, following merges; None if it is unknown."""
    lower = {n.lower(): n for n in book["codes"]}
    aliases = {n.lower(): into for n, into in book["aliases"].items()}
    name, seen = name.strip(), set()
    while name.lower() not in seen:
        seen.add(name.lower())
        if name.lower() in lower:
            return lower[name.lower()]
        if name.lower() not in aliases:
            return None
        name = aliases[name.lower()]
    return None


def render(book: dict, stats: dict | None = None,
           rng: random.Random | None = None) -> str:
    """The codebook as prompt text: sorted by focus and name, or shuffled given rng."""
    if not book["codes"]:
        return "(empty: this is the first paper)"
    names = sorted(book["codes"],
                   key=lambda n: (FOCI.index(book["codes"][n]["focus"]), n))
    if rng:
        rng.shuffle(names)
    lines = []
    for name in names:
        code = book["codes"][name]
        lines.append(f"- **{name}** ({code['focus']}): {code['definition']}")
        if stats:
            s = stats[name]
            stances = ", ".join(f"{k} {v}" for k, v in s["stances"].most_common())
            lines.append(f"  - {s['papers']} papers; stances: {stances}")
            lines.extend(f"  - position: {p}" for p in s["positions"])
    return "\n".join(lines)


def apply_codes(book: dict, stage: int, codes: list, log) -> list[dict]:
    """Resolve the codes the model applied, adding new ones to the codebook."""
    applied = []
    for c in codes:
        name = resolve(book, c["code"])
        if name is None:
            if not c["definition"].strip():
                log({"event": "dropped", "code": c["code"],
                     "reason": "unknown code without a definition"})
                continue
            name = c["code"].strip()
            book["codes"][name] = {"focus": c["focus"],
                                   "definition": c["definition"].strip(),
                                   "created": f"stage{stage}"}
            log({"event": "add", "code": name, "focus": c["focus"],
                 "definition": c["definition"].strip()})
        elif c["new"]:
            log({"event": "new code matched existing", "code": c["code"],
                 "existing": name})
        applied.append({"code": name, "stance": c["stance"], "position": c["position"],
                        "spans": c["spans"]})
    return applied


def merge(book: dict, code: str, into: str, definition: str = "", focus: str = ""):
    """Merge code into another, optionally redefining the result."""
    if code != into:
        del book["codes"][code]
        book["aliases"][code] = into
    if definition.strip():
        book["codes"][into]["definition"] = definition.strip()
    if focus:
        book["codes"][into]["focus"] = focus


def apply_revisions(book: dict, revisions: list, log):
    """Apply the merges and redefinitions the model proposed in stage 2."""
    for r in revisions:
        code = resolve(book, r["code"])
        if code is None:
            log({"event": "revision skipped", "revision": r, "reason": "unknown code"})
        elif r["action"] == "redefine":
            if r["definition"].strip():
                book["codes"][code]["definition"] = r["definition"].strip()
                log({"event": "redefine", "code": code, "definition": r["definition"],
                     "reason": r["reason"]})
        else:
            into = resolve(book, r["into"])
            if into is None or into == code:
                log({"event": "revision skipped", "revision": r,
                     "reason": "unknown or identical target"})
                continue
            merge(book, code, into, r["definition"])
            log({"event": "merge", "code": code, "into": into,
                 "definition": r["definition"], "reason": r["reason"]})


def code_stats(book: dict, codings: list[dict], rng: random.Random,
               n_positions: int = 3) -> dict:
    """Papers, stances and a few sample positions per current code."""
    stats = {n: {"papers": set(), "stances": Counter(), "positions": []}
             for n in book["codes"]}
    for coding in codings:
        for c in coding["codes"]:
            name = resolve(book, c["code"])
            if name is None:
                continue
            stats[name]["papers"].add(coding["key"])
            stats[name]["stances"][c["stance"]] += 1
            stats[name]["positions"].append(c["position"])
    for s in stats.values():
        s["papers"] = len(s["papers"])
        s["positions"] = rng.sample(s["positions"], min(n_positions, len(s["positions"])))
    return stats


def final_codings() -> tuple[dict, list[dict]]:
    """The fixed codebook and the codings in its terms; fails before ``freeze``."""
    final = OUT / "codebook_final.json"
    if not final.exists():
        raise typer.BadParameter("Run `freeze` first.")
    return read_json(final), read_jsonl(OUT / "codings_final.jsonl")


# ---- stages 1 and 2: coding ------------------------------------------------------------
@app.command()
def code(
    stage: int = typer.Option(..., help="1: open coding; 2: recoding with the codebook."),
    limit: int = typer.Option(None, help="Stop after this many papers (for testing)."),
    show_prompt: bool = typer.Option(False, help="Print the next prompt and stop."),
):
    """Code the papers one by one in random order, updating the shared codebook."""
    if stage not in (1, 2):
        raise typer.BadParameter("stage is 1 or 2")
    if (OUT / "codebook_final.json").exists():
        raise typer.BadParameter("The codebook is fixed already; nothing to code.")
    papers = corpus()
    out = stage_dir(stage)
    keys = order(f"stage{stage}", list(papers.index))
    book_path, codings_path = out / "codebook.json", out / "codings.jsonl"
    if book_path.exists():
        book = read_json(book_path)
    elif stage == 1:
        book = {"codes": {}, "aliases": {}, "retired": {}}
    else:
        previous = stage_dir(1)
        done1 = {c["key"] for c in read_jsonl(previous / "codings.jsonl")}
        if done1 != set(keys):
            raise typer.BadParameter("Finish stage 1 first.")
        book = read_json(previous / "codebook.json")
    done = {c["key"] for c in read_jsonl(codings_path)}
    todo = [k for k in keys if k not in done][:limit]
    typer.secho(f"Stage {stage}: {len(done)} of {len(keys)} papers coded, "
                f"{len(book['codes'])} codes; coding {len(todo)} now.")

    template, schema = (STAGE1, STAGE1_SCHEMA) if stage == 1 else (STAGE2, STAGE2_SCHEMA)
    for i, key in enumerate(todo, len(done) + 1):
        section, _ = paper_text(key, papers.loc[key])
        prompt = template.format(aim=AIM, rules=CODING_RULES, n=len(book["codes"]),
                                 codebook=render(book), paper=section)
        if show_prompt:
            typer.echo(prompt)
            return
        result, usage = call(prompt, schema)
        events = []

        def log(event, events=events, key=key):
            events.append(event)
            append_jsonl(out / "history.jsonl", {"key": key, **event})

        applied = apply_codes(book, stage, result["codes"], log)
        apply_revisions(book, result.get("revisions", []), log)
        append_jsonl(codings_path, {
            "key": key, "codes": applied, "revisions": result.get("revisions", []),
            "memo": result["memo"], "codebook_size": len(book["codes"]), "usage": usage,
        })
        write_json(book_path, book)  # after the coding, so a crash only redoes the paper
        changes = Counter(e["event"] for e in events)
        typer.secho(f"[{i}/{len(keys)}] {key}: {len(applied)} codes, "
                    f"{dict(changes) or 'no codebook changes'}; "
                    f"codebook {len(book['codes'])}")


# ---- fixing the codebook ---------------------------------------------------------------
@app.command()
def freeze():
    """Retire codes unused in stage 2, consolidate the codebook once, and fix it."""
    final = OUT / "codebook_final.json"
    if final.exists():
        raise typer.BadParameter(f"{final} exists; the codebook is fixed already.")
    out = stage_dir(2)
    codings = read_jsonl(out / "codings.jsonl")
    if {c["key"] for c in codings} != set(corpus().index):
        raise typer.BadParameter("Finish stage 2 first.")
    book = read_json(out / "codebook.json")
    rng = random.Random(f"{SEED}-freeze")
    log_path = OUT / "freeze_history.jsonl"

    stats = code_stats(book, codings, rng)
    for name in [n for n, s in stats.items() if s["papers"] == 0]:
        book["retired"][name] = book["codes"].pop(name)
        append_jsonl(log_path, {"event": "retire", "code": name,
                                "reason": "not applied in stage 2"})
    typer.secho(f"Retired {len(book['retired'])} stale codes; "
                f"consolidating {len(book['codes'])}.")

    stats = code_stats(book, codings, rng)
    result, usage = call(
        CONSOLIDATE.format(aim=AIM, n=len(book["codes"]),
                           codebook=render(book, stats, rng)),
        CONSOLIDATE_SCHEMA,
    )
    append_jsonl(log_path, {"event": "consolidation response", "result": result,
                            "usage": usage})
    for m in result["merges"]:
        codes = [c for c in (resolve(book, n) for n in m["codes"]) if c]
        if len(set(codes)) < 2:
            append_jsonl(log_path, {"event": "merge skipped", "merge": m,
                                    "reason": "fewer than two known codes"})
            continue
        into = resolve(book, m["into"]) or m["into"].strip()
        if into not in book["codes"]:
            book["codes"][into] = {"focus": m["focus"], "definition": m["definition"],
                                   "created": "freeze"}
        for c in set(codes):
            merge(book, c, into, m["definition"], m["focus"])
        append_jsonl(log_path, {"event": "merge", "codes": codes, "into": into,
                                "definition": m["definition"], "reason": m["reason"]})
    for r in result["redefinitions"]:
        name = resolve(book, r["code"])
        if name and r["definition"].strip():
            book["codes"][name]["definition"] = r["definition"].strip()
            append_jsonl(log_path, {"event": "redefine", "code": name, **r})

    # stage-2 codings in terms of the fixed codebook, one entry per paper and code
    papers = corpus()
    with (OUT / "codings_final.jsonl").open("w") as f:
        for coding in codings:
            _, source = paper_text(coding["key"], papers.loc[coding["key"]])
            codes = {}
            for c in coding["codes"]:
                name = resolve(book, c["code"])
                entry = codes.setdefault(name, {"code": name, "applications": []})
                entry["applications"].append({
                    "original_code": c["code"], "stance": c["stance"],
                    "position": c["position"],
                    "spans": [{"text": s, "verbatim": verbatim(s, source)}
                              for s in c["spans"]],
                })
            f.write(json.dumps({"key": coding["key"], "codes": list(codes.values())},
                               ensure_ascii=False) + "\n")
    book["frozen"] = True
    write_json(final, book)
    typer.secho(f"Fixed the codebook at {len(book['codes'])} codes: {final}")


# ---- thematic analysis -----------------------------------------------------------------
def final_stats(book: dict, codings: list[dict], rng: random.Random) -> dict:
    """code_stats for codings in terms of the fixed codebook."""
    flat = [{"key": c["key"], "codes": [
        {"code": code["code"], **a} for code in c["codes"] for a in code["applications"]
    ]} for c in codings]
    return code_stats(book, flat, rng)


@app.command()
def themes():
    """Group the fixed codes into themes (codes shown in random order)."""
    book, codings = final_codings()
    rng = random.Random(f"{SEED}-themes")
    stats = final_stats(book, codings, rng)
    prompt = THEMES.format(aim=AIM, n=len(book["codes"]),
                           codebook=render(book, stats, rng))
    for attempt in range(3):
        result, usage = call(prompt, THEMES_SCHEMA)
        assigned, problems = {}, []
        for theme in result["themes"]:
            for name in theme["codes"]:
                code = resolve(book, name)
                if code is None:
                    problems.append(f"unknown code {name!r}")
                elif code in assigned:
                    problems.append(f"{code!r} is in two themes")
                else:
                    assigned[code] = theme["name"]
        problems += [f"{c!r} is in no theme" for c in book["codes"] if c not in assigned]
        if not problems:
            break
        typer.secho(f"Attempt {attempt + 1}: {len(problems)} problems, "
                    f"e.g. {problems[:3]}", fg="yellow")
        prompt += ("\n\n## Your previous answer had these problems; fix them\n"
                   + "\n".join(f"- {p}" for p in problems))
    for theme in result["themes"]:
        theme["codes"] = [c for c in book["codes"] if assigned.get(c) == theme["name"]]
    unassigned = [c for c in book["codes"] if c not in assigned]
    if unassigned:
        result["themes"].append({"name": "Unassigned", "description": "",
                                 "codes": unassigned})
        typer.secho(f"{len(unassigned)} codes left unassigned.", fg="red")
    write_json(OUT / "themes.json", {"themes": result["themes"], "usage": usage})
    for theme in result["themes"]:
        typer.secho(f"{theme['name']} ({len(theme['codes'])} codes)")


@app.command()
def spans(limit: int = typer.Option(None, help="Stop after this many papers.")):
    """Record each paper's spans and position for every theme, in random order."""
    book, codings = final_codings()
    themes_ = read_json(OUT / "themes.json")["themes"]
    papers = corpus()
    keys = order("spans", list(papers.index))
    dest = OUT / "spans" / "theme_spans.jsonl"
    done = {r["key"] for r in read_jsonl(dest)}
    todo = [k for k in keys if k not in done][:limit]
    names = {t["name"].lower(): t["name"] for t in themes_}
    rendered = "\n".join(
        f"### {t['name']}\n{t['description']}\nCodes:\n"
        + "\n".join(f"- {c}: {book['codes'][c]['definition']}" for c in t["codes"])
        for t in themes_
    )
    by_key = {c["key"]: c for c in codings}
    lock = threading.Lock()

    def run(key):
        section, source = paper_text(key, papers.loc[key])
        codes = "\n".join(
            f"- {c['code']} ({a['stance']}): {a['position']}"
            for c in by_key[key]["codes"] for a in c["applications"]
        )
        result, usage = call(
            SPANS.format(aim=AIM, themes=rendered, codes=codes, paper=section),
            SPANS_SCHEMA)
        entries = []
        for t in result["themes"]:
            name = names.get(t["name"].strip().lower())
            if name is None:
                typer.secho(f"  {key}: unknown theme {t['name']!r} dropped", fg="yellow")
                continue
            entries.append({"theme": name, "position": t["position"], "spans": [
                {"text": s, "verbatim": verbatim(s, source)} for s in t["spans"]]})
        with lock:
            append_jsonl(dest, {"key": key, "themes": entries, "usage": usage})
            typer.secho(f"{key}: {len(entries)} themes")

    with ThreadPoolExecutor(SPAN_WORKERS) as pool:
        list(pool.map(run, todo))
    typer.secho(f"{len(read_jsonl(dest))} of {len(keys)} papers have theme spans.")


@app.command()
def representatives(k: int = typer.Option(8, help="Spans to select per theme.")):
    """Select the spans that best represent each theme, from all recorded spans."""
    themes_ = read_json(OUT / "themes.json")["themes"]
    records = read_jsonl(OUT / "spans" / "theme_spans.jsonl")
    if len(records) != len(corpus()):
        raise typer.BadParameter("Finish `spans` first.")
    rng = random.Random(f"{SEED}-representatives")
    anon = sorted({r["key"] for r in records})
    rng.shuffle(anon)
    anon = {key: f"P{i + 1:03d}" for i, key in enumerate(anon)}
    selected = {}
    for theme in themes_:
        pool = [
            {"key": r["key"], "position": t["position"], "text": s["text"]}
            for r in records for t in r["themes"] if t["theme"] == theme["name"]
            for s in t["spans"] if s["verbatim"]
        ]
        rng.shuffle(pool)
        for i, s in enumerate(pool):
            s["id"] = f"S{i + 1}"
        if not pool:
            selected[theme["name"]] = []
            continue
        listing = "\n".join(
            f"- [{s['id']}] {anon[s['key']]} (position: {s['position']}): \"{s['text']}\""
            for s in pool)
        prompt = REPRESENT.format(aim=AIM, k=min(k, len(pool)), name=theme["name"],
                                  description=theme["description"], spans=listing)
        result, _ = call(prompt, REPRESENT_SCHEMA)
        ids = {s["id"]: s for s in pool}
        selected[theme["name"]] = [
            {"key": ids[x["id"]]["key"], "text": ids[x["id"]]["text"],
             "position": ids[x["id"]]["position"], "why": x["why"]}
            for x in result["selected"] if x["id"] in ids
        ]
        typer.secho(f"{theme['name']}: {len(selected[theme['name']])} of {len(pool)} "
                    "spans")
    write_json(OUT / "representatives.json", selected)


# ---- report (the only place modes and fields appear) -----------------------------------
@app.command()
def report():
    """Write CSVs of codes, themes and spans joined with each paper's mode and field."""
    book, codings = final_codings()
    meta = E.load_classified()[["category", "Retrieval"]]
    meta = meta.rename(columns={"category": "mode", "Retrieval": "field"})
    theme_of = {}
    if (OUT / "themes.json").exists():
        for t in read_json(OUT / "themes.json")["themes"]:
            theme_of.update(dict.fromkeys(t["codes"], t["name"]))

    rows = [
        {"key": c["key"], "code": code["code"], "theme": theme_of.get(code["code"]),
         "focus": book["codes"][code["code"]]["focus"], "stance": a["stance"],
         "position": a["position"], "original_code": a["original_code"],
         "spans": " | ".join(s["text"] for s in a["spans"]),
         "spans_verbatim": all(s["verbatim"] for s in a["spans"])}
        for c in codings for code in c["codes"] for a in code["applications"]
    ]
    assignments = pd.DataFrame(rows).join(meta, on="key")
    assignments.to_csv(OUT / "assignments.csv", index=False)

    n_mode = meta.loc[assignments["key"].unique(), "mode"].value_counts()
    papers = assignments.drop_duplicates(["key", "code"])
    by_mode = pd.crosstab(papers["code"], papers["mode"])
    shares = (by_mode / n_mode).add_suffix(" share")
    codebook = pd.DataFrame.from_dict(book["codes"], orient="index")[
        ["focus", "definition"]]
    codebook["theme"] = codebook.index.map(theme_of)
    codebook["papers"] = papers["code"].value_counts()
    codebook = codebook.join(by_mode).join(shares).sort_values(
        ["theme", "papers"], ascending=[True, False])
    codebook.index.name = "code"
    codebook.to_csv(OUT / "codebook_final.csv")
    stances = pd.crosstab([assignments["code"], assignments["mode"]],
                          assignments["stance"])
    stances.to_csv(OUT / "stances_by_code_mode.csv")
    typer.secho(f"{len(assignments)} code applications on "
                f"{assignments['key'].nunique()} papers; "
                f"{assignments['spans_verbatim'].mean():.1%} with verbatim spans.")

    spans_path = OUT / "spans" / "theme_spans.jsonl"
    if spans_path.exists():
        rows = [
            {"key": r["key"], "theme": t["theme"], "position": t["position"],
             "span": s["text"], "verbatim": s["verbatim"]}
            for r in read_jsonl(spans_path) for t in r["themes"] for s in t["spans"]
        ]
        theme_spans = pd.DataFrame(rows).join(meta, on="key")
        reps = read_json(OUT / "representatives.json") \
            if (OUT / "representatives.json").exists() else {}
        chosen = {(r["key"], r["text"]) for v in reps.values() for r in v}
        theme_spans["representative"] = [
            (k, s) in chosen
            for k, s in zip(theme_spans["key"], theme_spans["span"], strict=True)]
        theme_spans.to_csv(OUT / "theme_spans.csv", index=False)
        tp = theme_spans.drop_duplicates(["key", "theme"])
        table = pd.crosstab(tp["theme"], tp["mode"])
        table.join((table / n_mode).add_suffix(" share")).to_csv(
            OUT / "themes_by_mode.csv")
        typer.secho(f"{len(theme_spans)} theme spans; "
                    f"{theme_spans['verbatim'].mean():.1%} verbatim.")
        typer.secho(table.to_string())
    typer.secho(f"Wrote the report to {OUT}")


@app.command()
def reset_test():
    """Delete the outputs of a test run (only when AISE_QUAL_VERSION starts with test)."""
    if not config.QUAL_VERSION.startswith("test"):
        raise typer.BadParameter("Set AISE_QUAL_VERSION=test... to use this.")
    shutil.rmtree(OUT, ignore_errors=True)


if __name__ == "__main__":
    app()
