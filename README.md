# Inference Agent Challenge

A two-part AI engineering challenge built against a fictional conference,
the Atlas AI Summit. Part 1 is a RAG system over structured documents
(CSV, JSON, PDF) with query routing between ID lookup, text-to-SQL, and
retrieval. Part 2 is an n8n agent that reconciles messy RSVP emails and
calendar invites against a speaker directory.

- **Live demo (Part 1):** https://inference-agent-challengegit-b83aw97ythualhczhx5zwf.streamlit.app
- **Technical summary:** [`docs/TECHNICAL_SUMMARY.md`](docs/TECHNICAL_SUMMARY.md)

## Review in 10 minutes

1. Open the [live demo](https://inference-agent-challengegit-b83aw97ythualhczhx5zwf.streamlit.app),
   go to "Part 1: RAG", and try a few of the sidebar's "Try these" example
   questions. They are picked from the gold eval set, one per route (ID
   lookup, SQL, retrieve, rewrite + retrieve), plus the planted traps: the
   cancelled session, the moved session, and a negative question with no
   answer in the knowledge base.
2. The 3 files most worth reading for code quality:
   - [`rag/pipeline.py`](rag/pipeline.py): the routing and answer
     orchestration.
   - [`rag/sql_agent.py`](rag/sql_agent.py): the guarded text-to-SQL layer
     (sqlglot validation, a locked-down DuckDB connection, a query
     timeout).
   - [`n8n/code/09_verify_and_dedupe.js`](n8n/code/09_verify_and_dedupe.js):
     Part 2's identity matching, correction, and ambiguity-flagging logic.
3. Results live in [`evals/ABLATION.md`](evals/ABLATION.md) (the retrieval
   configuration comparisons) and [`evals/baselines/`](evals/baselines/)
   (the raw eval run JSON, one file per configuration).

## Part 1: RAG over structured documents

Every question is routed to the path that can answer it correctly, rather
than sent to one retriever: exact ID lookup, text-to-SQL for counts and
aggregations, hybrid BM25 and vector retrieval, or an LLM query rewrite
first for keyword-free questions. Retrieved session chunks are expanded
through their foreign keys, so a session's room and speakers come along
automatically. The answer model cites its sources, and code strips any
citation that is not actually present in the retrieved context.

I measured retrieval before adding an LLM, and chose each component by
comparing it against the one before it on the same 19 retrievable gold
questions. The full comparison is in
[`evals/ABLATION.md`](evals/ABLATION.md); the knowledge base layout and the
planted edge cases (confusable speaker names, a cancelled session, a moved
session, and others) are in [`docs/architecture.md`](docs/architecture.md).

**Results** (30 development questions):

| Question type | Correct |
| --- | --- |
| ID lookup | 6/6 |
| Aggregation (SQL) | 7/7 |
| Semantic (no shared keywords) | 5/5 |
| Cross-document | 4/4 |
| Policy (from the PDF) | 4/4 |
| Unanswerable (must refuse) | 4/4 |

Route accuracy was 29/30, citation validity 26/26, and false refusals 0.
Median latency is 3.7 seconds, p95 is 6.4 seconds, and a full 30-question
run costs about US$0.16. These numbers are from the development set the
system was tuned against, so they are optimistic; a sealed 15-question
held-out set exists to give the unbiased number (see Evaluation below).

**Held-out results** (15 unseen questions, run once at submission): 14/15
correct.

| Question type | Correct |
| --- | --- |
| ID lookup | 3/3 |
| Aggregation (SQL) | 3/3 |
| Semantic (no shared keywords) | 1/2 |
| Cross-document | 3/3 |
| Policy (from the PDF) | 2/2 |
| Unanswerable (must refuse) | 2/2 |

Route accuracy was 15/15 and citation validity 12/12. The one miss was a
semantic question where the system answered "not in the knowledge base"
instead of finding the right session: a false refusal, not a wrong or
fabricated answer, so it failed safe. Full result in
[`evals/baselines/e2e_heldout_final.json`](evals/baselines/e2e_heldout_final.json).

## Part 2: n8n agent with tools, extraction and verification

The agent turns 9 messy RSVP emails and 2 calendar events into a verified
list of 8 unique attendees, with every correction and flag recorded per
field. It reads Gmail and Google Calendar, fetches the published speaker
directory over HTTP, and calls Claude for extraction only; everything
after extraction (validation, normalisation, matching, deduplication,
scoring) is deterministic code, not model judgement. The workflow,
its Code node source, and a local offline test are in [`n8n/`](n8n/);
setup steps and the full node list are in [`n8n/README.md`](n8n/README.md).

On live Gmail and Calendar data, the agent caught **8 of 8 planted
errors**:

| Planted error | Expected behaviour | Result |
| --- | --- | --- |
| Messy phone format | Match on digits | Passed |
| Misspelled organisation | Correct to directory spelling | Passed |
| Missing email | Fill from directory (name and phone agree) | Passed |
| Same person RSVPs twice, name typo | Merge into one attendee | Passed |
| Jon vs John Mokoena with the other's organisation | Flag ambiguous, do not guess | Passed |
| Email differs from directory | Flag the mismatch | Passed |
| Person not in the directory | Add as new delegate | Passed |
| Organisation typo in a calendar event | Correct via directory and email | Passed |

The errors were planted in synthetic seed data with a written answer key
(`data/synthetic/part2_answer_key.json`), so the score is checked, not
claimed. The same logic also passes 8/8 offline, with no network and no
LLM, in `n8n/test_logic.mjs`.

## Running locally

```bash
uv sync                                      # runtime deps
cp .env.example .env                         # then fill in ATLAS_ANTHROPIC_API_KEY
uv run streamlit run app/Home.py
```

The app needs `data/kb/`, `evals/kb_gold.jsonl`, and
`evals/baselines/e2e_v2.json` to already exist (they are generated and
checked into this repo, not built by the app itself). The first load
builds the index, the DuckDB store, and the embedding model once (cached
via `st.cache_resource`), so it takes a few seconds longer than later
loads.

### Deploying to Streamlit Community Cloud

- Set `ATLAS_ANTHROPIC_API_KEY` as a root-level **secret** in the app's
  settings. Streamlit Cloud exposes these as environment variables, which
  `Settings` reads directly; a `st.secrets` fallback also exists for local
  development with `.streamlit/secrets.toml` instead of `.env`. Never
  commit `.streamlit/secrets.toml` (already gitignored).
- `requirements.txt` is what Streamlit Cloud installs from. It is
  generated from `pyproject.toml`'s runtime dependencies only (no
  `dev`/`data` groups):
  ```bash
  uv export --no-dev --no-group data --no-hashes --format requirements-txt -o requirements.txt
  ```
  Regenerate it after any change to the main `dependencies` list in
  `pyproject.toml`.
- Streamlit runs each page as its own script with that script's own
  directory on `sys.path`, not the repo root. `app/Home.py`,
  `app/pages/1_Part1_RAG.py`, and `app/resources.py` each insert the repo
  root into `sys.path` before importing `rag`/`shared`, so this works the
  same locally and once deployed. Verify with
  `uv run streamlit run app/Home.py` from the repo root.

## Evaluation

- `evals/kb_gold.jsonl` (about 30 items) and `evals/sql_sanity.py` /
  `evals/retrieval_eval.py` are the iteration loop: run these as often as
  you like while tuning the pipeline. `evals/retrieval_eval.py --config
  <name> [--save-baseline <name>]` evaluates one named config from
  `rag/pipeline.py`'s `CONFIGS` against `kb_gold.jsonl`; `evals/ABLATION.md`
  tracks the resulting comparison across configs.
- `evals/kb_heldout.jsonl` (about 15 items) is **evaluated once, at
  submission time, by a human, not during iteration**. It exists to catch
  overfitting to the gold set's specific phrasing, which it cannot do if
  it is used for tuning. `retrieval_eval.py --heldout` refuses to run
  unless `--final` is also passed and the `HELDOUT_CONFIRM` environment
  variable is set to the exact current `kb_version`: two independent,
  hard-to-trigger-by-accident confirmations. See [`CLAUDE.md`](CLAUDE.md)
  for the standing rule against running it at all outside that one real
  evaluation, including to test the guard itself (a mistake made once
  during this project's development, see `CHANGELOG.md` 1.0.2).
- Every eval result records `kb_version` (read from
  `data/kb/manifest.json`) and `git_commit_sha`, so a result can always be
  matched back to the exact KB content and code it was produced against
  (see `docs/architecture.md`).

## Design decisions

- **Contact details (email, phone) are excluded from chunk text.**
  `rag/ingest.py`'s speaker chunks include name, job title, organisation,
  country, bio, and session list, never `email` or `phone`, which stay in
  `Chunk.metadata` only. This is Q&A grounded in retrieved text, not a
  contact directory, and surfacing personal contact details to an LLM as
  answerable context is the wrong default even for fictional data. One
  consequence worth naming: an email or phone lookup question is
  **unanswerable by the RAG pipeline by design**, not a retrieval failure.
  `evals/kb_gold.jsonl` does not include one (an earlier draft did, see
  `CHANGELOG.md`), and any future gold or held-out item should ask about a
  chunk-text field instead if it is meant to be answerable via retrieval.
- **Route by question type instead of one retriever.** I measured vector
  search alone and hybrid search alone before adding routing, and neither
  wins everywhere (see `evals/ABLATION.md`): hybrid wins lookups, vector
  wins semantic questions. The router sends each question type to the path
  that actually answers it.
- **The LLM extracts, deterministic code verifies (Part 2).** Correction,
  matching, and deduplication are rules I can test, not model judgement.
  The LLM only turns an email body into structured fields; everything that
  decides what to do with those fields is plain code.

## Limitations

- Evaluation is a single run on small sets (30 development, 15 held-out).
  The models used do not expose a temperature setting, so answers can vary
  slightly between runs.
- The rewrite route takes about 9 seconds because it makes extra LLM
  calls; answers are not streamed.
- Free hosting sleeps when idle, so the first load after a quiet period
  can take 30 to 60 seconds.
- Part 2 phone numbers are normalised for matching but not always
  rewritten in the output rows, and the change log sometimes records
  "email filled from directory" when an email was already present from
  another source for the same attendee.
- Part 2 runs on demand from the n8n canvas, not on a schedule or trigger.

See [`docs/TECHNICAL_SUMMARY.md`](docs/TECHNICAL_SUMMARY.md) for what I
would improve with more time.

## Repo layout

```
app/          Streamlit UI for Part 1 (Home page, chat page, cached resources)
rag/          Part 1 engine: ingestion, retrieval, routing, SQL agent, pipeline
shared/       Config and the Anthropic client wrapper, shared by rag/ and app/
data/kb/      The frozen, generated knowledge base (see docs/architecture.md)
data/synthetic/  Part 2 seed data (RSVP emails, calendar events, answer key)
evals/        Gold and held-out sets, eval scripts, ABLATION.md, baselines/
n8n/          Part 2: the importable workflow, its Code node source, README
scripts/      generate_kb.py, the deterministic knowledge base generator
docs/         Architecture notes and the technical summary
tests/        pytest suite (network blocked by default, see tests/conftest.py)
```
