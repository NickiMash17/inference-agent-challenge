# Changelog

All notable changes to the Part 1 knowledge base are recorded here, keyed
by `kb_version`. Per the freeze rule (see README "Knowledge base >
Conventions"), any change to `data/kb/` content or `evals/kb_gold.jsonl`
requires a `KB_VERSION` bump in `scripts/generate_kb.py` and a re-run of all
evals. Other notable project-wide fixes (not tied to a kb_version) are
recorded under "Unreleased" below.

## Unreleased

### Fixed

- **Tests could reach the live Anthropic API via `.env` loading.** A test
  simulating "API key absent" deleted the `ATLAS_ANTHROPIC_API_KEY`
  *process* environment variable, but `Settings` (pydantic-settings) reads
  `.env` directly as a second source. Once a real `.env` existed, that test
  no longer actually simulated "absent": `evals/e2e_eval.py`'s `main()` saw
  a real key, built a real `AnthropicClient`, and started a live eval run
  during a routine `uv run pytest` - caught because the command hung well
  past its normal runtime, not before it may have made one or more real,
  unbudgeted API calls. No `evals/results/e2e_*.json` or
  `evals/baselines/e2e_v1.json` was written, so no eval run completed, but
  the exact number of calls made before the process was killed is unknown.
  Fixed structurally, not by patching the one test:
  - `tests/conftest.py` adds two autouse fixtures (applied to every test not
    marked `@pytest.mark.live`): one forces
    `Settings.model_config["env_file"] = None` so no test can read the real
    `.env` regardless of whether an individual `Settings(...)` call
    remembers `_env_file=None`; the other makes constructing a real
    `anthropic.Anthropic` client raise immediately, so even a future code
    path that builds one directly (bypassing `shared.llm.AnthropicClient`)
    still can't reach the network from a test.
  - `evals/e2e_eval.py`'s missing-key and `--reps`/`--save-baseline` guards
    are now pure functions (`missing_api_key_error`, `reps_guard_error`,
    same pattern as `evals/retrieval_eval.py`'s `heldout_guard_error`),
    tested directly. No test in the suite calls `main()`.
  - `pytest-timeout` added with a 30s default per test, so a future hang
    fails loudly inside the test run instead of running silently until
    manually killed.

## 1.0.3 - Gold-item and scorer fixes

- **`G-003` replaced.** It asked for a speaker's email address, which is
  unanswerable by design (contact details are deliberately excluded from
  chunk text - see README "Design decisions"). Replaced with "What is the
  job title of speaker SP-009?" (same speaker, a field that's actually in
  the retrieved chunk text), computed from data.
- **`must_contain` added to all 4 `policy` gold items** (`G-018`-`G-021`):
  a short list of case-insensitive key facts (e.g. `"17:00"`,
  `"11 November 2026"`, `"dietary requirements"`), verified at generation
  time to actually appear in the policy's rule text. `evals/e2e_eval.py`
  now scores policy items on these facts rather than whole-sentence
  containment, so a correct paraphrase isn't marked wrong for not quoting
  the gold sentence verbatim.
- `sessions.csv`, `speakers.json`, and `venue_guide.pdf` are unchanged from
  1.0.2; only `kb_gold.jsonl` (5 lines: the `G-003` swap plus 4
  `must_contain` additions) and `manifest.json` differ.

### Fixed (evals/e2e_eval.py scorer, not a kb_version change on its own)

- **Number extraction mistook ID digits for the answer.** `_extract_number`
  used `-?\d[\d,]*\.?\d*`, which treated the hyphen in `SP-001` as a minus
  sign and matched `-001` before ever reaching the real number later in the
  sentence (seen live: `G-012`'s fully correct answer was scored wrong).
  Now strips recognized ID/date/time shapes (`S-\d{3}`, `SP-\d{3}`,
  `R-\d{2}`, `YYYY-MM-DD`, `HH:MM[-HH:MM]`) before searching, and a hyphen
  directly attached to a letter is never read as a minus sign.
- **List-valued `expected_value` (semicolon-separated IDs) required the
  literal ID string.** An answer naming co-speakers by full name rather
  than ID (the realistic LLM-prose case) was marked wrong even when fully
  correct. Now resolves each ID to its name/title from the KB
  (`load_entity_names`) and accepts either form, requiring every element of
  the list to be present as one or the other.

## 1.0.2 - Held-out set reset

Held-out set regenerated after an accidental evaluation run during
development. The previous set was retired; no tuning decisions used its
results.

- The kb_version 1.0.1 `evals/kb_heldout.jsonl` was evaluated once during
  development while testing the `--heldout`/`--final` CLI guard
  (`evals/retrieval_eval.py --heldout --final`), which should never have
  happened outside a real submission. That result has been deleted
  (`evals/results/retrieval_20261013T202555Z.json`).
- Replaced all 15 items with a fresh set in the same mix (3 lookup, 3
  aggregation, 3 cross_doc, 2 policy, 2 semantic, 2 negative), computed from
  the same frozen data, sharing no entity (session, speaker, or question)
  with either `evals/kb_gold.jsonl` or the retired 1.0.1 set.
- Strengthened the guard: `--heldout` now also requires the
  `HELDOUT_CONFIRM` environment variable to equal the current `kb_version`,
  on top of `--final`. Tests cover only refusal paths and never execute a
  held-out evaluation.
- Added `CLAUDE.md` at the repo root codifying this as a standing project
  rule: no evaluation of any kind against `kb_heldout.jsonl` outside a real,
  human-run submission.
- `sessions.csv`, `speakers.json`, `venue_guide.pdf`, and `kb_gold.jsonl` are
  unchanged from 1.0.1; only `kb_heldout.jsonl` and `manifest.json` differ.

## 1.0.1 - Gold label fix (patch, no data change)

- Relabelled `G-016` ("Which speaker has given the most sessions that are
  going ahead, and what is their job title?") from `answer_type: cross_doc`
  to `answer_type: aggregation`. It's a COUNT + GROUP BY + tie-broken
  ORDER BY query joined against speakers for the job title, the same shape
  as `G-010`, not a simple two-hop lookup. Added its hand-written SQL to
  `evals/sql_sanity.py` (confirmed passing).
- `sessions.csv`, `speakers.json`, and `venue_guide.pdf` are byte-identical
  to 1.0.0. Only `evals/kb_gold.jsonl`'s label changed, so `manifest.json`
  updates only the `kb_gold.jsonl` hash (plus the new `kb_heldout.jsonl`
  entry introduced in this version).
- Added `evals/kb_heldout.jsonl`: 15 items (3 lookup, 3 aggregation, 3
  cross_doc, 2 policy, 2 semantic, 2 negative), computed from the same
  frozen data but none paraphrasing an existing `kb_gold.jsonl` item.
  Reserved for a single evaluation at submission time (see README
  "Evaluation").

## 1.0.0 - Initial frozen knowledge base

- `data/kb/sessions.csv` (40 rows), `speakers.json` (25 records), and
  `venue_guide.pdf` (Rooms/Ticket Tiers/Key Policies tables) for the
  fictional Atlas AI Summit, generated deterministically by
  `scripts/generate_kb.py` (seed 42).
- `evals/kb_gold.jsonl`: 30 gold items (6 lookup, 6 aggregation, 5 cross_doc,
  4 policy, 4 negative, 5 semantic), computed programmatically from the
  generated data.
- Planted edge cases (confusable speaker names, a cancelled session, a moved
  session, a non-wheelchair-accessible room still in use, a co-presented
  session, a unique top speaker) each assigned to disjoint speakers, so no
  speaker carries more than one trap.
- `data/kb/manifest.json` introduced to pin the dataset: `kb_version`,
  generator seed, and a SHA-256 per KB file.
