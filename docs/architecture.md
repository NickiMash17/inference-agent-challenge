# Architecture notes

This is the detailed reference for Part 1's knowledge base and its
generator. For the overall system design and results, read
[`TECHNICAL_SUMMARY.md`](TECHNICAL_SUMMARY.md). For the retrieval
configuration comparisons, read [`../evals/ABLATION.md`](../evals/ABLATION.md).

## Knowledge base

The Part 1 knowledge base is a synthetic, fully fictional dataset for a
fictional conference, the **Atlas AI Summit** (18 to 19 November 2026,
Johannesburg). I generate it with
[`scripts/generate_kb.py`](../scripts/generate_kb.py), and it lives in
`data/kb/`:

| File | Contents | Why this format |
| --- | --- | --- |
| `sessions.csv` | About 40 conference sessions: title, track, level, day, time, room, speaker(s), capacity, status, notes | Tabular, one row per session. That is the natural shape for counting, filtering, and aggregation queries, and it forces the RAG system to route these to a structured query path rather than semantic search |
| `speakers.json` | About 25 speaker profiles: name, organisation, job title, email, phone, country, bio, LinkedIn, session IDs | Nested, record-shaped data (a list per speaker) does not flatten cleanly into a single CSV row, and JSON is the common interchange format for this kind of profile data |
| `venue_guide.pdf` | A 4-page PDF with three tables: Rooms, Ticket Tiers, and Key Policies | Tests PDF table extraction specifically: policies and room facts live only in prose-adjacent tabular PDF content, not in the structured CSV/JSON, which forces a genuinely multi-format RAG pipeline |
| `manifest.json` | `kb_version`, the generator seed, and a SHA-256 per KB file (including `evals/kb_gold.jsonl` and `evals/kb_heldout.jsonl`) | Pins the dataset to a version, so silent content drift is caught by a test instead of a confusing eval regression later |

`evals/kb_gold.jsonl` holds about 30 gold question/answer pairs computed
programmatically from the generated data, never hand-written. Each item has:

- `answer_type`: one of `lookup`, `aggregation`, `cross_doc`, `policy`,
  `negative`, `semantic`.
- `expected_answer`: a human-readable answer, for eyeballing and for
  LLM-judged grading.
- `expected_value`: a canonical, machine-checkable value for exact-match
  grading. An `int` for counts, `"Yes"`/`"No"` for boolean facts, the exact
  title/status/room_id string for lookups, a `;`-joined sorted list of
  speaker IDs for multi-speaker answers, or the literal string `"NOT_IN_KB"`
  for every `negative` item.
- `source_refs`: pointers back to the file(s)/record(s) that justify the
  answer, for example `"sessions.csv#S-017"`, `"speakers.json#SP-004"`, or
  `"venue_guide.pdf#rooms"`. A bare filename, e.g. `"sessions.csv"`, is used
  for aggregation answers computed across the whole file rather than one
  row. `negative` items have no `source_refs` since nothing backs them.

### Conventions

- **Cancelled sessions are excluded from aggregation counts** (e.g. "how many
  sessions are going ahead on Day 1") unless the question is explicitly about
  cancellations (e.g. "how many sessions have a status of Cancelled").
  Moved sessions are still counted: they are going ahead, just in a
  different room. Aggregation questions are worded with "going ahead" (or
  similar) wherever this exclusion applies, so a reader does not have to
  guess which convention is in effect. This applies anywhere a count is
  derived from sessions, not only to items tagged `aggregation`. For
  example, "who has given the most sessions" also excludes a speaker's
  cancelled session(s).
- **LinkedIn handles never resemble real profiles.** `linkedin_handle`
  follows the fixed pattern `atlas-demo-<speaker_id, lowercase>` (e.g.
  `atlas-demo-sp-001`) or is `null`, never derived from the speaker's name.
- **`semantic` gold items paraphrase a session's topic** with no shared key
  terms from its title and no IDs in the question, so they can only be
  answered by understanding the topic rather than matching keywords. This
  tests semantic retrieval, not keyword search.
- **Averages are rounded to 1 decimal place** (not the nearest whole number).
  Any gold item whose `expected_value` is an average also carries a
  `"tolerance": 0.05` field, so a grader can accept values within that
  tolerance instead of requiring an exact float match.
- **Every "highest/most" gold question has a unique answer.** Session
  capacity is derived from its room's capacity but shaved down slightly for
  each later use of that room (the first use keeps the full capacity), so no
  two sessions can tie for the global-highest capacity. One speaker is
  deliberately given one more session than anyone else, so "which speaker
  has given the most sessions" always has exactly one correct answer. Both
  are enforced by a generator-time self-check (`sanity_check`) and by
  `tests/test_kb.py`, so a future change that reintroduces a tie fails
  loudly instead of silently producing an ambiguous gold answer.
- **No speaker carries more than one planted edge case.** Each trap below
  has its own reserved speaker(s) (`EDGE_CASE_SPEAKER_INDICES` in the
  generator), pairwise disjoint from every other trap, so for example the
  confusable-name pair is never also the top speaker or the co-presenter.
  Enforced by `sanity_check` and
  `tests/test_kb.py::test_planted_edge_cases_have_disjoint_speakers`.
- **KB changes require a `kb_version` bump and a re-run of all evals.**
  `data/kb/manifest.json` pins `kb_version`, the generator seed, and a
  SHA-256 per KB file (including `evals/kb_gold.jsonl`).
  `tests/test_kb.py::test_manifest_hashes_match_files` fails if any file's
  content no longer matches its recorded hash. This is a deliberate
  trip-wire so a content change cannot silently ship without bumping
  `KB_VERSION` in `scripts/generate_kb.py` and re-running every eval against
  the new data. Each bump gets an entry in [`../CHANGELOG.md`](../CHANGELOG.md).

### Planted edge cases

All enforced by `tests/test_kb.py` and a self-check inside the generator.
IDs are not stable across regenerations, so re-read `data/kb/sessions.csv`
and `speakers.json` after any change.

- **Confusable speaker names.** `SP-001` "Jon Mokoena" (Nova Analytics) and
  `SP-002` "John Mokoena" (Ironwood AI Research) are different people at
  different organisations, to test whether retrieval disambiguates
  correctly. This is their only role: neither is the top speaker, a
  co-presenter, or involved in the moved or non-accessible-room sessions.
- **A cancelled session.** One session has `status=Cancelled`, presented by
  a speaker who carries no other trap.
- **A moved session.** One session has `status=Moved`, with `notes`
  recording which room it moved from and to, presented by a speaker who
  carries no other trap.
- **A non-wheelchair-accessible room that is still in use.** `R-04`
  (Workshop 2) is the only room marked `wheelchair_accessible=No` in the
  PDF, and it hosts at least one `Scheduled` session, so "is this session's
  room accessible?" requires a genuine cross-document join rather than an
  assumption. That session's speaker carries no other trap. (Other,
  unplanted sessions may also land in R-04 by ordinary chance; the trap is
  the one dedicated speaker, not every R-04 booking.)
- **A co-presented session.** One session lists two speaker IDs
  (semicolon-separated) to test multi-speaker lookups; both speakers carry
  no other trap.
- **A unique top speaker.** One speaker presents 3 going-ahead sessions;
  everyone else tops out at 2 (15 of the 25 speakers present 2 or more
  sessions each). This speaker carries no other trap.
- **Inconsistent phone formatting.** Speaker phone numbers deliberately mix
  E.164 (`+27...`), local (`0...`), spaced, and dashed formats.
- **Negative questions.** The gold set includes questions with no answer in
  the knowledge base (e.g. Wi-Fi password, a "Day 3", a hotel discount
  code), to test hallucination resistance.
- All speaker emails use only `@example.com` / `@example.org`, and all
  people, organisations and other identifying details are entirely
  fictional.

### Regenerating the knowledge base

```bash
uv sync --group data
uv run python scripts/generate_kb.py
```

The generator uses a fixed random seed (`42`); re-running it produces a
byte-identical `sessions.csv`, `speakers.json`, and `venue_guide.pdf`
(verified in `tests/test_kb.py::test_generator_is_deterministic`). Each
table draws from its own independent `random.Random(f"{SEED}:<table>")`
(and, for speakers, its own seeded `Faker` instance), for example
`"42:speakers"` and `"42:sessions"`, rather than one shared stream, so
adding or changing a field on one table can never shift another table's
realized values. The venue tables have no randomness, so no RNG is needed
there. `tests/test_kb.py::test_speaker_field_changes_do_not_affect_sessions`
proves this by monkeypatching an extra random draw into speaker-bio
generation and asserting `sessions.csv` is byte-for-byte unchanged. The PDF
is built with `reportlab`'s `invariant` mode (both the global `rl_config`
flag and the `Canvas`), which fixes its `/ID` and suppresses build-time
metadata that would otherwise differ between runs.
`tests/test_pdf_extraction.py` extracts the PDF's tables with `pdfplumber`
and checks that headers and row counts match the generator's source data
exactly.

## Knowledge base versioning in eval results

Every eval result JSON (`evals/results/*.json`, `evals/baselines/*.json`)
records `kb_version` (read from `data/kb/manifest.json`) and
`git_commit_sha`, so a result can always be matched back to the exact KB
content and code it was produced against, never silently misattributed to
a version it did not actually run on.
