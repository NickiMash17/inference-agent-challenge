# Changelog

All notable changes to the Part 1 knowledge base are recorded here. Per the
freeze rule (see README "Knowledge base > Conventions"), any change to
`data/kb/` content or `evals/kb_gold.jsonl` requires a `KB_VERSION` bump in
`scripts/generate_kb.py` and a re-run of all evals.

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
  ORDER BY query joined against speakers for the job title — the same shape
  as `G-010`, not a simple two-hop lookup. Added its hand-written SQL to
  `evals/sql_sanity.py` (confirmed passing).
- `sessions.csv`, `speakers.json`, and `venue_guide.pdf` are byte-identical
  to 1.0.0 — only `evals/kb_gold.jsonl`'s label changed, so `manifest.json`
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
