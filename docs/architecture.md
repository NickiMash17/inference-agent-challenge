# Architecture Notes

TODO: architecture notes and diagrams for Part 1 (RAG) and Part 2 (Agent).

## Knowledge base versioning in eval results

`data/kb/manifest.json` pins the Part 1 knowledge base to a `kb_version`
(see README "Knowledge base > Conventions" for the freeze rule). Any eval
script written later (e.g. an `evals/` runner that scores retrieval against
`evals/kb_gold.jsonl`) **must** record the `kb_version` it ran against in its
results JSON — e.g. a top-level `"kb_version": "1.1.0"` field — so that a
past eval run can always be matched back to the exact KB content it was
scored against, and a result can't be silently misattributed to a KB version
it didn't actually run on.
