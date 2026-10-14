# Project rules

These rules apply to any assistant or automated agent working in this
repository. They exist because of specific incidents during this project's
development (see `CHANGELOG.md`) — follow them exactly, not just in spirit.

## Held-out set

- **NEVER run any evaluation against `evals/kb_heldout.jsonl`, including to
  test the code path.** It runs once, by the human, at submission. This
  means: no `--heldout` runs, no `--heldout --final` runs, no "just
  checking the guard works" runs, no running it against a toy/mock config
  to verify behaviour. A held-out evaluation already happened by accident
  once during development (see `CHANGELOG.md` 1.0.2) — that set was
  retired and replaced. It must not happen again, for any reason, including
  verifying that the refusal guard itself works. Test the guard only via
  its pure logic function (`heldout_guard_error` in
  `evals/retrieval_eval.py`), never by invoking the CLI with `--heldout`.

## Git

- **NEVER commit or push.** The human commits. Leave the working tree as
  modified/untracked files and report what changed; do not run `git add`,
  `git commit`, or `git push` unless the human explicitly asks for that
  specific action in that specific message.

## Secrets

- **NEVER print, log, or echo secrets or `.env` contents.**
- The API key environment variable is **`ATLAS_ANTHROPIC_API_KEY`**. Never
  use or set `ANTHROPIC_API_KEY` — that name is not this project's key and
  must not appear in code, config, or shell commands here.

## Knowledge base versioning

- Any change to `data/kb/` content, `evals/kb_gold.jsonl`, or
  `evals/kb_heldout.jsonl` requires, together, in the same change:
  1. a `KB_VERSION` bump in `scripts/generate_kb.py`,
  2. a manifest regeneration (`uv run python scripts/generate_kb.py`), and
  3. a new entry in `CHANGELOG.md` describing what changed and why.
- Every eval result JSON (`evals/results/*.json`, `evals/baselines/*.json`)
  must record `kb_version`, `git_commit_sha`, and the model name(s) used
  (e.g. `embedding_model`), so a result can always be matched back to the
  exact KB content, code, and models it was produced against.

## Reporting results

- Report evaluation results as **counts**, e.g. `"3/5"`, not only as ratios
  or percentages. A bare ratio hides the sample size.

## Environment

- Windows, PowerShell, `uv` for Python dependency management and running
  scripts (`uv run python ...`, `uv sync`).
- Run `ruff check .` and `pytest` (via `uv run`) before reporting any task
  as complete.

## When a rule blocks a task

- **If a rule in this file blocks what a task is asking for, stop and
  ask — don't work around it.** This includes finding a technically-
  different way to achieve the same blocked outcome (e.g. a "dry run" that
  still reads `kb_heldout.jsonl`'s contents, or scripting around the
  `--heldout`/`--final`/`HELDOUT_CONFIRM` guard). If the rule seems wrong
  for a specific case, say so and ask, rather than deciding unilaterally
  that the exception is justified.
