# Part 2: Atlas Attendee Agent (n8n)

An n8n workflow that reconciles RSVP emails and calendar invites for the
Atlas AI Summit against the speaker directory (`data/kb/speakers.json`),
flags the planted data-quality issues described in
`data/synthetic/part2_seed.md`, and scores itself against
`data/synthetic/part2_answer_key.json`.

This directory is self-contained: nothing in `rag/`, `app/`, or `shared/`
(Part 1) depends on it, and it depends on nothing in Part 1 except reading
two files over HTTP for convenience (see below).

## Importing the workflow

1. Open n8n, go to **Workflows -> Import from File**.
2. Select `n8n/atlas_attendee_agent.json`.
3. n8n will create 27 nodes. Nodes using Gmail, Google Calendar, or the
   Anthropic HTTP call will show a red "credential missing" badge until you
   create and assign the three credentials below.
4. Click **Execute workflow** (the Manual Trigger) to run it end to end.

## Credentials to create

Create these three, then open each node that uses one (see the table
below) and select it from the credential dropdown - the JSON references
credentials **by name only**; it does not and cannot contain the actual
key/token.

| Credential name          | Type                                | Used by                                                      |
|---------------------------|--------------------------------------|---------------------------------------------------------------|
| `Gmail account`           | Gmail (OAuth2)                       | Gmail - Get Many                                              |
| `Google Calendar account` | Google Calendar (OAuth2)             | Google Calendar - Get Many                                    |
| `Anthropic API`           | Header Auth (generic), header `x-api-key` set to your Anthropic API key | Anthropic - Extract (Attempt 1), Anthropic - Extract (Retry) |

For `Anthropic API`: **Credentials -> New -> Header Auth**, set the header
name to `x-api-key` and the value to your Anthropic API key. Do not use a
"Bearer" style auth type - Anthropic's Messages API expects the raw key in
the `x-api-key` header, not an `Authorization: Bearer ...` header.

## What you must configure by hand after importing

- **Assign all three credentials** (above) on their respective nodes - a
  fresh import never carries real credentials with it.
- **Gmail node's `q` filter** is set to `to:your-address+atlas@gmail.com`, a
  placeholder. Open the **Gmail - Get Many** node and change the `q` filter
  under its Filters section to the real address you seeded, for example
  `to:you+atlas@gmail.com`, matching whatever address you used in
  `data/synthetic/part2_seed.md`.
- **Google Calendar node's calendar** is set to `primary`. Change it if the
  seed events live on a different calendar.
- **The two `HTTP Request` nodes that fetch `speakers.json` and
  `part2_answer_key.json`** read from this repo's `main` branch on GitHub
  (raw.githubusercontent.com). If your changes aren't pushed yet, or you
  want to test against local edits, point these two nodes at a different
  URL (or swap them for a Read/Write File node pointed at your local
  checkout) before running.
- **Model name**: the Anthropic call uses `claude-haiku-5-5`. Change the
  `model` field in both HTTP Request nodes' JSON body if you want a
  different model.
- Nodes were authored by hand against n8n's documented node/parameter
  schemas (not verified against a live n8n install). After import, open
  each Code, IF, and Merge node once and save the workflow - this lets n8n
  normalize any parameter shape it doesn't recognise rather than silently
  keeping a stale default.

## Node list (in execution order)

Vendor nodes are minimal; all matching/scoring logic lives in Code nodes
(whose JavaScript is also saved individually under `n8n/code/`, in the
same order, so it can be read or edited without opening n8n).

1. **Manual Trigger**
2. **Gmail - Get Many** - `resource: message`, `operation: getAll`,
   `returnAll: true`, Simplify **off**, `q: to:your-address+atlas@gmail.com`
   (placeholder, set to your real address, see below)
3. **Build Extraction Prompt** (Code) - decodes each Gmail message's
   base64url body (text/plain, falling back to stripped text/html) and
   builds the first extraction prompt. The prompt carries only the subject
   and body text - never the Gmail "From" header - so the model
   structurally cannot use sender identity even if it wanted to.
4. **Anthropic - Extract (Attempt 1)** (HTTP Request) - `POST
   https://api.anthropic.com/v1/messages`, model `claude-haiku-5-5`,
   `anthropic-version: 2023-06-01`, Header Auth credential `Anthropic API`
5. **Validate & Shape (Attempt 1)** (Code) - parses the model's JSON
   (tolerates ```json fences), checks `name` is present and `email` (if
   given) is a plausible address
6. **Valid? (Attempt 1)** (IF) - routes on `$json.valid`
7. **Build Retry Prompt** (Code, false branch only) - shows the model its
   own invalid output and the validation error, asks it to fix it
8. **Anthropic - Extract (Retry)** (HTTP Request) - same call shape as
   step 4
9. **Validate & Shape (Retry)** (Code) - same validation as step 5
10. **Valid? (Retry)** (IF)
11. **Mark Extraction Failed** (Code, false branch only) - nulls out
    unreliable fields, tags `extraction_failed`
12. **Merge - Combine Valid Extractions** - the two IF "true" branches
    (first-try and after-retry)
13. **Merge - Combine All Email Candidates** - adds the extraction-failed
    branch back in
14. **Google Calendar - Get Many** - `resource: event`, `operation:
    getAll`, calendar `primary`, 2026-11-18 to 2026-11-20, query filter
    `Atlas AI Summit`
15. **Parse Calendar** (Code) - deterministic, no LLM: keeps only events
    whose title contains "Atlas AI Summit", parses `"- Name,
    Organisation"` lines from the description
16. **Merge - Combine Email + Calendar Candidates**
17. **Normalise** (Code) - trims/lowercases names, lowercases emails,
    converts SA-local phone formats to E.164 (`+27...`)
18. **HTTP Request - Fetch Speakers Directory** - `GET
    .../data/kb/speakers.json` (n8n auto-splits the top-level JSON array
    into one item per speaker)
19. **Tag Directory** (Code) - adds the same normalised fields to each
    speaker record so they compare like-for-like with candidates
20. **Merge - Combine Candidates + Directory**
21. **Verify & Dedupe** (Code) - the core matching logic:
    - exact normalised-email or normalised-phone match wins outright
    - otherwise, best name match (Levenshtein) and best org match
      (Levenshtein, on whitespace-squashed strings) are computed
      independently; if they point at *different* speakers, the row is
      `flagged_ambiguous` and **not** auto-resolved (this is what catches
      the Jon/John Mokoena identity collision)
    - a strong match with a differing org/name corrects it to the
      directory value; a strong match with a missing email fills it from
      the directory; a strong match with a *conflicting* email is
      `flagged_email_mismatch` rather than silently overwritten
    - no match at all is `new_delegate`
    - resolved rows are then deduped (by speaker ID, else by the
      ambiguous-candidate set, else by normalised email/phone) so the same
      person RSVPing by email and appearing on a calendar invite collapses
      into one attendee row with merged `sources`
22. **HTTP Request - Fetch Answer Key** - `GET
    .../data/synthetic/part2_answer_key.json`
23. **Tag Answer Key** (Code)
24. **Merge - Combine Attendees + Answer Key**
25. **Score vs Answer Key** (Code) - checks each of the 8 planted errors
    against the final attendee list and reports `"Planted errors caught: X
    / 8"` plus a per-error pass/fail list (visible on this node's output)
26. **Split Attendees for CSV** (Code) - flattens the attendee rows (joins
    array fields like `sources`/`changes`/`flags` into `"; "`-separated
    strings) for a clean CSV
27. **Convert to File - CSV** - final attendee list as a downloadable CSV

## Local, offline test (no n8n, no network, no LLM)

`n8n/test_logic.mjs` re-implements the normalise/verify/dedupe/score logic
from the Code nodes above as plain functions, and replaces the Anthropic
call with a small regex-based mock extractor. It reads
`data/kb/speakers.json` and the 9 seed emails directly (the emails are
copied inline from `data/synthetic/part2_seed.md`) - no network calls, no
n8n install required.

```bash
node n8n/test_logic.mjs
```

Prints the mock-extracted fields per email, the final deduplicated
attendee list, and the same `"Planted errors caught: X / 8"` score. This
exercises the matching/dedup/scoring logic in isolation from both n8n and
the LLM, so a change to that logic can be sanity-checked in under a
second.
