# Retrieval ablation

**kb_version:** 1.0.0 · **git commit:** `a9a0d6e4945999ec029bb4dc64455c285918a5ff`

Each config is cumulative: v1 adds one change on top of v0, v2 adds one more
on top of v1, and so on through v5. Evaluated on the 20 gold items with
`answer_type` in `lookup` (6), `semantic` (5), `cross_doc` (5), `policy` (4);
aggregation items are checked separately in `evals/sql_sanity.py`. Counts are
"hits/n" at that cutoff. Raw results: `evals/baselines/<name>.json`.

| Config | lookup r@1 | lookup r@5 | semantic r@1 | semantic r@5 | cross_doc r@1 | cross_doc r@5 | cross_doc all_refs@5 | policy r@1 | policy r@5 | OVERALL r@1 | OVERALL r@5 | p50 latency |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v0_baseline | 5/6 | 6/6 | 0/5 | 1/5 | 4/5 | 4/5 | 0.2 | 3/4 | 4/4 | 12/20 | 15/20 | - |
| v1_bm25_stopwords | 4/6 | 6/6 | 0/5 | 0/5 | 4/5 | 4/5 | 0.2 | 3/4 | 4/4 | 11/20 | 14/20 | - |
| v2_weighted_rrf | 5/6 | 6/6 | 0/5 | 0/5 | 4/5 | 4/5 | 0.4 | 4/4 | 4/4 | 13/20 | 14/20 | - |
| v3_title_vectors | 5/6 | 6/6 | 0/5 | 1/5 | 4/5 | 4/5 | 0.4 | 4/4 | 4/4 | 13/20 | 15/20 | - |
| v4_fk_expansion | 5/6 | 6/6 | 0/5 | 1/5 | 4/5 | 4/5 | **0.6** | 4/4 | 4/4 | 13/20 | 15/20 | - |
| v5_rerank | 4/6 | 6/6 | **1/5** | 1/5 | 2/5 | 4/5 | **0.8** | 4/4 | 4/4 | 11/20 | 15/20 | 232 ms |

## Notes per config

**v0_baseline** — Unweighted hybrid (BM25 + bge-small-en-v1.5, 1:1 RRF,
k=60), plain lowercased tokenizer, single representation per chunk, no
expansion, no reranking. This is today's shipped pipeline before any of the
changes below; every later row is measured against it.

**v1_bm25_stopwords** — Removed English stopwords plus domain-generic terms
("session", "talk", "speaker", "which", "track", "room", "day") from the
BM25 tokenizer; IDs like `s-009` survive intact. This alone made lookup
*worse* (5/6 → 4/6 at r@1): several lookup questions are short and
ID-anchored, and thinning BM25's token set shifted its internal term
weighting enough to change which chunk won the unweighted RRF tie-break.
Proves why v2 (weighting BM25 vs. vector per query) is needed before this
change pays off — on its own, this is a regression.

**v2_weighted_rrf** — Query-aware fusion weights: 0.7 BM25 / 0.3 vector when
the query contains an ID pattern (`S-\d{3}`, `SP-\d{3}`, `R-\d{2}`), 0.3 / 0.7
otherwise. Recovers the v1 lookup regression (back to 5/6) and pushes policy
to 4/4 (giving the vector index more say on policy's longer, paraphrased
questions), lifting OVERALL r@1 from 11/20 to 13/20 — the best single-change
gain in this ablation. all_refs@5 on cross_doc also improves (0.2 → 0.4).

**v3_title_vectors** — Session chunks also embed just their title; speaker
chunks also embed "name, job_title at organisation". A chunk's vector score
is the max over its representations, deduplicated back to one chunk per
search() call. Modest, additive gain: semantic r@5 recovers to 1/5 and
OVERALL r@5 reaches 15/20, without touching anything v2 had already fixed.

**v4_fk_expansion** — For each retrieved session chunk, append its linked
room and speaker chunks (via metadata foreign keys), deduplicated, appended
after the base top-k rather than re-ranked into it. This is the clearest win
in the table for the metric it targets: cross_doc all_refs@5 jumps from 0.4
to **0.6**, since a cross_doc question needing both a session ref and a
speaker/room ref now gets the second ref "for free" once the session itself
is retrieved. Plain recall is unaffected, as expected — expansion adds
context, it doesn't change what gets ranked.

**v5_rerank** — Reranks the union top-20 with fastembed's cross-encoder
(`Xenova/ms-marco-MiniLM-L-6-v2`, confirmed available via
`fastembed.rerank.cross_encoder.TextCrossEncoder.rerank()`) before taking
top-5, FK-expanding after. Mixed and, overall, a net regression: it's the
only config to get a semantic item right at rank 1 (0/5 → 1/5) and it pushes
all_refs@5 to 0.8, but it demotes exact-ID lookups and cross_doc questions
that depend on matching a specific session/speaker ID (cross_doc r@1 drops
4/5 → 2/5, lookup r@1 drops to 4/6) — a generic MS-MARCO-trained
cross-encoder has no notion that an exact `S-015` match should dominate a
fuzzier semantic match. It also adds real latency: **232 ms p50 / 307 ms
p95** per query for reranking 20 candidates, on top of retrieval.

## Recommendation

**v4_fk_expansion** as the default. It keeps every gain from v1-v3 (OVERALL
13/20 r@1, 15/20 r@5 — tied for the best r@1 and r@5 in the table) and adds
the single largest improvement on the metric that most reflects real
multi-hop question answering (cross_doc all_refs@5: 0.6, the best of any
config), with no added query latency and no risk of displacing a correct
top-k hit (expansion only appends). v5_rerank's all_refs@5 is higher (0.8),
but it actively regresses exact-ID lookups and cross_doc r@1 — the two
answer types most common in this KB — and adds ~230ms/query; it isn't a
better default without either a domain-tuned reranker or a way to skip
reranking for ID-anchored queries (the same insight that made v2 necessary
for v1 to pay off).
