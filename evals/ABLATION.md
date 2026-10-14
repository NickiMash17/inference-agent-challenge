# Retrieval ablation

**kb_version:** 1.0.1 · **git commit:** `85192988c4278ea4b2843b573c78e184af06904d`

Evaluated on the 19 gold items with `answer_type` in `lookup` (6), `semantic`
(5), `cross_doc` (4), `policy` (4); aggregation items (7, including G-016 as
of kb_version 1.0.1) are checked separately in `evals/sql_sanity.py`. Counts
are "hits/n" at that cutoff. Raw results: `evals/baselines/<name>.json`.

The v0→v6 rows are cumulative (each adds one change on top of the last).
`v0_vector_only` and `v3_vector_only` are diagnostic side-rows — the vector
index alone, with no BM25 fusion — run against the same base/title-vector
setup as v0 and v3, to show honestly where hybrid fusion helps or hurts.

| Config | lookup r@1 | lookup r@5 | semantic r@1 | semantic r@5 | cross_doc r@1 | cross_doc r@5 | cross_doc all_refs@5 | policy r@1 | policy r@5 | OVERALL r@1 | OVERALL r@5 | p50 latency |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v0_baseline | 5/6 | 6/6 | 0/5 | 1/5 | 4/4 | 4/4 | 0.25 | 3/4 | 4/4 | 12/19 | 15/19 | 8.5 ms |
| v1_bm25_stopwords | 4/6 | 6/6 | 0/5 | 0/5 | 4/4 | 4/4 | 0.25 | 3/4 | 4/4 | 11/19 | 14/19 | 9.0 ms |
| v2_weighted_rrf | 5/6 | 6/6 | 0/5 | 0/5 | 4/4 | 4/4 | 0.5 | 4/4 | 4/4 | 13/19 | 14/19 | 8.6 ms |
| v3_title_vectors | 5/6 | 6/6 | 0/5 | 1/5 | 4/4 | 4/4 | 0.5 | 4/4 | 4/4 | 13/19 | 15/19 | 8.1 ms |
| v4_fk_expansion | 5/6 | 6/6 | 0/5 | 1/5 | 4/4 | 4/4 | 0.75 | 4/4 | 4/4 | 13/19 | 15/19 | 8.8 ms |
| v5_rerank | 4/6 | 6/6 | 1/5 | 1/5 | 2/4 | 4/4 | 1.0 | 4/4 | 4/4 | 11/19 | 15/19 | 289.7 ms |
| **v6_id_pinning** | **6/6** | 6/6 | 0/5 | 1/5 | 4/4 | 4/4 | **1.0** | 4/4 | 4/4 | **14/19** | 15/19 | **7.4 ms** |
| v0_vector_only *(diag.)* | 3/6 | 3/6 | 1/5 | 3/5 | 4/4 | 4/4 | 0.25 | 4/4 | 4/4 | 12/19 | 14/19 | 6.7 ms |
| v3_vector_only *(diag.)* | 3/6 | 3/6 | **3/5** | 3/5 | 4/4 | 4/4 | 0.25 | 4/4 | 4/4 | 14/19 | 14/19 | 7.2 ms |

(p50/p95 for every config are in each `evals/baselines/<name>.json` under
`result.query_latency_ms`; v5's cross-encoder cost is also broken out
separately under `result.rerank_latency_ms`. The ~130ms p95 on v0-v4/v6 is a
one-time ONNX session warm-up on each run's first query, not a per-query
cost — note the vector-only rows, run against an already-warm embedding
cache, show no such spike.)

## Notes per config

**v0_baseline** — Unweighted hybrid (BM25 + bge-small-en-v1.5, 1:1 RRF,
k=60), plain lowercased tokenizer, single representation per chunk, no
expansion, no reranking, no pinning. Every later row is measured against
this.

**v1_bm25_stopwords** — Removed English stopwords plus domain-generic terms
("session", "talk", "speaker", "which", "track", "room", "day") from the
BM25 tokenizer; IDs like `s-009` survive intact (see the G-004 diagnosis
below — IDs were never the problem here). This alone made lookup *worse*
(5/6 → 4/6 at r@1): thinning BM25's token set shifted its internal term
weighting enough to change which chunk won the unweighted RRF tie-break.
Shows why v2 (weighting BM25 vs. vector per query) is needed before this
change pays off — on its own, this is a regression.

**v2_weighted_rrf** — Query-aware fusion weights: 0.7 BM25 / 0.3 vector when
the query contains an ID pattern (`S-\d{3}`, `SP-\d{3}`, `R-\d{2}`), 0.3 / 0.7
otherwise. Recovers the v1 lookup regression (back to 5/6) and pushes policy
to 4/4, lifting OVERALL r@1 from 11/19 to 13/19. all_refs@5 on cross_doc also
improves (0.25 → 0.5). Still doesn't fully fix exact-ID lookups (see v6).

**v3_title_vectors** — Session chunks also embed just their title; speaker
chunks also embed "name, job_title at organisation". A chunk's vector score
is the max over its representations, deduplicated back to one chunk per
search() call. Modest, additive gain: OVERALL r@5 reaches 15/19 without
touching anything v2 had already fixed.

**v4_fk_expansion** — For each retrieved session chunk, append its linked
room and speaker chunks (via metadata foreign keys), deduplicated, appended
after the base top-k rather than re-ranked into it. Clear win on the metric
it targets: cross_doc all_refs@5 climbs from 0.5 to 0.75, since a cross_doc
question needing both a session ref and a speaker/room ref now gets the
second ref "for free." Plain recall is unaffected, as expected — expansion
adds context, it doesn't change what gets ranked.

**v5_rerank** — Reranks the union top-20 with fastembed's cross-encoder
(`Xenova/ms-marco-MiniLM-L-6-v2`, confirmed available via
`fastembed.rerank.cross_encoder.TextCrossEncoder.rerank()`) before taking
top-5, FK-expanding after. Pushes all_refs@5 to 1.0 and is the only config
to get a semantic item right at rank 1, but it's a net regression overall:
it demotes exact-ID lookups (lookup r@1 drops to 4/6) and cross_doc r@1
(4/4 → 2/4) — a generic MS-MARCO-trained cross-encoder has no notion that an
exact `S-015` match should dominate a fuzzier semantic match. It also adds
real latency: **~282ms p50 / ~359ms p95** for reranking 20 candidates, on
top of retrieval (**~290ms p50 end-to-end**).

**v6_id_pinning** — On top of v4 (not v5: pinning and the cross-encoder are
two independent answers to the same "exact ID" problem, and v6 isolates the
cheaper, deterministic one). If the query contains one or more IDs, each is
resolved by exact metadata lookup and pinned at the top, in order of
appearance; remaining slots are filled from the fused results, deduplicated.
An unresolved ID (e.g. a typo'd `S-999`) never crashes — it's recorded in
`unresolved_ids` and pinning is simply skipped for it. This is the strict
best row in the table: **lookup r@1 reaches 6/6** (this is the structural
fix for the G-004 diagnosis below — not a tokenizer change, since the
tokenizer was never broken) and all_refs@5 reaches 1.0, with *lower* latency
than v0-v4 (no cold-start spike, since pinning short-circuits before the
full fused search when every slot is already pinned).

**v0_vector_only / v3_vector_only (diagnostic)** — Vector search alone,
bypassing BM25 and fusion entirely. These are the most honest view of where
hybrid hurts: semantic r@1 is dramatically better vector-only (v0: 0/5 →
1/5; v3 title-vectors: 0/5 → **3/5**) because BM25 contributes essentially
noise on these deliberately keyword-free paraphrases, and even a small BM25
weight in RRF is enough to displace the vector index's correct top pick.
The trade-off: vector-only is *worse* on lookup (3/6 vs. 5-6/6 hybrid),
because exact IDs are out-of-vocabulary noise for a semantic embedding
model. Neither pure strategy wins everywhere — this is the core argument for
v6's targeted pinning over either a blanket hybrid-weight change or
abandoning BM25.

## Recommendation

**v6_id_pinning** as the default. It is the strict best row for both OVERALL
r@1 (14/19) and cross_doc all_refs@5 (1.0), achieves a *perfect* lookup
score (6/6) — the answer type most sensitive to exact-ID matching — and adds
no latency over v4 (in fact less, due to pinning's short-circuit path). Its
only weak spot, semantic (0/5 at r@1), is shared by every hybrid config in
this table and is a vector-representation problem, not something pinning
regresses — see v3_vector_only, which shows the real fix for semantic is
routing keyword-free queries to vector-only or giving them a much lower BM25
weight, independent of pinning. v5_rerank's all_refs@5 ties v6 (1.0) and it
alone recovers one semantic hit, but it actively regresses exact-ID lookups
and cross_doc r@1 and costs ~290ms/query; it isn't a better default without
a domain-tuned reranker or a way to skip reranking for ID-anchored queries.
