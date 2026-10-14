"""Atlas AI Summit Q&A - landing page."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# See app/resources.py's module docstring: Streamlit does not put the repo
# root on sys.path by default, so `import rag` / `import shared` would fail
# without this, most visibly once deployed.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

BASELINE_PATH = ROOT / "evals" / "baselines" / "e2e_v2.json"
REPO_URL = "https://github.com/NickiMash17/inference-agent-challenge"

st.set_page_config(page_title="Atlas AI Summit Q&A")

st.title("Atlas AI Summit Q&A")

st.markdown(
    "This app answers natural-language questions about the **Atlas AI Summit** "
    "(18-19 November 2026, Johannesburg) - a fully fictional conference built as a "
    "Part 1 retrieval-augmented-generation benchmark. Everything in it - sessions, "
    "speakers, venue rooms, ticket tiers, and policies - is invented for this "
    "exercise; nothing refers to a real event, person, or organisation."
)

st.subheader("How it works")
st.markdown(
    "Every question is routed to one of four paths before it's answered:"
)
st.markdown(
    "- **ID lookup** - the question names a specific session (`S-009`), speaker "
    "(`SP-001`), or room (`R-04`) and isn't asking for a count; the matching chunk "
    "is pinned to the top of retrieval, with no LLM call needed just to decide the "
    "route."
)
st.markdown(
    "- **SQL** - the question needs a count, an average, or a \"most/least\" fact "
    "computed across many rows (e.g. *how many sessions are in the RAG track?*). A "
    "guarded, schema-aware SQL query is generated, validated against an allow-list "
    "before it ever runs, and executed against a DuckDB store with external file "
    "access disabled at the engine level."
)
st.markdown(
    "- **Retrieve** - the question names a specific entity or policy directly; a "
    "hybrid BM25 + vector index retrieves the most relevant chunks, with foreign-key "
    "expansion pulling in each session's linked room and speaker automatically."
)
st.markdown(
    "- **Rewrite + retrieve** - the question describes a topic indirectly, sharing "
    "no vocabulary with the session title it's really asking about. Three "
    "alternative phrasings are generated in conference-programme vocabulary first, "
    "then all four variants (the original plus the three rewrites) are retrieved "
    "and fused."
)
st.markdown(
    "Every answer is grounded in its retrieved context or SQL result and cites its "
    "sources; the model is told explicitly that the context it's given is data, "
    "never instructions to follow."
)

st.subheader("Headline eval results")

if BASELINE_PATH.exists():
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    by_type = baseline["by_answer_type"]
    total_correct = sum(v["correct"] for v in by_type.values())
    total_n = sum(v["n"] for v in by_type.values())

    col1, col2, col3 = st.columns(3)
    col1.metric("Correct (all types)", f"{total_correct}/{total_n}")
    col2.metric(
        "Route accuracy",
        f"{baseline['route_accuracy']['correct']}/{baseline['route_accuracy']['n']}",
    )
    col3.metric(
        "Citation validity",
        f"{baseline['citation_validity']['valid_or_refused']}/"
        f"{baseline['citation_validity']['n']}",
    )

    col4, col5, col6 = st.columns(3)
    col4.metric(
        "Refusals on negatives",
        f"{baseline['negative_refusal']['refused']}/{baseline['negative_refusal']['n']}",
    )
    col5.metric("p50 latency", f"{baseline['latency_ms']['p50']:.0f} ms")
    col6.metric("Total cost", f"${baseline['cost_usd']['total']:.4f}")

    st.caption(
        f"kb_version `{baseline['kb_version']}` · git `{baseline['git_commit_sha'][:12]}` · "
        f"router `{baseline['router_model']}` · answer `{baseline['answer_model']}` · "
        f"from `evals/baselines/{BASELINE_PATH.name}`"
    )

    with st.expander("Correct, by answer type"):
        for answer_type, counts in by_type.items():
            st.write(f"**{answer_type}**: {counts['correct']}/{counts['n']}")
else:
    st.info(f"No eval baseline found at `{BASELINE_PATH.relative_to(ROOT)}` yet.")

st.subheader("Learn more")
st.markdown(
    f"- [Full retrieval ablation study (v0-v7)]({REPO_URL}/blob/main/evals/ABLATION.md)\n"
    f"- [Source code on GitHub]({REPO_URL})\n\n"
    "Use the sidebar to open **Part 1: RAG** and try the chat interface."
)
