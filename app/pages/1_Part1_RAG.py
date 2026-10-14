"""Part 1: RAG - chat interface over the Atlas AI Summit knowledge base."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# See app/resources.py's module docstring: this page is run as its own
# script by Streamlit, from app/pages/, with neither app/ nor the repo root
# on sys.path by default.
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.resources import EVALS_DIR, get_app_resources  # noqa: E402
from rag.pipeline import ID_PATTERN  # noqa: E402
from rag.pipeline import answer as answer_query  # noqa: E402

MAX_CHARS = 500
MAX_QUESTIONS = 20

ROUTE_LABELS = {
    "id_lookup": "🔑 ID lookup",
    "sql": "🧮 SQL",
    "retrieve": "🔍 Retrieve",
    "rewrite_retrieve": "🔁 Rewrite + retrieve",
    "out_of_scope": "❓ Out of scope",
}


@st.cache_data
def load_gold_items() -> list[dict]:
    path = EVALS_DIR / "kb_gold.jsonl"
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pick_example_questions(gold_items: list[dict]) -> dict[str, str]:
    """One example per route, plus the planted traps - all sourced from
    evals/kb_gold.jsonl (never the held-out set, per CLAUDE.md)."""

    def first(predicate, fallback_type: str | None = None) -> str | None:
        for item in gold_items:
            if predicate(item):
                return item["question"]
        if fallback_type:
            for item in gold_items:
                if item["answer_type"] == fallback_type:
                    return item["question"]
        return None

    candidates = {
        "🔑 An ID lookup": first(
            lambda i: i["answer_type"] == "lookup" and ID_PATTERN.search(i["question"])
        ),
        "🧮 A count question": first(lambda i: i["answer_type"] == "aggregation"),
        "🧭 A keyword-free topic": first(lambda i: i["answer_type"] == "semantic"),
        "🏢 A cross-document accessibility question": first(
            lambda i: i["answer_type"] == "cross_doc" and "accessible" in i["question"].lower()
        ),
        "🚫 The cancelled session": first(lambda i: i["expected_value"] == "Cancelled"),
        "📦 The moved session": first(lambda i: "moved" in i["question"].lower()),
        "🙅 A negative question": first(lambda i: i["answer_type"] == "negative"),
    }
    return {label: question for label, question in candidates.items() if question}


def is_over_question_limit(question_count: int) -> bool:
    return question_count >= MAX_QUESTIONS


def is_over_char_limit(question: str) -> bool:
    return len(question) > MAX_CHARS


def build_meta(result) -> dict:
    return {
        "route": result.route,
        "citations": result.citations,
        "rewrites": result.rewrites,
        "sql": result.sql_result.sql if result.sql_result else None,
        "sql_columns": result.sql_result.columns if result.sql_result else None,
        "sql_rows": result.sql_result.rows if result.sql_result else None,
        "latency_s": sum(result.stage_latencies_s.values()),
        "cost_usd": result.cost_usd,
        "not_in_kb": result.not_in_kb,
        "low_confidence": result.low_confidence,
    }


def render_answer_meta(meta: dict, chunk_by_ref: dict) -> None:
    badge = ROUTE_LABELS.get(meta["route"], meta["route"])
    cols = st.columns(3)
    cols[0].markdown(f"**Route**  \n{badge}")
    cols[1].markdown(f"**Latency**  \n{meta['latency_s'] * 1000:.0f} ms")
    cols[2].markdown(f"**Cost**  \n${meta['cost_usd']:.5f}")

    if meta["not_in_kb"]:
        st.info("🙅 Not in the knowledge base.")
    if meta["low_confidence"]:
        st.warning("⚠️ Low confidence: no citation could be verified against the context.")

    if meta["sql"]:
        st.caption("Generated SQL")
        st.code(meta["sql"], language="sql")
        if meta["sql_rows"] is not None:
            rows_as_dicts = [
                dict(zip(meta["sql_columns"], row, strict=True)) for row in meta["sql_rows"]
            ]
            st.dataframe(rows_as_dicts, width="stretch")

    if meta["rewrites"]:
        with st.expander(f"Rewrites used ({len(meta['rewrites'])})"):
            for i, rewrite in enumerate(meta["rewrites"], start=1):
                st.write(f"{i}. {rewrite}")

    if meta["citations"]:
        st.caption("Citations")
        for ref in meta["citations"]:
            chunk = chunk_by_ref.get(ref)
            with st.expander(ref):
                st.write(chunk.text if chunk else "_(not a chunk-backed citation)_")


def handle_question(question: str, resources: dict) -> None:
    st.session_state.question_count += 1
    st.session_state.messages.append({"role": "user", "content": question, "meta": None})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        meta = None
        answer_text = "Sorry, something went wrong answering that question. Please try again."
        try:
            with st.spinner("Thinking..."):
                result = answer_query(
                    question,
                    con=resources["con"],
                    schema=resources["schema"],
                    retrieval_pipelines=resources["pipelines"],
                    llm=resources["llm"],
                    router_model=resources["settings"].router_model,
                    answer_model=resources["settings"].answer_model,
                )
            answer_text = result.answer
            st.write(answer_text)
            meta = build_meta(result)
            render_answer_meta(meta, resources["chunk_by_ref"])
        except Exception:
            # Never a stack trace or any secret - a plain, user-safe message.
            st.error(answer_text)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer_text, "meta": meta}
    )


def main() -> None:
    st.set_page_config(page_title="Part 1: RAG", page_icon="💬")
    st.title("Part 1: RAG")
    st.caption("Ask about the Atlas AI Summit's sessions, speakers, venue, or policies.")

    resources = get_app_resources()

    if resources["llm"] is None:
        st.error(
            f"The app can't reach the Anthropic API: {resources['llm_error']} "
            "Set `ATLAS_ANTHROPIC_API_KEY` (in `.env` locally, or in Streamlit secrets) "
            "to use this page."
        )
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "question_count" not in st.session_state:
        st.session_state.question_count = 0

    limit_reached = is_over_question_limit(st.session_state.question_count)

    with st.sidebar:
        st.subheader("Try these")
        st.caption("Examples from the gold eval set, one per route plus the planted traps.")
        examples = pick_example_questions(load_gold_items())
        pending_question = None
        for label, question in examples.items():
            if st.button(label, width="stretch", disabled=limit_reached):
                pending_question = question
        st.divider()
        st.caption(
            f"Questions asked this session: {st.session_state.question_count}/{MAX_QUESTIONS}"
        )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and msg.get("meta"):
                render_answer_meta(msg["meta"], resources["chunk_by_ref"])

    if limit_reached and pending_question is None:
        st.info(
            f"You've reached the {MAX_QUESTIONS}-question limit for this browser session. "
            "Refresh the page to start a new session."
        )

    typed_question = st.chat_input(
        "Ask a question about the Atlas AI Summit...",
        max_chars=MAX_CHARS,
        disabled=limit_reached,
    )

    question = pending_question or typed_question
    if question and not limit_reached and not is_over_char_limit(question):
        handle_question(question, resources)


if __name__ == "__main__":
    main()
