from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from legal_chatbot.pipeline import LegalChatbotPipeline  # noqa: E402
from legal_chatbot.config import settings  # noqa: E402

ICON = "⚖️"
st.set_page_config(page_title="Drift-Aware Legal AI Chatbot", page_icon=ICON, layout="wide")

st.title(f"{ICON} Drift-Aware Legal AI Chatbot")
st.caption("UK legal and regulatory assistant with retrieval grounding and drift monitoring.")


@st.cache_resource
def get_pipeline() -> LegalChatbotPipeline:
    return LegalChatbotPipeline()


pipeline = get_pipeline()

with st.sidebar:
    st.subheader("Configuration")
    st.write(f"**Provider:** {settings.llm_provider}")
    st.write(f"**Embedding model:** {settings.embedding_model}")
    st.write(f"**Vector search:** {pipeline.retriever.search_backend}")
    st.write(f"**Top-k retrieval:** {settings.top_k}")
    st.write(f"**Confidence threshold:** {pipeline.min_confidence:.2f}")
    st.info(
        "This system is a portfolio demonstration and decision-support prototype. "
        "It does not constitute legal advice."
    )

default_question = "What are the key compliance considerations around data protection and consumer duty?"
question = st.text_area("Enter your legal or regulatory query", value=default_question, height=120)

if st.button("Generate response", type="primary"):
    with st.spinner("Retrieving sources and generating answer..."):
        result = pipeline.ask(question)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Answer")
        if result.low_confidence:
            st.warning(
                "Low retrieval confidence. No answer was generated; the closest source "
                "extracts are shown instead so you can check them yourself."
            )
        st.write(result.answer)

    with col2:
        st.subheader("Drift monitoring")
        dm = result.drift_metrics
        st.metric(
            "Top-1 similarity",
            f"{dm['top1_score']:.3f}",
            help=f"Answers are only generated at or above {dm['min_confidence']:.2f}.",
        )
        status = dm["status"]
        if status == "no_baseline":
            st.info("No baseline yet. Run scripts/build_index.py to create one.")
        elif status == "insufficient_data":
            st.info(
                f"Drift check needs {dm['needed']} queries to be meaningful; "
                f"{dm['n']} so far. PSI on a handful of queries is noise."
            )
        else:
            st.metric("PSI (recent queries vs baseline)", f"{dm['psi']:.3f}")
            if status == "significant":
                st.warning("Retrieval scores have shifted from baseline. Review the corpus and queries.")
            elif status == "moderate":
                st.info("Moderate shift from baseline. Keep an eye on it.")
            else:
                st.success("Retrieval behaviour is consistent with baseline.")

    st.subheader("Retrieved passages")
    for idx, passage in enumerate(result.passages, start=1):
        with st.expander(f"[{idx}] {passage['title']} | score={passage['score']:.4f}"):
            st.write(passage["content"])
            st.caption(f"Source file: {passage['source']}")
