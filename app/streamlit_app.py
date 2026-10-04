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

st.set_page_config(page_title="Drift-Aware Legal AI Chatbot", page_icon="⚖️", layout="wide")

st.title("⚖️ Drift-Aware Legal AI Chatbot")
st.caption("UK legal and regulatory assistant with retrieval grounding and drift monitoring.")

with st.sidebar:
    st.subheader("Configuration")
    st.write(f"**Provider:** {settings.llm_provider}")
    st.write(f"**Embedding model:** {settings.embedding_model}")
    st.write(f"**Top-k retrieval:** {settings.top_k}")
    st.info(
        "This system is intended for academic demonstration and decision support. "
        "It does not constitute legal advice."
    )

@st.cache_resource
def get_pipeline() -> LegalChatbotPipeline:
    return LegalChatbotPipeline()

pipeline = get_pipeline()

default_question = "What are the key compliance considerations around data protection and consumer duty?"
question = st.text_area("Enter your legal or regulatory query", value=default_question, height=120)

if st.button("Generate response", type="primary"):
    with st.spinner("Retrieving sources and generating answer..."):
        result = pipeline.ask(question)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Answer")
        st.write(result.answer)

    with col2:
        st.subheader("Drift monitoring")
        st.metric("PSI", f"{result.drift_metrics['psi']:.4f}")
        st.metric("KL divergence", f"{result.drift_metrics['kl_divergence']:.4f}")
        if result.drift_metrics["psi"] > 0.25:
            st.warning("Potential drift signal detected. Review the corpus and retrieval behaviour.")
        else:
            st.success("No strong drift signal detected for this retrieval pattern.")

    st.subheader("Retrieved passages")
    for idx, passage in enumerate(result.passages, start=1):
        with st.expander(f"[{idx}] {passage['title']} | score={passage['score']:.4f}"):
            st.write(passage["content"])
            st.caption(f"Source file: {passage['source']}")
