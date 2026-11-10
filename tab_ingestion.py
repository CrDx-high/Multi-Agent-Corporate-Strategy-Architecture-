"""Tab 1: Document Ingestion and Vector Indexing interface."""

import os
import tempfile
import streamlit as st

from ingestion import clear_pointers, ingest_docx
from schemas import DocumentPointer
from ui.async_utils import run_async
from ui.components import active_doc_card_html, section_tag_html


def render_ingestion_tab() -> None:
    """Renders Phase 1: Document Ingestion and Vector Indexing."""
    st.html(section_tag_html("Document Ingestion & Vector Indexing"))
    st.markdown(
        "Upload corporate <code>.docx</code> strategic plans, governance policies, and financial projections. "
        "The ingestion pipeline extracts section hierarchies and table matrices (budgets, KPIs), "
        "and indexes vector embeddings into ChromaDB for contextual retrieval.",
        unsafe_allow_html=True,
    )

    active_hash = st.session_state.get("latest_doc_hash")
    active_name = st.session_state.get("active_doc_name")
    latest_pointers: list[DocumentPointer] = st.session_state.get("latest_pointers", [])

    if active_hash and latest_pointers:
        st.html(active_doc_card_html(active_name or "Uploaded Document", active_hash, len(latest_pointers)))

        col_unload, _ = st.columns([1.5, 3.5])
        with col_unload:
            if st.button("Unload Active Document", use_container_width=True):
                clear_pointers(doc_hash=active_hash)
                st.session_state.pop("latest_pointers", None)
                st.session_state.pop("latest_doc_hash", None)
                st.session_state.pop("active_doc_name", None)
                st.session_state.pop("consensus_result", None)
                st.success("Active document unloaded from memory and vector vault.")
                st.rerun()

    uploaded_file = st.file_uploader(
        "Upload Strategic Word Document (.docx)",
        type=["docx"],
        help="Upload corporate strategy, M&A, or planning .docx files.",
    )

    col1, _ = st.columns([1.5, 2.5])
    with col1:
        replace_vault = st.checkbox(
            "Replace existing documents in vector vault",
            value=True,
            help="Purges previous document embeddings from ChromaDB so only this document is used.",
        )
        if uploaded_file is not None:
            if st.button("Process & Embed Document", type="primary", use_container_width=True):
                with st.spinner("Processing document structure and indexing into vector database..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    try:
                        if replace_vault:
                            clear_pointers()
                        pointers = run_async(ingest_docx(tmp_path))
                        st.session_state["latest_pointers"] = pointers
                        st.session_state["latest_doc_hash"] = (
                            pointers[0].doc_hash if pointers else ""
                        )
                        st.session_state["active_doc_name"] = uploaded_file.name
                        st.session_state.pop("consensus_result", None)
                        st.success(
                            f"Successfully processed and indexed {len(pointers)} evidence pointers from '{uploaded_file.name}'."
                        )
                        st.rerun()
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)

    latest_pointers = st.session_state.get("latest_pointers", [])
    if latest_pointers:
        st.html(section_tag_html("Indexed Evidence Pointers"))
        for p in latest_pointers[:10]:
            with st.expander(f"Pointer {p.pointer_id} — {p.section_hierarchy}"):
                st.write(p.content)
                st.caption(
                    f"Document Hash: {p.doc_hash[:16]}... | Relevance Score: {p.relevance_score:.2f}"
                )

