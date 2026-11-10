"""Corporate Strategy Consensus Engine — Streamlit Web Interface.

Visual Art Direction: Nocturnal Pixel-Art Aesthetic & High-End Executive Presentation.
Strict compliance with AGENTS.md architectural standards and zero-emoji mandates.
"""

from dotenv import load_dotenv
import streamlit as st

from ingestion import clear_pointers
from ui.assets import load_css, load_pixel_b64
from ui.components import banner_html
from ui.tab_consensus import render_consensus_tab
from ui.tab_ingestion import render_ingestion_tab

# Load environment configuration
load_dotenv()

# Streamlit Page Configuration
st.set_page_config(
    page_title="Corporate Strategy Consensus Engine",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)


def render_top_control_row() -> str:
    """Renders the model selector and session reset controls."""
    col_model, col_reset = st.columns([4.5, 1])
    with col_model:
        selected_model = st.selectbox(
            "Model",
            options=["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"],
            index=0,
            label_visibility="collapsed",
            help="Select the language model for multi-agent strategic consensus.",
        )
        st.session_state["selected_model"] = selected_model

    with col_reset:
        if st.button("Reset Session", use_container_width=True):
            st.session_state.clear()
            try:
                clear_pointers()
            except Exception as e:
                st.session_state["reset_error"] = str(e)
            st.rerun()

    return selected_model


def main() -> None:
    """Main application layout for Corporate Strategy Consensus Engine."""
    # Inject nocturnal pixel art stylesheet
    st.html(f"<style>{load_css()}</style>")

    # Top framed nocturnal banner
    buildings_b64 = load_pixel_b64("assets/buildings.png")
    if not buildings_b64:
        buildings_b64 = load_pixel_b64("assets/buildings.jpg")
    st.html(banner_html(buildings_b64))

    # Control bar
    selected_model = render_top_control_row()

    # Executive navigation tabs
    tab1, tab2 = st.tabs(
        ["1. Document Ingestion & Vault", "2. Strategy Consensus DAG"]
    )

    with tab1:
        render_ingestion_tab()

    with tab2:
        render_consensus_tab(selected_model=selected_model)


if __name__ == "__main__":
    main()
