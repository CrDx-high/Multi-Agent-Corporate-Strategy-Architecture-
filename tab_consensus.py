"""Tab 2: Multi-Agent Consensus Orchestration interface."""

import uuid
import streamlit as st

from infra import checkpointer
from ingestion import retrieve_pointers
from orchestrator import load_proposals, run_strategy_dag
from schemas import (
    BottleneckAuditBatch,
    ComplianceAuditBatch,
    ProposalBatch,
    StrategicHypothesis,
    StrategyConsensus,
)
from ui.async_utils import run_async
from ui.components import (
    initiative_card_html,
    pixel_hr_html,
    section_tag_html,
    summary_html,
    target_doc_badge_html,
)


def render_consensus_tab(selected_model: str | None = None) -> None:
    """Renders Phase 2: Multi-Agent Consensus Orchestrator."""
    st.html(section_tag_html("Multi-Agent Strategy Consensus DAG"))
    st.markdown(
        "Executes a structured consensus workflow across four specialized agent roles—**Proposer**, "
        "**Compliance Auditor**, **Bottleneck Auditor**, and **Arbiter**—with strict context isolation "
        "to evaluate strategic proposals against extracted evidence."
    )

    active_hash = st.session_state.get("latest_doc_hash")
    active_name = st.session_state.get("active_doc_name", "Corporate Strategic Plan")

    if active_hash:
        st.html(target_doc_badge_html(active_name, active_hash))
    else:
        st.warning("No active document currently loaded. Please upload a .docx file in Tab 1 before running the consensus DAG.")

    query = st.text_input(
        "Strategic Objective or Executive Inquiry",
        value="Evaluate strategic expansion into international retail markets and automated AI supply chain systems.",
        help="Enter the executive query to be analyzed against evidence pointers.",
    )

    col_ctx1, col_ctx2 = st.columns(2)
    with col_ctx1:
        policy_context = st.text_area(
            "Policy & Governance Directives (Compliance Node)",
            value="All initiatives must achieve compliance with GDPR data privacy directives, "
                  "meet corporate ESG zero-carbon 2030 targets, and maintain a minimum 15% EBITDA operating margin.",
            height=120,
        )
    with col_ctx2:
        bottleneck_context = st.text_area(
            "Operational & Technical Constraints (Bottleneck Node)",
            value="Cloud GPU compute quota is capped at 64 H100s. Supply chain hardware lead times are 12 weeks. "
                  "Maximum engineering allocation is 15 FTEs for international rollout.",
            height=120,
        )

    if st.button("Run Strategy Consensus DAG", type="primary"):
        if not active_hash:
            st.error("Cannot run consensus without an active document. Please upload and embed a .docx file in Tab 1 first.")
            return

        job_id = f"job_{uuid.uuid4().hex[:8]}"
        st.session_state["current_job_id"] = job_id

        with st.status("Executing Multi-Agent Strategy Consensus DAG...", expanded=True) as status:
            st.write(f"Step 1: Retrieving relevant evidence pointers for '{active_name}'...")
            pointers = retrieve_pointers(query, top_k=12, doc_hash=active_hash)

            if not pointers:
                status.update(label="No Evidence Pointers Found", state="error", expanded=True)
                st.error("No evidence pointers could be retrieved for the active document. Please verify the document content.")
                return

            st.write(f"Located {len(pointers)} relevant evidence pointers from active document.")
            st.write("Step 2: Orchestrating consensus across Proposer, Compliance, Bottleneck, and Arbiter...")
            consensus: StrategyConsensus = run_async(
                run_strategy_dag(
                    job_id=job_id,
                    query=query,
                    pointers=pointers,
                    policy_context=policy_context,
                    bottleneck_context=bottleneck_context,
                    model=selected_model,
                )
            )
            st.session_state["consensus_result"] = consensus
            status.update(label="Strategic Consensus Achieved", state="complete", expanded=False)

    consensus_result: StrategyConsensus | None = st.session_state.get("consensus_result")
    current_job_id = st.session_state.get("current_job_id", "")

    if consensus_result:
        st.html(pixel_hr_html())
        st.html(section_tag_html("Executive Consensus Summary"))
        st.html(summary_html(consensus_result.executive_summary))

        # Match evaluated initiatives with proposals checkpoint
        proposals_batch: ProposalBatch | None = (
            load_proposals(current_job_id)
            if current_job_id
            else None
        )
        proposals_map = (
            {h.initiative_id: h for h in proposals_batch.hypotheses}
            if proposals_batch
            else {}
        )

        st.html(section_tag_html("Evaluated Strategic Initiatives"))
        for init in consensus_result.evaluated_initiatives:
            matched_hypothesis: StrategicHypothesis | None = proposals_map.get(init.initiative_id)
            st.html(initiative_card_html(init, matched_hypothesis))

        # Checkpoint Inspector for this Job
        if current_job_id:
            st.html(pixel_hr_html())
            st.html(section_tag_html(f"System Checkpoint Inspection (Job ID: {current_job_id})"))
            c1, c2, c3 = st.columns(3)
            with c1:
                proposals = checkpointer.load_checkpoint(
                    current_job_id, "proposer", ProposalBatch
                )
                if proposals:
                    with st.expander("Proposals Checkpoint (Node 1)"):
                        st.json(proposals.model_dump())
            with c2:
                compliance = checkpointer.load_checkpoint(
                    current_job_id, "compliance", ComplianceAuditBatch
                )
                if compliance:
                    with st.expander("Compliance Audit Checkpoint (Node 2A)"):
                        st.json(compliance.model_dump())
            with c3:
                bottlenecks = checkpointer.load_checkpoint(
                    current_job_id, "bottleneck", BottleneckAuditBatch
                )
                if bottlenecks:
                    with st.expander("Operational Bottlenecks Checkpoint (Node 2B)"):
                        st.json(bottlenecks.model_dump())
