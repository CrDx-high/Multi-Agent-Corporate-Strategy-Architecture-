"""Orchestration DAG module: LCEL-based multi-agent strategic consensus with state checkpointing, rate limiting, and strict context isolation."""

import asyncio
import hashlib
import logging
import os
from typing import Any
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel

from infra import (
    checkpointer,
    limiter_arbiter,
    limiter_bottleneck,
    limiter_compliance,
    limiter_proposer,
)
from schemas import (
    BottleneckAuditBatch,
    ComplianceAuditBatch,
    DocumentPointer,
    ProposalBatch,
    StrategyConsensus,
)

logger = logging.getLogger(__name__)


def get_llm(temperature: float = 0.1, model: str | None = None) -> Any:
    """Lazily initializes and returns the Groq LLM client.

    Ensures offline module imports succeed without API keys.
    """
    load_dotenv()
    from langchain_groq import ChatGroq

    selected_model = model or os.environ.get(
        "GROQ_MODEL", "openai/gpt-oss-20b"
    )
    return ChatGroq(model=selected_model, temperature=temperature)


def get_structured_llm(
    schema: Any, temperature: float = 0.1, model: str | None = None
) -> Any:
    """Returns a Groq LLM bound with structured Pydantic schema validation using json_schema mode."""
    llm = get_llm(temperature=temperature, model=model)
    try:
        return llm.with_structured_output(schema, method="json_schema")
    except Exception:
        return llm.with_structured_output(schema)


def _format_evidence_pointers(pointers: list[DocumentPointer]) -> str:
    """Formats DocumentPointer objects into strict evidence strings."""
    if not pointers:
        return "No specific evidence pointers provided."
    lines = []
    for p in pointers:
        lines.append(
            f"Pointer ID: {p.pointer_id}\n"
            f"Section Hierarchy: {p.section_hierarchy}\n"
            f"Assertion: {p.content}"
        )
    return "\n---\n".join(lines)


def _format_proposals(batch: ProposalBatch) -> str:
    """Formats ProposalBatch hypotheses for downstream audit nodes."""
    lines = []
    for h in batch.hypotheses:
        pointers_str = (
            ", ".join(h.evidence_pointers) if h.evidence_pointers else "None"
        )
        lines.append(
            f"Initiative ID: {h.initiative_id}\n"
            f"Title: {h.title}\n"
            f"Strategic Goal Alignment: {h.strategic_goal_alignment}\n"
            f"Execution Blueprint: {h.execution_blueprint}\n"
            f"Selection Rationale: {h.selection_rationale}\n"
            f"Quantified Upside: {h.quantified_upside}\n"
            f"Supporting Evidence Pointers: {pointers_str}"
        )
    return "\n\n---\n\n".join(lines)


def _format_compliance_audits(batch: ComplianceAuditBatch) -> str:
    """Formats ComplianceAuditBatch checks for the Arbiter."""
    lines = []
    for a in batch.audits:
        lines.append(
            f"Initiative ID: {a.initiative_id}\n"
            f"Compliant: {a.is_compliant}\n"
            f"Policy Reference: {a.policy_reference or 'None'}\n"
            f"Remediation Note: {a.remediation_note or 'None'}"
        )
    return "\n\n".join(lines)


def _format_bottleneck_reports(batch: BottleneckAuditBatch) -> str:
    """Formats BottleneckAuditBatch reports for the Arbiter."""
    lines = []
    for r in batch.reports:
        lines.append(
            f"Initiative ID: {r.initiative_id}\n"
            f"Blocked: {r.is_blocked}\n"
            f"Friction Index: {r.friction_index:.2f}\n"
            f"Constraint Violated: {r.constraint_violated or 'None'}"
        )
    return "\n\n".join(lines)


# --- Node 1: Proposer Node ---

PROPOSER_SYSTEM_PROMPT = """You are a Principal Corporate Strategy Partner. Your mandate is to author exhaustive, deeply analytical strategic blueprints grounded strictly in the provided document pointers.

CRITICAL RULES:
- NEVER output generic, one-sentence summaries or vague platitudes (e.g., 'leverage synergies' or 'optimize workflows').
- Every field must be thorough and concrete:
  1. `strategic_goal_alignment`: Write a comprehensive analysis of the core business objective and the exact mechanism by which this initiative solves it.
  2. `execution_blueprint`: Detail a concrete, multi-phase implementation roadmap (who does what, technical/operational steps, and milestones).
  3. `selection_rationale`: Defend WHY you chose this specific strategic path over alternatives. Explicitly cite facts, metrics, and constraints from the provided Pointer IDs to justify your choice.
  4. `quantified_upside`: Break down the expected ROI, revenue impact, or cost reduction with specific numbers/ranges from the text."""


async def _with_retry(
    coro_factory: Any,
    node_name: str,
    max_attempts: int = 3,
    base_backoff: float = 0.5,
) -> Any:
    """Executes an async callable with bounded retries and exponential backoff."""
    for attempt in range(1, max_attempts + 1):
        try:
            return await coro_factory()
        except Exception as exc:
            if attempt >= max_attempts:
                logger.error(
                    "Node '%s' failed all %d retry attempts: %s",
                    node_name,
                    max_attempts,
                    exc,
                    exc_info=True,
                )
                raise
            backoff = base_backoff * (2 ** (attempt - 1))
            logger.warning(
                "Node '%s' attempt %d/%d failed with error (%s). Retrying in %.2fs...",
                node_name,
                attempt,
                max_attempts,
                exc,
                backoff,
            )
            await asyncio.sleep(backoff)


def _validate_initiative_ids(
    expected_ids: set[str], actual_ids: set[str], node_name: str
) -> None:
    """Validates that a node's output covers exactly the set of expected proposal initiative IDs."""
    if expected_ids != actual_ids:
        missing = expected_ids - actual_ids
        extra = actual_ids - expected_ids
        raise ValueError(
            f"{node_name} initiative ID mismatch after retry. "
            f"Expected {sorted(expected_ids)}, got {sorted(actual_ids)}. "
            f"Missing: {sorted(missing)}, Extra: {sorted(extra)}"
        )


async def proposer_node(
    job_id: str,
    query: str,
    pointers: list[DocumentPointer],
    model: str | None = None,
    ignore_cache: bool = False,
) -> ProposalBatch:
    """Node 1: Generates 2-4 strategic proposals from query and evidence pointers.

    Enforces strict context isolation (receives only query and pointers).
    """
    evidence_formatted = _format_evidence_pointers(pointers)
    inputs_str = f"query:{query}\nevidence:{evidence_formatted}"
    input_hash = hashlib.sha256(inputs_str.encode("utf-8")).hexdigest()[:12]
    checkpoint_key = f"proposer:{input_hash}"

    if not ignore_cache:
        cached = checkpointer.load_checkpoint(job_id, checkpoint_key, ProposalBatch)
        if cached is not None:
            logger.info(
                "Loaded cached checkpoint for job_id=%s, node=%s", job_id, checkpoint_key
            )
            return cached

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PROPOSER_SYSTEM_PROMPT),
            (
                "human",
                "Executive Query: {query}\n\n"
                "Document Evidence Pointers:\n{evidence_block}\n\n"
                "Formulate 2 to 4 exhaustive, deeply analytical strategic hypotheses.",
            ),
        ]
    )

    structured_llm = get_structured_llm(ProposalBatch, temperature=0.4, model=model)
    chain = prompt | structured_llm

    async def _invoke():
        async with limiter_proposer:
            return await chain.ainvoke(
                {"query": query, "evidence_block": evidence_formatted}
            )

    result: ProposalBatch = await _with_retry(_invoke, node_name="proposer")

    checkpointer.save_checkpoint(job_id, checkpoint_key, result)
    logger.info(
        "Completed and checkpointed proposer node for job_id=%s with %d hypotheses",
        job_id,
        len(result.hypotheses),
    )
    return result


# --- Node 2A: Compliance Node ---

COMPLIANCE_SYSTEM_PROMPT = """You are a Chief Compliance and Regulatory Officer.
Your objective is to independently audit proposed strategic initiatives against corporate policies, legal regulations, and governance constraints.

Requirements:
1. For every initiative in the proposal batch, produce an individual ComplianceCheck:
   - initiative_id: Matching the initiative under review.
   - is_compliant: Boolean flag (True if aligned with policy, False if non-compliant).
   - policy_reference: Specific policy section, regulation, or governance standard referenced (or None if fully compliant).
   - remediation_note: Actionable remediation needed to achieve compliance (or None if fully compliant).
2. Evaluate each initiative objectively against the provided policy context."""


async def compliance_node(
    job_id: str,
    proposals: ProposalBatch,
    policy_context: str,
    model: str | None = None,
    ignore_cache: bool = False,
) -> ComplianceAuditBatch:
    """Node 2A: Audits proposed initiatives against policy and regulatory standards.

    Enforces strict context isolation (receives only ProposalBatch and policy_context).
    """
    proposals_formatted = _format_proposals(proposals)
    effective_policy = policy_context or "Standard corporate regulatory governance."
    inputs_str = f"policy:{effective_policy}\nproposals:{proposals_formatted}"
    input_hash = hashlib.sha256(inputs_str.encode("utf-8")).hexdigest()[:12]
    checkpoint_key = f"compliance:{input_hash}"

    if not ignore_cache:
        cached = checkpointer.load_checkpoint(
            job_id, checkpoint_key, ComplianceAuditBatch
        )
        if cached is not None:
            logger.info(
                "Loaded cached checkpoint for job_id=%s, node=%s", job_id, checkpoint_key
            )
            return cached

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", COMPLIANCE_SYSTEM_PROMPT),
            (
                "human",
                "Policy & Governance Context:\n{policy_context}\n\n"
                "Proposed Initiatives to Audit:\n{proposals_block}\n\n"
                "Audit each proposed initiative.",
            ),
        ]
    )

    structured_llm = get_structured_llm(
        ComplianceAuditBatch, temperature=0.1, model=model
    )
    chain = prompt | structured_llm

    async def _invoke():
        async with limiter_compliance:
            return await chain.ainvoke(
                {
                    "policy_context": effective_policy,
                    "proposals_block": proposals_formatted,
                }
            )

    result: ComplianceAuditBatch = await _with_retry(_invoke, node_name="compliance")

    checkpointer.save_checkpoint(job_id, checkpoint_key, result)
    logger.info(
        "Completed and checkpointed compliance node for job_id=%s with %d audits",
        job_id,
        len(result.audits),
    )
    return result


# --- Node 2B: Bottleneck Node ---

BOTTLENECK_SYSTEM_PROMPT = """You are a Chief Operating & Technical Officer.
Your objective is to stress-test proposed strategic initiatives against operational capacity, technical architecture, and supply-chain friction.

Requirements:
1. For every initiative in the proposal batch, produce an individual BottleneckReport:
   - initiative_id: Matching the initiative under review.
   - is_blocked: Boolean flag indicating if critical resource/infrastructure bottlenecks block execution.
   - friction_index: Float from 0.0 (smooth execution, minimal friction) to 1.0 (extreme blockage/fatal friction).
   - constraint_violated: The specific operational, compute, staffing, or vendor constraint violated (or None if smooth).
2. Base evaluations strictly on the provided operational and technical constraints."""


async def bottleneck_node(
    job_id: str,
    proposals: ProposalBatch,
    bottleneck_context: str,
    model: str | None = None,
    ignore_cache: bool = False,
) -> BottleneckAuditBatch:
    """Node 2B: Evaluates operational and technical friction for proposed initiatives.

    Enforces strict context isolation (receives only ProposalBatch and bottleneck_context).
    """
    proposals_formatted = _format_proposals(proposals)
    effective_bottlenecks = (
        bottleneck_context or "Standard operational engineering capacity."
    )
    inputs_str = f"bottleneck:{effective_bottlenecks}\nproposals:{proposals_formatted}"
    input_hash = hashlib.sha256(inputs_str.encode("utf-8")).hexdigest()[:12]
    checkpoint_key = f"bottleneck:{input_hash}"

    if not ignore_cache:
        cached = checkpointer.load_checkpoint(
            job_id, checkpoint_key, BottleneckAuditBatch
        )
        if cached is not None:
            logger.info(
                "Loaded cached checkpoint for job_id=%s, node=%s", job_id, checkpoint_key
            )
            return cached

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", BOTTLENECK_SYSTEM_PROMPT),
            (
                "human",
                "Operational & Technical Constraints:\n{bottleneck_context}\n\n"
                "Proposed Initiatives to Stress-Test:\n{proposals_block}\n\n"
                "Stress-test each proposed initiative.",
            ),
        ]
    )

    structured_llm = get_structured_llm(
        BottleneckAuditBatch, temperature=0.1, model=model
    )
    chain = prompt | structured_llm

    async def _invoke():
        async with limiter_bottleneck:
            return await chain.ainvoke(
                {
                    "bottleneck_context": effective_bottlenecks,
                    "proposals_block": proposals_formatted,
                }
            )

    result: BottleneckAuditBatch = await _with_retry(_invoke, node_name="bottleneck")

    checkpointer.save_checkpoint(job_id, checkpoint_key, result)
    logger.info(
        "Completed and checkpointed bottleneck node for job_id=%s with %d reports",
        job_id,
        len(result.reports),
    )
    return result


# --- Node 3: Arbiter Node ---

ARBITER_SYSTEM_PROMPT = """You are the Lead Executive Strategy Arbiter.
Your objective is to reconcile the proposed strategic hypotheses with the independent compliance audits and operational bottleneck reports to synthesize a final executive consensus.

Requirements:
1. Provide a comprehensive executive_summary synthesizing the strategic landscape, risk profile, and resource allocation.
2. For every proposed initiative, deliver an EvaluatedInitiative:
   - initiative_id: Matching the initiative.
   - verdict: VerdictEnum value ('APPROVED', 'REJECTED', or 'CONDITIONALLY_APPROVED').
   - confidence_score: Float between 0.0 and 1.0 reflecting confidence in the verdict.
   - trade_off_resolution: Clear synthesis articulating how upside was balanced against compliance and operational risks.
   - mandatory_prerequisites: List of concrete conditions, remediation gates, or KPIs that must be satisfied.
3. Be rigorous: initiatives with high friction index (>0.7) or compliance violations should generally be CONDITIONALLY_APPROVED with strict prerequisites, or REJECTED."""


async def arbiter_node(
    job_id: str,
    proposals: ProposalBatch,
    compliance: ComplianceAuditBatch,
    bottlenecks: BottleneckAuditBatch,
    model: str | None = None,
    ignore_cache: bool = False,
) -> StrategyConsensus:
    """Node 3: Synthesizes proposals, compliance audits, and bottleneck reports into executive consensus.

    Enforces strict context isolation (receives only structured outputs of Nodes 1, 2A, 2B).
    """
    proposals_formatted = _format_proposals(proposals)
    compliance_formatted = _format_compliance_audits(compliance)
    bottlenecks_formatted = _format_bottleneck_reports(bottlenecks)

    inputs_str = (
        f"proposals:{proposals_formatted}\n"
        f"compliance:{compliance_formatted}\n"
        f"bottlenecks:{bottlenecks_formatted}"
    )
    input_hash = hashlib.sha256(inputs_str.encode("utf-8")).hexdigest()[:12]
    checkpoint_key = f"arbiter:{input_hash}"

    if not ignore_cache:
        cached = checkpointer.load_checkpoint(job_id, checkpoint_key, StrategyConsensus)
        if cached is not None:
            logger.info(
                "Loaded cached checkpoint for job_id=%s, node=%s", job_id, checkpoint_key
            )
            return cached

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ARBITER_SYSTEM_PROMPT),
            (
                "human",
                "Proposed Strategic Initiatives:\n{proposals_block}\n\n"
                "Compliance Audits:\n{compliance_block}\n\n"
                "Operational Bottleneck Reports:\n{bottlenecks_block}\n\n"
                "Synthesize the final executive strategy consensus.",
            ),
        ]
    )

    structured_llm = get_structured_llm(StrategyConsensus, temperature=0.1, model=model)
    chain = prompt | structured_llm

    async def _invoke():
        async with limiter_arbiter:
            return await chain.ainvoke(
                {
                    "proposals_block": proposals_formatted,
                    "compliance_block": compliance_formatted,
                    "bottlenecks_block": bottlenecks_formatted,
                }
            )

    result: StrategyConsensus = await _with_retry(_invoke, node_name="arbiter")

    checkpointer.save_checkpoint(job_id, checkpoint_key, result)
    logger.info(
        "Completed and checkpointed arbiter node for job_id=%s with %d evaluated initiatives",
        job_id,
        len(result.evaluated_initiatives),
    )
    return result


# --- Orchestrator DAG ---


async def run_strategy_dag(
    job_id: str,
    query: str,
    pointers: list[DocumentPointer],
    policy_context: str,
    bottleneck_context: str,
    model: str | None = None,
) -> StrategyConsensus:
    """Executes the strategic consensus DAG across Proposer, Compliance, Bottleneck, and Arbiter nodes with state checkpointing, rate limiting, and strict context isolation."""
    logger.info("Executing Strategy Consensus DAG for job_id=%s", job_id)

    # Node 1: Proposer (Context isolation: query and pointers only)
    proposals = await proposer_node(job_id, query, pointers, model=model)
    expected_ids = {h.initiative_id for h in proposals.hypotheses}

    # Nodes 2A & 2B: Parallel Audits (Context isolation: proposals and respective context only)
    compliance_task = compliance_node(job_id, proposals, policy_context, model=model)
    bottleneck_task = bottleneck_node(job_id, proposals, bottleneck_context, model=model)
    compliance_batch, bottleneck_batch = await asyncio.gather(
        compliance_task, bottleneck_task
    )

    # Validate Node 2A Compliance IDs; rerun once with ignore_cache=True on mismatch
    comp_ids = {a.initiative_id for a in compliance_batch.audits}
    if comp_ids != expected_ids:
        logger.warning(
            "Compliance ID mismatch for job_id=%s (expected %s, got %s). Retrying compliance node without cache...",
            job_id,
            expected_ids,
            comp_ids,
        )
        compliance_batch = await compliance_node(
            job_id, proposals, policy_context, model=model, ignore_cache=True
        )
        comp_ids = {a.initiative_id for a in compliance_batch.audits}
        _validate_initiative_ids(expected_ids, comp_ids, "Compliance node")

    # Validate Node 2B Bottleneck IDs; rerun once with ignore_cache=True on mismatch
    bot_ids = {r.initiative_id for r in bottleneck_batch.reports}
    if bot_ids != expected_ids:
        logger.warning(
            "Bottleneck ID mismatch for job_id=%s (expected %s, got %s). Retrying bottleneck node without cache...",
            job_id,
            expected_ids,
            bot_ids,
        )
        bottleneck_batch = await bottleneck_node(
            job_id, proposals, bottleneck_context, model=model, ignore_cache=True
        )
        bot_ids = {r.initiative_id for r in bottleneck_batch.reports}
        _validate_initiative_ids(expected_ids, bot_ids, "Bottleneck node")

    # Node 3: Arbiter (Context isolation: structured outputs of 1, 2A, 2B only)
    consensus = await arbiter_node(
        job_id, proposals, compliance_batch, bottleneck_batch, model=model
    )

    # Validate Node 3 Arbiter IDs; rerun once with ignore_cache=True on mismatch
    arb_ids = {i.initiative_id for i in consensus.evaluated_initiatives}
    if arb_ids != expected_ids:
        logger.warning(
            "Arbiter ID mismatch for job_id=%s (expected %s, got %s). Retrying arbiter node without cache...",
            job_id,
            expected_ids,
            arb_ids,
        )
        consensus = await arbiter_node(
            job_id,
            proposals,
            compliance_batch,
            bottleneck_batch,
            model=model,
            ignore_cache=True,
        )
        arb_ids = {i.initiative_id for i in consensus.evaluated_initiatives}
        _validate_initiative_ids(expected_ids, arb_ids, "Arbiter node")

    logger.info(
        "Consensus achieved for job_id=%s with %d evaluated initiatives",
        job_id,
        len(consensus.evaluated_initiatives),
    )
    return consensus


def load_proposals(job_id: str) -> ProposalBatch | None:
    """Loads the most recent ProposalBatch checkpoint for a given job_id."""
    return checkpointer.load_checkpoint(job_id, "proposer", ProposalBatch)


__all__ = [
    "run_strategy_dag",
    "proposer_node",
    "compliance_node",
    "bottleneck_node",
    "arbiter_node",
    "load_proposals",
    "get_llm",
    "get_structured_llm",
]
