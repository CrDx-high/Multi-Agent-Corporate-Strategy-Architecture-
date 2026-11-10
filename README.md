# Corporate Strategy Consensus Engine

A local, single-user strategic consensus engine powered by LangChain v0.3+ Expression Language (LCEL), Groq LLMs, SQLite WAL state checkpointing, and multi-agent DAG review.

---

## Architecture Overview

```
                      ┌────────────────────────────┐
                      │    Client .docx Document   │
                      └──────────────┬─────────────┘
                                     │
                        (python-docx: paragraphs & tables)
                                     │
                                     ▼
                      ┌────────────────────────────┐
                      │ Hierarchical Text Chunker  │
                      │ (Paragraph > Sentence >    │
                      │  800-char / 150-overlap)   │
                      └──────────────┬─────────────┘
                                     │
                                     ▼
                      ┌────────────────────────────┐
                      │    ChromaDB Vector Store   │
                      │  (Single-Writer Async Lock)│
                      └──────────────┬─────────────┘
                                     │
                             (retrieve_pointers)
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               Multi-Agent Consensus DAG                                │
│                                                                                        │
│                                  ┌───────────────────┐                                 │
│                                  │   Proposer Node   │                                 │
│                                  │ (Query + Pointers)│                                 │
│                                  └─────────┬─────────┘                                 │
│                                            │                                           │
│                             (ProposalBatch: 2-4 Hypotheses)                            │
│                                            │                                           │
│                     ┌──────────────────────┴──────────────────────┐                    │
│                     ▼                                             ▼                    │
│          ┌─────────────────────┐                       ┌─────────────────────┐         │
│          │   Compliance Node   │                       │   Bottleneck Node   │         │
│          │ (Proposals+Policies)│                       │ (Proposals+Capacity)│         │
│          └──────────┬──────────┘                       └──────────┬──────────┘         │
│                     │                                             │                    │
│            (ComplianceAuditBatch)                       (BottleneckAuditBatch)         │
│                     │                                             │                    │
│                     └──────────────────────┬──────────────────────┘                    │
│                                            │                                           │
│                                            ▼                                           │
│                                  ┌───────────────────┐                                 │
│                                  │   Arbiter Node    │                                 │
│                                  │ (Executive Synth) │                                 │
│                                  └─────────┬─────────┘                                 │
│                                            │                                           │
│                                            ▼                                           │
│                               ┌─────────────────────────┐                              │
│                               │    StrategyConsensus    │                              │
│                               │ (Verdicts & Prerequisites)                             │
│                               └─────────────────────────┘                              │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Engineering Principles

- **Single-User, Local Tool**: Built for local execution without multi-tenant overhead or PII masking layers. Document text is passed directly to the configured Groq LLM model in structured prompts.
- **Strict Context Isolation**:
  - `proposer_node` receives only executive `query` and retrieved `DocumentPointer` evidence.
  - `compliance_node` receives only the `ProposalBatch` and policy context.
  - `bottleneck_node` receives only the `ProposalBatch` and operational constraint context.
  - `arbiter_node` receives only the structured audit batches from Nodes 1, 2A, and 2B.
- **Input-Hashed Checkpointing**: Node checkpoints in `./checkpoints.db` are keyed by `job_id` and a 12-character SHA-256 hash of prompt inputs. Reusing a `job_id` with altered queries or context will not return stale data.
- **ID Consistency Enforcement**: Evaluated initiative IDs across compliance, bottleneck, and arbiter nodes are strictly validated against proposal IDs. Any mismatch triggers an automatic single-retry without cache, raising `ValueError` on failure.
- **Bounded Exponential Retries**: All LLM calls are wrapped in bounded retries (up to 3 attempts with exponential backoff) for transient errors or structured schema output validation faults.
- **Rate Limiting**: Request starts per minute are throttled via `AsyncTokenBucketLimiter` (`limiter_proposer: 10 RPM`, `limiter_compliance: 20 RPM`, `limiter_bottleneck: 20 RPM`, `limiter_arbiter: 5 RPM`).
- **Dead-Letter Queue (DLQ)**: Corrupted or unparseable `.docx` files are safely moved to `./dlq/` without crashing batch ingestion.
- **Zero Import Side-Effects**: All LLM clients, embedding engines, SQLite connections, and vector databases are initialized lazily inside functions. Backend modules import instantly offline with zero environment variables set and create no runtime artifacts.

---

## Project Structure

```
strategy_consensus/
├── app.py                      # Thin Streamlit entry point (~80 lines)
├── ui/
│   ├── __init__.py
│   ├── style.css               # Static pixel-art design system & layouts
│   ├── assets.py               # Base64 image and CSS file loaders
│   ├── components.py           # Pure HTML render functions (escaped)
│   ├── async_utils.py          # Streamlit async event-loop bridge
│   ├── tab_ingestion.py        # Tab 1: Document Ingestion & Vault
│   └── tab_consensus.py        # Tab 2: Strategy Consensus DAG & Checkpoints
├── schemas.py                  # Pydantic v2 data models (ConfigDict extra="forbid")
├── infra.py                    # Rate limiters and WAL-mode SQLite state checkpointer
├── ingestion.py                # Docx parsing, text chunking, and ChromaDB storage
├── orchestrator.py             # LCEL multi-agent consensus DAG and node executors
├── scripts/
│   └── generate_pixel_banner.py # Procedural pixel-art banner generator
├── tests/
│   ├── fixtures/
│   │   └── sample_strategic_plan.docx
│   ├── test_chunker.py         # Boundary, sizing, and overlap tests
│   ├── test_dag.py             # DAG orchestration, checkpointing, and ID checks
│   ├── test_imports.py         # Offline zero-side-effect import verification
│   └── test_ingestion.py       # Ingestion and DLQ failure routing tests
├── assets/                     # Pixel-art visual assets (buildings, clouds)
├── pyproject.toml              # Dependencies and pytest configuration
└── README.md
```

---

## Quickstart

### 1. Environment Setup

Manage all dependencies exclusively with `uv`:

```bash
uv sync
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_...
GROQ_MODEL=openai/gpt-oss-20b
```

Verified Groq models available in the interface:
- `openai/gpt-oss-20b` (Default)
- `openai/gpt-oss-120b`
- `qwen/qwen3.8-27b`

### 3. Launch the Application

```bash
uv run streamlit run app.py
```

---

## Module Reference & Exports

| Module | Core Exports | Responsibility |
|---|---|---|
| `schemas.py` | `VerdictEnum`, `DocumentPointer`, `StrategicHypothesis`, `ProposalBatch`, `ComplianceCheck`, `ComplianceAuditBatch`, `BottleneckReport`, `BottleneckAuditBatch`, `EvaluatedInitiative`, `StrategyConsensus` | Immutable Pydantic v2 data contracts with `model_config = ConfigDict(extra="forbid")`. |
| `infra.py` | `AsyncTokenBucketLimiter`, `StateCheckpointer`, `checkpointer`, `limiter_proposer`, `limiter_compliance`, `limiter_bottleneck`, `limiter_arbiter` | Asynchronous rate limiters and WAL-mode SQLite state checkpointer. |
| `ingestion.py` | `ingest_docx`, `retrieve_pointers`, `clear_pointers`, `TextChunker`, `DocxHierarchyParser`, `ChromaPointerStore`, `get_pipeline` | Object-oriented `.docx` hierarchy parsing, paragraph/sentence text chunking, and ChromaDB vector persistence. |
| `orchestrator.py` | `run_strategy_dag`, `proposer_node`, `compliance_node`, `bottleneck_node`, `arbiter_node`, `load_proposals`, `get_llm`, `get_structured_llm` | Multi-agent LCEL DAG execution with parallel compliance/bottleneck reviews, arbiter consensus, bounded retries, and ID consistency verification. |
| `app.py` | `main` | Clean, thin Streamlit entry point. |

---

## Verification & Testing

Execute the test suite and AGENTS.md verification gates:

```bash
# 1. Bytecode compilation across all source modules
uv run python -m py_compile schemas.py infra.py ingestion.py orchestrator.py app.py ui/*.py scripts/*.py

# 2. Offline import and interface safety check (creates 0 files, instantiates 0 clients)
uv run python -c "import schemas, infra, ingestion, orchestrator; print('ALL IMPORTS VERIFIED')"

# 3. Automated test suite (all 12 offline tests)
uv run pytest -q
```
