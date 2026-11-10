"""Ingestion module: Object-oriented .docx parsing, table extraction, DLQ error routing,
character chunking, and ChromaDB vector persistence using lightweight ONNX embeddings.
"""

import asyncio
import functools
import hashlib
import logging
import os
import re
import shutil
import time
from typing import Any

# Disable telemetry to avoid network round-trips and timeouts in sandboxed environments
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
import docx

from schemas import DocumentPointer

logger = logging.getLogger(__name__)

# Module-level writer locks for ChromaDB persistence operations per event loop
_chroma_writer_locks: dict[int, asyncio.Lock] = {}


def get_chroma_writer_lock() -> asyncio.Lock:
    """Lazily creates and returns the writer lock for the active running event loop."""
    try:
        loop = asyncio.get_running_loop()
        loop_id = id(loop)
    except RuntimeError:
        loop_id = 0
    if loop_id not in _chroma_writer_locks:
        _chroma_writer_locks[loop_id] = asyncio.Lock()
    return _chroma_writer_locks[loop_id]


class DocxHierarchyParser:
    """Parses Word .docx documents, extracting paragraphs and table matrices while preserving section hierarchies."""

    def __init__(self, dlq_dir: str = "./dlq"):
        self.dlq_dir = os.path.abspath(dlq_dir)

    @staticmethod
    def parse_heading_level(style_name: str) -> int | None:
        """Determines the numerical heading level from paragraph style name, or None if body text."""
        if not style_name:
            return None
        cleaned = style_name.strip()
        match = re.match(r"^(?:Heading|Header)\s*(\d+)$", cleaned, re.IGNORECASE)
        if match:
            return int(match.group(1))
        if cleaned.lower() == "title":
            return 1
        if cleaned.lower() == "subtitle":
            return 2
        return None

    def parse(self, file_path: str) -> tuple[str, list[tuple[str, str]]]:
        """Parses the document, extracting headings/body/tables, and returns (doc_hash, raw_sections).

        Moves corrupted files to dlq upon error.
        """
        if not os.path.exists(file_path):
            logger.error("Specified file path does not exist: %s", file_path)
            return "", []

        try:
            with open(file_path, "rb") as f:
                file_bytes = f.read()
            doc_hash = hashlib.sha256(file_bytes).hexdigest()

            doc = docx.Document(file_path)
            heading_stack: list[tuple[int, str]] = []
            raw_sections: list[tuple[str, str]] = []

            # 1. Parse paragraphs & headings
            for p in doc.paragraphs:
                text = p.text.strip()
                if not text:
                    continue

                style_name = p.style.name if p.style else ""
                level = self.parse_heading_level(style_name)

                if level is not None:
                    while heading_stack and heading_stack[-1][0] >= level:
                        heading_stack.pop()
                    heading_stack.append((level, text))
                else:
                    current_hierarchy = (
                        " > ".join(h[1] for h in heading_stack)
                        if heading_stack
                        else "Document"
                    )
                    raw_sections.append((current_hierarchy, text))

            # 2. Parse Word tables (budgets, KPI matrices)
            for table in doc.tables:
                current_hierarchy = (
                    " > ".join(h[1] for h in heading_stack)
                    if heading_stack
                    else "Document > Table"
                )
                for row in table.rows:
                    row_cells = [
                        cell.text.strip() for cell in row.cells if cell.text.strip()
                    ]
                    if row_cells:
                        row_text = " | ".join(dict.fromkeys(row_cells))
                        raw_sections.append((current_hierarchy, row_text))

            return doc_hash, raw_sections

        except Exception as exc:
            logger.error(
                "Failed parsing .docx file '%s'. Moving to DLQ: %s",
                file_path,
                exc,
                exc_info=True,
            )
            dest_filename = os.path.basename(file_path)
            dest_path = os.path.join(self.dlq_dir, dest_filename)
            if os.path.exists(dest_path):
                name, ext = os.path.splitext(dest_filename)
                dest_path = os.path.join(
                    self.dlq_dir, f"{name}_{int(time.time())}{ext}"
                )
            try:
                os.makedirs(self.dlq_dir, exist_ok=True)
                shutil.move(file_path, dest_path)
                logger.info("Moved corrupted file '%s' to '%s'", file_path, dest_path)
            except Exception as move_err:
                logger.error("Failed to move corrupted file to DLQ: %s", move_err)
            return "", []


class TextChunker:
    """Splits structured text into character-bounded windows with configurable overlap.

    Splits on paragraph breaks first, then sentence boundaries, and only falls back
    to a hard character cut for a single over-long sentence.
    """

    def __init__(self, chunk_size: int = 800, overlap: int = 150):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be strictly positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be non-negative and strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.stride = max(1, chunk_size - overlap)

    def chunk(self, text: str) -> list[str]:
        """Returns chunks of at most chunk_size characters with overlap characters."""
        if not text or not text.strip():
            return []
        cleaned = text.strip()
        if len(cleaned) <= self.chunk_size:
            return [cleaned]

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]
        splits: list[str] = []
        for p in paragraphs:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", p) if s.strip()]
            if not sentences:
                sentences = [p]
            for s in sentences:
                if len(s) <= self.chunk_size:
                    splits.append(s)
                else:
                    for start in range(0, len(s), self.stride):
                        end = min(start + self.chunk_size, len(s))
                        part = s[start:end].strip()
                        if part:
                            splits.append(part)
                        if end >= len(s):
                            break

        chunks: list[str] = []
        current_docs: list[str] = []
        current_len = 0

        for s in splits:
            sep = 1 if current_docs else 0
            if current_len + sep + len(s) > self.chunk_size:
                if current_docs:
                    chunks.append(" ".join(current_docs))
                    while current_docs and (
                        current_len > self.overlap
                        or (current_len + (1 if current_docs else 0) + len(s) > self.chunk_size)
                    ):
                        popped = current_docs.pop(0)
                        sep_loss = 1 if current_docs else 0
                        current_len = max(0, current_len - (len(popped) + sep_loss))
            sep = 1 if current_docs else 0
            current_docs.append(s)
            current_len += sep + len(s)

        if current_docs:
            chunks.append(" ".join(current_docs))

        return chunks


class ChromaPointerStore:
    """Manages single-writer ChromaDB vector storage and semantic retrieval using lightweight ONNX DefaultEmbeddingFunction."""

    def __init__(
        self,
        chroma_path: str = "./chroma_db",
        collection_name: str = "document_pointers",
    ):
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self._client: Any = None
        self._collection: Any = None
        self._embedding_function: Any = None

    def get_embedding_function(self):
        """Lazily instantiates the built-in lightweight ONNX DefaultEmbeddingFunction."""
        if self._embedding_function is None:
            self._embedding_function = DefaultEmbeddingFunction()
        return self._embedding_function

    def get_client(self):
        """Lazily opens and retains the persistent ChromaDB client."""
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=self.chroma_path,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
        return self._client

    def get_collection(self):
        """Initializes or opens the persistent ChromaDB collection."""
        if self._collection is None:
            client = self.get_client()
            ef = self.get_embedding_function()
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=ef,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    async def upsert_pointers(self, pointers: list[DocumentPointer]) -> None:
        """Asynchronously persists DocumentPointer objects into ChromaDB under single-writer lock."""
        if not pointers:
            return

        async with get_chroma_writer_lock():
            collection = self.get_collection()
            doc_texts = [p.content for p in pointers]

            collection.upsert(
                ids=[p.pointer_id for p in pointers],
                documents=doc_texts,
                metadatas=[
                    {
                        "doc_hash": p.doc_hash,
                        "section_hierarchy": p.section_hierarchy,
                        "relevance_score": float(p.relevance_score),
                    }
                    for p in pointers
                ],
            )

    def clear(self, doc_hash: str | None = None) -> None:
        """Deletes pointers matching doc_hash, or resets the collection if doc_hash is None."""
        try:
            collection = self.get_collection()
            if doc_hash:
                collection.delete(where={"doc_hash": doc_hash})
                logger.info("Cleared ChromaDB pointers for doc_hash=%s", doc_hash)
            else:
                client = self.get_client()
                client.delete_collection(name=self.collection_name)
                self._collection = None
                logger.info("Reset entire ChromaDB collection '%s'", self.collection_name)
        except Exception as e:
            logger.warning("ChromaDB clear operation encountered error: %s", e)

    def retrieve(
        self, query: str, top_k: int = 5, doc_hash: str | None = None
    ) -> list[DocumentPointer]:
        """Performs vector search in ChromaDB and reconstructs DocumentPointer instances.

        Optionally scopes retrieval to a specific document hash to prevent cross-document contamination.
        """
        if top_k <= 0 or not query or not query.strip():
            return []

        collection = self.get_collection()
        where_filter = {"doc_hash": doc_hash} if doc_hash else None

        try:
            results = collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_filter,
            )
        except Exception as exc:
            logger.debug(
                "Initial Chroma query failed with top_k=%d (%s). Checking matching document count...",
                top_k,
                exc,
            )
            try:
                matching = collection.get(where=where_filter, include=[])
                matching_count = len(matching.get("ids", [])) if matching else 0
            except Exception:
                matching_count = 0

            if matching_count == 0:
                logger.info("No matching pointers found for query/filter")
                return []

            results = collection.query(
                query_texts=[query],
                n_results=min(top_k, matching_count),
                where=where_filter,
            )

        pointers: list[DocumentPointer] = []
        if results and results.get("ids") and results["ids"][0]:
            ids = results["ids"][0]
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            distances = (
                results.get("distances", [[]])[0]
                if results.get("distances")
                else [0.0] * len(ids)
            )

            for p_id, text, meta, dist in zip(ids, docs, metas, distances):
                score = max(0.0, min(1.0, 1.0 - float(dist)))
                pointer = DocumentPointer(
                    pointer_id=p_id,
                    doc_hash=meta.get("doc_hash", ""),
                    section_hierarchy=meta.get("section_hierarchy", "Document"),
                    content=text,
                    relevance_score=round(score, 4),
                )
                pointers.append(pointer)

        return pointers


class DocxIngestionPipeline:
    """Coordinating pipeline orchestrating document parser, character chunker, and vector store."""

    def __init__(
        self,
        chroma_path: str = "./chroma_db",
        dlq_dir: str = "./dlq",
    ):
        self.parser = DocxHierarchyParser(dlq_dir=dlq_dir)
        self.chunker = TextChunker(chunk_size=800, overlap=150)
        self.store = ChromaPointerStore(chroma_path=chroma_path)

    async def ingest(self, file_path: str) -> list[DocumentPointer]:
        """Executes end-to-end ingestion pipeline on a .docx document."""
        doc_hash, raw_sections = self.parser.parse(file_path)
        if not doc_hash or not raw_sections:
            return []

        # Group consecutive paragraphs under identical section hierarchy
        grouped_sections: list[tuple[str, list[str]]] = []
        for sec_path, para in raw_sections:
            if grouped_sections and grouped_sections[-1][0] == sec_path:
                grouped_sections[-1][1].append(para)
            else:
                grouped_sections.append((sec_path, [para]))

        pointers: list[DocumentPointer] = []
        pointer_counter = 0

        for sec_path, paras in grouped_sections:
            full_text = "\n\n".join(paras)
            chunks = self.chunker.chunk(full_text)
            for chunk in chunks:
                pointer_counter += 1
                pointer_id = f"{doc_hash[:12]}_{pointer_counter:04d}"
                # Ingest-time pointers carry a neutral default score (1.0); query retrieval calculates and assigns actual cosine similarity.
                pointer = DocumentPointer(
                    pointer_id=pointer_id,
                    doc_hash=doc_hash,
                    section_hierarchy=sec_path,
                    content=chunk,
                    relevance_score=1.0,
                )
                pointers.append(pointer)

        if pointers:
            await self.store.upsert_pointers(pointers)

        return pointers

    def retrieve(
        self, query: str, top_k: int = 5, doc_hash: str | None = None
    ) -> list[DocumentPointer]:
        """Queries the vector store for top_k relevant DocumentPointer objects."""
        return self.store.retrieve(query, top_k=top_k, doc_hash=doc_hash)

    def clear(self, doc_hash: str | None = None) -> None:
        """Clears pointers matching doc_hash or resets the vector store."""
        self.store.clear(doc_hash=doc_hash)


@functools.lru_cache(maxsize=4)
def get_pipeline(
    chroma_path: str = "./chroma_db", dlq_dir: str = "./dlq"
) -> DocxIngestionPipeline:
    """Cached factory returning DocxIngestionPipeline instance on demand."""
    return DocxIngestionPipeline(chroma_path=chroma_path, dlq_dir=dlq_dir)


async def ingest_docx(
    file_path: str,
    chroma_path: str = "./chroma_db",
    dlq_dir: str = "./dlq",
) -> list[DocumentPointer]:
    """Public export: Parses .docx, chunks text, and stores DocumentPointers."""
    pipeline = get_pipeline(chroma_path=chroma_path, dlq_dir=dlq_dir)
    return await pipeline.ingest(file_path)


def retrieve_pointers(
    query: str,
    top_k: int = 5,
    doc_hash: str | None = None,
    chroma_path: str = "./chroma_db",
) -> list[DocumentPointer]:
    """Public export: Retrieves top_k relevant DocumentPointer objects from ChromaDB.

    Optionally filters by doc_hash to guarantee no cross-document contamination.
    """
    pipeline = get_pipeline(chroma_path=chroma_path)
    return pipeline.retrieve(query, top_k=top_k, doc_hash=doc_hash)


def clear_pointers(
    doc_hash: str | None = None,
    chroma_path: str = "./chroma_db",
) -> None:
    """Public export: Clears pointers for a specific doc_hash or resets the entire vector store."""
    pipeline = get_pipeline(chroma_path=chroma_path)
    pipeline.clear(doc_hash=doc_hash)


def chunk_text(
    text: str, chunk_size: int = 800, overlap: int = 150
) -> list[str]:
    """Public helper: Splits text into bounded chunks."""
    return TextChunker(chunk_size=chunk_size, overlap=overlap).chunk(text)


__all__ = [
    # Contracts
    "ingest_docx",
    "retrieve_pointers",
    "clear_pointers",
    # Classes
    "DocxHierarchyParser",
    "TextChunker",
    "ChromaPointerStore",
    "DocxIngestionPipeline",
    # Helpers
    "chunk_text",
]
