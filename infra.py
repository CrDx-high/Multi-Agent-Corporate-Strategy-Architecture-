"""Infrastructure modules: Async Token Bucket Rate Limiters and SQLite State Checkpointer."""

import asyncio
from contextlib import closing, contextmanager
import os
import sqlite3
import time
from typing import Any, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class AsyncTokenBucketLimiter:
    """Asynchronous Token Bucket Rate Limiter.

    Limits request starts per minute according to a configured RPM, not concurrency.
    Throttles operations using asyncio.Lock(), time.monotonic(), and non-blocking
    await asyncio.sleep().
    """

    def __init__(self, requests_per_minute: int):
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be strictly positive")
        self.requests_per_minute: int = requests_per_minute
        self.capacity: float = float(requests_per_minute)
        self.tokens: float = float(requests_per_minute)
        self.refill_rate: float = float(requests_per_minute) / 60.0
        self.last_refill: float = time.monotonic()
        self._locks: dict[int, asyncio.Lock] = {}

    @property
    def lock(self) -> asyncio.Lock:
        try:
            loop = asyncio.get_running_loop()
            loop_id = id(loop)
        except RuntimeError:
            loop_id = 0
        if loop_id not in self._locks:
            self._locks[loop_id] = asyncio.Lock()
        return self._locks[loop_id]

    async def acquire(self, tokens: float = 1.0) -> None:
        """Acquires tokens from the bucket, awaiting non-blocking sleep if rate limited."""
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.last_refill = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)

            if self.tokens < tokens:
                needed = tokens - self.tokens
                wait_time = needed / self.refill_rate
                await asyncio.sleep(wait_time)
                self.last_refill = time.monotonic()
                self.tokens = 0.0
            else:
                self.tokens -= tokens

    async def __aenter__(self):
        await self.acquire(1.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


class StateCheckpointer:
    """SQLite-backed state checkpointer operating in WAL mode."""

    def __init__(self, db_path: str = "./checkpoints.db"):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _get_connection(self):
        db_dir = os.path.dirname(os.path.abspath(self.db_path))
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        with closing(sqlite3.connect(self.db_path, timeout=30.0)) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            with conn:
                yield conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    job_id TEXT NOT NULL,
                    node_name TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (job_id, node_name)
                );
                """
            )

    def save_checkpoint(self, job_id: str, node_name: str, payload: BaseModel) -> None:
        """Serializes and persists a Pydantic model checkpoint for a given job and node."""
        payload_json = payload.model_dump_json()
        now = time.time()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO checkpoints (job_id, node_name, payload, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(job_id, node_name) DO UPDATE SET
                    payload = excluded.payload,
                    updated_at = excluded.updated_at;
                """,
                (job_id, node_name, payload_json, now),
            )

    def load_checkpoint(
        self, job_id: str, node_name: str, model_cls: type[T]
    ) -> T | None:
        """Loads and deserializes a Pydantic model checkpoint, or returns None if not found.

        Matches exact node_name, or if node_name has no hash suffix, matches latest node_name:*.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT payload FROM checkpoints
                WHERE job_id = ? AND (node_name = ? OR node_name LIKE ?)
                ORDER BY updated_at DESC LIMIT 1;
                """,
                (job_id, node_name, f"{node_name}:%"),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return model_cls.model_validate_json(row[0])


class _LazyCheckpointerProxy:
    """Lazy proxy ensuring StateCheckpointer is only instantiated on first attribute access."""

    def __init__(self, db_path: str = "./checkpoints.db"):
        self._db_path = db_path
        self._instance: StateCheckpointer | None = None

    def _get_instance(self) -> StateCheckpointer:
        if self._instance is None:
            self._instance = StateCheckpointer(db_path=self._db_path)
        return self._instance

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get_instance(), name)


# Exported rate limiter instances
limiter_proposer = AsyncTokenBucketLimiter(requests_per_minute=10)
limiter_compliance = AsyncTokenBucketLimiter(requests_per_minute=20)
limiter_bottleneck = AsyncTokenBucketLimiter(requests_per_minute=20)
limiter_arbiter = AsyncTokenBucketLimiter(requests_per_minute=5)

# Exported lazy checkpointer instance
checkpointer = _LazyCheckpointerProxy()

__all__ = [
    "AsyncTokenBucketLimiter",
    "limiter_proposer",
    "limiter_compliance",
    "limiter_bottleneck",
    "limiter_arbiter",
    "StateCheckpointer",
    "checkpointer",
]
