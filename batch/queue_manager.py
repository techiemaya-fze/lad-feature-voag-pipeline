"""Persistent file-backed queue management for batch call jobs."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

ISOFORMAT = "%Y-%m-%dT%H:%M:%SZ"


class CallQueueManager:
    """Stores batch call progress in a local JSON file for durability."""

    def __init__(self, file_path: Path) -> None:
        self._file_path = Path(file_path)
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Ensure the backing file exists with an empty structure."""
        async with self._lock:
            if self._file_path.exists():
                return
            await asyncio.to_thread(self._write_state, {"jobs": {}})

    async def snapshot(self) -> Dict[str, Any]:
        """Return a deep copy of the current queue state."""
        async with self._lock:
            return await asyncio.to_thread(self._read_state)

    async def register_job(
        self,
        job_id: str,
        metadata: Dict[str, Any],
        entries: Iterable[Dict[str, Any]],
    ) -> None:
        """Register a new batch job and its queue entries."""
        async with self._lock:
            state = await asyncio.to_thread(self._read_state)
            jobs = state.setdefault("jobs", {})
            if job_id in jobs:
                raise ValueError(f"Job {job_id} already registered in queue")

            now = datetime.now(timezone.utc).strftime(ISOFORMAT)
            jobs[job_id] = {
                "metadata": {
                    **metadata,
                    "job_id": job_id,
                    "created_at": now,
                    "updated_at": now,
                },
                "entries": [self._normalize_entry(idx, entry) for idx, entry in enumerate(entries)],
            }
            await asyncio.to_thread(self._write_state, state)

    async def iter_job_entries(self, job_id: str) -> List[Dict[str, Any]]:
        async with self._lock:
            state = await asyncio.to_thread(self._read_state)
            job = state.get("jobs", {}).get(job_id)
            if not job:
                return []
            return [dict(entry) for entry in job["entries"]]

    async def update_entry(
        self,
        job_id: str,
        index: int,
        **fields: Any,
    ) -> Dict[str, Any]:
        """Update an entry with arbitrary fields and persist state."""
        async with self._lock:
            state = await asyncio.to_thread(self._read_state)
            job = state.get("jobs", {}).get(job_id)
            if not job:
                raise KeyError(f"Job {job_id} not found")
            if index < 0 or index >= len(job["entries"]):
                raise IndexError(f"Job {job_id} has no entry at index {index}")

            entry = job["entries"][index]
            entry.update(fields)
            job["metadata"]["updated_at"] = datetime.now(timezone.utc).strftime(ISOFORMAT)
            await asyncio.to_thread(self._write_state, state)
            return dict(entry)

    async def next_pending_entry(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Return the next entry still marked as in_queue."""
        entries = await self.iter_job_entries(job_id)
        for entry in sorted(entries, key=lambda item: item["index"]):
            if entry.get("status") == "in_queue":
                return entry
        return None

    async def mark_status(
        self,
        job_id: str,
        index: int,
        status: str,
        error: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {"status": status, "error": error}
        if extra:
            payload.update(extra)
        return await self.update_entry(job_id, index, **payload)

    async def remove_job_if_complete(self, job_id: str) -> bool:
        """Remove a job once all entries have a terminal status."""
        terminal_statuses = {"succeeded", "failed", "error"}
        async with self._lock:
            state = await asyncio.to_thread(self._read_state)
            job = state.get("jobs", {}).get(job_id)
            if not job:
                return False
            if any(entry.get("status") not in terminal_statuses for entry in job["entries"]):
                return False
            state["jobs"].pop(job_id, None)
            await asyncio.to_thread(self._write_state, state)
            return True

    async def rebuild_jobs(self, builder: Callable[[str, Dict[str, Any]], None]) -> None:
        """Replay persisted jobs via the provided callback."""
        async with self._lock:
            state = await asyncio.to_thread(self._read_state)
        for job_id, job_state in state.get("jobs", {}).items():
            builder(job_id, job_state)

    def _normalize_entry(self, index: int, entry: Dict[str, Any]) -> Dict[str, Any]:
        normalized = {
            "index": index,
            "status": entry.get("status", "in_queue"),
            "error": entry.get("error"),
        }
        normalized.update(entry)
        return normalized

    def _read_state(self) -> Dict[str, Any]:
        if not self._file_path.exists():
            return {"jobs": {}}
        with self._file_path.open("r", encoding="utf-8") as handle:
            try:
                return json.load(handle)
            except json.JSONDecodeError:
                return {"jobs": {}}

    def _write_state(self, state: Dict[str, Any]) -> None:
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        with self._file_path.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, sort_keys=True)
