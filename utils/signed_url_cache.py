from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional


@dataclass(slots=True)
class CachedSignedUrl:
	"""Represents a cached signed URL entry with its expiry metadata."""

	key: str
	signed_url: str
	gs_url: str
	cached_at: datetime
	expires_at: datetime
	call_log_id: str | None = None


class SignedUrlCache:
	"""In-memory TTL cache for call recording signed URLs."""

	def __init__(self, ttl: timedelta) -> None:
		self._ttl = ttl
		self._cache: Dict[str, CachedSignedUrl] = {}
		self._lock = asyncio.Lock()

	@property
	def ttl(self) -> timedelta:
		"""Return the configured time-to-live window."""

		return self._ttl

	async def get(self, key: str) -> Optional[CachedSignedUrl]:
		"""Return a non-expired cache entry for the given cache key if available."""

		now = datetime.now(timezone.utc)
		async with self._lock:
			entry = self._cache.get(key)
			if entry is None:
				return None
			if entry.expires_at <= now:
				self._cache.pop(key, None)
				return None
			return entry

	async def set(self, key: str, signed_url: str, gs_url: str, *, call_log_id: str | None = None) -> CachedSignedUrl:
		"""Store a fresh cache entry for the key and return it."""

		now = datetime.now(timezone.utc)
		entry = CachedSignedUrl(
			key=key,
			signed_url=signed_url,
			gs_url=gs_url,
			cached_at=now,
			expires_at=now + self._ttl,
			call_log_id=call_log_id,
		)
		async with self._lock:
			self._cache[key] = entry
		return entry

	async def purge_expired(self) -> int:
		"""Remove expired entries; primarily useful for upkeep hooks."""

		now = datetime.now(timezone.utc)
		async with self._lock:
			expired_keys = [key for key, entry in self._cache.items() if entry.expires_at <= now]
			for key in expired_keys:
				self._cache.pop(key, None)
			return len(expired_keys)
