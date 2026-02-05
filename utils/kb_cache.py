"""
KB Stats Cache - Persistent cache for Knowledge Base document counts.

Avoids expensive Gemini API calls by caching document counts locally.
Cache is updated on: create KB, delete KB, upload doc, delete doc.
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# Cache file location
CACHE_DIR = Path(__file__).parent.parent / "data"
CACHE_FILE = CACHE_DIR / "kb_cache.json"


@dataclass
class StoreCache:
    """Cached stats for a single KB store."""
    gemini_store_name: str
    document_count: int
    documents: list[str]  # List of document IDs/names
    last_updated: str


class KBStatsCache:
    """Persistent file-based cache for KB document counts."""
    
    _instance: Optional['KBStatsCache'] = None
    
    def __new__(cls):
        """Singleton pattern - only one cache instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._cache: dict = {"stores": {}, "last_full_sync": None}
        self._load_cache()
    
    def _load_cache(self) -> None:
        """Load cache from disk."""
        try:
            if CACHE_FILE.exists():
                with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                    self._cache = json.load(f)
                logger.info("KB cache loaded: %d stores", len(self._cache.get("stores", {})))
            else:
                logger.info("No existing KB cache found, starting fresh")
        except Exception as e:
            logger.warning("Failed to load KB cache: %s", e)
            self._cache = {"stores": {}, "last_full_sync": None}
    
    def _save_cache(self) -> None:
        """Save cache to disk."""
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(self._cache, f, indent=2)
            logger.debug("KB cache saved")
        except Exception as e:
            logger.error("Failed to save KB cache: %s", e)
    
    def _now(self) -> str:
        """Get current timestamp as ISO string."""
        return datetime.now(timezone.utc).isoformat()
    
    def get_store_stats(self, store_id: str) -> Optional[dict]:
        """Get cached stats for a store by store_id or gemini name."""
        stores = self._cache.get("stores", {})
        
        # Direct lookup by key
        if store_id in stores:
            return stores[store_id]
        
        # Lookup by gemini_store_name
        for key, stats in stores.items():
            if stats.get("gemini_store_name") == store_id:
                return stats
            # Also check if key matches the short name from gemini path
            gemini_name = stats.get("gemini_store_name", "")
            short_name = gemini_name.replace("fileSearchStores/", "")
            if short_name == store_id or key == store_id:
                return stats
        
        return None
    
    def get_by_gemini_name(self, gemini_store_name: str) -> Optional[dict]:
        """Get cached stats by full gemini store name."""
        stores = self._cache.get("stores", {})
        
        # Extract short name
        short_name = gemini_store_name.replace("fileSearchStores/", "")
        
        # Try direct key lookup
        if short_name in stores:
            return stores[short_name]
        
        # Search by gemini_store_name field
        for stats in stores.values():
            if stats.get("gemini_store_name") == gemini_store_name:
                return stats
        
        return None
    
    def get_all_stores(self) -> dict:
        """Get all cached store stats."""
        return self._cache.get("stores", {})
    
    def has_store(self, store_id: str) -> bool:
        """Check if store exists in cache."""
        return self.get_store_stats(store_id) is not None
    
    def set_store_stats(
        self, 
        store_id: str, 
        gemini_store_name: str,
        document_count: int,
        documents: list[str]
    ) -> None:
        """Set/update full stats for a store (used after Gemini API sync)."""
        if "stores" not in self._cache:
            self._cache["stores"] = {}
        
        self._cache["stores"][store_id] = {
            "gemini_store_name": gemini_store_name,
            "document_count": document_count,
            "documents": documents,
            "last_updated": self._now()
        }
        self._save_cache()
        logger.info("Cache updated for store %s: %d docs", store_id, document_count)
    
    def update_on_create_store(self, store_id: str, gemini_store_name: str) -> None:
        """Update cache when a new KB store is created."""
        self.set_store_stats(store_id, gemini_store_name, 0, [])
        logger.info("Cache: new store created %s", store_id)
    
    def update_on_delete_store(self, store_id: str) -> None:
        """Update cache when a KB store is deleted."""
        if "stores" in self._cache and store_id in self._cache["stores"]:
            del self._cache["stores"][store_id]
            self._save_cache()
            logger.info("Cache: store deleted %s", store_id)
    
    def update_on_upload(self, store_id: str, doc_name: str) -> None:
        """Update cache when a document is uploaded."""
        if store_id in self._cache.get("stores", {}):
            store = self._cache["stores"][store_id]
            if doc_name not in store["documents"]:
                store["documents"].append(doc_name)
                store["document_count"] = len(store["documents"])
                store["last_updated"] = self._now()
                self._save_cache()
                logger.debug("Cache: doc added to %s, count=%d", store_id, store["document_count"])
    
    def update_on_delete_doc(self, store_id: str, doc_name: str) -> None:
        """Update cache when a document is deleted."""
        if store_id in self._cache.get("stores", {}):
            store = self._cache["stores"][store_id]
            # Handle both full path and just doc ID
            doc_id = doc_name.split("/")[-1] if "/" in doc_name else doc_name
            
            # Try to remove by exact match or by doc ID suffix
            removed = False
            for i, cached_doc in enumerate(store["documents"]):
                cached_id = cached_doc.split("/")[-1] if "/" in cached_doc else cached_doc
                if cached_id == doc_id or cached_doc == doc_name:
                    store["documents"].pop(i)
                    removed = True
                    break
            
            if removed:
                store["document_count"] = len(store["documents"])
                store["last_updated"] = self._now()
                self._save_cache()
                logger.debug("Cache: doc removed from %s, count=%d", store_id, store["document_count"])
    
    def mark_full_sync(self) -> None:
        """Mark that a full sync was performed."""
        self._cache["last_full_sync"] = self._now()
        self._save_cache()


# Global singleton instance
kb_cache = KBStatsCache()
