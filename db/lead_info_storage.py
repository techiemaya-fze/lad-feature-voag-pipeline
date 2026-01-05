"""
Lead Info Storage
Async storage for lead information extracted from call transcriptions.

Uses the same asyncpg + env-based DB config pattern as LeadBookingsStorage,
and writes to lad_dev.leads.
"""

import os
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

try:
    import asyncpg
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    logger.warning("asyncpg not installed. Install with: pip install asyncpg")


SCHEMA = "lad_dev"
LEADS_TABLE = "leads"


class LeadInfoStorageError(Exception):
    """Exception raised for lead info storage errors."""


class LeadInfoStorage:
    """
    Async storage helper for lad_dev.leads driven by lead_info_extractor results.

    Responsibilities:
    - Map extracted lead_info fields into lad_dev.leads columns
    - Upsert (find-or-create) by (tenant_id, phone)
    """

    def __init__(self) -> None:
        self.db_config = {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": int(os.getenv("DB_PORT", "5432")),
            "database": os.getenv("DB_NAME", "salesmaya_agent"),
            "user": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD", ""),
        }
        self._pool: Optional["asyncpg.pool.Pool"] = None

    async def _get_connection(self) -> "asyncpg.Connection":
        if not DB_AVAILABLE:
            raise ImportError("asyncpg not installed")

        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                host=self.db_config["host"],
                port=self.db_config["port"],
                database=self.db_config["database"],
                user=self.db_config["user"],
                password=self.db_config["password"],
                min_size=1,
                max_size=10,
            )

        return await self._pool.acquire()

    async def _return_connection(self, conn: "asyncpg.Connection") -> None:
        if self._pool and conn:
            await self._pool.release(conn)

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def _find_existing_lead_id(
        self,
        conn: "asyncpg.Connection",
        tenant_id: str,
        phone: str,
    ) -> Optional[str]:
        row = await conn.fetchrow(
            f"""
            SELECT id
            FROM {SCHEMA}.{LEADS_TABLE}
            WHERE tenant_id = $1::uuid
              AND phone = $2
            LIMIT 1
            """,
            tenant_id,
            phone,
        )
        return str(row["id"]) if row and row.get("id") else None

    async def save_lead_from_extraction(
        self,
        *,
        tenant_id: Optional[str],
        lead_info: Dict[str, Any],
        call_id: str,
    ) -> Optional[str]:
        """
        Upsert a lead row in lad_dev.leads from extracted lead_info.

        Args:
            tenant_id: Tenant UUID string (required to save)
            lead_info: Dict returned by LeadInfoExtractor.extract_lead_information
            call_id: Call UUID (stored inside custom_fields for traceability)

        Returns:
            Lead ID (UUID string) if saved/updated, else None.
        """
        if not DB_AVAILABLE:
            raise ImportError("asyncpg not installed")

        if not tenant_id:
            logger.warning("Tenant_id is missing, skipping lead save for call_id=%s", call_id)
            return None

        # Basic field mapping from lead_info
        first_name = (lead_info.get("first_name") or "").strip() or None
        full_name = (lead_info.get("full_name") or "").strip() or None
        email = (lead_info.get("email") or "").strip() or None
        phone = (lead_info.get("phone") or "").strip() or None
        whatsapp = (lead_info.get("whatsapp") or "").strip() or None
        notes = (lead_info.get("additional_notes") or "").strip() or None

        # Prefer explicit phone; fall back to whatsapp if needed
        if not phone and whatsapp:
            phone = whatsapp

        if not phone:
            logger.warning(
                "No phone/whatsapp in lead_info for tenant_id=%s, call_id=%s; "
                "skipping lad_dev.leads save.",
                tenant_id,
                call_id,
            )
            return None

        # Derive first_name from full_name if needed
        if not first_name and full_name:
            parts = full_name.split(maxsplit=1)
            first_name = parts[0]
            # We ignore last_name here; schema has first/last but it's optional

        # Prepare custom_fields JSONB with entire lead_info plus call_id
        custom_fields: Dict[str, Any] = dict(lead_info or {})
        custom_fields["source_call_id"] = call_id
        custom_fields_json = json.dumps(custom_fields)

        conn = await self._get_connection()
        try:
            existing_id = await self._find_existing_lead_id(conn, tenant_id, phone)

            if existing_id:
                # Update existing lead (do not overwrite with NULLs)
                result = await conn.fetchrow(
                    f"""
                    UPDATE {SCHEMA}.{LEADS_TABLE}
                    SET
                        first_name = COALESCE($3, first_name),
                        email      = COALESCE($4, email),
                        notes      = COALESCE($5, notes),
                        custom_fields = COALESCE(custom_fields, '{{}}'::jsonb) || $6::jsonb,
                        source     = COALESCE(source, 'voice_agent'),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = $1::uuid
                      AND tenant_id = $2::uuid
                    RETURNING id
                    """,
                    existing_id,
                    tenant_id,
                    first_name,
                    email,
                    notes,
                    custom_fields_json,
                )
                if result and result.get("id"):
                    lead_id = str(result["id"])
                    logger.info("Updated existing lead id=%s for phone=%s", lead_id, phone)
                    return lead_id

                logger.warning(
                    "UPDATE for existing lead id=%s, phone=%s affected no rows", existing_id, phone
                )
                return existing_id

            # Insert new lead
            result = await conn.fetchrow(
                f"""
                INSERT INTO {SCHEMA}.{LEADS_TABLE} (
                    tenant_id,
                    phone,
                    first_name,
                    email,
                    source,
                    notes,
                    custom_fields,
                    status
                ) VALUES (
                    $1::uuid,
                    $2,
                    $3,
                    $4,
                    $5,
                    $6,
                    $7::jsonb,
                    $8
                )
                RETURNING id
                """,
                tenant_id,
                phone,
                first_name,
                email,
                "voice_agent",
                notes,
                custom_fields_json,
                "new",
            )

            if result and result.get("id"):
                lead_id = str(result["id"])
                logger.info("Created new lead id=%s for phone=%s", lead_id, phone)
                return lead_id

            logger.warning(
                "INSERT into %s.%s for phone=%s returned no id",
                SCHEMA,
                LEADS_TABLE,
                phone,
            )
            return None
        except Exception as exc:
            logger.error("Error saving lead for tenant_id=%s, phone=%s: %s", tenant_id, phone, exc, exc_info=True)
            raise LeadInfoStorageError(f"Failed to save lead: {exc}") from exc
        finally:
            await self._return_connection(conn)


