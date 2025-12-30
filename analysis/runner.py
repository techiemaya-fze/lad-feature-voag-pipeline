from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Iterable, Mapping

from .merged_analytics import analytics

logger = logging.getLogger("post_call_analysis.runner")


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        text = value.rstrip("Z")
        if value.endswith("Z"):
            text = value[:-1] + "+00:00"
        return datetime.fromisoformat(text)
    except Exception:  # noqa: BLE001
        return None


def _build_conversation(transcription: Mapping[str, Any]) -> tuple[list[dict[str, Any]], datetime | None, datetime | None]:
    segments = transcription.get("segments") or []
    conversation: list[dict[str, Any]] = []
    for segment in segments:
        text = (segment.get("text") or "").strip()
        if not text:
            continue
        speaker = (segment.get("speaker") or "").strip().lower()
        role = "user" if speaker == "user" else "agent"
        conversation.append(
            {
                "role": role,
                "message": text,
                "timestamp": segment.get("timestamp"),
            }
        )
    started_at = transcription.get("started_at")
    ended_at = transcription.get("ended_at")
    return conversation, _parse_iso(started_at), _parse_iso(ended_at)


def _resolve_duration(
    provided_duration: float | int | None,
    call_details: Mapping[str, Any] | None,
    started_at: datetime | None,
    ended_at: datetime | None,
) -> int:
    if provided_duration and provided_duration > 0:
        return int(provided_duration)
    if call_details:
        start = call_details.get("started_at")
        end = call_details.get("ended_at")
        if start and end:
            try:
                return int((end - start).total_seconds())
            except Exception:  # noqa: BLE001
                pass
    if started_at and ended_at:
        try:
            return int((ended_at - started_at).total_seconds())
        except Exception:  # noqa: BLE001
            return 0
    return 0


async def run_post_call_analysis(
    *,
    call_log_id: str,
    transcription_json: str | dict,  # Can be dict (JSONB) or str (legacy TEXT)
    duration_seconds: float | int | None,
    call_details: Mapping[str, Any] | None,
    db_config: Mapping[str, Any] | None,
    tenant_id: str | None = None,  # Multi-tenancy support
) -> bool:
    """Invoke the legacy merged analytics pipeline and persist results."""

    if not transcription_json:
        logger.info("No transcription found for call_log_id=%s; skipping analysis", call_log_id)
        return False

    # Handle both dict (JSONB) and str (legacy TEXT) formats
    if isinstance(transcription_json, dict):
        transcription = transcription_json
    else:
        try:
            transcription = json.loads(transcription_json)
        except json.JSONDecodeError:
            logger.warning("Malformed transcription JSON for call_log_id=%s; skipping analysis", call_log_id)
            return False

    conversation, started_at, ended_at = _build_conversation(transcription)

    if not conversation:
        logger.info("Empty conversation for call_log_id=%s; skipping analysis", call_log_id)
        return False

    resolved_duration = _resolve_duration(duration_seconds, call_details, started_at, ended_at)

    try:
        analysis = await analytics.analyze_call(
            call_id=str(call_log_id),
            conversation_log=conversation,
            duration_seconds=resolved_duration,
            call_start_time=started_at,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Call analytics failed for call_log_id=%s: %s", call_log_id, exc, exc_info=True)
        return False

    if not db_config:
        logger.warning("No database configuration provided; analysis result not persisted for call_log_id=%s", call_log_id)
        return False

    # Add tenant_id to analysis dict for saving
    if tenant_id:
        analysis["tenant_id"] = tenant_id
    
    try:
        saved = analytics.save_to_database(analysis, call_log_id, dict(db_config))
    except Exception as exc:  # noqa: BLE001
        logger.error("Saving analytics failed for call_log_id=%s: %s", call_log_id, exc, exc_info=True)
        return False

    return bool(saved)
