"""Email template storage helpers for database-driven email templates."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, Sequence

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Import connection pool manager
from db.connection_pool import get_db_connection, return_connection, USE_CONNECTION_POOLING

load_dotenv()

logger = logging.getLogger(__name__)


class EmailTemplateStorage:
    """Provides CRUD access to email templates stored in the database."""

    def __init__(self) -> None:
        self.db_config = {
            "host": os.getenv("DB_HOST"),
            "port": int(os.getenv("DB_PORT", "5432")),
            "database": os.getenv("DB_NAME"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
        }

        required = ["DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD"]
        missing = [env for env in required if not os.getenv(env)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    def _get_connection(self) -> psycopg2.extensions.connection:
        """Get database connection (pooled or direct based on feature flag)"""
        return get_db_connection(self.db_config)

    def _return_connection(self, conn) -> None:
        """Return connection to pool if pooling is enabled"""
        if USE_CONNECTION_POOLING:
            return_connection(conn, self.db_config)

    async def get_template_by_id(self, template_id: int) -> Optional[dict[str, Any]]:
        """Fetch a template by its primary key ID."""
        if template_id is None:
            return None

        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        """
                        SELECT id, name, template_key, category, subject_template,
                               text_body_template, html_body_template, placeholders,
                               description, is_active, is_builtin, created_by,
                               agent_id, created_at, updated_at
                        FROM lad_dev.communication_templates
                        WHERE id = %s AND is_active = TRUE
                        LIMIT 1
                        """,
                        (template_id,),
                    )
                    row = cur.fetchone()
                    return dict(row) if row else None
        except Exception as exc:
            logger.error("Failed to fetch template id=%s: %s", template_id, exc, exc_info=True)
            return None

    async def get_template_by_key(
        self,
        template_key: str,
        agent_id: int | None = None,
    ) -> Optional[dict[str, Any]]:
        """
        Fetch a template by its unique key.
        
        If agent_id is provided, first looks for agent-specific template,
        then falls back to global templates (agent_id IS NULL).
        """
        if not template_key:
            return None

        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    if agent_id is not None:
                        # Try agent-specific first
                        cur.execute(
                            """
                            SELECT id, name, template_key, category, subject_template,
                                   text_body_template, html_body_template, placeholders,
                                   description, is_active, is_builtin, created_by,
                                   agent_id, created_at, updated_at
                            FROM lad_dev.communication_templates
                            WHERE template_key = %s
                              AND is_active = TRUE
                              AND (agent_id = %s OR agent_id IS NULL)
                            ORDER BY agent_id NULLS LAST
                            LIMIT 1
                            """,
                            (template_key, agent_id),
                        )
                    else:
                        # Global templates only
                        cur.execute(
                            """
                            SELECT id, name, template_key, category, subject_template,
                                   text_body_template, html_body_template, placeholders,
                                   description, is_active, is_builtin, created_by,
                                   agent_id, created_at, updated_at
                            FROM lad_dev.communication_templates
                            WHERE template_key = %s
                              AND is_active = TRUE
                              AND agent_id IS NULL
                            LIMIT 1
                            """,
                            (template_key,),
                        )
                    row = cur.fetchone()
                    return dict(row) if row else None
        except Exception as exc:
            logger.error("Failed to fetch template key=%s: %s", template_key, exc, exc_info=True)
            return None

    async def list_templates(
        self,
        agent_id: int | None = None,
        category: str | None = None,
        include_builtin: bool = True,
        active_only: bool = True,
    ) -> list[dict[str, Any]]:
        """
        List available email templates.
        
        Args:
            agent_id: Filter by agent (also includes global templates)
            category: Filter by category (e.g., 'meeting_invite', 'follow_up')
            include_builtin: Whether to include built-in system templates
            active_only: Whether to only return active templates
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    conditions = []
                    params: list[Any] = []

                    if active_only:
                        conditions.append("is_active = TRUE")

                    if not include_builtin:
                        conditions.append("is_builtin = FALSE")

                    if agent_id is not None:
                        conditions.append("(agent_id = %s OR agent_id IS NULL)")
                        params.append(agent_id)

                    if category:
                        conditions.append("category = %s")
                        params.append(category)

                    where_clause = " AND ".join(conditions) if conditions else "TRUE"

                    cur.execute(
                        f"""
                        SELECT id, name, template_key, category, subject_template,
                               placeholders, description, is_active, is_builtin,
                               agent_id, created_at
                        FROM lad_dev.communication_templates
                        WHERE {where_clause}
                        ORDER BY is_builtin DESC, name ASC
                        """,
                        params,
                    )
                    rows = cur.fetchall()
                    return [dict(row) for row in rows]
        except Exception as exc:
            logger.error("Failed to list templates: %s", exc, exc_info=True)
            return []

    async def create_template(
        self,
        name: str,
        template_key: str,
        subject_template: str,
        text_body_template: str,
        html_body_template: str | None = None,
        placeholders: list[str] | None = None,
        description: str | None = None,
        category: str = "general",
        is_builtin: bool = False,
        created_by: str | None = None,  # UUID
        agent_id: int | None = None,
    ) -> Optional[dict[str, Any]]:
        """
        Create a new email template.
        
        Args:
            name: Human-readable template name
            template_key: Unique key for the template (e.g., 'meeting_confirmation')
            subject_template: Subject line template with {{placeholders}}
            text_body_template: Plain text body template
            html_body_template: Optional HTML body template
            placeholders: List of placeholder names used in templates
            description: Description of when to use this template
            category: Category for organizing templates
            is_builtin: Whether this is a system-provided template
            created_by: User ID who created this template
            agent_id: Optional agent ID for agent-specific templates
        """
        import json

        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        """
                        INSERT INTO lad_dev.communication_templates
                            (name, template_key, category, subject_template,
                             text_body_template, html_body_template, placeholders,
                             description, is_builtin, created_by, agent_id)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id, name, template_key, category, subject_template,
                                  text_body_template, html_body_template, placeholders,
                                  description, is_active, is_builtin, created_by,
                                  agent_id, created_at, updated_at
                        """,
                        (
                            name,
                            template_key,
                            category,
                            subject_template,
                            text_body_template,
                            html_body_template,
                            json.dumps(placeholders) if placeholders else None,
                            description,
                            is_builtin,
                            created_by,
                            agent_id,
                        ),
                    )
                    row = cur.fetchone()
                    conn.commit()
                    logger.info("Created email template: %s (key=%s)", name, template_key)
                    return dict(row) if row else None
        except psycopg2.errors.UniqueViolation as exc:
            logger.warning("Template key already exists: %s", template_key)
            raise ValueError(f"Template with key '{template_key}' already exists") from exc
        except Exception as exc:
            logger.error("Failed to create template: %s", exc, exc_info=True)
            raise

    async def update_template(
        self,
        template_id: int,
        **updates: Any,
    ) -> Optional[dict[str, Any]]:
        """
        Update an existing template.
        
        Args:
            template_id: ID of the template to update
            **updates: Fields to update (name, subject_template, etc.)
        """
        import json

        allowed_fields = {
            "name", "subject_template", "text_body_template", "html_body_template",
            "placeholders", "description", "category", "is_active",
        }
        
        filtered = {k: v for k, v in updates.items() if k in allowed_fields}
        if not filtered:
            return await self.get_template_by_id(template_id)

        # Handle placeholders JSON encoding
        if "placeholders" in filtered and filtered["placeholders"] is not None:
            filtered["placeholders"] = json.dumps(filtered["placeholders"])

        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    set_clause = ", ".join(f"{k} = %s" for k in filtered)
                    values = list(filtered.values())
                    values.append(template_id)

                    cur.execute(
                        f"""
                        UPDATE lad_dev.communication_templates
                        SET {set_clause}, updated_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                        RETURNING id, name, template_key, category, subject_template,
                                  text_body_template, html_body_template, placeholders,
                                  description, is_active, is_builtin, created_by,
                                  agent_id, created_at, updated_at
                        """,
                        values,
                    )
                    row = cur.fetchone()
                    conn.commit()
                    return dict(row) if row else None
        except Exception as exc:
            logger.error("Failed to update template id=%s: %s", template_id, exc, exc_info=True)
            raise

    async def delete_template(self, template_id: int, soft_delete: bool = True) -> bool:
        """
        Delete a template.
        
        Args:
            template_id: ID of the template to delete
            soft_delete: If True, just marks as inactive; if False, hard deletes
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    if soft_delete:
                        cur.execute(
                            """
                            UPDATE lad_dev.communication_templates
                            SET is_active = FALSE, updated_at = CURRENT_TIMESTAMP
                            WHERE id = %s AND is_builtin = FALSE
                            """,
                            (template_id,),
                        )
                    else:
                        cur.execute(
                            """
                            DELETE FROM lad_dev.communication_templates
                            WHERE id = %s AND is_builtin = FALSE
                            """,
                            (template_id,),
                        )
                    affected = cur.rowcount
                    conn.commit()
                    return affected > 0
        except Exception as exc:
            logger.error("Failed to delete template id=%s: %s", template_id, exc, exc_info=True)
            return False


def render_template(
    template_str: str,
    placeholders: dict[str, str],
    strict: bool = False,
) -> str:
    """
    Render a template string by replacing placeholders.
    
    Placeholder format: {{placeholder_name}}
    
    Args:
        template_str: Template string with {{placeholders}}
        placeholders: Dictionary mapping placeholder names to values
        strict: If True, raises error for missing placeholders
        
    Returns:
        Rendered string with placeholders replaced
    """
    import re

    result = template_str
    
    # Find all placeholders in template
    pattern = r"\{\{(\w+)\}\}"
    found = set(re.findall(pattern, template_str))
    
    for key in found:
        if key in placeholders:
            result = result.replace(f"{{{{{key}}}}}", str(placeholders[key]))
        elif strict:
            raise ValueError(f"Missing required placeholder: {key}")
        # If not strict, leave placeholder as-is for debugging
    
    return result


def get_placeholder_list(template_str: str) -> list[str]:
    """
    Extract all placeholder names from a template string.
    
    Args:
        template_str: Template string with {{placeholders}} or {PLACEHOLDER}
        
    Returns:
        List of unique placeholder names found
    """
    import re
    # Support both {{placeholder}} and {PLACEHOLDER} formats
    pattern1 = r"\{\{(\w+)\}\}"
    pattern2 = r"\{([A-Z_]+)\}"
    found = set(re.findall(pattern1, template_str))
    found.update(re.findall(pattern2, template_str))
    return list(found)


def validate_placeholders(
    template_str: str,
    placeholders: dict[str, str],
) -> tuple[bool, list[str]]:
    """
    Check if all placeholders in the template have been filled.
    
    This is CRITICAL for preventing emails with unfilled placeholders
    like {STUDENT_NAME} from being sent.
    
    Args:
        template_str: Template string to check
        placeholders: Dictionary of provided placeholder values
        
    Returns:
        Tuple of (is_valid, list_of_unfilled_placeholders)
    """
    import re
    
    # Find all placeholder patterns
    pattern1 = r"\{\{(\w+)\}\}"
    pattern2 = r"\{([A-Z_]+)\}"
    
    found = set(re.findall(pattern1, template_str))
    found.update(re.findall(pattern2, template_str))
    
    # Check which are not provided or still have placeholder-like values
    unfilled = []
    for placeholder in found:
        value = placeholders.get(placeholder) or placeholders.get(placeholder.lower())
        if not value:
            unfilled.append(placeholder)
        # Also check if the value itself looks like a placeholder (wasn't replaced)
        elif value.startswith("{") and value.endswith("}"):
            unfilled.append(placeholder)
    
    return len(unfilled) == 0, unfilled


def render_glinks_template(
    template_str: str,
    placeholders: dict[str, str],
    strict: bool = True,
) -> str:
    """
    Render a Glinks template string by replacing placeholders.
    
    Supports both {{placeholder_name}} and {PLACEHOLDER_NAME} formats.
    
    Args:
        template_str: Template string with placeholders
        placeholders: Dictionary mapping placeholder names to values
        strict: If True (default), raises error for unfilled placeholders
        
    Returns:
        Rendered string with placeholders replaced
        
    Raises:
        ValueError: If strict=True and any placeholder is unfilled
    """
    import re
    
    result = template_str
    
    # Replace {{placeholder}} format (case-insensitive key lookup)
    pattern1 = r"\{\{(\w+)\}\}"
    for match in re.finditer(pattern1, template_str):
        key = match.group(1)
        value = placeholders.get(key) or placeholders.get(key.upper()) or placeholders.get(key.lower())
        if value:
            result = result.replace(f"{{{{{key}}}}}", str(value))
    
    # Replace {PLACEHOLDER} format (case-insensitive key lookup)
    pattern2 = r"\{([A-Z_]+)\}"
    for match in re.finditer(pattern2, template_str):
        key = match.group(1)
        value = placeholders.get(key) or placeholders.get(key.lower())
        if value:
            result = result.replace(f"{{{key}}}", str(value))
    
    # Validate no placeholders remain
    if strict:
        is_valid, unfilled = validate_placeholders(result, {})
        # Re-check with empty dict to find any remaining placeholders in result
        remaining1 = set(re.findall(pattern1, result))
        remaining2 = set(re.findall(pattern2, result))
        remaining = remaining1.union(remaining2)
        if remaining:
            raise ValueError(
                f"Cannot send email with unfilled placeholders: {', '.join(remaining)}. "
                f"Please provide values for all placeholders."
            )
    
    return result


__all__ = [
    "EmailTemplateStorage",
    "render_template",
    "render_glinks_template",
    "get_placeholder_list",
    "validate_placeholders",
]
