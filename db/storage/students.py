"""
Student Storage Module
======================

Handles student creation, lookup, and management for Glinks calls.
Mirrors the lead_storage.py pattern but uses the students_voiceagent table.

Students table uses UUID primary key (auto-generated), unlike leads which use BIGINT.
"""

import logging
from typing import Optional, Dict

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Import connection pool manager
from db.connection_pool import get_raw_connection, return_connection, USE_CONNECTION_POOLING
# Import centralized DB config (respects USE_LOCAL_DB toggle)
from db.db_config import get_db_config

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


class StudentStorage:
    """Manages student data in the database for Glinks calls"""
    
    def __init__(self):
        """Initialize database connection using centralized config"""
        self.db_config = get_db_config()
    
    def _get_connection(self):
        """Get raw database connection (must be returned with _return_connection)"""
        return get_raw_connection(self.db_config)
    
    def _return_connection(self, conn):
        """Return connection to pool if pooling is enabled"""
        if USE_CONNECTION_POOLING:
            return_connection(conn, self.db_config)
    
    async def get_student_by_contact(self, parent_contact: str) -> Optional[Dict]:
        """
        Get student by parent contact number (equivalent to lead_number for leads)
        
        Args:
            parent_contact: Phone number to search for
            
        Returns:
            Student dict if found, None otherwise
        """
        conn = None
        cursor = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute(
                """
                SELECT id, student_name, parent_name, parent_contact, email,
                       country_of_residence, grade_year, school_name,
                       counsellor_meeting_link, tags, stage, status,
                       counsellor_email, created_at
                FROM lad_dev.education_students
                WHERE parent_contact = %s
                LIMIT 1
                """,
                (parent_contact,)
            )
            
            result = cursor.fetchone()
            cursor.close()
            self._return_connection(conn)
            
            if result:
                # Convert UUID to string for consistent handling
                result_dict = dict(result)
                result_dict['id'] = str(result_dict['id'])
                logger.info(f"Found existing student: id={result_dict['id']}, contact={parent_contact}")
                return result_dict
            else:
                logger.debug(f"No student found for contact: {parent_contact}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting student by contact {parent_contact}: {e}", exc_info=True)
            if cursor:
                cursor.close()
            if conn:
                self._return_connection(conn)
            return None
    
    async def create_student(
        self,
        parent_contact: str,
        student_name: Optional[str] = None,
        parent_name: Optional[str] = None,
        email: Optional[str] = None,
        country_of_residence: str = "Unknown",
        lead_source: str = "Voice Agent",
        counsellor_meeting_link: str = "",
        grade_year: Optional[str] = None,
        school_name: Optional[str] = None,
        tags: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        Create a new student record (UUID is auto-generated)
        
        Args:
            parent_contact: Phone number for the student/parent
            student_name: Name of the student
            parent_name: Name of the parent
            email: Email address
            country_of_residence: Country (required field, defaults to "Unknown")
            lead_source: Source of the lead (required field, defaults to "Voice Agent")
            counsellor_meeting_link: Meeting link (required field, can be empty)
            grade_year: Grade/year of the student
            school_name: Name of school
            tags: Lead tag (hot/warm/cold)
            
        Returns:
            Student record dict with UUID id if successful, None otherwise
        """
        conn = None
        cursor = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute(
                """
                INSERT INTO lad_dev.education_students 
                (parent_contact, student_name, parent_name, email, 
                 country_of_residence, lead_source, counsellor_meeting_link,
                 grade_year, school_name, tags)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    parent_contact, 
                    student_name, 
                    parent_name, 
                    email,
                    country_of_residence,
                    lead_source,
                    counsellor_meeting_link,
                    grade_year,
                    school_name,
                    tags
                )
            )
            
            record = cursor.fetchone()
            student_id = str(record[0]) if record else None  # UUID as string
            conn.commit()
            cursor.close()
            self._return_connection(conn)
            
            if student_id is None:
                logger.warning(f"No student id returned when creating contact={parent_contact}")
                return None

            logger.info(f"Created new student: id={student_id}, contact={parent_contact}")
            return {
                "id": student_id,
                "parent_contact": parent_contact,
                "student_name": student_name,
                "parent_name": parent_name,
                "email": email,
                "country_of_residence": country_of_residence,
                "lead_source": lead_source,
                "counsellor_meeting_link": counsellor_meeting_link,
                "grade_year": grade_year,
                "school_name": school_name,
                "tags": tags,
            }
            
        except psycopg2.IntegrityError as e:
            # Handle potential duplicates
            logger.warning(f"Student with contact {parent_contact} may already exist: {e}")
            if cursor:
                cursor.close()
            if conn:
                conn.rollback()
                self._return_connection(conn)
            # Try to get existing student
            existing = await self.get_student_by_contact(parent_contact)
            return existing
            
        except Exception as e:
            logger.error(f"Error creating student for contact {parent_contact}: {e}", exc_info=True)
            if cursor:
                cursor.close()
            if conn:
                conn.rollback()
                self._return_connection(conn)
            return None

    async def _set_name_if_missing(self, student_id: str, student_name: str) -> bool:
        """Set the student name only when it is currently null or blank."""
        if not student_name:
            return False

        conn = None
        cursor = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE lad_dev.education_students
                SET student_name = %s
                WHERE id = %s::uuid
                AND (student_name IS NULL OR LENGTH(TRIM(student_name)) = 0)
                RETURNING id
                """,
                (student_name, student_id)
            )
            updated = cursor.fetchone() is not None
            conn.commit()
            cursor.close()
            self._return_connection(conn)
            if updated:
                logger.info(f"Updated missing name for student id={student_id}")
            return updated
        except Exception as e:
            logger.error(f"Error updating missing name for student {student_id}: {e}", exc_info=True)
            if cursor:
                cursor.close()
            if conn:
                conn.rollback()
                self._return_connection(conn)
            return False
    
    async def find_or_create_student(
        self,
        parent_contact: str,
        student_name: Optional[str] = None,
        parent_name: Optional[str] = None,
        country_of_residence: str = "Unknown",
        lead_source: str = "Voice Agent",
        counsellor_meeting_link: str = "",
    ) -> Optional[Dict]:
        """
        Find existing student by parent contact or create new one.
        Mirrors the lead_storage.find_or_create_lead pattern.
        
        Args:
            parent_contact: Phone number to search for or create
            student_name: Optional name for the student (for new students or to fill in missing)
            parent_name: Optional name for the parent
            country_of_residence: Country (for new students)
            lead_source: Source of the lead (for new students)
            counsellor_meeting_link: Meeting link (for new students)
            
        Returns:
            Student record dict with UUID id if successful, None otherwise
        """
        try:
            trimmed_name = student_name.strip() if student_name else None
            trimmed_parent = parent_name.strip() if parent_name else None

            # First try to find existing student
            existing_student = await self.get_student_by_contact(parent_contact)

            if existing_student:
                logger.info(f"Using existing student: id={existing_student['id']}, contact={parent_contact}")
                has_name = bool(existing_student.get('student_name') and existing_student['student_name'].strip())
                if trimmed_name and not has_name:
                    name_was_set = await self._set_name_if_missing(existing_student['id'], trimmed_name)
                    if name_was_set:
                        existing_student['student_name'] = trimmed_name
                return existing_student

            # If not found, create new student
            logger.info(f"Creating new student for contact: {parent_contact}")
            student_record = await self.create_student(
                parent_contact=parent_contact,
                student_name=trimmed_name,
                parent_name=trimmed_parent,
                country_of_residence=country_of_residence,
                lead_source=lead_source,
                counsellor_meeting_link=counsellor_meeting_link,
            )

            return student_record

        except Exception as e:
            logger.error(f"Error in find_or_create_student for {parent_contact}: {e}", exc_info=True)
            return None
    
    async def get_student_by_id(self, student_id: str) -> Optional[Dict]:
        """
        Get student by UUID
        
        Args:
            student_id: Student UUID as string
            
        Returns:
            Student dict if found, None otherwise
        """
        conn = None
        cursor = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute(
                """
                SELECT id, student_name, parent_name, parent_contact, email,
                       country_of_residence, grade_year, school_name,
                       counsellor_meeting_link, tags, stage, status,
                       counsellor_email, created_at
                FROM lad_dev.education_students
                WHERE id = %s::uuid
                """,
                (student_id,)
            )
            
            result = cursor.fetchone()
            cursor.close()
            self._return_connection(conn)
            
            if result:
                result_dict = dict(result)
                result_dict['id'] = str(result_dict['id'])
                return result_dict
            else:
                logger.debug(f"No student found for id: {student_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting student by id {student_id}: {e}", exc_info=True)
            if cursor:
                cursor.close()
            if conn:
                self._return_connection(conn)
            return None

    async def update_student_info(
        self,
        student_id: str,
        student_name: Optional[str] = None,
        parent_name: Optional[str] = None,
        tags: Optional[str] = None,
        stage: Optional[str] = None,
        status: Optional[str] = None,
    ) -> bool:
        """
        Update student information
        
        Args:
            student_id: Student UUID to update
            student_name: New student name (optional)
            parent_name: New parent name (optional)
            tags: New tag value (optional)
            stage: New stage value (optional)
            status: New status value (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            updates = {}
            if student_name is not None:
                updates["student_name"] = student_name
            if parent_name is not None:
                updates["parent_name"] = parent_name
            if tags is not None:
                updates["tags"] = tags
            if stage is not None:
                updates["stage"] = stage
            if status is not None:
                updates["status"] = status
            
            if not updates:
                logger.warning("No fields to update")
                return False
            
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Build dynamic UPDATE query
            set_clauses = [f"{key} = %s" for key in updates.keys()]
            params = list(updates.values())
            params.append(student_id)
            
            query = f"""
                UPDATE lad_dev.education_students
                SET {', '.join(set_clauses)}
                WHERE id = %s::uuid
            """
            
            cursor.execute(query, params)
            rows_updated = cursor.rowcount
            conn.commit()
            cursor.close()
            self._return_connection(conn)
            
            if rows_updated > 0:
                logger.info(f"Updated student: id={student_id}")
                return True
            else:
                logger.warning(f"No student found with id={student_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating student {student_id}: {e}", exc_info=True)
            if conn:
                conn.rollback()
                self._return_connection(conn)
            return False
