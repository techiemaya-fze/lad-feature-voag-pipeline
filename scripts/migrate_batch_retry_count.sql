-- Migration: Add retry_count column to voice_call_batch_entries
-- Date: 2026-01-16
-- Purpose: Track retry count for wave dispatch expired requests

-- Add retry_count column with default 0
ALTER TABLE lad_dev.voice_call_batch_entries 
ADD COLUMN IF NOT EXISTS retry_count INTEGER DEFAULT 0;

-- Add index for efficient status queries during wave processing
CREATE INDEX IF NOT EXISTS idx_batch_entries_status_retry 
ON lad_dev.voice_call_batch_entries(batch_id, status, retry_count) 
WHERE is_deleted = FALSE;

-- Update existing entries to have retry_count = 0 (just in case)
UPDATE lad_dev.voice_call_batch_entries 
SET retry_count = 0 
WHERE retry_count IS NULL;

-- Add comment for documentation
COMMENT ON COLUMN lad_dev.voice_call_batch_entries.retry_count IS 
'Number of times this entry was reset from dispatched to queued after wave timeout. Max 2 retries.';
