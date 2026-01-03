-- Migration: Expand varchar columns to TEXT in voice_call_analysis
-- Reason: Values like "PROCEED IMMEDIATELY" exceed varchar(20) limit
-- Date: 2024-12-30

-- Expand short varchar columns to TEXT for flexible content
ALTER TABLE lad_dev.voice_call_analysis 
    ALTER COLUMN lead_category TYPE TEXT,
    ALTER COLUMN engagement_level TYPE TEXT,
    ALTER COLUMN disposition TYPE TEXT;

-- Verify changes
SELECT column_name, data_type, character_maximum_length
FROM information_schema.columns
WHERE table_schema = 'lad_dev' 
  AND table_name = 'voice_call_analysis'
  AND column_name IN ('lead_category', 'engagement_level', 'disposition');
