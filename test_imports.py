#!/usr/bin/env python3
"""Integration test for v2 module imports"""

import sys
sys.path.insert(0, '.')

errors = []

# Test recording module
try:
    from recording.recorder import CallRecorder, TranscriptionSegment, CallTranscription, LocalBufferManager
    print("✓ recording.recorder imports OK")
except Exception as e:
    errors.append(f"recording.recorder: {e}")
    print(f"✗ recording.recorder: {e}")

try:
    from recording.transcription import TranscriptionTracker, attach_transcription_tracker
    print("✓ recording.transcription imports OK")
except Exception as e:
    errors.append(f"recording.transcription: {e}")
    print(f"✗ recording.transcription: {e}")

try:
    from recording.api import RecordingAPI
    print("✓ recording.api imports OK")
except Exception as e:
    errors.append(f"recording.api: {e}")
    print(f"✗ recording.api: {e}")

# Test storage module
try:
    from storage.gcs import GCSStorageManager
    print("✓ storage.gcs imports OK")
except Exception as e:
    errors.append(f"storage.gcs: {e}")
    print(f"✗ storage.gcs: {e}")

# Test batch module
try:
    from batch.queue_manager import CallQueueManager
    print("✓ batch.queue_manager imports OK")
except Exception as e:
    errors.append(f"batch.queue_manager: {e}")
    print(f"✗ batch.queue_manager: {e}")

# Test db storage
try:
    from db.storage import BatchStorage, CallStorage
    print("✓ db.storage imports OK")
except Exception as e:
    errors.append(f"db.storage: {e}")
    print(f"✗ db.storage: {e}")

# Test tools
try:
    from tools.google_workspace import AgentGoogleWorkspace
    print("✓ tools.google_workspace imports OK")
except Exception as e:
    errors.append(f"tools.google_workspace: {e}")
    print(f"✗ tools.google_workspace: {e}")

try:
    from tools.microsoft_bookings import AgentMicrosoftBookings
    print("✓ tools.microsoft_bookings imports OK")
except Exception as e:
    errors.append(f"tools.microsoft_bookings: {e}")
    print(f"✗ tools.microsoft_bookings: {e}")

try:
    from tools.file_search_tool import FileSearchTool
    print("✓ tools.file_search_tool imports OK")
except Exception as e:
    errors.append(f"tools.file_search_tool: {e}")
    print(f"✗ tools.file_search_tool: {e}")

# Summary
print("\n" + "="*50)
if errors:
    print(f"FAILED: {len(errors)} import error(s)")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("SUCCESS: All v2 module imports passed!")
    sys.exit(0)
