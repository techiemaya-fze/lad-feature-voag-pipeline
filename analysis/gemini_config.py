"""
Gemini API Configuration

Centralized configuration for all Gemini API calls in the analysis module.
"""

import os

# Model Configuration
MODEL_NAME = "gemini-3-flash-preview"  # Gemini 3 Flash with thinking support

# Default Generation Parameters
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_OUTPUT_TOKENS = 16384  # Very high limit to prevent any truncation

# API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Retry Configuration  
MAX_RETRIES = 5  # Increased retries for robustness
RETRY_DELAY_BASE = 1  # Base delay in seconds for exponential backoff

# Timeout Configuration
REQUEST_TIMEOUT = 60.0  # seconds
