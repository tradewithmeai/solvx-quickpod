#!/usr/bin/env python3
"""
SolvX QuickPod - Configuration Module

Manages application configuration including:
- Pod state persistence (pod ID, model info)
- API credentials from environment
- Runtime constants

State is stored in ~/.myai/pod.json for persistence across sessions.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

# =============================================================================
# PATHS
# =============================================================================

STATE_DIR: Path = Path.home() / ".myai"
STATE_FILE: Path = STATE_DIR / "pod.json"

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# HuggingFace model identifier for vLLM
PRIMARY_MODEL: str = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"

# Default system prompt for chat sessions
SYSTEM_PROMPT: str = "You are a helpful assistant."

# =============================================================================
# RUNTIME STATE (loaded from state file)
# =============================================================================

# These values are populated from the state file if it exists.
# None values indicate no active pod session.

POD_ID: Optional[str] = None
MODEL: Optional[str] = None
LLM_PORT: int = 8000
LLM_BASE_URL: Optional[str] = None

# Load state from file if it exists
if STATE_FILE.exists():
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            _state = json.load(f)
            POD_ID = _state.get("pod_id")
            MODEL = _state.get("model", PRIMARY_MODEL)

        if POD_ID:
            LLM_BASE_URL = f"https://{POD_ID}-{LLM_PORT}.proxy.runpod.net"
    except (json.JSONDecodeError, KeyError, IOError):
        # Invalid or unreadable state file - treat as no active pod
        pass

# =============================================================================
# API CREDENTIALS
# =============================================================================

API_KEY: Optional[str] = os.getenv("VLLM_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "VLLM_API_KEY environment variable is not set. "
        "Please run the application to complete onboarding."
    )
