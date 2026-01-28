#!/usr/bin/env python3
"""
SolvX QuickPod - Storage Module

Handles local persistence for chat transcripts and user profiles.

Storage locations:
    ~/.myai/chat_logs/{session_uuid}.jsonl  - Chat session logs
    ~/.myai/user.json                       - User profile data

File handling:
    All writes use open/write/flush/close pattern to ensure data integrity.
    No file handles are held between operations.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

# =============================================================================
# STORAGE PATHS
# =============================================================================

STORAGE_DIR: Path = Path.home() / ".myai"
CHAT_LOGS_DIR: Path = STORAGE_DIR / "chat_logs"
USER_FILE: Path = STORAGE_DIR / "user.json"


# =============================================================================
# SESSION MANAGEMENT
# =============================================================================

def new_session() -> str:
    """
    Generate a unique session identifier.

    Returns:
        A UUID string for the new chat session.
    """
    return str(uuid.uuid4())


# =============================================================================
# CHAT LOGGING
# =============================================================================

def log_message(session_id: str, role: str, content: str) -> None:
    """
    Append a message to the session's chat log.

    Each message is stored as a JSON line with timestamp, role, and content.
    The file is opened, written, flushed, and closed on each call to ensure
    data persistence without holding file handles.

    Args:
        session_id: The UUID of the current chat session.
        role: The message sender role ('user', 'assistant', or 'system').
        content: The message content.
    """
    CHAT_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = CHAT_LOGS_DIR / f"{session_id}.jsonl"

    entry = {
        "ts": _utc_timestamp(),
        "role": role,
        "content": content,
    }

    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        f.flush()


# =============================================================================
# USER PROFILE
# =============================================================================

def get_or_create_user() -> Dict[str, str]:
    """
    Retrieve or create the user profile.

    The profile is stored at ~/.myai/user.json and contains:
        - user_id: User identifier (default: "default")
        - created_at: Profile creation timestamp
        - last_seen: Most recent activity timestamp (updated on each call)

    Returns:
        Dictionary containing user profile data.
    """
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    if USER_FILE.exists():
        with open(USER_FILE, "r", encoding="utf-8") as f:
            user = json.load(f)
    else:
        user = {
            "user_id": "default",
            "created_at": _utc_timestamp(),
        }

    # Update last seen timestamp
    user["last_seen"] = _utc_timestamp()

    with open(USER_FILE, "w", encoding="utf-8") as f:
        json.dump(user, f, indent=2)
        f.flush()

    return user


def touch_user() -> None:
    """
    Update the user's last_seen timestamp.

    Convenience wrapper around get_or_create_user() for activity tracking.
    """
    get_or_create_user()


# =============================================================================
# UTILITIES
# =============================================================================

def _utc_timestamp() -> str:
    """
    Generate an ISO 8601 UTC timestamp.

    Returns:
        Timestamp string in format: 2024-01-15T10:30:00Z
    """
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
