"""
Local storage module for chat transcripts and user profiles.

Saves chat logs to ~/.myai/chat_logs/{session_uuid}.jsonl
Each message is written with open/write/flush/close (no held handles).
"""

import uuid
import json
from pathlib import Path
from datetime import datetime, timezone

STORAGE_DIR = Path.home() / ".myai"
CHAT_LOGS_DIR = STORAGE_DIR / "chat_logs"
USER_FILE = STORAGE_DIR / "user.json"


def new_session() -> str:
    """Generate UUID for new session."""
    return str(uuid.uuid4())


def log_message(session_id: str, role: str, content: str):
    """
    Append message to session JSONL file.

    Opens, writes, flushes, and closes file each call.
    No file handles are held between calls.
    """
    CHAT_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = CHAT_LOGS_DIR / f"{session_id}.jsonl"

    entry = {
        "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "role": role,
        "content": content
    }

    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        f.flush()
    # File closed on exit - no held handles


def get_or_create_user() -> dict:
    """
    Read or create user profile at ~/.myai/user.json

    Returns dict with user_id, created_at, last_seen.
    Updates last_seen on each call.
    """
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    if USER_FILE.exists():
        with open(USER_FILE, "r", encoding="utf-8") as f:
            user = json.load(f)
    else:
        user = {
            "user_id": "default",
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        }

    # Update last_seen
    user["last_seen"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    with open(USER_FILE, "w", encoding="utf-8") as f:
        json.dump(user, f, indent=2)
        f.flush()

    return user


def touch_user():
    """Update user's last_seen timestamp."""
    get_or_create_user()  # This already updates last_seen
