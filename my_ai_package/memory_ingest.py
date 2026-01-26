"""
Memory Ingest - Event bus for capturing chat and tool events.

Events are stored as JSONL files (append-only) with offset tracking
to support incremental processing by the memory worker.

Event types:
- chat.user: User message
- chat.assistant: Assistant response
- tool.call: Tool invocation
- tool.result: Tool success result
- tool.error: Tool failure
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Memory events directory
MEMORY_EVENTS_DIR = Path.home() / ".myai" / "memory_events"


def _ensure_dir():
    """Create memory events directory if it doesn't exist."""
    MEMORY_EVENTS_DIR.mkdir(parents=True, exist_ok=True)


def _events_path(session_id: str) -> Path:
    """Get path to events JSONL file for a session."""
    return MEMORY_EVENTS_DIR / f"{session_id}.jsonl"


def _offset_path(session_id: str) -> Path:
    """Get path to offset file for a session."""
    return MEMORY_EVENTS_DIR / f"{session_id}.offset"


def emit_event(
    session_id: str,
    event_type: str,
    payload: dict[str, Any],
    meta: dict[str, Any] | None = None,
) -> None:
    """
    Emit an event to the session's event stream.

    Args:
        session_id: Session identifier
        event_type: One of chat.user, chat.assistant, tool.call, tool.result, tool.error
        payload: Event-specific data
        meta: Optional metadata
    """
    _ensure_dir()

    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "type": event_type,
        "session_id": session_id,
        "payload": payload,
        "meta": meta or {},
    }

    # Append to JSONL file (open/write/flush/close pattern)
    events_file = _events_path(session_id)
    with open(events_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
        f.flush()
        os.fsync(f.fileno())


def get_events(
    session_id: str, since_offset: int = 0, max_events: int = 0
) -> tuple[list[dict[str, Any]], int]:
    """
    Read events from a session's event stream starting at offset.

    Args:
        session_id: Session identifier
        since_offset: Byte offset to start reading from
        max_events: Maximum events to read (0 = unlimited)

    Returns:
        Tuple of (list of events, new byte offset after last event read)
    """
    events_file = _events_path(session_id)

    if not events_file.exists():
        return [], 0

    events = []
    new_offset = since_offset

    with open(events_file, "r", encoding="utf-8") as f:
        # Seek to offset
        f.seek(since_offset)

        # Read lines using readline() to allow f.tell() to work
        while True:
            line = f.readline()
            if not line:
                break  # EOF

            line_stripped = line.strip()
            if line_stripped:
                try:
                    events.append(json.loads(line_stripped))
                    # Update offset AFTER successfully parsing the line
                    new_offset = f.tell()

                    # Stop if we've reached max_events
                    if max_events > 0 and len(events) >= max_events:
                        break
                except json.JSONDecodeError:
                    # Skip malformed lines but still advance offset
                    new_offset = f.tell()
                    continue
            else:
                # Empty line, still advance offset
                new_offset = f.tell()

    return events, new_offset


def get_offset(session_id: str) -> int:
    """
    Get the last processed byte offset for a session.

    Args:
        session_id: Session identifier

    Returns:
        Byte offset (0 if no offset file exists)
    """
    offset_file = _offset_path(session_id)

    if not offset_file.exists():
        return 0

    try:
        with open(offset_file, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    except (ValueError, IOError):
        return 0


def save_offset(session_id: str, offset: int) -> None:
    """
    Save the processed byte offset for a session.

    Args:
        session_id: Session identifier
        offset: Byte offset to save
    """
    _ensure_dir()

    offset_file = _offset_path(session_id)
    with open(offset_file, "w", encoding="utf-8") as f:
        f.write(str(offset))
        f.flush()
        os.fsync(f.fileno())


def has_events(session_id: str) -> bool:
    """Check if a session has any events."""
    events_file = _events_path(session_id)
    return events_file.exists() and events_file.stat().st_size > 0
