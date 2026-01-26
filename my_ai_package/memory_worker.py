"""
Memory Worker - Background batch extractor for memory events.

Processes events in batches, extracts facts/tasks/preferences using LLM,
and stores results in memory_store for context injection.
"""

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from my_ai_package import config
from my_ai_package.memory_ingest import get_events, get_offset, save_offset
from my_ai_package.memory_services import MemoryServices
from my_ai_package.memory_logging import log_info, log_error, log_debug, log_warning

# Memory store directory
MEMORY_STORE_DIR = Path.home() / ".myai" / "memory_store"

# Extraction prompt template
EXTRACTION_PROMPT = '''You are a memory extraction assistant. Analyze the following conversation events and extract structured information.

EVENTS:
{events_text}

Extract and return a JSON object with these fields:
{{
  "episode_summary": "Brief 1-2 sentence summary of what happened in this conversation segment",
  "facts": ["List of factual statements learned (user preferences, project details, etc.)"],
  "open_tasks": ["List of tasks or action items that remain incomplete"],
  "preferences": {{"key": "value"}}  // User preferences like style, tools, etc.
}}

Rules:
- Only include facts explicitly stated or strongly implied
- Keep facts concise and actionable
- Only list tasks that are clearly unfinished
- For preferences, use snake_case keys

Return ONLY the JSON object, no other text.'''


def _ensure_dir():
    """Create memory store directory if it doesn't exist."""
    MEMORY_STORE_DIR.mkdir(parents=True, exist_ok=True)


def _store_path(session_id: str) -> Path:
    """Get path to memory store file for a session."""
    return MEMORY_STORE_DIR / f"{session_id}.json"


def _load_store(session_id: str) -> dict[str, Any]:
    """Load existing memory store or create empty one."""
    store_file = _store_path(session_id)

    if store_file.exists():
        try:
            with open(store_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    # Return empty store structure
    return {
        "session_id": session_id,
        "updated_at": None,
        "last_event_offset": 0,
        "episode_summary": "",
        "facts": [],
        "open_tasks": [],
        "preferences": {},
    }


def _save_store(session_id: str, store: dict[str, Any]) -> None:
    """Save memory store to disk."""
    _ensure_dir()
    store_file = _store_path(session_id)

    store["updated_at"] = datetime.now(timezone.utc).isoformat()

    with open(store_file, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


def _format_events(events: list[dict]) -> str:
    """Format events into text for LLM processing."""
    lines = []
    for i, event in enumerate(events):
        event_type = event.get("type", "unknown")
        payload = event.get("payload", {})
        ts = event.get("ts", "")

        if event_type == "chat.user":
            lines.append(f"[{i}] USER: {payload.get('content', '')}")
        elif event_type == "chat.assistant":
            lines.append(f"[{i}] ASSISTANT: {payload.get('content', '')}")
        elif event_type == "tool.call":
            name = payload.get("name", "unknown")
            args = payload.get("args", {})
            lines.append(f"[{i}] TOOL_CALL: {name}({json.dumps(args)})")
        elif event_type == "tool.result":
            name = payload.get("name", "unknown")
            result = payload.get("result", "")[:200]  # Truncate long results
            lines.append(f"[{i}] TOOL_RESULT: {name} -> {result}")
        elif event_type == "tool.error":
            name = payload.get("name", "unknown")
            error = payload.get("error", "")
            lines.append(f"[{i}] TOOL_ERROR: {name} -> {error}")

    return "\n".join(lines)


def _parse_extraction(response: str) -> Optional[dict]:
    """Parse LLM extraction response as JSON."""
    try:
        # Find JSON in response
        response = response.strip()

        # Remove markdown code fences if present
        if response.startswith("```"):
            lines = response.split("\n")
            lines = lines[1:]  # Remove first line
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            response = "\n".join(lines).strip()

        # Find first { and last }
        start = response.find("{")
        end = response.rfind("}")

        if start == -1 or end == -1:
            return None

        json_str = response[start:end + 1]
        return json.loads(json_str)

    except (json.JSONDecodeError, ValueError):
        return None


def _merge_extraction(store: dict, extraction: dict) -> None:
    """Merge extraction results into existing store."""
    # Update episode summary (replace with latest)
    if extraction.get("episode_summary"):
        if store["episode_summary"]:
            store["episode_summary"] = f"{store['episode_summary']} {extraction['episode_summary']}"
        else:
            store["episode_summary"] = extraction["episode_summary"]

    # Append new facts (deduplicate)
    new_facts = extraction.get("facts", [])
    existing_facts = {f.get("text") if isinstance(f, dict) else f for f in store["facts"]}

    for fact in new_facts:
        fact_text = fact if isinstance(fact, str) else str(fact)
        if fact_text and fact_text not in existing_facts:
            store["facts"].append({
                "text": fact_text,
                "ts": datetime.now(timezone.utc).isoformat(),
            })

    # Append new tasks (deduplicate)
    new_tasks = extraction.get("open_tasks", [])
    existing_tasks = {t.get("text") if isinstance(t, dict) else t for t in store["open_tasks"]}

    for task in new_tasks:
        task_text = task if isinstance(task, str) else str(task)
        if task_text and task_text not in existing_tasks:
            store["open_tasks"].append({
                "text": task_text,
                "status": "open",
                "created_at": datetime.now(timezone.utc).isoformat(),
            })

    # Merge preferences (update/add)
    new_prefs = extraction.get("preferences", {})
    if isinstance(new_prefs, dict):
        store["preferences"].update(new_prefs)


def process_session(session_id: str, services: MemoryServices) -> Optional[dict]:
    """
    Process a session's events and extract memory.

    Args:
        session_id: Session identifier
        services: MemoryServices instance for LLM access

    Returns:
        Updated memory store dict, or None if nothing to process/failure
    """
    # Get current offset
    offset = get_offset(session_id)

    # Read new events (limited to batch size, offset tracks actual position)
    events, new_offset = get_events(
        session_id,
        since_offset=offset,
        max_events=config.MEMORY_WORKER_BATCH_SIZE
    )

    # Nothing to process
    if not events:
        log_debug(f"[{session_id[:8]}] No new events to process")
        return None

    log_info(f"[{session_id[:8]}] Processing {len(events)} events (offset {offset} -> {new_offset})")

    # Format events for LLM
    events_text = _format_events(events)

    # Build extraction prompt
    prompt = EXTRACTION_PROMPT.format(events_text=events_text)

    # Call LLM
    start_time = time.time()
    response = services.llm_complete(prompt)
    elapsed = time.time() - start_time

    if not response:
        log_error(f"[{session_id[:8]}] LLM extraction failed ({elapsed:.1f}s)")
        return None

    log_info(f"[{session_id[:8]}] LLM extraction success ({elapsed:.1f}s)")

    # Parse extraction
    extraction = _parse_extraction(response)

    if not extraction:
        log_error(f"[{session_id[:8]}] Failed to parse LLM response")
        log_debug(f"[{session_id[:8]}] Raw response: {response[:200]}...")
        return None

    # Load existing store
    store = _load_store(session_id)

    # Merge extraction
    _merge_extraction(store, extraction)

    facts_count = len(extraction.get("facts", []))
    tasks_count = len(extraction.get("open_tasks", []))
    log_info(f"[{session_id[:8]}] Extracted {facts_count} facts, {tasks_count} tasks")

    # Update offset in store
    store["last_event_offset"] = new_offset

    # Save store
    _save_store(session_id, store)

    # Save offset separately (for ingest tracking)
    save_offset(session_id, new_offset)

    # Merge into user-level memory (cross-session persistence)
    try:
        from my_ai_package.memory_user import merge_session_memory
        merge_session_memory(
            session_id,
            store,
            pod_id=services.pod_id,
            api_key=services.api_key,
        )
        log_info(f"[{session_id[:8]}] Merged into user memory")
    except Exception as e:
        log_error(f"[{session_id[:8]}] User memory merge failed: {e}")

    return store


class MemoryWorker:
    """
    Background worker for processing memory events.

    Runs in a daemon thread and processes events at regular intervals.
    """

    def __init__(self, session_id: str, services: MemoryServices):
        """
        Initialize memory worker.

        Args:
            session_id: Session identifier to process
            services: MemoryServices instance for LLM access
        """
        self.session_id = session_id
        self.services = services
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def process_batch(self) -> bool:
        """
        Process one batch of events (one-shot).

        Returns:
            True if events were processed, False otherwise
        """
        result = process_session(self.session_id, self.services)
        return result is not None

    def start_background(self) -> None:
        """Start the background worker thread."""
        if self._thread is not None and self._thread.is_alive():
            return  # Already running

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name=f"memory-worker-{self.session_id[:8]}",
        )
        self._thread.start()
        log_info(f"[{self.session_id[:8]}] Worker started (interval={config.MEMORY_WORKER_INTERVAL_SEC}s)")

    def stop(self) -> None:
        """Stop the background worker thread."""
        self._stop_event.set()

        if self._thread is not None:
            # Wait briefly for clean shutdown
            self._thread.join(timeout=2.0)
            self._thread = None
            log_info(f"[{self.session_id[:8]}] Worker stopped")

    def _run_loop(self) -> None:
        """Background worker loop."""
        while not self._stop_event.is_set():
            try:
                self.process_batch()
            except Exception as e:
                log_error(f"[{self.session_id[:8]}] Worker error: {e}")

            # Wait for interval or stop signal
            self._stop_event.wait(timeout=config.MEMORY_WORKER_INTERVAL_SEC)
