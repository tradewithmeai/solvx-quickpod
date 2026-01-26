"""
Memory Context - Build context packs for injection into LLM prompts.

Combines session memory and user memory (cross-session) for rich context.
Supports semantic retrieval when query and pod info are provided.
"""

import json
from pathlib import Path
from typing import Optional

# Memory store directory
MEMORY_STORE_DIR = Path.home() / ".myai" / "memory_store"


def _store_path(session_id: str) -> Path:
    """Get path to memory store file for a session."""
    return MEMORY_STORE_DIR / f"{session_id}.json"


def has_memory(session_id: str) -> bool:
    """Check if a session has stored memory."""
    # Check session memory
    store_file = _store_path(session_id)
    if store_file.exists():
        try:
            with open(store_file, "r", encoding="utf-8") as f:
                store = json.load(f)
                if any([
                    store.get("episode_summary"),
                    store.get("facts"),
                    store.get("open_tasks"),
                    store.get("preferences"),
                ]):
                    return True
        except (json.JSONDecodeError, IOError):
            pass

    # Check user memory
    try:
        from my_ai_package.memory_user import has_user_memory
        if has_user_memory():
            return True
    except Exception:
        pass

    return False


def build_context_pack(
    session_id: str,
    max_facts: int = 8,
    max_tasks: int = 5,
    query: str = None,
    pod_id: str = None,
    api_key: str = None,
) -> Optional[str]:
    """
    Build a context pack from stored memory for prompt injection.

    Combines:
    - User memory (cross-session facts, preferences, recent summaries)
    - Session memory (current session facts/tasks)

    Args:
        session_id: Session identifier
        max_facts: Maximum number of facts to include
        max_tasks: Maximum number of tasks to include
        query: Optional query for semantic retrieval
        pod_id: RunPod pod ID (for semantic retrieval)
        api_key: API key (for semantic retrieval)

    Returns:
        Formatted context string, or None if no memory exists
    """
    lines = []
    has_content = False

    # === USER MEMORY (cross-session) ===
    try:
        from my_ai_package.memory_user import (
            get_relevant_facts,
            get_open_tasks,
            get_preferences,
            get_recent_summaries,
            load_user_memory,
        )

        # Get facts (semantic search if query provided, else recent)
        if query and pod_id and api_key:
            user_facts = get_relevant_facts(query, pod_id, api_key, max_facts=max_facts)
        else:
            user_memory = load_user_memory()
            user_facts = user_memory.get("facts", [])[-max_facts:]

        user_tasks = get_open_tasks(max_tasks=max_tasks)
        user_prefs = get_preferences()
        user_summaries = get_recent_summaries(max_summaries=2)

        if user_facts or user_tasks or user_prefs or user_summaries:
            has_content = True
            lines.append("=== USER MEMORY ===")

            # Recent session summaries
            if user_summaries:
                lines.append("Recent Sessions:")
                for s in user_summaries:
                    lines.append(f"- {s.get('summary', '')[:100]}")
                lines.append("")

            # User facts
            if user_facts:
                lines.append("Known Facts:")
                for fact in user_facts:
                    fact_text = fact.get("text") if isinstance(fact, dict) else str(fact)
                    if fact_text:
                        lines.append(f"- {fact_text}")
                lines.append("")

            # User tasks
            if user_tasks:
                lines.append("Open Tasks:")
                for task in user_tasks:
                    task_text = task.get("text", "")
                    if task_text:
                        lines.append(f"- [ ] {task_text}")
                lines.append("")

            # User preferences
            if user_prefs:
                pref_items = [f"{k}: {v}" for k, v in user_prefs.items()]
                lines.append(f"Preferences: {', '.join(pref_items)}")
                lines.append("")

    except Exception:
        pass  # User memory unavailable

    # === SESSION MEMORY (current session) ===
    store_file = _store_path(session_id)
    if store_file.exists():
        try:
            with open(store_file, "r", encoding="utf-8") as f:
                store = json.load(f)

            summary = store.get("episode_summary", "")
            facts = store.get("facts", [])
            tasks = store.get("open_tasks", [])

            if summary or facts or tasks:
                has_content = True
                lines.append("=== CURRENT SESSION ===")

                if summary:
                    lines.append(f"Context: {summary}")
                    lines.append("")

                # Session-specific facts (not already in user memory)
                if facts:
                    lines.append("Session Facts:")
                    for fact in facts[-5:]:  # Last 5 session facts
                        fact_text = fact.get("text") if isinstance(fact, dict) else str(fact)
                        if fact_text:
                            lines.append(f"- {fact_text}")
                    lines.append("")

        except (json.JSONDecodeError, IOError):
            pass

    if not has_content:
        return None

    lines.append("=======================")
    return "\n".join(lines)


def build_query_context(
    query: str,
    pod_id: str,
    api_key: str,
    max_facts: int = 5,
) -> Optional[str]:
    """
    Build query-specific context using semantic retrieval.

    This is meant to be called per-query to augment with relevant facts.

    Args:
        query: The user's query
        pod_id: RunPod pod ID
        api_key: API key
        max_facts: Maximum facts to include

    Returns:
        Formatted context string, or None if no relevant facts
    """
    try:
        from my_ai_package.memory_user import get_relevant_facts

        facts = get_relevant_facts(query, pod_id, api_key, max_facts=max_facts)

        if not facts:
            return None

        lines = ["[Relevant context from memory:]"]
        for fact in facts:
            fact_text = fact.get("text") if isinstance(fact, dict) else str(fact)
            if fact_text:
                lines.append(f"- {fact_text}")

        return "\n".join(lines)

    except Exception:
        return None
