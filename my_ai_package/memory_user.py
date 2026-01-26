"""
User Memory - Cross-session persistent memory with semantic retrieval.

Stores facts, preferences, and tasks at the user level so they persist
across chat sessions. Uses pod embeddings service (port 8008) for
semantic search of relevant facts.
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx

from my_ai_package.memory_logging import log_info, log_error, log_debug, log_warning

# User memory file
MEMORY_STORE_DIR = Path.home() / ".myai" / "memory_store"
USER_MEMORY_FILE = MEMORY_STORE_DIR / "user.json"

# Limits
MAX_USER_FACTS = 100
MAX_USER_TASKS = 50


def _ensure_dir():
    """Create memory store directory if it doesn't exist."""
    MEMORY_STORE_DIR.mkdir(parents=True, exist_ok=True)


def _empty_user_memory() -> dict[str, Any]:
    """Return empty user memory structure."""
    return {
        "updated_at": None,
        "facts": [],           # List of {text, ts, source_session, embedding}
        "open_tasks": [],
        "completed_tasks": [],
        "preferences": {},
        "session_summaries": [],
    }


def load_user_memory() -> dict[str, Any]:
    """Load user memory from disk."""
    if not USER_MEMORY_FILE.exists():
        return _empty_user_memory()

    try:
        with open(USER_MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return _empty_user_memory()


def save_user_memory(memory: dict[str, Any]) -> None:
    """Save user memory to disk."""
    _ensure_dir()
    memory["updated_at"] = datetime.now(timezone.utc).isoformat()

    with open(USER_MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


# =============================================================================
# Embeddings Service (port 8008)
# =============================================================================

def get_embedding(text: str, pod_id: str, api_key: str) -> Optional[list[float]]:
    """
    Get embedding vector from pod embeddings service.

    Args:
        text: Text to embed
        pod_id: RunPod pod ID
        api_key: API key

    Returns:
        Embedding vector or None on failure
    """
    url = f"https://{pod_id}-8008.proxy.runpod.net/embed"

    start_time = time.time()
    try:
        response = httpx.post(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            json={"text": text},
            timeout=30,
        )
        elapsed = time.time() - start_time
        if response.status_code == 200:
            data = response.json()
            log_debug(f"Embedding success ({elapsed:.2f}s)")
            return data.get("embedding")
        else:
            log_warning(f"Embedding HTTP {response.status_code} ({elapsed:.2f}s)")
    except Exception as e:
        elapsed = time.time() - start_time
        log_error(f"Embedding failed ({elapsed:.2f}s): {e}")

    return None


def get_embeddings_batch(texts: list[str], pod_id: str, api_key: str) -> list[Optional[list[float]]]:
    """
    Get embeddings for multiple texts.

    Args:
        texts: List of texts to embed
        pod_id: RunPod pod ID
        api_key: API key

    Returns:
        List of embedding vectors (None for failures)
    """
    url = f"https://{pod_id}-8008.proxy.runpod.net/embed_batch"

    log_info(f"Embedding batch of {len(texts)} texts")
    start_time = time.time()
    try:
        response = httpx.post(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            json={"texts": texts},
            timeout=60,
        )
        elapsed = time.time() - start_time
        if response.status_code == 200:
            data = response.json()
            log_info(f"Batch embedding success ({elapsed:.2f}s)")
            return data.get("embeddings", [None] * len(texts))
        else:
            log_warning(f"Batch embedding HTTP {response.status_code}, falling back to individual")
    except Exception as e:
        log_warning(f"Batch embedding failed: {e}, falling back to individual")

    # Fallback: try individual embeddings
    return [get_embedding(t, pod_id, api_key) for t in texts]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


# =============================================================================
# Memory Operations
# =============================================================================

def merge_session_memory(
    session_id: str,
    session_store: dict[str, Any],
    pod_id: str = None,
    api_key: str = None,
) -> dict[str, Any]:
    """
    Merge session memory into user memory with optional embeddings.

    Args:
        session_id: Session identifier
        session_store: Session memory store dict
        pod_id: RunPod pod ID (for embeddings)
        api_key: API key (for embeddings)

    Returns:
        Updated user memory dict
    """
    log_info(f"[{session_id[:8]}] Merging session into user memory")
    user_memory = load_user_memory()
    existing_fact_texts = {f.get("text") for f in user_memory["facts"]}

    # Collect new facts
    new_facts = []
    for fact in session_store.get("facts", []):
        fact_text = fact.get("text") if isinstance(fact, dict) else str(fact)
        if fact_text and fact_text not in existing_fact_texts:
            new_facts.append({
                "text": fact_text,
                "ts": fact.get("ts") if isinstance(fact, dict) else datetime.now(timezone.utc).isoformat(),
                "source_session": session_id,
                "embedding": None,
            })
            existing_fact_texts.add(fact_text)

    # Get embeddings for new facts if pod available
    if new_facts and pod_id and api_key:
        texts = [f["text"] for f in new_facts]
        embeddings = get_embeddings_batch(texts, pod_id, api_key)
        for fact, emb in zip(new_facts, embeddings):
            fact["embedding"] = emb

    # Add new facts
    user_memory["facts"].extend(new_facts)

    # Trim to max
    if len(user_memory["facts"]) > MAX_USER_FACTS:
        user_memory["facts"] = user_memory["facts"][-MAX_USER_FACTS:]

    # Merge tasks
    existing_task_texts = {t.get("text") for t in user_memory["open_tasks"]}
    for task in session_store.get("open_tasks", []):
        task_text = task.get("text") if isinstance(task, dict) else str(task)
        if task_text and task_text not in existing_task_texts:
            user_memory["open_tasks"].append({
                "text": task_text,
                "status": "open",
                "created_at": task.get("created_at") if isinstance(task, dict) else datetime.now(timezone.utc).isoformat(),
                "source_session": session_id,
            })
            existing_task_texts.add(task_text)

    if len(user_memory["open_tasks"]) > MAX_USER_TASKS:
        user_memory["open_tasks"] = user_memory["open_tasks"][-MAX_USER_TASKS:]

    # Merge preferences
    session_prefs = session_store.get("preferences", {})
    if isinstance(session_prefs, dict):
        user_memory["preferences"].update(session_prefs)

    # Add session summary
    summary = session_store.get("episode_summary", "")
    if summary:
        user_memory["session_summaries"].append({
            "session_id": session_id,
            "summary": summary[:200],
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        if len(user_memory["session_summaries"]) > 20:
            user_memory["session_summaries"] = user_memory["session_summaries"][-20:]

    save_user_memory(user_memory)
    log_info(f"[{session_id[:8]}] User memory now has {len(user_memory['facts'])} facts, {len(user_memory['open_tasks'])} tasks")
    return user_memory


def get_relevant_facts(
    query: str,
    pod_id: str,
    api_key: str,
    max_facts: int = 8,
    min_similarity: float = 0.3,
) -> list[dict]:
    """
    Get semantically relevant facts using embeddings.

    Args:
        query: Query text to find relevant facts for
        pod_id: RunPod pod ID
        api_key: API key
        max_facts: Maximum facts to return
        min_similarity: Minimum cosine similarity threshold

    Returns:
        List of relevant fact dicts sorted by relevance
    """
    user_memory = load_user_memory()
    facts = user_memory.get("facts", [])

    if not facts:
        return []

    # Get query embedding
    query_embedding = get_embedding(query, pod_id, api_key)

    if not query_embedding:
        # Fallback: return recent facts
        return facts[-max_facts:]

    # Score facts by similarity
    scored_facts = []
    for fact in facts:
        fact_embedding = fact.get("embedding")
        if fact_embedding:
            similarity = cosine_similarity(query_embedding, fact_embedding)
            if similarity >= min_similarity:
                scored_facts.append((similarity, fact))
        else:
            # No embedding - include with low score
            scored_facts.append((0.1, fact))

    # Sort by similarity descending
    scored_facts.sort(key=lambda x: x[0], reverse=True)

    # Return top facts
    return [f for _, f in scored_facts[:max_facts]]


def get_open_tasks(max_tasks: int = 5) -> list[dict]:
    """Get open tasks from user memory."""
    user_memory = load_user_memory()
    tasks = user_memory.get("open_tasks", [])
    open_tasks = [t for t in tasks if t.get("status") == "open"]
    return open_tasks[-max_tasks:]


def get_preferences() -> dict[str, Any]:
    """Get user preferences."""
    user_memory = load_user_memory()
    return user_memory.get("preferences", {})


def get_recent_summaries(max_summaries: int = 3) -> list[dict]:
    """Get recent session summaries."""
    user_memory = load_user_memory()
    summaries = user_memory.get("session_summaries", [])
    return summaries[-max_summaries:]


def has_user_memory() -> bool:
    """Check if user has any stored memory."""
    if not USER_MEMORY_FILE.exists():
        return False
    user_memory = load_user_memory()
    return bool(
        user_memory.get("facts")
        or user_memory.get("open_tasks")
        or user_memory.get("preferences")
    )
