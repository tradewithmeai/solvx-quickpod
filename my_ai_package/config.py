import json
import os
from pathlib import Path

STATE_FILE = Path.home() / ".myai" / "pod.json"

# Don't crash if missing - let ai.py handle reconnection flow
POD_ID = None
MODEL = None
LLM_PORT = 8000  # Active LLM port (8000=primary, 8001=small)
LLM_BASE_URL = None

# Model paths
PRIMARY_MODEL = "/workspace/models/mistral-7b-instruct-awq"
SMALL_MODEL = "/workspace/models/stable-code-instruct-3b-awq"

if STATE_FILE.exists():
    try:
        with open(STATE_FILE) as f:
            _state = json.load(f)
            POD_ID = _state.get("pod_id")
            _model = _state.get("model", PRIMARY_MODEL)
            # Handle special mode identifiers
            if _model in ("dual-llm", "full-stack"):
                MODEL = PRIMARY_MODEL  # Primary model for chat
            elif _model in ("shell-only", "agentic"):
                MODEL = None
            else:
                MODEL = _model
        if POD_ID:
            LLM_BASE_URL = f"https://{POD_ID}-{LLM_PORT}.proxy.runpod.net"
    except (json.JSONDecodeError, KeyError):
        pass  # Invalid state file, treat as no pod


def set_active_llm(port: int, model: str):
    """
    Set the active LLM port and model. Updates LLM_BASE_URL accordingly.

    Args:
        port: 8000 for primary (Mistral), 8001 for small (Stable-Code)
        model: Model path string
    """
    global LLM_PORT, LLM_BASE_URL, MODEL
    LLM_PORT = port
    MODEL = model
    if POD_ID:
        LLM_BASE_URL = f"https://{POD_ID}-{LLM_PORT}.proxy.runpod.net"

API_KEY = os.getenv("VLLM_API_KEY")
if not API_KEY:
    raise RuntimeError("VLLM_API_KEY is not set")

# System prompt presets
SYSTEM_PRESETS = {
    "helpful": "You are a helpful assistant.",
    "critic": "You are a strong critic. Analyze ideas rigorously, identify weaknesses, and provide honest, constructive feedback. Don't sugarcoat issues.",
    "engineer": "You are a senior software engineer. Provide practical coding advice, identify bugs, suggest improvements, and explain technical concepts clearly.",
    "creative": "You are a creative collaborator. Think outside the box, suggest unconventional ideas, and encourage experimentation. Be imaginative and inspiring.",
}

# Temperature presets
TEMPERATURE_PRESETS = {
    "precise": 0.1,
    "balanced": 0.5,
    "creative": 0.9,
}

# Default for backwards compatibility
SYSTEM_PROMPT = SYSTEM_PRESETS["helpful"]

# Memory style options (conversation context)
MEMORY_STYLES = {
    "sliding": {
        "description": "Keep last 10 turns (current behavior)",
        "max_turns": 10,
    },
    "full": {
        "description": "Keep entire conversation history",
        "max_turns": None,
    },
}

# Memory ingest system
MEMORY_ENABLED = True
MEMORY_WORKER_BATCH_SIZE = 10  # Events per batch
MEMORY_WORKER_INTERVAL_SEC = 60  # Seconds between worker runs

# Dual LLM configuration (Full Stack mode)
PRIMARY_LLM_PORT = 8000  # Mistral-7B
SMALL_LLM_PORT = 8001  # Stable-Code-3B
PREFER_SMALL_LLM = False  # Use primary LLM (8000) by default
