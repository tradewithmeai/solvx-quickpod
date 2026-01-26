import json
import os
from pathlib import Path

STATE_FILE = Path.home() / ".myai" / "pod.json"

# Don't crash if missing - let ai.py handle reconnection flow
POD_ID = None
MODEL = None
LLM_PORT = 8000
LLM_BASE_URL = None

# Model path
PRIMARY_MODEL = "/workspace/models/mistral-7b-instruct-awq"

if STATE_FILE.exists():
    try:
        with open(STATE_FILE) as f:
            _state = json.load(f)
            POD_ID = _state.get("pod_id")
            MODEL = _state.get("model", PRIMARY_MODEL)
        if POD_ID:
            LLM_BASE_URL = f"https://{POD_ID}-{LLM_PORT}.proxy.runpod.net"
    except (json.JSONDecodeError, KeyError):
        pass  # Invalid state file, treat as no pod

API_KEY = os.getenv("VLLM_API_KEY")
if not API_KEY:
    raise RuntimeError("VLLM_API_KEY is not set")

# Default system prompt
SYSTEM_PROMPT = "You are a helpful assistant."
