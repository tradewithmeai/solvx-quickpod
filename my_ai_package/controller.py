"""
Agent Controller Module (Chunk 2)

Agent controller that enforces JSON-structured responses and executes tools.
Supports both /v1/completions and /v1/chat/completions endpoints.
"""

import json
from dataclasses import dataclass
from typing import Union

import httpx

from my_ai_package.tools.registry import registry
from my_ai_package.tools.websearch import websearch

# Register tools on module load
registry.register("websearch", websearch)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FinalResult:
    """Returned when the model provides a final answer."""
    content: str


@dataclass
class ToolRequest:
    """Returned when the model requests a tool call."""
    name: str
    args: dict


# =============================================================================
# System Contract
# =============================================================================

SYSTEM_CONTRACT = '''You are an AI assistant that MUST respond with exactly ONE JSON object.

You MUST output ONE of these two JSON structures (no arrays, no strings, just one object):

1. For final answers:
{"type": "final", "content": "your answer here"}

2. For tool requests:
{"type": "tool", "name": "tool_name", "args": {"arg1": "value1"}}

Available tools:
- websearch: Search the web. Args: query (required string), k (optional int, default 5)

WHEN TO USE TOOLS (IMPORTANT):
- If the user asks you to "use websearch" or "search for", you MUST call the websearch tool first
- For current events, recent statistics, or real-time data, use websearch
- Do NOT answer from memory when the user explicitly requests a tool
- After using websearch, include the source URL in your final answer

Rules:
- Output ONLY the JSON object
- No explanations before or after
- No markdown code fences
- No arrays or multiple objects
- ONE JSON OBJECT ONLY'''


# =============================================================================
# Helper Functions
# =============================================================================

def extract_json(text: str) -> dict:
    """Extract first complete JSON object from text (balanced braces)."""
    text = text.strip()

    # Remove markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```)
        lines = lines[1:]
        # Remove last line if it's ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Find first {
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in: {text[:100]}...")

    # Find matching } by counting balanced braces
    depth = 0
    in_string = False
    escape_next = False
    end = -1

    for i in range(start, len(text)):
        char = text[i]

        if escape_next:
            escape_next = False
            continue

        if char == '\\' and in_string:
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                end = i
                break

    if end == -1:
        raise ValueError(f"No complete JSON object found in: {text[:100]}...")

    json_str = text[start:end+1]
    return json.loads(json_str)


def validate_response(obj: dict) -> tuple[bool, str]:
    """Validate response matches contract. Returns (is_valid, error_msg)."""
    if not isinstance(obj, dict):
        return False, "Response must be a JSON object"
    if "type" not in obj:
        return False, "Missing 'type' field"
    if obj["type"] == "final":
        if "content" not in obj or not isinstance(obj["content"], str):
            return False, "Final response must have 'content' string"
        return True, ""
    if obj["type"] == "tool":
        if "name" not in obj or not isinstance(obj["name"], str):
            return False, "Tool request must have 'name' string"
        if "args" not in obj or not isinstance(obj["args"], dict):
            return False, "Tool request must have 'args' object"
        return True, ""
    return False, f"Invalid type '{obj['type']}'. Must be 'final' or 'tool'"


def validate_tool_args(name: str, args: dict) -> tuple[bool, str]:
    """Validate tool-specific args. Returns (is_valid, error_msg)."""
    if name == "websearch":
        if "query" not in args or not isinstance(args["query"], str):
            return False, "websearch requires 'query' string argument"
        if "k" in args and not isinstance(args["k"], int):
            return False, "websearch 'k' argument must be an integer"
        return True, ""
    return False, f"Unknown tool: {name}"


def repair_prompt(bad_output: str, error: str) -> str:
    """Build repair prompt that re-includes contract and demands compliance."""
    return f'''{SYSTEM_CONTRACT}

Your previous response was invalid.
Error: {error}

Your output was:
{bad_output}

You MUST try again. Output ONE JSON OBJECT ONLY.
No text before or after. Just the JSON object.'''


# =============================================================================
# Controller Class
# =============================================================================

class Controller:
    """
    Agent controller that enforces JSON-structured responses.

    Supports both /v1/completions and /v1/chat/completions endpoints.
    """

    def __init__(self, base_url: str, api_key: str, model: str, mode: str = "completion"):
        """
        Initialize the controller.

        Args:
            base_url: vLLM server URL (e.g., "http://localhost:8000")
            api_key: API key for authentication
            model: Model path or name
            mode: "completion" for /v1/completions (default, no chat template needed)
                  "chat" for /v1/chat/completions (requires working chat template)
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.mode = mode

    def run(self, user_text: str, max_tool_calls: int = 2, debug: bool = False, session_id: str = None) -> FinalResult:
        """
        Run the agent loop: call model, execute tools if needed, return final result.

        Args:
            user_text: The user's input/query
            max_tool_calls: Maximum tool executions allowed (default 2)
            debug: If True, print tool execution info
            session_id: Session ID for memory event logging (optional)

        Returns:
            FinalResult (always - tool requests are handled internally)

        Raises:
            ValueError: If response is invalid after retry or too many tool calls
            RuntimeError: If API call fails
        """
        conversation = user_text
        tool_calls = 0

        while True:
            # Build prompt and call model
            full_prompt = f"{SYSTEM_CONTRACT}\n\nUSER:\n{conversation}\n\nASSISTANT:\n"

            if self.mode == "chat":
                raw = self._call_chat(conversation)
            else:
                raw = self._call_completion(full_prompt)

            if debug:
                print(f"[DEBUG] Raw model response: {raw[:200]}...")

            # Extract and validate
            error = ""
            try:
                obj = extract_json(raw)
                valid, error = validate_response(obj)
                if not valid:
                    raise ValueError(error)
            except (ValueError, json.JSONDecodeError) as e:
                error = str(e)
                # Retry once with repair prompt
                repair = repair_prompt(raw, error)
                if self.mode == "chat":
                    raw = self._call_chat(repair)
                else:
                    raw = self._call_completion(f"{SYSTEM_CONTRACT}\n\n{repair}\n\nASSISTANT:\n")
                obj = extract_json(raw)
                valid, error = validate_response(obj)
                if not valid:
                    raise ValueError(f"Invalid response after retry: {error}")

            # Handle final result
            if obj["type"] == "final":
                if debug:
                    print(f"[DEBUG] Final result after {tool_calls} tool call(s)")
                return FinalResult(content=obj["content"])

            # Handle tool request
            if obj["type"] == "tool":
                tool_name = obj["name"]
                tool_args = obj["args"]

                # Validate tool args
                valid, error = validate_tool_args(tool_name, tool_args)
                if not valid:
                    raise ValueError(f"Invalid tool args: {error}")

                # Check tool call limit
                tool_calls += 1
                if tool_calls > max_tool_calls:
                    raise ValueError(f"Exceeded max tool calls ({max_tool_calls})")

                # Execute tool
                if debug:
                    print(f"[DEBUG] Tool call #{tool_calls}: {tool_name}({tool_args})")

                # Emit tool.call event before execution
                if session_id:
                    try:
                        from my_ai_package.memory_ingest import emit_event
                        emit_event(session_id, "tool.call", {"name": tool_name, "args": tool_args})
                    except Exception:
                        pass  # Graceful degradation

                try:
                    result = registry.run(tool_name, tool_args)
                    # Emit tool.result event after success
                    if session_id:
                        try:
                            from my_ai_package.memory_ingest import emit_event
                            emit_event(session_id, "tool.result", {"name": tool_name, "result": str(result)[:500], "success": True})
                        except Exception:
                            pass  # Graceful degradation
                except Exception as e:
                    # Emit tool.error event on failure
                    if session_id:
                        try:
                            from my_ai_package.memory_ingest import emit_event
                            emit_event(session_id, "tool.error", {"name": tool_name, "error": str(e)[:500]})
                        except Exception:
                            pass  # Graceful degradation
                    raise  # Re-raise to preserve existing behavior

                if debug:
                    print(f"[DEBUG] Tool result: {str(result)[:100]}...")

                # Append tool result to conversation
                conversation = f"{conversation}\n\nTOOL RESULT ({tool_name}):\n{result}\n\nNow provide your final answer."

    def _build_result(self, obj: dict) -> Union[FinalResult, ToolRequest]:
        """Convert validated dict to result object."""
        if obj["type"] == "final":
            return FinalResult(content=obj["content"])
        else:
            return ToolRequest(name=obj["name"], args=obj["args"])

    def _call_completion(self, prompt: str) -> str:
        """Use /v1/completions endpoint (no chat template needed)."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": 0.1,
            "max_tokens": 200,
            "stream": False,
        }
        response = httpx.post(
            f"{self.base_url}/v1/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=60,
        )
        if response.status_code != 200:
            raise RuntimeError(f"API error {response.status_code}: {response.text}")
        data = response.json()
        return data["choices"][0]["text"]

    def _call_chat(self, user_text: str) -> str:
        """Use /v1/chat/completions endpoint (requires working chat template)."""
        messages = [
            {"role": "system", "content": SYSTEM_CONTRACT},
            {"role": "user", "content": user_text},
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 200,
            "stream": False,
        }
        response = httpx.post(
            f"{self.base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=60,
        )
        if response.status_code != 200:
            raise RuntimeError(f"API error {response.status_code}: {response.text}")
        data = response.json()
        return data["choices"][0]["message"]["content"]
