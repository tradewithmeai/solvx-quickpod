"""
Judge Service - Stable-Code-3B as strict evaluator.

FROZEN CONTRACT:
- Stable-Code is ONLY used as a judge
- Never for routing, classification, or tool selection
- Judge invocation is EXPLICIT ONLY (via /judge or programmatic call)
- No retries, no fallbacks, no schema inference
"""

from dataclasses import dataclass
from typing import Optional
import json

import httpx

from my_ai_package.memory_logging import log_info, log_error


# Judge configuration - FROZEN
JUDGE_MODEL = "/workspace/models/stable-code-instruct-3b-awq"
JUDGE_PORT = 8001
JUDGE_MAX_TOKENS = 180
JUDGE_TEMPERATURE = 0

# Judge prompt template - FROZEN (CRITERIA and OUTPUT on their own lines)
JUDGE_PROMPT = '''You are a strict evaluator.

Evaluate the OUTPUT according to the CRITERIA.

Return your final verdict in a JSON object matching the schema below.
The JSON MUST appear between the markers <JSON> and </JSON>.
Do not include any other JSON objects.

CRITERIA:
{criteria}

OUTPUT:
{output}

SCHEMA:
{{
"pass": boolean,
"score": number,
"issues": string[],
"summary": string
}}

<JSON>'''


@dataclass
class JudgeVerdict:
    """Result from judge evaluation."""
    pass_: bool          # "pass" is reserved keyword
    score: float
    issues: list[str]
    summary: str
    raw_response: str    # Full response for debugging


def judge(criteria: str, output: str, pod_id: str, api_key: str) -> Optional[JudgeVerdict]:
    """
    Evaluate output against criteria using Stable-Code judge.

    Judge invocation is EXPLICIT ONLY:
    - via /judge command
    - or explicit programmatic evaluation
    - NEVER auto-routed from normal chat input

    Args:
        criteria: Evaluation criteria
        output: Content to evaluate
        pod_id: RunPod pod ID
        api_key: vLLM API key

    Returns:
        JudgeVerdict or None on failure (no retries, no fallbacks)
    """
    prompt = JUDGE_PROMPT.format(criteria=criteria, output=output)

    url = f"https://{pod_id}-{JUDGE_PORT}.proxy.runpod.net/v1/completions"

    # Log input lengths only (not full content)
    log_info(f"Judge: criteria={len(criteria)} chars, output={len(output)} chars")

    try:
        response = httpx.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": JUDGE_MODEL,  # MUST be explicit, do not rely on server defaults
                "prompt": prompt,
                "max_tokens": JUDGE_MAX_TOKENS,
                "temperature": JUDGE_TEMPERATURE,
                "stop": ["</JSON>"],
            },
            timeout=30,
        )

        if response.status_code != 200:
            log_error(f"Judge HTTP error: {response.status_code}")
            return None

        data = response.json()
        raw = data["choices"][0]["text"]

        return parse_verdict(raw)

    except Exception as e:
        log_error(f"Judge call failed: {e}")
        return None


def parse_verdict(raw: str) -> Optional[JudgeVerdict]:
    """
    JUDGE PARSING (STRICT, FAIL-HARD):
    - Requires JSON between <JSON> and </JSON> markers
    - Returns None if markers missing (logged as error)
    - NO fallback parsing, NO schema inference
    - Contrast with CRITIC parsing which is PERMISSIVE and fail-soft

    STRICT RULES (judge contract is frozen):
    - JSON must be between <JSON> and </JSON> markers
    - If <JSON> is NOT found: log error, return None, NO recovery
    - Ignore all other text
    - No retries
    - Any deviation is a FAILURE, not a soft error
    """
    # Find <JSON> marker - REQUIRED
    start = raw.find("<JSON>")
    if start == -1:
        log_error("Judge parse failed: missing <JSON> marker")
        return None  # NO FALLBACK - contract requires markers

    start += len("<JSON>")

    # Find </JSON> marker (may not exist if stop token worked - that's OK)
    end = raw.find("</JSON>", start)
    if end == -1:
        end = len(raw)

    json_str = raw[start:end].strip()

    try:
        obj = json.loads(json_str)
        verdict = JudgeVerdict(
            pass_=obj.get("pass", False),
            score=float(obj.get("score", 0)),
            issues=obj.get("issues", []),
            summary=obj.get("summary", ""),
            raw_response=raw,
        )
        # Log verdict only (not full content)
        log_info(f"Judge verdict: pass={verdict.pass_}, score={verdict.score}")
        return verdict
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        log_error(f"Judge parse failed: invalid JSON - {e}")
        return None  # NO FALLBACK
