"""
Orchestrator - Code-only tool routing with LLM critic validation.

ROLE DEFINITIONS (LOCKED):

ROUTER (Code-Only):
- Authority: FULL - makes all routing decisions
- Implementation: Deterministic keyword/pattern matching
- Output: action = "respond" or "tool"
- LLM involvement: NONE

CRITIC (SMALL LLM - Advisory Only):
- Authority: NONE - cannot make or change decisions
- Role: Bounded advisory validator of code-made decisions
- Output: { "approved": true | false } only
- Fallback: At most ONE code-verified fallback
- On invalid output: Ignored entirely

The critic is NOT a router. The critic is a bounded validator.

PRE-RAG STABILITY (January 2025):
- Router + critic + judge are considered stable
- Next phase introduces embeddings, reranking, and RAG
- RAG will CONSUME critic/judge outputs but NOT alter authority rules
- Router remains code-only and authoritative
"""

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from my_ai_package.memory_services import MemoryServices
from my_ai_package.tools.registry import ToolRegistry
from my_ai_package.memory_logging import log_info, log_error, log_debug


# =============================================================================
# Training Data Storage
# =============================================================================

TRAINING_DATA_DIR = Path.home() / ".myai"
CRITIC_TRAINING_FILE = TRAINING_DATA_DIR / "critic_training.jsonl"


# =============================================================================
# Code-Only Tool Router (AUTHORITATIVE)
# =============================================================================

# Keywords that trigger websearch tool (case insensitive)
TOOL_KEYWORDS = [
    r"\bsearch\b",
    r"\blook\s+up\b",
    r"\bfind\b.*\b(online|web|internet)\b",
    r"\bcurrent\b",
    r"\brecent\b",
    r"\blatest\b",
    r"\btoday\b",
    r"\bnews\b",
    r"\bweather\b",
    r"\bstock\s*(price)?\b",
    r"\bprice\s+of\b",
]

# Patterns that always route to direct response (override tool keywords)
RESPOND_PATTERNS = [
    r"^(hi|hello|hey|greetings)\b",
    r"^thanks?\b",
    r"^thank\s+you\b",
    r"\bhelp\b.*\bwith\b",
    r"^(what|how|why|when|where|who)\s+(is|are|do|does|can|should|would)\b",
    r"^explain\b",
    r"^tell\s+me\s+about\b",
]


# =============================================================================
# Pattern Matching Helpers (Single Source of Truth)
# =============================================================================

def _matches_tool_patterns(user_input: str) -> bool:
    """Check if input matches any TOOL_KEYWORDS pattern."""
    text = user_input.strip()
    for pattern in TOOL_KEYWORDS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def _matches_respond_patterns(user_input: str) -> bool:
    """Check if input matches any RESPOND_PATTERNS pattern."""
    text = user_input.strip()
    for pattern in RESPOND_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def route(user_input: str) -> dict:
    """
    Deterministic code-only tool routing. AUTHORITATIVE.

    RULES:
    - Check RESPOND_PATTERNS first (override tool keywords)
    - Then check TOOL_KEYWORDS
    - Default to respond
    - NO LLM calls, NO semantic reasoning

    Returns:
        dict with keys: action, tool_name, tool_args, confidence, reason
    """
    # Check respond patterns first (override tool keywords)
    if _matches_respond_patterns(user_input):
        return {
            "action": "respond",
            "tool_name": None,
            "tool_args": None,
            "confidence": 1.0,
            "reason": "matched_respond_pattern"
        }

    # Check tool keywords
    if _matches_tool_patterns(user_input):
        return {
            "action": "tool",
            "tool_name": "websearch",
            "tool_args": {"query": user_input},
            "confidence": 1.0,
            "reason": "matched_tool_keyword"
        }

    # Default: respond directly (no tool)
    return {
        "action": "respond",
        "tool_name": None,
        "tool_args": None,
        "confidence": 1.0,
        "reason": "no_tool_keywords_matched"
    }


def is_tool_action_valid(user_input: str) -> bool:
    """Check if tool action would be valid per router rules."""
    return _matches_tool_patterns(user_input)


def is_respond_action_valid(user_input: str) -> bool:
    """Respond is always valid as the default fallback."""
    return True


# =============================================================================
# Routing Critic (ADVISORY ONLY)
# =============================================================================
# CRITIC AUTHORITY BOUNDARY (DO NOT EXPAND)
# - The critic NEVER decides routing
# - The critic NEVER proposes tools
# - The critic may influence behaviour AT MOST ONCE via code-validated fallback
# - Future changes: if you need more authority, use a DIFFERENT component
# =============================================================================

# Action descriptions for the critic prompt
ACTION_DESCRIPTIONS = {
    "respond": "Respond directly to the user without using any tools.",
    "tool": "Use the websearch tool to fetch current information.",
}

# Module-level counters for critic observability (reset per import/session)
_critic_stats = {
    "approved": 0,
    "rejected": 0,
    "invalid": 0,
}

# One-time logging flag for model resolution
_critic_model_logged = False


def _record_critic_outcome(outcome: str):
    """Track critic outcomes for observability. No behaviour change."""
    _critic_stats[outcome] = _critic_stats.get(outcome, 0) + 1
    total = sum(_critic_stats.values())
    if total > 0 and total % 10 == 0:  # Check every 10 calls
        reject_rate = _critic_stats["rejected"] / total
        if reject_rate > 0.3:
            log_info(f"CRITIC_NOISE_WARNING: {reject_rate:.0%} rejection rate ({_critic_stats})")


def get_critic_stats() -> dict:
    """Return current critic outcome stats."""
    return dict(_critic_stats)


@dataclass
class CriticResult:
    """Result from routing critic evaluation."""
    approved: Optional[bool]  # True/False/None (invalid)
    raw_output: str           # For logging


def evaluate_routing_decision(
    user_input: str,
    proposed_action: str,
    services: MemoryServices
) -> CriticResult:
    """
    Ask SMALL LLM to evaluate the code router's decision.
    ADVISORY ONLY - cannot make or change routing decisions.

    Returns CriticResult with:
        approved: True = proceed, False = consider fallback, None = invalid
        raw_output: For logging critic disagreements
    """
    global _critic_model_logged

    action_desc = ACTION_DESCRIPTIONS.get(proposed_action, f"Action: {proposed_action}")

    # NOTE: JSON-first prompt enforcement DEFERRED pending live pod testing.
    # Current prompt structure tested and confirmed working with Stable-Code.
    critic_prompt = (
        "You are a routing critic. You must NOT answer the user's question. "
        "You only evaluate whether the proposed action is appropriate.\n\n"
        "You MUST include a JSON object with a boolean field called approved. "
        "After the JSON, you may add ONE short sentence explaining your reasoning.\n\n"
        f"User input:\n{user_input}\n\n"
        f"Proposed action:\n{action_desc}"
    )

    response = services.llm_complete(
        critic_prompt,
        prefer_small=True,
        max_tokens=80,
        temperature=0
    )

    # One-time log of model resolution for auditability
    if not _critic_model_logged:
        log_info("CRITIC_MODEL: using prefer_small=True (port 8001, Stable-Code)")
        _critic_model_logged = True

    if response is None:
        return CriticResult(approved=None, raw_output="")

    approved = parse_critic_response(response)
    return CriticResult(approved=approved, raw_output=response)


def parse_critic_response(raw: str) -> Optional[bool]:
    """
    CRITIC PARSING (PERMISSIVE, FAIL-SOFT):
    - Accepts JSON anywhere in first 200 chars
    - Returns None on parse failure (ignored, proceed with code decision)
    - Contrast with JUDGE parsing which is STRICT and requires markers
    """
    # Only look in first 200 chars to avoid matching example JSON later in response
    search_window = raw[:200]
    match = re.search(r'\{[^}]*"approved"\s*:\s*(true|false)[^}]*\}', search_window, re.IGNORECASE)
    if not match:
        log_debug("Critic: no valid JSON in first 200 chars")
        return None

    approved_str = match.group(1).lower()
    return approved_str == "true"


def log_critic_disagreement(
    user_input: str,
    proposed_action: str,
    critic_output: str,
    final_action: str,
    fallback_applied: bool = False
):
    """
    Log critic rejection for training data / router refinement.

    Writes structured JSONL to ~/.myai/critic_training.jsonl for:
    - Router pattern analysis
    - Keyword refinement
    - Understanding critic behavior
    """
    # Console log (truncated)
    truncated = critic_output[:100] + "..." if len(critic_output) > 100 else critic_output
    log_info(
        f"CRITIC_REJECT: proposed={proposed_action} final={final_action} "
        f"input={user_input[:50]}... critic={truncated}"
    )

    # Structured training data
    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "user_input": user_input,
        "proposed_action": proposed_action,
        "final_action": final_action,
        "fallback_applied": fallback_applied,
        "critic_output": critic_output,
    }

    # Write to JSONL file
    try:
        TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(CRITIC_TRAINING_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        log_debug(f"Training data written to {CRITIC_TRAINING_FILE}")
    except Exception as e:
        log_error(f"Failed to write training data: {e}")


def get_training_data(limit: int = 100) -> list[dict]:
    """
    Read critic training data for analysis.

    Args:
        limit: Maximum records to return (newest first)

    Returns:
        List of training records
    """
    if not CRITIC_TRAINING_FILE.exists():
        return []

    records = []
    try:
        with open(CRITIC_TRAINING_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except Exception as e:
        log_error(f"Failed to read training data: {e}")
        return []

    # Return newest first
    return records[-limit:][::-1]


def summarize_training_data() -> dict:
    """
    Summarize critic training data for router refinement.

    Returns:
        Summary dict with counts and patterns
    """
    records = get_training_data(limit=1000)

    if not records:
        return {"total": 0, "message": "No training data yet"}

    summary = {
        "total": len(records),
        "fallback_applied": sum(1 for r in records if r.get("fallback_applied")),
        "by_proposed_action": {},
        "recent_inputs": [],
    }

    for r in records:
        action = r.get("proposed_action", "unknown")
        summary["by_proposed_action"][action] = summary["by_proposed_action"].get(action, 0) + 1

    # Last 10 user inputs for pattern analysis
    summary["recent_inputs"] = [r.get("user_input", "")[:80] for r in records[:10]]

    return summary


# =============================================================================
# Orchestrator Class
# =============================================================================

class Orchestrator:
    """
    Code-only tool router with LLM critic validation.

    Routing: Deterministic code (AUTHORITATIVE)
    Critic: SMALL LLM (ADVISORY - validates code decisions)
    Responses: Mistral (PRIMARY LLM)
    """

    def __init__(
        self,
        services: MemoryServices,
        tool_registry: ToolRegistry,
        system_prompt: str = "You are a helpful assistant."
    ):
        """
        Initialize orchestrator.

        Args:
            services: MemoryServices instance for LLM access
            tool_registry: ToolRegistry with registered tools
            system_prompt: System prompt for final response generation
        """
        self.services = services
        self.tools = tool_registry
        self.system_prompt = system_prompt

    def run(self, session_id: str, user_input: str) -> str:
        """
        Execute one orchestrator pass for the user input.

        Flow:
        1. Code router decides action (AUTHORITATIVE)
        2. Critic evaluates decision (ADVISORY)
        3. Apply ONE fallback if critic rejects AND alternative is valid
        4. Execute action and return response

        Args:
            session_id: Current session ID (for logging)
            user_input: The user's message

        Returns:
            Final assistant response string
        """
        log_info(f"Orchestrator: processing input ({len(user_input)} chars)")

        # Step 1: Code-only routing decision (AUTHORITATIVE)
        decision = route(user_input)
        proposed_action = decision["action"]
        log_debug(f"Router: action={proposed_action}, reason={decision['reason']}")

        # Step 2: Critic evaluates the decision (ADVISORY)
        critic_result = evaluate_routing_decision(user_input, proposed_action, self.services)

        if critic_result.approved is False:
            _record_critic_outcome("rejected")
            # AUTHORITY BOUNDARY: At most ONE fallback, ONLY if code validates alternative
            # Do NOT add retries, loops, or additional LLM calls here
            fallback_applied = False

            if proposed_action == "respond" and is_tool_action_valid(user_input):
                decision["action"] = "tool"
                decision["tool_name"] = "websearch"
                decision["tool_args"] = {"query": user_input}
                decision["reason"] = "critic_fallback_to_tool"
                fallback_applied = True
            elif proposed_action == "tool" and is_respond_action_valid(user_input):
                decision["action"] = "respond"
                decision["tool_name"] = None
                decision["tool_args"] = None
                decision["reason"] = "critic_fallback_to_respond"
                fallback_applied = True

            # Always log critic disagreement (learning hook → training data)
            log_critic_disagreement(
                user_input=user_input,
                proposed_action=proposed_action,
                critic_output=critic_result.raw_output,
                final_action=decision["action"],
                fallback_applied=fallback_applied
            )

            if fallback_applied:
                log_debug(f"Critic override: action={decision['action']}")
            else:
                log_debug("Critic rejected but alternative not valid, proceeding with original")

        elif critic_result.approved is True:
            _record_critic_outcome("approved")
            log_debug("Critic: approved")
        else:
            _record_critic_outcome("invalid")
            log_debug("Critic: invalid output, proceeding with code decision")

        # Step 3: Execute based on final decision
        if decision["action"] == "tool":
            return self._handle_tool(user_input, decision)
        else:
            return self._handle_respond(user_input)

    def _handle_tool(self, user_input: str, decision: dict) -> str:
        """
        Execute tool and generate response with result.
        Uses PRIMARY LLM (Mistral) for response generation.
        """
        tool_name = decision["tool_name"]
        tool_args = decision["tool_args"]

        # Validate tool exists
        if not tool_name or not self.tools.has_tool(tool_name):
            log_error(f"Unknown tool requested: {tool_name}")
            return self._handle_respond(user_input)  # Fallback to direct response

        # Execute tool
        log_info(f"Executing tool: {tool_name}({tool_args})")
        try:
            tool_result = self.tools.run(tool_name, tool_args)
        except Exception as e:
            log_error(f"Tool execution failed: {e}")
            return f"I tried to search for that information but encountered an error: {str(e)}"

        # Generate final response with tool result (PRIMARY LLM - Mistral)
        response_prompt = f"""{self.system_prompt}

The user asked: {user_input}

I searched and found this information:
{tool_result}

Based on this information, provide a helpful response to the user. Include relevant details from the search results."""

        response = self.services.llm_complete(response_prompt, prefer_small=False)

        if response is None:
            log_error("Final response LLM call failed")
            return "I found some information but had trouble formulating a response. Please try again."

        return response

    def _handle_respond(self, user_input: str) -> str:
        """
        Generate direct response without tool use.
        Uses PRIMARY LLM (Mistral) for response generation.
        """
        response_prompt = f"""{self.system_prompt}

User: {user_input}

Provide a helpful and friendly response."""

        # Use PRIMARY LLM for final response (Mistral)
        response = self.services.llm_complete(response_prompt, prefer_small=False)

        if response is None:
            log_error("Final response LLM call failed")
            return "I'm having trouble connecting right now. Please try again."

        return response
