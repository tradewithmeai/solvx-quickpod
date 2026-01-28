#!/usr/bin/env python3
"""
SolvX QuickPod - Demo Script

Simulates a chat session showcasing the /json debug mode.
Designed for YouTube demos - highlights response times and raw API exchanges.

Usage:
    python demo/demo_script.py [--fast] [--no-typing]

Options:
    --fast       Reduce delays for quick run-through
    --no-typing  Instant text display (for testing)
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any, Dict, List

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

console = Console()

# =============================================================================
# DEMO CONFIGURATION
# =============================================================================

# Typing speed (characters per second)
TYPING_SPEED = 12  # Slow, readable typing for user input
TYPING_SPEED_FAST = 100  # For --fast mode only

# AI response speed (words per second) - slower than typing
AI_RESPONSE_SPEED = 8  # Deliberate, readable pace

# Delays between actions (seconds)
DELAY_AFTER_USER_INPUT = 1.5  # Pause after user types, before AI responds
DELAY_AFTER_AI_RESPONSE = 4.0  # Long pause to read the response
DELAY_AFTER_JSON_BLOCK = 5.0  # Extra time to examine JSON
DELAY_AFTER_JSON_TOGGLE = 3.0  # Pause after /json mode enabled
DELAY_BETWEEN_EXCHANGES = 2.0  # Pause before next exchange starts

# Simulated response times (seconds) - realistic for cloud GPU
RESPONSE_TIMES = [1.8, 2.1, 1.4, 2.7]

# System prompt (shown in JSON)
SYSTEM_PROMPT = "You are a helpful AI assistant."


# =============================================================================
# DEMO CONVERSATION - META AND FUN
# =============================================================================

DEMO_EXCHANGES = [
    # Exchange 1: Normal chat, set up the reveal
    {
        "user": "Hey! Quick question - when I chat with you, what actually happens behind the scenes?",
        "assistant": "Great question! When you send me a message, it gets wrapped in a JSON request and sent to my API endpoint. The request includes our entire conversation history so I have context. I process it and stream back tokens one by one.\n\nWant to see exactly what that looks like? Type /json to toggle debug mode!",
        "show_json": False,
    },
    # Exchange 2: Enable JSON mode - THE BIG REVEAL
    {
        "command": "/json",
        "output": "JSON display: ON",
    },
    # Exchange 3: First JSON-visible exchange
    {
        "user": "So you can see ALL our previous messages in each request?",
        "assistant": "Yep! Look at the request above - see the \"messages\" array? That's our entire conversation! Every time you chat, I receive ALL previous messages plus your new one. That's how I \"remember\" what we talked about.\n\nIt's not magic - it's just... a really long JSON array that keeps growing!",
        "show_json": True,
    },
    # Exchange 4: Show history building up
    {
        "user": "That's wild. So the conversation history is basically copy-pasted every time?",
        "assistant": "Exactly! And here's the fun part - look at how much bigger this request is compared to the last one. Each exchange adds more messages to the array.\n\nThis is why there's a context limit - eventually the conversation gets too long and we have to trim older messages. SolvX QuickPod keeps the last 10 turns.\n\nPretty cool to see how the sausage is made, right?",
        "show_json": True,
    },
]


# =============================================================================
# DEMO FUNCTIONS
# =============================================================================

def type_text(text: str, speed: int = TYPING_SPEED) -> None:
    """Simulate typing text character by character."""
    for char in text:
        console.print(char, end="", highlight=False)
        time.sleep(1 / speed)
    print()


def stream_response(text: str, words_per_second: int = 8) -> None:
    """Simulate streaming response from AI with word-by-word display."""
    words = text.split(" ")
    for i, word in enumerate(words):
        if i > 0:
            console.print(" ", end="")
        console.print(word, end="", highlight=False, soft_wrap=True)
        time.sleep(1.0 / words_per_second)
    print()


def show_response_time(seconds: float) -> None:
    """Display response time with visual emphasis."""
    # Color based on speed
    if seconds < 2.0:
        color = "green"
        label = "FAST"
    elif seconds < 3.0:
        color = "yellow"
        label = "OK"
    else:
        color = "red"
        label = "SLOW"

    console.print(f"[bold {color} on black] {label}: {seconds:.1f}s [/bold {color} on black]")


def build_request_json(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """Build a realistic API request payload."""
    return {
        "model": "TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
        "messages": messages,
        "temperature": 0.5,
        "max_tokens": 500,
        "stream": True,
    }


def show_json_request(payload: Dict[str, Any]) -> None:
    """Display the JSON request with syntax highlighting."""
    console.print()
    console.print("[bold magenta on black] >>> API REQUEST >>> [/bold magenta on black]")
    console.print()
    json_str = json.dumps(payload, indent=2)
    syntax = Syntax(json_str, "json", theme="monokai", line_numbers=False, background_color="default")
    console.print(Panel(syntax, border_style="magenta", padding=(0, 1)))
    console.print()


def show_json_response(content: str) -> None:
    """Display the JSON response with syntax highlighting."""
    response = {"role": "assistant", "content": content}
    console.print()
    console.print("[bold blue on black] <<< API RESPONSE <<< [/bold blue on black]")
    console.print()
    json_str = json.dumps(response, indent=2)
    syntax = Syntax(json_str, "json", theme="monokai", line_numbers=False, background_color="default")
    console.print(Panel(syntax, border_style="blue", padding=(0, 1)))


def show_welcome() -> None:
    """Display the demo welcome banner."""
    console.print()
    console.print(Panel(
        Text.from_markup(
            "[bold cyan]SolvX QuickPod Demo[/bold cyan]\n\n"
            "[bold]The /json Debug Mode Showcase[/bold]\n\n"
            "Watch how LLM API requests work under the hood.\n"
            "See your conversation history grow in real-time!\n\n"
            "[dim]Press Ctrl+C to exit at any time.[/dim]"
        ),
        title="Behind the Scenes",
        border_style="cyan",
    ))
    console.print()


def run_demo(fast: bool = False, no_typing: bool = False) -> None:
    """Run the demo showcasing /json mode."""
    typing_speed = TYPING_SPEED_FAST if fast else TYPING_SPEED
    response_speed = 100 if fast else AI_RESPONSE_SPEED

    if no_typing:
        typing_speed = 10000
        response_speed = 10000

    show_welcome()
    time.sleep(2.0 if not fast else 0.3)

    # Simulate app header with version
    console.print("[bold]=== SolvX QuickPod v1.0.0 ===[/bold]")
    console.print()
    time.sleep(1.0 if not fast else 0.3)

    # Session info with cost
    console.print("[dim]GPU: RTX 3090 (~$0.44/hour) | Model: Mistral-7B[/dim]")
    console.print("[dim]Temperature: 0.5 | History: Last 10 turns[/dim]")
    console.print()
    console.print("[bold]Chat started. Commands: /json, /stop, /help. Ctrl+C to exit.[/bold]")
    console.print()

    time.sleep(2.0 if not fast else 0.3)

    # Track conversation for JSON display
    messages: List[Dict[str, str]] = []
    response_idx = 0

    for exchange in DEMO_EXCHANGES:
        # Handle /json command
        if "command" in exchange:
            console.print("[bold white on blue] YOU [/bold white on blue] ", end="")
            type_text(exchange["command"], typing_speed)
            time.sleep(1.0 if not fast else 0.1)
            # Highlight the JSON toggle prominently
            console.print()
            console.print("[bold yellow on black] ** JSON DEBUG MODE: ON ** [/bold yellow on black]")
            console.print("[yellow]You will now see raw API requests and responses![/yellow]")
            console.print()
            time.sleep(DELAY_AFTER_JSON_TOGGLE if not fast else 0.3)
            continue

        user_msg = exchange["user"]
        assistant_msg = exchange["assistant"]
        show_json = exchange.get("show_json", False)

        # Build message with system prompt prepended to first user message
        if not messages:
            user_content = f"{SYSTEM_PROMPT}\n\n{user_msg}"
        else:
            user_content = user_msg

        # Visual separator for new exchange
        console.print("[dim]" + "-" * 60 + "[/dim]")
        console.print()
        time.sleep(DELAY_BETWEEN_EXCHANGES if not fast else 0.2)

        # User input with prominent styling
        console.print("[bold white on blue] YOU [/bold white on blue] ", end="")
        type_text(user_msg, typing_speed)  # Show original, not with system prompt

        # Add to messages
        messages.append({"role": "user", "content": user_content})

        # Pause after user input before showing JSON or AI response
        time.sleep(DELAY_AFTER_USER_INPUT if not fast else 0.1)

        # Show JSON request if enabled
        if show_json:
            payload = build_request_json(messages)
            show_json_request(payload)
            # Long pause to let viewer examine the JSON
            time.sleep(DELAY_AFTER_JSON_BLOCK if not fast else 0.5)

        # AI response with prominent styling
        console.print()
        console.print("[bold white on green] AI [/bold white on green] ", end="")
        # Brief pause to simulate AI "thinking"
        time.sleep(0.8 if not fast else 0.1)
        stream_response(assistant_msg, response_speed)

        # Show response time with emphasis
        response_time = RESPONSE_TIMES[response_idx % len(RESPONSE_TIMES)]
        show_response_time(response_time)

        # Pause to read AI response
        time.sleep(DELAY_AFTER_AI_RESPONSE if not fast else 0.3)

        # Show JSON response if enabled
        if show_json:
            show_json_response(assistant_msg)
            # Pause to examine JSON response
            time.sleep(DELAY_AFTER_JSON_BLOCK if not fast else 0.3)

        console.print()

        # Add assistant response to history
        messages.append({"role": "assistant", "content": assistant_msg})
        response_idx += 1

    # End demo with summary
    console.print()
    console.print(Panel(
        Text.from_markup(
            "[bold green]Demo Complete![/bold green]\n\n"
            "[bold]What you just saw:[/bold]\n"
            "- Real-time streaming responses from Mistral-7B\n"
            "- The /json command revealing raw API exchanges\n"
            "- Conversation history growing with each message\n"
            "- Response times from cloud GPU inference\n\n"
            "[dim]This transparency is invaluable for learning\n"
            "how LLM APIs work under the hood![/dim]"
        ),
        title="That's a Wrap!",
        border_style="green",
    ))


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    """Parse arguments and run demo."""
    fast = "--fast" in sys.argv
    no_typing = "--no-typing" in sys.argv

    try:
        run_demo(fast=fast, no_typing=no_typing)
    except KeyboardInterrupt:
        console.print("\n\n[dim]Demo interrupted.[/dim]")
        sys.exit(0)


if __name__ == "__main__":
    main()
