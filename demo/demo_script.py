#!/usr/bin/env python3
"""
SolvX QuickPod - Demo Script

Simulates a chat session for screenshots and screen recordings.
Uses pre-defined prompts with realistic typing delays.

Usage:
    python demo/demo_script.py [--fast] [--no-typing]

Options:
    --fast       Reduce delays for quick run-through
    --no-typing  Instant text display (for testing)
"""

from __future__ import annotations

import sys
import time
from typing import List, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

# =============================================================================
# DEMO CONFIGURATION
# =============================================================================

# Typing speed (characters per second)
TYPING_SPEED = 25  # Normal human typing speed
TYPING_SPEED_FAST = 100

# Delays between actions (seconds)
DELAY_AFTER_PROMPT = 0.5
DELAY_AFTER_RESPONSE = 1.5
DELAY_BETWEEN_SECTIONS = 2.0

# Demo conversation
DEMO_CONVERSATION: List[Tuple[str, str]] = [
    (
        "Hello! What can you help me with?",
        "Hello! I'm your AI assistant running on RunPod's cloud GPUs. I can help you with:\n\n"
        "- **Coding questions** - debugging, code review, explanations\n"
        "- **Writing tasks** - drafting, editing, summarizing\n"
        "- **Research** - explaining concepts, answering questions\n"
        "- **Brainstorming** - ideas, problem-solving, planning\n\n"
        "What would you like to explore today?"
    ),
    (
        "Write a Python function to check if a number is prime",
        "Here's an efficient prime number checker:\n\n"
        "```python\n"
        "def is_prime(n: int) -> bool:\n"
        "    \"\"\"Check if a number is prime.\"\"\"\n"
        "    if n < 2:\n"
        "        return False\n"
        "    if n == 2:\n"
        "        return True\n"
        "    if n % 2 == 0:\n"
        "        return False\n"
        "    for i in range(3, int(n**0.5) + 1, 2):\n"
        "        if n % i == 0:\n"
        "            return False\n"
        "    return True\n"
        "```\n\n"
        "This uses the square root optimization - we only need to check divisors up to √n. "
        "Would you like me to explain how it works or add any features?"
    ),
    (
        "That's great! How do I use this app efficiently?",
        "Here are some tips for getting the most out of SolvX QuickPod:\n\n"
        "**Commands:**\n"
        "- `/json` - Toggle debug mode to see raw API requests\n"
        "- `/stop` - Terminate the pod (stops billing)\n"
        "- `/help` - Show available commands\n\n"
        "**Best Practices:**\n"
        "- Be specific in your questions for better responses\n"
        "- The conversation history is preserved (last 10 turns)\n"
        "- Use `/stop` when done to avoid unnecessary charges\n\n"
        "**Cost Efficiency:**\n"
        "- RTX 3090 runs at ~$0.44/hour\n"
        "- Your $15 credit = ~34 hours of chat time\n\n"
        "Anything else you'd like to know?"
    ),
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


def stream_response(text: str, speed: int = 50) -> None:
    """Simulate streaming response from AI."""
    words = text.split(" ")
    for i, word in enumerate(words):
        if i > 0:
            console.print(" ", end="")
        console.print(word, end="", highlight=False, soft_wrap=True)
        time.sleep(1 / speed)
    print()


def show_welcome() -> None:
    """Display the demo welcome banner."""
    console.print()
    console.print(Panel(
        Text.from_markup(
            "[bold cyan]SolvX QuickPod Demo[/bold cyan]\n\n"
            "This demo simulates a chat session for screenshots.\n"
            "Press [bold]Ctrl+C[/bold] to exit at any time."
        ),
        title="Demo Mode",
        border_style="cyan",
    ))
    console.print()


def run_demo(fast: bool = False, no_typing: bool = False) -> None:
    """Run the demo conversation."""
    typing_speed = TYPING_SPEED_FAST if fast else TYPING_SPEED
    response_speed = 100 if fast else 50

    if no_typing:
        typing_speed = 10000
        response_speed = 10000

    show_welcome()

    # Simulate app header
    console.print("[bold]=== SolvX QuickPod ===[/bold]")
    console.print()
    time.sleep(DELAY_BETWEEN_SECTIONS if not fast else 0.5)

    # Simulate pod ready
    console.print("[dim]Temperature: 0.5[/dim]")
    console.print("[dim]History: Last 10 turns[/dim]")
    console.print()
    console.print("[bold]Chat started. Commands: /json, /stop, /help. Ctrl+C to exit.[/bold]")
    console.print()

    time.sleep(DELAY_BETWEEN_SECTIONS if not fast else 0.5)

    # Run conversation
    for user_input, ai_response in DEMO_CONVERSATION:
        # User input
        console.print("[bold cyan]You > [/bold cyan]", end="")
        type_text(user_input, typing_speed)
        time.sleep(DELAY_AFTER_PROMPT if not fast else 0.2)

        # AI response
        console.print()
        console.print("[bold green]AI >[/bold green] ", end="")
        stream_response(ai_response, response_speed)
        console.print(f"[dim](2.3s)[/dim]")
        console.print()

        time.sleep(DELAY_AFTER_RESPONSE if not fast else 0.5)

    # End demo
    console.print()
    console.print(Panel(
        "[bold green]Demo Complete![/bold green]\n\n"
        "You've seen the key features of SolvX QuickPod.",
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
