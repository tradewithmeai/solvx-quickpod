#!/usr/bin/env python3
"""
SolvX QuickPod - Social Media Demo (Portrait 9:16)

Short, punchy demo for TikTok, Instagram Reels, and YouTube Shorts.
Runs in a narrow 42-character terminal to simulate mobile portrait view.

Usage:
    python demo/demo_social.py [--fast] [--no-typing]

Recording tip:
    Resize your terminal to ~42 chars wide and as tall as possible,
    then screen record the terminal window.
"""

from __future__ import annotations

import os
import sys
import time

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Portrait width - narrow for 9:16 ratio
CONSOLE_WIDTH = 42

# Resize terminal on Windows
if sys.platform == "win32":
    os.system(f"mode con cols={CONSOLE_WIDTH} lines=60")

console = Console(width=CONSOLE_WIDTH)

# =============================================================================
# CONFIGURATION
# =============================================================================

TYPING_SPEED = 18  # Faster for short-form
TYPING_SPEED_FAST = 100
AI_RESPONSE_SPEED = 12  # Faster streaming

DEMO_GPUS = [
    {"display_name": "RTX 3080",  "vram_gb": 10, "price_hr": 0.19},
    {"display_name": "RTX A4000", "vram_gb": 16, "price_hr": 0.24},
    {"display_name": "RTX 3090",  "vram_gb": 24, "price_hr": 0.44},
    {"display_name": "RTX 4090",  "vram_gb": 24, "price_hr": 0.69},
]


# =============================================================================
# DEMO FUNCTIONS
# =============================================================================

def type_text(text: str, speed: int = TYPING_SPEED) -> None:
    for char in text:
        console.print(char, end="", highlight=False)
        time.sleep(1 / speed)
    print()


def stream_response(text: str, wps: int = AI_RESPONSE_SPEED) -> None:
    words = text.split(" ")
    for i, word in enumerate(words):
        if i > 0:
            console.print(" ", end="")
        console.print(word, end="", highlight=False, soft_wrap=True)
        time.sleep(1.0 / wps)
    print()


def delay(seconds: float, fast: bool) -> None:
    time.sleep(seconds * (0.15 if fast else 1.0))


# =============================================================================
# DEMO FLOW
# =============================================================================

def run_demo(fast: bool = False, no_typing: bool = False) -> None:
    typing_speed = TYPING_SPEED_FAST if fast else TYPING_SPEED
    response_speed = 100 if fast else AI_RESPONSE_SPEED

    if no_typing:
        typing_speed = 10000
        response_speed = 10000

    # --- Title Card ---
    console.print()
    console.print(Panel(
        Text.from_markup(
            "[bold cyan]SolvX QuickPod[/bold cyan]\n"
            "[dim]Your own AI for $0.19/hr[/dim]"
        ),
        border_style="cyan",
    ))
    delay(2.5, fast)

    # --- GPU Selection (quick) ---
    console.print()
    console.print("[bold]=== Pick Your GPU ===[/bold]")
    console.print()
    delay(1.0, fast)

    console.print("  1. [bold]Cheapest (auto)[/bold]")
    for i, gpu in enumerate(DEMO_GPUS, start=2):
        console.print(
            f"  {i}. {gpu['display_name']:10s} "
            f"{gpu['vram_gb']:2d}GB "
            f"${gpu['price_hr']:.2f}/hr"
        )
        delay(0.3, fast)

    delay(2.0, fast)

    # Select cheapest
    console.print()
    console.print("[bold]> [/bold]", end="")
    delay(0.5, fast)
    type_text("1", typing_speed)
    delay(0.5, fast)

    selected = DEMO_GPUS[0]
    console.print(
        f"[green]{selected['display_name']} "
        f"${selected['price_hr']:.2f}/hr[/green]"
    )
    delay(1.5, fast)

    # --- Pod Launch (compressed) ---
    console.print()
    console.print("[bold]=== Launching ===[/bold]")
    delay(0.8, fast)

    steps = [
        "Finding GPU...",
        "Container starting...",
        "Loading Mistral-7B...",
    ]
    for step in steps:
        console.print(f"[dim]  {step}[/dim]")
        delay(1.0, fast)

    # Progress dots
    console.print("[dim]  [/dim]", end="")
    for _ in range(6):
        console.print("[dim].[/dim]", end="")
        delay(0.4, fast)
    console.print()
    delay(0.5, fast)

    console.print("[bold green]  Ready![/bold green]")
    delay(1.5, fast)

    # --- Chat Exchange ---
    console.print()
    console.print("[dim]" + "-" * (CONSOLE_WIDTH - 2) + "[/dim]")
    console.print()

    # User message
    console.print("[bold blue]You >[/bold blue] ", end="")
    type_text("What can you help me with?", typing_speed)
    delay(1.0, fast)

    # AI response
    console.print()
    console.print("[bold green]AI >[/bold green] ", end="")
    delay(0.5, fast)
    stream_response(
        "I can help with coding, writing, "
        "analysis, brainstorming, math, "
        "and much more. Ask me anything!",
        response_speed,
    )

    console.print("[dim](1.2s)[/dim]")
    delay(2.5, fast)

    # --- Second quick exchange ---
    console.print()
    console.print("[bold blue]You >[/bold blue] ", end="")
    type_text("Write a Python hello world", typing_speed)
    delay(1.0, fast)

    console.print()
    console.print("[bold green]AI >[/bold green] ", end="")
    delay(0.5, fast)
    stream_response(
        'print("Hello, World!")\n\n'
        "That's it! Python keeps things "
        "simple. Run it and you'll see "
        "the greeting in your terminal.",
        response_speed,
    )
    console.print("[dim](0.9s)[/dim]")
    delay(2.0, fast)

    # --- End Card ---
    console.print()
    console.print(Panel(
        Text.from_markup(
            "[bold cyan]SolvX QuickPod[/bold cyan]\n\n"
            "[bold]Private AI chat[/bold]\n"
            "[bold]From $0.12/hour[/bold]\n\n"
            "[dim]Link in bio[/dim]"
        ),
        border_style="cyan",
    ))


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    fast = "--fast" in sys.argv
    no_typing = "--no-typing" in sys.argv

    try:
        run_demo(fast=fast, no_typing=no_typing)
    except KeyboardInterrupt:
        console.print("\n\n[dim]Demo interrupted.[/dim]")
        sys.exit(0)


if __name__ == "__main__":
    main()
