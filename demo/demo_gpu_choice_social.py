#!/usr/bin/env python3
"""
SolvX QuickPod - GPU Choice Demo (Portrait 9:16)

Short social media demo showcasing GPU selection:
  - VRAM filter to unlock cheap GPUs
  - Cheapest available auto-selection
  - Change your mind before launch

For TikTok, Instagram Reels, YouTube Shorts (~30-45 sec).

Usage:
    python demo/demo_gpu_choice_social.py [--fast] [--no-typing]

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

TYPING_SPEED = 18
TYPING_SPEED_FAST = 100
AI_RESPONSE_SPEED = 12

GPUS_16 = [
    {"display_name": "RTX A4000",  "vram_gb": 16, "price_hr": 0.24},
    {"display_name": "RTX 4070 Ti", "vram_gb": 16, "price_hr": 0.28},
    {"display_name": "RTX 4080",   "vram_gb": 16, "price_hr": 0.36},
    {"display_name": "RTX 3090",   "vram_gb": 24, "price_hr": 0.44},
    {"display_name": "RTX 4090",   "vram_gb": 24, "price_hr": 0.69},
]

GPUS_8 = [
    {"display_name": "RTX 3070",   "vram_gb":  8, "price_hr": 0.16},
    {"display_name": "RTX 3080",   "vram_gb": 10, "price_hr": 0.19},
    {"display_name": "RTX 3080 Ti", "vram_gb": 12, "price_hr": 0.22},
    {"display_name": "RTX A4000",  "vram_gb": 16, "price_hr": 0.24},
    {"display_name": "RTX 3090",   "vram_gb": 24, "price_hr": 0.44},
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


def print_gpu_list(gpus: list) -> None:
    console.print("  1. [bold]Cheapest (auto)[/bold]")
    for i, gpu in enumerate(gpus, start=2):
        console.print(
            f"  {i}. {gpu['display_name']:10s} "
            f"{gpu['vram_gb']:2d}GB "
            f"${gpu['price_hr']:.2f}/hr"
        )


# =============================================================================
# DEMO FLOW
# =============================================================================

def run_demo(fast: bool = False, no_typing: bool = False) -> None:
    typing_speed = TYPING_SPEED_FAST if fast else TYPING_SPEED
    response_speed = 100 if fast else AI_RESPONSE_SPEED

    if no_typing:
        typing_speed = 10000
        response_speed = 10000

    # --- Title ---
    console.print()
    console.print(Panel(
        Text.from_markup(
            "[bold cyan]SolvX QuickPod[/bold cyan]\n"
            "[dim]Pick your GPU. Pay by the hour.[/dim]"
        ),
        border_style="cyan",
    ))
    delay(2.5, fast)

    # --- Default 16 GB ---
    console.print()
    console.print("[bold]=== Pick Your GPU ===[/bold]")
    console.print()
    console.print("[bold]Min VRAM [16]: [/bold]", end="")
    delay(1.0, fast)
    console.print("[dim](enter)[/dim]")
    delay(1.0, fast)

    console.print()
    print_gpu_list(GPUS_16)
    delay(2.5, fast)

    console.print()
    console.print(
        f"[dim]Cheapest: ${GPUS_16[0]['price_hr']:.2f}/hr[/dim]"
    )
    delay(1.5, fast)

    # --- Lower to 8 GB ---
    console.print()
    console.print("[bold]Want cheaper? Lower the VRAM.[/bold]")
    delay(1.5, fast)

    console.print()
    console.print("[bold]Min VRAM [16]: [/bold]", end="")
    delay(0.8, fast)
    type_text("8", typing_speed)
    delay(1.0, fast)

    console.print()
    print_gpu_list(GPUS_8)
    delay(2.5, fast)

    console.print()
    console.print(
        f"[bold green]Now from "
        f"${GPUS_8[0]['price_hr']:.2f}/hr![/bold green]"
    )
    delay(2.0, fast)

    # --- Pick cheapest ---
    console.print()
    console.print("[bold]> [/bold]", end="")
    delay(0.5, fast)
    type_text("1", typing_speed)
    delay(0.5, fast)

    picked = GPUS_8[0]
    console.print(
        f"[green]{picked['display_name']} "
        f"${picked['price_hr']:.2f}/hr[/green]"
    )
    delay(1.0, fast)

    console.print("[bold]Proceed? (y/n): [/bold]", end="")
    delay(1.0, fast)
    type_text("y", typing_speed)
    delay(1.5, fast)

    # --- Quick launch ---
    console.print()
    console.print("[bold]=== Launching ===[/bold]")
    delay(0.5, fast)

    steps = ["Finding GPU...", "Starting...", "Loading model..."]
    for step in steps:
        console.print(f"[dim]  {step}[/dim]")
        delay(0.8, fast)

    console.print("[dim]  [/dim]", end="")
    for _ in range(5):
        console.print("[dim].[/dim]", end="")
        delay(0.3, fast)
    console.print()
    delay(0.3, fast)

    console.print("[bold green]  Ready![/bold green]")
    delay(1.5, fast)

    # --- One chat exchange ---
    console.print()
    console.print("[dim]" + "-" * (CONSOLE_WIDTH - 2) + "[/dim]")
    console.print()

    console.print("[bold blue]You >[/bold blue] ", end="")
    type_text("How much is this costing me?", typing_speed)
    delay(0.8, fast)

    console.print()
    console.print("[bold green]AI >[/bold green] ", end="")
    delay(0.4, fast)
    stream_response(
        f"You're on an {picked['display_name']} at "
        f"${picked['price_hr']:.2f}/hr. A full hour "
        "costs less than a coffee. Type /stop "
        "when you're done to stop billing.",
        response_speed,
    )
    console.print("[dim](1.1s)[/dim]")
    delay(2.0, fast)

    # --- End Card ---
    console.print()
    console.print(Panel(
        Text.from_markup(
            "[bold cyan]SolvX QuickPod[/bold cyan]\n\n"
            "[bold]30+ GPUs from $0.12/hr[/bold]\n"
            "[dim]You pick. You pay. You own it.[/dim]\n\n"
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
