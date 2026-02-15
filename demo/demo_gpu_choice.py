#!/usr/bin/env python3
"""
SolvX QuickPod - GPU Choice Demo (YouTube Landscape)

Showcases the GPU selection experience in v1.1.0:
  - VRAM filtering with custom thresholds
  - Cheapest available auto-selection
  - Manual GPU picking from a sorted price list
  - Confirmation before launch

Designed for YouTube demos at full screen resolution (~2-3 min).

Usage:
    python demo/demo_gpu_choice.py [--fast] [--no-typing]

Options:
    --fast       Reduce delays for quick run-through
    --no-typing  Instant text display (for testing)
"""

from __future__ import annotations

import ctypes
import sys
import time

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Go fullscreen on Windows by simulating F11
if sys.platform == "win32":
    try:
        user32 = ctypes.windll.user32
        VK_F11 = 0x7A
        user32.keybd_event(VK_F11, 0, 0, 0)
        user32.keybd_event(VK_F11, 0, 0x0002, 0)
    except Exception:
        pass

console = Console()

# =============================================================================
# CONFIGURATION
# =============================================================================

TYPING_SPEED = 12
TYPING_SPEED_FAST = 100
AI_RESPONSE_SPEED = 8

# Full GPU list matching the real app - sorted by price
ALL_GPUS = [
    {"display_name": "RTX A2000",       "vram_gb":  6, "price_hr": 0.12, "cloud": "Community"},
    {"display_name": "RTX 3070",        "vram_gb":  8, "price_hr": 0.16, "cloud": "Community"},
    {"display_name": "RTX 3080",        "vram_gb": 10, "price_hr": 0.19, "cloud": "Community"},
    {"display_name": "RTX 3080 Ti",     "vram_gb": 12, "price_hr": 0.22, "cloud": "Community"},
    {"display_name": "RTX A4000",       "vram_gb": 16, "price_hr": 0.24, "cloud": "Community"},
    {"display_name": "RTX 4070 Ti",     "vram_gb": 16, "price_hr": 0.28, "cloud": "Community"},
    {"display_name": "RTX 4080",        "vram_gb": 16, "price_hr": 0.36, "cloud": "Community"},
    {"display_name": "RTX 4000 Ada",    "vram_gb": 20, "price_hr": 0.34, "cloud": "Secure"},
    {"display_name": "RTX 3090",        "vram_gb": 24, "price_hr": 0.44, "cloud": "Community"},
    {"display_name": "RTX 4090",        "vram_gb": 24, "price_hr": 0.69, "cloud": "Secure"},
    {"display_name": "RTX A6000",       "vram_gb": 48, "price_hr": 0.79, "cloud": "Secure"},
    {"display_name": "L40S",            "vram_gb": 48, "price_hr": 0.94, "cloud": "Secure"},
    {"display_name": "A100 SXM 80GB",   "vram_gb": 80, "price_hr": 1.94, "cloud": "Secure"},
]


# =============================================================================
# DEMO FUNCTIONS
# =============================================================================

def type_text(text: str, speed: int = TYPING_SPEED) -> None:
    for char in text:
        console.print(char, end="", highlight=False)
        time.sleep(1 / speed)
    print()


def stream_response(text: str, words_per_second: int = 8) -> None:
    words = text.split(" ")
    for i, word in enumerate(words):
        if i > 0:
            console.print(" ", end="")
        console.print(word, end="", highlight=False, soft_wrap=True)
        time.sleep(1.0 / words_per_second)
    print()


def delay(seconds: float, fast: bool) -> None:
    time.sleep(seconds * (0.15 if fast else 1.0))


def filter_gpus(min_vram: int) -> list:
    return [g for g in ALL_GPUS if g["vram_gb"] >= min_vram]


def print_gpu_list(gpus: list) -> None:
    console.print()
    console.print(
        "  [bold cyan]1.[/bold cyan] [bold]Cheapest available "
        "(auto-select)[/bold]"
    )
    for i, gpu in enumerate(gpus, start=2):
        cloud_tag = (
            f"[green]{gpu['cloud']}[/green]"
            if gpu["cloud"] == "Secure"
            else f"[yellow]{gpu['cloud']}[/yellow]"
        )
        console.print(
            f"  [bold cyan]{i:2d}.[/bold cyan] "
            f"{gpu['display_name']:20s} "
            f"[dim]{gpu['vram_gb']:3d} GB[/dim]  "
            f"[bold]${gpu['price_hr']:.2f}/hr[/bold]  "
            f"{cloud_tag}"
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

    # --- Welcome ---
    console.print()
    console.print(Panel(
        Text.from_markup(
            "[bold cyan]SolvX QuickPod v1.1.0[/bold cyan]\n\n"
            "[bold]GPU Selection Deep Dive[/bold]\n\n"
            "Choose from 30+ GPUs starting at $0.12/hr.\n"
            "Filter by VRAM, pick the cheapest, or go manual.\n\n"
            "[dim]Press Ctrl+C to exit at any time.[/dim]"
        ),
        title="One-Click AI Chat",
        border_style="cyan",
    ))
    delay(4.0, fast)

    # --- App header ---
    console.print()
    console.print("[bold]=== SolvX QuickPod v1.1.0 ===[/bold]")
    console.print()
    delay(1.5, fast)

    # =========================================================================
    # SCENARIO 1: Default VRAM (16 GB) + Cheapest auto-select
    # =========================================================================
    console.print(Panel(
        "[bold]Scenario 1:[/bold] Just give me the cheapest GPU",
        border_style="dim",
    ))
    delay(2.0, fast)

    console.print("[bold]=== GPU Selection ===[/bold]")
    console.print("[bold]Minimum VRAM in GB [16]: [/bold]", end="")
    delay(1.5, fast)
    # User presses Enter for default
    console.print("[dim](enter)[/dim]")
    delay(1.0, fast)

    console.print("\n[dim]Fetching available GPUs...[/dim]")
    delay(2.0, fast)

    gpus_16 = filter_gpus(16)
    print_gpu_list(gpus_16)
    delay(3.0, fast)

    console.print(
        f"\n[bold]Select GPU (1-{len(gpus_16) + 1}) [1]: [/bold]", end=""
    )
    delay(1.5, fast)
    type_text("1", typing_speed)
    delay(0.5, fast)

    cheapest = gpus_16[0]
    console.print(
        f"\n[bold green]  >>> {cheapest['display_name']} "
        f"({cheapest['vram_gb']} GB) - "
        f"${cheapest['price_hr']:.2f}/hr "
        f"[{cheapest['cloud']}][/bold green]"
    )
    console.print("[bold]Proceed? (y/n) [y]: [/bold]", end="")
    delay(1.0, fast)
    type_text("y", typing_speed)
    delay(1.0, fast)

    console.print(
        "\n[green]Pod would launch here with "
        f"{cheapest['display_name']} at "
        f"${cheapest['price_hr']:.2f}/hr[/green]"
    )
    delay(3.0, fast)

    # =========================================================================
    # SCENARIO 2: Low VRAM filter (8 GB) to find budget GPUs
    # =========================================================================
    console.print()
    console.print(Panel(
        "[bold]Scenario 2:[/bold] I want the absolute cheapest - "
        "lower the VRAM filter",
        border_style="dim",
    ))
    delay(2.5, fast)

    console.print("[bold]=== GPU Selection ===[/bold]")
    console.print("[bold]Minimum VRAM in GB [16]: [/bold]", end="")
    delay(1.5, fast)
    type_text("8", typing_speed)
    delay(1.0, fast)

    console.print("\n[dim]Fetching available GPUs...[/dim]")
    delay(2.0, fast)

    gpus_8 = filter_gpus(8)
    print_gpu_list(gpus_8)
    delay(3.0, fast)

    console.print(
        f"\n  [bold]More GPUs unlocked![/bold] "
        f"[dim]({len(gpus_8)} options from "
        f"${gpus_8[0]['price_hr']:.2f}/hr)[/dim]"
    )
    delay(2.0, fast)

    # User picks cheapest again
    console.print(
        f"\n[bold]Select GPU (1-{len(gpus_8) + 1}) [1]: [/bold]", end=""
    )
    delay(1.5, fast)
    type_text("1", typing_speed)
    delay(0.5, fast)

    budget = gpus_8[0]
    console.print(
        f"\n[bold green]  >>> {budget['display_name']} "
        f"({budget['vram_gb']} GB) - "
        f"${budget['price_hr']:.2f}/hr "
        f"[{budget['cloud']}][/bold green]"
    )
    console.print("[bold]Proceed? (y/n) [y]: [/bold]", end="")
    delay(1.0, fast)
    type_text("y", typing_speed)
    delay(1.0, fast)

    console.print(
        f"\n[green]Even cheaper! "
        f"${budget['price_hr']:.2f}/hr - that's "
        f"${budget['price_hr'] * 10:.2f} for a 10 hour session[/green]"
    )
    delay(3.5, fast)

    # =========================================================================
    # SCENARIO 3: Manual pick - user wants a specific GPU
    # =========================================================================
    console.print()
    console.print(Panel(
        "[bold]Scenario 3:[/bold] I want a specific GPU - "
        "the RTX 4090",
        border_style="dim",
    ))
    delay(2.5, fast)

    console.print("[bold]=== GPU Selection ===[/bold]")
    console.print("[bold]Minimum VRAM in GB [16]: [/bold]", end="")
    delay(1.5, fast)
    type_text("24", typing_speed)
    delay(1.0, fast)

    console.print("\n[dim]Fetching available GPUs...[/dim]")
    delay(2.0, fast)

    gpus_24 = filter_gpus(24)
    print_gpu_list(gpus_24)
    delay(3.0, fast)

    # Find RTX 4090 index
    target_idx = next(
        i for i, g in enumerate(gpus_24) if "4090" in g["display_name"]
    )
    console.print(
        f"\n[bold]Select GPU (1-{len(gpus_24) + 1}) [1]: [/bold]", end=""
    )
    delay(1.5, fast)
    type_text(str(target_idx + 2), typing_speed)  # +2 because cheapest is 1
    delay(0.5, fast)

    picked = gpus_24[target_idx]
    console.print(
        f"\n[bold green]  >>> {picked['display_name']} "
        f"({picked['vram_gb']} GB) - "
        f"${picked['price_hr']:.2f}/hr "
        f"[{picked['cloud']}][/bold green]"
    )

    # --- User changes their mind ---
    console.print("[bold]Proceed? (y/n) [y]: [/bold]", end="")
    delay(2.0, fast)
    type_text("n", typing_speed)
    delay(1.0, fast)

    console.print("\n[dim]Going back to GPU selection...[/dim]")
    delay(2.0, fast)

    # User picks cheapest instead
    console.print(
        f"\n[bold]Select GPU (1-{len(gpus_24) + 1}) [1]: [/bold]", end=""
    )
    delay(1.0, fast)
    type_text("1", typing_speed)
    delay(0.5, fast)

    final = gpus_24[0]
    console.print(
        f"\n[bold green]  >>> {final['display_name']} "
        f"({final['vram_gb']} GB) - "
        f"${final['price_hr']:.2f}/hr "
        f"[{final['cloud']}][/bold green]"
    )
    console.print("[bold]Proceed? (y/n) [y]: [/bold]", end="")
    delay(1.0, fast)
    type_text("y", typing_speed)
    delay(1.5, fast)

    # --- Quick launch + chat teaser ---
    console.print("\n[bold]=== Starting Pod ===[/bold]")
    console.print(f"[bold]GPU: {final['display_name']}[/bold]")
    console.print("[bold]Model: Mistral-7B[/bold]")
    delay(1.5, fast)

    stages = [
        "Finding available GPU...",
        "GPU assigned, preparing container...",
        "Container starting...",
        "GPU pod is running!",
    ]
    for stage in stages:
        delay(1.0, fast)
        console.print(f"  {stage}")

    delay(1.0, fast)
    console.print("[dim]Loading AI model...[/dim]", end="")
    for _ in range(6):
        delay(0.4, fast)
        console.print("[dim].[/dim]", end="")
    console.print()
    delay(0.5, fast)

    console.print("[bold green]  Model loaded![/bold green]")
    delay(1.5, fast)

    # One quick chat to prove it works
    console.print("\n[bold]=== Chat Ready ===[/bold]")
    console.print(
        f"[dim]GPU: {final['display_name']} "
        f"(~${final['price_hr']:.2f}/hour) | Model: Mistral-7B[/dim]"
    )
    console.print()
    delay(1.5, fast)

    console.print("[bold white on blue] YOU [/bold white on blue] ", end="")
    type_text(
        "What's the cheapest way to run AI privately?", typing_speed
    )
    delay(1.0, fast)

    console.print()
    console.print("[bold white on green] AI [/bold white on green] ", end="")
    delay(0.5, fast)
    stream_response(
        "With QuickPod you can rent a GPU from $0.12/hour and run "
        "models like Mistral-7B completely privately. Your data never "
        "leaves the pod. A 10-hour session on the cheapest GPU costs "
        "under $2 - far less than any monthly AI subscription.",
        response_speed,
    )
    console.print("[bold green on black] FAST: 1.1s [/bold green on black]")
    delay(3.0, fast)

    # --- End ---
    console.print()
    console.print(Panel(
        Text.from_markup(
            "[bold green]Demo Complete![/bold green]\n\n"
            "[bold]GPU Selection Features:[/bold]\n"
            "  - Filter by minimum VRAM (6-80 GB)\n"
            "  - Cheapest available auto-selection\n"
            "  - Manual pick from sorted price list\n"
            "  - Change your mind before launch\n"
            "  - Secure and Community cloud options\n\n"
            "[bold]30+ GPUs from $0.12/hour[/bold]\n"
            "[dim]github.com/tradewithmeai/solvx-quickpod[/dim]"
        ),
        title="SolvX QuickPod v1.1.0",
        border_style="green",
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
