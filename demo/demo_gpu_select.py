#!/usr/bin/env python3
"""
SolvX QuickPod - GPU Selection Demo (YouTube Landscape)

Simulates the full v1.1.0 experience: GPU selection, pod launch, and chat.
Designed for YouTube demos at full screen resolution.

Usage:
    python demo/demo_gpu_select.py [--fast] [--no-typing]

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
RESPONSE_TIMES = [1.2, 1.8, 0.9]

DEMO_GPUS = [
    {"display_name": "RTX 3080",     "vram_gb": 10, "price_hr": 0.19},
    {"display_name": "RTX A4000",    "vram_gb": 16, "price_hr": 0.24},
    {"display_name": "RTX 4000 Ada", "vram_gb": 20, "price_hr": 0.34},
    {"display_name": "RTX 3090",     "vram_gb": 24, "price_hr": 0.44},
    {"display_name": "RTX 4090",     "vram_gb": 24, "price_hr": 0.69},
    {"display_name": "RTX A6000",    "vram_gb": 48, "price_hr": 0.79},
    {"display_name": "A100 SXM",     "vram_gb": 80, "price_hr": 1.94},
]

DEMO_EXCHANGES = [
    {
        "user": "What makes cloud GPUs useful for running AI models?",
        "assistant": "Cloud GPUs let you access powerful hardware without buying expensive equipment. A single RTX 3090 costs over $1500 to buy, but you can rent one for under $0.50/hour. You only pay for what you use - spin it up when you need it, shut it down when you're done.\n\nFor running models like Mistral-7B, you need at least 10GB of VRAM. Cloud GPUs give you that on demand, anywhere in the world.",
    },
    {
        "user": "How fast are the responses compared to ChatGPT?",
        "assistant": "Response times depend on your GPU choice and the model. With Mistral-7B on an RTX 3090, you typically get 1-2 second response times for short answers. Longer responses stream in word by word, so you start reading immediately.\n\nThe big advantage is privacy - your conversations never leave the GPU pod. No data logging, no training on your inputs. It's your own private AI.",
    },
    {
        "user": "Can I use a cheaper GPU to save money?",
        "assistant": "Absolutely! Mistral-7B with AWQ quantization only needs about 4GB of VRAM, so even a 10GB RTX 3080 at $0.19/hour works great. That's less than $2 for a full 10-hour session.\n\nThe cheapest available option auto-selects the best deal for your VRAM needs. If budget is the priority, go with the lowest price. If you want faster inference, a beefier GPU will generate tokens quicker.",
    },
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


def show_response_time(seconds: float) -> None:
    if seconds < 1.5:
        color, label = "green", "FAST"
    elif seconds < 2.5:
        color, label = "yellow", "OK"
    else:
        color, label = "red", "SLOW"
    console.print(f"[bold {color} on black] {label}: {seconds:.1f}s [/bold {color} on black]")


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

    # --- Welcome ---
    console.print()
    console.print(Panel(
        Text.from_markup(
            "[bold cyan]SolvX QuickPod v1.1.0[/bold cyan]\n\n"
            "[bold]GPU Selection Demo[/bold]\n\n"
            "Choose your GPU, launch a pod, and chat.\n"
            "Cloud AI on your terms.\n\n"
            "[dim]Press Ctrl+C to exit at any time.[/dim]"
        ),
        title="One-Click AI Chat",
        border_style="cyan",
    ))
    delay(3.0, fast)

    # --- App header ---
    console.print()
    console.print("[bold]=== SolvX QuickPod v1.1.0 ===[/bold]")
    console.print()
    delay(1.5, fast)

    # --- GPU Selection ---
    console.print("[bold]=== GPU Selection ===[/bold]")
    console.print("[bold]Minimum VRAM in GB [16]: [/bold]", end="")
    delay(1.0, fast)
    type_text("16", typing_speed)
    delay(1.0, fast)

    console.print("\n[dim]Fetching available GPUs...[/dim]")
    delay(2.0, fast)

    # Display GPU list
    console.print()
    console.print("  1. [bold]Cheapest available (auto-select)[/bold]")
    for i, gpu in enumerate(DEMO_GPUS, start=2):
        console.print(
            f"  {i}. {gpu['display_name']:20s} | "
            f"{gpu['vram_gb']:3d} GB | "
            f"${gpu['price_hr']:.2f}/hr"
        )

    delay(4.0, fast)

    # User selects option 1 (cheapest)
    console.print(f"\n[bold]Select GPU (1-{len(DEMO_GPUS) + 1}) [1]: [/bold]", end="")
    delay(1.5, fast)
    type_text("1", typing_speed)
    delay(0.5, fast)

    selected = DEMO_GPUS[0]
    console.print(
        f"\n[bold green]Selected: {selected['display_name']} "
        f"({selected['vram_gb']} GB) - ${selected['price_hr']:.2f}/hr[/bold green]"
    )
    console.print("[bold]Proceed? (y/n) [y]: [/bold]", end="")
    delay(1.0, fast)
    type_text("y", typing_speed)
    delay(1.5, fast)

    # --- Pod Launch ---
    console.print("\n[bold]=== Starting Pod ===[/bold]")
    console.print(f"[bold]GPU: {selected['display_name']}[/bold]")
    console.print("[bold]Model: Mistral-7B[/bold]")
    delay(1.5, fast)

    console.print("[dim]Starting GPU pod...[/dim]")
    stages = [
        "Finding available GPU...",
        "GPU assigned, preparing container...",
        "Container starting...",
        "GPU pod is running!",
    ]
    for stage in stages:
        delay(1.5, fast)
        console.print(f"  {stage}")

    delay(1.5, fast)

    # --- Model Loading ---
    console.print("[dim]Loading AI model - chat will be ready in 8-9 minutes...[/dim]", end="")
    for _ in range(8):
        delay(0.5, fast)
        console.print("[dim].[/dim]", end="")
    console.print()
    console.print("[dim]  Model loaded: TheBloke/Mistral-7B-Instruct-v0.2-AWQ[/dim]")
    delay(1.5, fast)

    # --- Pod Ready ---
    console.print("\n[bold]=== Pod Ready ===[/bold]\n")
    delay(1.0, fast)

    console.print(
        f"[dim]GPU: {selected['display_name']} "
        f"(~${selected['price_hr']:.2f}/hour) | Model: Mistral-7B[/dim]"
    )
    console.print("[dim]Temperature: 0.5 | History: Last 10 turns[/dim]")
    console.print()
    console.print("[bold]Chat started. Commands: /json, /stop, /help. Ctrl+C to exit.[/bold]")
    console.print()
    delay(2.0, fast)

    # --- Chat Exchanges ---
    for idx, exchange in enumerate(DEMO_EXCHANGES):
        console.print("[dim]" + "-" * 60 + "[/dim]")
        console.print()
        delay(1.5, fast)

        # User input
        console.print("[bold white on blue] YOU [/bold white on blue] ", end="")
        type_text(exchange["user"], typing_speed)
        delay(1.5, fast)

        # AI response
        console.print()
        console.print("[bold white on green] AI [/bold white on green] ", end="")
        delay(0.8, fast)
        stream_response(exchange["assistant"], response_speed)

        # Response time
        show_response_time(RESPONSE_TIMES[idx % len(RESPONSE_TIMES)])
        delay(3.5, fast)
        console.print()

    # --- End ---
    console.print()
    console.print(Panel(
        Text.from_markup(
            "[bold green]Demo Complete![/bold green]\n\n"
            "[bold]What you just saw:[/bold]\n"
            "- Live GPU selection from 30+ options\n"
            "- Cheapest available auto-selection\n"
            "- Full pod launch and model loading\n"
            "- Streaming AI chat with fast response times\n\n"
            "[bold]Starting at $0.12/hour[/bold]\n"
            "[dim]Download at github.com/tradewithmeai/solvx-quickpod[/dim]"
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
