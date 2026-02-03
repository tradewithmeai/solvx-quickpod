#!/usr/bin/env python3
"""
SolvX QuickPod - Core AI Module

Main application logic for the SolvX QuickPod chat interface, including:
- First-run onboarding orchestration
- Pod lifecycle management (create, reconnect, terminate)
- Interactive streaming chat with vLLM
- Session logging and history management

This module serves as the primary orchestrator, coordinating between
the launcher, config, storage, and onboarding modules.
"""

from __future__ import annotations

import atexit
import importlib
import json
import os
import signal
import sys
import time
from typing import Dict, List, Optional, Tuple

import httpx
import requests
from dotenv import load_dotenv
from rich.console import Console

# =============================================================================
# FIRST-RUN ONBOARDING (must run before loading environment)
# =============================================================================

from solvx_quickpod.onboarding import (
    check_first_run,
    get_env_path,
    run_onboarding,
    save_env_file,
)

if check_first_run():
    runpod_key, vllm_key = run_onboarding()
    save_env_file(runpod_key, vllm_key)
    print("\nStarting SolvX QuickPod...\n")

# Load environment after onboarding completes
load_dotenv(get_env_path(), override=True)

# =============================================================================
# MODULE IMPORTS (after environment is loaded)
# =============================================================================

from solvx_quickpod import storage
from solvx_quickpod.launcher import (
    clear_state,
    create_pod,
    is_pod_running,
    terminate_pod,
    wait_for_running,
    write_state,
)

# =============================================================================
# ENVIRONMENT VALIDATION
# =============================================================================

RUNPOD_API_KEY: Optional[str] = os.getenv("RUNPOD_API_KEY")
VLLM_API_KEY: Optional[str] = os.getenv("VLLM_API_KEY")

if not RUNPOD_API_KEY or not VLLM_API_KEY:
    print("[ERROR] Configuration failed. Please delete ~/.myai/.env and try again.")
    sys.exit(1)

# =============================================================================
# CONSTANTS
# =============================================================================

# GPU Configuration
GPU_TYPE: str = "NVIDIA GeForce RTX 3090"
GPU_COUNT: int = 1
GPU_COST_PER_HOUR: float = 0.44  # Approximate cost in USD

# Model Configuration
MODEL_ID: str = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"
SYSTEM_PROMPT: str = "You are a helpful AI assistant."

# Chat Configuration
MAX_TURNS: int = 10
TEMPERATURE: float = 0.5
MAX_TOKENS: int = 500

# Timing Configuration
POD_CHECK_INTERVAL: int = 10  # Seconds between pod status checks

# Marketing
REFERRAL_LINK: str = "https://runpod.io?ref=q04x36mf"

# Rich Console for formatted output
console: Console = Console()

# Global state for emergency cleanup
_active_pod_id: Optional[str] = None


# =============================================================================
# EMERGENCY POD TERMINATION (handles terminal close)
# =============================================================================

def _emergency_terminate() -> None:
    """
    Emergency pod termination when app is forcibly closed.

    Called by atexit and signal handlers when the terminal window is closed
    or the process receives a termination signal. Silently terminates any
    running pod to prevent unexpected billing charges.
    """
    global _active_pod_id

    if _active_pod_id:
        try:
            # Silent termination - no user interaction possible
            terminate_pod(_active_pod_id)
            clear_state()
        except Exception:
            pass  # Best effort - can't do much if this fails


def _signal_handler(signum, frame) -> None:
    """Handle termination signals by cleaning up and exiting."""
    _emergency_terminate()
    sys.exit(0)


def _setup_emergency_handlers() -> None:
    """
    Register handlers for emergency pod termination.

    Sets up atexit, signal handlers, and Windows console handlers to ensure
    pods are terminated when the application exits unexpectedly.
    """
    # Register atexit handler for normal exits
    atexit.register(_emergency_terminate)

    if sys.platform == "win32":
        # Windows: Use SetConsoleCtrlHandler to catch terminal window close.
        # signal.SIGBREAK does NOT catch the close button - we need the
        # Windows API directly. The handler gets ~5 seconds to complete.
        import ctypes

        @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_ulong)
        def _win_console_handler(event):
            # CTRL_CLOSE_EVENT=2, CTRL_LOGOFF_EVENT=5, CTRL_SHUTDOWN_EVENT=6
            if event in (2, 5, 6):
                _emergency_terminate()
                return True
            return False

        # Store reference at module level to prevent garbage collection
        global _win_handler_ref
        _win_handler_ref = _win_console_handler
        ctypes.windll.kernel32.SetConsoleCtrlHandler(_win_handler_ref, True)
    else:
        # Unix: SIGTERM is sent when terminal is closed, SIGHUP for hangup
        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGHUP, _signal_handler)


# Module-level reference for Windows console handler (prevents GC)
_win_handler_ref = None

# Initialize emergency handlers at module load
_setup_emergency_handlers()


# =============================================================================
# POD MANAGEMENT
# =============================================================================

def check_existing_pod() -> Optional[str]:
    """
    Check for an existing running pod from a previous session.

    Reloads the config module to get fresh state, then verifies
    the pod is still running via the RunPod API.

    Returns:
        Pod ID if a running pod exists, None otherwise.
    """
    from solvx_quickpod import config
    importlib.reload(config)

    if not config.POD_ID:
        return None

    if is_pod_running(config.POD_ID):
        return config.POD_ID

    # Pod exists in state but isn't running - clear stale state
    clear_state()
    return None


def launch_pod() -> Tuple[str, str]:
    """
    Launch a new RunPod GPU pod with vLLM.

    Creates the pod, waits for it to be running, and returns
    the connection details.

    Returns:
        Tuple of (pod_id, base_url)

    Raises:
        SystemExit: If pod creation fails.
    """
    print("\n=== Starting Pod ===")
    print(f"GPU: {GPU_TYPE}")
    print(f"Model: Mistral-7B")

    pod_id = create_pod(GPU_TYPE, GPU_COUNT)
    if not pod_id:
        print("[ERROR] Failed to create pod")
        sys.exit(1)

    write_state(pod_id, MODEL_ID)
    wait_for_running(pod_id)

    base_url = f"https://{pod_id}-8000.proxy.runpod.net"
    return pod_id, base_url


def wait_for_vllm_ready(pod_id: str) -> None:
    """
    Wait for the vLLM API to be fully operational.

    Polls the /v1/models endpoint until it returns a successful response,
    indicating the model is loaded and ready to serve requests.

    Args:
        pod_id: The pod identifier to check.

    Raises:
        SystemExit: If the pod terminates while waiting.
    """
    print("Loading AI model - chat will be ready in 8-9 minutes...")
    check_count = 0
    dot_count = 0

    while True:
        try:
            response = requests.get(
                f"https://{pod_id}-8000.proxy.runpod.net/v1/models",
                headers={"Authorization": f"Bearer {VLLM_API_KEY}"},
                timeout=10,
            )

            if response.status_code == 200:
                data = response.json()
                models = [m.get("id") for m in data.get("data", [])]
                print(f"\n  Model loaded: {models[0] if models else 'unknown'}")
                return

            # Show progress dots
            dot_count += 1
            if dot_count % 5 == 0:
                print(".", end="", flush=True)

        except requests.RequestException:
            pass  # Retry on network errors

        # Periodically verify pod is still running
        check_count += 1
        if check_count >= 10:
            check_count = 0
            if not is_pod_running(pod_id):
                print("\n\nPod terminated unexpectedly. This may be due to:")
                print("  - Insufficient RunPod credit")
                print("  - Manual termination from RunPod console")
                print("  - GPU availability issues")
                sys.exit(0)

        time.sleep(3)


# =============================================================================
# CHAT HISTORY MANAGEMENT
# =============================================================================

def trim_history(messages: List[Dict[str, str]], max_turns: int = 10) -> List[Dict[str, str]]:
    """
    Trim conversation history to maintain context window limits.

    Keeps the first message (contains system prompt) plus the most recent
    turns to stay within the model's context limits.

    Args:
        messages: The full message history.
        max_turns: Maximum number of conversation turns to keep.

    Returns:
        Trimmed message list.
    """
    if max_turns is None:
        return messages

    # Calculate max messages: first message + (max_turns * 2 for user/assistant pairs)
    max_messages = 1 + max_turns * 2

    if len(messages) > max_messages:
        return [messages[0]] + messages[-(max_messages - 1):]

    return messages


# =============================================================================
# INTERACTIVE CHAT
# =============================================================================

def run_chat(pod_id: str) -> None:
    """
    Run the interactive chat interface.

    Handles user input, streams responses from vLLM, and manages
    the conversation history. Supports commands for debugging and control.

    Available commands:
        /json  - Toggle JSON request/response display
        /stop  - Terminate pod and exit
        /help  - Show available commands

    Args:
        pod_id: The active pod identifier.
    """
    from solvx_quickpod import config
    importlib.reload(config)

    # Initialize session
    session_id = storage.new_session()
    storage.touch_user()

    # Conversation state
    # Note: Mistral doesn't support system role, so we prepend to first user message
    messages: List[Dict[str, str]] = []
    first_message = True
    show_json = False

    # Pod health monitoring
    last_pod_check = time.time()

    # Display session info
    console.print(f"[dim]GPU: RTX 3090 (~${GPU_COST_PER_HOUR:.2f}/hour) | Model: Mistral-7B[/dim]")
    console.print(f"[dim]Temperature: {TEMPERATURE} | History: Last {MAX_TURNS} turns[/dim]\n")
    console.print("[bold]Chat started. Commands: /json, /stop, /help. Ctrl+C to exit.[/bold]\n")

    while True:
        try:
            # Periodic pod health check
            if time.time() - last_pod_check > POD_CHECK_INTERVAL:
                if not is_pod_running(pod_id):
                    console.print("\n[dim]Pod terminated. Exiting.[/dim]")
                    sys.exit(0)
                last_pod_check = time.time()

            # Get user input
            user_input = console.input("[bold cyan]You > [/bold cyan]").strip()
            if not user_input:
                continue

            # Handle commands
            if user_input.lower() == "/help":
                _show_help()
                continue

            if user_input.lower() == "/json":
                show_json = not show_json
                console.print(f"[dim]JSON display: {'ON' if show_json else 'OFF'}[/dim]\n")
                continue

            if user_input.lower() == "/stop":
                if _confirm_stop(pod_id):
                    return
                continue

            # Prepare message content
            # Mistral requires alternating user/assistant roles, no system role
            if first_message:
                user_content = f"{SYSTEM_PROMPT}\n\n{user_input}"
                first_message = False
            else:
                user_content = user_input

            messages.append({"role": "user", "content": user_content})
            storage.log_message(session_id, "user", user_input)
            messages = trim_history(messages, MAX_TURNS)

            # Send request and stream response
            assistant_text = _stream_response(
                config=config,
                messages=messages,
                show_json=show_json,
            )

            # Save assistant response
            if assistant_text:
                messages.append({"role": "assistant", "content": assistant_text})
                storage.log_message(session_id, "assistant", assistant_text)
                messages = trim_history(messages, MAX_TURNS)

        except KeyboardInterrupt:
            _handle_exit(pod_id)
            sys.exit(0)

        except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError) as e:
            if not is_pod_running(pod_id):
                console.print("\n[dim]Pod terminated. This may be due to:[/dim]")
                console.print("[dim]  - Insufficient RunPod credit[/dim]")
                console.print("[dim]  - Manual termination from RunPod console[/dim]")
                console.print("[dim]  - Session timeout[/dim]")
                sys.exit(0)
            console.print("\n[yellow]Connection issue - retrying...[/yellow]")
            console.print("[dim]If this persists, check your internet connection.[/dim]\n")


def _show_help() -> None:
    """Display available chat commands."""
    console.print("[dim]Available commands:[/dim]")
    console.print("[dim]  /json - Toggle JSON request/response display[/dim]")
    console.print("[dim]  /stop - Terminate pod and stop billing[/dim]")
    console.print("[dim]  /help - Show this help[/dim]")
    console.print(f"[dim]\nGPU Cost: ~${GPU_COST_PER_HOUR:.2f}/hour (RTX 3090)[/dim]")
    console.print("[dim]Use /stop when done to avoid unnecessary charges.[/dim]\n")


def _confirm_stop(pod_id: str) -> bool:
    """
    Prompt for confirmation and terminate the pod.

    Args:
        pod_id: The pod to terminate.

    Returns:
        True if pod was terminated, False if cancelled.
    """
    confirm = input("Terminate pod? This will stop billing. (y/n): ").strip().lower()

    if confirm == "y":
        global _active_pod_id
        if terminate_pod(pod_id):
            _active_pod_id = None  # Clear so atexit doesn't double-terminate
            clear_state()
            console.print("[bold]Pod terminated.[/bold]")
            sys.exit(0)
        else:
            console.print("[red]Failed to terminate pod.[/red]")

    return False


def _handle_exit(pod_id: str) -> None:
    """
    Handle application exit - prompt to terminate running pod.

    Checks if the pod is still running and offers to terminate it
    to prevent unexpected billing charges.

    Args:
        pod_id: The pod identifier to check/terminate.
    """
    console.print("\n")

    # Verify pod is actually still running via API
    if not is_pod_running(pod_id):
        console.print("[dim]Pod already stopped.[/dim]")
        clear_state()
        return

    # Pod is running - warn user and offer to terminate
    console.print(f"[yellow]Warning: Your GPU pod is still running![/yellow]")
    console.print(f"[yellow]It will continue billing at ~${GPU_COST_PER_HOUR:.2f}/hour until stopped.[/yellow]\n")

    try:
        choice = input("Terminate pod now? (y/n): ").strip().lower()

        if choice == "y":
            global _active_pod_id
            console.print("[dim]Terminating pod...[/dim]")
            if terminate_pod(pod_id):
                _active_pod_id = None  # Clear so atexit doesn't double-terminate
                clear_state()
                console.print("[bold green]Pod terminated. Billing stopped.[/bold green]")
            else:
                console.print("[red]Failed to terminate pod automatically.[/red]")
                console.print("[yellow]Please terminate manually at: https://www.runpod.io/console/pods[/yellow]")
        else:
            console.print("[dim]Pod left running. Use /stop next time or terminate at:[/dim]")
            console.print("[dim]https://www.runpod.io/console/pods[/dim]")

    except (KeyboardInterrupt, EOFError):
        # User pressed Ctrl+C again - leave pod running
        console.print("\n[dim]Exiting. Pod left running.[/dim]")
        console.print("[dim]Terminate at: https://www.runpod.io/console/pods[/dim]")


def _stream_response(config, messages: List[Dict[str, str]], show_json: bool) -> str:
    """
    Send a chat request and stream the response.

    Args:
        config: The config module (reloaded for fresh state).
        messages: The conversation history.
        show_json: Whether to display JSON debug output.

    Returns:
        The complete assistant response text.
    """
    request_start = time.time()

    payload = {
        "model": config.MODEL,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "stream": True,
    }

    headers = {
        "Authorization": f"Bearer {config.API_KEY}",
        "Content-Type": "application/json",
    }

    # Debug: show request JSON
    if show_json:
        console.print("\n[dim]>>> REQUEST:[/dim]")
        console.print(f"[dim]{json.dumps(payload, indent=2)}[/dim]\n")

    console.print("\n[bold green]AI >[/bold green] ", end="")
    assistant_text = ""

    with httpx.Client(timeout=None) as client:
        with client.stream(
            "POST",
            config.LLM_BASE_URL.rstrip("/") + "/v1/chat/completions",
            headers=headers,
            json=payload,
        ) as response:

            if response.status_code != 200:
                error_text = response.read().decode()
                if response.status_code == 502:
                    console.print("\n[yellow]Model is still loading, please wait a moment and try again...[/yellow]")
                elif response.status_code == 503:
                    console.print("\n[yellow]Server temporarily unavailable. Please try again.[/yellow]")
                elif response.status_code == 401:
                    console.print("\n[red]Authentication error. Your API key may be invalid.[/red]")
                else:
                    console.print(f"\n[red]Server error ({response.status_code}): {error_text[:200]}[/red]")
                return ""

            # Stream response tokens
            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue

                data = line[6:]
                if data == "[DONE]":
                    break

                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"]
                    if "content" in delta:
                        token = delta["content"]
                        assistant_text += token
                        console.print(token, end="", soft_wrap=True)
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    # Display timing
    elapsed = time.time() - request_start
    console.print(f"\n[dim]({elapsed:.1f}s)[/dim]")

    # Debug: show response JSON
    if show_json and assistant_text:
        console.print("\n[dim]<<< RESPONSE:[/dim]")
        console.print(f"[dim]{json.dumps({'role': 'assistant', 'content': assistant_text}, indent=2)}[/dim]")

    console.print()
    return assistant_text


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main() -> None:
    """
    Main application entry point.

    Orchestrates the complete user flow:
    1. Display welcome message
    2. Check for existing running pod (offer reconnection)
    3. Launch new pod if needed
    4. Wait for vLLM to be ready
    5. Start interactive chat session
    """
    global _active_pod_id

    try:
        # Welcome with version
        from solvx_quickpod import __version__
        print(f"\n=== SolvX QuickPod v{__version__} ===\n")

        # Check for existing pod
        existing_pod_id = check_existing_pod()

        if existing_pod_id:
            print("=== Existing Pod Found ===")
            print(f"Pod ID: {existing_pod_id}")
            print(f"[Note: Pod is still running and billing at ~${GPU_COST_PER_HOUR:.2f}/hour]")

            while True:
                choice = input("\nReconnect to this pod? (y/n): ").strip().lower()

                if choice == "y":
                    _active_pod_id = existing_pod_id
                    print("\nReconnecting...")
                    wait_for_vllm_ready(existing_pod_id)
                    print("\n=== Reconnected ===\n")
                    run_chat(existing_pod_id)
                    return

                if choice == "n":
                    print("\nStarting new pod instead...")
                    print("(Previous pod will keep running - use RunPod console to stop it)")
                    break

                print("Please enter 'y' or 'n'")

        # Launch new pod
        pod_id, _base_url = launch_pod()
        _active_pod_id = pod_id

        # Wait for vLLM to be ready
        wait_for_vllm_ready(pod_id)

        # Start chat
        print("\n=== Pod Ready ===\n")
        run_chat(pod_id)

    except KeyboardInterrupt:
        # Check if we have a pod running that needs cleanup
        existing_pod = check_existing_pod()
        if existing_pod:
            _handle_exit(existing_pod)
        else:
            print("\n\nInterrupted by user")
        sys.exit(1)

    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
