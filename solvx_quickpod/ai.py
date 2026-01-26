#!/usr/bin/env python3

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())  # Find and load .env file (searches up directory tree)

import os
import sys
import time
import json
import requests
import httpx
from rich.console import Console
import importlib

from solvx_quickpod.launcher import create_pod, wait_for_running, wait_for_proxy, write_state, clear_state, is_pod_running, terminate_pod
from solvx_quickpod import storage


# ==================== ENVIRONMENT CHECKS ====================

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
VLLM_API_KEY = os.getenv("VLLM_API_KEY")

if not RUNPOD_API_KEY:
    print("[ERROR] RUNPOD_API_KEY is not set")
    sys.exit(1)

if not VLLM_API_KEY:
    print("[ERROR] VLLM_API_KEY is not set")
    sys.exit(1)

# ==================== CONSTANTS ====================

# Hardcoded configuration
GPU_TYPE = "NVIDIA GeForce RTX 3090"
GPU_COUNT = 1
MODEL_PATH = "/workspace/models/mistral-7b-instruct-awq"
REFERRAL_LINK = "https://runpod.io?ref=q04x36mf"

console = Console()
MAX_TURNS = 10
SYSTEM_PROMPT = "You are a helpful AI assistant."
TEMPERATURE = 0.5


# ==================== POD MANAGEMENT ====================

def check_existing_pod():
    """Check if there's a running pod we can reconnect to."""
    from solvx_quickpod import config
    importlib.reload(config)

    if not config.POD_ID:
        return None

    if is_pod_running(config.POD_ID):
        return config.POD_ID

    # Pod exists in state but isn't running - clear stale state
    clear_state()
    return None


def launch_pod():
    """Launch pod using launcher.py functions and return pod details."""
    print("\n=== Starting Pod ===")
    print(f"GPU: {GPU_TYPE}")
    print(f"Model: Mistral-7B")

    pod_id = create_pod(GPU_TYPE, GPU_COUNT)
    if not pod_id:
        print("[ERROR] Failed to create pod")
        sys.exit(1)

    write_state(pod_id, MODEL_PATH)
    wait_for_running(pod_id)

    base_url = f"https://{pod_id}-8000.proxy.runpod.net"

    return pod_id, base_url


def wait_for_vllm_ready(pod_id):
    """Wait for vLLM API to be fully ready by checking /v1/models endpoint."""
    print("Waiting for vLLM API to be ready...")
    check_count = 0

    while True:
        try:
            r = requests.get(
                f"https://{pod_id}-8000.proxy.runpod.net/v1/models",
                headers={"Authorization": f"Bearer {VLLM_API_KEY}"},
                timeout=10
            )
            if r.status_code == 200:
                print("vLLM API ready ✓")
                return
        except requests.RequestException:
            pass

        # Check pod status every 10 iterations (~30 seconds)
        check_count += 1
        if check_count >= 10:
            check_count = 0
            if not is_pod_running(pod_id):
                print("\nPod terminated. Exiting.")
                sys.exit(0)

        time.sleep(3)


# ==================== CHAT FUNCTIONS ====================

def trim_history(messages, max_turns=10):
    """Trim history based on max_turns. None = keep all."""
    if max_turns is None:
        return messages  # Full history

    max_messages = 1 + max_turns * 2
    if len(messages) > max_messages:
        return [messages[0]] + messages[-(max_messages - 1):]
    return messages


def run_chat(pod_id):
    """Run the interactive chat interface with streaming responses."""
    from solvx_quickpod import config
    importlib.reload(config)

    # Initialize storage session
    session_id = storage.new_session()
    storage.touch_user()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    storage.log_message(session_id, "system", SYSTEM_PROMPT)

    # Pod check timing
    last_pod_check = time.time()
    POD_CHECK_INTERVAL = 10  # seconds

    console.print(f"[dim]Temperature: {TEMPERATURE}[/dim]")
    console.print(f"[dim]History: Last {MAX_TURNS} turns[/dim]\n")
    console.print("[bold]Chat started. Commands: /stop, /help. Ctrl+C to exit.[/bold]\n")

    while True:
        try:
            # Check pod status periodically
            if time.time() - last_pod_check > POD_CHECK_INTERVAL:
                if not is_pod_running(pod_id):
                    console.print("\n[dim]Pod terminated. Exiting.[/dim]")
                    sys.exit(0)
                last_pod_check = time.time()

            user_input = console.input("[bold cyan]You > [/bold cyan]").strip()
            if not user_input:
                continue

            # Show command help
            if user_input.lower() == "/help":
                console.print("[dim]Available commands:[/dim]")
                console.print("[dim]  /stop - Terminate pod and exit[/dim]")
                console.print("[dim]  /help - Show this help[/dim]\n")
                continue

            # Handle /stop command
            if user_input.lower() == "/stop":
                confirm = input("Terminate pod? This will stop billing. (y/n): ").strip().lower()
                if confirm == 'y':
                    if terminate_pod(pod_id):
                        console.print("[bold]Pod terminated.[/bold]")
                        sys.exit(0)
                    else:
                        console.print("[red]Failed to terminate pod.[/red]")
                continue

            messages.append({"role": "user", "content": user_input})
            storage.log_message(session_id, "user", user_input)
            messages = trim_history(messages, MAX_TURNS)

            # Start timing from prompt send
            request_start = time.time()

            payload = {
                "model": config.MODEL,
                "messages": messages,
                "temperature": TEMPERATURE,
                "max_tokens": 500,
                "stream": True,
            }

            headers = {
                "Authorization": f"Bearer {config.API_KEY}",
                "Content-Type": "application/json",
            }

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
                        console.print(
                            f"\n[bold red]HTTP {response.status_code}[/bold red]"
                        )
                        continue

                    for line in response.iter_lines():
                        if not line:
                            continue

                        if not line.startswith("data: "):
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
                        except Exception:
                            continue

            elapsed = time.time() - request_start
            console.print(f"\n[dim]({elapsed:.1f}s)[/dim]\n")

            if assistant_text:
                messages.append(
                    {"role": "assistant", "content": assistant_text}
                )
                storage.log_message(session_id, "assistant", assistant_text)
                messages = trim_history(messages, MAX_TURNS)

        except KeyboardInterrupt:
            console.print("\n[bold]Session ended.[/bold]")
            sys.exit(0)

        except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError) as e:
            # Connection error - check if pod is still running
            if not is_pod_running(pod_id):
                console.print("\n[dim]Pod terminated. Exiting.[/dim]")
                sys.exit(0)
            else:
                console.print(f"\n[bold red]Connection error:[/bold red] {e}")
                continue


# ==================== MAIN ORCHESTRATION ====================

def main():
    """Main function orchestrating the complete user flow."""
    try:
        # Welcome message with referral
        print("\n=== SolvX QuickPod ===")
        print(f"Don't have RunPod? Get $5 free credit: {REFERRAL_LINK}\n")

        # Step 1: Check for existing running pod
        existing_pod_id = check_existing_pod()
        if existing_pod_id:
            print(f"=== Existing Pod Found ===")
            print(f"Pod ID: {existing_pod_id}")

            while True:
                choice = input("\nReconnect to this pod? (y/n): ").strip().lower()
                if choice == 'y':
                    print("\nReconnecting...")
                    wait_for_vllm_ready(existing_pod_id)
                    print("\n=== Reconnected ===\n")
                    run_chat(existing_pod_id)
                    return
                elif choice == 'n':
                    print("\nStarting new pod instead...")
                    break
                else:
                    print("Please enter 'y' or 'n'")

        # Step 2: Launch Pod (no menus, just launch)
        pod_id, base_url = launch_pod()

        # Step 3: Wait for vLLM API to be fully ready
        wait_for_vllm_ready(pod_id)

        # Step 4: Start chat
        print("\n=== Pod Ready ===\n")
        run_chat(pod_id)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
