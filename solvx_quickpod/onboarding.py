#!/usr/bin/env python3
"""
First-run onboarding flow for SolvX QuickPod.
Guides users through RunPod signup and API key configuration.
"""

import os
import sys
import webbrowser
from pathlib import Path


SIGNUP_URL = "https://runpod.io?ref=q04x36mf"
SETTINGS_URL = "https://www.runpod.io/console/user/settings"


def get_env_path() -> Path:
    """Get the path to the .env file. Always uses ~/.myai/.env for consistency."""
    myai_dir = Path.home() / ".myai"
    myai_dir.mkdir(exist_ok=True)
    return myai_dir / ".env"


def check_first_run() -> bool:
    """
    Return True if onboarding is needed.
    Onboarding is needed if .env doesn't exist or is missing required keys.
    """
    env_path = get_env_path()

    if not env_path.exists():
        return True

    # Check if file has required keys
    try:
        content = env_path.read_text()
        has_runpod = False
        has_vllm = False

        for line in content.splitlines():
            line = line.strip()
            if line.startswith('#') or '=' not in line:
                continue

            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key == "RUNPOD_API_KEY" and value and not value.startswith("your_"):
                has_runpod = True
            if key == "VLLM_API_KEY" and value and not value.startswith("your_"):
                has_vllm = True

        return not (has_runpod and has_vllm)
    except Exception:
        return True


def run_onboarding() -> tuple[str, str]:
    """
    Guide user through RunPod signup and API key creation.
    Returns (runpod_api_key, vllm_api_key).
    """
    print("\n" + "=" * 50)
    print("       Welcome to SolvX QuickPod")
    print("=" * 50)
    print("\nChat with AI using RunPod's cloud GPUs.")
    print("\nYou'll need a RunPod account and API key to continue.")
    print("New users get $5 FREE credit when adding $10.")

    # Ask before opening signup page
    print("\n" + "-" * 50)
    open_signup = input("Open RunPod signup page? (y/n): ").strip().lower()
    if open_signup == 'y':
        webbrowser.open(SIGNUP_URL)
        print("Signup page opened.")
        input("\nPress Enter when ready to continue...")

    # Ask before opening settings page
    print("\n" + "-" * 50)
    print("To get your API key:")
    print("  1. Go to RunPod Settings > API Keys")
    print("  2. Click 'Create API Key'")
    print("  3. Name it anything, select 'All' permissions")
    print("  4. Copy the key")

    open_settings = input("\nOpen RunPod settings page? (y/n): ").strip().lower()
    if open_settings == 'y':
        webbrowser.open(SETTINGS_URL)
        print("Settings page opened.")

    # Get RunPod API key
    print("\n" + "-" * 50)
    while True:
        runpod_key = input("Paste your RunPod API key: ").strip()

        if not runpod_key:
            print("API key cannot be empty. Please try again.")
            continue

        if len(runpod_key) < 10:
            print("That doesn't look like a valid API key. Please try again.")
            continue

        break

    # Get VLLM password
    print("\n" + "-" * 50)
    print("Create a password for your AI server.")
    print("(Tip: Use something memorable, like 'myai123')")

    while True:
        vllm_key = input("\nEnter password: ").strip()

        if not vllm_key:
            print("Password cannot be empty. Please try again.")
            continue

        break

    return runpod_key, vllm_key


def save_env_file(runpod_key: str, vllm_key: str):
    """Save API keys to .env file."""
    env_path = get_env_path()

    content = f"""# SolvX QuickPod Configuration
# Generated during first-run setup

RUNPOD_API_KEY={runpod_key}
VLLM_API_KEY={vllm_key}
"""

    env_path.write_text(content)
    print("\nConfiguration saved!")
