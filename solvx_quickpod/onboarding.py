#!/usr/bin/env python3
"""
SolvX QuickPod - Onboarding Module

Handles first-run setup for new users, including:
- RunPod account signup guidance
- API key configuration
- Server password setup
- Desktop shortcut creation

Configuration is stored in ~/.myai/.env for persistence across sessions.
"""

from __future__ import annotations

import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Tuple

# =============================================================================
# CONSTANTS
# =============================================================================

# RunPod URLs with referral tracking
SIGNUP_URL: str = "https://runpod.io?ref=q04x36mf"
SETTINGS_URL: str = "https://www.runpod.io/console/user/settings"

# Minimum length for API key validation
MIN_API_KEY_LENGTH: int = 10


# =============================================================================
# PATH MANAGEMENT
# =============================================================================

def get_env_path() -> Path:
    """
    Get the path to the environment configuration file.

    The .env file is stored in ~/.myai/ to ensure consistent behavior
    regardless of where the executable is located.

    Returns:
        Path to the .env file (~/.myai/.env)
    """
    myai_dir = Path.home() / ".myai"
    myai_dir.mkdir(exist_ok=True)
    return myai_dir / ".env"


# =============================================================================
# FIRST RUN DETECTION
# =============================================================================

def check_first_run() -> bool:
    """
    Determine if onboarding is required.

    Onboarding is needed when:
    - The .env file does not exist
    - Required keys (RUNPOD_API_KEY, VLLM_API_KEY) are missing or invalid

    Returns:
        True if onboarding should be run, False otherwise.
    """
    env_path = get_env_path()

    if not env_path.exists():
        return True

    try:
        content = env_path.read_text(encoding="utf-8")
        has_runpod = False
        has_vllm = False

        for line in content.splitlines():
            line = line.strip()

            # Skip comments and empty lines
            if line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            # Check for valid (non-placeholder) values
            if key == "RUNPOD_API_KEY" and value and not value.startswith("your_"):
                has_runpod = True
            if key == "VLLM_API_KEY" and value and not value.startswith("your_"):
                has_vllm = True

        return not (has_runpod and has_vllm)

    except (IOError, OSError):
        return True


# =============================================================================
# ONBOARDING FLOW
# =============================================================================

def run_onboarding() -> Tuple[str, str]:
    """
    Execute the interactive onboarding flow.

    Guides the user through:
    1. RunPod account creation (optional browser launch)
    2. API key retrieval (optional browser launch)
    3. API key input
    4. Server password creation

    Returns:
        Tuple of (runpod_api_key, vllm_api_key)
    """
    _print_welcome()
    _handle_signup()
    _handle_api_key_instructions()

    runpod_key = _get_api_key()
    vllm_key = _get_password()

    return runpod_key, vllm_key


def _print_welcome() -> None:
    """Display the welcome message."""
    print("\n" + "=" * 50)
    print("       Welcome to SolvX QuickPod")
    print("=" * 50)
    print("\nChat with AI using RunPod's cloud GPUs.")
    print("\nYou'll need a RunPod account and API key to continue.")
    print("New users get $5 FREE credit when adding $10.")


def _handle_signup() -> None:
    """Offer to open the RunPod signup page."""
    print("\n" + "-" * 50)
    response = input("Open RunPod signup page? (y/n): ").strip().lower()

    if response == "y":
        webbrowser.open(SIGNUP_URL)
        print("Signup page opened.")
        input("\nPress Enter when ready to continue...")


def _handle_api_key_instructions() -> None:
    """Display API key instructions and offer to open settings page."""
    print("\n" + "-" * 50)
    print("To get your API key:")
    print("  1. Go to RunPod Settings > API Keys")
    print("  2. Click 'Create API Key'")
    print("  3. Name it anything, select 'All' permissions")
    print("  4. Copy the key")

    response = input("\nOpen RunPod settings page? (y/n): ").strip().lower()

    if response == "y":
        webbrowser.open(SETTINGS_URL)
        print("Settings page opened.")
    else:
        # Check if user might already have an API key configured
        env_path = get_env_path()
        has_existing_key = False

        if env_path.exists():
            try:
                content = env_path.read_text(encoding="utf-8")
                for line in content.splitlines():
                    if line.strip().startswith("RUNPOD_API_KEY="):
                        value = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if value and not value.startswith("your_"):
                            has_existing_key = True
                            break
            except Exception:
                pass

        if not has_existing_key:
            print("\n" + "-" * 50)
            print("No worries! If you already created an API key but didn't")
            print("copy it, the key won't be shown again on the site.")
            print("\nTo create a new one:")
            print("  1. Visit: https://www.runpod.io/console/user/settings")
            print("  2. Scroll down to the 'API Keys' section")
            print("  3. Click 'Create API Key'")
            print("  4. Copy the new key")
            input("\nPress Enter when you have your API key ready...")


def _get_api_key() -> str:
    """
    Prompt for and validate the RunPod API key.

    Returns:
        The validated API key string.
    """
    print("\n" + "-" * 50)

    while True:
        api_key = input("Paste your RunPod API key: ").strip()

        if not api_key:
            print("API key cannot be empty. Please try again.")
            continue

        if len(api_key) < MIN_API_KEY_LENGTH:
            print("That doesn't look like a valid API key. Please try again.")
            continue

        return api_key


def _get_password() -> str:
    """
    Prompt for the vLLM server password.

    Returns:
        The password string.
    """
    print("\n" + "-" * 50)
    print("Create a connection key for your AI server.")
    print("\nThis is just to secure your connection to the server -")
    print("it's stored locally and you won't need to remember it.")
    print("\nFeel free to mash your keyboard to create a random string!")
    print("(Example: kj3h4kj5h4 or anything you like)")

    while True:
        password = input("\nEnter connection key: ").strip()

        if not password:
            print("Connection key cannot be empty. Please try again.")
            continue

        if len(password) < 4:
            print("Please enter at least 4 characters.")
            continue

        return password


# =============================================================================
# CONFIGURATION PERSISTENCE
# =============================================================================

def save_env_file(runpod_key: str, vllm_key: str) -> None:
    """
    Save API credentials to the .env file.

    Args:
        runpod_key: The RunPod API key.
        vllm_key: The vLLM server password.
    """
    env_path = get_env_path()

    content = f"""# SolvX QuickPod Configuration
# Generated during first-run setup

RUNPOD_API_KEY={runpod_key}
VLLM_API_KEY={vllm_key}
"""

    env_path.write_text(content, encoding="utf-8")
    print("\nConfiguration saved!")

    # Offer to create desktop shortcut
    _offer_desktop_shortcut()


# =============================================================================
# DESKTOP SHORTCUT CREATION
# =============================================================================

def _offer_desktop_shortcut() -> None:
    """Offer to create a desktop shortcut for the application."""
    if sys.platform != "win32":
        return  # Only supported on Windows currently

    print("\n" + "-" * 50)
    response = input("Create desktop shortcut? (y/n): ").strip().lower()

    if response == "y":
        if _create_desktop_shortcut():
            print("Desktop shortcut created!")
        else:
            print("Could not create shortcut automatically.")
            print("You can create one manually by right-clicking the .exe")


def _get_desktop_path() -> Path:
    """
    Get the actual Desktop folder path, handling OneDrive redirection.

    Returns:
        Path to the user's Desktop folder.
    """
    try:
        # Use PowerShell to get the correct desktop path (handles OneDrive)
        result = subprocess.run(
            ["powershell", "-Command", "[Environment]::GetFolderPath('Desktop')"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return Path(result.stdout.strip())
    except Exception:
        pass

    # Fallback to standard path
    return Path.home() / "Desktop"


def _create_desktop_shortcut() -> bool:
    """
    Create a Windows desktop shortcut for the application.

    Returns:
        True if successful, False otherwise.
    """
    try:
        # Get the executable path
        if getattr(sys, "frozen", False):
            exe_path = Path(sys.executable)
        else:
            # Running as script - find the exe in dist/
            exe_path = Path(__file__).parent.parent / "dist" / "solvx-quickpod.exe"
            if not exe_path.exists():
                return False

        desktop = _get_desktop_path()
        shortcut_path = desktop / "SolvX QuickPod.lnk"

        # Try to find the icon
        icon_path = None
        if getattr(sys, "frozen", False):
            # Icon is embedded in exe when frozen
            icon_path = exe_path
        else:
            # Development mode - use icons folder
            icon_candidate = Path(__file__).parent.parent / "icons" / "favicon.ico"
            if icon_candidate.exists():
                icon_path = icon_candidate

        # PowerShell script to create shortcut
        ps_script = f'''
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("{shortcut_path}")
$Shortcut.TargetPath = "{exe_path}"
$Shortcut.WorkingDirectory = "{exe_path.parent}"
$Shortcut.Description = "AI Chat on RunPod Cloud GPUs"
'''

        if icon_path:
            ps_script += f'$Shortcut.IconLocation = "{icon_path}"\n'

        ps_script += "$Shortcut.Save()"

        result = subprocess.run(
            ["powershell", "-Command", ps_script],
            capture_output=True,
            text=True,
        )

        return result.returncode == 0 and shortcut_path.exists()

    except Exception:
        return False
