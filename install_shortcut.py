#!/usr/bin/env python3
"""
SolvX QuickPod - Desktop Shortcut Installer

Creates a desktop shortcut for the application with the proper icon.
Works on Windows. Run this after downloading the .exe.

Usage:
    python install_shortcut.py

Or double-click install_shortcut.py if Python is associated.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def get_exe_path() -> Path:
    """Find the solvx-quickpod executable."""
    # Check common locations
    candidates = [
        Path.cwd() / "solvx-quickpod.exe",
        Path.cwd() / "dist" / "solvx-quickpod.exe",
        Path(__file__).parent / "solvx-quickpod.exe",
        Path(__file__).parent / "dist" / "solvx-quickpod.exe",
    ]

    for path in candidates:
        if path.exists():
            return path.resolve()

    return None


def get_icon_path() -> Path:
    """Find the favicon.ico file."""
    candidates = [
        Path.cwd() / "icons" / "favicon.ico",
        Path(__file__).parent / "icons" / "favicon.ico",
    ]

    for path in candidates:
        if path.exists():
            return path.resolve()

    return None


def create_windows_shortcut(exe_path: Path, icon_path: Path | None) -> bool:
    """
    Create a Windows desktop shortcut using PowerShell.

    Args:
        exe_path: Path to the executable
        icon_path: Optional path to the icon file

    Returns:
        True if successful, False otherwise
    """
    try:
        import subprocess

        desktop = Path.home() / "Desktop"
        shortcut_path = desktop / "SolvX QuickPod.lnk"

        # PowerShell script to create shortcut
        ps_script = f'''
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("{shortcut_path}")
$Shortcut.TargetPath = "{exe_path}"
$Shortcut.WorkingDirectory = "{exe_path.parent}"
$Shortcut.Description = "AI Chat on RunPod Cloud GPUs"
'''

        if icon_path and icon_path.exists():
            ps_script += f'$Shortcut.IconLocation = "{icon_path}"\n'

        ps_script += '$Shortcut.Save()'

        result = subprocess.run(
            ["powershell", "-Command", ps_script],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0 and shortcut_path.exists():
            return True
        else:
            print(f"PowerShell error: {result.stderr}")
            return False

    except Exception as e:
        print(f"Error creating shortcut: {e}")
        return False


def create_linux_desktop_entry(exe_path: Path, icon_path: Path | None) -> bool:
    """
    Create a Linux .desktop file.

    Args:
        exe_path: Path to the executable
        icon_path: Optional path to the icon file

    Returns:
        True if successful, False otherwise
    """
    try:
        desktop_dir = Path.home() / ".local" / "share" / "applications"
        desktop_dir.mkdir(parents=True, exist_ok=True)

        desktop_file = desktop_dir / "solvx-quickpod.desktop"

        icon_line = f"Icon={icon_path}\n" if icon_path else ""

        content = f"""[Desktop Entry]
Type=Application
Name=SolvX QuickPod
Comment=AI Chat on RunPod Cloud GPUs
Exec={exe_path}
{icon_line}Terminal=true
Categories=Utility;Development;
"""

        desktop_file.write_text(content)
        desktop_file.chmod(0o755)

        # Also create on Desktop if it exists
        desktop_folder = Path.home() / "Desktop"
        if desktop_folder.exists():
            desktop_shortcut = desktop_folder / "SolvX QuickPod.desktop"
            desktop_shortcut.write_text(content)
            desktop_shortcut.chmod(0o755)

        return True

    except Exception as e:
        print(f"Error creating desktop entry: {e}")
        return False


def main() -> None:
    """Create desktop shortcut for the current platform."""
    print("=== SolvX QuickPod - Shortcut Installer ===\n")

    # Find executable
    exe_path = get_exe_path()
    if not exe_path:
        print("ERROR: Could not find solvx-quickpod.exe")
        print("\nMake sure this script is in the same folder as the .exe")
        print("Or run from the project directory after building.")
        input("\nPress Enter to exit...")
        sys.exit(1)

    print(f"Found executable: {exe_path}")

    # Find icon
    icon_path = get_icon_path()
    if icon_path:
        print(f"Found icon: {icon_path}")
    else:
        print("Icon not found (shortcut will use default icon)")

    # Create shortcut based on platform
    print("\nCreating desktop shortcut...")

    if sys.platform == "win32":
        success = create_windows_shortcut(exe_path, icon_path)
    elif sys.platform.startswith("linux"):
        success = create_linux_desktop_entry(exe_path, icon_path)
    else:
        print(f"Platform {sys.platform} not supported for shortcuts.")
        print(f"You can manually create an alias to: {exe_path}")
        success = False

    if success:
        print("\nDesktop shortcut created successfully!")
        print("You can now launch SolvX QuickPod from your desktop.")
    else:
        print("\nFailed to create shortcut.")
        print(f"You can manually create a shortcut to: {exe_path}")

    input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()
