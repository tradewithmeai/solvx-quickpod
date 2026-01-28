#!/usr/bin/env python3
"""
SolvX QuickPod - Entry Point

A streamlined CLI application for running AI chat sessions on RunPod cloud GPUs.
This module serves as the application entry point.

Usage:
    python -m solvx_quickpod.main
"""

from solvx_quickpod.ai import main as ai_main


def main() -> None:
    """Launch the SolvX QuickPod application."""
    ai_main()


if __name__ == "__main__":
    main()
