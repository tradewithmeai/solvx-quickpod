#!/usr/bin/env python3
"""
Icon Generator for SolvX QuickPod

Generates all required icon sizes from a master 1024x1024 PNG.

Usage:
    pip install Pillow
    python demo/create_icons.py path/to/master-icon-1024.png

Output:
    Creates icons/ directory with all sizes
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Please install Pillow: pip install Pillow")
    sys.exit(1)


# Icon sizes needed for different platforms
ICON_SIZES = [
    (16, "icon-16.png"),
    (32, "icon-32.png"),
    (48, "icon-48.png"),
    (64, "icon-64.png"),
    (128, "icon-128.png"),
    (256, "icon-256.png"),
    (512, "icon-512.png"),
    (1024, "icon-1024.png"),
]


def create_icons(master_path: str, output_dir: str = "icons") -> None:
    """
    Generate all icon sizes from master image.

    Args:
        master_path: Path to 1024x1024 master PNG
        output_dir: Output directory for icons
    """
    # Load master image
    master = Image.open(master_path)

    if master.size != (1024, 1024):
        print(f"Warning: Master image is {master.size}, not 1024x1024")
        print("Resizing may affect quality")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"Generating icons in {output_path}/")
    print("-" * 40)

    # Generate each size
    for size, filename in ICON_SIZES:
        resized = master.resize((size, size), Image.Resampling.LANCZOS)
        output_file = output_path / filename
        resized.save(output_file, "PNG")
        print(f"  {filename:20} ({size}x{size})")

    # Create ICO file with multiple sizes
    ico_sizes = [(16, 16), (32, 32), (48, 48), (256, 256)]
    ico_images = [master.resize(size, Image.Resampling.LANCZOS) for size in ico_sizes]

    ico_path = output_path / "favicon.ico"
    ico_images[0].save(
        ico_path,
        format="ICO",
        sizes=ico_sizes,
        append_images=ico_images[1:],
    )
    print(f"  {'favicon.ico':20} (multi-size ICO)")

    print("-" * 40)
    print(f"Done! {len(ICON_SIZES) + 1} files created.")


def main() -> None:
    """Parse arguments and run."""
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nExample:")
        print("  python demo/create_icons.py my-logo-1024.png")
        sys.exit(1)

    master_path = sys.argv[1]

    if not Path(master_path).exists():
        print(f"Error: File not found: {master_path}")
        sys.exit(1)

    output_dir = sys.argv[2] if len(sys.argv) > 2 else "icons"
    create_icons(master_path, output_dir)


if __name__ == "__main__":
    main()
