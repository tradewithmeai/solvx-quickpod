#!/usr/bin/env python3
"""
SolvX QuickPod - Master Icon Generator

Creates a 1024x1024 master icon programmatically using the brand colors.
Then run create_icons.py to generate all required sizes.

Usage:
    pip install Pillow
    python demo/generate_master_icon.py
    python demo/create_icons.py icons/icon-master-1024.png icons
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Please install Pillow: pip install Pillow")
    sys.exit(1)


# Brand colors from RELEASE_CHECKLIST.md
DEEP_BLUE = "#1a365d"
BRIGHT_BLUE = "#3182ce"
ORANGE = "#ed8936"
SILVER = "#a0aec0"
WHITE = "#ffffff"


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def create_master_icon(output_path: str = "icons/icon-master-1024.png") -> None:
    """
    Create a 1024x1024 master icon with SQ initials.

    Design:
    - Full bleed rounded rectangle (no padding)
    - MASSIVE "SQ" text filling nearly all the space
    - Clean, bold look that scales well
    """
    size = 1024
    corner_radius = 120  # Smaller corners to maximize space

    # Create image with transparent background
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Draw rounded rectangle background - FULL BLEED
    bg_color = hex_to_rgb(DEEP_BLUE)
    draw.rounded_rectangle(
        [0, 0, size, size],
        radius=corner_radius,
        fill=bg_color,
    )

    # Try to load a BOLD font - start with massive size
    font_size = 750  # Much bigger!
    try:
        # Prefer bold/black fonts for maximum visibility
        font_paths = [
            "C:/Windows/Fonts/arialbd.ttf",  # Arial Bold
            "C:/Windows/Fonts/ariblk.ttf",   # Arial Black
            "C:/Windows/Fonts/impact.ttf",   # Impact
            "C:/Windows/Fonts/segoeuib.ttf", # Segoe UI Bold
            "C:/Windows/Fonts/arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
        font = None
        for font_path in font_paths:
            if Path(font_path).exists():
                font = ImageFont.truetype(font_path, font_size)
                break
        if font is None:
            font = ImageFont.load_default()
            font_size = 100
    except Exception:
        font = ImageFont.load_default()
        font_size = 100

    # Draw "SQ" text
    text = "SQ"
    text_color = hex_to_rgb(ORANGE)

    # Calculate text position (centered)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (size - text_width) // 2
    y = (size - text_height) // 2 - 20

    # Draw text - no shadow, just solid color for maximum clarity
    draw.text((x, y), text, font=font, fill=text_color)

    # Create output directory
    output_dir = Path(output_path).parent
    output_dir.mkdir(exist_ok=True)

    # Save the master icon
    img.save(output_path, "PNG")
    print(f"Created master icon: {output_path}")
    print(f"Size: {size}x{size}")
    print(f"\nNext step: python demo/create_icons.py {output_path} icons")


def main() -> None:
    """Generate the master icon."""
    output_path = "icons/icon-master-1024.png"

    if len(sys.argv) > 1:
        output_path = sys.argv[1]

    create_master_icon(output_path)


if __name__ == "__main__":
    main()
