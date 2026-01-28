#!/usr/bin/env python3
"""
Create a proper Windows ICO file with multiple embedded sizes.

This writes the ICO binary format directly to ensure Windows
recognizes all embedded sizes correctly.
"""

from __future__ import annotations

import struct
from io import BytesIO
from pathlib import Path

from PIL import Image


def create_ico_from_png(master_png: str, output_ico: str) -> None:
    """
    Create a proper multi-resolution ICO file from a master PNG.

    Args:
        master_png: Path to the source PNG (should be 256x256 or larger)
        output_ico: Path for the output ICO file
    """
    # Load master image
    master = Image.open(master_png)
    if master.mode != "RGBA":
        master = master.convert("RGBA")

    # Sizes to include (Windows standard sizes)
    # 256x256 is stored as PNG, smaller sizes as BMP
    sizes = [256, 128, 64, 48, 32, 24, 16]

    images_data = []

    for size in sizes:
        # Resize with high quality
        img = master.resize((size, size), Image.Resampling.LANCZOS)

        # For 256x256, store as PNG (better quality, smaller size)
        # For smaller sizes, also use PNG for best quality
        buf = BytesIO()
        img.save(buf, format="PNG")
        png_data = buf.getvalue()

        images_data.append({
            "size": size,
            "data": png_data,
            "width": size if size < 256 else 0,  # 0 means 256
            "height": size if size < 256 else 0,
        })

    # Build ICO file
    # ICO Header: 6 bytes
    # - Reserved: 2 bytes (0)
    # - Type: 2 bytes (1 for ICO)
    # - Count: 2 bytes (number of images)

    num_images = len(images_data)
    header = struct.pack("<HHH", 0, 1, num_images)

    # Calculate offsets
    # Each directory entry is 16 bytes
    dir_size = 16 * num_images
    data_offset = 6 + dir_size  # Header + directory entries

    directory = b""
    image_data = b""

    for img_info in images_data:
        # Directory entry: 16 bytes
        # - Width: 1 byte (0 = 256)
        # - Height: 1 byte (0 = 256)
        # - Colors: 1 byte (0 = no palette)
        # - Reserved: 1 byte (0)
        # - Planes: 2 bytes (1)
        # - Bit count: 2 bytes (32 for RGBA)
        # - Size: 4 bytes
        # - Offset: 4 bytes

        width = img_info["width"]
        height = img_info["height"]
        data = img_info["data"]

        entry = struct.pack(
            "<BBBBHHII",
            width,      # Width (0 = 256)
            height,     # Height (0 = 256)
            0,          # Color palette
            0,          # Reserved
            1,          # Color planes
            32,         # Bits per pixel
            len(data),  # Size of image data
            data_offset + len(image_data),  # Offset to image data
        )

        directory += entry
        image_data += data

    # Write ICO file
    with open(output_ico, "wb") as f:
        f.write(header)
        f.write(directory)
        f.write(image_data)

    print(f"Created {output_ico} with {num_images} sizes: {sizes}")
    print(f"File size: {Path(output_ico).stat().st_size:,} bytes")


def main():
    master_png = "icons/icon-master-1024.png"
    output_ico = "icons/favicon.ico"

    create_ico_from_png(master_png, output_ico)


if __name__ == "__main__":
    main()
