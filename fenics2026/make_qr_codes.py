# pip install qrcode-artistic
import segno

import tempfile
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


def generate_qr_with_text(text: str, url: str):
    bg_size = 600
    bg_img = Image.new("RGBA", (bg_size, bg_size), "white")
    draw = ImageDraw.Draw(bg_img)

    # Set up the text and font
    font = ImageFont.load_default(size=78)

    # Calculate text size to perfectly center it in the 600x600 box
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    text_width = right - left
    text_height = bottom - top

    x = (bg_size - text_width) // 2
    y = (bg_size - text_height) // 2

    # Draw the text using the orange RGB tuple
    draw.text((x, y), text, fill=(241, 90, 34), font=font)

    # Save the intermediate image
    with tempfile.NamedTemporaryFile(suffix=".png") as temp_file:
        bg_img.save(temp_file.name)
        temp_file.seek(0)

        qrcode = segno.make_qr(url, error="h", version=15)
        cwd = Path.cwd() / "qr_codes"
        cwd.mkdir(exist_ok=True)
        # Blend the QR code with the intermediate image
        qrcode.to_artistic(
            background=temp_file.name,
            target=(cwd / f"{text}_qr.png").as_posix(),
            scale=10,
            dark="black",  # The color of the QR code's data modules
        )


projects = [
    ("scifem", "https://scientificcomputing.github.io/scifem"),
    ("Irksome", "https://github.com/firedrakeproject/Irksome/pull/234"),
    ("FEniCSx_ii", "https://scientificcomputing.github.io/fenicsx_ii"),
    ("DOLFINx_adjoint", "https://scientificcomputing.github.io/dolfinx-adjoint"),
    ("io4dolfinx", "https://scientificcomputing.github.io/io4dolfinx"),
    ("networks_FEniCSx", "http://scientificcomputing.github.io/networks_fenicsx"),
    ("FEniCSx_JAX", "https://github.com/scientificcomputing/fenicsx-jax"),
]


# 1. Generate the QR code with High error correction ('h')
for text, url in projects:
    generate_qr_with_text(text, url)
