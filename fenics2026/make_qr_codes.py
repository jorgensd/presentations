# pip install qrcode-artistic
import segno

import tempfile
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


def generate_qr_with_text(text: str, url: str, custom_text: str = None):
    bg_size = 600
    bg_img = Image.new("RGBA", (bg_size, bg_size), "white")
    draw = ImageDraw.Draw(bg_img)

    # Set up the text and font
    font = ImageFont.load_default(size=100)
    # 1. Determine the actual text to display FIRST
    display_text = custom_text if custom_text is not None else text

    # 2. Find the exact center of the background
    center_x = bg_size / 2
    center_y = bg_size / 2

    # 3. Draw the text using Pillow's alignment features
    draw.text(
        (center_x, center_y), 
        display_text, 
        fill=(241, 90, 34), 
        font=font, 
        anchor="mm",      # "mm" anchors the exact Middle-Middle of the text to the x,y coordinates
        align="center"    # "center" perfectly aligns multiline text blocks relative to each other
    )

    # Save the intermediate image
    with tempfile.NamedTemporaryFile(suffix=".png") as temp_file:
        bg_img.save(temp_file.name)
        temp_file.seek(0)

        qrcode = segno.make_qr(url, error="h", version=6)
        cwd = Path.cwd() / "qr_codes"
        cwd.mkdir(exist_ok=True)
        
        # Blend the QR code with the intermediate image
        qrcode.to_artistic(
            background=temp_file.name,
            target=(cwd / f"{text}_qr.png").as_posix(),
            scale=5,
            dark="black"
        )

projects = [
    ("scifem", "https://github.com/scientificcomputing/scifem", None),
    ("Irksome", "https://github.com/firedrakeproject/Irksome/", None),
    ("FEniCSx_ii", "https://github.com/scientificcomputing/fenicsx_ii", "FEniCSx_ii"),
    ("DOLFINx_adjoint", "https://github.com/scientificcomputing/dolfinx-adjoint", "DOLFINx\nAdjoint"),
    ("io4dolfinx", "https://github.com/scientificcomputing/io4dolfinx", None),
    ("networks_FEniCSx", "https://github.com/scientificcomputing/networks_fenicsx", "Networks\nFEniCSx"),
    ("FEniCSx_JAX", "https://github.com/scientificcomputing/fenicsx-jax", "FEniCSx\nJAX"),
]


# 1. Generate the QR code with High error correction ('h')
for text, url, custom_text in projects:
    generate_qr_with_text(text, url, custom_text)
