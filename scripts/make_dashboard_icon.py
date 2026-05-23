"""Generate the desktop-shortcut icon for the Dashboard.

Draws a candlestick chart on a dark navy background and writes a
multi-resolution .ico file at assets/app_icon.ico.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "assets" / "app_icon.ico"

SIZE = 256
BG = (11, 18, 32)
GRID = (28, 38, 58)
GREEN = (34, 197, 94)
RED = (239, 68, 68)

CANDLES = [
    dict(cx=40,  high=110, low=220, open_y=130, close_y=200, color=RED),
    dict(cx=84,  high=80,  low=200, open_y=180, close_y=110, color=GREEN),
    dict(cx=128, high=50,  low=170, open_y=150, close_y=70,  color=GREEN),
    dict(cx=172, high=30,  low=140, open_y=120, close_y=50,  color=GREEN),
    dict(cx=216, high=20,  low=130, open_y=40,  close_y=110, color=RED),
]

BODY_W = 24
WICK_W = 4


def render() -> Image.Image:
    img = Image.new("RGBA", (SIZE, SIZE), BG)
    draw = ImageDraw.Draw(img)

    for y in (60, 120, 180):
        draw.line([(12, y), (SIZE - 12, y)], fill=GRID, width=1)

    for c in CANDLES:
        cx = c["cx"]
        wx0, wx1 = cx - WICK_W // 2, cx + WICK_W // 2
        bx0, bx1 = cx - BODY_W // 2, cx + BODY_W // 2
        draw.rectangle([wx0, c["high"], wx1, c["low"]], fill=c["color"])
        top, bot = sorted([c["open_y"], c["close_y"]])
        draw.rectangle([bx0, top, bx1, bot], fill=c["color"])

    return img


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    img = render()
    img.save(
        OUT_PATH,
        format="ICO",
        sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
