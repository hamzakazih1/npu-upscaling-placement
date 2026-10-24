"""Draw the app icon and write PadMIDI.ico / PadMIDI.icns (needs Pillow)."""

import sys
from pathlib import Path

from PIL import Image, ImageDraw

out = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "build")
out.mkdir(parents=True, exist_ok=True)

S = 1024
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
d.rounded_rectangle((64, 64, S - 64, S - 64), radius=210, fill=(61, 155, 240, 255))
bar_w, gap, base = 112, 52, S - 250
for i, h in enumerate((220, 400, 580, 330)):
    x = 236 + i * (bar_w + gap)
    d.rounded_rectangle((x, base - h, x + bar_w, base), radius=48, fill=(255, 255, 255, 255))

img.save(out / "PadMIDI.png")
img.save(out / "PadMIDI.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
img.save(out / "PadMIDI.icns")
print("icons written to", out)
