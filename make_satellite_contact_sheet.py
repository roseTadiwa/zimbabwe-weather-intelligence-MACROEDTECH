
from pathlib import Path
from PIL import Image, ImageDraw
import math

folder = Path("reports/satellite")

files = sorted(
    p for p in folder.iterdir()
    if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
)

thumb_width = 420
thumb_height = 300
label_height = 55
columns = 2
rows = math.ceil(len(files) / columns)

sheet = Image.new(
    "RGB",
    (columns * thumb_width, rows * (thumb_height + label_height)),
    "white"
)

draw = ImageDraw.Draw(sheet)

for i, path in enumerate(files):
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail((thumb_width - 20, thumb_height - 20))

    x = (i % columns) * thumb_width
    y = (i // columns) * (thumb_height + label_height)

    sheet.paste(image, (x + 10, y + 10))
    draw.text(
        (x + 10, y + thumb_height),
        path.name,
        fill="black"
    )

output = folder / "satellite_contact_sheet.jpg"
sheet.save(output, quality=90)

print("Images included:", len(files))
print("Contact sheet saved:", output.resolve())
