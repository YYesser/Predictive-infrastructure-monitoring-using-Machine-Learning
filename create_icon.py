from pathlib import Path
from PIL import Image, ImageDraw

icon_file = Path(r"C:\Users\yasser.rached\Desktop\Zabbix_Dashboard.ico")
icon_file.parent.mkdir(parents=True, exist_ok=True)

img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

draw.ellipse((24, 24, 232, 232), fill=(13, 110, 253, 255))
bars = [((74, 150), (104, 240)), ((114, 110), (144, 240)), ((154, 70), (184, 240)), ((194, 40), (224, 240))]
for top_left, bottom_right in bars:
    draw.rectangle([top_left, bottom_right], fill=(255, 255, 255, 255))

line_points = [(94, 150), (124, 110), (164, 70), (204, 40)]
draw.line(line_points, fill=(255, 255, 255, 255), width=18)
for x, y in line_points:
    draw.ellipse((x - 12, y - 12, x + 12, y + 12), fill=(255, 255, 255, 255))

img.save(icon_file, format="ICO", sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
print(f"Created icon at {icon_file}")
