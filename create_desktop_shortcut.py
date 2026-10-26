from pathlib import Path
from PIL import Image, ImageDraw

repo_root = Path(__file__).resolve().parent
desktop = Path.home() / "Desktop"
batch_file = repo_root / "start_dashboard.bat"
icon_file = repo_root / "Zabbix_Dashboard.ico"
shortcut_file = desktop / "Zabbix Dashboard.lnk"

# Generate a simple dashboard icon
size = (256, 256)
img = Image.new("RGBA", size, (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# Background circle
circle_bounds = (24, 24, 232, 232)
draw.ellipse(circle_bounds, fill=(13, 110, 253, 255))

# Bar chart elements
bars = [((74, 150), (104, 240)), ((114, 110), (144, 240)), ((154, 70), (184, 240)), ((194, 40), (224, 240))]
for top_left, bottom_right in bars:
    draw.rectangle([top_left, bottom_right], fill=(255, 255, 255, 255))

# Chart line
line_points = [(94, 150), (124, 110), (164, 70), (204, 40)]
draw.line(line_points, fill=(255, 255, 255, 255), width=18, joint="curve")
for x, y in line_points:
    draw.ellipse((x - 12, y - 12, x + 12, y + 12), fill=(255, 255, 255, 255))

# Save icon
img.save(icon_file, format="ICO", sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])

# Create Windows shortcut
import pythoncom
from win32com.shell import shell
from win32com.client import Dispatch

shell_obj = Dispatch("WScript.Shell")
shortcut = shell_obj.CreateShortcut(str(shortcut_file))
shortcut.TargetPath = str(batch_file)
shortcut.WorkingDirectory = str(repo_root)
shortcut.IconLocation = str(icon_file)
shortcut.Save()

print(f"Created shortcut: {shortcut_file}")
print(f"Created icon: {icon_file}")
