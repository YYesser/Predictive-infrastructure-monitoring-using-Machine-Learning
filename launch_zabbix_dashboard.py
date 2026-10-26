from __future__ import annotations

import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

WORKDIR = Path(__file__).resolve().parent
HOST = "127.0.0.1"
PORT = 8501
URL = f"http://{HOST}:{PORT}"
MARKER_FILE = WORKDIR / ".dashboard_opened"


def is_port_open(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.8)
            return sock.connect_ex((host, port)) == 0
    except OSError:
        return False


def get_python_executable() -> Path:
    pythonw = WORKDIR / ".venv" / "Scripts" / "pythonw.exe"
    if pythonw.exists():
        return pythonw
    return Path(sys.executable)


def start_streamlit() -> subprocess.Popen:
    python_exe = get_python_executable()
    app_file = WORKDIR / "app.py"
    command = [
        str(python_exe),
        "-m",
        "streamlit",
        "run",
        str(app_file),
        "--server.port",
        str(PORT),
        "--server.headless",
        "true",
        "--server.maxUploadSize",
        "10000",
    ]
    return subprocess.Popen(
        command,
        cwd=str(WORKDIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        close_fds=True,
    )


def wait_for_server(timeout: int = 20) -> bool:
    for _ in range(timeout * 2):
        if is_port_open(HOST, PORT):
            return True
        time.sleep(0.5)
    return False


def main() -> None:
    if is_port_open(HOST, PORT):
        if MARKER_FILE.exists():
            return
        webbrowser.open(URL, new=0, autoraise=True)
        MARKER_FILE.write_text("opened")
        return

    process = start_streamlit()
    if wait_for_server():
        webbrowser.open(URL, new=0, autoraise=True)
        MARKER_FILE.write_text("opened")
        return

    print("Failed to start the dashboard server.")
    try:
        process.kill()
    except Exception:
        pass


if __name__ == "__main__":
    main()
