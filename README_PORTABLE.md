# Portable Project Launcher

This project can be launched on Windows with a desktop shortcut and a portable virtual environment.

## Files added for portability

- `start_dashboard.bat` — starts the Streamlit dashboard using the repo-local Python runtime.
- `start_pipeline.bat` — starts the continuous Zabbix pipeline in the background.
- `stop_pipeline.bat` — stops the continuous Zabbix pipeline.
- `create_shortcut.ps1` — creates a desktop shortcut to `start_dashboard.bat` from the repo root.
- `create_desktop_shortcut.py` — alternative shortcut creator using Python.

## How to use

1. Create and activate the virtual environment if you have not already:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. Create the desktop shortcut:

   ```powershell
   .\create_shortcut.ps1
   ```

3. Start the dashboard from the shortcut on your desktop.

4. Use `start_pipeline.bat` to start the background pipeline.
5. Use `stop_pipeline.bat` to stop it.

## Notes

- These launchers preserve the existing project behavior.
- They use relative paths so the project can be moved to another folder.
- `start_dashboard.bat` uses `pythonw.exe` if available to avoid a console window.
