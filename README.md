# Zabbix ML Dashboard

Windows Streamlit dashboard for training lightweight 7-day-ahead LightGBM
forecasting models from Zabbix metric history.

> **Repository hygiene:** `users.json`, `.env`, virtual environments, generated
> models/reports, and local Zabbix data are intentionally excluded from Git.
> Never commit real credentials, API tokens, personal email addresses, or
> exported monitoring data.

Purpose
-------
This project trains lightweight 7-day-ahead LightGBM regression models on
Zabbix metric history and exposes training reports, validation plots and
short-term forecasts via a Streamlit dashboard.

Quickstart
----------
1. Create and activate a Python virtual environment (Windows example):

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Configure optional email settings:

   ```powershell
   Copy-Item .env.example .env
   ```

   Edit `.env` only on your local machine. The dashboard creates a local
   `users.json` automatically on first start with the initial credentials
   `admin` / `admin`; change the password immediately.

4. Train models (process may take time for large datasets):

   ```powershell
   python zabbix_pipeline.py
   ```

5. Start the dashboard (or use the launcher):

   ```powershell
   python launch_zabbix_dashboard.py
   ```

Files & folders
---------------
- `zabbix_pipeline.py` — ETL, feature engineering, model training and report/plot generation.
- `app.py` — Streamlit dashboard that loads `reports/` and `models/` to show results and forecasts.
- `data/hosts/` — per-host CSV files used for training and forecasting.
- `models/` — trained model files saved as `<sanitized_item_key>_dayN.txt` per host.
- `reports/` — `training_summary.csv`, `skipped_series.csv`, and `plots/`.
- `users.json` — local user store used by the dashboard; generated locally and
  excluded from the repository.

Configuration
-------------
You can provide optional environment values in a `.env` file at the project root. Supported variables:

- `SMTP_SERVER`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`, `EMAIL_FROM` — email settings for notification of new users.
- `ZABBIX_DASHBOARD_USER`, `ZABBIX_DASHBOARD_PASS` — override default dashboard credentials.

Notes & next steps
------------------
- The admin "Train Your Own" UI is a placeholder; automation (scheduling retraining) and CI/tests are not included.
- Paths in `reports/training_summary.csv` are absolute Windows paths — consider making them relative for portability.

If you want, I can add a scheduled retraining scaffold, CI checks, or a packaged installer next.
