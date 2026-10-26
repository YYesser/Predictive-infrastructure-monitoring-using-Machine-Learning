from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Callable
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from zabbix_pipeline import MODELS_DIR, REPORTS_DIR, PLOTS_DIR, sanitize_name, split_by_host, train_models

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "pipeline"
RAW_DATA_PATH = DATA_DIR / "zabbix_continuous.csv"
STATE_PATH = DATA_DIR / "pipeline_state.json"
PID_PATH = DATA_DIR / "continuous_pipeline.pid"
LOCK_PATH = DATA_DIR / "continuous_pipeline.lock"
LOG_PATH = DATA_DIR / "continuous_pipeline.log"
DEFAULT_API_URL = os.environ.get("ZABBIX_API_URL", "http://192.168.100.60/zabbix/api_jsonrpc.php")
DEFAULT_API_TOKEN = os.environ.get(
    "ZABBIX_API_TOKEN",
    "c37bbc02ded0e038c8a16c9441a9d1a4f52c5720c49f3c8a4a7e2051f1192830",
)
DEFAULT_METRICS = [
    "CPU",
    "Memory",
    "Disk",
    "Network",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_dirs() -> None:
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def _write_log(message: str) -> None:
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        timestamp = _utc_now().isoformat()
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] {message}\n")
    except Exception:
        pass


def _load_state() -> dict:
    if not STATE_PATH.exists():
        return {"last_run_at": None, "last_timestamp": None}
    try:
        return pd.read_json(STATE_PATH).to_dict(orient="records")[0]
    except Exception:
        return {"last_run_at": None, "last_timestamp": None}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = STATE_PATH.with_suffix(STATE_PATH.suffix + ".tmp")
    pd.DataFrame([state]).to_json(temp_path, orient="records", indent=2)
    temp_path.replace(STATE_PATH)


def _read_pid() -> Optional[int]:
    if not PID_PATH.exists():
        return None
    try:
        return int(PID_PATH.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def _remove_stale_pid() -> None:
    try:
        if PID_PATH.exists():
            PID_PATH.unlink()
    except Exception:
        pass


def _acquire_lock(timeout_seconds: int = 10) -> bool:
    try:
        LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        start = time.time()
        while LOCK_PATH.exists():
            stale_pid = None
            try:
                stale_pid = int(LOCK_PATH.read_text(encoding="utf-8").strip())
            except Exception:
                pass

            if stale_pid is not None and not _is_pid_running(stale_pid):
                try:
                    LOCK_PATH.unlink()
                    _write_log(f"Removed stale pipeline lock from PID {stale_pid}")
                except Exception:
                    pass
                break

            if (time.time() - start) >= timeout_seconds:
                return False
            time.sleep(0.1)

        LOCK_PATH.write_text(str(os.getpid()), encoding="utf-8")
        return True
    except Exception:
        return False


def _release_lock() -> None:
    try:
        if LOCK_PATH.exists():
            LOCK_PATH.unlink()
    except Exception:
        pass


def _is_pid_running(pid: Optional[int]) -> bool:
    if pid is None:
        return False

    if os.name == "nt":
        try:
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            completed = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True,
                text=True,
                check=False,
                startupinfo=startupinfo,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            output = completed.stdout.strip()
            if not output:
                return False
            return "No tasks are running" not in output
        except Exception:
            return False

    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except ProcessLookupError:
        return False
    except OSError:
        return False


def resolve_running_pipeline_pid(
    pid: Optional[int],
    running_pid: Optional[Callable[[Optional[int]], bool]] = None,
    discover_pids: Optional[Callable[[], list[int]]] = None,
) -> Optional[int]:
    running_pid = running_pid or _is_pid_running
    if running_pid(pid):
        return pid

    discover = discover_pids or _discover_pipeline_pids
    for candidate in discover():
        if running_pid(candidate):
            return candidate
    return None


def _discover_pipeline_pids() -> list[int]:
    if os.name != "nt":
        return []

    try:
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE

        completed = subprocess.run(
            [
                "wmic",
                "process",
                "where",
                "CommandLine like '%continuous_zabbix_pipeline.py%'",
                "get",
                "ProcessId",
                "/format:csv",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            startupinfo=startupinfo,
            creationflags=subprocess.CREATE_NO_WINDOW,
            check=False,
        )
        output = completed.stdout or ""
        pids = []
        for line in output.splitlines():
            line = line.strip()
            if not line or line.lower().startswith("node"):
                continue
            try:
                parts = [p.strip() for p in line.split(",") if p.strip()]
                if parts:
                    pids.append(int(parts[-1]))
            except ValueError:
                continue
        return pids
    except Exception:
        return []


def get_pipeline_status() -> dict:
    state = _load_state()
    pid = _read_pid()
    resolved_pid = resolve_running_pipeline_pid(pid)
    return {
        "running": resolved_pid is not None,
        "pid": resolved_pid,
        "last_run_at": state.get("last_run_at"),
        "last_timestamp": state.get("last_timestamp"),
        "last_model_refresh_at": state.get("last_model_refresh_at"),
        "log_path": str(LOG_PATH),
    }


def should_trigger_model_refresh(state: dict, previous_row_count: int, current_row_count: int) -> bool:
    if current_row_count <= previous_row_count:
        return False
    last_refresh = state.get("last_model_refresh_at")
    if not last_refresh:
        return True
    try:
        last_refresh_dt = datetime.fromisoformat(last_refresh)
        return (datetime.now(timezone.utc) - last_refresh_dt).total_seconds() >= 60
    except Exception:
        return True


def start_continuous_pipeline(api_url: str, api_token: str | None = None, metric_keywords: Optional[list[str]] = None, interval_seconds: int = 300) -> dict:
    existing_pid = _read_pid()
    if _is_pid_running(existing_pid):
        return {"started": False, "pid": existing_pid, "message": "Continuous pipeline already running."}

    _remove_stale_pid()
    metrics = metric_keywords or DEFAULT_METRICS
    command = [
        sys.executable,
        str(BASE_DIR / "continuous_zabbix_pipeline.py"),
        "--api-url",
        api_url,
        "--api-token",
        api_token or "",
        "--interval-seconds",
        str(interval_seconds),
        "--metric-keywords",
        *metrics,
    ]
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_handle = LOG_PATH.open("a", encoding="utf-8")
    popen_kwargs = {
        "stdout": log_handle,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
        "text": True,
        "close_fds": True,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW | 0x00000008
        try:
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            popen_kwargs["startupinfo"] = startupinfo
        except Exception:
            pass
        try:
            # Prefer pythonw to avoid a visible console window when launching the background worker
            pythonw_path = Path(sys.executable).with_name("pythonw.exe")
            if pythonw_path.exists():
                command[0] = str(pythonw_path)
        except Exception:
            pass
    else:
        popen_kwargs["start_new_session"] = True

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _write_log("Starting continuous pipeline process")
    log_handle = LOG_PATH.open("a", encoding="utf-8")
    popen_kwargs["stdout"] = log_handle
    popen_kwargs["stderr"] = subprocess.STDOUT

    process = subprocess.Popen(command, **popen_kwargs)
    PID_PATH.write_text(str(process.pid), encoding="utf-8")
    time.sleep(2)
    if not _is_pid_running(process.pid):
        _write_log(f"Background process failed to stay alive after launch: PID {process.pid}")
        return {"started": False, "pid": process.pid, "message": "The background process exited immediately. Check the log file for details."}
    _write_log(f"Background pipeline launched with PID {process.pid}")
    return {"started": True, "pid": process.pid, "message": "Continuous pipeline started."}


def stop_continuous_pipeline() -> dict:
    pid = _read_pid()
    pids_to_stop: set[int] = set()

    if pid is not None and _is_pid_running(pid):
        pids_to_stop.add(pid)

    for discovered_pid in _discover_pipeline_pids():
        if _is_pid_running(discovered_pid):
            pids_to_stop.add(discovered_pid)

    if not pids_to_stop:
        if PID_PATH.exists():
            PID_PATH.unlink(missing_ok=True)
        return {"stopped": False, "pid": pid, "message": "No running continuous pipeline was found."}

    for target_pid in sorted(pids_to_stop):
        try:
            if os.name == "nt":
                # Run taskkill without showing a window
                try:
                    startupinfo = subprocess.STARTUPINFO()
                    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                    subprocess.run(
                        ["taskkill", "/PID", str(target_pid), "/T", "/F"],
                        capture_output=True,
                        text=True,
                        check=False,
                        startupinfo=startupinfo,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                    )
                except Exception:
                    subprocess.run(["taskkill", "/PID", str(target_pid), "/T", "/F"], capture_output=True, text=True, check=False)
            else:
                os.kill(target_pid, 15)
        except Exception:
            pass

    time.sleep(0.5)
    remaining = [p for p in pids_to_stop if _is_pid_running(p)]
    if remaining:
        return {"stopped": False, "pid": pid, "message": f"Could not stop pipeline PIDs: {remaining}."}

    if PID_PATH.exists():
        PID_PATH.unlink(missing_ok=True)
    return {"stopped": True, "pid": pid, "message": "Continuous pipeline stopped."}


def merge_extracted_rows(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """Append only new rows to the existing continuous dataset."""
    if existing.empty and incoming.empty:
        return pd.DataFrame(columns=["host_name", "item_name", "item_key", "timestamp", "value", "value_type"])

    combined = pd.concat([existing, incoming], ignore_index=True)
    if combined.empty:
        return combined

    combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")
    combined = combined.dropna(subset=["timestamp"]).copy()

    deduped = combined.drop_duplicates(
        subset=["host_name", "item_key", "timestamp"],
        keep="last",
    )
    return deduped.sort_values(["host_name", "item_key", "timestamp"]).reset_index(drop=True)


def zabbix_call(method: str, params: dict, api_url: str, api_token: str | None = None, retries: int = 3, backoff_seconds: float = 1.0) -> list | None:
    payload = {"jsonrpc": "2.0", "method": method, "params": params, "id": 1}
    headers = {"Content-Type": "application/json-rpc"}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    for attempt in range(retries):
        try:
            response = requests.post(api_url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            if "error" in data:
                return None
            return data.get("result")
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(backoff_seconds * (attempt + 1))
    return None


def zabbix_get_hosts(api_url: str, api_token: str | None) -> list[dict]:
    result = zabbix_call("host.get", {"output": ["hostid", "host", "status"], "filter": {"status": 0}}, api_url, api_token)
    return result or []


def zabbix_get_items(api_url: str, api_token: str | None, host_id: str, metric_keywords: list[str]) -> list[dict]:
    all_items: list[dict] = []
    for keyword in metric_keywords:
        result = zabbix_call(
            "item.get",
            {"hostids": host_id, "search": {"name": keyword}, "output": ["itemid", "name", "key_", "value_type"]},
            api_url,
            api_token,
        )
        if result:
            all_items.extend(result)

    seen: set[str] = set()
    unique_items: list[dict] = []
    for item in all_items:
        if item.get("itemid") not in seen:
            seen.add(item.get("itemid"))
            unique_items.append(item)
    return unique_items


def zabbix_get_history(api_url: str, api_token: str | None, item_id: str, value_type: int, time_from: int, time_till: int) -> list[dict]:
    all_history: list[dict] = []
    current_from = time_from
    while current_from < time_till:
        page = zabbix_call(
            "history.get",
            {
                "itemids": item_id,
                "history": value_type,
                "time_from": current_from,
                "time_till": time_till,
                "output": "extend",
                "sortfield": "clock",
                "sortorder": "ASC",
                "limit": 10000,
            },
            api_url,
            api_token,
        )
        page = page or []
        if not page:
            break
        all_history.extend(page)
        last_clock = int(page[-1].get("clock", 0))
        next_from = last_clock + 1
        if next_from >= time_till or len(page) < 10000:
            break
        current_from = next_from
        time.sleep(0.15)
    return all_history


def extract_new_rows(api_url: str, api_token: str | None, metric_keywords: Optional[list[str]] = None, lookback_days: int = 1) -> pd.DataFrame:
    _ensure_dirs()
    metrics = metric_keywords or DEFAULT_METRICS
    time_till = int(_utc_now().timestamp())
    time_from = int((_utc_now() - timedelta(days=lookback_days)).timestamp())

    hosts = zabbix_get_hosts(api_url, api_token)
    rows: list[dict] = []
    for host in hosts:
        host_id = host.get("hostid")
        host_name = host.get("host", "unknown")
        items = zabbix_get_items(api_url, api_token, host_id, metrics)
        for item in items:
            history = zabbix_get_history(api_url, api_token, item.get("itemid"), int(item.get("value_type", 0)), time_from, time_till)
            for point in history:
                rows.append(
                    {
                        "host_name": host_name,
                        "item_name": item.get("name"),
                        "item_key": item.get("key_"),
                        "timestamp": pd.to_datetime(int(point.get("clock", 0)), unit="s", utc=True).tz_convert(None),
                        "value": point.get("value"),
                        "value_type": int(item.get("value_type", 0)),
                    }
                )
    return pd.DataFrame(rows, columns=["host_name", "item_name", "item_key", "timestamp", "value", "value_type"])


def refresh_models_from_continuous_data() -> None:
    if not RAW_DATA_PATH.exists():
        return

    temp_host_dir = BASE_DIR / "data" / "pipeline" / "refresh_hosts"
    temp_host_dir.mkdir(parents=True, exist_ok=True)
    split_by_host(RAW_DATA_PATH, temp_host_dir)
    host_files = []
    for host_file in sorted(temp_host_dir.glob("*.csv")):
        host_files.append({"host_name": host_file.stem, "file_path": str(host_file)})

    if not host_files:
        return

    train_models(
        host_files,
        models_dir=MODELS_DIR,
        reports_dir=REPORTS_DIR,
        plots_dir=PLOTS_DIR,
        progress_callback=None,
        min_improvement=0.05,
    )


def update_continuous_dataset(api_url: str, api_token: str | None, metric_keywords: Optional[list[str]] = None, lookback_days: int = 1) -> Path:
    _ensure_dirs()
    if not _acquire_lock():
        return RAW_DATA_PATH

    try:
        current = pd.DataFrame(columns=["host_name", "item_name", "item_key", "timestamp", "value", "value_type"])
        if RAW_DATA_PATH.exists():
            current = pd.read_csv(RAW_DATA_PATH)

        previous_row_count = len(current)
        incoming = extract_new_rows(api_url, api_token, metric_keywords=metric_keywords, lookback_days=lookback_days)
        merged = merge_extracted_rows(current, incoming)
        merged.to_csv(RAW_DATA_PATH, index=False)

        state = _load_state()
        state["last_run_at"] = _utc_now().isoformat()
        state["last_timestamp"] = merged["timestamp"].max().isoformat() if not merged.empty else None
        if should_trigger_model_refresh(state, previous_row_count, len(merged)):
            try:
                refresh_models_from_continuous_data()
                state["last_model_refresh_at"] = _utc_now().isoformat()
            except Exception:
                state["last_model_refresh_at"] = _utc_now().isoformat()
        _save_state(state)
        return RAW_DATA_PATH
    finally:
        _release_lock()


def run_continuous_loop(api_url: str, api_token: str | None, metric_keywords: Optional[list[str]] = None, interval_seconds: int = 300, max_runs: Optional[int] = None) -> None:
    run_count = 0
    while True:
        try:
            _write_log(f"Running continuous update cycle #{run_count + 1}")
            update_continuous_dataset(api_url, api_token, metric_keywords=metric_keywords, lookback_days=1)
            _write_log(f"Continuous update cycle #{run_count + 1} completed")
        except Exception as exc:
            _write_log(f"Continuous loop exception: {type(exc).__name__}: {exc}")
        run_count += 1
        if max_runs is not None and run_count >= max_runs:
            _write_log("Maximum run count reached, exiting continuous loop")
            break
        _write_log(f"Sleeping {interval_seconds} seconds before next cycle")
        time.sleep(interval_seconds)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Continuously pull Zabbix metrics into a growing CSV dataset.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--api-token", default=DEFAULT_API_TOKEN)
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--metric-keywords", nargs="+", default=DEFAULT_METRICS)
    args = parser.parse_args()

    _write_log("continuous_zabbix_pipeline.py started")
    run_continuous_loop(args.api_url, args.api_token, metric_keywords=args.metric_keywords, interval_seconds=args.interval_seconds, max_runs=args.max_runs)
    _write_log("continuous_zabbix_pipeline.py exiting")
