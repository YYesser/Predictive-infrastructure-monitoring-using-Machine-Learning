from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import base64
import hashlib
import json
import os
import re
import secrets
import shutil
import smtplib
import ssl
import subprocess
import threading
import time
import uuid
from datetime import datetime, timedelta
from email.message import EmailMessage

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
import lightgbm as lgb
import numpy as np

from continuous_zabbix_pipeline import (
    RAW_DATA_PATH as CONTINUOUS_DATA_PATH,
    STATE_PATH as CONTINUOUS_STATE_PATH,
    get_pipeline_status,
    start_continuous_pipeline,
    stop_continuous_pipeline,
    update_continuous_dataset,
)
from zabbix_pipeline import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    build_feature_vector_from_history,
    count_training_tasks,
    generate_simple_forecast,
    generate_trend_forecast,
    generate_rolling_mean_forecast,
    normalize_metric_frame,
    split_by_host,
    train_models,
)


def _load_continuous_pipeline_state(path: Path) -> pd.DataFrame:
    try:
        return pd.read_json(path)
    except Exception:
        try:
            import json

            raw_text = path.read_text(encoding="utf-8")
            data = json.loads(raw_text)
            return pd.DataFrame(data)
        except Exception:
            raise


def normalize_forecast_strategy(value: object) -> str:
    """Normalize strategy labels to a safe, UI-friendly value."""
    candidate = str(value or "").strip().lower()
    if candidate in {"lightgbm", "baseline", "trend_baseline", "rolling_baseline"}:
        return candidate
    return "lightgbm"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

BASE_DIR = Path(__file__).resolve().parent
load_env_file(BASE_DIR / ".env")
REPORTS_DIR = BASE_DIR / "reports"
TRAINING_SUMMARY_PATH = REPORTS_DIR / "training_summary.csv"
SKIPPED_SERIES_PATH = REPORTS_DIR / "skipped_series.csv"
PLOTS_DIR = REPORTS_DIR / "plots"
MODELS_DIR = BASE_DIR / "models"
DRAFTS_DIR = BASE_DIR / "drafts"
LIVE_DATA_PATH = BASE_DIR / "zabbix_data_live.csv"
LOGIN_TOKEN_PATH = BASE_DIR / ".login_token"
USER_STORE_PATH = BASE_DIR / "users.json"

DEFAULT_USERNAME = "admin"
DEFAULT_PASSWORD = "admin"

SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USERNAME = os.environ.get("SMTP_USERNAME", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
EMAIL_FROM = os.environ.get("EMAIL_FROM", SMTP_USERNAME)


PBKDF2_ITERATIONS = 280_000
PASSWORD_HASH_PREFIX = "pbkdf2_sha256"


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return f"{PASSWORD_HASH_PREFIX}${PBKDF2_ITERATIONS}${base64.b64encode(salt).decode('ascii')}${base64.b64encode(derived).decode('ascii')}"


def verify_password(password: str, stored_hash: str) -> bool:
    if stored_hash.startswith(f"{PASSWORD_HASH_PREFIX}$"):
        try:
            _, iterations_str, salt_b64, derived_b64 = stored_hash.split("$", 3)
            iterations = int(iterations_str)
            salt = base64.b64decode(salt_b64)
            expected = base64.b64decode(derived_b64)
            derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
            return secrets.compare_digest(derived, expected)
        except Exception:
            return False
    return hashlib.sha256(password.encode("utf-8")).hexdigest() == stored_hash


def get_default_user_store() -> dict[str, dict[str, str]]:
    return {
        DEFAULT_USERNAME: {
            "password": hash_password(DEFAULT_PASSWORD),
            "role": "admin",
        }
    }


def load_user_store() -> dict[str, dict[str, str]]:
    if not USER_STORE_PATH.exists():
        return get_default_user_store()

    try:
        loaded = json.loads(USER_STORE_PATH.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            sample_value = next(iter(loaded.values()), None)
            if isinstance(sample_value, dict) and "password" in sample_value:
                users = loaded
            else:
                # Backwards compatibility for old store format
                users = {
                    username: {
                        "password": password_hash,
                        "role": "admin" if username == DEFAULT_USERNAME else "user",
                    }
                    for username, password_hash in loaded.items()
                }

            if DEFAULT_USERNAME not in users:
                users[DEFAULT_USERNAME] = {
                    "password": hash_password(DEFAULT_PASSWORD),
                    "role": "admin",
                }
            elif not isinstance(users[DEFAULT_USERNAME], dict) or "password" not in users[DEFAULT_USERNAME]:
                users[DEFAULT_USERNAME] = {
                    "password": hash_password(DEFAULT_PASSWORD),
                    "role": "admin",
                }

            if not any(
                isinstance(record, dict) and record.get("role") == "admin"
                for record in users.values()
            ):
                users[DEFAULT_USERNAME]["role"] = "admin"

            return users
    except Exception:
        pass

    return get_default_user_store()


EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

def normalize_username(username: str) -> str:
    return username.strip()


def normalize_email(email: str) -> str:
    return email.strip().lower()


def is_valid_email(email: str) -> bool:
    return bool(EMAIL_REGEX.match(email))


def needs_password_change(users: dict[str, dict[str, str]], username: str) -> bool:
    record = users.get(username, {})
    return bool(isinstance(record, dict) and record.get("force_password_change", False))


def is_email_in_use(users: dict[str, dict[str, str]], email: str, ignore_username: str | None = None) -> bool:
    normalized_email = normalize_email(email)
    return any(
        isinstance(record, dict)
        and username != ignore_username
        and normalize_email(record.get("email", "")) == normalized_email
        for username, record in users.items()
    )


def get_password_status(password_hash: str) -> str:
    if password_hash.startswith(f"{PASSWORD_HASH_PREFIX}$"):
        return "Secure"
    if password_hash:
        return "Legacy SHA256"
    return "Unknown"


def get_sorted_user_rows(users: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for username, record in users.items():
        rows.append(
            {
                "Username": username,
                "Email": record.get("email", ""),
                "Role": record.get("role", "user"),
                "Password status": get_password_status(record.get("password", "")),
            }
        )
    return sorted(rows, key=lambda row: (row["Role"], row["Username"]))


def ensure_draft_dirs() -> None:
    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)


def get_draft_dir(draft_name: str) -> Path:
    return DRAFTS_DIR / sanitize_name(draft_name)


def get_draft_metadata_path(draft_dir: Path) -> Path:
    return draft_dir / "draft_metadata.json"


def save_draft_metadata(draft_dir: Path, metadata: dict) -> None:
    try:
        draft_dir.mkdir(parents=True, exist_ok=True)
        get_draft_metadata_path(draft_dir).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    except Exception:
        pass


def load_draft_metadata(draft_dir: Path) -> dict:
    metadata_path = get_draft_metadata_path(draft_dir)
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.setdefault("name", draft_dir.name)
            metadata.setdefault("published", False)
            metadata.setdefault("created_at", "")
            metadata.setdefault("created_by", "")
            metadata.setdefault("source_file", "")
            metadata.setdefault("published_at", "")
            metadata.setdefault("published_by", "")
            metadata.setdefault("disabled", False)
            metadata.setdefault("pending_review", False)
            metadata.setdefault("requested_by", "")
            metadata.setdefault("requested_at", "")
            metadata.setdefault("review_status", "draft")
            metadata.setdefault("approved_by", "")
            metadata.setdefault("approved_at", "")
            metadata.setdefault("rejected_by", "")
            metadata.setdefault("rejected_at", "")
            metadata.setdefault("rejection_reason", "")
            return metadata
        except Exception:
            return {}
    return {
        "name": draft_dir.name,
        "published": False,
        "created_at": "",
        "created_by": "",
        "source_file": "",
        "published_at": "",
        "published_by": "",
        "disabled": False,
        "pending_review": False,
        "requested_by": "",
        "requested_at": "",
        "review_status": "draft",
        "approved_by": "",
        "approved_at": "",
        "rejected_by": "",
        "rejected_at": "",
        "rejection_reason": "",
    }


def list_drafts() -> list[dict]:
    ensure_draft_dirs()
    drafts = []
    for draft_dir in sorted(DRAFTS_DIR.iterdir()):
        if not draft_dir.is_dir():
            continue
        metadata = load_draft_metadata(draft_dir)
        drafts.append({"slug": draft_dir.name, **metadata})
    return sorted(drafts, key=lambda item: item.get("created_at", ""), reverse=True)


def get_published_dashboards(include_disabled: bool = False) -> list[dict]:
    dashboards = [draft for draft in list_drafts() if draft.get("published")]
    if not include_disabled:
        dashboards = [draft for draft in dashboards if not draft.get("disabled", False)]
    return dashboards


def get_pending_review_drafts() -> list[dict]:
    return [draft for draft in list_drafts() if draft.get("pending_review", False)]


def request_draft_review(draft_dir: Path, requested_by: str | None = None) -> tuple[bool, str]:
    metadata = load_draft_metadata(draft_dir)
    if metadata.get("published"):
        return False, "This draft has already been published."
    if metadata.get("pending_review"):
        return False, "The draft is already pending review."

    metadata["pending_review"] = True
    metadata["requested_by"] = requested_by or metadata.get("requested_by", "")
    metadata["requested_at"] = datetime.utcnow().isoformat()
    metadata["review_status"] = "pending"
    metadata["approved_by"] = ""
    metadata["approved_at"] = ""
    metadata["rejected_by"] = ""
    metadata["rejected_at"] = ""
    metadata["rejection_reason"] = ""
    save_draft_metadata(draft_dir, metadata)
    return True, "Draft submitted for admin review."


def reject_draft_review(draft_dir: Path, rejected_by: str | None = None, reason: str = "") -> tuple[bool, str]:
    metadata = load_draft_metadata(draft_dir)
    if not metadata.get("pending_review"):
        return False, "This draft is not currently pending review."

    metadata["pending_review"] = False
    metadata["review_status"] = "rejected"
    metadata["rejected_by"] = rejected_by or ""
    metadata["rejected_at"] = datetime.utcnow().isoformat()
    metadata["rejection_reason"] = reason or ""
    save_draft_metadata(draft_dir, metadata)
    return True, "Draft review has been denied."


def load_dashboard_data(dashboard_dir: Path | None) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    if dashboard_dir is None:
        training_summary = pd.DataFrame()
        skipped_series = pd.DataFrame()
        if TRAINING_SUMMARY_PATH.exists():
            training_summary = pd.read_csv(TRAINING_SUMMARY_PATH)
        if SKIPPED_SERIES_PATH.exists():
            skipped_series = pd.read_csv(SKIPPED_SERIES_PATH)
        return training_summary, skipped_series, MODELS_DIR, PLOTS_DIR

    summary_path = dashboard_dir / "training_summary.csv"
    skipped_path = dashboard_dir / "skipped_series.csv"
    training_summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    skipped_series = pd.read_csv(skipped_path) if skipped_path.exists() else pd.DataFrame()
    return training_summary, skipped_series, dashboard_dir / "models", dashboard_dir / "plots"


def copy_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.is_dir():
            copy_tree(item, dst / item.name)
        else:
            try:
                shutil.copy2(item, dst / item.name)
            except Exception:
                pass


def save_uploaded_csv(uploaded_file, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as out:
        out.write(uploaded_file.getbuffer())


def remove_path_safe(path: Path, retries: int = 5, delay: float = 0.2) -> None:
    def _onerror(func, path_str, exc_info):
        try:
            os.chmod(path_str, 0o700)
            func(path_str)
        except Exception:
            pass

    for _ in range(retries):
        if not path.exists():
            return
        try:
            shutil.rmtree(path, onerror=_onerror)
            return
        except Exception:
            time.sleep(delay)

    if not path.exists():
        return

    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            file_path = Path(root) / name
            try:
                os.chmod(file_path, 0o700)
                file_path.unlink()
            except Exception:
                pass
        for name in dirs:
            dir_path = Path(root) / name
            try:
                os.chmod(dir_path, 0o700)
                dir_path.rmdir()
            except Exception:
                pass

    try:
        path.rmdir()
    except Exception:
        pass


def get_draft_reports(draft_dir: Path) -> tuple[Path, Path]:
    return draft_dir / "training_summary.csv", draft_dir / "skipped_series.csv"


def get_draft_plots_dir(draft_dir: Path) -> Path:
    return draft_dir / "plots"


def get_draft_models_dir(draft_dir: Path) -> Path:
    return draft_dir / "models"


def get_uploaded_file_path(draft_dir: Path) -> Path:
    return draft_dir / "uploaded.csv"


def get_draft_metrics_frame(draft_dir: Path) -> pd.DataFrame:
    uploaded_path = get_uploaded_file_path(draft_dir)
    if uploaded_path.exists():
        try:
            uploaded_df = pd.read_csv(uploaded_path)
            return normalize_metric_frame(uploaded_df, host_fallback="unknown", item_fallback="unknown")
        except Exception:
            pass

    host_dir = draft_dir / "hosts"
    if host_dir.exists() and host_dir.is_dir():
        metrics_frames: list[pd.DataFrame] = []
        for host_file in sorted(host_dir.glob("*.csv")):
            try:
                host_df = pd.read_csv(host_file)
                metrics_frames.append(normalize_metric_frame(host_df, host_fallback="unknown", item_fallback="unknown"))
            except Exception:
                continue
        if metrics_frames:
            return pd.concat(metrics_frames, ignore_index=True)

    return pd.DataFrame(columns=["host_name", "item_name", "item_key", "timestamp", "value", "value_type"])


def get_draft_model_paths(draft_dir: Path, host_name: str, item_key: str) -> Optional[list[str]]:
    host_dir = get_draft_models_dir(draft_dir) / sanitize_name(host_name)
    model_paths: list[str] = []
    for day in range(1, 8):
        model_path = host_dir / f"{sanitize_name(str(item_key))}_day{day}.txt"
        if not model_path.exists():
            return None
        model_paths.append(str(model_path))
    return model_paths


def rename_draft(draft_dir: Path, new_name: str) -> tuple[bool, str]:
    target_dir = DRAFTS_DIR / sanitize_name(new_name)
    if target_dir.exists():
        return False, "A draft with that name already exists."
    try:
        draft_dir.rename(target_dir)
        metadata = load_draft_metadata(target_dir)
        metadata["name"] = new_name
        save_draft_metadata(target_dir, metadata)
        return True, "Draft renamed successfully."
    except Exception as exc:
        return False, str(exc)


def delete_draft(draft_dir: Path) -> tuple[bool, str]:
    try:
        if draft_dir.exists():
            shutil.rmtree(draft_dir)
        return True, "Draft deleted successfully."
    except Exception as exc:
        return False, str(exc)


def render_draft_actions(draft_dir: Path) -> None:
    st.markdown("---")
    st.markdown("#### Draft actions")
    metadata = load_draft_metadata(draft_dir)
    col1, col2, col3 = st.columns([1, 1, 1])

    if metadata.get("published"):
        col1.button("This draft is live", disabled=True, key=f"publish_draft_{draft_dir.name}")
    elif metadata.get("pending_review"):
        col1.button("Pending admin review", disabled=True, key=f"publish_draft_{draft_dir.name}")
    else:
        if col1.button("Send draft for admin review", key=f"publish_draft_{draft_dir.name}"):
            requester = st.session_state.get("current_user", "unknown")
            success, message = request_draft_review(draft_dir, requested_by=requester)
            if success:
                st.session_state.draft_published_message = message
                st.session_state.previewed_draft = draft_dir.name
                st.rerun()
            else:
                st.error(message)

    if col2.button("Rename draft", key=f"rename_draft_{draft_dir.name}"):
        st.session_state.rename_draft_target = draft_dir.name

    if col3.button("Delete draft", key=f"delete_draft_{draft_dir.name}"):
        st.session_state.delete_draft_target = draft_dir.name

    if st.session_state.get("rename_draft_target") == draft_dir.name:
        new_name = st.text_input("New draft name", value=draft_dir.name, key=f"rename_input_{draft_dir.name}")
        if st.button("Confirm rename", key=f"confirm_rename_{draft_dir.name}"):
            success, message = rename_draft(draft_dir, new_name)
            if success:
                st.success(message)
                st.session_state.previewed_draft = sanitize_name(new_name)
                st.session_state.rename_draft_target = None
                st.rerun()
            else:
                st.error(message)

    if st.session_state.get("delete_draft_target") == draft_dir.name:
        st.warning("This will permanently delete the selected draft.")
        if st.button("Confirm delete", key=f"confirm_delete_{draft_dir.name}"):
            success, message = delete_draft(draft_dir)
            if success:
                st.success(message)
                st.session_state.delete_draft_target = None
                st.session_state.previewed_draft = None
                st.rerun()
            else:
                st.error(message)


def render_draft_preview(draft_dir: Path) -> None:
    st.markdown(f"#### Preview draft: {draft_dir.name}")
    summary_path, skipped_path = get_draft_reports(draft_dir)
    fallback_path = draft_dir / "fallback_report.csv"
    audit_log_path = draft_dir / "forecast_audit_log.json"
    if not summary_path.exists():
        st.warning("This draft has not been trained yet.")
        return

    summary_df = pd.read_csv(summary_path)
    skipped_df = pd.read_csv(skipped_path) if skipped_path.exists() else pd.DataFrame()

    if summary_df.empty:
        st.info("Draft training completed but no metrics were recorded.")
    else:
        st.dataframe(
            summary_df[["host_name", "item_key", "item_name", "mae", "rmse", "mape"]],
            use_container_width=True,
        )

    if skipped_df.empty:
        st.info("No skipped series were recorded for this draft.")
    else:
        with st.expander("View skipped series", expanded=False):
            st.dataframe(skipped_df, use_container_width=True)

    if fallback_path.exists():
        with st.expander("View fallback report", expanded=False):
            st.dataframe(pd.read_csv(fallback_path), use_container_width=True)
    else:
        st.info("No fallback report was generated for this draft.")

    if audit_log_path.exists():
        with st.expander("View audit log", expanded=False):
            st.json(json.loads(audit_log_path.read_text(encoding="utf-8")))
    else:
        st.info("No audit log was generated for this draft.")

    draft_csv_path = get_uploaded_file_path(draft_dir)
    if not draft_csv_path.exists():
        st.warning("Draft CSV file is missing, so predictions cannot be previewed.")
        return

    try:
        uploaded_df = pd.read_csv(draft_csv_path)
    except Exception as exc:
        st.warning(f"Unable to read draft CSV: {exc}")
        return

    normalized_uploaded = normalize_metric_frame(uploaded_df, host_fallback="unknown", item_fallback="unknown")

    host_names = sorted(set(summary_df["host_name"].astype(str).tolist()))
    if not host_names:
        host_names = sorted(set(normalized_uploaded["host_name"].astype(str).tolist()))

    if not host_names:
        st.info("No hosts are available in this draft for preview.")
        return

    selected_host = st.selectbox(
        "Select host for draft forecast preview",
        host_names,
        key=f"draft_preview_host_{draft_dir.name}",
    )

    item_keys = sorted(
        summary_df.loc[summary_df["host_name"].astype(str) == selected_host, "item_key"].astype(str).unique().tolist()
    )
    if not item_keys:
        item_keys = sorted(
            set(
                normalized_uploaded.loc[
                    normalized_uploaded["host_name"].astype(str).str.strip() == selected_host,
                    "item_key",
                ].astype(str).tolist()
            )
        )

    if not item_keys:
        fallback_host = host_names[0]
        selected_host = fallback_host
        item_keys = sorted(
            set(
                normalized_uploaded.loc[
                    normalized_uploaded["host_name"].astype(str).str.strip() == fallback_host,
                    "item_key",
                ].astype(str).tolist()
            )
        )

    if not item_keys:
        st.warning("No metrics found for the selected host in this draft.")
        return

    selected_item = st.selectbox(
        "Select metric to preview",
        item_keys,
        key=f"draft_preview_metric_{draft_dir.name}",
    )

    preview_data = normalize_metric_frame(uploaded_df, host_fallback=selected_host, item_fallback=selected_item)
    preview_data = preview_data[
        preview_data["host_name"].astype(str).str.strip() == selected_host
    ].copy()
    preview_data = preview_data[
        (preview_data["item_key"].astype(str).str.strip() == selected_item)
        | (preview_data["item_name"].astype(str).str.strip() == selected_item)
    ].copy()
    preview_data["value"] = pd.to_numeric(preview_data["value"], errors="coerce")
    preview_data = preview_data.dropna(subset=["value"])

    if preview_data.empty:
        st.warning("The uploaded draft data does not contain any usable numeric values for the selected host/metric.")
        return

    last_values = preview_data["value"].tail(9).values.astype(float)
    if len(last_values) == 0:
        st.warning("The uploaded draft data does not contain any usable numeric values for the selected host/metric.")
        return

    latest_timestamp = pd.Timestamp.utcnow()
    if not preview_data.empty and "timestamp" in preview_data.columns:
        latest_timestamp = pd.to_datetime(preview_data["timestamp"], errors="coerce").dropna().max()
        if pd.isna(latest_timestamp):
            latest_timestamp = pd.Timestamp.utcnow()

    feature_vector = build_feature_vector_from_history(last_values, timestamp=latest_timestamp)

    model_paths = get_draft_model_paths(draft_dir, selected_host, selected_item)
    forecast_strategy = "lightgbm"
    if not summary_df.empty and "forecast_strategy" in summary_df.columns:
        selector = (
            (summary_df["host_name"].astype(str).str.strip() == selected_host)
            & (summary_df["item_key"].astype(str).str.strip() == selected_item)
        )
        if selector.any():
            forecast_strategy = normalize_forecast_strategy(summary_df.loc[selector, "forecast_strategy"].iloc[0])
    forecast_values: list[float] = []
    if forecast_strategy == "lightgbm" and model_paths:
        for model_path in model_paths:
            booster = load_booster_cached(str(model_path))
            if booster is None:
                st.warning(f"Unable to load draft model: {model_path}")
                break
            try:
                prediction = float(booster.predict([feature_vector])[0])
                forecast_values.append(prediction)
            except Exception as exc:
                st.warning(f"Draft forecast error: {exc}")
                forecast_values = []
                break

    if str(forecast_strategy).strip().lower() == "trend_baseline":
        forecast_values = generate_trend_forecast(last_values, days=7)
    elif str(forecast_strategy).strip().lower() == "rolling_baseline":
        forecast_values = generate_rolling_mean_forecast(last_values, days=7, window=3)
    elif len(forecast_values) != 7:
        forecast_values = generate_simple_forecast(last_values, days=7)

    st.markdown("##### Draft 7-day forecast")
    threshold_value = float(st.session_state.get("forecast_threshold_selector", 0.05))
    st.caption(f"Strategy: {forecast_strategy} | threshold: {threshold_value:.2f}")
    cols = st.columns(7)
    for day, (col, value) in enumerate(zip(cols, forecast_values), start=1):
        col.metric(f"Day {day}", f"{value:.2f}")

    forecast_df = pd.DataFrame({"Day": list(range(1, 8)), "Prediction": forecast_values})
    st.line_chart(forecast_df.set_index("Day"))

    plot_dir = get_draft_plots_dir(draft_dir) / sanitize_name(selected_host) / sanitize_name(str(selected_item))
    actual_plot = plot_dir / "actual_vs_predicted.png"
    if actual_plot.exists():
        st.markdown("##### Draft validation plot")
        st.image(str(actual_plot), caption="Draft actual vs predicted validation plot")
    else:
        st.info("Draft validation plot is not available yet.")

    st.caption("This preview uses the draft's trained models and data, before publishing to the dashboard.")


def format_iso_datetime(timestamp: str) -> str:
    try:
        parsed = datetime.fromisoformat(timestamp)
        return parsed.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return timestamp


def publish_draft(draft_dir: Path, published_by: str | None = None, approved_by: str | None = None) -> tuple[bool, str]:
    summary_path, skipped_path = get_draft_reports(draft_dir)
    models_src = get_draft_models_dir(draft_dir)
    plots_src = get_draft_plots_dir(draft_dir)

    if not summary_path.exists() or not skipped_path.exists():
        return False, "Draft reports are missing. Train before publishing."

    try:
        # Copy published reports and models from the draft into the published dashboard.
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(summary_path, REPORTS_DIR / summary_path.name)
        shutil.copy2(skipped_path, REPORTS_DIR / skipped_path.name)

        if models_src.exists():
            copy_tree(models_src, MODELS_DIR)
        if plots_src.exists():
            copy_tree(plots_src, PLOTS_DIR)

        metadata = load_draft_metadata(draft_dir)
        metadata["published"] = True
        metadata["pending_review"] = False
        metadata["review_status"] = "published"
        metadata["published_at"] = datetime.utcnow().isoformat()
        metadata["published_by"] = published_by or metadata.get("requested_by", metadata.get("published_by", ""))
        if approved_by:
            metadata["approved_by"] = approved_by
            metadata["approved_at"] = datetime.utcnow().isoformat()
        save_draft_metadata(draft_dir, metadata)
        return True, "Draft published successfully."
    except Exception as exc:
        return False, str(exc)


def save_user_store(store: dict[str, dict[str, str]]) -> None:
    try:
        USER_STORE_PATH.write_text(json.dumps(store, indent=2), encoding="utf-8")
    except Exception:
        pass


def load_smtp_config() -> tuple[str, int, str, str, bool]:
    load_env_file(BASE_DIR / ".env")
    smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_username = os.environ.get("SMTP_USERNAME", "")
    smtp_password = os.environ.get("SMTP_PASSWORD", "")
    email_from = os.environ.get("EMAIL_FROM", smtp_username)
    smtp_configured = bool(smtp_username and smtp_password and email_from)
    return smtp_server, smtp_port, smtp_username, email_from, smtp_configured


def send_email(smtp_server: str, smtp_port: int, smtp_username: str, smtp_password: str, email_from: str, to_address: str, subject: str, body: str) -> tuple[bool, str]:
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = email_from
    message["To"] = to_address
    message.set_content(body)

    context = ssl.create_default_context()
    try:
        if smtp_port == 465:
            with smtplib.SMTP_SSL(smtp_server, smtp_port, context=context) as server:
                server.login(smtp_username, smtp_password)
                server.send_message(message)
        else:
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls(context=context)
                server.login(smtp_username, smtp_password)
                server.send_message(message)
        return True, ""
    except Exception as exc:
        return False, str(exc)


def send_user_notification(email_address: str, username: str, password: str) -> tuple[bool, str]:
    smtp_server, smtp_port, smtp_username, email_from, smtp_configured = load_smtp_config()
    if not smtp_configured:
        return False, "SMTP settings are not configured."

    subject = "Your new dashboard credentials"
    body = (
        f"Hello,\n\n"
        f"Your account has been created for the Zabbix dashboard.\n"
        f"Username: {username}\n"
        f"Password: {password}\n\n"
        f"Please sign in and change your password if required.\n"
    )
    smtp_password = os.environ.get("SMTP_PASSWORD", "")

    return send_email(smtp_server, smtp_port, smtp_username, smtp_password, email_from, email_address, subject, body)


def get_app_credentials() -> tuple[str, str]:
    """Get the dashboard login credentials from environment variables or defaults."""
    username = os.environ.get("ZABBIX_DASHBOARD_USER", DEFAULT_USERNAME)
    password = os.environ.get("ZABBIX_DASHBOARD_PASS", DEFAULT_PASSWORD)
    return username, password


def get_login_token(username: str) -> str:
    users = load_user_store()
    user_record = users.get(username, {})
    if isinstance(user_record, dict):
        password_hash = user_record.get("password", "")
    else:
        password_hash = user_record
    return hashlib.sha256(f"{username}:{password_hash}".encode("utf-8")).hexdigest()


def save_login_token(username: str) -> None:
    try:
        LOGIN_TOKEN_PATH.write_text(get_login_token(username), encoding="utf-8")
    except Exception:
        pass


def clear_login_token() -> None:
    try:
        if LOGIN_TOKEN_PATH.exists():
            LOGIN_TOKEN_PATH.unlink()
    except Exception:
        pass


def is_user_logged_in() -> bool:
    """Return True when the user has authenticated successfully."""
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in and LOGIN_TOKEN_PATH.exists():
        try:
            saved_token = LOGIN_TOKEN_PATH.read_text(encoding="utf-8").strip()
            users = load_user_store()
            for username, record in users.items():
                if saved_token == get_login_token(username):
                    st.session_state.logged_in = True
                    st.session_state.current_user = username
                    st.session_state.current_role = record.get("role", "user")
                    break
        except Exception:
            pass

    return bool(st.session_state.logged_in)


def render_login_form() -> None:
    """Show a simple login form before allowing access to the dashboard."""
    st.title("Zabbix Dashboard Login")
    st.write("Please sign in to continue.")

    if "login_error" not in st.session_state:
        st.session_state.login_error = ""

    with st.form("login_form"):
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        submitted = st.form_submit_button("Sign in")

        if submitted:
            users = load_user_store()
            user_record = users.get(username)
            if isinstance(user_record, dict) and verify_password(password, user_record.get("password", "")):
                st.session_state.logged_in = True
                st.session_state.current_user = username
                st.session_state.current_role = user_record.get("role", "user")
                st.session_state.login_error = ""
                if not user_record.get("password", "").startswith(f"{PASSWORD_HASH_PREFIX}$"):
                    user_record["password"] = hash_password(password)
                    save_user_store(users)
                st.session_state.require_password_change = needs_password_change(users, username)
                if not st.session_state.require_password_change:
                    save_login_token(username)
                st.rerun()
            else:
                st.session_state.logged_in = False
                st.session_state.login_error = "Invalid username or password."

    if st.session_state.login_error:
        st.error(st.session_state.login_error)


def render_my_dashboards_page() -> None:
    current_user = st.session_state.get("current_user", "")
    role = st.session_state.get("current_role", "user")
    if not current_user:
        st.error("Unable to determine the current user.")
        return

    if role not in ("admin", "engineer"):
        st.error("Access denied. My Dashboards is only available to admin and engineer roles.")
        return

    st.title("My Dashboards")
    st.write("Track dashboards you created or requested, including review status and admin feedback.")

    all_drafts = list_drafts()
    user_dashboards = [
        draft
        for draft in all_drafts
        if draft.get("created_by") == current_user
        or draft.get("requested_by") == current_user
        or draft.get("published_by") == current_user
    ]

    if not user_dashboards:
        st.info("You have not created or requested any dashboards yet.")
        return

    def format_status(draft: dict) -> str:
        if draft.get("published"):
            status = "Published"
            if draft.get("disabled"):
                status += " (disabled)"
            return status
        if draft.get("review_status") == "pending" or draft.get("pending_review"):
            return "Under review"
        if draft.get("review_status") == "rejected":
            return "Denied"
        return "Draft saved"

    rows = []
    for draft in user_dashboards:
        rows.append(
            {
                "Name": draft.get("name", ""),
                "Status": format_status(draft),
                "Created by": draft.get("created_by", ""),
                "Requested by": draft.get("requested_by", ""),
                "Created at": format_iso_datetime(draft.get("created_at", "")),
                "Requested at": format_iso_datetime(draft.get("requested_at", "")),
                "Published at": format_iso_datetime(draft.get("published_at", "")),
                "Rejected at": format_iso_datetime(draft.get("rejected_at", "")),
                "Review reason": draft.get("rejection_reason", ""),
            }
        )

    st.table(rows)

    rejected_dashboards = [draft for draft in user_dashboards if draft.get("review_status") == "rejected"]
    if rejected_dashboards:
        st.markdown("### Rejected dashboards")
        for draft in rejected_dashboards:
            st.markdown(f"**{draft.get('name', draft.get('slug'))}**")
            st.markdown(f"- Denied by: {draft.get('rejected_by', 'admin')}")
            st.markdown(f"- Denied at: {format_iso_datetime(draft.get('rejected_at', ''))}")
            if draft.get("rejection_reason"):
                st.markdown(f"- Reason: {draft.get('rejection_reason')}")
            else:
                st.markdown("- Reason: Not provided")
            st.write("---")


def render_live_data_page() -> None:
    current_user = st.session_state.get("current_user", "")
    role = st.session_state.get("current_role", "user")
    if not current_user:
        st.error("Unable to determine the current user.")
        return

    if role not in ("admin", "engineer"):
        st.error("Access denied. Extract Data is only available to admin and engineer roles.")
        return

    st.title("Extract Data")
    st.write("Data extraction and ingestion from Zabbix.")
    st.info("Use this page to configure metric extraction directly from your Zabbix server.")

    st.markdown("### Extract Data Source")
    source = st.radio(
        "Select the data source for training and model feedback:",
        ["Zabbix Instant Extraction"],
        index=0,
        horizontal=True,
        key="live_data_source",
    )

    st.markdown("#### Zabbix Instant Extraction")
    st.write(
        "When enabled, the app will pull metric history directly from Zabbix and make it available as an extracted data source. "
        "This section only configures the live pipeline; actual pipeline execution is handled by the Zabbix extraction backend."
    )

    zabbix_api_url = st.text_input(
        "Zabbix API URL",
        value=os.environ.get("ZABBIX_API_URL", "http://192.168.100.60/zabbix/api_jsonrpc.php"),
        help="The Zabbix JSON-RPC API endpoint.",
        key="zabbix_api_url",
    )
    zabbix_api_token = st.text_input(
        "Zabbix API Token",
        value=os.environ.get("ZABBIX_API_TOKEN", ""),
        placeholder="Enter your Zabbix API token",
        help="Your Zabbix API bearer token. Leave blank if you prefer username/password login.",
        type="password",
        key="zabbix_api_token",
    )
    zabbix_username = st.text_input(
        "Zabbix username (optional)",
        value=os.environ.get("ZABBIX_USERNAME", ""),
        placeholder="Optional fallback login username",
        help="Use this if your Zabbix server requires username/password authentication instead of an API token.",
        key="zabbix_username",
    )
    zabbix_password = st.text_input(
        "Zabbix password (optional)",
        value=os.environ.get("ZABBIX_PASSWORD", ""),
        placeholder="Optional fallback login password",
        help="Password for the optional Zabbix username login.",
        type="password",
        key="zabbix_password",
    )
    zabbix_days = st.number_input(
        "Days of history to extract",
        min_value=1,
        max_value=365,
        value=7,
        step=1,
        help="Number of days of historical metric data to pull from Zabbix.",
        key="zabbix_days",
    )
    available_metric_keywords = [
        "CPU",
        "Memory",
        "Disk",
        "Network",
        "Load",
        "Temperature",
        "Storage",
        "Process",
        "IO",
        "Interface",
    ]
    zabbix_metric_filter = st.multiselect(
        "Select metric keywords",
        options=available_metric_keywords,
        default=list(
            filter(
                None,
                [keyword.strip() for keyword in os.environ.get("ZABBIX_METRIC_FILTER", "CPU,Memory,Disk,Network").split(",") if keyword.strip()],
            )
        ),
        help="Choose one or more keywords to select matching Zabbix items.",
        key="zabbix_metric_filter",
    )
    st.caption("Pick the metric categories you want the live pipeline to pull from Zabbix.")

    st.markdown("#### Extraction Controls")
    output_filename = st.text_input(
        "Output file name",
        value="zabbix_data_live.csv",
        help="Choose the name of the CSV file that will store the extracted data.",
        key="zabbix_output_filename",
    )
    if st.button("Run Zabbix Data Extraction", key="run_zabbix_live_extraction"):
        if not zabbix_api_url or (not zabbix_api_token and not zabbix_username):
            st.error("Zabbix API URL plus either a token or a username is required to run the live pipeline.")
        elif zabbix_username and not zabbix_password:
            st.error("A Zabbix password is required when using username-based login.")
        else:
            output_path = BASE_DIR / output_filename
            try:
                run_zabbix_live_pipeline(
                    zabbix_api_url,
                    zabbix_api_token,
                    int(zabbix_days),
                    zabbix_metric_filter,
                    output_path,
                    zabbix_username,
                    zabbix_password,
                )
                st.success(f"Data extraction complete. Data saved to {output_path.name}.")
            except Exception as exc:
                st.error(f"Zabbix extraction failed: {exc}")

    if Path(LIVE_DATA_PATH).exists():
        live_summary = load_live_data_summary(LIVE_DATA_PATH)
        st.markdown("#### Last extraction summary")
        st.table(live_summary)
        st.markdown(
            "Use this live data file as a source for training in the Train Your Own page by selecting the file upload option and choosing `zabbix_data_live.csv`."
        )


@st.cache_data(show_spinner=False)
def load_data(training_summary_mtime: float = 0.0, skipped_series_mtime: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the training and skipped-series reports from disk.

    The modification times are passed as cache keys so Streamlit refreshes the cached
    data automatically whenever the underlying report files change on disk.
    """
    training_summary = pd.DataFrame()
    skipped_series = pd.DataFrame()

    if TRAINING_SUMMARY_PATH.exists():
        training_summary = pd.read_csv(TRAINING_SUMMARY_PATH)
    if SKIPPED_SERIES_PATH.exists():
        skipped_series = pd.read_csv(SKIPPED_SERIES_PATH)

    return training_summary, skipped_series


def load_live_data_summary(path: Path) -> pd.DataFrame:
    """Read the live extraction file and summarize row counts by host and item."""
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame()

    summary = (
        df.groupby(["host_name", "item_key"])
        .agg(rows=("timestamp", "count"), first_timestamp=("timestamp", "min"), last_timestamp=("timestamp", "max"))
        .reset_index()
    )
    summary["first_timestamp"] = pd.to_datetime(summary["first_timestamp"], errors="coerce")
    summary["last_timestamp"] = pd.to_datetime(summary["last_timestamp"], errors="coerce")
    return summary


def summarize_live_data_rows(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate live extraction rows without breaking on datetime columns."""
    if summary_df.empty:
        return summary_df.copy()

    summary_df = summary_df.copy()
    summary_df["rows"] = pd.to_numeric(summary_df["rows"], errors="coerce").fillna(0).astype(int)
    summary_df["first_timestamp"] = pd.to_datetime(summary_df["first_timestamp"], errors="coerce")
    summary_df["last_timestamp"] = pd.to_datetime(summary_df["last_timestamp"], errors="coerce")

    grouped = (
        summary_df.groupby(["host_name", "item_key"], as_index=False)
        .agg(
            rows=("rows", "sum"),
            first_timestamp=("first_timestamp", "min"),
            last_timestamp=("last_timestamp", "max"),
        )
    )
    return grouped


def zabbix_call(method: str, params: dict, api_url: str = "http://192.168.100.60/zabbix/api_jsonrpc.php", api_token: str | None = None) -> list | None:
    """Send a single JSON-RPC call to the Zabbix API and return the result."""
    payload = {
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": 1,
    }
    headers = {
        "Content-Type": "application/json-rpc",
        "Authorization": f"Bearer {api_token}" if api_token else "",
    }
    try:
        response = requests.post(api_url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            print(f"[API ERROR] {method}: {data['error']}")
            return None
        return data.get("result")
    except requests.exceptions.RequestException as e:
        print(f"[REQUEST FAILED] {method}: {e}")
        return None
    finally:
        time.sleep(0.15)


def zabbix_get_hosts(api_url: str, api_token: str) -> list[dict]:
    result = zabbix_call("host.get", {
        "output": ["hostid", "host", "status"],
        "filter": {"status": 0},
    }, api_url=api_url, api_token=api_token)
    return result or []


def zabbix_get_items(api_url: str, api_token: str, host_id: str, metric_keywords: list[str]) -> list[dict]:
    all_items: list[dict] = []
    for keyword in metric_keywords:
        result = zabbix_call("item.get", {
            "hostids": host_id,
            "search": {"name": keyword},
            "output": ["itemid", "name", "key_", "value_type"],
        }, api_url=api_url, api_token=api_token)
        if result:
            all_items.extend(result)

    seen: set[str] = set()
    unique_items: list[dict] = []
    for item in all_items:
        if item["itemid"] not in seen:
            seen.add(item["itemid"])
            unique_items.append(item)
    return unique_items


def zabbix_get_history(api_url: str, api_token: str, item_id: str, value_type: int, time_from: int, time_till: int) -> list[dict]:
    all_history: list[dict] = []
    current_from = time_from
    while current_from < time_till:
        page = zabbix_call("history.get", {
            "itemids": item_id,
            "history": value_type,
            "time_from": current_from,
            "time_till": time_till,
            "output": "extend",
            "sortfield": "clock",
            "sortorder": "ASC",
            "limit": 10000,
        }, api_url=api_url, api_token=api_token)
        page = page or []
        if not page:
            break

        all_history.extend(page)
        last_clock = int(page[-1]["clock"])
        next_from = last_clock + 1
        if next_from >= time_till or len(page) < 10000:
            break

        current_from = next_from

    return all_history


def run_zabbix_live_pipeline(
    api_url: str,
    api_token: str,
    days_back: int,
    metric_filter: str,
    output_path: Path | None = None,
    username: str | None = None,
    password: str | None = None,
) -> None:
    time_till = int(datetime.utcnow().timestamp())
    time_from = int((datetime.utcnow() - timedelta(days=days_back)).timestamp())
    metrics = [keyword.strip() for keyword in metric_filter if keyword.strip()]

    hosts = zabbix_get_hosts(api_url, api_token)
    rows: list[dict] = []
    summary_rows: list[dict] = []

    total_steps = max(len(hosts), 1)
    progress_bar = st.progress(0)
    progress_text = st.empty()

    for index, host in enumerate(hosts, start=1):
        host_id = host["hostid"]
        host_name = host["host"]
        progress_text.text(f"Processing host {index}/{len(hosts)}: {host_name}")
        progress_bar.progress(index / total_steps)

        items = zabbix_get_items(api_url, api_token, host_id, metrics)
        item_total = max(len(items), 1)
        for item_index, item in enumerate(items, start=1):
            progress_text.text(
                f"Processing host {index}/{len(hosts)} | item {item_index}/{item_total}: {item['name']}"
            )
            history = zabbix_get_history(
                api_url,
                api_token,
                item["itemid"],
                int(item["value_type"]),
                time_from,
                time_till,
            )
            if not history:
                continue
            for point in history:
                rows.append({
                    "host_name": host_name,
                    "item_name": item["name"],
                    "item_key": item["key_"],
                    "timestamp": int(point["clock"]),
                    "value": point["value"],
                    "value_type": int(item["value_type"]),
                })
            summary_rows.append({
                "host_name": host_name,
                "item_key": item["key_"],
                "rows": len(history),
                "first_timestamp": int(history[0]["clock"]),
                "last_timestamp": int(history[-1]["clock"]),
            })

    progress_bar.progress(1.0)
    progress_text.text("Finalizing extraction and writing output file...")

    output_path = output_path or LIVE_DATA_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["host_name", "item_name", "item_key", "timestamp", "value", "value_type"])
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df.to_csv(output_path, index=False)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df["first_timestamp"] = pd.to_datetime(summary_df["first_timestamp"], unit="s")
        summary_df["last_timestamp"] = pd.to_datetime(summary_df["last_timestamp"], unit="s")
        summary_df = summarize_live_data_rows(summary_df)
        summary_df.to_csv(BASE_DIR / "live_data_summary.csv", index=False)


def load_7day_models(host_name: str, item_key: str, models_dir: Path) -> Optional[list]:
    """Load all 7 models for a given host and item key."""
    host_dir = models_dir / sanitize_name(host_name)
    model_paths: list = []

    for day in range(1, 8):
        model_path = host_dir / f"{sanitize_name(item_key)}_day{day}.txt"
        if model_path.exists():
            model_paths.append(model_path)
        else:
            # Backwards compatibility: accept single-file model saved without _day suffix
            if day == 1:
                old_model_path = host_dir / f"{sanitize_name(item_key)}.txt"
                if old_model_path.exists():
                    model_paths.append(old_model_path)
                else:
                    return None
            else:
                return None

    return model_paths if len(model_paths) == 7 else None


@st.cache_resource
def load_booster_cached(model_path: str):
    """Cache and return a LightGBM Booster loaded from disk.

    Returns None on error.
    """
    try:
        return lgb.Booster(model_file=str(model_path))
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def read_host_csv_cached(path: str) -> pd.DataFrame:
    """Cached CSV reader for host data files."""
    return pd.read_csv(path)


def preload_models_async(model_paths: list) -> None:
    """Start a background thread to warm the model cache (non-blocking)."""
    def _preload():
        for p in model_paths:
            try:
                load_booster_cached(str(p))
            except Exception:
                pass

    thread = threading.Thread(target=_preload, daemon=True)
    thread.start()


def ensure_data_ready(training_summary: pd.DataFrame, skipped_series: pd.DataFrame) -> None:
    """Show a clear message if the report files are missing."""
    if training_summary.empty and not TRAINING_SUMMARY_PATH.exists():
        st.error("Training summary is not available yet. Please run zabbix_pipeline.py first.")
    if skipped_series.empty and not SKIPPED_SERIES_PATH.exists():
        st.error("Skipped series report is not available yet. Please run zabbix_pipeline.py first.")


def stop_streamlit_processes(delay_ms: int = 5000) -> None:
    marker_file = BASE_DIR / ".dashboard_opened"
    if marker_file.exists():
        try:
            marker_file.unlink()
        except Exception:
            pass

    def _kill_process():
        powershell_command = (
            "Get-CimInstance Win32_Process | "
            "Where-Object { $_.CommandLine -and $_.CommandLine -match 'streamlit.*run.*app\\.py' } | "
            "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
        )

        try:
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            subprocess.Popen(
                [
                    "powershell",
                    "-NoProfile",
                    "-WindowStyle",
                    "Hidden",
                    "-Command",
                    powershell_command,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                startupinfo=startupinfo,
                close_fds=True,
            )
        except Exception:
            pass

    timer = threading.Timer(delay_ms / 1000.0, _kill_process)
    timer.daemon = True
    timer.start()


def render_data_pipeline_page() -> None:
    """Render the dedicated data pipeline page for admin and engineer users."""
    st.title("Data Pipeline")
    st.caption("Manage and review the continuous Zabbix data extraction workflow for training data.")

    st.info("This pipeline continuously appends fresh Zabbix history into a growing CSV dataset for retraining and model refreshes.")

    status = get_pipeline_status()
    st.subheader("Background pipeline control")
    st.caption(f"Status: {'Running' if status['running'] else 'Stopped'} | PID: {status['pid'] or 'N/A'}")
    st.markdown(
        f"**Continuous dataset file name:** `{CONTINUOUS_DATA_PATH.name}`  \\n"        f"**Saved at:** `{CONTINUOUS_DATA_PATH}`"
    )

    # Small control to force a status refresh without waiting for automatic rerun
    if st.button("Refresh status", key="refresh_pipeline_status"):
        try:
            st.experimental_rerun()
        except Exception:
            pass

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Start background pipeline", key="start_background_pipeline"):
            try:
                result = start_continuous_pipeline(
                    os.environ.get("ZABBIX_API_URL", "http://192.168.100.60/zabbix/api_jsonrpc.php"),
                    os.environ.get(
                        "ZABBIX_API_TOKEN",
                        "c37bbc02ded0e038c8a16c9441a9d1a4f52c5720c49f3c8a4a7e2051f1192830",
                    ),
                    metric_keywords=["CPU", "Memory", "Disk", "Network"],
                    interval_seconds=300,
                )
                st.success(result["message"])
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to start background pipeline: {exc}")
    with col2:
        if st.button("Stop background pipeline", key="stop_background_pipeline"):
            try:
                result = stop_continuous_pipeline()
                st.success(result["message"])
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to stop background pipeline: {exc}")

    if st.button("Run one continuous refresh", key="run_continuous_refresh"):
        try:
            output_path = update_continuous_dataset(
                os.environ.get("ZABBIX_API_URL", "http://192.168.100.60/zabbix/api_jsonrpc.php"),
                os.environ.get(
                    "ZABBIX_API_TOKEN",
                    "c37bbc02ded0e038c8a16c9441a9d1a4f52c5720c49f3c8a4a7e2051f1192830",
                ),
                metric_keywords=["CPU", "Memory", "Disk", "Network"],
                lookback_days=1,
            )
            st.success(f"Continuous refresh completed. Data saved to {output_path}")
        except Exception as exc:
            st.error(f"Continuous refresh failed: {exc}")

    if CONTINUOUS_DATA_PATH.exists():
        st.info(f"Continuous dataset file: {CONTINUOUS_DATA_PATH}")
        df = pd.read_csv(CONTINUOUS_DATA_PATH)
        if not df.empty:
            st.subheader("Continuous dataset preview")
            st.dataframe(df.tail(10), use_container_width=True)
            st.caption(f"Rows in dataset: {len(df)}")
        else:
            st.warning("The continuous dataset exists but has no rows yet.")
    else:
        st.warning("No continuous data file has been created yet.")

    st.subheader("Latest background log")
    try:
        if os.path.exists("data/pipeline/continuous_pipeline.log"):
            with open("data/pipeline/continuous_pipeline.log", "r", encoding="utf-8") as handle:
                lines = handle.read().splitlines()
            preview = lines[-20:] if len(lines) > 20 else lines
            if preview:
                st.text_area("Recent log lines", "\n".join(preview), height=220)
            else:
                st.info("No log output yet. The background process will write here once it starts running.")
        else:
            st.info("No log file found yet.")
    except Exception as exc:
        st.warning(f"Unable to read the background log: {exc}")



def render_admin_page() -> None:
    """Render the dedicated admin page with server shutdown controls."""
    st.title("Admin Panel")
    st.caption("Manage users, review drafts, and control dashboard access.")

    threshold_value = st.number_input(
        "Minimum improvement threshold",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.get("forecast_threshold_selector", 0.05),
        step=0.01,
        format="%.2f",
        help="Higher values make the trained model less likely to be selected unless it clearly beats the baselines.",
        key="forecast_threshold_selector",
    )
    st.caption("Small threshold = more model reliance. Large threshold = more baseline reliance.")

    with st.expander("Forecast threshold guidance", expanded=False):
        st.write(
            "This is the training threshold used when deciding whether the trained model should be preferred over the simple baselines."
        )
        st.markdown("- Small threshold = more model reliance")
        st.markdown("- Large threshold = more baseline reliance")

    st.markdown("")
    stop_col, info_col = st.columns([1, 2])
    with stop_col:
        if st.button("Stop Streamlit server", key="stop_server"):
            clear_login_token()
            st.warning("Stopping the Streamlit server...")
            components.html(
                "<script>function redirectToBlank(){try{if(window.top!==window){window.top.location.href='about:blank';}else{window.location.href='about:blank';}}catch(e){window.location.href='about:blank';}}redirectToBlank();</script>",
                height=0,
                width=0,
            )
            stop_streamlit_processes(delay_ms=2500)
            st.stop()
    with info_col:
        st.info("Stopping the server will clear the saved login token and navigate away from the Streamlit app.")

    st.divider()
    smtp_server, smtp_port, smtp_username, email_from, smtp_configured = load_smtp_config()
    with st.expander("SMTP configuration", expanded=True):
        st.write(f"**Server:** `{smtp_server}`")
        st.write(f"**Port:** `{smtp_port}`")
        st.write(f"**Username:** `{smtp_username or 'Not configured'}`")
        st.write(f"**From address:** `{email_from or 'Not configured'}`")
        if smtp_configured:
            st.success("SMTP settings appear configured.")
            st.caption("If you are using Gmail, make sure 2FA is enabled and you are using an app password.")
        else:
            st.warning(
                "SMTP credentials are missing or incomplete. Update `.env` and restart the app."
            )
            st.markdown(
                "- Set `SMTP_USERNAME`, `SMTP_PASSWORD`, and `EMAIL_FROM` in `.env`\n"
                "- For Gmail, enable 2FA and use an App Password instead of your regular password."
            )

    st.divider()
    st.subheader("Create a new user")
    with st.form("create_user_form"):
        raw_username = st.text_input("New username", key="new_username")
        raw_email = st.text_input("User email address", key="new_email")
        new_password = st.text_input("New password", type="password", key="new_password")
        confirm_password = st.text_input("Confirm password", type="password", key="confirm_password")
        new_role = st.selectbox("Privilege level", ["user", "engineer", "admin"], index=0)
        create_submitted = st.form_submit_button("Create user")

        if create_submitted:
            new_username = normalize_username(raw_username)
            new_email = normalize_email(raw_email)
            users = load_user_store()
            existing_usernames = {username.lower() for username in users.keys()}

            if not new_username or not new_password or not new_email:
                st.error("Username, email, and password are required.")
            elif new_username.lower() in existing_usernames:
                st.error("That username already exists.")
            elif is_email_in_use(users, new_email):
                st.error("That email address is already in use.")
            elif not is_valid_email(new_email):
                st.error("Enter a valid email address.")
            elif len(new_password) < 8:
                st.error("Password must be at least 8 characters long.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                users[new_username] = {
                    "password": hash_password(new_password),
                    "role": new_role,
                    "email": new_email,
                    "force_password_change": True,
                }
                save_user_store(users)
                email_sent, email_error = send_user_notification(new_email, new_username, new_password)
                st.success(f"User '{new_username}' created successfully with role '{new_role}'.")
                if email_sent:
                    st.info(f"Credentials were emailed to {new_email}.")
                else:
                    st.warning(
                        "Email was not sent. Check SMTP settings and the app logs or restart the app if you just created .env."
                    )
                    if email_error:
                        st.write(f"Error: {email_error}")

    st.divider()
    st.subheader("Manage existing users")
    users = load_user_store()
    if users:
        st.table(get_sorted_user_rows(users))

        with st.form("edit_user_form"):
            selected_user = st.selectbox("Select user", sorted(users.keys()), key="selected_user")
            selected_record = users.get(selected_user, {})
            edit_username = st.text_input("Username", value=selected_user, key="edit_username")
            edit_email = st.text_input("Email", value=selected_record.get("email", ""), key="edit_email")
            edit_role = st.selectbox(
                "Role",
                ["user", "engineer", "admin"],
                index=["user", "engineer", "admin"].index(selected_record.get("role", "user")),
                key="edit_role",
            )
            confirm_delete = st.checkbox("Confirm permanent deletion", key="confirm_delete")
            delete_password = None
            selected_role = selected_record.get("role", "user")
            current_user = st.session_state.get("current_user", "")
            if selected_role == "admin" and selected_user != current_user:
                delete_password = st.text_input(
                    "Enter your password to delete this admin account",
                    type="password",
                    key="delete_admin_password",
                )
            update_user = st.form_submit_button("Update user")
            delete_user = st.form_submit_button("Delete user")

            if update_user:
                normalized_username = normalize_username(edit_username)
                normalized_email = normalize_email(edit_email)
                existing_usernames = {username.lower() for username in users.keys() if username != selected_user}

                if not normalized_username or not normalized_email:
                    st.error("Username and email are required.")
                elif normalized_username.lower() in existing_usernames:
                    st.error("That username already exists.")
                elif is_email_in_use(users, normalized_email, ignore_username=selected_user):
                    st.error("That email address is already in use.")
                elif not is_valid_email(normalized_email):
                    st.error("Enter a valid email address.")
                else:
                    record = users.pop(selected_user, {})
                    record["email"] = normalized_email
                    record["role"] = edit_role
                    users[normalized_username] = record
                    save_user_store(users)
                    if "current_user" in st.session_state and st.session_state.current_user == selected_user:
                        st.session_state.current_user = normalized_username
                        save_login_token(normalized_username)
                    st.success(f"User '{selected_user}' updated successfully.")
                    st.rerun()

            if delete_user:
                if not confirm_delete:
                    st.error("Please confirm deletion before deleting a user.")
                elif delete_password is not None:
                    current_user_record = users.get(current_user, {})
                    if not delete_password:
                        st.error("You must enter your password to delete another admin.")
                    elif not isinstance(current_user_record, dict) or not verify_password(delete_password, current_user_record.get("password", "")):
                        st.error("Password validation failed. Cannot delete the admin account.")
                    else:
                        users.pop(selected_user, None)
                        save_user_store(users)
                        if "current_user" in st.session_state and st.session_state.current_user == selected_user:
                            st.session_state.logged_in = False
                            st.session_state.current_user = ""
                            st.session_state.current_role = ""
                            clear_login_token()
                        st.success(f"User '{selected_user}' has been deleted.")
                        st.rerun()
                else:
                    users.pop(selected_user, None)
                    save_user_store(users)
                    if "current_user" in st.session_state and st.session_state.current_user == selected_user:
                        st.session_state.logged_in = False
                        st.session_state.current_user = ""
                        st.session_state.current_role = ""
                        clear_login_token()
                    st.success(f"User '{selected_user}' has been deleted.")
                    st.rerun()
    else:
        st.info("No users have been created yet.")

    st.divider()
    st.subheader("Pending draft reviews")
    pending_drafts = get_pending_review_drafts()
    if pending_drafts:
        selected_pending = st.selectbox(
            "Select a draft pending review",
            pending_drafts,
            format_func=lambda item: f"{item.get('name', item['slug'])} — requested by {item.get('requested_by', 'unknown')} on {format_iso_datetime(item.get('requested_at', ''))}",
            key="admin_pending_draft",
        )

        selected_pending_dir = get_draft_dir(selected_pending["slug"])
        st.markdown(f"**Draft:** {selected_pending.get('name', selected_pending['slug'])}")
        st.markdown(f"- **Requested by:** {selected_pending.get('requested_by', 'unknown')}")
        st.markdown(f"- **Requested at:** {format_iso_datetime(selected_pending.get('requested_at', ''))}")
        st.markdown(f"- **Created at:** {format_iso_datetime(selected_pending.get('created_at', ''))}")
        st.markdown(f"- **Current status:** {selected_pending.get('review_status', 'pending')}")

        with st.expander("Preview draft before decision", expanded=True):
            render_draft_preview(selected_pending_dir)

        col_a, col_b = st.columns([1, 1])
        if col_a.button("Accept and publish", key=f"accept_pending_{selected_pending['slug']}"):
            approver = st.session_state.get("current_user", "admin")
            published_by = selected_pending.get("requested_by", approver) or approver
            success, message = publish_draft(selected_pending_dir, published_by=published_by, approved_by=approver)
            if success:
                st.success("Draft accepted and published successfully.")
                st.session_state.admin_review_message = message
                st.rerun()
            else:
                st.error(message)

        deny_reason = st.text_area("Deny reason (optional)", key=f"deny_reason_{selected_pending['slug']}")
        if col_b.button("Deny request", key=f"deny_pending_{selected_pending['slug']}"):
            reviewer = st.session_state.get("current_user", "admin")
            success, message = reject_draft_review(selected_pending_dir, rejected_by=reviewer, reason=deny_reason)
            if success:
                st.success("Draft review request has been denied.")
                st.session_state.admin_review_message = message
                st.rerun()
            else:
                st.error(message)
    else:
        st.info("No drafts are currently pending admin review.")

    st.divider()
    st.subheader("Manage published dashboards")
    published_dashboards = get_published_dashboards(include_disabled=True)
    if not published_dashboards:
        st.info("No published dashboards have been created yet.")
    else:
        dashboard_selection = []
        for item in published_dashboards:
            state_label = "[disabled]" if item.get("disabled", False) else "[visible]"
            published_by = item.get("published_by", "unknown") or "unknown"
            dashboard_selection.append(
                f"{item.get('name', item['slug'])} — by {published_by} {state_label}"
            )

        selected_index = st.selectbox(
            "Select dashboard to manage",
            list(range(len(dashboard_selection))),
            format_func=lambda idx: dashboard_selection[idx],
            key="admin_published_dashboard",
        )
        selected_dashboard = published_dashboards[selected_index]
        selected_draft_dir = get_draft_dir(selected_dashboard["slug"])

        st.markdown(f"**Dashboard:** {selected_dashboard.get('name', selected_dashboard['slug'])}")
        st.markdown(f"- **Created by:** {selected_dashboard.get('published_by', 'unknown')}")
        st.markdown(f"- **Published at:** {format_iso_datetime(selected_dashboard.get('published_at', ''))}")
        if selected_dashboard.get("source_file"):
            st.markdown(f"- **Source file:** `{selected_dashboard.get('source_file')}`")
        st.markdown(f"- **Status:** {'Disabled from viewing' if selected_dashboard.get('disabled', False) else 'Visible to viewers'}")

        st.write("")
        toggle_label = (
            "Enable dashboard for viewing" if selected_dashboard.get("disabled", False) else "Disable dashboard from viewing"
        )
        if st.button(toggle_label, key=f"toggle_visibility_{selected_dashboard['slug']}"):
            selected_dashboard["disabled"] = not selected_dashboard.get("disabled", False)
            save_draft_metadata(selected_draft_dir, selected_dashboard)
            status_text = "enabled" if not selected_dashboard.get("disabled", False) else "disabled"
            st.success(f"Dashboard has been {status_text} for viewers.")
            st.rerun()

        delete_confirm = st.checkbox("Confirm permanent deletion of this published dashboard", key="confirm_delete_published")
        if st.button("Delete published dashboard", key=f"delete_published_{selected_dashboard['slug']}"):
            if not delete_confirm:
                st.error("Please confirm deletion before deleting the published dashboard.")
            else:
                success, message = delete_draft(selected_draft_dir)
                if success:
                    st.success("Published dashboard deleted successfully.")
                    st.rerun()
                else:
                    st.error(f"Failed to delete dashboard: {message}")

    st.divider()
    st.subheader("Admin Actions")
    st.markdown(
        "- **Stop Streamlit server**: shuts down the app process.\n"
        "- **Clear login token**: forces the next visit to require login.\n"
        "- **Add user**: register a new login for the dashboard."
    )


def render_force_password_change_page() -> None:
    current_user = st.session_state.get("current_user", "")
    if not current_user:
        st.error("Unable to determine the current user.")
        return

    users = load_user_store()
    user_record = users.get(current_user, {})
    if not isinstance(user_record, dict):
        st.error("Unable to load your account details.")
        return

    st.title("Change Password Required")
    st.warning("You must change your password before continuing.")
    st.write("Enter your current password and choose a new password.")

    with st.form("force_password_change_form"):
        current_password = st.text_input("Current password", type="password", key="force_current_password")
        new_password = st.text_input("New password", type="password", key="force_new_password")
        confirm_password = st.text_input("Confirm new password", type="password", key="force_confirm_password")
        submitted = st.form_submit_button("Change password")

        if submitted:
            if not current_password or not new_password or not confirm_password:
                st.error("All fields are required.")
            elif not verify_password(current_password, user_record.get("password", "")):
                st.error("Current password is incorrect.")
            elif len(new_password) < 8:
                st.error("New password must be at least 8 characters long.")
            elif new_password == current_password:
                st.error("New password must be different from the current password.")
            elif new_password != confirm_password:
                st.error("New passwords do not match.")
            else:
                user_record["password"] = hash_password(new_password)
                user_record["force_password_change"] = False
                users[current_user] = user_record
                save_user_store(users)
                st.session_state.require_password_change = False
                save_login_token(current_user)
                st.success("Your password has been updated. You can now continue.")
                st.rerun()


def render_train_your_own_page() -> None:
    """Render the Train Your Own page for admins and engineers."""
    st.title("Train Your Own")
    st.write("Upload new metric data, retrain models, and save results as drafts or publish them.")
    st.info("This page is available for admin and engineer roles.")

    ensure_draft_dirs()

    if "draft_published_message" in st.session_state:
        st.success(st.session_state.pop("draft_published_message"))

    existing_drafts_expanded = st.session_state.get(
        "existing_drafts_expanded",
        bool(st.session_state.get("previewed_draft")),
    )
    with st.expander("Existing drafts", expanded=existing_drafts_expanded):
        drafts = list_drafts()
        if drafts:
            draft_rows = [
                {
                    "Draft name": draft.get("name", ""),
                    "Created by": draft.get("created_by", ""),
                    "Review status": draft.get("review_status", "draft"),
                    "Created at": format_iso_datetime(draft.get("created_at", "")),
                    "Requested by": draft.get("requested_by", ""),
                    "Requested at": format_iso_datetime(draft.get("requested_at", "")),
                    "Published at": format_iso_datetime(draft.get("published_at", "")),
                }
                for draft in drafts
            ]
            st.table(draft_rows)

            draft_choices = [draft["slug"] for draft in drafts]
            selected_draft = st.selectbox("Preview an existing draft", draft_choices, key="preview_existing_draft")
            col1, col2 = st.columns([1, 1])
            if col1.button("Preview selected draft", key="preview_draft_button"):
                st.session_state.previewed_draft = selected_draft
                st.session_state.existing_drafts_expanded = True
            if col2.button("Delete selected draft", key="delete_existing_draft_button"):
                st.session_state.delete_existing_draft = selected_draft

            if st.session_state.get("delete_existing_draft") == selected_draft:
                st.warning("Are you sure you want to permanently delete this draft? This action cannot be undone.")
                if st.button("Confirm delete", key=f"confirm_delete_existing_draft_{selected_draft}"):
                    success, message = delete_draft(get_draft_dir(selected_draft))
                    if success:
                        st.success(message)
                        st.session_state.previewed_draft = None
                        st.session_state.delete_existing_draft = None
                        st.rerun()
                    else:
                        st.error(message)

            previewed = st.session_state.get("previewed_draft")
            if previewed:
                preview_dir = get_draft_dir(previewed)
                if preview_dir.exists():
                    render_draft_preview(preview_dir)
                    render_draft_actions(preview_dir)
                else:
                    st.warning("The selected draft no longer exists.")
        else:
            st.info("No drafts have been created yet.")

    st.divider()
    uploaded_file = st.file_uploader("Upload CSV data for retraining", type=["csv"], help="Upload a valid Zabbix metrics CSV.")
    default_name = f"draft_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    draft_name = st.text_input("Draft name", value=default_name, key="train_draft_name")
    st.caption("Enter a unique name for this draft. You can publish it now or save it for later.")

    threshold_value = st.number_input(
        "Minimum improvement threshold",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.get("forecast_threshold_selector", 0.05),
        step=0.01,
        format="%.2f",
        help="Higher values make the trained model less likely to be selected unless it clearly beats the baselines.",
        key="forecast_threshold_selector",
    )
    st.caption("Small threshold = more model reliance. Large threshold = more baseline reliance.")

    if uploaded_file is not None:
        st.markdown(f"**Selected file:** {uploaded_file.name}")

    train_draft_clicked = st.button("Train Draft", key="train_draft")

    if uploaded_file is not None and train_draft_clicked:
        draft_dir = get_draft_dir(draft_name)
        uploaded_path = get_uploaded_file_path(draft_dir)
        save_uploaded_csv(uploaded_file, uploaded_path)

        host_dir = draft_dir / "hosts"
        if host_dir.exists():
            remove_path_safe(host_dir)

        try:
            host_files = split_by_host(uploaded_path, host_dir)
        except Exception as exc:
            st.error(f"Failed to split uploaded CSV by host: {exc}")
            return

        if not host_files:
            st.error("No host data was found in the uploaded CSV.")
            return

        total_tasks = count_training_tasks(host_files)
        progress_bar = st.progress(0)
        progress_text = st.empty()

        def training_progress(current: int, message: str) -> None:
            if total_tasks > 0:
                progress = min(float(current) / float(total_tasks), 1.0)
                progress_bar.progress(progress)
            progress_text.text(message)

        draft_models_dir = get_draft_models_dir(draft_dir)
        draft_plots_dir = get_draft_plots_dir(draft_dir)

        st.info("Retraining models from uploaded file. This may take a few minutes.")
        try:
            train_models(
                host_files,
                models_dir=draft_models_dir,
                reports_dir=draft_dir,
                plots_dir=draft_plots_dir,
                progress_callback=training_progress,
                min_improvement=float(st.session_state.get("forecast_threshold_selector", threshold_value)),
            )
            progress_bar.progress(1.0)
            progress_text.text("Training complete.")

            metadata = {
                "name": draft_name,
                "created_at": datetime.utcnow().isoformat(),
                "created_by": st.session_state.get("current_user", "unknown"),
                "source_file": uploaded_file.name,
                "published": False,
                "published_at": "",
                "published_by": "",
                "disabled": False,
                "pending_review": False,
                "requested_by": "",
                "requested_at": "",
                "review_status": "draft",
                "approved_by": "",
                "approved_at": "",
                "rejected_by": "",
                "rejected_at": "",
                "rejection_reason": "",
            }
            save_draft_metadata(draft_dir, metadata)
            st.session_state.previewed_draft = draft_name
            st.session_state.existing_drafts_expanded = True

            st.success("Retraining completed successfully. The draft has been saved.")
            threshold_value = float(st.session_state.get("forecast_threshold_selector", 0.05))
            st.info(f"Selected strategy threshold for this run: {threshold_value:.2f} (higher values are more conservative).")
            st.info("You can preview the draft from the Existing drafts section, then publish, rename, or delete it.")

            summary_path, skipped_path = get_draft_reports(draft_dir)
            if summary_path.exists():
                st.markdown("### Draft training summary")
                st.dataframe(pd.read_csv(summary_path), use_container_width=True)
            if skipped_path.exists():
                st.markdown("### Draft skipped series")
                st.dataframe(pd.read_csv(skipped_path), use_container_width=True)

            st.session_state.previewed_draft = draft_name
        except Exception as exc:
            st.error(f"Draft training failed: {exc}")


def render_profile_page() -> None:
    current_user = st.session_state.get("current_user", "")
    if not current_user:
        st.error("Unable to determine the current user.")
        return

    users = load_user_store()
    user_record = users.get(current_user, {})
    if not isinstance(user_record, dict):
        st.error("Unable to load your account details.")
        return

    current_email = user_record.get("email", "")

    st.title("My Profile")
    st.write("Update your email address or change your password.")

    with st.form("profile_form"):
        new_email = st.text_input("Email", value=current_email, key="profile_email")
        current_password = st.text_input("Current password", type="password", key="profile_current_password")
        new_password = st.text_input("New password", type="password", key="profile_new_password")
        confirm_password = st.text_input("Confirm new password", type="password", key="profile_confirm_password")
        update_submitted = st.form_submit_button("Save changes")

        if update_submitted:
            normalized_email = normalize_email(new_email)
            email_changed = normalized_email != normalize_email(current_email)
            wants_password_change = bool(new_password or confirm_password or current_password)

            if email_changed and not normalized_email:
                st.error("Enter a valid email address.")
            elif email_changed and not is_valid_email(normalized_email):
                st.error("Enter a valid email address.")
            elif email_changed and is_email_in_use(users, normalized_email, ignore_username=current_user):
                st.error("That email address is already in use.")
            elif wants_password_change and not current_password:
                st.error("Current password is required to change your password.")
            elif wants_password_change and not verify_password(current_password, user_record.get("password", "")):
                st.error("Current password is incorrect.")
            elif wants_password_change and len(new_password) < 8:
                st.error("New password must be at least 8 characters long.")
            elif wants_password_change and new_password == current_password:
                st.error("New password must be different from the current password.")
            elif wants_password_change and new_password != confirm_password:
                st.error("New passwords do not match.")
            else:
                changes = []
                if email_changed:
                    user_record["email"] = normalized_email
                    changes.append("email")
                if wants_password_change:
                    user_record["password"] = hash_password(new_password)
                    changes.append("password")
                if changes:
                    users[current_user] = user_record
                    save_user_store(users)
                    st.success("Your profile has been updated.")
                    if "password" in changes:
                        st.info("Your password has been changed successfully.")
                else:
                    st.info("No changes were made.")


def render_host_view(training_summary: pd.DataFrame, skipped_series: pd.DataFrame, models_dir: Path, plots_dir: Path) -> None:
    """Render the host-specific view with all metrics displayed without expanders."""
    host_names = []
    if not training_summary.empty:
        host_names.extend(training_summary["host_name"].dropna().astype(str).tolist())
    if not skipped_series.empty:
        host_names.extend(skipped_series["host_name"].dropna().astype(str).tolist())

    host_names = sorted(set(host_names))

    if not host_names:
        st.info("No host data is available yet. Run zabbix_pipeline.py to generate the reports.")
        return

    selected_host = st.sidebar.selectbox("Select Host", host_names, key="host_selector")
    st.title(selected_host)

    host_summary = training_summary[training_summary["host_name"].astype(str) == selected_host]
    host_skipped = skipped_series[skipped_series["host_name"].astype(str) == selected_host]

    trained_count = int(len(host_summary))
    skipped_count = int(len(host_skipped))
    avg_mae = float(host_summary["mae"].mean()) if not host_summary.empty else float("nan")
    avg_rmse = float(host_summary["rmse"].mean()) if not host_summary.empty else float("nan")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Metrics trained", trained_count)
    col2.metric("Metrics skipped", skipped_count)
    col3.metric("Average MAE", f"{avg_mae:.4f}" if pd.notna(avg_mae) else "N/A")
    col4.metric("Average RMSE", f"{avg_rmse:.4f}" if pd.notna(avg_rmse) else "N/A")

    combined_metrics = []
    for _, row in host_summary.iterrows():
        combined_metrics.append((str(row["item_key"]), row, True))
    for _, row in host_skipped.iterrows():
        combined_metrics.append((str(row["item_key"]), row, False))

    combined_metrics.sort(key=lambda item: (str(item[1].get("item_name", "")), str(item[1].get("item_key", ""))))

    metric_labels = ["All metrics"] + [f"{row.get('item_name', '')} ({item_key})" for item_key, row, _ in combined_metrics]
    selected_metric_label = st.sidebar.selectbox(
        "Select Metric",
        metric_labels,
        key=f"metric_selector_{sanitize_name(selected_host)}",
    )

    selected_metric_key = None
    if selected_metric_label != "All metrics":
        selected_metric_key = next(
            (
                item_key
                for item_key, row, _ in combined_metrics
                if f"{row.get('item_name', '')} ({item_key})" == selected_metric_label
            ),
            None,
        )

    if "last_metric_key" not in st.session_state:
        st.session_state.last_metric_key = None

    ordered_metrics = list(combined_metrics)
    if selected_metric_key:
        ordered_metrics = [
            metric for metric in ordered_metrics if metric[0] == selected_metric_key
        ] + [
            metric for metric in ordered_metrics if metric[0] != selected_metric_key
        ]

    for idx, (item_key, row, is_trained) in enumerate(ordered_metrics):
        st.divider()
        anchor_id = f"metric-{sanitize_name(selected_host)}-{sanitize_name(item_key)}"
        st.markdown(f'<a id="{anchor_id}" style="display:block; position:relative; top:-90px;"></a>', unsafe_allow_html=True)
        header_text = f"{row.get('item_name', '')} ({item_key})"
        st.markdown(f"### {header_text}")
        
        if is_trained:
            metric_cols = st.columns(4)
            metric_cols[0].metric("Samples", int(row.get("num_samples", 0)))
            metric_cols[1].metric("MAE", f"{row.get('mae', float('nan')):.4f}" if pd.notna(row.get("mae")) else "N/A")
            metric_cols[2].metric("RMSE", f"{row.get('rmse', float('nan')):.4f}" if pd.notna(row.get("rmse")) else "N/A")
            metric_cols[3].metric("MAPE", f"{row.get('mape', float('nan')):.4f}" if pd.notna(row.get("mape")) else "N/A")

            plot_dir = plots_dir / sanitize_name(str(selected_host)) / sanitize_name(str(item_key))
            actual_plot = plot_dir / "actual_vs_predicted.png"

            if actual_plot.exists():
                st.markdown("#### Validation plot")
                st.image(str(actual_plot), caption="Actual vs Predicted (Day 1 validation)")
                st.caption("Validation plot compares actual day-1 test values with the model's day-1 predictions, not the 7-day forecast.")
            else:
                st.info("Plot not available")

            st.caption(f"Model path: {row.get('model_path', '')}")
            forecast_strategy = normalize_forecast_strategy(row.get("forecast_strategy", "lightgbm"))
            threshold_value = float(st.session_state.get("forecast_threshold_selector", 0.05))
            st.caption(f"Selected strategy: {forecast_strategy} | threshold: {threshold_value:.2f}")
            if pd.notna(row.get("selection_reason")):
                st.caption(f"Reason: {row.get('selection_reason')}")
            if pd.notna(row.get("health_score")):
                st.caption(f"Health score: {float(row.get('health_score')):.3f}")
            if pd.notna(row.get("backtest_confidence")):
                st.caption(f"Backtest confidence: {float(row.get('backtest_confidence')):.3f}")
            
            # Load 7-day forecast model paths (lazy load actual models)
            model_paths = load_7day_models(selected_host, item_key, models_dir)
            forecast_strategy = normalize_forecast_strategy(row.get("forecast_strategy", "lightgbm"))
            if model_paths and len(model_paths) == 7 and forecast_strategy == "lightgbm":
                st.markdown("#### 7-Day Forecast")
                st.info("This forecast uses recent observed values to predict the next 7 future data points.")

                try:
                    data_dir = BASE_DIR / "data" / "hosts"
                    published_host_dir = BASE_DIR / "drafts"
                    candidate_dirs = [data_dir]
                    if (BASE_DIR / "drafts").exists():
                        for draft_dir in (BASE_DIR / "drafts").iterdir():
                            if draft_dir.is_dir() and (draft_dir / "hosts").exists():
                                candidate_dirs.append(draft_dir / "hosts")
                    host_csv = None
                    for candidate_dir in candidate_dirs:
                        candidate_path = candidate_dir / f"{sanitize_name(selected_host)}.csv"
                        if candidate_path.exists():
                            host_csv = candidate_path
                            break

                    if host_csv is None:
                        st.warning("Host data file not found")
                    else:
                        # Read CSV via cached reader
                        try:
                            df = read_host_csv_cached(str(host_csv))
                            df.rename(columns=lambda x: x.strip(), inplace=True)

                            metric_data = df[df["item_key"].astype(str).str.strip() == str(item_key).strip()]

                            if metric_data.empty:
                                st.info(f"No matching data for {item_key} in CSV")
                            else:
                                metric_data = metric_data.copy()
                                metric_data["value"] = pd.to_numeric(metric_data["value"], errors="coerce")
                                metric_data = metric_data.dropna(subset=["value"])

                                if len(metric_data) < 1:
                                    st.info("No numeric values are available for forecasting.")
                                else:
                                    # Warm the model cache in background (non-blocking)
                                    preload_models_async(model_paths)

                                    recent_values = metric_data["value"].tail(10).values.astype(float)
                                    latest_timestamp = pd.to_datetime(metric_data["timestamp"], errors="coerce").dropna().max()
                                    if pd.isna(latest_timestamp):
                                        latest_timestamp = pd.Timestamp.utcnow()
                                    feature_vector = build_feature_vector_from_history(recent_values, timestamp=latest_timestamp)

                                    X_latest = np.array([feature_vector])

                                    # Get predictions by lazily loading cached boosters
                                    forecast_values = []
                                    for path in model_paths:
                                        try:
                                            booster = load_booster_cached(str(path))
                                            if booster is None:
                                                st.warning(f"Failed to load model: {path}")
                                                break
                                            pred = float(booster.predict(X_latest)[0])
                                            forecast_values.append(pred)
                                        except Exception as e:
                                            st.warning(f"Prediction error: {e}")
                                            break

                                    if len(forecast_values) != 7:
                                        forecast_values = generate_simple_forecast(recent_values, days=7)

                                    cols = st.columns(7)
                                    for day, (col, val) in enumerate(zip(cols, forecast_values), 1):
                                        col.metric(f"Day {day}", f"{val:.1f}")

                                    forecast_df = pd.DataFrame({
                                        "Day": list(range(1, 8)),
                                        "Prediction": forecast_values
                                    })
                                    st.dataframe(
                                        forecast_df,
                                        hide_index=True,
                                        use_container_width=True
                                    )

                                    st.markdown("##### Forecast curve")
                                    st.line_chart(
                                        forecast_df.set_index("Day")
                                    )
                        except Exception as csv_err:
                            st.warning(f"CSV read error: {str(csv_err)[:100]}")
                except Exception as e:
                    st.warning(f"Forecast setup error: {str(e)[:100]}")
            else:
                st.markdown("#### 7-Day Forecast")
                if forecast_strategy == "trend_baseline":
                    st.info("This series used the trend baseline because it outperformed the trained model and the persistence baseline on validation data.")
                elif forecast_strategy == "rolling_baseline":
                    st.info("This series used the rolling-mean baseline because it outperformed the trained model and the other simple baselines on validation data.")
                else:
                    st.info("This series used the simple persistence baseline because it outperformed the trained model on validation data.")
                recent_values = []
                try:
                    data_dir = BASE_DIR / "data" / "hosts"
                    published_host_dir = BASE_DIR / "drafts"
                    candidate_dirs = [data_dir]
                    if (BASE_DIR / "drafts").exists():
                        for draft_dir in (BASE_DIR / "drafts").iterdir():
                            if draft_dir.is_dir() and (draft_dir / "hosts").exists():
                                candidate_dirs.append(draft_dir / "hosts")
                    host_csv = None
                    for candidate_dir in candidate_dirs:
                        candidate_path = candidate_dir / f"{sanitize_name(selected_host)}.csv"
                        if candidate_path.exists():
                            host_csv = candidate_path
                            break

                    if host_csv is not None:
                        df = read_host_csv_cached(str(host_csv))
                        df.rename(columns=lambda x: x.strip(), inplace=True)
                        metric_data = df[df["item_key"].astype(str).str.strip() == str(item_key).strip()]
                        if not metric_data.empty:
                            metric_data = metric_data.copy()
                            metric_data["value"] = pd.to_numeric(metric_data["value"], errors="coerce")
                            metric_data = metric_data.dropna(subset=["value"])
                            recent_values = metric_data["value"].tail(10).values.astype(float)

                except Exception:
                    recent_values = []

                if forecast_strategy == "trend_baseline":
                    forecast_values = generate_trend_forecast(recent_values, days=7)
                elif forecast_strategy == "rolling_baseline":
                    forecast_values = generate_rolling_mean_forecast(recent_values, days=7, window=3)
                else:
                    forecast_values = generate_simple_forecast(recent_values, days=7)
                st.caption(f"Strategy: {forecast_strategy}")
            if pd.notna(row.get("selection_reason")):
                st.caption(f"Reason: {row.get('selection_reason')}")
            if pd.notna(row.get("health_score")):
                st.caption(f"Health score: {float(row.get('health_score')):.3f}")
            if pd.notna(row.get("backtest_confidence")):
                st.caption(f"Backtest confidence: {float(row.get('backtest_confidence')):.3f}")
                cols = st.columns(7)
                for day, (col, val) in enumerate(zip(cols, forecast_values), 1):
                    col.metric(f"Day {day}", f"{val:.1f}")

                forecast_df = pd.DataFrame({"Day": list(range(1, 8)), "Prediction": forecast_values})
                st.dataframe(forecast_df, hide_index=True, use_container_width=True)
                st.markdown("##### Forecast curve")
                st.line_chart(forecast_df.set_index("Day"))
        else:
            st.warning(f"⚠️ Skipped: {row.get('reason_skipped', 'Unknown reason')}")
            st.caption(f"Row count: {row.get('row_count', 'N/A')}")

    if selected_metric_key and st.session_state.last_metric_key != selected_metric_key:
        st.session_state.last_metric_key = selected_metric_key
        scroll_to_metric_anchor(f"metric-{sanitize_name(selected_host)}-{sanitize_name(selected_metric_key)}")


def sanitize_name(value: str) -> str:
    """Mirror the sanitization used by the training pipeline for file paths."""
    import re

    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return sanitized.strip("_") or "unknown"


def scroll_to_metric_anchor(anchor_id: str) -> None:
    """Scroll the current Streamlit page to the selected metric anchor."""
    if not anchor_id:
        return

    script = f"""
    <script>
      window.setTimeout(function() {{
        const target = document.getElementById({anchor_id!r});
        if (target) {{
          target.scrollIntoView({{behavior: 'smooth', block: 'start'}});
          window.location.hash = {anchor_id!r};
        }}
      }}, 450);
    </script>
    """
    components.html(script, height=0, width=0)


def hide_streamlit_deploy_button() -> None:
    """Hide Streamlit's top-right deploy button if present."""
    hide_html = """
        <style>
        .hide-deploy-button { display:none !important; }
        </style>
        <script>
        function hideDeployElements() {
            const root = window.parent?.document || document;
            const all = Array.from(root.querySelectorAll('*'));
            all.forEach((el) => {
                if (!el || !el.textContent) return;
                const text = el.textContent.trim().toLowerCase();
                const title = (el.getAttribute && el.getAttribute('title') || '').trim().toLowerCase();
                const aria = (el.getAttribute && el.getAttribute('aria-label') || '').trim().toLowerCase();
                const dataId = (el.getAttribute && el.getAttribute('data-testid') || '').trim().toLowerCase();
                const href = (el.getAttribute && el.getAttribute('href') || '').trim().toLowerCase();

                const isDeploy = text === 'deploy' || title === 'deploy' || aria === 'deploy' || dataId === 'deploy' || href.includes('deploy');
                if (!isDeploy) return;

                const wrapper = el.closest('button, a, [role="button"], div');
                if (wrapper) {
                    wrapper.classList.add('hide-deploy-button');
                } else {
                    el.classList.add('hide-deploy-button');
                }
            });
        }
        setInterval(hideDeployElements, 200);
        window.addEventListener('load', hideDeployElements);
        const observer = new MutationObserver(hideDeployElements);
        observer.observe((window.parent?.document || document).body, { childList: true, subtree: true });
        </script>
    """
    components.html(hide_html, height=0)


def render_logout_button() -> None:
    """Render a top-right logout button that clears login state."""
    if "current_user" not in st.session_state:
        st.session_state.current_user = ""

    if st.button("Logout", key="logout", help="Sign out and return to the login screen"):
        clear_login_token()
        st.session_state.logged_in = False
        st.session_state.current_user = ""
        st.session_state.current_role = "user"
        st.rerun()


def build_powerbi_export_frame(
    training_summary: pd.DataFrame,
    skipped_series: pd.DataFrame,
    dashboard_name: str | None = None,
) -> pd.DataFrame:
    """Create a Power BI-friendly export frame from the current dashboard metrics."""
    rows: list[dict[str, object]] = []

    if not training_summary.empty:
        for _, row in training_summary.iterrows():
            rows.append(
                {
                    "dashboard_name": dashboard_name or "",
                    "record_type": "trained",
                    "host_name": row.get("host_name", ""),
                    "item_key": row.get("item_key", ""),
                    "item_name": row.get("item_name", ""),
                    "num_samples": row.get("num_samples"),
                    "num_train": row.get("num_train"),
                    "num_test": row.get("num_test"),
                    "mae": row.get("mae"),
                    "rmse": row.get("rmse"),
                    "mape": row.get("mape"),
                    "mape_capped": row.get("mape_capped") if "mape_capped" in training_summary.columns else (float(np.clip(float(row.get("mape", 0) or 0), 0.0, 200.0)) if pd.notna(row.get("mape")) else None),
                    "baseline_mae": row.get("baseline_mae"),
                    "baseline_rmse": row.get("baseline_rmse"),
                    "baseline_mape": row.get("baseline_mape"),
                    "trend_mae": row.get("trend_mae"),
                    "trend_rmse": row.get("trend_rmse"),
                    "trend_mape": row.get("trend_mape"),
                    "rolling_mae": row.get("rolling_mae"),
                    "rolling_rmse": row.get("rolling_rmse"),
                    "rolling_mape": row.get("rolling_mape"),
                    "forecast_strategy": row.get("forecast_strategy", ""),
                    "selection_reason": row.get("selection_reason", ""),
                    "health_score": row.get("health_score"),
                    "backtest_confidence": row.get("backtest_confidence") if pd.notna(row.get("backtest_confidence")) else None,
                    "confidence_threshold": row.get("confidence_threshold") if "confidence_threshold" in training_summary.columns else DEFAULT_CONFIDENCE_THRESHOLD,
                    "mape_status": row.get("mape_status", "ok") if "mape_status" in training_summary.columns else ("ok" if pd.notna(row.get("mape_capped")) else "undefined_zero_actuals"),
                    "model_path": Path(str(row.get("model_path", ""))).name if str(row.get("model_path", "")) else "",
                    "reason_skipped": "",
                    "row_count": "",
                    "exported_at": datetime.utcnow().isoformat(),
                }
            )

    if not skipped_series.empty:
        for _, row in skipped_series.iterrows():
            rows.append(
                {
                    "dashboard_name": dashboard_name or "",
                    "record_type": "skipped",
                    "host_name": row.get("host_name", ""),
                    "item_key": row.get("item_key", ""),
                    "item_name": row.get("item_name", ""),
                    "num_samples": "",
                    "num_train": "",
                    "num_test": "",
                    "mae": "",
                    "rmse": "",
                    "mape": "",
                    "baseline_mae": "",
                    "baseline_rmse": "",
                    "baseline_mape": "",
                    "trend_mae": "",
                    "trend_rmse": "",
                    "trend_mape": "",
                    "rolling_mae": "",
                    "rolling_rmse": "",
                    "rolling_mape": "",
                    "forecast_strategy": "",
                    "selection_reason": "",
                    "health_score": "",
                    "backtest_confidence": "",
                    "confidence_threshold": "",
                    "mape_status": "skipped_insufficient_data",
                    "model_path": "",
                    "reason_skipped": row.get("reason_skipped", ""),
                    "row_count": row.get("row_count", ""),
                    "exported_at": datetime.utcnow().isoformat(),
                }
            )

    columns = [
        "dashboard_name",
        "record_type",
        "host_name",
        "item_key",
        "item_name",
        "num_samples",
        "num_train",
        "num_test",
        "mae",
        "rmse",
        "mape",
        "mape_capped",
        "baseline_mae",
        "baseline_rmse",
        "baseline_mape",
        "trend_mae",
        "trend_rmse",
        "trend_mape",
        "rolling_mae",
        "rolling_rmse",
        "rolling_mape",
        "forecast_strategy",
        "selection_reason",
        "health_score",
        "backtest_confidence",
        "confidence_threshold",
        "mape_status",
        "model_path",
        "reason_skipped",
        "row_count",
        "exported_at",
    ]
    return pd.DataFrame(rows, columns=columns)


def export_dashboard_to_powerbi(
    training_summary: pd.DataFrame,
    skipped_series: pd.DataFrame,
    dashboard_name: str | None = None,
    export_dir: Path | None = None,
) -> Path:
    """Export the current dashboard metrics to a CSV file suitable for Power BI."""
    export_dir = export_dir or REPORTS_DIR / "powerbi_exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    export_frame = build_powerbi_export_frame(training_summary, skipped_series, dashboard_name=dashboard_name)
    export_filename = f"{sanitize_name(dashboard_name or 'dashboard')}_powerbi_export.csv"
    export_path = export_dir / export_filename
    export_frame.to_csv(export_path, index=False)
    return export_path


def build_powerbi_metrics_export_frame(
    metrics_frame: pd.DataFrame,
    dashboard_name: str | None = None,
) -> pd.DataFrame:
    """Create a Power BI-friendly export frame from raw metric rows."""
    metrics_frame = metrics_frame.copy()
    if metrics_frame.empty:
        return pd.DataFrame(
            columns=[
                "dashboard_name",
                "host_name",
                "item_key",
                "item_name",
                "timestamp",
                "value",
                "value_type",
                "exported_at",
            ]
        )

    metrics_frame["timestamp"] = pd.to_datetime(metrics_frame["timestamp"], errors="coerce")
    metrics_frame["value"] = pd.to_numeric(metrics_frame["value"], errors="coerce")
    metrics_frame = metrics_frame.dropna(subset=["timestamp", "value"]).copy()
    metrics_frame["dashboard_name"] = dashboard_name or ""
    metrics_frame["exported_at"] = datetime.utcnow().isoformat()

    columns = [
        "dashboard_name",
        "host_name",
        "item_key",
        "item_name",
        "timestamp",
        "value",
        "value_type",
        "exported_at",
    ]
    return metrics_frame[columns]


def export_dashboard_metrics_to_powerbi(
    metrics_frame: pd.DataFrame,
    dashboard_name: str | None = None,
    export_dir: Path | None = None,
) -> Path:
    """Export the current raw metrics to a CSV file suitable for Power BI."""
    export_dir = export_dir or REPORTS_DIR / "powerbi_exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    export_frame = build_powerbi_metrics_export_frame(metrics_frame, dashboard_name=dashboard_name)
    export_filename = f"{sanitize_name(dashboard_name or 'dashboard')}_powerbi_metrics_export.csv"
    export_path = export_dir / export_filename
    export_frame.to_csv(export_path, index=False)
    return export_path


def main() -> None:
    """Entry point for the Streamlit dashboard."""
    st.set_page_config(page_title="Zabbix ML Results Viewer", layout="wide")
    hide_streamlit_deploy_button()

    if not is_user_logged_in():
        render_login_form()
        if not st.session_state.logged_in:
            return

    training_summary_mtime = TRAINING_SUMMARY_PATH.stat().st_mtime if TRAINING_SUMMARY_PATH.exists() else 0.0
    skipped_series_mtime = SKIPPED_SERIES_PATH.stat().st_mtime if SKIPPED_SERIES_PATH.exists() else 0.0
    training_summary, skipped_series = load_data(training_summary_mtime, skipped_series_mtime)
    ensure_data_ready(training_summary, skipped_series)

    top_left, top_right = st.columns([18, 1])
    with top_right:
        if st.button("Refresh data", key="refresh_dashboard_data", help="Reload the latest reports from disk"):
            load_data.clear()
            st.rerun()
        render_logout_button()

    pages = ["Dashboard", "My Profile"]
    role = st.session_state.get("current_role", "user")
    if role in ("admin", "engineer"):
        pages.append("My Dashboards")
        pages.append("Train Your Own")
        pages.append("Extract Data")
        pages.append("Data Pipeline")
    if role == "admin":
        pages.append("Admin")

    page = st.sidebar.radio("Page", pages, index=0)
    if page == "Admin":
        render_admin_page()
        return
    if page == "Extract Data":
        render_live_data_page()
        return
    if page == "Data Pipeline":
        render_data_pipeline_page()
        return
    if page == "Train Your Own":
        render_train_your_own_page()
        return
    if page == "My Dashboards":
        render_my_dashboards_page()
        return
    if page == "My Profile":
        render_profile_page()
        return

    if st.session_state.get("require_password_change", False):
        render_force_password_change_page()
        return

    published_dashboards = get_published_dashboards()
    if not published_dashboards:
        st.info("No published dashboards are available yet. Publish a draft from Train Your Own first.")
        return

    dashboard_options = [draft["name"] for draft in published_dashboards]
    selected_dashboard = st.sidebar.selectbox(
        "Select dashboard to view",
        dashboard_options,
        key="selected_dashboard",
    )

    selected_dashboard_dir = get_draft_dir(
        next(
            draft["slug"] for draft in published_dashboards if draft["name"] == selected_dashboard
        )
    )
    selected_dashboard_name = selected_dashboard

    training_summary, skipped_series, models_dir, plots_dir = load_dashboard_data(selected_dashboard_dir)

    if training_summary.empty and skipped_series.empty:
        return

    st.markdown(f"### Viewing dashboard: **{selected_dashboard_name}**")

    export_col, download_col = st.columns([4, 2])
    with export_col:
        if st.button("Export metrics to PowerBI", key="export_metrics_to_powerbi", use_container_width=True):
            metrics_frame = get_draft_metrics_frame(selected_dashboard_dir)
            export_path = export_dashboard_metrics_to_powerbi(
                metrics_frame,
                dashboard_name=selected_dashboard_name,
            )
            st.session_state["powerbi_export_path"] = str(export_path)
            st.session_state["powerbi_export_ready"] = True
            st.success(f"Metrics export created: {export_path.name}")
    with download_col:
        if st.session_state.get("powerbi_export_ready") and st.session_state.get("powerbi_export_path"):
            export_path = Path(st.session_state["powerbi_export_path"])
            if export_path.exists():
                with export_path.open("rb") as handle:
                    st.download_button(
                        "Download CSV",
                        data=handle.read(),
                        file_name=export_path.name,
                        mime="text/csv",
                        key="download_powerbi_export",
                    )

    if selected_dashboard_dir is not None and selected_dashboard_dir.exists():
        dashboard_metadata = load_draft_metadata(selected_dashboard_dir)
        published_by = dashboard_metadata.get("published_by")
        source_file = dashboard_metadata.get("source_file")
        published_at = dashboard_metadata.get("published_at")
        if published_by or source_file or published_at:
            info_parts = []
            if published_by:
                info_parts.append(f"published by **{published_by}**")
            if published_at:
                info_parts.append(f"on {format_iso_datetime(published_at)}")
            if source_file:
                info_parts.append(f"from `{source_file}`")
            st.markdown("*" + " ".join(info_parts) + "*")

    render_host_view(training_summary, skipped_series, models_dir, plots_dir)


if __name__ == "__main__":
    main()
