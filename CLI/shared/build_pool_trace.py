"""
Build Pool Trace - Track Cloud Build Worker Pool Runs

Logs build runs to ARR_COC/Training/logs/cloudbuild-pool.json with:
- Timing per phase (infra, build, push, submit)
- Cost calculation based on provision price
- Automatic cleanup of stale/lost builds
- Max 500 entries (auto-rotate oldest)
"""

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List

from ..config.constants import LOGS_DIR


# Build log path
BUILD_LOG_JSON = LOGS_DIR / "cloudbuild-pool.json"
MAX_ENTRIES = 500


# Phase descriptions
PHASE_DESCRIPTIONS = {
    "infra": "Check quota, verify worker pool, GCS bucket, Artifact Registry",
    "build": "Compile PyTorch (7517 tasks), torchvision, torchaudio with ccache",
    "push": "Push arr-pytorch-base image to Artifact Registry",
    "submit": "Submit W&B Launch job to Vertex AI",
}


def _ensure_log_file():
    """Ensure log directory and file exist"""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if not BUILD_LOG_JSON.exists():
        with open(BUILD_LOG_JSON, "w") as f:
            json.dump({"runs": []}, f, indent=2)


def _load_builds() -> Dict:
    """Load builds from JSON"""
    _ensure_log_file()
    with open(BUILD_LOG_JSON) as f:
        return json.load(f)


def _save_builds(data: Dict):
    """Save builds to JSON"""
    with open(BUILD_LOG_JSON, "w") as f:
        json.dump(data, f, indent=2)


def create_build_entry(
    machine: str,
    vcpus: int,
    provision_price_usd_per_hr: Optional[float] = None
) -> str:
    """
    Create new build entry with status=in_process

    Args:
        machine: Machine type (e.g. "c3-highcpu-176")
        vcpus: Number of vCPUs
        provision_price_usd_per_hr: Price per hour at provision time

    Returns:
        build_id (UUID string)
    """
    build_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat() + "Z"

    entry = {
        "build_id": build_id,
        "timestamp": timestamp,
        "status": "in_process",
        "machine": machine,
        "vcpus": vcpus,
        "provision_price_usd_per_hr": provision_price_usd_per_hr,
        "cost_estimate_usd": None,
        "phases": {}
    }

    # Initialize all phases as pending
    for phase_name, desc in PHASE_DESCRIPTIONS.items():
        entry["phases"][phase_name] = {
            "desc": desc,
            "start": None,
            "end": None,
            "duration_sec": None,
            "status": "pending"
        }

    entry["error"] = None

    # Add to log
    data = _load_builds()
    data["runs"].append(entry)
    _save_builds(data)

    return build_id


def update_phase(
    build_id: str,
    phase_name: str,
    status: str,
    start: Optional[str] = None,
    end: Optional[str] = None
):
    """
    Update phase timing and status

    Args:
        build_id: Build UUID
        phase_name: Phase name (infra/build/push/submit)
        status: Phase status (pending/in_process/success/failure)
        start: ISO timestamp for phase start
        end: ISO timestamp for phase end
    """
    data = _load_builds()

    # Find build
    build = None
    for entry in data["runs"]:
        if entry["build_id"] == build_id:
            build = entry
            break

    if not build:
        raise ValueError(f"Build {build_id} not found")

    # Update phase
    phase = build["phases"][phase_name]
    phase["status"] = status

    if start:
        phase["start"] = start
    if end:
        phase["end"] = end

        # Calculate duration if both start and end are set
        if phase["start"]:
            start_dt = datetime.fromisoformat(phase["start"].replace("Z", ""))
            end_dt = datetime.fromisoformat(end.replace("Z", ""))
            duration_sec = (end_dt - start_dt).total_seconds()
            phase["duration_sec"] = int(duration_sec)

    _save_builds(data)


def mark_build_complete(
    build_id: str,
    status: str,
    error: Optional[str] = None
):
    """
    Mark build as complete with final status

    Calculates total cost based on all phase durations and provision price.

    Args:
        build_id: Build UUID
        status: Final status (success/failure/lost)
        error: Error message if failure
    """
    data = _load_builds()

    # Find build
    build = None
    for entry in data["runs"]:
        if entry["build_id"] == build_id:
            build = entry
            break

    if not build:
        raise ValueError(f"Build {build_id} not found")

    # Update status and error
    build["status"] = status
    if error:
        build["error"] = error

    # Calculate total cost
    if build["provision_price_usd_per_hr"]:
        total_duration_sec = 0
        for phase in build["phases"].values():
            if phase["duration_sec"]:
                total_duration_sec += phase["duration_sec"]

        duration_hours = total_duration_sec / 3600
        build["cost_estimate_usd"] = round(
            duration_hours * build["provision_price_usd_per_hr"], 4
        )

    _save_builds(data)


def cleanup_lost_builds():
    """
    Mark builds >2 days old with status=in_process as 'lost'
    """
    data = _load_builds()
    cutoff = datetime.utcnow() - timedelta(days=2)

    for entry in data["runs"]:
        if entry["status"] == "in_process":
            timestamp = datetime.fromisoformat(entry["timestamp"].replace("Z", ""))
            if timestamp < cutoff:
                entry["status"] = "lost"
                entry["error"] = "Build marked as lost (>2 days old, no completion)"

    _save_builds(data)


def rotate_old_entries():
    """
    Keep only newest 500 entries, delete oldest
    """
    data = _load_builds()

    if len(data["runs"]) > MAX_ENTRIES:
        # Sort by timestamp (newest first)
        data["runs"].sort(
            key=lambda x: x["timestamp"],
            reverse=True
        )
        # Keep only newest 500
        data["runs"] = data["runs"][:MAX_ENTRIES]
        _save_builds(data)


def get_build(build_id: str) -> Optional[Dict]:
    """
    Get build entry by ID

    Args:
        build_id: Build UUID

    Returns:
        Build entry dict or None if not found
    """
    data = _load_builds()
    for entry in data["runs"]:
        if entry["build_id"] == build_id:
            return entry
    return None


def get_all_builds() -> List[Dict]:
    """
    Get all build entries

    Returns:
        List of build entry dicts
    """
    data = _load_builds()
    return data["runs"]
