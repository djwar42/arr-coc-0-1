"""
🤖 MECHA Registry System (メカ) + Fatigue Tracking

Tracks acquired MECHA (regional worker pools) and their machine types.
Automatically wipes all pools when CPU NUMBER changes.

MECHA Collection Strategy:
- Progressive acquisition: Try to acquire missing MECHAs each launch
- Full wipe on CPU change: Machine type change = delete ALL and restart
- Fatigue system: Failed MECHAs need rest before retry
- Full fleet goal: Collect all 18 MECHA regions

MECHA Fatigue System:
- 1st timeout (15 min) → FATIGUED 4 hours 😴
- 2nd timeout → FATIGUED 4 hours 😴
- 3rd timeout in 24h period → EXHAUSTED 24 hours (whole day) 🛌
- Auto-recovery: MECHAs can battle again after rest period

Registry tracks:
- Region name
- Machine type (e.g., c3-standard-176)
- Status (RUNNING, CREATING, FAILED)
- Last attempt timestamp
- Fatigue state (fatigued_until, failures_today)
- Fatigue reason (human-readable + machine-readable code)
"""

# <claudes_code_comments>
# ** Function List **
# load_registry() - Load MECHA registry using SafeJSON (auto-handles corruption)
# save_registry(registry) - Save registry using SafeJSON (atomic + backups)
# check_machine_type_changed(registry, current_machine) - Detect CPU NUMBER change
# wipe_all_mechas(registry, new_machine_type) - Nuclear reset on machine type change
# get_deployed_mechas(registry, machine_type) - List RUNNING MECHAs
# get_missing_mechas(registry, all_regions) - List unacquired regions
# update_mecha_status(...) - Update MECHA status (RUNNING/CREATING/FAILED)
# get_mecha_fleet_status(registry, total_regions) - Overall fleet health
# is_mecha_fatigued(mecha_info) - Check if MECHA needs rest
# _log_godzilla_incident(...) - Record failure incident to campaign_stats
# record_mecha_timeout(...) - Apply fatigue penalties on timeout
# get_available_mechas(registry, all_regions) - Get rested MECHAs ready for battle
# get_fatigue_status_display(registry) - Human-readable fatigue summary
#
# ** Technical Review **
# Progressive MECHA acquisition system with fatigue tracking to prevent queue thrashing.
# Flow: check_machine_type_changed() → wipe if CPU changed → get_missing_mechas() →
# attempt acquisition → update_mecha_status() → on timeout: record_mecha_timeout() →
# apply 4h/24h fatigue → campaign_stats.record_fatigue_event().
#
# Fatigue escalation (3-strike system):
# - 1st timeout: FATIGUED 4 hours (fatigued_until = now + 4h)
# - 2nd timeout (within 24h): FATIGUED 4 hours again
# - 3rd timeout (within 24h): EXHAUSTED 24 hours (fatigued_until = now + 24h)
# Reset: failures_today cleared after 24h window expires.
#
# Machine type changes (CPU NUMBER): Triggers wipe_all_mechas() → deletes ALL pools →
# fresh start (prevents mismatched machine types across regions).
#
# File I/O: Uses SafeJSON for both registry (mecha_hangar.json) and godzilla incidents
# (godzilla_incidents.json). Returns {} if missing/corrupt. 20 versioned backups per file.
# Atomic writes prevent corruption if 2 builds finish simultaneously.
# </claudes_code_comments>

import time
from pathlib import Path
from typing import Dict, Optional, Tuple, List

# Import SafeJSON for production-grade JSON I/O
from ...shared.safe_json import SafeJSON

# Registry file location - stored in mecha/data/ folder
MECHA_REGISTRY_PATH = Path(__file__).parent / "data" / "mecha_hangar.json"


def load_registry() -> Dict:
    """
    Load MECHA registry from disk using SafeJSON.

    Returns:
        {
            "machine_type": "c3-standard-176",  # Current CPU config
            "last_updated": 1234567890,
            "mechas": {
                "us-central1": {
                    "machine_type": "c3-standard-176",
                    "operational_status": "OPERATIONAL" | "NONOPERATIONAL" | "CREATING",
                    "created_at": 1234567890,
                    "last_attempt": 1234567890,

                    # Fatigue tracking (only present after failure)
                    "fatigued_until": 1234567890,  # Unix timestamp when fatigue expires
                    "failures_today": [1234567890, ...],  # Failure timestamps
                    "failure_count_today": 2,
                    "fatigue_message": "FATIGUED for 4h",
                    "last_failure_reason": "Queue timeout - 45 minutes in QUEUED state",
                    "last_failure_reason_code": "queue-timeout"  # Machine-readable classification
                },
                ...
            }
        }
    """
    # SafeJSON handles all error cases and returns {} if file doesn't exist or is corrupt
    registry = SafeJSON.read(MECHA_REGISTRY_PATH)

    # If empty (new file or corruption), initialize with defaults
    if not registry:
        registry = {
            "machine_type": None,
            "last_updated": time.time(),
            "mechas": {}
        }

    return registry


def save_registry(registry: Dict):
    """Save MECHA registry to disk using SafeJSON"""
    registry["last_updated"] = time.time()

    # SafeJSON handles all locking, atomic writes, and backups automatically
    SafeJSON.write(MECHA_REGISTRY_PATH, registry)


def check_machine_type_changed(registry: Dict, current_machine: str) -> bool:
    """
    Check if machine type changed (CPU NUMBER changed).

    Returns True if we need to wipe all pools.
    """
    stored_machine = registry.get("machine_type")

    if stored_machine is None:
        # First run - no wipe needed
        return False

    if stored_machine != current_machine:
        # MACHINE TYPE CHANGED! Wipe everything!
        return True

    return False


def wipe_all_mechas(registry: Dict, new_machine_type: str) -> Dict:
    """
    Wipe all MECHA deployments (CPU NUMBER changed).

    Returns fresh registry with new machine type.
    """
    return {
        "machine_type": new_machine_type,
        "last_updated": time.time(),
        "mechas": {},
        "wiped_at": time.time(),
        "wiped_reason": f"Machine type changed to {new_machine_type}"
    }


def get_deployed_mechas(registry: Dict, machine_type: str) -> List[str]:
    """
    Get list of RUNNING MECHA regions for given machine type.

    Returns: ["us-east4", "europe-west1", ...]
    """
    mechas = registry.get("mechas", {})

    deployed = []
    for region, mecha_info in mechas.items():
        if (mecha_info.get("machine_type") == machine_type and
            mecha_info.get("operational_status") == "OPERATIONAL"):
            deployed.append(region)

    return deployed


def get_missing_mechas(registry: Dict, all_regions: List[str]) -> List[str]:
    """
    Get list of MECHA regions that haven't been acquired yet.

    Returns: ["asia-south1", "southamerica-east1", ...]
    """
    mechas = registry.get("mechas", {})
    operational_regions = {r for r, info in mechas.items() if info.get("operational_status") == "OPERATIONAL"}

    missing = [r for r in all_regions if r not in operational_regions]
    return missing


def update_mecha_status(
    registry: Dict,
    region: str,
    machine_type: str,
    status: str,
    **kwargs
):
    """
    Update MECHA deployment status.

    Args:
        region: Region name (e.g., "us-east4")
        machine_type: Machine type (e.g., "c3-standard-176")
        status: "OPERATIONAL", "CREATING", "NONOPERATIONAL"
        **kwargs: Additional fields (error_message, etc.)
    """
    if "mechas" not in registry:
        registry["mechas"] = {}

    mecha_info = registry["mechas"].get(region, {})

    mecha_info.update({
        "machine_type": machine_type,
        "operational_status": status,
        "last_attempt": time.time(),
        **kwargs
    })

    if status == "OPERATIONAL" and "created_at" not in mecha_info:
        mecha_info["created_at"] = time.time()

    registry["mechas"][region] = mecha_info

    # Update global machine type
    registry["machine_type"] = machine_type


def get_mecha_fleet_status(registry: Dict, total_regions: int) -> Tuple[int, int, bool]:
    """
    Get MECHA fleet completion status.

    Returns:
        (deployed_count, total_count, is_full_fleet)
    """
    mechas = registry.get("mechas", {})
    operational_count = sum(1 for info in mechas.values() if info.get("operational_status") == "OPERATIONAL")

    is_full = operational_count == total_regions

    return (operational_count, total_regions, is_full)


# ============================================================
# MECHA FATIGUE SYSTEM 😴💤🛌
# ============================================================

def is_mecha_fatigued(mecha_info: Dict) -> Tuple[bool, Optional[str]]:
    """
    Check if MECHA is currently fatigued and needs rest.
    
    Returns:
        (is_fatigued, message)
        - (True, "Fatigued until 2025-01-10 18:00") if fatigued
        - (False, None) if ready to battle
    """
    now = time.time()
    
    # Check if MECHA has fatigue timestamp
    fatigued_until = mecha_info.get("fatigued_until", 0)
    
    if fatigued_until > now:
        # Still fatigued!
        remaining_hours = (fatigued_until - now) / 3600
        return (True, f"Fatigued until {time.strftime('%Y-%m-%d %H:%M', time.localtime(fatigued_until))} ({remaining_hours:.1f}h remaining)")
    
    # Fatigue expired - ready to battle!
    return (False, None)


def _log_godzilla_incident(
    region: str,
    reason: str,
    fatigue_type: str,
    fatigue_duration_hours: int,
    failure_count_today: int,
    error_message: str,
    build_id: str
):
    """
    Log Godzilla incident to historical log file using SafeJSON.

    Creates a permanent record of ALL fatigues for analysis.
    SafeJSON handles corruption, backups, and atomic writes automatically.
    """
    import datetime
    from pathlib import Path

    # Godzilla incidents file path
    data_dir = Path(__file__).parent / "data"
    incidents_file = data_dir / "godzilla_incidents.json"

    # Create incident record
    now = time.time()
    incident = {
        "timestamp": now,
        "date_human": datetime.datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S"),
        "region": region,
        "reason": reason,
        "fatigue_type": fatigue_type,  # FATIGUED or EXHAUSTED
        "fatigue_duration_hours": fatigue_duration_hours,
        "failure_count_today": failure_count_today,
        "error_message": error_message if error_message else "No error details provided",
        "build_id": build_id if build_id else "N/A"
    }

    # Load existing incidents using SafeJSON (returns {} if missing/corrupt)
    data = SafeJSON.read(incidents_file)

    # Initialize structure if empty
    if not data or "incidents" not in data:
        data = {"incidents": []}

    # Append new incident
    data["incidents"].append(incident)

    # Save using SafeJSON (atomic write + 20 versioned backups)
    if not SafeJSON.write(incidents_file, data):
        print(f"ERROR: Failed to log Godzilla incident: {region} - {reason}")
        print(f"Lost incident: {incident}")
    # else: Success! SafeJSON handled everything automatically


def record_mecha_timeout(
    registry: Dict,
    region: str,
    reason: str = "Queue timeout",
    reason_code: str = None,
    error_message: str = "",
    build_id: str = ""
):
    """
    Record that MECHA timed out (15 min deployment failure).

    Applies fatigue rules:
    - 1st timeout → 4 hours fatigue
    - 2nd timeout → 4 hours fatigue
    - 3rd timeout in 24h period → 24 hours fatigue (EXHAUSTED!)

    Args:
        registry: MECHA registry dict
        region: Region name
        reason: Human-readable reason for timeout
        reason_code: Machine-readable reason code (campaign_stats.REASON_*)
        error_message: Full error message (optional, can be long)
        build_id: CloudBuild ID (optional)
    """
    now = time.time()

    if "mechas" not in registry:
        registry["mechas"] = {}

    mecha_info = registry["mechas"].get(region, {})

    # Get or initialize failure tracking
    failures_today = mecha_info.get("failures_today", [])

    # Clean out failures older than 24 hours
    cutoff = now - (24 * 3600)
    failures_today = [ts for ts in failures_today if ts > cutoff]

    # Add this failure
    failures_today.append(now)

    # Determine fatigue duration based on failure count
    failure_count = len(failures_today)

    if failure_count >= 3:
        # 3rd failure in 24h → EXHAUSTED! 24 hours rest (whole day)
        fatigue_hours = 24
        fatigue_message = "EXHAUSTED"
    else:
        # 1st or 2nd failure → FATIGUED 4 hours
        fatigue_hours = 4
        fatigue_message = "FATIGUED"

    fatigued_until = now + (fatigue_hours * 3600)

    # Update MECHA info
    mecha_info.update({
        "operational_status": "NONOPERATIONAL",
        "last_attempt": now,
        "failures_today": failures_today,
        "failure_count_today": failure_count,
        "fatigued_until": fatigued_until,
        "fatigue_message": f"{fatigue_message} for {fatigue_hours}h",
        "last_failure_reason": reason,  # Human-readable reason
        "last_failure_reason_code": reason_code  # Machine-readable classification
    })

    registry["mechas"][region] = mecha_info

    # LOG TO GODZILLA INCIDENTS HISTORY!
    _log_godzilla_incident(
        region=region,
        reason=reason,
        fatigue_type=fatigue_message,
        fatigue_duration_hours=fatigue_hours,
        failure_count_today=failure_count,
        error_message=error_message,
        build_id=build_id
    )

    # RECORD TO CAMPAIGN STATS!
    from .campaign_stats import record_fatigue_event, REASON_UNKNOWN
    # Default to REASON_UNKNOWN if no code provided (backward compatibility)
    final_reason_code = reason_code if reason_code else REASON_UNKNOWN
    record_fatigue_event(region, reason, final_reason_code, error_message)

    return (failure_count, fatigue_hours, fatigue_message)


def get_available_mechas(registry: Dict, all_regions: List[str]) -> Tuple[List[str], List[str]]:
    """
    Get MECHAs available for deployment (not fatigued).
    
    Returns:
        (available_regions, fatigued_regions)
    """
    available = []
    fatigued = []
    
    for region in all_regions:
        mecha_info = registry.get("mechas", {}).get(region, {})
        
        # Check if MECHA is already RUNNING
        if mecha_info.get("operational_status") == "OPERATIONAL":
            continue
        
        # Check if fatigued
        is_fatigued, _ = is_mecha_fatigued(mecha_info)
        
        if is_fatigued:
            fatigued.append(region)
        else:
            available.append(region)
    
    return (available, fatigued)


def get_fatigue_status_display(registry: Dict) -> str:
    """
    Get human-readable fatigue status for MECHA Hangar display.
    
    Returns ASCII art showing fatigued MECHAs with recovery times.
    """
    mechas = registry.get("mechas", {})
    fatigued_mechas = []
    
    for region, info in mechas.items():
        is_fatigued, message = is_mecha_fatigued(info)
        if is_fatigued:
            failure_count = info.get("failure_count_today", 0)
            if failure_count >= 3:
                emoji = "🛌"  # Exhausted
            else:
                emoji = "😴"  # Fatigued
            
            fatigued_mechas.append(f"     {emoji} {region}: {message}")

    if not fatigued_mechas:
        return ""

    lines = ["\n   😴 FATIGUED MECHAS:"] + fatigued_mechas
    return "\n".join(lines)
