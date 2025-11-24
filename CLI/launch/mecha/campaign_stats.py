"""
MECHA Campaign Stats Tracker - Simple Build Metrics

Tracks build performance with:
- All-time aggregates (fast lookups)
- Last 100 builds per region with basic timing
- 30/60/90 day windowing calculated on-read
- RUN-based CHONK markers (in logs only, not parsed)

File: CLI/launch/mecha/data/campaign_stats.json
"""

# <claudes_code_comments>
# ** Function List **
# load_campaign_stats() - Load campaign stats using SafeJSON (handles corruption automatically)
# save_campaign_stats(stats) - Save with SafeJSON (atomic writes + file locking + backups)
# get_queue_metrics(build_id, region) - Get queue/working seconds from CloudBuild API
# get_build_timing(build_id, region) - Parse CloudBuild timing (build/push duration)
# record_build_result(...) - Record new build with basic timing
# update_pending_build(region, build_id, status) - Update build_id="pending" records
# update_build_completion(...) - Update existing build with final stats
# record_fatigue_event(...) - Track region fatigue incidents
# get_recent_builds(...) - Filter builds by time window
# get_windowed_stats(region, days) - Calculate 30/60/90 day metrics
# get_all_windowed_stats(...) - Get multi-window stats for all regions
#
# ** Technical Review **
# Simple build tracking with RUN-based CHONK markers:
# 1. record_build_result() or update_build_completion() triggered after build
# 2. Calls get_build_timing() to extract CloudBuild timing
# 3. get_build_timing() flow:
#    - Calls `gcloud builds describe` to get timing.BUILD and timing.PUSH
#    - Extracts start/end times, calculates duration in minutes
#    - Returns: build_duration_minutes, push_duration_minutes, timestamps
# 4. CHONK markers (💎🔷🔶) are RUN-based - inline with Dockerfile work steps
#    - Visible in CloudBuild logs for progress tracking
#    - NOT parsed or stored in JSON (too complex, not needed)
# 5. Cost estimation: Simple hourly_rate × duration calculation
#    - Field: cost_estimated_usd (likely inaccurate, more research needed)
#    - Doesn't account for: spot preemption, network egress, storage, GCP billing complexity
#
# Data storage: campaign_stats.json contains:
# - stats["regions"][region]["recent_builds"] = last 100 builds with basic metrics
# - Each build has: timestamps, duration, success/fail, estimated cost
#
# Error handling: All parsing wrapped in try/except, non-blocking.
# If parsing fails, build still recorded with basic data (backward compatible).
#
# File I/O: Uses SafeJSON for production-grade JSON operations (atomic writes, file locking,
# 20 versioned backups, auto-corruption detection). Returns {} if file missing/corrupt.
# </claudes_code_comments>

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Import SafeJSON for production-grade JSON I/O
from ...shared.safe_json import SafeJSON

# No CHONK parsing - markers are RUN-based (visible in logs only)


# Campaign stats file location
CAMPAIGN_STATS_FILE = Path(__file__).parent / "data" / "campaign_stats.json"

# Keep last N builds per region for windowing
MAX_RECENT_BUILDS = 100


# ============================================================================
# FATIGUE REASON CODES (Formal Classification System)
# ============================================================================
# Machine-readable codes for fatigue incidents.
# Enable smart filtering, beacon optimization, and analytics.

# Current MECHA fatigue types (expand as needed)
REASON_QUEUE_TIMEOUT = "queue-timeout"        # 45-min QUEUED state (Queue Godzilla)
REASON_BEACON_TIMEOUT = "beacon-timeout"      # Pool creation timeout (5-min beacon)
REASON_UNKNOWN = "unknown"                    # Unclassified fatigue (fallback)

# All valid reason codes (for validation)
VALID_REASON_CODES = {
    REASON_QUEUE_TIMEOUT,
    REASON_BEACON_TIMEOUT,
    REASON_UNKNOWN,
}


def load_campaign_stats() -> Dict[str, Any]:
    """
    Load campaign stats from JSON file using SafeJSON.

    Returns:
        Stats dict with structure:
        {
            "campaign_start": timestamp,
            "last_updated": timestamp,
            "total_builds_all_regions": int,
            "regions": {
                "us-west2": {
                    # All-time aggregates
                    "total_builds": int,
                    "successes": int,
                    "failures": int,
                    "success_rate": float,
                    ...
                    # Rich build history
                    "recent_builds": [
                        {
                            "timestamp": float,
                            "build_id": str,
                            "build_type": str,
                            "machine_type": str,
                            "success": bool,
                            "status": str,
                            "duration_seconds": int,
                            "duration_minutes": float,
                            "queue_wait_seconds": int,
                            "queue_wait_minutes": float,
                            "working_seconds": int,
                            "total_steps": int,
                            "failed_step": int or None,
                            "spot_price_per_hour": float,
                            "cost_estimated_usd": float,
                            "error": str or None,
                            "timeout_reason": str or None
                        }
                    ]
                }
            }
        }
    """
    # SafeJSON handles all error cases and returns {} if file doesn't exist or is corrupt
    stats = SafeJSON.read(CAMPAIGN_STATS_FILE)

    # If empty (new file or corruption), initialize with defaults
    if not stats:
        stats = {
            "campaign_start": time.time(),
            "last_updated": time.time(),
            "total_builds_all_regions": 0,
            "regions": {}
        }

    return stats


def save_campaign_stats(stats: Dict[str, Any], max_retries: int = 3) -> bool:
    """
    Save campaign stats to JSON file using SafeJSON.

    Args:
        stats: Stats dictionary
        max_retries: Max retry attempts (kept for backward compatibility, SafeJSON handles retries internally)

    Returns:
        True if saved successfully
    """
    # SafeJSON handles all locking, atomic writes, and backups automatically
    return SafeJSON.write(CAMPAIGN_STATS_FILE, stats)


def _extract_concise_error(error_message: str, max_chars: int = 80) -> str:
    """
    Extract concise error message from full error text.

    Args:
        error_message: Full error message (can be multi-line)
        max_chars: Max length of concise error

    Returns:
        Concise error string
    """
    if not error_message:
        return "Unknown error"

    # Take first line only
    first_line = error_message.split('\n')[0].strip()

    # Look for common error patterns
    error_markers = [
        "ERROR:", "Error:", "FAILED:", "Failed:",
        "TIMEOUT:", "Timeout:", "Exception:"
    ]

    for marker in error_markers:
        if marker in first_line:
            parts = first_line.split(marker, 1)
            if len(parts) > 1:
                first_line = parts[1].strip()
                break

    # Truncate if too long
    if len(first_line) > max_chars:
        first_line = first_line[:max_chars-3] + "..."

    return first_line


def get_queue_metrics(build_id: str, region: str) -> dict:
    """
    Fetch detailed timing metrics from CloudBuild API for a completed build.

    Returns dict with timing breakdown, log URL, worker pool, etc.
    Returns empty dict if fetch fails (non-blocking).
    """
    try:
        import subprocess
        result = subprocess.run(
            [
                "gcloud", "builds", "describe", build_id,
                f"--region={region}",
                "--format=json"
            ],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            return {}

        import json
        from datetime import datetime
        data = json.loads(result.stdout)

        # Parse timing breakdown
        timing = data.get("timing", {})
        metrics = {}

        # Calculate queue wait and working time
        create_time = datetime.fromisoformat(data["createTime"].replace("Z", "+00:00"))
        start_time = datetime.fromisoformat(data["startTime"].replace("Z", "+00:00"))
        finish_time = datetime.fromisoformat(data["finishTime"].replace("Z", "+00:00"))

        metrics["queue_wait_seconds"] = int((start_time - create_time).total_seconds())
        metrics["working_seconds"] = int((finish_time - start_time).total_seconds())

        # Extract phase timings
        for phase in ["FETCHSOURCE", "BUILD", "PUSH"]:
            if phase in timing:
                phase_start = datetime.fromisoformat(timing[phase]["startTime"].replace("Z", "+00:00"))
                phase_end = datetime.fromisoformat(timing[phase]["endTime"].replace("Z", "+00:00"))
                duration = int((phase_end - phase_start).total_seconds())

                if phase == "FETCHSOURCE":
                    metrics["fetch_source_seconds"] = duration
                elif phase == "BUILD":
                    metrics["build_phase_seconds"] = duration
                elif phase == "PUSH":
                    metrics["push_image_seconds"] = duration

        # Extract log URL and worker pool
        metrics["log_url"] = data.get("logUrl")
        pool = data.get("options", {}).get("pool", {})
        metrics["worker_pool"] = pool.get("name")

        return metrics

    except Exception as e:
        # Non-blocking - if metrics fetch fails, just return empty dict
        print(f"⚠️  Could not fetch CloudBuild metrics: {e}")
        return {}


def get_build_timing(build_id: str, region: str) -> dict:
    """
    Get build timing from CloudBuild metadata.

    Extracts timing data from CloudBuild's metadata JSON (BUILD, PUSH phases).
    Returns timing metrics (build duration, push duration, timestamps).
    Returns empty dict if parse fails (non-blocking).
    """
    try:
        import subprocess
        import json
        from datetime import datetime

        metrics = {}

        # ═══════════════════════════════════════════════════════════════════
        # PART 1: CloudBuild Metadata Timing (from gcloud builds describe)
        # ═══════════════════════════════════════════════════════════════════
        result = subprocess.run(
            ["gcloud", "builds", "describe", build_id, f"--region={region}", "--format=json"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            build_metadata = json.loads(result.stdout)
            timing = build_metadata.get("timing", {})

            # Extract BUILD phase timing
            build_phase = timing.get("BUILD", {})
            if build_phase:
                start = datetime.fromisoformat(build_phase["startTime"].replace("Z", "+00:00"))
                end = datetime.fromisoformat(build_phase["endTime"].replace("Z", "+00:00"))
                build_duration_min = (end - start).total_seconds() / 60
                metrics["build_duration_minutes"] = round(build_duration_min, 2)
                metrics["build_start_time"] = build_phase["startTime"]
                metrics["build_end_time"] = build_phase["endTime"]

            # Extract PUSH phase timing
            push_phase = timing.get("PUSH", {})
            if push_phase:
                start = datetime.fromisoformat(push_phase["startTime"].replace("Z", "+00:00"))
                end = datetime.fromisoformat(push_phase["endTime"].replace("Z", "+00:00"))
                push_duration_min = (end - start).total_seconds() / 60
                metrics["push_duration_minutes"] = round(push_duration_min, 2)

        # CHONK markers (💎🔷🔶) are RUN-based - inline with Dockerfile work steps
        # Example: RUN git clone ... && echo "🔹CHONK: [10%] Git clone MUHNCHED! 💎"
        # Visible in CloudBuild logs for progress tracking, not parsed/stored here

        return metrics

    except Exception as e:
        # Non-blocking - if parsing fails, just return empty dict
        print(f"⚠️  Could not parse build timing: {e}")
        return {}


def record_build_result(
    region: str,
    success: bool,
    duration_minutes: float,
    queue_wait_minutes: float = 0.0,
    error_message: Optional[str] = None,
    # Rich build details
    build_id: Optional[str] = None,
    build_type: Optional[str] = None,
    machine_type: Optional[str] = None,
    status: Optional[str] = None,
    duration_seconds: Optional[int] = None,
    queue_wait_seconds: Optional[int] = None,
    working_seconds: Optional[int] = None,
    total_steps: Optional[int] = None,
    failed_step: Optional[int] = None,
    spot_price_per_hour: Optional[float] = None,
    timeout_reason: Optional[str] = None,
    fatigues: Optional[List[Dict]] = None
) -> Tuple[int, float]:
    """
    Record a build result with rich details.

    Updates:
    - All-time aggregates (fast stats)
    - Adds to recent_builds array (windowing + rich details)

    Args:
        region: Region name
        success: True if build succeeded
        duration_minutes: Total build duration
        queue_wait_minutes: Time in queue
        error_message: Error if failed

        # Rich details (optional):
        build_id: Cloud Build ID
        build_type: "arr-pytorch-base", "arr-ml-stack", "arr-trainer", "arr-vertex-launcher"
        machine_type: "c3-standard-176"
        status: "SUCCESS", "FAILURE", "TIMEOUT"
        duration_seconds: Precise duration
        queue_wait_seconds: Precise queue time
        working_seconds: Precise working time
        total_steps: Total Docker steps
        failed_step: Which step failed
        spot_price_per_hour: Spot price used
        timeout_reason: "QUEUED" or "WORKING"

    Returns:
        (total_builds, success_rate) for region
    """
    now = time.time()

    # Load current stats
    stats = load_campaign_stats()

    # Initialize region if new
    if region not in stats["regions"]:
        stats["regions"][region] = {
            "total_builds": 0,
            "successes": 0,
            "failures": 0,
            "success_rate": 0.0,

            "total_duration_minutes": 0.0,
            "total_queue_wait_minutes": 0.0,
            "avg_duration_minutes": 0.0,
            "avg_queue_wait_minutes": 0.0,
            "fastest_minutes": None,
            "slowest_minutes": None,

            "last_error": None,
            "last_error_time": None,

            "fatigue_incidents": 0,
            "last_fatigue_reason": None,
            "last_fatigue_reason_code": None,
            "last_fatigue_time": None,

            "current_streak": 0,
            "last_used": None,

            "recent_builds": []  # Rich build history
        }

    region_stats = stats["regions"][region]

    # ═══════════════════════════════════════════════════════════
    # Update All-Time Aggregates (Fast Stats)
    # ═══════════════════════════════════════════════════════════
    region_stats["total_builds"] += 1
    stats["total_builds_all_regions"] += 1

    if success:
        region_stats["successes"] += 1
        # Update streak (positive = wins)
        if region_stats["current_streak"] >= 0:
            region_stats["current_streak"] += 1
        else:
            region_stats["current_streak"] = 1
    else:
        region_stats["failures"] += 1
        # Update streak (negative = losses)
        if region_stats["current_streak"] <= 0:
            region_stats["current_streak"] -= 1
        else:
            region_stats["current_streak"] = -1

        # Record error
        if error_message:
            region_stats["last_error"] = _extract_concise_error(error_message)
            region_stats["last_error_time"] = now

    # Success rate
    region_stats["success_rate"] = round(
        (region_stats["successes"] / region_stats["total_builds"]) * 100, 1
    )

    # Duration stats
    region_stats["total_duration_minutes"] += duration_minutes
    region_stats["total_queue_wait_minutes"] += queue_wait_minutes
    region_stats["avg_duration_minutes"] = round(
        region_stats["total_duration_minutes"] / region_stats["total_builds"], 1
    )
    region_stats["avg_queue_wait_minutes"] = round(
        region_stats["total_queue_wait_minutes"] / region_stats["total_builds"], 1
    )

    # Fastest/slowest
    if region_stats["fastest_minutes"] is None or duration_minutes < region_stats["fastest_minutes"]:
        region_stats["fastest_minutes"] = round(duration_minutes, 1)
    if region_stats["slowest_minutes"] is None or duration_minutes > region_stats["slowest_minutes"]:
        region_stats["slowest_minutes"] = round(duration_minutes, 1)

    region_stats["last_used"] = now

    # ═══════════════════════════════════════════════════════════
    # Add to Recent Builds (Rich Details for Windowing)
    # ═══════════════════════════════════════════════════════════

    # Calculate cost estimate (SIMPLE APPROXIMATION - likely inaccurate!)
    # Just: hourly_rate × duration
    # Doesn't account for: spot preemption overhead, network egress,
    # Artifact Registry storage, or GCP billing model complexity
    # More research needed for accurate cost tracking
    cost_estimated_usd = None
    if spot_price_per_hour and duration_seconds:
        cost_estimated_usd = round((spot_price_per_hour * duration_seconds) / 3600, 3)

    # Fetch rich CloudBuild metrics if we have a build_id
    cloudbuild_metrics = {}
    resource_metrics = {}
    if build_id and build_id != "pending":
        cloudbuild_metrics = get_queue_metrics(build_id, region)
        resource_metrics = get_build_timing(build_id, region)

    build_record = {
        "timestamp": now,
        "build_id": build_id,
        "build_type": build_type,
        "machine_type": machine_type,

        "success": success,
        "status": status or ("SUCCESS" if success else "FAILURE"),

        "duration_seconds": duration_seconds or int(duration_minutes * 60),
        "duration_minutes": round(duration_minutes, 1),
        "queue_wait_seconds": cloudbuild_metrics.get("queue_wait_seconds") or queue_wait_seconds or int(queue_wait_minutes * 60),
        "queue_wait_minutes": round((cloudbuild_metrics.get("queue_wait_seconds") or queue_wait_seconds or int(queue_wait_minutes * 60)) / 60, 1),
        "working_seconds": cloudbuild_metrics.get("working_seconds") or working_seconds,

        # Phase timing from CloudBuild API
        "fetch_source_seconds": cloudbuild_metrics.get("fetch_source_seconds", 0),
        "build_phase_seconds": cloudbuild_metrics.get("build_phase_seconds", 0),
        "push_image_seconds": cloudbuild_metrics.get("push_image_seconds", 0),
        "log_url": cloudbuild_metrics.get("log_url"),
        "worker_pool": cloudbuild_metrics.get("worker_pool"),

        # Build timing from CloudBuild metadata
        "build_duration_minutes": resource_metrics.get("build_duration_minutes", 0),
        "build_start_time": resource_metrics.get("build_start_time", ""),
        "build_end_time": resource_metrics.get("build_end_time", ""),
        "push_duration_minutes": resource_metrics.get("push_duration_minutes", 0),

        "total_steps": total_steps,
        "failed_step": failed_step,

        "spot_price_per_hour": spot_price_per_hour,
        "cost_estimated_usd": cost_estimated_usd,

        "error": _extract_concise_error(error_message) if error_message else None,
        "timeout_reason": timeout_reason,

        "fatigues": fatigues or []
    }

    # Add to recent builds
    region_stats["recent_builds"].append(build_record)

    # Keep only last MAX_RECENT_BUILDS
    if len(region_stats["recent_builds"]) > MAX_RECENT_BUILDS:
        region_stats["recent_builds"] = region_stats["recent_builds"][-MAX_RECENT_BUILDS:]

    # Update timestamps
    stats["last_updated"] = now

    # Save updated stats
    save_campaign_stats(stats)

    return (region_stats["total_builds"], region_stats["success_rate"])


def update_pending_build(
    region: str,
    build_id: str,
    status: str = "QUEUED"
):
    """
    Update the pending build record with actual CloudBuild ID.

    Finds the most recent build with build_id="pending" OR matching build_id and updates status.

    Args:
        region: GCP region
        build_id: Actual CloudBuild ID
        status: New status (default: "QUEUED")
    """

    stats = load_campaign_stats()

    # Initialize region if it doesn't exist yet (CRITICAL FIX!)
    if region not in stats["regions"]:
        stats["regions"][region] = {
            "total_builds": 0,
            "successes": 0,
            "failures": 0,
            "success_rate": 0.0,
            "total_duration_minutes": 0.0,
            "total_queue_wait_minutes": 0.0,
            "avg_duration_minutes": 0.0,
            "avg_queue_wait_minutes": 0.0,
            "fastest_minutes": None,
            "slowest_minutes": None,
            "last_error": None,
            "last_error_time": None,
            "fatigue_incidents": 0,
            "last_fatigue_reason": None,
            "last_fatigue_reason_code": None,
            "last_fatigue_time": None,
            "current_streak": 0,
            "last_used": None,
            "recent_builds": []
        }

    region_stats = stats["regions"][region]
    recent_builds = region_stats.get("recent_builds", [])

    # Find most recent pending build OR matching build_id
    found = False
    for build in reversed(recent_builds):
        # Match either "pending" (first call) or actual build_id (subsequent calls)
        if build.get("build_id") == "pending" or build.get("build_id") == build_id:
            old_status = build.get("status")
            # Update with actual build ID and status
            build["build_id"] = build_id
            build["status"] = status
            found = True
            break

    if not found:
        pass  # Build not found (possibly already completed or error state)

    # Save updated stats
    save_campaign_stats(stats)


def update_build_completion(
    region: str,
    build_id: str,
    success: bool,
    duration_minutes: float,
    queue_wait_minutes: float = 0.0,
    error_message: Optional[str] = None,
    duration_seconds: Optional[int] = None,
    queue_wait_seconds: Optional[int] = None,
    working_seconds: Optional[int] = None,
    total_steps: Optional[int] = None,
    failed_step: Optional[int] = None,
    spot_price_per_hour: Optional[float] = None,
    timeout_reason: Optional[str] = None,
    fatigues: Optional[List[Dict]] = None
):
    """
    Update existing build record with completion data (success/failure).

    Finds build by ID and updates with final status, durations, errors, etc.
    Also updates region aggregates (success_rate, avg_duration, etc.).

    Args:
        region: GCP region
        build_id: CloudBuild ID to find and update
        success: Whether build succeeded
        duration_minutes: Total build duration
        queue_wait_minutes: Time spent in QUEUED state
        error_message: Error message if failed
        (... other rich details ...)
    """
    stats = load_campaign_stats()

    if region not in stats["regions"]:
        return  # No region entry

    region_stats = stats["regions"][region]
    recent_builds = region_stats.get("recent_builds", [])

    # Fetch rich CloudBuild and resource metrics if we have a valid build_id
    cloudbuild_metrics = {}
    resource_metrics = {}
    if build_id and build_id != "pending":
        cloudbuild_metrics = get_queue_metrics(build_id, region)
        resource_metrics = get_build_timing(build_id, region)

    # Find build by ID
    build_found = False
    for build in recent_builds:
        if build.get("build_id") == build_id:
            # Update with completion data
            build["success"] = success
            build["status"] = "SUCCESS" if success else "FAILURE"
            build["duration_minutes"] = duration_minutes
            build["queue_wait_minutes"] = queue_wait_minutes

            if duration_seconds is not None:
                build["duration_seconds"] = duration_seconds
            if queue_wait_seconds is not None:
                build["queue_wait_seconds"] = queue_wait_seconds
            if working_seconds is not None:
                build["working_seconds"] = working_seconds
            if total_steps is not None:
                build["total_steps"] = total_steps
            if failed_step is not None:
                build["failed_step"] = failed_step
            if spot_price_per_hour is not None:
                build["spot_price_per_hour"] = spot_price_per_hour
                # Calculate cost
                hours = duration_minutes / 60.0
                build["cost_usd"] = round(spot_price_per_hour * hours, 4)
            if error_message:
                build["error"] = error_message
            if timeout_reason:
                build["timeout_reason"] = timeout_reason
            if fatigues:
                build["fatigues"] = fatigues

            # Enrich with CloudBuild timing metrics (if available)
            if cloudbuild_metrics:
                build["queue_wait_seconds"] = cloudbuild_metrics.get("queue_wait_seconds", build.get("queue_wait_seconds", 0))
                build["queue_wait_minutes"] = round(build["queue_wait_seconds"] / 60, 1)
                build["working_seconds"] = cloudbuild_metrics.get("working_seconds", build.get("working_seconds", 0))
                build["fetch_source_seconds"] = cloudbuild_metrics.get("fetch_source_seconds", 0)
                build["build_phase_seconds"] = cloudbuild_metrics.get("build_phase_seconds", 0)
                build["push_image_seconds"] = cloudbuild_metrics.get("push_image_seconds", 0)
                build["log_url"] = cloudbuild_metrics.get("log_url")
                build["worker_pool"] = cloudbuild_metrics.get("worker_pool")

            # Enrich with timing + CHONK metrics (if available)
            if resource_metrics:
                # Build timing from CloudBuild metadata
                build["build_duration_minutes"] = resource_metrics.get("build_duration_minutes", 0)
                build["build_start_time"] = resource_metrics.get("build_start_time", "")
                build["build_end_time"] = resource_metrics.get("build_end_time", "")
                build["push_duration_minutes"] = resource_metrics.get("push_duration_minutes", 0)

            build_found = True
            break

    if not build_found:
        return  # Build not found (shouldn't happen)

    # Update region aggregates
    region_stats["total_builds"] = len(recent_builds)
    region_stats["successes"] = sum(1 for b in recent_builds if b.get("success"))
    region_stats["failures"] = sum(1 for b in recent_builds if not b.get("success"))
    region_stats["success_rate"] = region_stats["successes"] / region_stats["total_builds"] if region_stats["total_builds"] > 0 else 0.0

    # Update duration averages
    durations = [b["duration_minutes"] for b in recent_builds if b.get("duration_minutes", 0) > 0]
    if durations:
        region_stats["avg_duration_minutes"] = sum(durations) / len(durations)
        region_stats["fastest_minutes"] = min(durations)
        region_stats["slowest_minutes"] = max(durations)

    # Update queue wait averages
    queue_waits = [b["queue_wait_minutes"] for b in recent_builds if b.get("queue_wait_minutes", 0) > 0]
    if queue_waits:
        region_stats["avg_queue_wait_minutes"] = sum(queue_waits) / len(queue_waits)

    # Update error tracking
    if error_message:
        region_stats["last_error"] = error_message
        region_stats["last_error_time"] = time.time()

    # Update streak
    if success:
        if region_stats.get("current_streak", 0) >= 0:
            region_stats["current_streak"] = region_stats.get("current_streak", 0) + 1
        else:
            region_stats["current_streak"] = 1
    else:
        if region_stats.get("current_streak", 0) <= 0:
            region_stats["current_streak"] = region_stats.get("current_streak", 0) - 1
        else:
            region_stats["current_streak"] = -1

    region_stats["last_used"] = time.time()

    # Update global stats
    stats["total_builds_all_regions"] = sum(r["total_builds"] for r in stats["regions"].values())
    stats["last_updated"] = time.time()

    # Save updated stats
    save_campaign_stats(stats)


def record_fatigue_event(
    region: str,
    reason: str,
    reason_code: str,
    error_message: Optional[str] = None
):
    """
    Record a fatigue/Godzilla incident.

    Args:
        region: Region name
        reason: Human-readable reason ("Queue timeout - 45 minutes in QUEUED state")
        reason_code: Machine-readable code (REASON_QUEUE_TIMEOUT, REASON_BEACON_TIMEOUT, etc.)
        error_message: Full error message (optional, will be made concise)
    """
    now = time.time()

    # Validate reason_code
    if reason_code not in VALID_REASON_CODES:
        # Invalid code - fall back to REASON_UNKNOWN
        reason_code = REASON_UNKNOWN

    stats = load_campaign_stats()

    # Initialize region if new
    if region not in stats["regions"]:
        stats["regions"][region] = {
            "total_builds": 0,
            "successes": 0,
            "failures": 0,
            "success_rate": 0.0,

            "total_duration_minutes": 0.0,
            "total_queue_wait_minutes": 0.0,
            "avg_duration_minutes": 0.0,
            "avg_queue_wait_minutes": 0.0,
            "fastest_minutes": None,
            "slowest_minutes": None,

            "last_error": None,
            "last_error_time": None,

            "fatigue_incidents": 0,
            "last_fatigue_reason": None,
            "last_fatigue_reason_code": None,
            "last_fatigue_time": None,

            "current_streak": 0,
            "last_used": None,

            "recent_builds": []
        }

    region_stats = stats["regions"][region]

    # Record fatigue
    region_stats["fatigue_incidents"] += 1
    region_stats["last_fatigue_reason"] = _extract_concise_error(
        error_message if error_message else reason, max_chars=60
    )
    region_stats["last_fatigue_reason_code"] = reason_code
    region_stats["last_fatigue_time"] = now

    stats["last_updated"] = now

    save_campaign_stats(stats)


def get_stats_for_window(region: str, days: Optional[int] = None) -> Dict[str, Any]:
    """
    Get stats for a time window.

    Args:
        region: Region name
        days: Days to look back (None = all-time)

    Returns:
        Stats dict for the window
    """
    stats = load_campaign_stats()

    if region not in stats["regions"]:
        return None

    region_stats = stats["regions"][region]

    # All-time: return pre-calculated aggregates
    if days is None:
        return region_stats

    # Windowed: calculate from recent_builds
    cutoff = time.time() - (days * 86400)
    recent = [b for b in region_stats["recent_builds"] if b["timestamp"] > cutoff]

    if not recent:
        return None

    # Calculate window stats
    successes = sum(1 for b in recent if b["success"])
    failures = len(recent) - successes

    return {
        "total_builds": len(recent),
        "successes": successes,
        "failures": failures,
        "success_rate": round((successes / len(recent)) * 100, 1) if recent else 0.0,
        "avg_duration_minutes": round(sum(b["duration_minutes"] for b in recent) / len(recent), 1),
        "avg_queue_wait_minutes": round(sum(b["queue_wait_minutes"] for b in recent) / len(recent), 1),
        "fastest_minutes": round(min(b["duration_minutes"] for b in recent), 1),
        "slowest_minutes": round(max(b["duration_minutes"] for b in recent), 1),
        "window_days": days
    }


def get_top_regions(metric: str = "success_rate", limit: int = 5) -> List[Tuple[str, float]]:
    """
    Get top regions by metric.

    Args:
        metric: "success_rate", "avg_duration_minutes", "total_builds"
        limit: Number of results

    Returns:
        [(region, value), ...] sorted by metric
    """
    stats = load_campaign_stats()

    regions = []
    for region, data in stats["regions"].items():
        if data["total_builds"] > 0:
            regions.append((region, data.get(metric, 0)))

    # Sort (descending for success_rate/total_builds, ascending for duration)
    reverse = metric != "avg_duration_minutes"
    sorted_regions = sorted(regions, key=lambda x: x[1], reverse=reverse)

    return sorted_regions[:limit]
