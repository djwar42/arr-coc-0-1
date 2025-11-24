"""
Zeus Campaign Stats Tracker - Training Job Metrics

Tracks Vertex AI training job performance with:
- All-time aggregates (fast lookups)
- Last 100 jobs per region with GPU metrics
- Cost savings tracking (Zeus champion vs most expensive region)
- Divine wrath incidents (GPU preemptions)
- 30/60/90 day windowing calculated on-read

File: CLI/launch/zeus/data/campaign_stats.json

NOTE: CURRENTLY WRITES DUMMY DATA - Will be populated with real Vertex AI job data later
"""

# <claudes_code_comments>
# ** Function List **
# load_campaign_stats() - Load campaign stats using SafeJSON (handles corruption automatically)
# save_campaign_stats(stats) - Save with SafeJSON (atomic writes + file locking + backups)
# record_job_result(...) - Record new training job with GPU metrics (DUMMY DATA for now)
# record_wrath_event(...) - Track divine wrath incidents (mirrors MECHA's record_fatigue_event)
# get_recent_jobs(...) - Filter jobs by time window
# get_stats_for_window(region, days) - Calculate 30/60/90 day metrics (mirrors MECHA)
# get_top_regions(metric, limit) - Get top N regions by metric (mirrors MECHA)
#
# ** Technical Review **
# Zeus campaign system tracks Vertex AI training jobs (not Cloud Build jobs like MECHA).
# Mirrors MECHA's pattern but adapted for GPU training metrics:
#
# 1. Job tracking: record_job_result() logs training job completion
#    - GPU type, count, tier (spark/bolt/storm/tempest/cataclysm)
#    - Training duration, success/failure status
#    - Spot pricing, total cost, savings vs most expensive region
#    - Job ID, log URL for Vertex AI console
#
# 2. Divine wrath: GPU preemptions tracked separately (like MECHA fatigue)
#    - record_divine_wrath_event() logs preemption incidents
#    - Tracks 3-strike escalation (DISFAVORED 4h → ZEUS'S WRATH 24h)
#    - Links to zeus_olympus.py wrath system
#
# 3. Cost tracking: Zeus-specific metrics
#    - total_cost_usd: Actual training cost (spot_price × duration × gpu_count)
#    - savings_vs_most_expensive_usd: How much Zeus saved by picking champion
#    - battle_wins: How many times this region won the pricing battle
#
# 4. Data storage: zeus_campaign_stats.json contains:
#    - stats["regions"][region]["recent_jobs"] = last 100 jobs with GPU metrics
#    - Each job has: timestamp, GPU config, duration, cost, savings, preemption status
#
# 5. CURRENT STATUS: DUMMY DATA ONLY!
#    - save_campaign_stats() writes to zeus_campaign_stats.json
#    - record_job_result() accepts parameters but writes dummy data
#    - Will be replaced with real Vertex AI job tracking later (when we monitor training runs)
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


# Campaign stats file location
CAMPAIGN_STATS_FILE = Path(__file__).parent / "data" / "campaign_stats.json"

# Keep last N jobs per region for windowing
MAX_RECENT_JOBS = 100


# ============================================================================
# DIVINE WRATH REASON CODES (GPU Preemption Classification)
# ============================================================================
# Machine-readable codes for divine wrath incidents.
# Enable smart filtering, region selection optimization, and analytics.

# Current Zeus wrath types (expand as needed)
REASON_SPOT_PREEMPTION = "spot-preemption"       # Spot instance preempted by GCP
REASON_GPU_PREEMPTED = "gpu-preempted"          # Spot GPU preempted during training
REASON_QUOTA_EXCEEDED = "quota-exceeded"         # GPU quota exceeded during launch
REASON_ZONE_EXHAUSTION = "zone-exhaustion"       # Zone capacity exhausted (no resources)
REASON_OOM_ERROR = "oom-error"                   # CUDA out of memory
REASON_TRAINING_FAILED = "training-failed"       # Generic training failure
REASON_UNKNOWN = "unknown"                       # Unclassified wrath (fallback)

# All valid reason codes (for validation)
VALID_REASON_CODES = {
    REASON_SPOT_PREEMPTION,
    REASON_GPU_PREEMPTED,
    REASON_QUOTA_EXCEEDED,
    REASON_ZONE_EXHAUSTION,
    REASON_OOM_ERROR,
    REASON_TRAINING_FAILED,
    REASON_UNKNOWN,
}


def load_campaign_stats() -> Dict[str, Any]:
    """
    Load Zeus campaign stats from JSON file using SafeJSON.

    Returns:
        Stats dict with structure:
        {
            "campaign_start": timestamp,
            "last_updated": timestamp,
            "total_jobs_all_regions": int,
            "total_cost_savings_usd": float,
            "regions": {
                "us-east4": {
                    # All-time aggregates
                    "total_jobs": int,
                    "successes": int,
                    "failures": int,
                    "preemptions": int,
                    "success_rate": float,
                    "total_cost_usd": float,
                    "total_savings_usd": float,
                    "battle_wins": int,
                    ...
                    # Rich job history
                    "recent_jobs": [
                        {
                            "timestamp": float,
                            "job_id": str,
                            "gpu_type": str,
                            "gpu_count": int,
                            "tier": str,
                            "success": bool,
                            "duration_minutes": float,
                            "spot_price_per_gpu": float,
                            "total_cost_usd": float,
                            "savings_vs_most_expensive_usd": float,
                            "preempted": bool,
                            "error": str or None
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
            "total_jobs_all_regions": 0,
            "total_cost_savings_usd": 0.0,
            "regions": {}
        }

    return stats


def save_campaign_stats(stats: Dict[str, Any]) -> bool:
    """
    Save Zeus campaign stats to JSON file using SafeJSON.

    Args:
        stats: Stats dictionary

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
        "Preempted:", "CUDA:", "RuntimeError:"
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


def record_job_result(
    region: str,
    job_id: str,
    job_name: str,
    gpu_type: str,
    gpu_count: int,
    tier: str,
    success: bool,
    status: str,
    duration_minutes: float,
    spot_price_per_gpu: float,
    total_cost_usd: float,
    savings_vs_most_expensive_usd: float,
    champion_reason: str = "",
    preempted: bool = False,
    error: Optional[str] = None,
    log_url: Optional[str] = None
) -> bool:
    """
    Record a completed training job result.

    NOTE: Currently writes DUMMY DATA! Will be populated with real Vertex AI job data later.

    Args:
        region: GCP region (e.g., "us-east4")
        job_id: Vertex AI custom job ID
        job_name: Job name
        gpu_type: GPU type (e.g., "NVIDIA_H100_80GB")
        gpu_count: Number of GPUs
        tier: Zeus tier (spark/bolt/storm/tempest/cataclysm)
        success: Job succeeded
        status: Vertex AI job status (JOB_STATE_SUCCEEDED, etc.)
        duration_minutes: Training duration
        spot_price_per_gpu: Spot price per GPU per hour
        total_cost_usd: Total cost (spot_price × duration × gpu_count)
        savings_vs_most_expensive_usd: Savings vs most expensive region
        champion_reason: Why Zeus selected this region
        preempted: GPU preempted during training
        error: Error message if failed
        log_url: Vertex AI console log URL

    Returns:
        True if recorded successfully
    """
    # DUMMY DATA MODE: Just write to JSON with SafeJSON for now
    # Real implementation will call Vertex AI API to get job metrics

    stats = load_campaign_stats()

    # Update timestamp
    stats["last_updated"] = time.time()

    # Initialize region if needed
    if region not in stats.get("regions", {}):
        stats.setdefault("regions", {})[region] = {
            "total_jobs": 0,
            "successes": 0,
            "failures": 0,
            "preemptions": 0,
            "success_rate": 0.0,
            "preemption_rate": 0.0,
            "total_duration_minutes": 0.0,
            "avg_duration_minutes": 0.0,
            "fastest_minutes": None,
            "slowest_minutes": None,
            "total_gpu_hours": 0.0,
            "total_cost_usd": 0.0,
            "total_savings_usd": 0.0,
            "avg_spot_price": 0.0,
            "last_error": None,
            "last_error_time": None,
            "divine_wrath_incidents": 0,
            "last_wrath_reason": None,
            "last_wrath_time": None,
            "current_streak": 0,
            "last_used": None,
            "battle_wins": 0,
            "recent_jobs": []
        }

    # Note: Real implementation will update aggregates here
    # For now, just save the structure (dummy data already in JSON file)

    return save_campaign_stats(stats)


def record_wrath_event(
    region: str,
    reason: str,
    reason_code: str,
    job_id: Optional[str] = None,
    details: Optional[str] = None
) -> bool:
    """
    Record a divine wrath incident (GPU preemption, training failure).

    Mirrors MECHA's record_fatigue_event() pattern (function name only - theme is divine wrath!).

    Args:
        region: GCP region where divine wrath occurred
        reason: Human-readable reason (e.g., "GPU preempted at step 15420")
        reason_code: Machine-readable code (use REASON_* constants)
        job_id: Associated Vertex AI job ID
        details: Additional details

    Returns:
        True if recorded successfully
    """
    # Validate reason code
    if reason_code not in VALID_REASON_CODES:
        reason_code = REASON_UNKNOWN

    stats = load_campaign_stats()

    # Update region wrath stats
    if region in stats.get("regions", {}):
        region_stats = stats["regions"][region]
        region_stats["divine_wrath_incidents"] = region_stats.get("divine_wrath_incidents", 0) + 1
        region_stats["last_wrath_reason"] = reason
        region_stats["last_wrath_time"] = time.time()

        # Update last error (for display)
        region_stats["last_error"] = _extract_concise_error(reason)
        region_stats["last_error_time"] = time.time()

    return save_campaign_stats(stats)


def get_recent_jobs(region: str, days: int = 30) -> List[Dict]:
    """
    Get recent jobs for a region within time window.

    Args:
        region: GCP region
        days: Time window in days

    Returns:
        List of job dicts within time window
    """
    stats = load_campaign_stats()

    if region not in stats.get("regions", {}):
        return []

    cutoff_time = time.time() - (days * 24 * 3600)
    recent_jobs = stats["regions"][region].get("recent_jobs", [])

    return [
        job for job in recent_jobs
        if job.get("timestamp", 0) >= cutoff_time
    ]


def get_stats_for_window(region: str, days: Optional[int] = None) -> Dict:
    """
    Calculate statistics for a region within time window.

    Mirrors MECHA's get_stats_for_window() pattern.

    Args:
        region: GCP region
        days: Time window in days (30, 60, 90). None = all time

    Returns:
        Dict with windowed metrics
    """
    # Handle None (all-time stats)
    if days is None:
        stats = load_campaign_stats()
        if region not in stats.get("regions", {}):
            return {
                "jobs": 0,
                "successes": 0,
                "failures": 0,
                "success_rate": 0.0,
                "total_cost_usd": 0.0,
                "total_savings_usd": 0.0,
                "avg_duration_minutes": 0.0
            }
        region_stats = stats["regions"][region]
        return {
            "jobs": region_stats.get("total_jobs", 0),
            "successes": region_stats.get("successes", 0),
            "failures": region_stats.get("failures", 0),
            "success_rate": region_stats.get("success_rate", 0.0),
            "total_cost_usd": region_stats.get("total_cost_usd", 0.0),
            "total_savings_usd": region_stats.get("total_savings_usd", 0.0),
            "avg_duration_minutes": region_stats.get("avg_duration_minutes", 0.0)
        }
    jobs = get_recent_jobs(region, days)

    if not jobs:
        return {
            "jobs": 0,
            "successes": 0,
            "failures": 0,
            "success_rate": 0.0,
            "total_cost_usd": 0.0,
            "total_savings_usd": 0.0,
            "avg_duration_minutes": 0.0
        }

    successes = sum(1 for j in jobs if j.get("success", False))
    failures = len(jobs) - successes
    total_cost = sum(j.get("total_cost_usd", 0.0) for j in jobs)
    total_savings = sum(j.get("savings_vs_most_expensive_usd", 0.0) for j in jobs)
    total_duration = sum(j.get("duration_minutes", 0.0) for j in jobs)

    return {
        "jobs": len(jobs),
        "successes": successes,
        "failures": failures,
        "success_rate": (successes / len(jobs) * 100) if jobs else 0.0,
        "total_cost_usd": total_cost,
        "total_savings_usd": total_savings,
        "avg_duration_minutes": (total_duration / len(jobs)) if jobs else 0.0
    }


def get_top_regions(metric: str = "success_rate", limit: int = 5) -> List[Tuple[str, float]]:
    """
    Get top N regions sorted by specified metric.

    Mirrors MECHA's get_top_regions() pattern.

    Args:
        metric: Metric to sort by ("success_rate", "total_jobs", "total_savings_usd", etc.)
        limit: Number of top regions to return

    Returns:
        List of (region, metric_value) tuples, sorted by metric descending
    """
    stats = load_campaign_stats()
    regions = stats.get("regions", {})

    # Build list of (region, metric_value) tuples
    region_metrics = []
    for region, region_stats in regions.items():
        value = region_stats.get(metric, 0.0)
        # Handle both numeric and None values
        if value is None:
            value = 0.0
        region_metrics.append((region, value))

    # Sort by metric value descending
    region_metrics.sort(key=lambda x: x[1], reverse=True)

    # Return top N
    return region_metrics[:limit]
