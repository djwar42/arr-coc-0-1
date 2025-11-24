"""
⚡ Zeus Olympus Registry System - Thunder Fleet State + Divine Wrath Tracking

Tracks thunder-ready GPU regions across multiple tiers (Spark, Bolt, Storm, Tempest, Cataclysm).
Automatically wipes tier fleet when GPU TYPE changes within that tier.

Thunder Discovery Strategy:
- Full quota snapshot: ONE API call discovers ALL regions with quota (15× faster!)
- Tier-specific wipe: GPU type change within tier = delete tier and restart
- Divine wrath system: Preempted regions need Zeus's favor to return
- Per-tier fleets: Each tier (T4, L4, A100, H100, H200) tracked independently

Divine Wrath System:
- 1st preemption (24h) → DISFAVORED 4 hours ⚡
- 2nd preemption → DISFAVORED 4 hours ⚡
- 3rd preemption in 24h → ZEUS'S WRATH 24 hours (whole day) ⚡⚡⚡
- Auto-recovery: Regions can battle again after wrath period

Registry tracks (per tier):
- Region name
- GPU type (e.g., NVIDIA_H100_80GB)
- Quota status (THUNDER_READY, QUEST_LOCKED)
- Last training timestamp
- Divine wrath state (wrathful_until, preemptions_today)
- Wrath reason (human-readable + machine-readable code)
"""

# <claudes_code_comments>
# ** Function List **
# load_olympus_registry() - Load Zeus registry using SafeJSON (auto-handles corruption)
# save_olympus_registry(registry) - Save registry using SafeJSON (atomic + backups)
# check_thunder_tier_changed(registry, tier, gpu_type) - Detect GPU type change within tier
# wipe_tier_fleet(registry, tier_name, new_gpu_type) - Reset specific tier on GPU type change
# _get_tier_emoji(tier_name) - Get emoji for tier (⚡☇⛈️⚡⚡⚡⚡⚡⚡⚡)
# get_thunder_ready_regions(registry, tier) - List regions with GPU quota
# get_quest_locked_regions(registry, tier) - List regions needing quota requests
# update_region_status(...) - Update region status (THUNDER_READY/QUEST_LOCKED)
# get_tier_fleet_status(registry, tier, total_regions) - Overall tier health
# is_region_wrathful(region_info) - Check if region under Zeus's wrath
# _log_divine_incident(...) - Record preemption incident to divine_incidents.json
# record_divine_wrath(registry, tier, region, reason, reason_code, error_message, job_id) - Apply wrath penalties, returns (count, hours, msg)
# get_available_thunder_regions(registry, tier, all_regions, outlawed_regions) - Get wrath-free + non-outlawed regions
# get_wrath_status_display(registry, tier) - Human-readable wrath summary
#
# ** Technical Review **
# Multi-tier thunder fleet registry with divine wrath tracking. Each tier (spark/bolt/storm/
# tempest/cataclysm) has independent fleet state. User selects tier via GPU_TYPE config,
# Zeus tracks quota acquisition progress for THAT tier's regions.
#
# Key insight: Unlike MECHA (which auto-selects machine), Zeus tracks state for ALL 5 tiers
# simultaneously. User's GPU_TYPE picks which tier to use, but registry maintains all tiers
#
# ═══════════════════════════════════════════════════════════════════════════════════════
# WIRE-UP: Divine Wrath System (Preemption Detection)
# ═══════════════════════════════════════════════════════════════════════════════════════
# Divine wrath system is READY but not yet wired to job monitoring. To enable:
#
# 1. Add preemption detection in job monitoring flow (core.py or job monitor):
#    - After Vertex AI job completes: Check job.state == "JOB_STATE_FAILED"
#    - Parse job.error for preemption indicators:
#      * "spot instance"
#      * "preempted"
#      * "reclaimed"
#    - If preemption detected:
#      from zeus.zeus_olympus import record_divine_wrath, REASON_SPOT_PREEMPTION
#      registry = load_olympus_registry()
#      count, hours, msg = record_divine_wrath(
#          registry, tier_name, region_name,
#          reason_code=REASON_SPOT_PREEMPTION,
#          error_message=job.error,
#          job_id=job.name
#      )
#      save_olympus_registry(registry)
#      print(f"⚡ {region_name} DISFAVORED for {hours}h (preemption #{count})")
#
# 2. Wrathful regions are already filtered by get_available_thunder_regions()
#    - Integration with zeus_battle.py is complete
#    - Regions with active wrath are automatically excluded from pricing battles
#
# 3. Wrath tracking is already integrated with campaign_stats.py
#    - record_wrath_event() logs all incidents
#    - Divine incidents saved to zeus/data/divine_incidents.json
# ═══════════════════════════════════════════════════════════════════════════════════════
# for quick switching (e.g., user changes H100 → T4, tier data already tracked).
#
# Registry structure: tiers[tier_name] = {gpu_type, regions{region_name: {status, price, ...}}}
# Region status: THUNDER_READY (has quota), QUEST_LOCKED (no quota, needs manual request).
#
# Acquisition flow (for user's chosen tier):
# 1. check_thunder_tier_changed() → Did GPU type change within tier? (e.g., H100 → H100_MEGA)
# 2. If changed: wipe_tier_fleet() → Delete tier, restart acquisition for new GPU type
# 3. get_quest_locked_regions() → Find regions without quota
# 4. Passive collection attempts quota check → If exists: update_region_status(THUNDER_READY)
# 5. On preemption: record_divine_wrath() → Apply wrath penalties (4h/24h)
#
# Divine wrath escalation (3-strike system for preempted regions):
# - 1st preemption: DISFAVORED 4 hours (wrathful_until = now + 4h)
# - 2nd preemption (within 24h): DISFAVORED 4 hours again
# - 3rd preemption (within 24h): ZEUS'S WRATH 24 hours (wrathful_until = now + 24h)
# Reset: preemptions_today cleared after 24h window expires. Wrathful regions excluded from battle.
#
# Tier wipe scope: ONLY wipes changed tier, other tiers unaffected (user might switch GPUs).
#
# File I/O: SafeJSON for zeus_olympus.json + divine_incidents.json. Returns {} if missing/corrupt.
# Atomic writes + 20 versioned backups prevent data loss during concurrent launches.
# </claudes_code_comments>

import time
from pathlib import Path
from typing import Dict, Optional, Tuple, List

# Import SafeJSON for production-grade JSON I/O
from ...shared.safe_json import SafeJSON

# Import reason codes and wrath tracking from campaign_stats (single source of truth)
from .campaign_stats import (
    REASON_SPOT_PREEMPTION,
    REASON_QUOTA_EXCEEDED,
    REASON_ZONE_EXHAUSTION,
    REASON_UNKNOWN,
    record_wrath_event
)

# Registry file locations - stored in zeus/data/ folder
ZEUS_OLYMPUS_PATH = Path(__file__).parent / "data" / "zeus_olympus.json"
DIVINE_INCIDENTS_PATH = Path(__file__).parent / "data" / "divine_incidents.json"


def load_olympus_registry() -> Dict:
    """
    Load Zeus Olympus registry from disk using SafeJSON.

    Returns:
        {
            "last_updated": 1234567890,
            "thunder_fleets": {
                "tempest": {
                    "tier_emoji": "⚡⚡⚡⚡",
                    "gpu_type": "NVIDIA_H100_80GB",
                    "total_regions": 8,
                    "thunder_ready_count": 6,
                    "quest_locked_count": 2,
                    "regions": {
                        "us-central1": {
                            "gpu_type": "NVIDIA_H100_80GB",
                            "quota_gpus": 8,
                            "spot_price_per_hour": 2.10,
                            "operational_status": "THUNDER_READY" | "QUEST_LOCKED",
                            "created_at": 1234567890,
                            "last_training": 1234567890,

                            # Divine wrath tracking (only after preemption)
                            "divine_wrath": {
                                "wrathful_until": 1234567890,  # Unix timestamp
                                "preemptions_today": [1234567890, ...],
                                "preemption_count_today": 2,
                                "wrath_message": "DISFAVORED for 4h",
                                "last_preemption_reason": "Spot preemption - 30 sec warning",
                                "last_preemption_reason_code": "spot-preemption"
                            }
                        },
                        ...
                    }
                },
                "spark": { ... },  # T4 fleet
                "bolt": { ... },   # L4 fleet
                "storm": { ... },  # A100 fleet
                "cataclysm": { ... }  # H200 fleet
            }
        }
    """
    registry = SafeJSON.read(ZEUS_OLYMPUS_PATH)
    if not registry:
        registry = {
            "last_updated": time.time(),
            "thunder_fleets": {}
        }
    return registry


def save_olympus_registry(registry: Dict) -> None:
    """
    Save Zeus Olympus registry to disk using SafeJSON (atomic write + backups).
    """
    registry["last_updated"] = time.time()
    SafeJSON.write(ZEUS_OLYMPUS_PATH, registry)


def check_thunder_tier_changed(registry: Dict, tier_name: str, current_gpu_type: str) -> bool:
    """
    Check if GPU type changed within this tier.

    Returns:
        True if tier needs wipe (GPU type changed)
        False if same GPU type or tier doesn't exist yet
    """
    if "thunder_fleets" not in registry:
        return False

    if tier_name not in registry["thunder_fleets"]:
        return False

    tier = registry["thunder_fleets"][tier_name]
    stored_gpu_type = tier.get("gpu_type")

    if stored_gpu_type and stored_gpu_type != current_gpu_type:
        return True

    return False


def wipe_tier_fleet(registry: Dict, tier_name: str, new_gpu_type: str) -> Dict:
    """
    Nuclear reset for specific tier when GPU type changes.
    Other tiers remain untouched.

    Returns updated registry.
    """
    if "thunder_fleets" not in registry:
        registry["thunder_fleets"] = {}

    # Wipe this tier only
    registry["thunder_fleets"][tier_name] = {
        "gpu_type": new_gpu_type,
        "tier_emoji": _get_tier_emoji(tier_name),
        "total_regions": 0,
        "thunder_ready_count": 0,
        "quest_locked_count": 0,
        "regions": {}
    }

    return registry


def _get_tier_emoji(tier_name: str) -> str:
    """Get tier emoji based on tier name."""
    tier_emojis = {
        "spark": "⚡",
        "bolt": "⚡⚡",
        "storm": "⚡⚡⚡",
        "tempest": "⚡⚡⚡⚡",
        "cataclysm": "⚡⚡⚡⚡⚡"
    }
    return tier_emojis.get(tier_name, "⚡")


def get_thunder_ready_regions(registry: Dict, tier_name: str) -> List[str]:
    """
    Get list of regions with GPU quota (thunder-ready).

    Returns list of region names.
    """
    if "thunder_fleets" not in registry:
        return []

    if tier_name not in registry["thunder_fleets"]:
        return []

    tier = registry["thunder_fleets"][tier_name]
    thunder_ready = []

    for region_name, region_info in tier.get("regions", {}).items():
        if region_info.get("operational_status") == "THUNDER_READY":
            thunder_ready.append(region_name)

    return thunder_ready


def get_quest_locked_regions(registry: Dict, tier_name: str) -> List[str]:
    """
    Get list of regions needing quota requests (quest-locked).

    Returns list of region names.
    """
    if "thunder_fleets" not in registry:
        return []

    if tier_name not in registry["thunder_fleets"]:
        return []

    tier = registry["thunder_fleets"][tier_name]
    quest_locked = []

    for region_name, region_info in tier.get("regions", {}).items():
        if region_info.get("operational_status") == "QUEST_LOCKED":
            quest_locked.append(region_name)

    return quest_locked


def is_region_wrathful(region_info: Dict) -> Tuple[bool, Optional[str]]:
    """
    Check if region is under divine wrath.

    Returns:
        (True, "Wrathful until 18:00 (3.5h remaining)") if wrathful
        (False, None) if wrath cleared
    """
    wrath = region_info.get("divine_wrath", {})
    wrathful_until = wrath.get("wrathful_until")

    if not wrathful_until:
        return (False, None)

    now = time.time()
    if now < wrathful_until:
        # Still wrathful
        remaining_seconds = wrathful_until - now
        remaining_hours = remaining_seconds / 3600

        wrath_msg = wrath.get("wrath_message", "Under divine wrath")
        time_msg = f"{remaining_hours:.1f}h remaining"

        return (True, f"{wrath_msg} ({time_msg})")
    else:
        # Wrath expired
        return (False, None)


def record_divine_wrath(
    registry: Dict,
    tier_name: str,
    region_name: str,
    reason: str = "Spot preemption",
    reason_code: str = None,
    error_message: str = "",
    job_id: Optional[str] = None
):
    """
    Record divine wrath after GPU preemption (matches MECHA's record_mecha_timeout pattern).

    Applies wrath rules:
    - 1st preemption in 24h → 4 hours DISFAVORED
    - 2nd preemption in 24h → 4 hours DISFAVORED
    - 3rd preemption in 24h → 24 hours ZEUS'S WRATH! (EXHAUSTED!)

    Args:
        registry: Zeus Olympus registry dict
        tier_name: Thunder tier name (spark, bolt, storm, tempest, cataclysm)
        region_name: Region name
        reason: Human-readable reason for preemption
        reason_code: Machine-readable reason code (campaign_stats.REASON_*)
        error_message: Full error message (optional, can be long)
        job_id: Vertex AI job ID (optional)

    Returns:
        (preemption_count, wrath_hours, wrath_message) tuple

    Wire-up Location:
        Call this from job monitoring when Vertex AI job is preempted:
        - After job completes: Check job.state == "JOB_STATE_FAILED"
        - Parse error for preemption: "spot instance reclaimed" / "preempted"
        - Call: record_divine_wrath(registry, tier, region, reason_code=REASON_SPOT_PREEMPTION)
        - Save registry: save_olympus_registry(registry)
        - Region will be excluded from battles until wrath expires
    """
    if "thunder_fleets" not in registry or tier_name not in registry["thunder_fleets"]:
        return (0, 0, "ERROR: Tier not found")

    tier = registry["thunder_fleets"][tier_name]
    if region_name not in tier["regions"]:
        return (0, 0, "ERROR: Region not found")

    region = tier["regions"][region_name]
    now = time.time()

    # Initialize wrath tracking if needed
    if "divine_wrath" not in region:
        region["divine_wrath"] = {
            "wrathful_until": None,
            "preemptions_today": [],
            "preemption_count_today": 0,
            "wrath_message": None,
            "last_preemption_reason": None,
            "last_preemption_reason_code": None
        }

    wrath = region["divine_wrath"]

    # Get or initialize preemption tracking
    preemptions_today = wrath.get("preemptions_today", [])

    # Clean out preemptions older than 24 hours
    cutoff = now - (24 * 3600)
    preemptions_today = [ts for ts in preemptions_today if ts > cutoff]

    # Add this preemption
    preemptions_today.append(now)

    # Determine wrath duration based on preemption count
    preemption_count = len(preemptions_today)

    if preemption_count >= 3:
        # 3rd preemption in 24h → ZEUS'S WRATH! 24 hours rest (whole day)
        wrath_hours = 24
        wrath_message = "⚡⚡⚡ ZEUS'S WRATH!"
    else:
        # 1st or 2nd preemption → DISFAVORED 4 hours
        wrath_hours = 4
        wrath_message = "DISFAVORED"

    wrathful_until = now + (wrath_hours * 3600)

    # Update region divine wrath info (matches MECHA's mecha_info pattern)
    wrath.update({
        "wrathful_until": wrathful_until,
        "preemptions_today": preemptions_today,
        "preemption_count_today": preemption_count,
        "wrath_message": f"{wrath_message} for {wrath_hours}h",
        "last_preemption_reason": reason,  # Human-readable reason
        "last_preemption_reason_code": reason_code  # Machine-readable classification
    })

    # LOG TO DIVINE INCIDENTS HISTORY! (matches MECHA's _log_godzilla_incident)
    _log_divine_incident(
        tier_name=tier_name,
        region_name=region_name,
        reason_code=reason_code if reason_code else REASON_UNKNOWN,
        error_msg=error_message,
        job_id=job_id,
        preemption_count=preemption_count
    )

    # RECORD TO CAMPAIGN STATS! (matches MECHA's record_fatigue_event)
    # Default to REASON_UNKNOWN if no code provided (backward compatibility)
    final_reason_code = reason_code if reason_code else REASON_UNKNOWN
    record_wrath_event(region_name, reason, final_reason_code, error_message)

    return (preemption_count, wrath_hours, wrath_message)


def _log_divine_incident(
    tier_name: str,
    region_name: str,
    reason_code: str,
    error_msg: str,
    job_id: Optional[str],
    preemption_count: int
) -> None:
    """
    Log divine wrath incident to permanent divine_incidents.json file.
    """
    incidents = SafeJSON.read(DIVINE_INCIDENTS_PATH)
    if not incidents:
        incidents = {"incidents": []}

    incident = {
        "timestamp": time.time(),
        "tier": tier_name,
        "region": region_name,
        "reason_code": reason_code,
        "error_message": error_msg,
        "job_id": job_id,
        "preemption_count_today": preemption_count,
        "wrath_level": "ZEUS'S WRATH" if preemption_count >= 3 else "DISFAVORED"
    }

    incidents["incidents"].append(incident)
    SafeJSON.write(DIVINE_INCIDENTS_PATH, incidents)


def get_available_thunder_regions(registry: Dict, tier_name: str, all_regions: List[str], outlawed_regions: Optional[List[str]] = None) -> List[str]:
    """
    Get thunder-ready regions that are NOT under divine wrath or outlawed.

    Filters out:
    - Wrathful regions (preemptions)
    - Outlawed regions (ZEUS_OUTLAWED_REGIONS from config)

    Returns list of region names ready for battle.
    """
    outlawed_regions = outlawed_regions or []
    thunder_ready = get_thunder_ready_regions(registry, tier_name)

    if not thunder_ready:
        return []

    # Filter out wrathful AND outlawed regions
    available = []
    tier = registry["thunder_fleets"][tier_name]

    for region_name in thunder_ready:
        # Skip outlawed regions
        if region_name in outlawed_regions:
            continue

        region_info = tier["regions"].get(region_name, {})
        is_wrathful, _ = is_region_wrathful(region_info)

        if not is_wrathful:
            available.append(region_name)

    return available


def update_region_status(
    registry: Dict,
    tier_name: str,
    region_name: str,
    status: str,
    gpu_type: str,
    quota_gpus: int = 0,
    spot_price: float = 0.0
) -> Dict:
    """
    Update region status in tier fleet.

    status: "THUNDER_READY" or "QUEST_LOCKED"
    """
    if "thunder_fleets" not in registry:
        registry["thunder_fleets"] = {}

    if tier_name not in registry["thunder_fleets"]:
        registry["thunder_fleets"][tier_name] = {
            "gpu_type": gpu_type,
            "tier_emoji": _get_tier_emoji(tier_name),
            "total_regions": 0,
            "thunder_ready_count": 0,
            "quest_locked_count": 0,
            "regions": {}
        }

    tier = registry["thunder_fleets"][tier_name]

    if region_name not in tier["regions"]:
        tier["regions"][region_name] = {
            "gpu_type": gpu_type,
            "created_at": time.time()
        }

    region = tier["regions"][region_name]
    region["operational_status"] = status
    region["quota_gpus"] = quota_gpus
    region["spot_price_per_hour"] = spot_price

    # Update counts
    tier["total_regions"] = len(tier["regions"])
    tier["thunder_ready_count"] = len(get_thunder_ready_regions(registry, tier_name))
    tier["quest_locked_count"] = len(get_quest_locked_regions(registry, tier_name))

    return registry


def get_tier_fleet_status(registry: Dict, tier_name: str, total_possible: int) -> Dict:
    """
    Get overall tier fleet health summary.

    Returns:
        {
            "thunder_ready": 6,
            "quest_locked": 2,
            "missing": 0,
            "total_possible": 8,
            "completion": 0.75
        }
    """
    thunder_ready = len(get_thunder_ready_regions(registry, tier_name))
    quest_locked = len(get_quest_locked_regions(registry, tier_name))
    tracked = thunder_ready + quest_locked
    missing = max(0, total_possible - tracked)

    return {
        "thunder_ready": thunder_ready,
        "quest_locked": quest_locked,
        "missing": missing,
        "total_possible": total_possible,
        "completion": thunder_ready / total_possible if total_possible > 0 else 0.0
    }


def get_wrath_status_display(registry: Dict, tier_name: str) -> str:
    """
    Get human-readable wrath status for Zeus Olympus display (matches MECHA's get_fatigue_status_display).
    
    Returns formatted string showing wrathful regions with recovery times.
    
    Args:
        registry: Zeus Olympus registry dict
        tier_name: Thunder tier name
        
    Returns:
        Formatted string with wrathful regions (empty string if none)
    """
    if "thunder_fleets" not in registry or tier_name not in registry["thunder_fleets"]:
        return ""
        
    tier = registry["thunder_fleets"][tier_name]
    regions = tier.get("regions", {})
    wrathful_regions = []
    
    for region_name, region_info in regions.items():
        is_wrathful, message = is_region_wrathful(region_info)
        if is_wrathful:
            preemption_count = region_info.get("divine_wrath", {}).get("preemption_count_today", 0)
            if preemption_count >= 3:
                emoji = "⚡⚡⚡"  # ZEUS'S WRATH (exhausted - 24h)
            else:
                emoji = "☁️"  # DISFAVORED (4h rest)
            
            wrathful_regions.append(f"     {emoji} {region_name}: {message}")
    
    if not wrathful_regions:
        return ""
        
    lines = ["\n   ⚡ REGIONS UNDER DIVINE WRATH:"] + wrathful_regions
    return "\n".join(lines)
