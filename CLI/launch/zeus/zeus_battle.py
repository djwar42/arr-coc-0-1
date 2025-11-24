"""
⚡ Zeus Battle Orchestrator - Quota Discovery + Thunder Champion Selection

Core Logic:
1. Load olympus registry at launch
2. Check GPU TYPE changed within tier → WIPE TIER (not all tiers)
3. Refresh quota snapshot → discover ALL regions with quota (1 API call!)
4. Select thunder champion from regions with quota
5. Update registry after discovery
6. Show epic thunder battle phrases

Strategy:
- Full Discovery: ONE API call checks ALL regions for quota every launch
- Instant Detection: New quota approvals discovered immediately
- Tier-specific Wipe: GPU type change within tier = DELETE tier, restart
- Progressive: User requests quota manually → Zeus discovers automatically

Quota Discovery (not creation):
- Zeus DISCOVERS existing quota (user requests via GCP console)
- API returns ALL regions in single call (15× faster than old method!)
- zeus_acquire.py provides optional Thunder Fleet Blast display
- HERMES passage shown when quota needed
"""

# <claudes_code_comments>
# ** Function List **
# check_quota_exists(region, tier, gpu_count, project_id, use_spot) - Check single region GPU quota
# get_spot_pricing(region, tier, gpu_count, pricing_data) - Get live GCP spot pricing for region
# refresh_quota_snapshot(registry, tier, gpu_count, project_id, print_fn, pricing_data) - Discover ALL regions with quota, return (registry, newly_acquired, newly_lost)
# show_hermes_passage(region, tier, gpu_count, print_fn) - Show divine quota request guidance
# select_thunder_champion(thunder_ready, tier, gpu_count, print_fn, pricing_data) - Battle regions, return champion + prices (5 values like MECHA!)
#
# ** Technical Review **
# Zeus battles REGIONS for user's chosen GPU type (NOT auto-selecting GPU like MECHA!).
# User explicitly specifies GPU_TYPE in config (e.g., "NVIDIA_H100_80GB") → maps to tier.
# Zeus ONLY battles to find cheapest REGION for that specific GPU tier.
#
# Core flow: User sets GPU_TYPE → core.py validates → maps to tier → Zeus battles regions:
# 1. Load olympus registry (multi-tier state)
# 2. Check tier GPU type changed → wipe tier if changed (restart discovery)
# 3. Refresh quota snapshot → ONE API call discovers ALL regions with quota (15× faster!)
# 4. Separate regions: thunder-ready (have quota) vs quest-locked (no quota)
# 5. Run select_thunder_champion() → battle thunder-ready regions → select cheapest
# 6. Update olympus registry → save state
# 7. Return (champion, champion_price, compare_region, compare_price, savings) to integration
# 8. Integration shows WINS! banner + battle summary (matching MECHA format!)
#
# Quota Discovery (not creation):
# - User requests quota via GCP console (1-3 days approval)
# - Zeus discovers quota via get_all_gpu_quotas_for_type() → returns ALL regions at once
# - Snapshot refresh runs EVERY launch → instant discovery of new approvals
# - No region lists needed → API tells us which regions have each GPU type!
#
# Pricing: Uses shared/pricing get_spot_price() for live GCP spot pricing data.
# CRITICAL: NO FALLBACK PRICING! If live data unavailable, battle HALTS (raises RuntimeError).
# We NEVER use fake prices for real money decisions - fail hard with clear error instead.
#
# Data structures: THUNDER_TIERS defines 5 tiers with GPU types, memory, emoji (NO region lists!).
# Regions discovered via API - get_all_gpu_quotas_for_type() returns {region: quota} for ALL regions.
# Fully integrated with shared modules: get_all_gpu_quotas_for_type() from shared/quota/gpu_quota.py,
# get_spot_price() from shared/pricing/__init__.py (mirrors MECHA pattern - same shared code).
# </claudes_code_comments>

import json
import random
import subprocess
from typing import Dict, List, Optional, Tuple

from ...shared.pricing import get_spot_price

# Import shared modules for quota and pricing
from ...shared.quota.gpu_quota import get_all_gpu_quotas_for_type, get_vertex_gpu_quotas
from .zeus_battle_epic import (
    BATTLE_ROUND_PHRASES,
    SIZING_UP_PHRASES,
    VICTORY_DECLARATIONS,
    VICTORY_EMERGENCE_PHRASES,
    DivineThunderPrinter,
    categorize_price,
    get_price_tier,
    get_thunder_phrase,
)
from .zeus_display import REGION_FLAGS, get_region_display_name
from .zeus_olympus import (
    check_thunder_tier_changed,
    get_quest_locked_regions,
    get_thunder_ready_regions,
    get_tier_fleet_status,
    load_olympus_registry,
    save_olympus_registry,
    update_region_status,
    wipe_tier_fleet,
)

# ═══════════════════════════════════════════════════════════════════════════════
# TIER CONSTANTS (Zeus GPU capabilities)
# ═══════════════════════════════════════════════════════════════════════════════
# Regions are discovered via API - get_all_gpu_quotas_for_type() returns ALL regions!
# No need to maintain region lists manually.

# ═══════════════════════════════════════════════════════════════════════════════
# ⚡ ZEUS THUNDER MANIFEST - SOURCE OF TRUTH FOR GPU TIERS
# ═══════════════════════════════════════════════════════════════════════════════
# This is the canonical definition for ZEUS thunder tiers.
# Other modules should import from here, NOT from gcp-manifest.json
# gcp-manifest.json just points here as the source of truth.

ZEUS_THUNDER_MANIFEST = {
    "system": "zeus",
    "description": "ZEUS Thunder Battle System - GPU tier pricing battles for Vertex AI",
    "registry_file": "CLI/launch/zeus/data/zeus_olympus.json",
    "total_tiers": 5,
    "tiers": [
        {
            "name": "spark",
            "emoji": "⚡",
            "gpu_type": "NVIDIA_TESLA_T4",
            "display_name": "T4",
            "memory_gb": 16,
            "machine_type": "n1-standard-4",  # T4 attaches to N1 machines
            "cost_tier": "low",
        },
        {
            "name": "bolt",
            "emoji": "⚡⚡",
            "gpu_type": "NVIDIA_L4",
            "display_name": "L4",
            "memory_gb": 24,
            "machine_type": "g2-standard-4",  # L4 uses G2 machines
            "cost_tier": "medium",
        },
        {
            "name": "storm",
            "emoji": "⚡⚡⚡",
            "gpu_type": "NVIDIA_TESLA_A100",
            "display_name": "A100",
            "memory_gb": 40,
            "machine_type": "a2-highgpu-1g",  # A100 40GB pre-attached
            "cost_tier": "high",
        },
        {
            "name": "tempest",
            "emoji": "⚡⚡⚡⚡",
            "gpu_type": "NVIDIA_H100_80GB",
            "display_name": "H100 80GB",
            "memory_gb": 80,
            "machine_type": "a3-highgpu-8g",  # H100 80GB pre-attached (8 GPUs)
            "cost_tier": "premium",
        },
        {
            "name": "cataclysm",
            "emoji": "⚡⚡⚡⚡⚡",
            "gpu_type": "NVIDIA_H200",
            "display_name": "H200",
            "memory_gb": 141,
            "machine_type": "a3-highgpu-8g",  # H200 uses same A3 machines
            "cost_tier": "premium",
        },
    ],
}

# Runtime lookup dict (derived from manifest)
THUNDER_TIERS = {
    "spark": {"emoji": "⚡", "gpu_types": ["NVIDIA_TESLA_T4"], "memory_gb": 16},
    "bolt": {"emoji": "⚡⚡", "gpu_types": ["NVIDIA_L4"], "memory_gb": 24},
    "storm": {"emoji": "⚡⚡⚡", "gpu_types": ["NVIDIA_TESLA_A100"], "memory_gb": 40},
    "tempest": {
        "emoji": "⚡⚡⚡⚡",
        "gpu_types": ["NVIDIA_H100_80GB"],
        "memory_gb": 80,
    },
    "cataclysm": {
        "emoji": "⚡⚡⚡⚡⚡",
        "gpu_types": ["NVIDIA_H200"],
        "memory_gb": 141,
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# QUOTA CHECKING
# ═══════════════════════════════════════════════════════════════════════════════


def check_quota_exists(
    region: str, tier: str, gpu_count: int, project_id: str, use_spot: bool = True
) -> Tuple[bool, int]:
    """
    Check if GPU quota exists in region for this tier using shared quota module.

    Args:
        region: GCP region (e.g., "us-central1")
        tier: Thunder tier (e.g., "tempest")
        gpu_count: Number of GPUs needed
        project_id: GCP project ID
        use_spot: Check spot quota (default True)

    Returns:
        (exists, quota_limit)
    """
    if tier not in THUNDER_TIERS:
        return (False, 0)

    gpu_type = THUNDER_TIERS[tier]["gpu_types"][0]

    try:
        # Use shared quota module (Vertex AI quotas, not Compute Engine!)
        quota_limit = get_vertex_gpu_quotas(project_id, region, gpu_type, use_spot)

        if quota_limit >= gpu_count:
            return (True, quota_limit)
        else:
            return (False, quota_limit)

    except Exception:
        return (False, 0)


# ═══════════════════════════════════════════════════════════════════════════════
# PRICING
# ═══════════════════════════════════════════════════════════════════════════════


def get_spot_pricing(
    region: str, tier: str, gpu_count: int, pricing_data: Optional[Dict] = None
) -> float:
    """
    Get LIVE GCP spot pricing for GPUs in specified region.

    Uses Cloud Billing Catalog API data from Artifact Registry (mirrors MECHA pattern).

    CRITICAL: NO FALLBACK PRICING! If live data unavailable, returns None (caller must handle).

    Args:
        region: GCP region (e.g., "us-central1")
        tier: Thunder tier (e.g., "tempest")
        gpu_count: Number of GPUs
        pricing_data: GPU pricing dictionary from Artifact Registry (REQUIRED!)

    Returns:
        Total hourly cost (e.g., $16.40/hr for 8×H100) or None if pricing unavailable
    """
    if tier not in THUNDER_TIERS:
        return None  # Invalid tier

    gpu_type = THUNDER_TIERS[tier]["gpu_types"][0]

    # ═══════════════════════════════════════════════════════════════
    # REQUIRE LIVE PRICING DATA (no fallback!)
    # ═══════════════════════════════════════════════════════════════
    if not pricing_data or "gpus_spot" not in pricing_data:
        return None  # No pricing data available

    region_gpu_skus = pricing_data["gpus_spot"].get(region, [])

    if not region_gpu_skus:
        return None  # No SKUs for this region

    # Filter SKUs for this specific GPU type
    # GPU type in THUNDER_TIERS: "NVIDIA_H100_80GB"
    # Description in SKU: "Nvidia Tesla H100 80GB GPU running in region"
    # Match strategy: Check if GPU family appears in description

    # Extract GPU family for matching (H100, T4, L4, A100, H200)
    gpu_family = gpu_type.replace("NVIDIA_", "").replace("TESLA_", "").split("_")[0]
    # "NVIDIA_H100_80GB" → "H100"
    # "NVIDIA_TESLA_T4" → "T4"

    matching_skus = [
        sku
        for sku in region_gpu_skus
        if gpu_family.lower() in sku.get("description", "").lower()
    ]

    if not matching_skus:
        return None  # No matching SKUs for this GPU type

    # Use shared pricing module to get cheapest spot price
    price_per_gpu = get_spot_price(matching_skus)

    if price_per_gpu is None:
        return None  # Shared module couldn't extract price

    total_price = price_per_gpu * gpu_count
    return round(total_price, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# QUOTA SNAPSHOT REFRESH (Acquisition = Discovery!)
# ═══════════════════════════════════════════════════════════════════════════════


def refresh_quota_snapshot(
    registry: Dict,
    tier: str,
    gpu_count: int,
    project_id: str,
    print_fn,
    pricing_data: Optional[Dict] = None,
) -> Tuple[Dict, List[str], List[str]]:
    """
    Refresh quota snapshot for ALL regions in tier (full discovery sweep).

    This is NOT creating quota - it's DISCOVERING what quota we have RIGHT NOW!
    User manually requests quota via GCP console (1-3 days approval).
    Zeus discovers new approvals by checking all regions each battle.

    Acquisition = Quota discovery (we found you have it now!)

    Returns:
        (updated_registry, newly_acquired_regions, newly_lost_regions)
    """

    # Run SILENTLY - output shown later after MOUNT OLYMPUS STATUS

    newly_acquired = []
    newly_lost = []
    gpu_type = THUNDER_TIERS[tier]["gpu_types"][0]

    # ONE API CALL - Get ALL regions at once! (15× faster!)
    quotas = get_all_gpu_quotas_for_type(project_id, gpu_type, use_spot=True)

    # Process each region from quota data
    for region, quota_limit in quotas.items():
        # Get previous status from registry
        tier_data = registry.get("thunder_fleets", {}).get(tier, {})
        regions_data = tier_data.get("regions", {})
        region_info = regions_data.get(region, {})
        previous_status = region_info.get("operational_status", "UNKNOWN")

        has_quota = quota_limit >= gpu_count

        if has_quota and previous_status != "THUNDER_READY":
            # NEW QUOTA DISCOVERED! (user got approval!)
            spot_price = get_spot_pricing(region, tier, gpu_count, pricing_data)

            registry = update_region_status(
                registry,
                tier,
                region,
                status="THUNDER_READY",
                gpu_type=gpu_type,
                quota_gpus=quota_limit,
                spot_price=spot_price,
            )

            newly_acquired.append(region)
            # Output shown later after MOUNT OLYMPUS STATUS

        elif not has_quota and previous_status == "THUNDER_READY":
            # LOST QUOTA (rare but possible - quota revoked/expired)
            registry = update_region_status(
                registry,
                tier,
                region,
                status="QUEST_LOCKED",
                gpu_type=gpu_type,
                quota_gpus=quota_limit,
                spot_price=0.0,
            )

            newly_lost.append(region)
            # Output shown later after MOUNT OLYMPUS STATUS

        elif not has_quota and previous_status == "UNKNOWN":
            # FIRST RUN: Region has no quota - mark as QUEST_LOCKED
            registry = update_region_status(
                registry,
                tier,
                region,
                status="QUEST_LOCKED",
                gpu_type=gpu_type,
                quota_gpus=quota_limit,
                spot_price=0.0,
            )
            # Don't print anything (keeps output clean on first run)

    # Return lists of regions - output shown later after MOUNT OLYMPUS STATUS
    return (registry, newly_acquired, newly_lost)


# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# HERMES DIVINE GUIDANCE
# ═══════════════════════════════════════════════════════════════════════════════


def show_hermes_passage(region: str, tier: str, gpu_count: int, print_fn) -> None:
    """
    Show HERMES divine guidance when quota needed (manual request).
    """
    gpu_type = THUNDER_TIERS[tier]["gpu_types"][0]
    tier_emoji = THUNDER_TIERS[tier]["emoji"]

    print_fn("\n" + "═" * 70)
    print_fn("   ⚡ HERMES TRISMEGISTUS - DIVINE QUOTA GUIDANCE ⚡")
    print_fn("═" * 70)
    print_fn(f"\n   Quest-Locked Region: {region.upper()}")
    print_fn(f"   Required Tier: {tier.upper()} {tier_emoji}")
    print_fn(f"   GPU Type: {gpu_type}")
    print_fn(f"   GPU Count: {gpu_count}")
    print_fn("\n   Zeus requires your mortal intervention to unlock this region!")
    print_fn("\n   Steps to Request Quota:")
    print_fn("   1. Visit GCP Console → IAM & Admin → Quotas")
    print_fn(f"   2. Filter by: Region = {region}, Metric = {gpu_type}")
    print_fn(f"   3. Request increase to {gpu_count} GPUs (or more)")
    print_fn("   4. Justification: 'Training vision-language models on Vertex AI'")
    print_fn("   5. Wait 1-3 days for approval (Zeus's divine timeline)")
    print_fn("\n   Once approved, this region will join the thunder fleet!")
    print_fn("\n" + "═" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# THUNDER BATTLE (PRICING)
# ═══════════════════════════════════════════════════════════════════════════════


def select_thunder_champion(
    thunder_ready: List[str],
    tier: str,
    gpu_count: int,
    print_fn,
    pricing_data: Optional[Dict] = None,
) -> Optional[str]:
    """
    Run thunder pricing battle and select cheapest region.

    Uses LIVE GCP pricing data (mirrors MECHA pattern).
    FAILS HARD if pricing unavailable (no fallback!).

    Returns:
        Champion region name (e.g., "us-east4") or None if pricing failed
    """
    if not thunder_ready:
        return None

    # ═══════════════════════════════════════════════════════════════
    # CRITICAL: Verify pricing data available BEFORE battle
    # ═══════════════════════════════════════════════════════════════
    if not pricing_data or "gpus_spot" not in pricing_data:
        print_fn("")
        print_fn("[red]❌ ZEUS BATTLE HALTED - NO LIVE PRICING DATA![/red]")
        print_fn("")
        print_fn("   Zeus requires LIVE GCP pricing from Artifact Registry.")
        print_fn("   Pricing data fetch failed or returned empty.")
        print_fn("")
        print_fn("   Check:")
        print_fn("   1. Artifact Registry container build succeeded")
        print_fn("   2. Pricing fetch container ran successfully")
        print_fn("   3. pricing_gcr_latest.json exists in Artifact Registry")
        print_fn("")
        print_fn("   Cannot proceed with Zeus battle without real pricing!")
        print_fn("   (We never use fake/fallback prices for real money decisions)")
        print_fn("")
        return None

    # Get pricing for all thunder-ready regions
    pricing = []
    failed_regions = []

    for region in thunder_ready:
        price = get_spot_pricing(region, tier, gpu_count, pricing_data)
        if price is None:
            failed_regions.append(region)
        else:
            pricing.append((region, price))

    # Check if we got pricing for at least one region
    if not pricing:
        print_fn("")
        print_fn("[red]❌ ZEUS BATTLE HALTED - NO PRICING FOR ANY REGION![/red]")
        print_fn("")
        print_fn(f"   Attempted regions: {', '.join(thunder_ready)}")
        print_fn(f"   Tier: {tier} ({THUNDER_TIERS[tier]['gpu_types'][0]})")
        print_fn("")
        print_fn("   All regions returned None for pricing!")
        print_fn("   Check if GPU type exists in pricing data SKUs.")
        print_fn("")
        return None

    # Warn about failed regions but continue with available pricing
    if failed_regions:
        print_fn("")
        print_fn(
            f"[yellow]⚠️  Pricing unavailable for: {', '.join(failed_regions)}[/yellow]"
        )
        print_fn(f"   Continuing battle with {len(pricing)} region(s)...")
        print_fn("")

    # Sort by price (cheapest first)
    pricing.sort(key=lambda x: x[1])

    champion_region, champion_price = pricing[0]
    tier_emoji = THUNDER_TIERS[tier]["emoji"]

    # Get price range for tier calculation
    min_price = pricing[0][1]
    max_price = pricing[-1][1]
    most_expensive_region = pricing[-1][0]
    most_expensive_price = pricing[-1][1]

    # Show battle header with region list
    print_fn("")
    print_fn(f"⚡ THUNDER-READY REGIONS ({tier.upper()} TIER {tier_emoji}):")
    print_fn("")

    # Roll call display (4 regions per line, random order)
    from .zeus_display import roll_call_display

    roll_call_display(thunder_ready, print_fn)
    print_fn("")
    print_fn(f"   Battling with {len(thunder_ready)} divine regions!")
    print_fn("")

    # ========================================
    # CONDENSED EPIC BATTLE (MECHA FORMAT!)
    # ========================================

    print_fn("   ∿◇∿ ZEUS THUNDER PRICING BATTLE BEGINS ∿◇∿\n")

    # Create printer for comic book style output
    printer = DivineThunderPrinter(status_callback=print_fn)

    # Get pricing dict
    prices = {region: price for region, price in pricing}

    # Baseline region (cheapest if available, otherwise fallback)
    baseline_region = champion_region
    baseline_price = champion_price

    # Pre-contender (baseline sets the bar)
    printer.print_panel(
        f'⚡ {baseline_region.upper()} sets the bar |${baseline_price:.2f}/hr| - "Challenge the gods if you dare!"'
    )

    # 2 challengers approach (skip baseline)
    other_regions = [r for r, p in pricing if r != baseline_region]
    num_challengers = min(2, len(other_regions))
    challengers = random.sample(other_regions, num_challengers) if other_regions else []

    last_phrase_template = None
    for challenger in challengers:
        price = prices[challenger]
        category = categorize_price(price, baseline_price)

        # Pick phrase, avoid repeats
        phrase_template = random.choice(SIZING_UP_PHRASES[category])
        if (
            phrase_template == last_phrase_template
            and len(SIZING_UP_PHRASES[category]) > 1
        ):
            for _ in range(3):
                phrase_template = random.choice(SIZING_UP_PHRASES[category])
                if phrase_template != last_phrase_template:
                    break

        phrase = phrase_template.format(region=challenger.upper(), price=price)
        last_phrase_template = phrase_template
        printer.print_panel(phrase)

    # 1 battle round (two regions fight!)
    if len(pricing) >= 2:
        pair = random.sample([r for r, p in pricing], 2)
        region1, region2 = pair
        price1, price2 = prices[region1], prices[region2]
        diff = abs(price1 - price2)

        if diff > 1.0:
            category = "much_cheaper"
        elif diff > 0.3:
            category = "cheaper"
        else:
            category = "close"

        if price1 < price2:
            winner, loser = region1, region2
        else:
            winner, loser = region2, region1

        if category == "close":
            phrase = random.choice(BATTLE_ROUND_PHRASES[category]).format(
                mecha1=region1.upper(), mecha2=region2.upper(), diff=diff
            )
        else:
            phrase = random.choice(BATTLE_ROUND_PHRASES[category]).format(
                winner=winner.upper(), loser=loser.upper(), diff=diff
            )
        printer.print_panel(phrase)

    # Victory emergence!
    emergence_phrase = random.choice(VICTORY_EMERGENCE_PHRASES)
    printer.print_panel(emergence_phrase)

    # Savings announcement (vs most expensive)
    savings_amt = most_expensive_price - champion_price
    savings_pct = int((savings_amt / most_expensive_price) * 100)
    printer.print_panel(
        f"⬡ {champion_region.upper()} |${champion_price:.2f}/hr| saves ${savings_amt:.2f} ({savings_pct}%) vs {most_expensive_region.upper()} |${most_expensive_price:.2f}/hr|!"
    )

    # Final divine blessing
    blessing = random.choice(VICTORY_DECLARATIONS)
    printer.print_panel(
        f"🔱 ⚡✨ {champion_region.upper()} |${champion_price:.2f}/hr| ✨⚡ {blessing}"
    )

    print_fn("")

    # Return values matching MECHA's format (savings_amt already calculated above!)
    return (
        champion_region,
        champion_price,
        most_expensive_region,
        most_expensive_price,
        savings_amt,
    )
