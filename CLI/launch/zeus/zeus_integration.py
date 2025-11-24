"""
⚡ Zeus Integration Wrapper - Thunder Battle System Entry Point

Integrates complete Zeus Thunder Battle System into Vertex AI launch CLI.

This module is called from CLI/launch/core.py BEFORE Vertex AI job submission.

Integration Flow:
1. Load Zeus Olympus (registry)
2. Check GPU TYPE change within tier → WIPE TIER if needed
3. Refresh quota snapshot for ALL tier regions → instant acquisition discovery
4. Run Zeus thunder pricing battle → select optimal region
5. Return selected champion region for Vertex AI submission
"""

# <claudes_code_comments>
# ** Function List **
# run_thunder_battle(project_id, tier_name, gpu_count, primary_region, pricing_data, status_callback, override_region, outlawed_regions) - Main entry point (called from core.py)
# run_thunder_battle_display_only(tier_name, gpu_count, status_callback) - Display-only mode (safe testing)
#
# ** Technical Review **
# Main integration point between Zeus thunder battle and Vertex AI launch workflow.
# Called from core.py AFTER user's GPU_TYPE has been validated and mapped to tier.
#
# Critical design: USER chooses GPU tier (via GPU_TYPE config), Zeus battles REGIONS only.
# Unlike MECHA (which auto-selects best C3 machine), Zeus receives tier as input parameter.
#
# Decision tree flow:
# 1. Override check: ZEUS_SINGLE_REGION_OVERRIDE set? → Skip battle, return override
# 2. Outlaw filtering: Filter ZEUS_OUTLAWED_REGIONS from eligible regions
# 3. Tier wipe check: GPU type changed within tier? → Wipe tier fleet, restart acquisition
# 4. Quota snapshot refresh: Check ALL tier regions for quota → discover new approvals instantly
# 5. Quota separation: Split regions into thunder-ready (have quota) vs quest-locked (no quota)
# 6. Battle paths:
#    - No thunder-ready: Return primary_region (quota snapshot refreshed for next launch)
#    - One thunder-ready: Instant victory → return solo region
#    - Multiple thunder-ready: EPIC BATTLE → select_thunder_champion() → return cheapest
# 7. Registry update: Saved after quota snapshot refresh (always fresh state)
#
# Acquisition = Quota discovery (finding existing quota user manually requested via GCP console).
# Snapshot runs BEFORE battle → instant discovery of ALL new permissions each launch.
# Receives pricing_data from core.py (same data MECHA uses) → threads to battle functions.
# Returns champion region string for Vertex AI job submission (e.g., "us-east4").
#
# Display-only mode: Shows Zeus output without affecting launch (testing/demo).
# Full mode: Controlled by ZEUS_ENABLED env var (default: true).
# </claudes_code_comments>

import random
from typing import List, Optional

from .zeus_battle import THUNDER_TIERS, refresh_quota_snapshot, select_thunder_champion
from .zeus_olympus import (
    check_thunder_tier_changed,
    get_available_thunder_regions,
    get_thunder_ready_regions,
    load_olympus_registry,
    save_olympus_registry,
    wipe_tier_fleet,
)

# Divine geometry symbols (random winner banners)
DIVINE_INNER = [
    "◢",
    "◣",
    "◤",
    "◥",
    "▲",
    "▼",
    "◄",
    "►",
    "⬡",
    "⬢",
    "⬣",
    "◆",
    "◇",
    "▶",
    "◀",
    "▷",
    "◁",
    "▸",
    "◂",
    "▹",
    "◃",
    "△",
    "▽",
    "▻",
    "◅",
    "◊",
    "❖",
    "◉",
    "◎",
    "●",
    "◐",
    "◑",
    "◒",
    "◓",
    "▪",
    "▫",
    "■",
    "□",
    "▢",
    "✦",
    "✧",
    "✶",
    "✷",
    "✸",
    "⚡",
    "☇",
    "☈",
]
DIVINE_OUTER = [
    "˙",
    "˚",
    "·",
    "∙",
    "◦",
    "⋅",
    "∘",
    "˗",
    "˜",
    "¯",
    "˸",
    "‧",
    "˓",
    "˔",
    "˕",
    "˖",
    "˱",
    "˲",
    "˳",
    "˴",
    "˵",
    "˶",
    "˷",
    "˸",
    "°",
    "⁰",
    "-",
    "~",
    "˭",
    "˯",
    "˰",
    "⁕",
    "⁘",
    "⁙",
    "⁚",
    "⁛",
    "⚡",
    "☁️",
]


def run_thunder_battle(
    project_id: str,
    tier_name: str,
    gpu_count: int,
    primary_region: str,
    pricing_data: dict,
    status_callback=None,
    override_region: Optional[str] = None,
    outlawed_regions: Optional[List[str]] = None,
) -> str:
    """
    Run complete Zeus thunder battle system and return CHAMPION region! ⚡

    Args:
        project_id: GCP project ID
        tier_name: Thunder tier (e.g., "tempest" for H100)
        gpu_count: Number of GPUs needed
        primary_region: Default region (fallback)
        pricing_data: Pricing data from battle
        status_callback: Optional callback for status updates (TUI compatibility)
        override_region: ZEUS_SINGLE_REGION_OVERRIDE from config (skips battle!)
        outlawed_regions: ZEUS_OUTLAWED_REGIONS from config (exclude these)

    Returns:
        Champion thunder region (override, price battle winner, or primary fallback)

    Side Effects:
        - May wipe tier fleet if GPU type changed
        - Runs epic Zeus thunder pricing battle to select CHAMPION
        - Acquires ONE missing region in background (passive collection)
        - Updates Zeus Olympus registry
    """

    # Initialize outlawed_regions if None
    outlawed_regions = outlawed_regions or []

    # Helper function for output (works in both CLI and TUI)
    def output(msg=""):
        if status_callback:
            status_callback(msg)
        else:
            print(msg)

    # ═══════════════════════════════════════════════════════════════
    # ZEUS_SINGLE_REGION_OVERRIDE: Instant Victory (Skip Battle!)
    # ═══════════════════════════════════════════════════════════════
    if override_region:
        output("\n---")
        output("⚡ ZEUS PRICE BATTLE SYSTEM GO! (Zeus finds cheapest GPU pricing)")
        output("---")
        output("")

        # Validate tier exists
        if tier_name not in THUNDER_TIERS:
            output(f"❌ Invalid tier: '{tier_name}'")
            output(f"   Valid tiers: {', '.join(THUNDER_TIERS.keys())}")
            raise ValueError(f"Invalid tier: {tier_name}")

        tier_regions = THUNDER_TIERS[tier_name]["regions"]

        # Validate region is in tier
        if override_region not in tier_regions:
            output(
                f"❌ ZEUS_SINGLE_REGION_OVERRIDE = '{override_region}' INVALID for {tier_name}!"
            )
            output(f"   Valid {tier_name} regions: {', '.join(tier_regions[:5])}...")
            output(f"   Total valid: {len(tier_regions)} regions")
            output("")
            raise ValueError(
                f"Invalid override: {override_region} (not in {tier_name} tier)"
            )

        # INSTANT VICTORY!
        tier_emoji = THUNDER_TIERS[tier_name]["emoji"]

        output(f"   🎯 ZEUS_SINGLE_REGION_OVERRIDE = {override_region}")
        output(f"   {tier_emoji} {tier_name.upper()} tier validated!")
        output("")
        output(f"   ⚡ INSTANT VICTORY: {override_region.upper()}!")
        output(f"   ⚡ No battle needed - Zeus has spoken!")
        output("")

        return override_region

    # ═══════════════════════════════════════════════════════════════
    # Validate tier
    # ═══════════════════════════════════════════════════════════════

    if tier_name not in THUNDER_TIERS:
        output(f"❌ Invalid tier: '{tier_name}'")
        raise ValueError(f"Invalid tier: {tier_name}")

    # Outlawed regions are filtered later from thunder-ready regions
    # (After API discovers which regions have quota)

    # ═══════════════════════════════════════════════════════════════
    # Load Registry + Check GPU Type Change
    # ═══════════════════════════════════════════════════════════════

    registry = load_olympus_registry()
    gpu_type = THUNDER_TIERS[tier_name]["gpu_types"][0]

    if check_thunder_tier_changed(registry, tier_name, gpu_type):
        output("\n⚡ GPU TYPE CHANGED! WIPING {tier_name.upper()} TIER!")
        output(f"   Previous: {registry['thunder_fleets'][tier_name]['gpu_type']}")
        output(f"   Current: {gpu_type}")
        output(f"   Wiping {tier_name} tier fleet...")
        output("")

        registry = wipe_tier_fleet(registry, tier_name, gpu_type)
        save_olympus_registry(registry)

    # ═══════════════════════════════════════════════════════════════
    # QUOTA SNAPSHOT REFRESH - Check ALL regions for new quota!
    # ═══════════════════════════════════════════════════════════════
    # This discovers new quota approvals BEFORE battle (instant acquisition!)

    registry, newly_acquired, newly_lost = refresh_quota_snapshot(
        registry, tier_name, gpu_count, project_id, output, pricing_data
    )
    save_olympus_registry(registry)  # Save snapshot immediately

    # ═══════════════════════════════════════════════════════════════
    # Get Thunder-Ready Regions (filter out wrathful)
    # ═══════════════════════════════════════════════════════════════

    # Get ALL regions for this tier from the registry
    tier_data = registry["thunder_fleets"].get(tier_name, {})
    all_regions = list(tier_data.get("regions", {}).keys())

    thunder_ready = get_available_thunder_regions(
        registry, tier_name, all_regions, outlawed_regions
    )

    # ═══════════════════════════════════════════════════════════════
    # Special Case: NO Thunder-Ready Regions
    # ═══════════════════════════════════════════════════════════════

    if not thunder_ready:
        output("\n---")
        output("⚡ ZEUS PRICE BATTLE SYSTEM GO! (Zeus finds cheapest GPU pricing)")
        output("---")
        output(f"\n⚠️  No thunder-ready regions in {tier_name} tier yet!")
        output(f"   Falling back to primary: {primary_region}")
        output(f"\n   Quota snapshot refreshed - check again next launch...")
        output("")

        return primary_region

    # ═══════════════════════════════════════════════════════════════
    # Special Case: Solo Thunder Region
    # ═══════════════════════════════════════════════════════════════

    if len(thunder_ready) == 1:
        solo_region = thunder_ready[0]
        tier_emoji = THUNDER_TIERS[tier_name]["emoji"]

        output("\n---")
        output("⚡ ZEUS PRICE BATTLE SYSTEM GO! (Zeus finds cheapest GPU pricing)")
        output("---")
        output("")
        output("⚡  THIS THUNDER REGION BATTLES ALONE AND WINS! ⚡")
        output("")

        # Get price for display
        from .zeus_battle import get_spot_pricing

        price = get_spot_pricing(solo_region, tier_name, gpu_count, pricing_data)

        output(
            f"   🏆 CHAMPION: {solo_region} ({gpu_count}×{THUNDER_TIERS[tier_name]['gpu_types'][0].replace('NVIDIA_', '').replace('_80GB', '')}) |${price:.2f}/hr|"
        )
        output("")

        return solo_region

    # ═══════════════════════════════════════════════════════════════
    # FULL THUNDER BATTLE (2+ regions)
    # ═══════════════════════════════════════════════════════════════

    tier_emoji = THUNDER_TIERS[tier_name]["emoji"]

    output("\n---")
    output("⚡ ZEUS THUNDER PRICING SYSTEM GO!")
    output("---")

    # Calculate quest-locked count
    quest_locked_count = len(all_regions) - len(thunder_ready)

    output("⚡☁️ MOUNT OLYMPUS STATUS:")
    output(f"   Thunder-Ready: {len(thunder_ready)}/{len(all_regions)} regions")
    output(f"   Quest-Locked: {quest_locked_count} (awaiting quota)")
    output(f"   Thunder Tier: {tier_name.upper()} {tier_emoji}")

    # Show acquisition results (if any regions were acquired/lost this launch)
    if newly_acquired or newly_lost:
        output("")
        # Random thunder symbols for flair (matching MECHA's postfix style!)
        thunder_flair = ["⚡", "✦", "✧", "✨", "★", "☆", "◆", "◇", "◉", "○"]
        postfix = "".join(random.sample(thunder_flair, 3))
        output(f"   ⚡ NEW THUNDER ARRIVALS! {postfix}")
        output("")

        for region in newly_acquired:
            output(f"      ⚡ {region.upper()} now thunder-ready! (quota approved!)")

        for region in newly_lost:
            output(f"      ⚠️  {region.upper()} quota revoked or expired!")

    output("")

    # Run pricing battle (returns 5 values like MECHA!)
    champion, champion_price, compare_region, compare_price, savings = (
        select_thunder_champion(
            thunder_ready, tier_name, gpu_count, output, pricing_data
        )
    )

    if not champion:
        # Battle failed (pricing unavailable) - HALT LAUNCH!
        output("")
        output("[red]❌ LAUNCH HALTED - Zeus battle failed![/red]")
        output("")
        output("   Cannot proceed without live GCP pricing data.")
        output("   Fix pricing data fetch and try again.")
        output("")
        raise RuntimeError("Zeus battle failed: No live pricing data available")

    savings_percent = (savings / compare_price * 100) if compare_price > 0 else 0

    # Build winner banner with random geometry (MATCH MECHA FORMAT!)
    outer_left = "".join(random.choice(DIVINE_OUTER) for _ in range(3))
    inner_left = "".join(random.choice(DIVINE_INNER) for _ in range(4))
    inner_right = "".join(random.choice(DIVINE_INNER) for _ in range(4))
    outer_right = "".join(random.choice(DIVINE_OUTER) for _ in range(3))

    output("")
    output(
        f"   {outer_left}{inner_left} [bold cyan]{champion.upper()}[/bold cyan] WINS! {inner_right}{outer_right}"
    )
    output("")
    output(f"   ∿◇∿ ZEUS BATTLE COMPLETE ∿◇∿")
    output(f"   ∿◇∿ CHAMPION:  |${champion_price:.2f}/hr| {champion} ∿◇∿")
    output(
        f"   ∿◇∿ SAVES:     {savings_percent:.0f}% |${champion_price:.2f}/hr| vs {compare_region} |${compare_price:.2f}/hr| ∿◇∿\n"
    )

    return champion


def run_thunder_battle_display_only(
    tier_name: str, gpu_count: int, status_callback=None
) -> None:
    """
    Display-only Zeus output (Phase 0: safe testing).

    Shows canonical Zeus thunder battle WITHOUT selecting region.
    Used for user testing before wiring up real backend.

    Returns:
        None (display-only, doesn't affect launch)
    """

    def output(msg=""):
        if status_callback:
            status_callback(msg)
        else:
            print(msg)

    if tier_name not in THUNDER_TIERS:
        output(f"⚠️  Invalid tier: {tier_name}")
        return

    tier_emoji = THUNDER_TIERS[tier_name]["emoji"]
    tier_regions = THUNDER_TIERS[tier_name]["regions"]

    output("\n---")
    output(
        "⚡ ZEUS PRICE BATTLE SYSTEM GO! (Zeus finds cheapest GPU pricing) [DISPLAY ONLY]"
    )
    output("---")
    output(f"\n   ⚡☁️ MOUNT OLYMPUS STATUS:")
    output(f"   Thunder Tier: {tier_name.upper()} {tier_emoji}")
    output(
        f"   Thunder-Ready: {len(tier_regions) - 2}/{len(tier_regions)} regions (simulated)"
    )
    output(f"   Quest-Locked: 2 (Zeus's trial awaits!)")
    output(f"   Current Quest: {tier_name} × {gpu_count} GPUs")
    output("")
    output(f"   ⚡ THUNDER-READY REGIONS ({tier_name.upper()} {tier_emoji}):")
    output(f"   {' ∿ '.join(tier_regions[:3])} ∿ ...")
    output("")
    output("   ∿◇∿ ZEUS THUNDER PRICING BATTLE BEGINS ∿◇∿")
    output("")
    output("        ⚡ US-CENTRAL1 summons lightning |$16.80/hr| (8×$2.10)")
    output("   ☁️  ASIA-SOUTHEAST1 |$22.40/hr| (8×$2.80) arrives...")
    output("        ⚡ US-EAST4 CHANNELS PURE DIVINE POWER! |$16.40/hr|")
    output("             ● THE CHAMPION DESCENDS FROM OLYMPUS!")
    output("")
    output("   ˰˹·◈⚡▫◄ US-EAST4 CLAIMS ZEUS'S THUNDER! ◒⚡▢⬥˚˔˴")
    output("")
    output(f"   ∿◇∿ CHAMPION:  |$16.40/hr| us-east4 ({gpu_count}×H100) ∿◇∿")
    output("   ∿◇∿ SAVES:     27% vs asia-southeast1 ∿◇∿")
    output("   ∿◇∿ 24h DIVINE FAVOR: $144 saved! ∿◇∿")
    output("")
    output("   ℹ️  DISPLAY ONLY - Region selection unchanged")
    output("")
