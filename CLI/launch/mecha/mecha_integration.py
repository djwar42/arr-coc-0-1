"""
🤖 MECHA Integration Wrapper (メカ)

Integrates complete MECHA Battle System into launch CLI.

This module is called from CLI/launch/core.py BEFORE worker pool creation.

Integration Flow:
1. Load MECHA Hangar (registry)
2. Check CPU change → WIPE ALL 18 MECHAS if needed
3. Run MECHA price battle → select optimal region
4. Passively deploy ONE missing MECHA (background)
5. Return selected region for pool creation
"""

import random
import subprocess
import sys
from pathlib import Path

# Relative imports (... = up 3 levels to CLI/)
from ...config.constants import PROJECT_ROOT
from ...shared.quota import get_cloud_build_c3_quotas

# Same package imports (. = current mecha/ directory)
from .mecha_hangar import (load_registry, save_registry, check_machine_type_changed,
                           wipe_all_mechas, get_deployed_mechas, get_missing_mechas)
from .mecha_phrases import get_mecha_phrase
from .mecha_regions import ALL_MECHA_REGIONS, C3_REGIONS
from .mecha_acquire import BEACON_WAIT_MINUTES
from .mecha_display import (separate_by_quota, display_sidelined_mechas,
                            display_battle_ready_mechas)

sys.path.insert(0, str(PROJECT_ROOT))


# Winner banner geometry symbols
INNER_GEOMETRY = ['◢', '◣', '◤', '◥', '▲', '▼', '◄', '►', '⬡', '⬢', '⬣', '◆', '◇',
                  '▶', '◀', '▷', '◁', '▸', '◂', '▹', '◃', '△', '▽', '▻', '◅', '▴', '▾', '▵', '▿',
                  '◊', '❖', '⟐', '⟡', '⬖', '⬗', '⬘', '⬙', '⬥', '⬦', '⬧',
                  '◉', '◎', '●', '◐', '◑', '◒', '◓', '▪', '▫', '■', '□', '▢',
                  '✦', '✧', '✶', '✷', '✸', '◭', '◮']
OUTER_GEOMETRY = ['˙', '˚', '·', '∙', '◦', '⋅', '∘', '∗', '⁎', '˗', '˜', '¯', '˸', '‧',
                  '˓', '˔', '˕', '˖', '˱', '˲', '˳', '˴', '˵', '˶', '˷', '˸', '˹', '˺', '˻', '˼',
                  '°', '⁰', '¹', '²', '³', '⁴', '⁵', '-', '~', '˭', '˯', '˰',
                  '⁕', '⁘', '⁙', '⁚', '⁛', '⁜', '⁝']


def run_mecha_battle(project_id: str, best_machine: str, primary_region: str, pricing_data: dict, status_callback=None, override_region: str = None, outlawed_regions: list = None) -> str:
    """
    Run complete MECHA battle system and return CHAMPION region! ⚔️

    Args:
        project_id: GCP project ID
        best_machine: Machine type (e.g., "c3-standard-176")
        primary_region: Default region (fallback)
        pricing_data: Pricing data from epic battle
        status_callback: Optional callback for status updates (for TUI compatibility)
        override_region: C3_SINGLE_REGION_OVERRIDE from config (skips battle if set!)
        outlawed_regions: MECHA_OUTLAWED_REGIONS from config (exclude these from battle)

    Returns:
        Champion MECHA region (override, price battle winner, or primary fallback)

    Side Effects:
        - May wipe ALL 15 pools globally if CPU changed
        - Runs epic MECHA price battle to select CHAMPION
        - Deploys ONE missing MECHA in background (passive collection)
        - Updates MECHA Hangar registry
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
    # C3_SINGLE_REGION_OVERRIDE: Instant Victory (Skip Battle!)
    # ═══════════════════════════════════════════════════════════════
    if override_region:
        output("\n---")
        output("🤖 MECHA PRICE BATTLE SYSTEM GO! (Mecha finds cheapest Cloud Build pricing)")
        output("---")
        output("")

        # Validate region is in the 18 valid C3 regions
        if override_region not in ALL_MECHA_REGIONS:
            output(f"❌ C3_SINGLE_REGION_OVERRIDE = '{override_region}' is INVALID!")
            output(f"   Valid C3 regions: {', '.join(ALL_MECHA_REGIONS[:5])}...")
            output(f"   Total valid: {len(ALL_MECHA_REGIONS)} regions")
            output("")
            raise ValueError(f"Invalid C3_SINGLE_REGION_OVERRIDE: {override_region} (not in valid C3 regions)")

        # Load registry to check MECHA status
        registry = load_registry()
        mechas = registry.get("mechas", {})

        # Check if MECHA exists for override region
        if override_region not in mechas:
            output(f"❌ C3_SINGLE_REGION_OVERRIDE = '{override_region}' MECHA NOT ACQUIRED!")
            output(f"   MECHA Hangar has no worker pool for {override_region}")
            output(f"   Acquired MECHAs: {len(mechas)}/{len(ALL_MECHA_REGIONS)}")
            output("")
            raise ValueError(f"C3_SINGLE_REGION_OVERRIDE region '{override_region}' has no MECHA (run setup first)")

        # Check if MECHA is sidelined by quota
        vcpus_needed = int(best_machine.split("-")[-1])
        quotas = get_cloud_build_c3_quotas(project_id)
        region_quota = quotas.get(override_region, {}).get("limit", 0)

        if region_quota < vcpus_needed:
            output(f"❌ C3_SINGLE_REGION_OVERRIDE = '{override_region}' MECHA SIDELINED (NO QUOTA)!")
            output(f"   Need: {vcpus_needed} vCPUs for {best_machine}")
            output(f"   Have: {region_quota} vCPUs quota in {override_region}")
            output("")
            raise ValueError(f"C3_SINGLE_REGION_OVERRIDE region '{override_region}' has insufficient quota ({region_quota}/{vcpus_needed} vCPUs)")

        # Check if MECHA is tired (FAILED status)
        mecha_status = mechas[override_region].get("operational_status", "UNKNOWN")
        if mecha_status != "OPERATIONAL":
            output(f"⚠️  C3_SINGLE_REGION_OVERRIDE = '{override_region}' MECHA is TIRED (status: {mecha_status})")
            output(f"   Proceeding anyway (override is a hard requirement)")
            output("")

        # ✅ INSTANT VICTORY - One line, no battle!
        output(f"🏆 MECHA {override_region.upper()} CLAIMS INSTANT VICTORY!")
        output(f"   (C3_SINGLE_REGION_OVERRIDE was set, using {override_region})")
        output("")

        return override_region

    # ═══════════════════════════════════════════════════════════════
    # Normal MECHA Price Battle (No Override Set)
    # ═══════════════════════════════════════════════════════════════

    # ═══════════════════════════════════════════════════════════════
    # Filter Out Outlawed Regions (Quiet Validation)
    # ═══════════════════════════════════════════════════════════════

    # Validate outlaws quietly (no verbose output - shown later in hangar status)
    valid_outlaws = []
    if outlawed_regions:
        for region in outlawed_regions:
            if region in ALL_MECHA_REGIONS:
                valid_outlaws.append(region)

    # Filter outlawed regions from consideration
    eligible_regions = [r for r in ALL_MECHA_REGIONS if r not in valid_outlaws]

    # Safety check: prevent total lockout
    if not eligible_regions:
        output("\n❌ ALL VALID REGIONS OUTLAWED! Cannot proceed.")
        output(f"   {len(ALL_MECHA_REGIONS)} C3 regions available")
        output(f"   {len(valid_outlaws)} regions banned")
        output("")
        raise ValueError("MECHA_OUTLAWED_REGIONS bans all valid C3 regions - no regions available for battle")

    # Pricing check now happens in core.py (before MECHA is called)
    output("\n---")
    output("🤖 MECHA PRICE BATTLE SYSTEM GO! (Mecha finds cheapest Cloud Build pricing)")
    output("---")
    
    # Step 1: Load MECHA Hangar registry
    registry = load_registry()
    
    # Step 2: Check for CPU change (WIPE ALL if changed!)
    current_vcpus = int(best_machine.split("-")[-1])
    
    if check_machine_type_changed(registry, best_machine):
        output(f"\n⚠️  CPU NUMBER CHANGED!")
        output(f"   Previous: {registry.get('machine_type', 'UNKNOWN')}")
        output(f"   Current:  {best_machine} ({current_vcpus} vCPUs)")
        output(f"\n🔥 WIPING ALL 15 MECHAS GLOBALLY! 🔥")
        output(f"   Starting fresh collection with new CPU count...")
        output("")
        
        # WIPE ALL MECHAS in registry
        registry = wipe_all_mechas(registry, best_machine)
        save_registry(registry)
        
        output(f"✓ All pools wiped! Fresh start initiated.")
        output("")
    
    # Step 3: Get deployed MECHAs (all pools in registry, filtered by eligible_regions)
    all_deployed = get_deployed_mechas(registry, best_machine)
    deployed_regions = [r for r in all_deployed if r in eligible_regions]  # Filter outlawed
    total_mechas = len(eligible_regions)  # Total eligible (excluding outlawed)
    deployed_count = len(deployed_regions)  # Total deployed (usable + sidelined)

    # Get missing and fatigue info (filtered by eligible_regions)
    all_missing = get_missing_mechas(registry, ALL_MECHA_REGIONS)
    missing_regions = [r for r in all_missing if r in eligible_regions]  # Filter outlawed
    from .mecha_hangar import get_available_mechas
    all_available, all_fatigued = get_available_mechas(registry, ALL_MECHA_REGIONS)
    available_regions = [r for r in all_available if r in eligible_regions]  # Filter outlawed
    fatigued_regions = [r for r in all_fatigued if r in eligible_regions]  # Filter outlawed

    # Filter MECHAs by quota: separate battle-ready from sidelined
    vcpus_needed = int(best_machine.split("-")[-1])  # "c3-standard-176" → 176
    quotas = get_cloud_build_c3_quotas(project_id)
    battle_ready_regions, sidelined_regions = separate_by_quota(deployed_regions, project_id, vcpus_needed)

    # "Acquired" = Battle-ready (user-facing term for "ready to go")
    acquired_regions = battle_ready_regions
    acquired_count = len(acquired_regions)

    # Check tiredness status (only for battle-ready regions)
    rested_regions = []
    tired_regions = []
    for region in battle_ready_regions:
        mecha_info = registry.get("mechas", {}).get(region, {})
        status = mecha_info.get("operational_status", "OPERATIONAL")
        # RUNNING = rested, FAILED/other = tired
        if status == "OPERATIONAL":
            rested_regions.append(region)
        else:
            tired_regions.append(region)

    output(f"🏭 MECHA HANGAR STATUS:")

    # Build acquired line with optional fatigue and outlaw brackets
    acquired_line = f"   Acquired: {acquired_count}/{total_mechas} MECHAs"

    # Add fatigue bracket if any
    if fatigued_regions:
        acquired_line += f" ({len(fatigued_regions)} fatigued)"

    # Add outlaw bracket if any (compact, inline with acquired count)
    if valid_outlaws:
        if len(valid_outlaws) == 1:
            # Single outlaw: show region name + flag emoji
            region = valid_outlaws[0]
            flag = C3_REGIONS.get(region, {}).get("flag", "🌍")
            acquired_line += f" (1 outlawed: {region} {flag})"
        else:
            # Multiple outlaws: just show count (too long to list all)
            acquired_line += f" ({len(valid_outlaws)} outlawed)"

    output(acquired_line)

    # Show sidelined count (no quota)
    if sidelined_regions:
        output(f"   Sidelined: {len(sidelined_regions)} (no quota)")

    output(f"   Hyperarmour: {best_machine}")

    output("")

    # Display sidelined and usable MECHAs
    if sidelined_regions:
        display_sidelined_mechas(sidelined_regions, quotas, project_id, output)

    if battle_ready_regions:
        display_battle_ready_mechas(battle_ready_regions, quotas, output)

    # Step 4: CHAMPION SELECTION - Battle with battle-ready MECHAs only!

    # Track if we already ran a fleet blast to avoid double blast situation.
    # When EMPTY HANGAR blast acquires some MECHAs but others are quota-blocked,
    # we don't want to immediately blast those same quota-blocked regions again.
    already_ran_fleet_blast = False

    # Handle special cases: Check EMPTY HANGAR FIRST!
    if not deployed_regions:
        # EMPTY HANGAR → LAUNCH FLEET BLAST IMMEDIATELY! 🚀
        output(f"📍 EMPTY HANGAR DETECTED!")
        output(f"   No MECHAs acquired yet!")
        output(f"   WE ARE UNABLE TO PRICE BATTLE OVER C3 WORKER POOLS!")
        output(f"   Launching FLEET BLAST to acquire MECHAs...")
        output("")

        # Import and run Fleet Blast (OUTLAW PROTECTED!)
        from .mecha_acquire import blast_mecha_fleet

        successful, failed = blast_mecha_fleet(
            project_id,
            best_machine,
            status_callback=status_callback,
            eligible_regions=eligible_regions  # Excludes outlaws!
        )

        # Reload registry (Fleet Blast registered the MECHAs)
        registry = load_registry()
        all_deployed = get_deployed_mechas(registry, best_machine)

        # OUTLAW FILTER: Only use eligible regions (no outlaws!)
        deployed_regions_new = [r for r in all_deployed if r in eligible_regions]
        deployed_count_new = len(deployed_regions_new)

        # Re-do quota filtering for newly acquired MECHAs
        battle_ready_regions, sidelined_regions = separate_by_quota(deployed_regions_new, project_id, vcpus_needed)

        # "Acquired" = Battle-ready after fleet blast
        acquired_count_new = len(battle_ready_regions)

        output("")
        output(f"🏭 FLEET BLAST RESULTS:")
        output(f"   Acquired: {acquired_count_new}/{total_mechas} MECHAs")
        output("")

        # Mark that we already ran fleet blast (avoids trying quota-blocked regions again)
        already_ran_fleet_blast = True

        # Display sidelined/battle-ready
        if sidelined_regions:
            display_sidelined_mechas(sidelined_regions, quotas, project_id, output)

        if battle_ready_regions:
            display_battle_ready_mechas(battle_ready_regions, quotas, output)

        # After fleet blast, check if we have battle-ready MECHAs
        if len(battle_ready_regions) == 0:
            output("")
            output("❌ NO BATTLE-READY MECHAS AFTER FLEET BLAST!")
            output(f"   All acquired MECHAs need quota increases")
            output(f"   Falling back to PRIMARY: {primary_region}")
            output("")
            return primary_region
        elif len(battle_ready_regions) == 1:
            # Solo MECHA after fleet blast
            solo_region = battle_ready_regions[0]
            output("")
            output("⚔️  THIS MECHA BATTLES ALONE AND WINS! ⚔️")
            output("")
            from .mecha_battle_epic import get_region_price
            price = get_region_price(solo_region, best_machine)
            output(f"   🏆 CHAMPION: {solo_region} |${price:.2f}/hr|")
            output("")
            return solo_region
        # Otherwise fall through to battle logic below

    elif len(battle_ready_regions) == 0:
        # HAVE MECHAs but ALL need quota increases
        output("")
        output("❌ NO BATTLE-READY MECHAS!")
        output(f"   All acquired MECHAs need quota increases")
        output(f"   Falling back to PRIMARY: {primary_region}")
        output("")
        return primary_region

    elif len(battle_ready_regions) == 1:
        # SOLO MECHA WINS!
        solo_region = battle_ready_regions[0]
        output("")
        output("⚔️  THIS MECHA BATTLES ALONE AND WINS! ⚔️")
        output("")

        # Get price for display
        from .mecha_battle_epic import get_region_price
        price = get_region_price(solo_region, best_machine)

        output(f"   🏆 CHAMPION: {solo_region} |${price:.2f}/hr|")
        output("")

        return solo_region

    # If we get here, we have 2+ battle-ready MECHAs - proceed with normal battle logic
    if len(battle_ready_regions) == len(eligible_regions):
        # FULL FLEET → EPIC BATTLE! 🎮 (All eligible regions acquired!)
        output(" ⚡⚡⚡⚡⚡⚡⚡⚡⚡")
        output(f"⚡ FULL FLEET ASSEMBLED ⚡  ALL {len(eligible_regions)} ELIGIBLE REGIONS ONLINE!")
        output(" ⚡⚡⚡⚡⚡⚡⚡⚡⚡")
        output("")

        # Go straight to EPIC battle - pick from rested (or all battle-ready if none rested)
        battle_ready = rested_regions if rested_regions else battle_ready_regions

        from .mecha_battle_epic import epic_mecha_price_battle

        selected_region, champion_price, compare_region, compare_price, savings = epic_mecha_price_battle(battle_ready, pricing_data, status_callback, best_machine)

        savings_percent = (savings / compare_price * 100) if compare_price > 0 else 0

        # Build winner banner with random geometry
        outer_left = ''.join(random.choice(OUTER_GEOMETRY) for _ in range(3))
        inner_left = ''.join(random.choice(INNER_GEOMETRY) for _ in range(4))
        inner_right = ''.join(random.choice(INNER_GEOMETRY) for _ in range(4))
        outer_right = ''.join(random.choice(OUTER_GEOMETRY) for _ in range(3))

        output("")
        output(f"   {outer_left}{inner_left} [bold cyan]{selected_region.upper()}[/bold cyan] WINS! {inner_right}{outer_right}")
        output("")
        output(f"   ∿◇∿ MECHA BATTLE COMPLETE ∿◇∿")
        output(f"   ∿◇∿ CHAMPION:  |${champion_price:.2f}/hr| {selected_region} ∿◇∿")
        output(f"   ∿◇∿ SAVES:     {savings_percent:.0f}% |${champion_price:.2f}/hr| vs {compare_region} |${compare_price:.2f}/hr| ∿◇∿\n")

    else:
        # PARTIAL HANGAR (less than full fleet)
        # Check if there are non-fatigued missing regions we can acquire.
        # Skip if we just ran EMPTY HANGAR blast (avoids re-attempting quota-blocked regions)
        if available_regions and not already_ran_fleet_blast:
            # Send beacon to available regions!
            output(f"📡 ACQUIRING MORE MECHAs!")
            output(f"   Current: {acquired_count}/{total_mechas}")  # Only count MECHAs with quota
            output(f"   Available: {len(available_regions)} non-fatigued regions")
            output(f"   Fatigued: {len(fatigued_regions)} regions resting")
            output("")

            from .mecha_acquire import blast_mecha_fleet
            successful, failed = blast_mecha_fleet(
                project_id,
                best_machine,
                status_callback=status_callback,
                eligible_regions=eligible_regions  # Excludes outlaws!
            )

            # Reload registry (OUTLAW PROTECTED!)
            registry = load_registry()
            all_deployed_new = get_deployed_mechas(registry, best_machine)
            deployed_regions_new2 = [r for r in all_deployed_new if r in eligible_regions]  # Filter outlaws!
            deployed_count_new2 = len(deployed_regions_new2)

            # Re-filter by quota after acquisition
            battle_ready_regions_new, _ = separate_by_quota(deployed_regions_new2, project_id, vcpus_needed)
            acquired_count_after = len(battle_ready_regions_new)

            output("")
            output(f"🏭 ACQUISITION RESULTS:")
            output(f"   Acquired: {acquired_count_after}/{total_mechas} MECHAs")
            output("")
        else:
            # All missing regions are fatigued - battle with what we have!
            output(f"   Battling with {len(battle_ready_regions)} MECHAs!")
            output("")

        # Run battle with what we have (battle-ready MECHAs only)
        battle_ready = rested_regions if rested_regions else battle_ready_regions
        from .mecha_battle_epic import epic_mecha_price_battle
        selected_region, champion_price, compare_region, compare_price, savings = epic_mecha_price_battle(battle_ready, pricing_data, status_callback, best_machine)

        savings_percent = (savings / compare_price * 100) if compare_price > 0 else 0

        # Build winner banner with random geometry
        outer_left = ''.join(random.choice(OUTER_GEOMETRY) for _ in range(3))
        inner_left = ''.join(random.choice(INNER_GEOMETRY) for _ in range(4))
        inner_right = ''.join(random.choice(INNER_GEOMETRY) for _ in range(4))
        outer_right = ''.join(random.choice(OUTER_GEOMETRY) for _ in range(3))

        output("")
        output(f"   {outer_left}{inner_left} [bold cyan]{selected_region.upper()}[/bold cyan] WINS! {inner_right}{outer_right}")
        output("")
        output(f"   ∿◇∿ MECHA BATTLE COMPLETE ∿◇∿")
        output(f"   ∿◇∿ CHAMPION:  |${champion_price:.2f}/hr| {selected_region} ∿◇∿")
        output(f"   ∿◇∿ SAVES:     {savings_percent:.0f}% |${champion_price:.2f}/hr| vs {compare_region} |${compare_price:.2f}/hr| ∿◇∿\n")

    # Record initial build entry to campaign stats (before CloudBuild starts)
    try:
        from .campaign_stats import record_build_result
        record_build_result(
            region=selected_region,
            success=False,  # In progress
            duration_minutes=0.0,
            queue_wait_minutes=0.0,
            build_id="pending",  # Will be updated when CloudBuild ID is extracted
            build_type="arr-pytorch-base",
            machine_type=best_machine,
            status="MECHA_SELECTED"  # Initial status: MECHA battle winner chosen
        )
        # Silent success - campaign stats initialized
    except Exception as e:
        # Show error but don't fail MECHA battle
        output(f"[dim yellow]   ⚠️  Campaign stats error: {str(e)}[/dim yellow]")

    return selected_region
