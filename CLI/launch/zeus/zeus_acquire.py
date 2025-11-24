"""
⚡ Thunder Fleet Blast - Simultaneous Multi-Region GPU Quota Discovery

Checks GPU quota in ALL regions simultaneously (ONE API call).
Shows epic thunder-themed announcements when regions have quota!
Those without quota are marked QUEST_LOCKED.

Provides visual display for quota discovery (used by refresh_quota_snapshot in zeus_battle.py).
"""

# <claudes_code_comments>
# ** Function List **
# animate_arrival() - Show animated ASCII arrival effect (◦·• progression)
# thunder_fleet_blast(registry, tier, gpu_count, project_id, pricing_data, print_fn) - Discover ALL regions with quota (1 API call)
#
# ** Technical Review **
# Optional Thunder Fleet Blast system for Zeus - provides epic visual display when discovering GPU quotas.
# Mirrors MECHA's mecha_acquire.py but for GPU regions instead of C3 build regions.
#
# thunder_fleet_blast() calls get_all_gpu_quotas_for_type() which returns ALL regions in ONE API call.
# For each region with quota ≥ gpu_count:
# - Shows animated arrival effect (◦·• with random delays)
# - Displays epic announcement ("⚡ US-CENTRAL1 THUNDER READY!")
# - Updates Olympus registry to THUNDER_READY
# Regions without quota are marked QUEST_LOCKED silently.
#
# This is OPTIONAL display-only enhancement - zeus_battle.py's refresh_quota_snapshot() does the actual work.
# Fleet blast provides cool visual feedback on first launch or when showing user which regions are available.
# Both use same underlying API call (get_all_gpu_quotas_for_type) for consistency.
# </claudes_code_comments>

import time
import random
import sys
from typing import Dict, Optional, Callable
from ...shared.quota.gpu_quota import get_all_gpu_quotas_for_type
from .zeus_olympus import update_region_status
from .zeus_battle import get_spot_pricing, THUNDER_TIERS

# Center-based ASCII chars for arrival animation (25 total)
ARRIVAL_SYMBOLS = [
    '-', '~', '=', '+', '∿', '≈', '∼', '∽', '∾', '≋',
    '◦', '∘', '·', '•', '○', '◌', '◯', '∙', '⋅', '∶',
    '|', '¦', '⎢', '⎥', '┃'
]

# Cool ASCII chars for arrival flair (50 total) - geometric, lighting, energy
ARRIVAL_FLAIR = [
    '⚡', '✦', '✧', '✨', '★', '☆', '✴', '✵', '✶',
    '✷', '✸', '✹', '✺', '✻', '✼', '✽', '✾', '✿', '❀', '❁',
    '◆', '◇', '◈', '◉', '◊', '○', '◌', '◍', '◎', '●',
    '◐', '◑', '◒', '◓', '◔', '◕', '◖', '◗', '◘', '◙',
    '◚', '◛', '◜', '◝', '◞', '◟', '◠', '◡', '◢', '◣'
]

# Winding/dingbat-like symbols (25 total) - cryptic beacon hash characters
WINDING_SYMBOLS = [
    '▀', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█',
    '▌', '▐', '░', '▒', '▓', '├', '┤', '┬', '┴',
    '┼', '═', '║', '╔', '╚', '╬', '╳'
]


def animate_arrival():
    """
    Show 3-char arrival animation with 200ms between each char
    Using random center-based ASCII symbols
    Prepends to the arrival line (no newline after)
    """
    symbols = random.sample(ARRIVAL_SYMBOLS, 3)

    # Two spaces indent before animation
    sys.stdout.write('  ')
    sys.stdout.flush()

    # First char
    sys.stdout.write(symbols[0])
    sys.stdout.flush()
    time.sleep(0.2)

    # Second char
    sys.stdout.write(symbols[1])
    sys.stdout.flush()
    time.sleep(0.2)

    # Third char
    sys.stdout.write(symbols[2])
    sys.stdout.flush()
    time.sleep(0.2)

    # Space before arrival message (stays on same line)
    sys.stdout.write(' ')
    sys.stdout.flush()


def output_arrival(preline: str, message: str, print_fn: Optional[Callable] = None):
    """
    Output thunder region arrival message with timing/animation

    Args:
        preline: Pre-line symbols to prepend (for TUI compatibility)
        message: Full arrival message to display
        print_fn: Print function (if provided, skip animation)
    """
    if print_fn:
        # TUI or status callback: Dump full line immediately
        print_fn("  " + preline + message)
    else:
        # CLI: Full cinematic experience with animation and timing
        animate_arrival()
        print(message)
        time.sleep(random.uniform(0.5, 0.9))


def announce_thunder_ready(region: str, quota: int, gpu_type: str, print_fn: Optional[Callable] = None):
    """
    ⚡ Epic Thunder arrival announcement!

    Args:
        region: GCP region
        quota: Number of GPUs available
        gpu_type: GPU type (for display)
        print_fn: Optional print function
    """
    # Thunder-themed announcements (generic for all regions)
    announcements = [
        f"⚡ {region.upper()} THUNDER READY! ({quota} GPUs available)",
        f"⚡ {region.upper()} CHANNELS DIVINE POWER! ({quota} GPUs ready)",
        f"⚡ {region.upper()} JOINS OLYMPUS! ({quota} GPUs granted)",
        f"⚡ {region.upper()} LIGHTNING STRIKES! ({quota} GPUs unlocked)",
        f"⚡ {region.upper()} ZEUS'S BLESSING! ({quota} GPUs active)",
        f"⚡ {region.upper()} THUNDER AWAKENS! ({quota} GPUs online)",
        f"⚡ {region.upper()} BOLT DESCENDS! ({quota} GPUs charged)",
        f"⚡ {region.upper()} STORM ARRIVES! ({quota} GPUs ready)",
        f"⚡ {region.upper()} DIVINE FAVOR! ({quota} GPUs bestowed)",
        f"⚡ {region.upper()} OLYMPIAN MIGHT! ({quota} GPUs deployed)",
    ]

    message = random.choice(announcements)
    preline = "⚡⚡⚡ "
    output_arrival(preline, message, print_fn)


def thunder_fleet_blast(
    registry: Dict,
    tier: str,
    gpu_count: int,
    project_id: str,
    pricing_data: Optional[Dict],
    print_fn: Callable
) -> Dict:
    """
    ⚡⚡⚡ THUNDER FLEET BLAST ⚡⚡⚡

    Check GPU quota for ALL regions in ONE API call!
    Update Olympus registry with thunder-ready vs quest-locked status.

    Args:
        registry: Olympus registry
        tier: Thunder tier (e.g., "tempest")
        gpu_count: Number of GPUs needed
        project_id: GCP project ID
        pricing_data: Pricing data (optional)
        print_fn: Print function

    Returns:
        Updated registry
    """
    gpu_type = THUNDER_TIERS[tier]["gpu_types"][0]

    print_fn("")
    print_fn("   ⚡⚡⚡ THUNDER FLEET BLAST! ⚡⚡⚡")
    print_fn(f"   Tier: {tier.upper()} ({gpu_type})")
    print_fn(f"   Checking ALL regions for quota...")
    print_fn("")

    # ONE API CALL - Get ALL regions at once!
    quotas = get_all_gpu_quotas_for_type(project_id, gpu_type, use_spot=True)

    if not quotas:
        print_fn("   ⚠️  No quota data returned from API")
        return registry

    thunder_ready = []
    quest_locked = []

    # Process each region
    for region, quota_limit in quotas.items():
        if quota_limit >= gpu_count:
            # Region has quota!
            spot_price = get_spot_pricing(region, tier, gpu_count, pricing_data)

            registry = update_region_status(
                registry, tier, region,
                status="THUNDER_READY",
                gpu_type=gpu_type,
                quota_gpus=quota_limit,
                spot_price=spot_price
            )

            thunder_ready.append(region)
            announce_thunder_ready(region, quota_limit, gpu_type, print_fn)

        else:
            # No quota
            registry = update_region_status(
                registry, tier, region,
                status="QUEST_LOCKED",
                gpu_type=gpu_type,
                quota_gpus=quota_limit,
                spot_price=0.0
            )
            quest_locked.append(region)

    # Summary
    print_fn("")
    print_fn(f"   ⚡ Thunder Fleet Blast Complete!")
    print_fn(f"   ⚡ {len(thunder_ready)}/{len(quotas)} regions thunder-ready")

    if quest_locked:
        print_fn(f"   🚫 {len(quest_locked)} regions quest-locked (no quota)")

    print_fn("")

    return registry
