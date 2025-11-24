"""
🎮 EPIC MECHA PRICE BATTLE - Comic Book Strip Style!

Multi-phase battle system with:
1. Pre-Contender Introduction (baseline MECHA)
2. Sizing Up Phase (3 challengers approach)
3. Battle Phase (3 rounds of price comparisons)
4. Victory Phase (champion emerges with savings reveal)

All output uses comic book horizontal positioning for visual drama!
"""

import random
import time
import json
import re
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict


def random_battle_decoration() -> str:
    """
    Return random battle decoration - either emoji or ASCII.
    40% chance ASCII, 60% chance emoji.
    """
    # Battle-themed emojis
    battle_emojis = [
        "⚔️", "💥", "🔥", "⚡", "💪", "🎯", "✨", "🌟", "💎",
        "👊", "🏆", "⭐", "🔱", "💫", "🌪️", "🚀", "⚡", "✨"
    ]

    # Battle-themed ASCII
    battle_ascii = [
        "◆", "▸", "◇", "∿", "※", "★", "☆", "◈", "◉", "●",
        "○", "▲", "►", "◀", "▼", "◊", "⬡", "⬢", "⬣"
    ]

    # 40% ASCII, 60% emoji
    if random.random() < 0.4:
        return random.choice(battle_ascii)
    else:
        return random.choice(battle_emojis)


class ComicBookPrinter:
    """Prints text with smooth horizontal position changes like comic book panels"""

    def __init__(self, status_callback=None):
        self.current_indent = 2  # Start at middle level
        self.indent_levels = [3, 8, 13, 18, 23, 28]  # Minimum 3 spaces from left edge
        self.status_callback = status_callback

    def print_panel(self, text: str, pause: float = 0.05):
        """
        Print text with smooth horizontal movement.

        Position changes smoothly (+/- 1 level max) to create
        comic book strip effect going side to side.

        Replaces leading emoji/decoration with random battle decoration.
        """
        # Strip leading emoji/special symbols and replace with random decoration
        # Pattern matches: emoji, special symbols at start of line
        pattern = r'^[\s]*[^\w\s"\'-]+[\s]*'
        match = re.match(pattern, text)

        if match:
            # Found leading decoration - strip it and add random one
            rest_of_text = text[match.end():]
            decoration = random_battle_decoration()
            decorated_text = f"{decoration} {rest_of_text}"
        else:
            # No leading decoration - add random one
            decoration = random_battle_decoration()
            decorated_text = f"{decoration} {text.lstrip()}"

        # Smooth transition - can only move +/- 1 level or stay
        possible_moves = []

        if self.current_indent > 0:
            possible_moves.append(self.current_indent - 1)

        possible_moves.append(self.current_indent)

        if self.current_indent < len(self.indent_levels) - 1:
            possible_moves.append(self.current_indent + 1)

        self.current_indent = random.choice(possible_moves)
        indent_spaces = self.indent_levels[self.current_indent]

        output_text = " " * indent_spaces + decorated_text
        if self.status_callback:
            self.status_callback(output_text)
        else:
            print(output_text)
        time.sleep(pause)


# Sizing Up Phrases (Pre-Battle)
SIZING_UP_PHRASES = {
    "very_expensive": [
        "💎 {region} |${price:.2f}/hr| approaches... PREMIUM-CLASS armor detected!",
        "👑 {region} |${price:.2f}/hr| enters... ROYAL pricing!",
        "⭐ {region} |${price:.2f}/hr| steps forward... LUXURY-TIER detected!",
        "🏰 {region} |${price:.2f}/hr| appears... FORTRESS-GRADE costs!",
        "💠 {region} |${price:.2f}/hr| arrives... ELITE-CLASS expenses confirmed!",
        "🔱 {region} |${price:.2f}/hr| joins... SUPREME-CLASS pricing!",
        "✨ {region} |${price:.2f}/hr| enters... PLATINUM-GRADE costs detected!",
        "🎩 {region} |${price:.2f}/hr| approaches... HIGH-SOCIETY pricing!",
        "💫 {region} |${price:.2f}/hr| appears... PRESTIGE-CLASS unit!",
        "🌟 {region} |${price:.2f}/hr| steps forward... DELUXE-TIER confirmed!",
        "🏛️ {region} |${price:.2f}/hr| arrives... IMPERIAL pricing detected!",
        "💰 {region} |${price:.2f}/hr| enters... EXECUTIVE-CLASS costs!",
    ],
    "expensive": [
        "🔷 {region} |${price:.2f}/hr| joins... Above-average pricing detected.",
        "📈 {region} |${price:.2f}/hr| enters... Higher cost bracket identified.",
        "💰 {region} |${price:.2f}/hr| approaches... Premium unit confirmed.",
        "🔶 {region} |${price:.2f}/hr| arrives... Elevated costs detected!",
        "📊 {region} |${price:.2f}/hr| steps forward... Upper-tier pricing!",
        "💼 {region} |${price:.2f}/hr| appears... Professional-grade expenses!",
    ],
    "same_price": [
        "⚖️ {region} |${price:.2f}/hr| approaches... EQUAL POWER detected!",
        "🤝 {region} |${price:.2f}/hr| enters... MATCHED pricing!",
        "◇ {region} |${price:.2f}/hr| steps forward... BALANCED competitor!",
        "🔄 {region} |${price:.2f}/hr| joins... EQUIVALENT class confirmed!",
        "⚡ {region} |${price:.2f}/hr| arrives... PARALLEL pricing detected!",
        "🎯 {region} |${price:.2f}/hr| appears... IDENTICAL costs!",
        "✨ {region} |${price:.2f}/hr| steps in... MIRROR-MATCHED unit!",
        "◆ {region} |${price:.2f}/hr| enters... UNIFORM pricing confirmed!",
    ],
    "cheap": [
        "💚 {region} |${price:.2f}/hr| enters... BUDGET-FRIENDLY detected!",
        "📉 {region} |${price:.2f}/hr| approaches... Lower cost bracket!",
        "✅ {region} |${price:.2f}/hr| steps forward... ECONOMICAL unit!",
        "🌿 {region} |${price:.2f}/hr| arrives... VALUE-PRICED confirmed!",
        "💡 {region} |${price:.2f}/hr| joins... COST-EFFECTIVE option!",
        "🎈 {region} |${price:.2f}/hr| appears... THRIFTY pricing detected!",
    ],
    "very_cheap": [
        "⚡ {region} |${price:.2f}/hr| bursts in... BARGAIN-CLASS pricing!",
        "🎯 {region} |${price:.2f}/hr| charges forward... ULTRA-VALUE!",
        "💥 {region} |${price:.2f}/hr| appears... DISCOUNT-TIER powerhouse!",
        "🌟 {region} |${price:.2f}/hr| enters... LEGENDARY low price!",
        "🚀 {region} |${price:.2f}/hr| blazes in... STEAL-DEAL detected!",
        "💫 {region} |${price:.2f}/hr| arrives... CHAMPION-VALUE pricing!",
        "🎁 {region} |${price:.2f}/hr| steps forward... GIFT-TIER costs!",
        "⭐ {region} |${price:.2f}/hr| charges... ROCK-BOTTOM excellence!",
    ],
}


# Battle Round Phrases
BATTLE_ROUND_PHRASES = {
    "much_cheaper": [
        "⚡ {winner} SLASHES through {loser}'s defense! |${diff:.2f} advantage!|",
        "💥 {winner} deals CRITICAL DAMAGE! {loser} reels back! |${diff:.2f} gap!|",
        "🎯 {winner} strikes with VALUE-BEAM! {loser} staggers! |${diff:.2f} saved!|",
    ],
    "cheaper": [
        "🔹 {winner} lands a solid hit! {loser} takes damage! |${diff:.2f} edge|",
        "⚔️ {winner} gains advantage! {loser} defends! |${diff:.2f} cheaper|",
        "✨ {winner} outmaneuvers {loser}! |${diff:.2f} better!|",
    ],
    "close": [
        "⚖️ {mecha1} and {mecha2} clash evenly! |${diff:.2f} difference|",
        "🤝 {mecha1} vs {mecha2} - MATCHED POWER! Nearly equal pricing!",
        "◇ {mecha1} and {mecha2} trade blows! Tight competition!",
    ],
}


# Victory Phrases (Champion Emerges)
VICTORY_EMERGENCE_PHRASES = [
    "🏆 THE CHAMPION RISES FROM THE CHAOS!",
    "👑 A NEW KING CLAIMS THE THRONE!",
    "⭐ THE VICTOR EMERGES TRIUMPHANT!",
    "💎 THE ULTIMATE MECHA STANDS TALL!",
    "🌟 THE LEGENDARY CHAMPION REVEALED!",
]


# Victory Celebration Phrases (15 Epic Variations!)
VICTORY_CELEBRATION_PHRASES = [
    "⚡ \"This is the power of OPTIMAL PRICING!\"",
    "💪 \"No mecha can match my VALUE!\"",
    "🎯 \"I am the SAVINGS CHAMPION!\"",
    "✨ \"Behold the might of EFFICIENCY!\"",
    "🔥 \"My savings... are OVER 9000!\"",
    "👊 \"This is what PEAK PERFORMANCE looks like!\"",
    "💎 \"I am the LEGENDARY LOW-COST WARRIOR!\"",
    "🌪️ \"My VALUE TORNADO sweeps all competition!\"",
    "⚔️ \"None can withstand my EFFICIENCY STRIKE!\"",
    "🏆 \"The championship belt... is MINE!\"",
    "🌟 \"Witness the ULTIMATE ECONOMY MODE!\"",
    "💥 \"My PRICE-OPTIMIZATION POWER is unstoppable!\"",
    "🎮 \"MAXIMUM SAVINGS UNLOCKED! Game over!\"",
    "⚡ \"I've transcended... ULTRA EFFICIENCY FORM!\"",
    "🚀 \"This... is my FINAL VALUE BOOST!\"",
]


def categorize_price(price: float, baseline: float) -> str:
    """Categorize price relative to baseline"""
    diff = price - baseline
    percent_diff = (diff / baseline) * 100

    if percent_diff > 50:
        return "very_expensive"
    elif percent_diff > 15:
        return "expensive"
    elif abs(percent_diff) <= 15:
        return "same_price"
    elif percent_diff > -50:
        return "cheap"
    else:
        return "very_cheap"


def get_region_price(region: str, pricing_data: dict, machine_type: str = "c3-standard-176") -> float:
    """
    Get LIVE GCP pricing for any C3 machine type in specified region.

    Supports all C3 families:
    - c3-standard: 4GB RAM per vCPU (balanced, default)
    - c3-highcpu: 2GB RAM per vCPU (CPU-optimized, cheaper)
    - c3-highmem: 8GB RAM per vCPU (memory-optimized, expensive)

    Uses Cloud Billing Catalog API data from Artifact Registry.

    Args:
        region: GCP region (e.g., "us-central1")
        pricing_data: C3/GPU pricing dictionary from Artifact Registry
        machine_type: C3 machine type (e.g., "c3-standard-176", "c3-highcpu-88", "c3-highmem-44")

    Returns:
        Price per hour for the specified machine type (spot pricing)
    """
    # Use pricing data passed in (fetched from Artifact Registry)
    pricing = pricing_data

    # Extract vCPUs and calculate RAM based on C3 family
    vcpus = int(machine_type.split("-")[-1])

    if "c3-standard" in machine_type:
        ram_gb = vcpus * 4  # c3-standard: 4GB RAM per vCPU
    elif "c3-highcpu" in machine_type:
        ram_gb = vcpus * 2  # c3-highcpu: 2GB RAM per vCPU
    elif "c3-highmem" in machine_type:
        ram_gb = vcpus * 8  # c3-highmem: 8GB RAM per vCPU
    else:
        raise ValueError(f"Unsupported machine type: {machine_type}")

    # Get region pricing (spot prices - cheapest option for Cloud Build)
    from ...shared.pricing import get_spot_price

    region_pricing = pricing["c3_machines"].get(region, {})
    cpu_skus = region_pricing.get("cpu_per_core_spot", [])
    ram_skus = region_pricing.get("ram_per_gb_spot", [])

    cpu_per_core = get_spot_price(cpu_skus)
    ram_per_gb = get_spot_price(ram_skus)

    if cpu_per_core is None or ram_per_gb is None:
        raise ValueError(f"✗ No spot pricing data available for {region}")

    # Calculate total hourly price
    price = (vcpus * cpu_per_core) + (ram_gb * ram_per_gb)
    return price


def epic_mecha_price_battle(acquired_mechas: List[str], pricing_data: dict, status_callback=None, machine_type: str = "c3-standard-176") -> tuple:
    """
    🎮 EPIC MECHA PRICE BATTLE - Full Comic Book Experience!

    Uses LIVE GCP Cloud Billing API pricing (auto-updates weekly).

    Supports all C3 families:
    - c3-standard: 4GB RAM per vCPU (balanced, default)
    - c3-highcpu: 2GB RAM per vCPU (CPU-optimized, cheaper)
    - c3-highmem: 8GB RAM per vCPU (memory-optimized, expensive)

    Args:
        acquired_mechas: List of acquired MECHA regions
        status_callback: Optional callback for status updates (for TUI compatibility)
        machine_type: C3 machine type to use for pricing (default: "c3-standard-176")

    Returns:
        (champion_region, champion_price, compare_region, compare_price, savings)
    """

    printer = ComicBookPrinter(status_callback)

    # Helper for plain output (non-panel text)
    def output(msg=""):
        if status_callback:
            status_callback(msg)
        else:
            print(msg)

    # Pricing was already checked/refreshed in core.py (shown before MECHA GO!)
    # Get all prices (live GCP spot pricing) for the specified machine type
    prices = {mecha: get_region_price(mecha, pricing_data, machine_type) for mecha in acquired_mechas}

    # DEBUG: Show all region prices sorted (cheapest to most expensive) - HIDDEN
    # print("   ╔═══════════════════════════════════════════════════════════")
    # print("   ║ DEBUG: All Region Prices (Sorted Cheapest → Most Expensive)")
    # print("   ╠═══════════════════════════════════════════════════════════")
    # sorted_prices = sorted(prices.items(), key=lambda x: x[1])
    # for i, (region, price) in enumerate(sorted_prices, 1):
    #     print(f"   ║ {i:2d}. {region:30s} |${price:.2f}/hr|")
    # print("   ╚═══════════════════════════════════════════════════════════\n")

    # ========================================
    # BATTLE START MARKER
    # ========================================
    output("   ∿◇∿ MECHA PRICE BATTLE BEGINS ∿◇∿\n")

    # ========================================
    # CONDENSED EPIC BATTLE (1/3 length, same theme!)
    # ========================================

    baseline = "us-central1" if "us-central1" in acquired_mechas else "us-east1"
    if baseline not in acquired_mechas:
        baseline = acquired_mechas[0]
    baseline_price = prices[baseline]

    # Pre-contender
    printer.print_panel(f"⚔️  {baseline.upper()} sets the bar |${baseline_price:.2f}/hr| - \"Beat me if you can!\"")

    # 2 challengers approach
    other_mechas = [m for m in acquired_mechas if m != baseline]
    num_challengers = min(2, len(other_mechas))
    challengers = random.sample(other_mechas, num_challengers) if other_mechas else []

    last_phrase_template = None  # Track last phrase to avoid immediate repeats
    for mecha in challengers:
        price = prices[mecha]
        category = categorize_price(price, baseline_price)

        # Pick phrase, re-roll if same as last one
        phrase_template = random.choice(SIZING_UP_PHRASES[category])
        if phrase_template == last_phrase_template and len(SIZING_UP_PHRASES[category]) > 1:
            # Try again to get different phrase (max 3 attempts)
            for _ in range(3):
                phrase_template = random.choice(SIZING_UP_PHRASES[category])
                if phrase_template != last_phrase_template:
                    break

        phrase = phrase_template.format(region=mecha.upper(), price=price)
        last_phrase_template = phrase_template
        printer.print_panel(phrase)

    # 1 battle round
    if len(acquired_mechas) >= 2:
        pair = random.sample(acquired_mechas, 2)
        mecha1, mecha2 = pair
        price1, price2 = prices[mecha1], prices[mecha2]
        diff = abs(price1 - price2)

        if diff > 1.0:
            category = "much_cheaper"
        elif diff > 0.3:
            category = "cheaper"
        else:
            category = "close"

        if price1 < price2:
            winner, loser = mecha1, mecha2
        else:
            winner, loser = mecha2, mecha1

        if category == "close":
            phrase = random.choice(BATTLE_ROUND_PHRASES[category]).format(
                mecha1=mecha1.upper(), mecha2=mecha2.upper(), diff=diff
            )
        else:
            phrase = random.choice(BATTLE_ROUND_PHRASES[category]).format(
                winner=winner.upper(), loser=loser.upper(), diff=diff
            )
        printer.print_panel(phrase)

    # Victory! (With US region preference as tiebreaker)
    # Sort by: (price, not_us_region)
    # If all prices equal, US regions win!
    champion = min(acquired_mechas, key=lambda m: (prices[m], not m.startswith('us-')))
    champion_price = prices[champion]

    emergence_phrase = random.choice(VICTORY_EMERGENCE_PHRASES)
    printer.print_panel(emergence_phrase)

    # Always compare against the MOST EXPENSIVE region for maximum savings display
    other_regions = sorted(
        [m for m in acquired_mechas if m != champion],
        key=lambda m: prices[m],
        reverse=True  # Most expensive first
    )

    # Pick most expensive region (first in sorted list)
    if other_regions:
        compare_against = other_regions[0]
        compare_price = prices[compare_against]
    else:
        # Only one MECHA - compare against itself (no savings)
        compare_against = champion
        compare_price = champion_price

    savings = compare_price - champion_price
    savings_percent = (savings / compare_price * 100) if compare_price > 0 else 0

    printer.print_panel(f"🏆 {champion.upper()} |${champion_price:.2f}/hr| saves ${savings:.2f} ({savings_percent:.0f}%) vs {compare_against.upper()} |${compare_price:.2f}/hr|!")

    celebration = random.choice(VICTORY_CELEBRATION_PHRASES)
    printer.print_panel(f"🎙️ ⚡✨ {champion.upper()} |${champion_price:.2f}/hr| ✨⚡ {celebration}")
    output("")

    return champion, champion_price, compare_against, compare_price, savings
