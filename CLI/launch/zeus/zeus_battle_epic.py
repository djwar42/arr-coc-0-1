"""
⚡ EPIC ZEUS PRICE BATTLE - Divine Thunder Style!

Condensed battle system (matches MECHA format):
1. Baseline sets the bar (1 region declares challenge)
2. Challengers approach (2 regions sizing up)
3. Battle round (1 direct confrontation)
4. Victory emergence (champion crowned with savings)

All output uses mythological thunder/lightning theme for visual drama!
Uses DivineThunderPrinter for comic book panel indentation (smooth +/- 1 level transitions).
"""

# <claudes_code_comments>
# ** Function List **
# DivineThunderPrinter - Comic book panel printer (smooth horizontal transitions)
# DivineThunderPrinter.print_panel(text, pause) - Print with random decoration and smooth indent
# categorize_price(price, baseline) - Categorize price relative to baseline (very_cheap→very_expensive)
# random_battle_decoration() - Return random emoji/ASCII decoration (40% ASCII, 60% emoji)
# get_thunder_phrase(scenario, region, price, savings) - Get random divine phrase for price tier (LEGACY - still used by some flows)
# get_price_tier(price, min_price, max_price) - Categorize price into tier (very_cheap→very_expensive)
#
# ** Technical Review **
# Epic battle phrase system for Zeus GPU pricing battles. Mirrors MECHA's mecha_battle_epic.py structure.
# Uses condensed battle format: baseline → 2 challengers → 1 battle → victory.
#
# DivineThunderPrinter provides comic book panel indentation (matches MECHA's ComicBookPrinter):
# - Smooth +/- 1 level horizontal transitions
# - Randomly replaces leading emoji with battle decorations
# - Indent levels: [3, 8, 13, 18, 23, 28] spaces
#
# SIZING_UP_PHRASES categorize regions before battle (based on price vs baseline):
# - very_cheap: "sets the bar" phrases
# - cheap/competitive/expensive/very_expensive: Sizing up variations
#
# BATTLE_ROUND_PHRASES for direct confrontations:
# - much_cheaper (>$1.00 diff): "THUNDERS through defense!"
# - cheaper (>$0.30 diff): "OUTMANEUVERS!"
# - close (<$0.30 diff): "clash evenly!"
#
# VICTORY_EMERGENCE_PHRASES: Champion announcements
# VICTORY_DECLARATIONS: Final blessing phrases
# </claudes_code_comments>

import random
import re
import time
from typing import Dict, List


class DivineThunderPrinter:
    """Prints text with smooth horizontal position changes like comic book panels (Zeus theme!)"""

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
            rest_of_text = text[match.end() :]
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


# Sizing Up Phrases (Pre-Battle) - Zeus Thunder Theme
SIZING_UP_PHRASES = {
    "very_expensive": [
        "☁️ {region} |${price:.2f}/hr| approaches... OLYMPIAN premium pricing!",
        "👑 {region} |${price:.2f}/hr| enters... ROYAL divine tribute!",
        "⭐ {region} |${price:.2f}/hr| arrives... CELESTIAL-CLASS costs!",
        "🏛️ {region} |${price:.2f}/hr| appears... IMPERIAL thunder detected!",
        "💎 {region} |${price:.2f}/hr| joins... LEGENDARY pricing!",
        "🔱 {region} |${price:.2f}/hr| steps forward... SUPREME divine power costs!",
    ],
    "expensive": [
        "☁️ {region} |${price:.2f}/hr| joins... Above-average divine tribute.",
        "⚡ {region} |${price:.2f}/hr| enters... Higher thunder costs detected.",
        "💰 {region} |${price:.2f}/hr| approaches... Premium Zeus pricing.",
        "📈 {region} |${price:.2f}/hr| appears... Elevated divine rates.",
    ],
    "competitive": [
        "⚡ {region} |${price:.2f}/hr| summons lightning... BALANCED competitor!",
        "☁️ {region} |${price:.2f}/hr| channels thunder... Competitive pricing!",
        "⚔️ {region} |${price:.2f}/hr| enters battle... Fair divine tribute!",
        "🎯 {region} |${price:.2f}/hr| takes position... Standard Zeus blessing!",
    ],
    "cheap": [
        "⚡ {region} |${price:.2f}/hr| summons lightning... Zeus FAVORS this domain!",
        "✨ {region} |${price:.2f}/hr| glows golden... Divine DISCOUNT detected!",
        "🎁 {region} |${price:.2f}/hr| offers gift... BLESSED pricing!",
    ],
    "very_cheap": [
        '⚡ {region} sets the bar |${price:.2f}/hr| - "Challenge the gods if you dare!"',
    ],
}

# Battle Round Phrases (Direct Competition)
BATTLE_ROUND_PHRASES = {
    "much_cheaper": [
        "🔥 {winner} THUNDERS through {loser}'s defense! |${diff:.2f} advantage!|",
        "⚡ {winner} STRIKES {loser} with DIVINE POWER! |${diff:.2f} cheaper!|",
    ],
    "cheaper": [
        "💥 {winner} OUTMANEUVERS {loser}! |${diff:.2f} savings!|",
        "⚡ {winner} bests {loser} in pricing! |${diff:.2f} advantage!|",
    ],
    "close": [
        "⚔️ {mecha1} and {mecha2} clash evenly! |${diff:.2f} gap - FIERCE!|",
    ],
}

# Victory Emergence Phrases
VICTORY_EMERGENCE_PHRASES = [
    "⚔️ THE THUNDER CHAMPION DESCENDS FROM OLYMPUS!",
    "⚡ ZEUS'S CHOSEN REGION EMERGES VICTORIOUS!",
    "⚔️ THE DIVINE CHAMPION CLAIMS THE THRONE!",
]

# Victory Declaration Phrases
VICTORY_DECLARATIONS = [
    '⚡ "ZEUS\'S BLESSING BESTOWED! None can withstand my DIVINE EFFICIENCY!"',
    '⚡ "OLYMPIAN FAVOR GRANTED! Thunder optimized!"',
    '⚡ "DIVINE WISDOM PREVAILS! Perfect selection made!"',
]


def categorize_price(price: float, baseline: float) -> str:
    """Categorize price relative to baseline for phrase selection"""
    if price < baseline * 0.85:
        return "very_cheap"
    elif price < baseline * 0.95:
        return "cheap"
    elif price < baseline * 1.05:
        return "competitive"
    elif price < baseline * 1.15:
        return "expensive"
    else:
        return "very_expensive"


def random_battle_decoration() -> str:
    """
    Return random battle decoration - either emoji or ASCII.
    40% chance ASCII, 60% chance emoji.
    """
    # Thunder/lightning-themed emojis
    battle_emojis = [
        "⚡",
        "⚡",
        "⚡",  # More lightning for Zeus!
        "✨",
        "💫",
        "🌟",
        "⭐",
        "💎",
        "🏆",
        "👑",
        "🔱",
        "🌪️",
        "☁️",
    ]

    # Battle-themed ASCII
    battle_ascii = [
        "◆",
        "▸",
        "◇",
        "∿",
        "※",
        "★",
        "☆",
        "◈",
        "◉",
        "●",
        "○",
        "▲",
        "►",
        "◀",
        "▼",
        "◊",
        "⬡",
        "⬢",
        "⬣",
    ]

    # 40% ASCII, 60% emoji
    if random.random() < 0.4:
        return random.choice(battle_ascii)
    else:
        return random.choice(battle_emojis)


# Divine Thunder Battle Phrases (price tier based)
THUNDER_PHRASES: Dict[str, List[str]] = {
    # Very expensive regions (Zeus's premium domains)
    "very_expensive": [
        "☁️ {region} |${price:.2f}/hr| approaches... OLYMPIAN premium pricing!",
        "👑 {region} |${price:.2f}/hr| enters... ROYAL divine tribute detected!",
        "⭐ {region} |${price:.2f}/hr| arrives... CELESTIAL-CLASS costs!",
        "🏛️ {region} |${price:.2f}/hr| appears... IMPERIAL thunder detected!",
        "💎 {region} |${price:.2f}/hr| joins... LEGENDARY pricing - Zeus's finest!",
        "🔱 {region} |${price:.2f}/hr| steps forward... SUPREME divine power costs!",
        "✨ {region} |${price:.2f}/hr| enters... PLATINUM thunder confirmed!",
        "⚡ {region} |${price:.2f}/hr| approaches... ELITE divine favor pricing!",
        "💫 {region} |${price:.2f}/hr| appears... PRESTIGE-CLASS thunder!",
        "🌟 {region} |${price:.2f}/hr| arrives... DELUXE Olympian rates!",
    ],
    # Expensive regions
    "expensive": [
        "☁️ {region} |${price:.2f}/hr| joins... Above-average divine tribute.",
        "⚡ {region} |${price:.2f}/hr| enters... Higher thunder costs detected.",
        "💰 {region} |${price:.2f}/hr| approaches... Premium Zeus pricing.",
        "📈 {region} |${price:.2f}/hr| appears... Elevated divine rates.",
        "🔷 {region} |${price:.2f}/hr| arrives... Upper-tier Olympian pricing.",
        "💠 {region} |${price:.2f}/hr| steps forward... Enhanced thunder costs.",
        "⭕ {region} |${price:.2f}/hr| joins... High divine favor required.",
        "🌐 {region} |${price:.2f}/hr| enters... Above-median Zeus rates.",
    ],
    # Competitive regions (middle tier)
    "competitive": [
        "⚡ {region} |${price:.2f}/hr| summons lightning... Competitive divine rates!",
        "☁️ {region} |${price:.2f}/hr| channels thunder... Balanced Olympian pricing!",
        "⚔️ {region} |${price:.2f}/hr| enters battle... Fair divine tribute!",
        "🎯 {region} |${price:.2f}/hr| takes position... Standard Zeus blessing!",
        "💪 {region} |${price:.2f}/hr| flexes power... Median thunder costs!",
        "🔆 {region} |${price:.2f}/hr| shines bright... Mid-tier divine favor!",
        "⚖️ {region} |${price:.2f}/hr| maintains balance... Equilibrium pricing!",
        "♡⃤ {region} |${price:.2f}/hr| rides waves... Steady Olympian rates!",
    ],
    # Cheap regions (Zeus favors these!)
    "cheap": [
        "⚡ {region} |${price:.2f}/hr| summons lightning... Zeus FAVORS this domain!",
        "✨ {region} |${price:.2f}/hr| glows golden... Divine DISCOUNT detected!",
        "🎁 {region} |${price:.2f}/hr| offers gift... BLESSED pricing from Olympus!",
        "💝 {region} |${price:.2f}/hr| shows mercy... Zeus's GENEROSITY!",
        "🌟 {region} |${price:.2f}/hr| shines bright... FAVORABLE divine rates!",
        "💚 {region} |${price:.2f}/hr| radiates warmth... Zeus APPROVES!",
        "🍀 {region} |${price:.2f}/hr| brings fortune... LUCKY thunder pricing!",
        "⭐ {region} |${price:.2f}/hr| sparkles... PREFERRED Olympian domain!",
    ],
    # Very cheap regions (CHAMPION material!)
    "very_cheap": [
        "⚡⚡⚡ {region} |${price:.2f}/hr| CHANNELS PURE DIVINE POWER!",
        "✨✨✨ {region} |${price:.2f}/hr| RADIATES ZEUS'S BLESSING!",
        "🏆🏆🏆 {region} |${price:.2f}/hr| UNLEASHES CHAMPION THUNDER!",
        "💎💎💎 {region} |${price:.2f}/hr| REVEALS LEGENDARY PRICING!",
        "⚡⚡⚡ {region} |${price:.2f}/hr| SUMMONS OLYMPIAN MIGHT!",
        "🌟🌟🌟 {region} |${price:.2f}/hr| IGNITES DIVINE SUPREMACY!",
        "👑👑👑 {region} |${price:.2f}/hr| CLAIMS ROYAL THUNDER THRONE!",
        "💫💫💫 {region} |${price:.2f}/hr| MANIFESTS CELESTIAL POWER!",
    ],
    # Champion announcement (the winner!)
    "champion": [
        "● THE CHAMPION DESCENDS FROM OLYMPUS!",
        "● ZEUS'S CHOSEN REGION EMERGES VICTORIOUS!",
        "● DIVINE CHAMPION CLAIMS THE THRONE!",
        "● THE THUNDER BEARER RISES!",
        "● OLYMPUS CROWNS ITS CHAMPION!",
        "● THE LIGHTNING LORD REVEALS ITSELF!",
        "● ZEUS'S FAVOR SHINES BRIGHTEST HERE!",
        "● THE DIVINE VICTOR STANDS ALONE!",
    ],
    # Divine blessing (champion follow-up)
    "blessing": [
        '⚡ "ZEUS\'S BLESSING BESTOWED! Divine efficiency achieved!"',
        '⚡ "OLYMPIAN FAVOR GRANTED! Thunder optimized!"',
        '⚡ "DIVINE WISDOM REVEALED! Perfect selection made!"',
        '⚡ "ZEUS SMILES UPON THIS CHOICE! Maximum power!"',
        '⚡ "THUNDER PERFECTION ACHIEVED! Olympus approves!"',
        '⚡ "DIVINE CONVERGENCE COMPLETE! Optimal domain found!"',
        '⚡ "ZEUS\'S WILL MANIFESTS! Champion confirmed!"',
        '⚡ "OLYMPIAN EXCELLENCE ATTAINED! Victory secured!"',
    ],
}


def get_thunder_phrase(
    scenario: str, region: str = "", price: float = 0.0, savings: str = ""
) -> str:
    """
    Get random divine thunder phrase for scenario.

    Args:
        scenario: Phrase category (very_expensive, expensive, competitive,
                  cheap, very_cheap, champion, blessing)
        region: Region name (e.g., "us-east4")
        price: Price per hour (e.g., 16.40)
        savings: Savings string (e.g., "$6.00/hr")

    Returns:
        Formatted divine phrase
    """
    if scenario not in THUNDER_PHRASES:
        return f"⚡ {region.upper()} |${price:.2f}/hr|"

    phrases = THUNDER_PHRASES[scenario]
    phrase = random.choice(phrases)

    # Format with provided values
    if region and price:
        return phrase.format(region=region.upper(), price=price, savings=savings)
    else:
        return phrase


def get_price_tier(price: float, min_price: float, max_price: float) -> str:
    """
    Determine price tier for phrase selection.

    Returns:
        Tier name: very_cheap, cheap, competitive, expensive, very_expensive
    """
    price_range = max_price - min_price

    if price_range == 0:
        return "competitive"

    # Calculate percentile
    percentile = (price - min_price) / price_range

    if percentile <= 0.15:
        return "very_cheap"  # Bottom 15% (champion material!)
    elif percentile <= 0.35:
        return "cheap"  # 15-35%
    elif percentile <= 0.65:
        return "competitive"  # 35-65% (middle)
    elif percentile <= 0.85:
        return "expensive"  # 65-85%
    else:
        return "very_expensive"  # Top 15%
