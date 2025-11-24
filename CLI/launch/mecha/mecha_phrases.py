"""
Mecha Phrase System - Regional Worker Pool Battle Terminology

メカ (Mecha) - Mechanical compute units / Giant anime robots
  - To coders: Mechanical worker pools, robot deployments
  - To non-coders: Giant anime robots! 🤖

Perfect insider joke: We're deploying "mechanical workers"
but it sounds like anime robot battles! 😄

Usage:
    from .mecha_phrases import get_mecha_phrase

    # Get random phrase for scenario
    phrase = get_mecha_phrase("super_effective", region="europe-west1", savings="1.40")
    print(phrase)
    # Output: "💥 MECHA DEPLOYED! Using europe-west1 unit! Saved $1.40/hr!"
"""

import random
from typing import Dict, List

# Mecha phrase collections (10 variants each scenario)
MECHA_PHRASES: Dict[str, List[str]] = {
    # Scenario 1: Found cheaper regions
    "found_cheaper": [
        "🤖 Wild {region} MECHA spotted! It has PRICE-CUT capability!",
        "⚡ {region} MECHA deployed COST-SLASH! Super effective savings!",
        "❄️ {region} MECHA appeared! Armed with DISCOUNT-CANNON!",
        "🔥 Critical hit! {region} MECHA unleashes DEAL-BUSTER!",
        "♡⃤ {region} MECHA surfaced! Wave-motion savings detected!",
        "💎 Rare {region} MECHA found with legendary pricing!",
        "⚔️ {region} MECHA challenges with competitive arsenal!",
        "🎯 {region} MECHA deployed HYPER-VALUE missiles!",
        "✨ Legendary pricing MECHA appeared in {region}!",
        "🌪️ {region} MECHA unleashed SAVINGS-TYPHOON!",
    ],
    # Scenario 2: Checking mecha hangar
    "checking_hangar": [
        "🏭 Checking Mecha Hangar... Do we have this unit deployed?",
        "🎒 Opening Mecha Bay to check for regional units...",
        "💾 Accessing Mecha Storage... Searching for deployed robots...",
        '📋 Commander: "Let me check your Mecha Fleet!"',
        '🔍 Engineer: "Scanning hangar for regional mechas..."',
        "📚 Consulting Regional Mecha Database...",
        "🗂️ Checking Mecha Deployment System...",
        "📖 Opening Regional Robot Catalog...",
        "🎮 Accessing Mission Data... Loading mecha roster...",
        "🌐 Checking Cloud Mecha Registry...",
    ],
    # Scenario 3: Successfully using cheaper region (super effective!)
    "super_effective": [
        "💥 MECHA DEPLOYED! Using {region} unit! Saved ${savings}/hr!",
        "⚡ CRITICAL STRIKE! {region} MECHA deals massive value!",
        "🎯 PERFECT DEPLOYMENT! {region} MECHA optimized!",
        "✨ PREMIUM UNIT! {region} MECHA activated!",
        "🔥 HYPER MODE! {region} MECHA maximizes efficiency!",
        "💎 RARE MECHA! {region} unit unlocked!",
        "⚔️ DEVASTATING ATTACK! {region} MECHA deployed!",
        "♡⃤ TSUNAMI STRIKE! {region} MECHA washed costs away!",
        "❄️ FREEZE MODE! {region} MECHA locked optimal rate!",
        "🌟 LEGENDARY DEPLOYMENT! {region} MECHA online!",
    ],
    # Scenario 4: Don't have cheaper region
    "dont_have": [
        "❌ {region} MECHA... Not in your hangar!",
        "💨 {region} MECHA unavailable! Not deployed yet...",
        "🚫 {region} MECHA... Haven't built this unit yet!",
        "😔 Need REGIONAL-LICENSE for {region} MECHA!",
        "⛔ {region} MECHA locked! Deploy via Regional Blast!",
        "🔒 {region} MECHA... Hangar empty for this region!",
        "💭 {region} MECHA spotted... but not in inventory!",
        "🎒 Bay is empty! No {region} MECHA available!",
        "❌ {region} MECHA can't deploy! Not built yet...",
        "🚷 {region} MECHA blocked! Complete Regional Quest!",
    ],
    # Scenario 5: PRIMARY_REGION already cheapest
    "already_best": [
        "✅ {region} MECHA has POWER-ADVANTAGE! Best unit already!",
        "🛡️ Perfect armor! {region} MECHA resists competitors!",
        "👑 {region} is CHAMPION-CLASS MECHA! Unbeatable!",
        "⭐ {region} MECHA already MAXIMUM-POWER level!",
        "🏆 Prime unit! {region} MECHA holds championship!",
        "💯 {region} MECHA optimized! Nothing beats it!",
        "🥇 ELITE-CLASS MECHA! {region} tops rankings!",
        "🌟 {region} MECHA evolved to MEGA-MODE!",
        "👊 {region} MECHA too powerful! No challenger wins!",
        "💪 {region} MECHA maxed out! Ultimate configuration!",
    ],
    # Scenario 6: Big savings (>$1.00/hr)
    "big_savings": [
        "💥 CRITICAL STRIKE! ${savings}/hr saved! ({percent}% off)",
        "⚡ SUPER-COMBO ATTACK! Save ${total_savings} on 4hr mission!",
        "🎯 QUADRUPLE-DAMAGE! Costs obliterated!",
        "✨ RARE {percent}% DISCOUNT! Ultra-power mecha found!",
        "🔥 MELTDOWN ATTACK! Prices reduced dramatically!",
        "❄️ FREEZE-RAY! Costs locked at minimum!",
        "💎 ULTRA-RARE POWER! LEGENDARY-class savings!",
        "⚔️ HYPER-CANNON! Maximum savings deployed!",
        "♡⃤ TSUNAMI CANNON! Expenses obliterated!",
        "🌪️ TORNADO STRIKE! Prices swept to minimum!",
    ],
    # Scenario 7: Small savings (<$1.00/hr)
    "small_savings": [
        "✓ Deployed! {region} MECHA saves ${savings}/hr.",
        "👍 Standard power. Modest ${savings}/hr saved.",
        "💚 Good unit! {region} MECHA reduces cost {percent}%.",
        "⚪ Baseline deployment. Saved ${total_savings} on mission.",
        "📊 Effective! {region} MECHA cuts {percent}% of cost.",
        "💵 Minor optimization! Every dollar counts!",
        "🎯 Launched! {region} MECHA chips away at price.",
        "✨ Small but steady! ${savings}/hr compounds!",
        "⭐ Consistent damage! Efficiency accumulates!",
        "💪 Solid unit! {region} MECHA reduces burden.",
    ],
    # Scenario 8: Regional blast creating pools
    "deploying": [
        "🚀 Launching {region} MECHA... Deployment successful!",
        "⚡ ULTRA-LAUNCH! {region} MECHA secured!",
        "💫 MASTER-CLASS launch! {region} MECHA online!",
        "🎯 PREMIUM deployment! {region}... MECHA ready!",
        "✨ CLOUD launch! {region} MECHA operational!",
        "🌟 PRIORITY deployment! {region} MECHA joins fleet!",
        "🎪 BATCH launch! {region} MECHA re-deployed!",
        "🎁 PREMIUM setup! {region} MECHA is ready!",
        "⚡ RAPID deployment! {region} MECHA online instantly!",
        "🏅 NETWORK launch! {region} MECHA added to hangar!",
    ],
    # Scenario 9: Full mecha fleet complete
    "full_fleet": [
        "🎊 FULL MECHA FLEET! All 18 regional mechas acquired!",
        "👑 COMMANDER STATUS! Complete Regional Mecha Fleet!",
        '🏆 PERFECT HANGAR! "Magnificent robot collection!"',
        "⭐⭐⭐ ELITE-RANK! Maximum deployment flexibility!",
        "💯 FLEET 100%! All regional mechas operational!",
        "🌍 GLOBAL COMMANDER! Worldwide mecha mastery!",
        "🎯 PERFECT DEPLOYMENT! All mecha variants acquired!",
        "🌟 LEGENDARY STATUS! Full mecha fleet online!",
        "👊 UNBEATABLE! All regional mechas acquired!",
        "💎 COMPLETE HANGAR! Rare full-fleet bonus active!",
    ],
}


def get_mecha_phrase(scenario: str, **kwargs) -> str:
    """
    Get random mecha phrase for scenario.

    Mecha (メカ): "Mechanical workers" to coders, "Giant robots" to everyone else!

    Args:
        scenario: One of the keys from MECHA_PHRASES:
            - found_cheaper: Discovered cheaper regional pricing
            - checking_hangar: Checking deployed mecha fleet
            - super_effective: Successfully using cheaper mecha
            - dont_have: Don't have this mecha deployed
            - already_best: PRIMARY_REGION is cheapest
            - big_savings: Large savings (>$1.00/hr)
            - small_savings: Small savings (<$1.00/hr)
            - deploying: Creating new worker pool
            - full_fleet: All 18 regions acquired
        **kwargs: Format variables (region, savings, percent, total_savings)

    Returns:
        Formatted mecha phrase (random selection from 10 variants)

    Examples:
        >>> get_mecha_phrase("super_effective", region="europe-west1", savings="1.40")
        "💥 MECHA DEPLOYED! Using europe-west1 unit! Saved $1.40/hr!"

        >>> get_mecha_phrase("full_fleet")
        "🎊 FULL MECHA FLEET! All 18 regional mechas acquired!"

        >>> get_mecha_phrase("dont_have", region="asia-northeast1")
        "❌ asia-northeast1 MECHA... Not in your hangar!"
    """
    phrases = MECHA_PHRASES.get(scenario, [])
    if not phrases:
        return f"⚠️ Unknown mecha scenario: {scenario}"

    phrase = random.choice(phrases)

    try:
        return phrase.format(**kwargs)
    except KeyError as e:
        return f"⚠️ Missing format variable for mecha phrase: {e}"


# Convenience function for testing
if __name__ == "__main__":
    print("🤖 MECHA PHRASE SYSTEM TEST\n")

    test_cases = [
        ("found_cheaper", {"region": "europe-west1"}),
        ("checking_hangar", {}),
        ("super_effective", {"region": "us-east4", "savings": "1.40"}),
        ("dont_have", {"region": "asia-northeast1"}),
        ("already_best", {"region": "us-central1"}),
        ("big_savings", {"savings": "1.60", "percent": "25", "total_savings": "6.40"}),
        (
            "small_savings",
            {
                "region": "europe-west2",
                "savings": "0.40",
                "percent": "8",
                "total_savings": "1.60",
            },
        ),
        ("deploying", {"region": "australia-southeast1"}),
        ("full_fleet", {}),
    ]

    for scenario, kwargs in test_cases:
        print(f"Scenario: {scenario}")
        print(f"  {get_mecha_phrase(scenario, **kwargs)}")
        print()
