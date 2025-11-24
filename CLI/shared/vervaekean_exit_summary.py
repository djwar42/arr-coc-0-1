#!/usr/bin/env python3
"""
Simple Exit Summary - Timing and Stats with Vervaekean Framing

Shows what happened in the session with focus on timing and performance.
"""

from typing import Any, Dict


def generate_exit_summary(session_stats: Dict[str, Any]) -> str:
    """
    Generate simple exit summary focusing on timing and stats.

    Args:
        session_stats: Dictionary with:
            - session_duration: float (seconds)
            - builds_completed: int
            - builds_skipped: int
            - total_builds: int
            - hash_detection_working: bool
            - errors: int
            - build_times: list of (name, duration)
            - avg_build_time: float

    Returns:
        Formatted exit summary string
    """

    duration = session_stats.get("session_duration", 0.0)
    builds_done = session_stats.get("builds_completed", 0)
    builds_skip = session_stats.get("builds_skipped", 0)
    total = session_stats.get("total_builds", 0)
    hash_working = session_stats.get("hash_detection_working", False)
    errors = session_stats.get("errors", 0)
    build_times = session_stats.get("build_times", [])
    avg_time = session_stats.get("avg_build_time", 0.0)

    # Calculate quality
    if builds_skip == total and errors == 0:
        quality = "HIGH"
        emoji = "✨"
    elif builds_done < total // 2 and errors < 3:
        quality = "MEDIUM"
        emoji = "◇"
    else:
        quality = "LOW"
        emoji = "⚠️"

    lines = []

    # Header
    lines.append("")
    lines.append("╔" + "═" * 78)
    lines.append("║")
    lines.append("║  🔄 ARR-COC Training TUI - Session Summary")
    lines.append("║  " + f"{emoji} Relevance Realization Quality: {quality}")
    lines.append("║")
    lines.append("╚" + "═" * 78)
    lines.append("")

    # Session timing
    lines.append("⏱️  SESSION TIMING")
    lines.append("─" * 80)
    lines.append(f"Duration: {duration:.1f}s ({duration / 60:.1f} min)")
    if build_times:
        lines.append("")
        lines.append("Build Times:")
        for name, time in build_times:
            lines.append(f"  • {name}: {time:.1f}s ({time / 60:.1f} min)")
        if len(build_times) > 1:
            lines.append(f"  • Average: {avg_time:.1f}s")
    lines.append("")

    # Build stats
    lines.append("📊 BUILD STATISTICS")
    lines.append("─" * 80)
    lines.append(f"Total builds: {total}")
    lines.append(f"  • Completed: {builds_done}")
    lines.append(
        f"  • Skipped: {builds_skip} {'✓ (hash detection working)' if hash_working else ''}"
    )

    if total > 0:
        skip_rate = (builds_skip / total) * 100
        lines.append(f"  • Skip rate: {skip_rate:.0f}%")
    lines.append("")

    # Errors
    if errors > 0:
        lines.append(f"❌ Errors encountered: {errors}")
        lines.append("")

    # What this means (Vervaekean framing)
    lines.append("🎯 WHAT HAPPENED")
    lines.append("─" * 80)

    if quality == "HIGH":
        lines.append("✓ TUI attended to signal, ignored noise")
        lines.append(f"  • Hash detection prevented {builds_skip} redundant rebuild(s)")
        lines.append(f"  • Session completed in {duration:.0f}s (fast!)")
        lines.append("  • No errors - system realized what mattered")
    elif quality == "MEDIUM":
        lines.append("◇ TUI improving - some signal detected")
        if hash_working:
            lines.append("  • Hash detection working")
        if builds_done > 0:
            lines.append(f"  • {builds_done} build(s) completed successfully")
        if errors > 0:
            lines.append(f"  • {errors} error(s) to address")
    else:
        lines.append("⚠️  TUI attended to noise - opportunity for improvement")
        if builds_done == total and total > 0:
            lines.append(f"  • All {total} build(s) ran (should have skipped some)")
        if not hash_working:
            lines.append("  • Hash detection not working properly")
        if errors > 0:
            lines.append(f"  • {errors} error(s) encountered")

    lines.append("")

    # Footer
    lines.append("╔" + "═" * 78)
    lines.append("║")
    lines.append("║  Realizing relevance = attending to what matters. ◇◇◇")
    lines.append("║")
    lines.append("╚" + "═" * 78)
    lines.append("")

    return "\n".join(lines)


# Test
if __name__ == "__main__":
    # Simulate HIGH quality session
    stats = {
        "session_duration": 8.5,
        "builds_completed": 0,
        "builds_skipped": 3,
        "total_builds": 3,
        "hash_detection_working": True,
        "errors": 0,
        "build_times": [],
        "avg_build_time": 0.0,
    }

    print("=== HIGH QUALITY SESSION ===")
    print(generate_exit_summary(stats))

    # Simulate LOW quality session
    stats_low = {
        "session_duration": 4500.0,
        "builds_completed": 5,
        "builds_skipped": 0,
        "total_builds": 5,
        "hash_detection_working": False,
        "errors": 12,
        "build_times": [
            ("base", 900.0),
            ("base", 920.0),
            ("base", 895.0),
            ("base", 910.0),
            ("base", 875.0),
        ],
        "avg_build_time": 900.0,
    }

    print("\n=== LOW QUALITY SESSION ===")
    print(generate_exit_summary(stats_low))
