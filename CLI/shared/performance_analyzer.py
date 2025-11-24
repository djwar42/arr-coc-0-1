#!/usr/bin/env python3
"""
Performance Log Analyzer

Parses performance.json logs and generates human-readable reports showing:
- What operations are blocking the UI
- CPU hotspots and thread bottlenecks
- Category breakdown (API vs UI vs GCP vs Docker)
- Timeline visualization of operations

Usage:
    python CLI/shared/performance_analyzer.py
    python CLI/shared/performance_analyzer.py --timeline
    python CLI/shared/performance_analyzer.py --category API
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
from datetime import datetime


def parse_log_file(log_file: Path) -> List[Dict[str, Any]]:
    """Parse JSONL performance log file"""
    events = []
    with open(log_file) as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    return events


def analyze_blocking_operations(events: List[Dict]) -> Dict[str, Any]:
    """Find all blocking operations and their impact"""
    blocking = []
    main_thread_blocks = []
    worker_thread_slows = []

    for event in events:
        if event.get("type") == "blocking_warning":
            blocking.append(event)
            if event.get("main_thread"):
                main_thread_blocks.append(event)
            else:
                worker_thread_slows.append(event)

    return {
        "total_blocking": len(blocking),
        "main_thread_blocks": len(main_thread_blocks),
        "worker_thread_slows": len(worker_thread_slows),
        "blocking_events": blocking,
        "main_thread_events": main_thread_blocks,
        "worker_thread_events": worker_thread_slows
    }


def analyze_categories(events: List[Dict]) -> Dict[str, Any]:
    """Break down timing by category (UI, API, GCP, etc.)"""
    categories = defaultdict(lambda: {
        "total_time": 0.0,
        "count": 0,
        "operations": []
    })

    for event in events:
        if event.get("type") == "operation_end":
            cat = event.get("category", "General")
            duration = event.get("duration", 0.0)
            categories[cat]["total_time"] += duration
            categories[cat]["count"] += 1
            categories[cat]["operations"].append({
                "name": event.get("name"),
                "duration": duration,
                "blocking": event.get("blocking", False)
            })

    # Sort operations by duration within each category
    for cat_data in categories.values():
        cat_data["operations"].sort(key=lambda x: x["duration"], reverse=True)

    return dict(categories)


def analyze_threads(events: List[Dict]) -> Dict[str, Any]:
    """Break down timing by thread"""
    threads = defaultdict(lambda: {
        "total_time": 0.0,
        "count": 0,
        "operations": []
    })

    for event in events:
        if event.get("type") == "operation_end":
            thread_name = event.get("thread_name", "unknown")
            duration = event.get("duration", 0.0)
            threads[thread_name]["total_time"] += duration
            threads[thread_name]["count"] += 1
            threads[thread_name]["operations"].append({
                "name": event.get("name"),
                "duration": duration,
                "category": event.get("category")
            })

    # Sort operations by duration within each thread
    for thread_data in threads.values():
        thread_data["operations"].sort(key=lambda x: x["duration"], reverse=True)

    return dict(threads)


def generate_timeline(events: List[Dict], max_width: int = 80) -> str:
    """Generate ASCII timeline visualization of operations"""
    # Find operation pairs (start → end)
    operations = {}
    for event in events:
        if event.get("type") == "operation_start":
            operations[event["id"]] = {
                "name": event["name"],
                "category": event["category"],
                "start": datetime.fromisoformat(event["timestamp"]),
                "thread": event.get("thread_name", "unknown")
            }
        elif event.get("type") == "operation_end":
            op_id = event.get("id")
            if op_id in operations:
                operations[op_id]["end"] = datetime.fromisoformat(event["timestamp"])
                operations[op_id]["duration"] = event.get("duration", 0.0)
                operations[op_id]["blocking"] = event.get("blocking", False)

    # Filter complete operations
    complete_ops = [op for op in operations.values() if "end" in op]
    if not complete_ops:
        return "No complete operations found"

    # Sort by start time
    complete_ops.sort(key=lambda x: x["start"])

    # Find time range
    start_time = complete_ops[0]["start"]
    end_time = max(op["end"] for op in complete_ops)
    total_duration = (end_time - start_time).total_seconds()

    # Generate timeline
    lines = []
    lines.append("=" * max_width)
    lines.append(f"Timeline: {total_duration:.2f}s total")
    lines.append("=" * max_width)

    for op in complete_ops:
        # Calculate position and width
        start_offset = (op["start"] - start_time).total_seconds()
        start_pos = int((start_offset / total_duration) * max_width)
        duration = op["duration"]
        width = max(1, int((duration / total_duration) * max_width))

        # Truncate if exceeds max_width
        if start_pos + width > max_width:
            width = max_width - start_pos

        # Build bar
        bar = " " * start_pos + "█" * width

        # Color code by category (simple text marker)
        marker = "!" if op.get("blocking") else " "
        category_short = op["category"][:3].upper()

        lines.append(f"{bar} {marker}{category_short} {op['name'][:30]} ({duration:.2f}s)")

    lines.append("=" * max_width)
    lines.append("Legend: ! = Blocking operation")
    return "\n".join(lines)


def print_report(events: List[Dict], args):
    """Generate and print comprehensive performance report"""
    print("\n" + "=" * 80)
    print("PERFORMANCE ANALYSIS REPORT")
    print("=" * 80)
    print()

    # Summary statistics
    total_events = len(events)
    operation_ends = sum(1 for e in events if e.get("type") == "operation_end")
    print(f"Total Events: {total_events}")
    print(f"Completed Operations: {operation_ends}")
    print()

    # Blocking operations analysis
    print("-" * 80)
    print("BLOCKING OPERATIONS (UI Responsiveness Issues)")
    print("-" * 80)
    blocking_analysis = analyze_blocking_operations(events)
    print(f"Total Blocking Operations: {blocking_analysis['total_blocking']}")
    print(f"  • Main Thread Blocks (>100ms): {blocking_analysis['main_thread_blocks']} 🚨")
    print(f"  • Worker Thread Slows (>5s): {blocking_analysis['worker_thread_slows']}")
    print()

    if blocking_analysis['main_thread_events']:
        print("🚨 CRITICAL: Main Thread Blocking Operations")
        print("-" * 80)
        for event in blocking_analysis['main_thread_events'][:10]:
            print(f"  • {event['operation']:40} {event['duration']:8.2f}s  [{event['thread_name']}]")
        print()

    if blocking_analysis['worker_thread_events']:
        print("⚠️  Worker Thread Slow Operations")
        print("-" * 80)
        for event in blocking_analysis['worker_thread_events'][:10]:
            print(f"  • {event['operation']:40} {event['duration']:8.2f}s  [{event['thread_name']}]")
        print()

    # Category breakdown
    if args.category or not args.timeline:
        print("-" * 80)
        print("CATEGORY BREAKDOWN")
        print("-" * 80)
        categories = analyze_categories(events)

        # Sort by total time
        sorted_cats = sorted(
            categories.items(),
            key=lambda x: x[1]["total_time"],
            reverse=True
        )

        for cat_name, cat_data in sorted_cats:
            if args.category and cat_name != args.category:
                continue  # Skip if filtering by category

            print(f"\n{cat_name}")
            print(f"  Total Time: {cat_data['total_time']:.2f}s")
            print(f"  Operations: {cat_data['count']}")
            print(f"  Avg: {cat_data['total_time']/cat_data['count']:.3f}s")
            print()
            print("  Top 10 Slowest:")
            for op in cat_data["operations"][:10]:
                blocking_marker = "🚨" if op["blocking"] else "  "
                print(f"    {blocking_marker} {op['name']:35} {op['duration']:8.3f}s")
        print()

    # Thread breakdown
    if not args.category:
        print("-" * 80)
        print("THREAD BREAKDOWN")
        print("-" * 80)
        threads = analyze_threads(events)

        # Sort by total time
        sorted_threads = sorted(
            threads.items(),
            key=lambda x: x[1]["total_time"],
            reverse=True
        )

        for thread_name, thread_data in sorted_threads:
            print(f"\n{thread_name}")
            print(f"  Total Time: {thread_data['total_time']:.2f}s")
            print(f"  Operations: {thread_data['count']}")
            print()
            print("  Top 5 Operations:")
            for op in thread_data["operations"][:5]:
                print(f"    • {op['name']:35} {op['duration']:8.3f}s  [{op['category']}]")
        print()

    # Timeline visualization
    if args.timeline:
        print("-" * 80)
        print("TIMELINE VISUALIZATION")
        print("-" * 80)
        print(generate_timeline(events))
        print()

    # Recommendations
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    if blocking_analysis['main_thread_blocks'] > 0:
        print("🚨 CRITICAL: Main thread blocking detected!")
        print("   → Move long operations to worker threads (use Textual workers)")
        print("   → Use async/await for I/O operations")
        print("   → Cache expensive computations")
        print()

    categories = analyze_categories(events)
    if "API" in categories and categories["API"]["total_time"] > 5.0:
        print("⚠️  High API latency detected")
        print("   → Cache API responses with TTL")
        print("   → Use staggered refresh (already implemented in Monitor)")
        print("   → Consider local API proxy/cache")
        print()

    if "GCP" in categories and categories["GCP"]["total_time"] > 30.0:
        print("⚠️  GCP operations are slow")
        print("   → Use background workers for gcloud commands")
        print("   → Pre-fetch infrastructure status")
        print("   → Consider GCP API direct calls (faster than gcloud CLI)")
        print()

    if "Docker" in categories and categories["Docker"]["total_time"] > 60.0:
        print("⚠️  Docker operations are very slow")
        print("   → Use cached base images")
        print("   → Parallelize image builds where possible")
        print("   → Consider pre-built images in CI/CD")
        print()

    print("=" * 80)
    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze TUI performance logs")
    parser.add_argument(
        "--log-file",
        default="ARR_COC/Training/logs/performance.json",
        help="Path to performance log file (default: ARR_COC/Training/logs/performance.json)"
    )
    parser.add_argument(
        "--timeline",
        action="store_true",
        help="Show timeline visualization"
    )
    parser.add_argument(
        "--category",
        help="Filter by category (UI, API, GCP, Docker, Disk, Compute)"
    )
    args = parser.parse_args()

    log_file = Path(args.log_file)
    if not log_file.exists():
        print(f"❌ Log file not found: {log_file}")
        print()
        print("Run the TUI with performance monitoring enabled first:")
        print("  python CLI/tui.py")
        print()
        print("Then run this analyzer:")
        print("  python CLI/shared/performance_analyzer.py")
        return

    print(f"📊 Analyzing: {log_file}")
    events = parse_log_file(log_file)

    if not events:
        print("❌ No events found in log file")
        return

    print_report(events, args)


if __name__ == "__main__":
    main()
