#!/usr/bin/env python3
"""
spacecheck.py - Check HuggingFace Space runtime status via API

Usage:
    python spacecheck.py username/space-name
    python spacecheck.py username/space-name --log  # Also write to spacecheck.log

Exit codes:
    0: Space is RUNNING
    1: Space has RUNTIME_ERROR
    2: Space is BUILDING/PAUSED (pending)
"""
import sys
from datetime import datetime
from huggingface_hub import HfApi

def check_space_status(space_id, write_log=False):
    """Check Space runtime status and display any errors."""
    api = HfApi()

    print(f"🔍 Checking Space: {space_id}")
    print("=" * 60)

    try:
        runtime = api.get_space_runtime(space_id)

        # Display status
        print(f"Stage: {runtime.stage}")
        print(f"Hardware: {runtime.hardware or 'None (using requested)'}")
        print(f"Requested Hardware: {runtime.requested_hardware}")

        # Check for errors
        if runtime.stage == "RUNTIME_ERROR":
            print("\n❌ RUNTIME ERROR DETECTED!")
            print("-" * 60)

            error_msg = runtime.raw.get('errorMessage', 'No error message available')
            print(error_msg)
            print("-" * 60)

            # Write to log file if requested
            if write_log:
                with open('spacecheck.log', 'a') as f:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                    f.write(f"Space: {space_id}\n")
                    f.write(f"Stage: {runtime.stage}\n")
                    f.write(f"{'='*60}\n")
                    f.write(error_msg)
                    f.write('\n')
                print(f"📝 Error logged to: spacecheck.log")

            return False

        elif runtime.stage == "RUNNING":
            print("\n✅ Space is RUNNING")
            domains = runtime.raw.get('domains', [])
            if domains:
                print(f"URL: https://{domains[0]['domain']}")
            return True

        elif runtime.stage in ["BUILDING", "RUNNING_BUILDING", "RUNNING_APP_STARTING"]:
            print("\n🔨 Space is BUILDING...")
            return None  # Pending

        elif runtime.stage == "PAUSED":
            print("\n⏸️  Space is PAUSED")
            return None

        else:
            print(f"\n⚠️  Unknown stage: {runtime.stage}")
            return None

    except Exception as e:
        print(f"\n💥 Failed to check Space status: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python spacecheck.py username/space-name [--log]")
        sys.exit(2)

    space_id = sys.argv[1]
    write_log = '--log' in sys.argv

    status = check_space_status(space_id, write_log=write_log)

    # Exit codes for automation
    if status is True:
        sys.exit(0)  # Running
    elif status is False:
        sys.exit(1)  # Error
    else:
        sys.exit(2)  # Pending/Unknown
