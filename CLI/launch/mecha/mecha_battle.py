"""
🤖 MECHA Orchestrator (メカ) - Passive Collection + Full Wipe System

Core Logic:
1. Load registry at launch
2. Check CPU NUMBER changed → WIPE ALL MECHAS globally
3. Deploy PRIMARY region (for actual launch)
4. Passively deploy ONE missing MECHA (progressive collection)
5. Update registry after deployments
6. Show epic MECHA battle phrases

Strategy:
- Passive: Each launch attempts to collect one missing MECHA
- Persistent: Keep trying failed deployments
- Full Wipe: CPU change = DELETE ALL pools everywhere, restart collection
- Progressive: Gradually build up to Full MECHA Fleet (15 regions)
"""

import subprocess
import json
from typing import Dict, List, Optional, Tuple
from .mecha_hangar import (
    load_registry,
    save_registry,
    check_machine_type_changed,
    wipe_all_mechas,
    get_deployed_mechas,
    get_missing_mechas,
    update_mecha_status,
    get_mecha_fleet_status
)
from .mecha_regions import ALL_REGIONS
from .mecha_phrases import get_mecha_phrase


def check_pool_exists(region: str, pool_name: str, project_id: str) -> Tuple[bool, Optional[str]]:
    """
    Check if worker pool exists and get its machine type.

    Returns:
        (exists, machine_type)
    """
    try:
        result = subprocess.run(
            ["gcloud", "builds", "worker-pools", "describe", pool_name,
             "--region", region, f"--project={project_id}", "--format=json"],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            pool_info = json.loads(result.stdout)
            machine_type = pool_info.get("privatePoolV1Config", {}).get("workerConfig", {}).get("machineType", "")
            status = pool_info.get("state", "UNKNOWN")

            # Only consider RUNNING pools as "existing"
            if status == "OPERATIONAL":
                return (True, machine_type)

        return (False, None)
    except Exception:
        return (False, None)


def delete_pool(region: str, pool_name: str, project_id: str) -> bool:
    """
    Delete worker pool.

    Returns: True if successful
    """
    try:
        result = subprocess.run(
            ["gcloud", "builds", "worker-pools", "delete", pool_name,
             "--region", region, f"--project={project_id}", "--quiet"],
            capture_output=True,
            timeout=120
        )
        return result.returncode == 0
    except Exception:
        return False


def wipe_all_pools_globally(
    pool_name: str,
    project_id: str,
    all_regions: List[str],
    print_fn
) -> int:
    """
    WIPE ALL MECHAS GLOBALLY (CPU NUMBER changed).

    Returns: Number of pools deleted
    """
    print_fn("\n💥 CPU NUMBER CHANGED! WIPING ALL MECHAS GLOBALLY!")
    print_fn("=" * 60)

    deleted_count = 0

    for region in all_regions:
        exists, _ = check_pool_exists(region, pool_name, project_id)

        if exists:
            print_fn(f"   💣 Deleting {region} MECHA...")
            if delete_pool(region, pool_name, project_id):
                deleted_count += 1
                print_fn(f"      ✓ Deleted")
            else:
                print_fn(f"      ⚠️  Delete failed (will retry)")

    print_fn(f"\n💥 WIPE COMPLETE: Deleted {deleted_count} MECHAs")
    print_fn("   🔄 Starting fresh collection...")
    print_fn("=" * 60)

    return deleted_count


def passive_mecha_collection(
    registry: Dict,
    machine_type: str,
    pool_name: str,
    project_id: str,
    all_regions: List[str],
    primary_region: str,
    print_fn
) -> Tuple[Dict, int]:
    """
    Passively collect ONE missing MECHA (progressive deployment).

    Strategy:
    - Skip PRIMARY region (already deployed for launch)
    - Find next missing MECHA
    - Attempt deployment
    - Update registry
    - Return updated registry + success count

    Returns:
        (updated_registry, success_count)
    """
    # Get missing MECHAs
    missing = get_missing_mechas(registry, all_regions)

    # Remove PRIMARY region from missing (already deployed)
    missing = [r for r in missing if r != primary_region]

    if not missing:
        # FULL MECHA FLEET ACHIEVED!
        deployed, total, is_full = get_mecha_fleet_status(registry, len(all_regions))

        if is_full:
            print_fn(f"\n{get_mecha_phrase('full_fleet')}")
            print_fn(f"   🎊 All {total} MECHA regions acquired!")
            print_fn(f"   💰 Maximum price flexibility unlocked!\n")

        return (registry, 0)

    # Try to deploy ONE missing MECHA (passive collection)
    target_region = missing[0]  # Take first missing

    print_fn(f"\n🤖 PASSIVE MECHA COLLECTION")
    print_fn("=" * 60)

    deployed, total, _ = get_mecha_fleet_status(registry, len(all_regions))
    print_fn(f"   Current Fleet: {deployed}/{total} MECHAs")
    print_fn(f"   Missing: {len(missing)} regions")

    print_fn(f"\n{get_mecha_phrase('deploying', region=target_region)}")

    # Check if pool exists
    exists, current_machine = check_pool_exists(target_region, pool_name, project_id)

    if exists and current_machine == machine_type:
        # MECHA already exists with correct machine type!
        print_fn(f"   ✅ {target_region} MECHA already deployed!")

        update_mecha_status(registry, target_region, machine_type, "OPERATIONAL")
        save_registry(registry)

        return (registry, 1)

    elif exists and current_machine != machine_type:
        # Wrong machine type - delete it
        print_fn(f"   ⚠️  Wrong machine: {current_machine} → {machine_type}")
        print_fn(f"   💣 Deleting old MECHA...")

        delete_pool(target_region, pool_name, project_id)

    # Create the MECHA
    print_fn(f"   🚀 Deploying {target_region} MECHA ({machine_type})...")

    try:
        result = subprocess.run(
            ["gcloud", "builds", "worker-pools", "create", pool_name,
             "--region", target_region, f"--project={project_id}",
             f"--worker-machine-type={machine_type}",
             "--worker-disk-size=100"],
            capture_output=True,
            text=True,
            timeout=2700  # 45 min
        )

        if result.returncode == 0:
            # SUCCESS!
            print_fn(f"   {get_mecha_phrase('super_effective', region=target_region, savings='unknown')}")

            update_mecha_status(registry, target_region, machine_type, "OPERATIONAL")
            save_registry(registry)

            deployed, total, is_full = get_mecha_fleet_status(registry, len(all_regions))
            print_fn(f"   📊 Fleet Progress: {deployed}/{total} MECHAs\n")

            if is_full:
                print_fn(f"{get_mecha_phrase('full_fleet')}\n")

            return (registry, 1)
        else:
            # FAILED!
            print_fn(f"   ❌ Deployment failed: {result.stderr[:200]}")

            update_mecha_status(
                registry, target_region, machine_type, "FAILED",
                error_message=result.stderr[:500]
            )
            save_registry(registry)

            return (registry, 0)

    except Exception as e:
        print_fn(f"   ❌ Exception: {str(e)[:200]}")

        update_mecha_status(
            registry, target_region, machine_type, "FAILED",
            error_message=str(e)[:500]
        )
        save_registry(registry)

        return (registry, 0)


def orchestrate_mecha_system(
    machine_type: str,
    pool_name: str,
    project_id: str,
    primary_region: str,
    print_fn
) -> Tuple[Dict, bool]:
    """
    Main MECHA orchestration logic.

    1. Load registry
    2. Check CPU change → WIPE ALL if changed
    3. Return registry for launch to use
    4. (Passive collection happens after launch)

    Returns:
        (registry, cpu_changed)
    """
    print_fn("\n🤖 MECHA SYSTEM INITIALIZING...")
    print_fn("=" * 60)

    # Load registry
    registry = load_registry()

    # Check CPU NUMBER changed
    cpu_changed = check_machine_type_changed(registry, machine_type)

    if cpu_changed:
        # WIPE ALL POOLS GLOBALLY!
        old_machine = registry.get("machine_type", "unknown")
        print_fn(f"   💥 CPU CHANGE DETECTED: {old_machine} → {machine_type}")

        wipe_all_pools_globally(pool_name, project_id, ALL_REGIONS, print_fn)

        # Create fresh registry
        registry = wipe_all_mechas(registry, machine_type)
        save_registry(registry)

    else:
        # No CPU change - show current fleet status
        deployed, total, is_full = get_mecha_fleet_status(registry, len(ALL_REGIONS))

        print_fn(f"   🤖 MECHA Fleet: {deployed}/{total} deployed")
        print_fn(f"   💾 Machine Type: {machine_type}")

        if is_full:
            print_fn(f"   ✨ FULL MECHA FLEET ACTIVE!")

    print_fn("=" * 60)

    return (registry, cpu_changed)
