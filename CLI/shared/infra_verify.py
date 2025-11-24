"""
Unified Infrastructure Verifier

ONE function to verify ALL infrastructure (GCP, W&B, HuggingFace, Local).

Used by:
- Setup screen (verify what exists)
- Teardown screen (check credentials)
- Infrastructure screen (display status)
- Launch core (boot checks)
- Monitor (display infra status)

Benefits:
- Distributed source of truth (imports from _MANIFEST consts)
- Consistent checks everywhere
- Billing checked FIRST (prevents cascade failures)
- Parallel execution with accumulators
- Local + cloud checks in one place

Source of Truth Architecture:
- MECHA_C3_MANIFEST → mecha_regions.py (18 C3 regions, worker pool config)
- GPU_TYPES_MANIFEST → gpu_quota.py (10 GPU types)
- gcp-manifest.json → apis, iam_roles, critical_items (manifest-specific)

CACHING NOTE (IMPORTANT!):
- GPU and C3 quota checks in THIS FILE use 30-minute in-memory cache
- This is for infra_verify.py ONLY (TUI display, status checks)
- Launch-time quota checks (launch/core.py) are ALWAYS FRESH - no cache!
- Cache is in-memory only, no files, clears on process restart
"""

from typing import Dict, Optional, Callable, Any
from pathlib import Path
import subprocess
import json
import time

# ═══════════════════════════════════════════════════════════════════════════════
# 🔥 STEVEN_INFRA_VERIFY_DEBUG Flag (2025-11-21):
# ═══════════════════════════════════════════════════════════════════════════════
# Toggle verbose timing logs for infra_verify quota checks.
# When True: Logs timing for GPU/C3 checks, cache hits/misses
# When False: Silent operation (production mode)
# Log file: ARR_COC/Training/logs/infra_verify_timing.log
STEVEN_INFRA_VERIFY_DEBUG = True  # 🔥 Turn ON to see timing!

# ═══════════════════════════════════════════════════════════════════════════════
# QUOTA CACHE (infra_verify ONLY - launch checks are ALWAYS fresh!)
# ═══════════════════════════════════════════════════════════════════════════════
# 30-minute TTL for GPU and C3 quota checks
# In-memory only - no file storage - clears on process restart
_QUOTA_CACHE_TTL_SECONDS = 30 * 60  # 30 minutes
_quota_cache: Dict[str, Any] = {}  # {"gpu": {...}, "c3": {...}, "gpu_ts": float, "c3_ts": float}

# Staggered warming state
_warming_state: Dict[str, Any] = {
    "gpu_idx": 0,      # Current GPU type index
    "gpu_region_idx": 0,  # Current GPU region index
    "c3_idx": 0,       # Current C3 region index
    "gpu_results": [],  # Accumulated GPU results
    "c3_results": {},   # Accumulated C3 results
    "is_warming": False,
}


def _get_cached_quotas(cache_key: str) -> Optional[Any]:
    """Get cached quota data if still valid (within TTL)"""
    ts_key = f"{cache_key}_ts"
    if cache_key in _quota_cache and ts_key in _quota_cache:
        age = time.time() - _quota_cache[ts_key]
        if age < _QUOTA_CACHE_TTL_SECONDS:
            return _quota_cache[cache_key]
    return None


def _set_cached_quotas(cache_key: str, data: Any) -> None:
    """Store quota data in cache with current timestamp"""
    _quota_cache[cache_key] = data
    _quota_cache[f"{cache_key}_ts"] = time.time()

    # 🦡🔍 DEBUG: Log cache write
    from CLI.shared.stevens_dance import stevens_log
    stevens_log("cache_write", f"🔍 CACHE_WRITE: key={cache_key}, data_size={len(str(data))} bytes, cache_keys={list(_quota_cache.keys())}")


def is_quota_cache_warm() -> bool:
    """Check if quota cache warming is COMPLETE (100% done)

    Returns True only when warming finished AND cache has data.
    This prevents:
    - Starting when is_warming=False but no data yet (initial state)
    - Stopping after first batch when partial data exists
    """
    # Must have BOTH caches with data AND warming must be complete
    gpu_cache = _get_cached_quotas("gpu")
    c3_cache = _get_cached_quotas("c3")
    has_data = gpu_cache is not None and c3_cache is not None
    warming_done = not _warming_state.get("is_warming", False)

    # Only warm if we have data AND warming is complete
    return has_data and warming_done


def warm_quota_cache_batch(project_id: str, batch_size: int = 8) -> Dict[str, Any]:
    """
    Warm ONE batch of quota cache (batch_size TOTAL checks split between GPU and C3).

    Default batch_size=8 means UP TO 8 checks total (split between GPU and C3).
    Example: 4 GPU + 4 C3 = 8 total, or 8 GPU + 0 C3 if C3 is done.
    Call this every 1 second from TUI set_interval until cache is warm.
    Returns: {"done": bool, "gpu_progress": "2/20", "c3_progress": "3/18", "elapsed_ms": int}

    NOTE: This is for TUI background warming ONLY.
    Launch-time checks are ALWAYS FRESH - no cache!
    """
    from .quota import get_vertex_gpu_quotas, get_cloud_build_c3_region_quota
    from ..launch.mecha.mecha_regions import MECHA_C3_MANIFEST

    global _warming_state

    # 🔥 STEVEN'S TIMING - Start measuring batch time!
    start_time = time.time()

    # GPU check config
    gpu_priority_order = [
        ("NVIDIA_TESLA_T4", "T4", "n1-standard-4"),
        ("NVIDIA_L4", "L4", "g2-standard-4"),
        ("NVIDIA_TESLA_A100", "A100", "a2-highgpu-1g"),
        ("NVIDIA_H100_80GB", "H100", "a3-highgpu-8g"),
        ("NVIDIA_H200", "H200", "a3-highgpu-8g"),
    ]
    gpu_regions = ["us-central1", "us-east1", "us-west1", "europe-west4"]
    total_gpu_checks = len(gpu_priority_order) * len(gpu_regions)  # 5 * 4 = 20

    # C3 check config
    c3_regions = [r["code"] for r in MECHA_C3_MANIFEST["regions"]]
    total_c3_checks = len(c3_regions)  # 18

    # Start warming if not already
    if not _warming_state["is_warming"]:
        _warming_state = {
            "gpu_idx": 0,
            "gpu_region_idx": 0,
            "c3_idx": 0,
            "gpu_results": [],
            "c3_results": {},
            "is_warming": True,
        }

    # Calculate current positions
    gpu_flat_idx = _warming_state["gpu_idx"] * len(gpu_regions) + _warming_state["gpu_region_idx"]
    gpu_done = gpu_flat_idx >= total_gpu_checks
    c3_done = _warming_state["c3_idx"] >= total_c3_checks

    # 🦡🎩 PARALLEL BATCH EXECUTION with GeneralAccumulator!
    # With batch_size=8: Launch UP TO 8 checks in PARALLEL (split between GPU and C3)

    from .api_helpers import GeneralAccumulator

    # Calculate split: 50/50 if both have work, otherwise give remaining to whichever still needs it
    gpu_remaining = total_gpu_checks - gpu_flat_idx
    c3_remaining = total_c3_checks - _warming_state["c3_idx"]

    if gpu_remaining > 0 and c3_remaining > 0:
        # Both need work - split 50/50
        gpu_batch = min(batch_size // 2, gpu_remaining)
        c3_batch = min(batch_size - gpu_batch, c3_remaining)  # Remainder goes to C3
    elif gpu_remaining > 0:
        # Only GPU needs work - use full batch
        gpu_batch = min(batch_size, gpu_remaining)
        c3_batch = 0
    elif c3_remaining > 0:
        # Only C3 needs work - use full batch
        gpu_batch = 0
        c3_batch = min(batch_size, c3_remaining)
    else:
        # Both done!
        gpu_batch = 0
        c3_batch = 0

    # Create accumulator and launch ALL checks in parallel!
    acc = GeneralAccumulator(max_workers=batch_size)

    # Launch GPU checks (in parallel!)
    gpu_checks = []  # Track metadata for results
    for i in range(gpu_batch):
        gpu_flat_idx = _warming_state["gpu_idx"] * len(gpu_regions) + _warming_state["gpu_region_idx"]
        if gpu_flat_idx >= total_gpu_checks or _warming_state["gpu_idx"] >= len(gpu_priority_order):
            break

        gpu_internal, gpu_display, machine_type = gpu_priority_order[_warming_state["gpu_idx"]]
        region = gpu_regions[_warming_state["gpu_region_idx"] % len(gpu_regions)]

        # Store metadata
        check_key = f"gpu_{i}"
        gpu_checks.append({
            "key": check_key,
            "gpu_display": gpu_display,
            "machine_type": machine_type,
            "region": region,
        })

        # Launch check in parallel!
        acc.start(check_key, lambda p=project_id, r=region, g=gpu_internal: get_vertex_gpu_quotas(p, r, g, use_spot=True))

        # Advance GPU position
        _warming_state["gpu_region_idx"] += 1
        if _warming_state["gpu_region_idx"] >= len(gpu_regions):
            _warming_state["gpu_region_idx"] = 0
            _warming_state["gpu_idx"] += 1

    # Launch C3 checks (in parallel!)
    c3_checks = []  # Track metadata for results
    for i in range(c3_batch):
        if _warming_state["c3_idx"] >= total_c3_checks or _warming_state["c3_idx"] >= len(c3_regions):
            break

        region = c3_regions[_warming_state["c3_idx"]]

        # Store metadata
        check_key = f"c3_{i}"
        c3_checks.append({
            "key": check_key,
            "region": region,
        })

        # Launch check in parallel!
        acc.start(check_key, lambda p=project_id, r=region: get_cloud_build_c3_region_quota(p, r))

        _warming_state["c3_idx"] += 1

    # ⏳ WAIT for ALL checks to complete (blocks until done!)
    results = acc.get_all()
    acc.shutdown()

    # Process GPU results
    for check in gpu_checks:
        try:
            quota_limit = results.get(check["key"], 0)
            if quota_limit and quota_limit > 0:
                _warming_state["gpu_results"].append({
                    "gpu": check["gpu_display"],
                    "machine_type": check["machine_type"],
                    "region": check["region"],
                    "quota": quota_limit
                })
        except Exception:
            pass

    # Process C3 results
    for check in c3_checks:
        try:
            quota_info = results.get(check["key"])
            if quota_info and quota_info.get("vcpus", 0) > 0:
                _warming_state["c3_results"][check["region"]] = quota_info
        except Exception:
            pass

    # Recalculate progress
    gpu_flat_idx = _warming_state["gpu_idx"] * len(gpu_regions) + _warming_state["gpu_region_idx"]
    gpu_done = gpu_flat_idx >= total_gpu_checks
    c3_done = _warming_state["c3_idx"] >= total_c3_checks
    all_done = gpu_done and c3_done

    # 🦡⚡ WRITE CACHE AFTER EVERY BATCH (not just when 100% done!)
    # This way partial cache is available even if warmup gets cancelled!
    vertex_gpu = {}
    for g in _warming_state["gpu_results"]:
        r = g["region"]
        if r not in vertex_gpu:
            vertex_gpu[r] = {"granted": [], "pending": []}
        vertex_gpu[r]["granted"].append(g)

    # Store in cache (ALWAYS, not just when all_done!)
    _set_cached_quotas("gpu", {"vertex_gpu": vertex_gpu, "all_gpu_found": _warming_state["gpu_results"]})
    _set_cached_quotas("c3", {"c3_build": _warming_state["c3_results"], "all_c3_found": [
        {"region": r, "vcpus": v.get("vcpus", 0), "machine": v.get("machine_type", "unknown")}
        for r, v in _warming_state["c3_results"].items()
    ]})

    # Mark warming done if both complete
    if all_done:
        _warming_state["is_warming"] = False

    # 🔥 STEVEN'S TIMING - Calculate elapsed time!
    elapsed_ms = int((time.time() - start_time) * 1000)

    return {
        "done": all_done,
        "gpu_progress": f"{min(gpu_flat_idx, total_gpu_checks)}/{total_gpu_checks}",
        "c3_progress": f"{min(_warming_state['c3_idx'], total_c3_checks)}/{total_c3_checks}",
        "elapsed_ms": elapsed_ms  # 🌶️ For Steven's complaint logs!
    }

# Import source of truth manifests
from ..launch.mecha.mecha_regions import MECHA_C3_MANIFEST
from .quota.gpu_quota import GPU_TYPES_MANIFEST

# Type aliases
StatusCallback = Callable[[str], None]


def _load_manifest() -> Dict:
    """Load GCP infrastructure manifest (canonical requirements)"""
    manifest_path = Path(__file__).parent.parent / "config" / "gcp-manifest.json"
    with open(manifest_path, "r") as f:
        return json.load(f)


def verify_all_infrastructure(
    helper,  # WandBHelper
    config: Dict[str, str],
    status: Optional[StatusCallback] = None,
    app = None,  # Textual app for toasts (deprecated - use callbacks!)
    gpu_progress_callback = None,  # 🦡⚡ Called when GPU quota check completes
    c3_progress_callback = None,   # 🦡⚡ Called when C3 quota check completes
) -> Dict:
    """
    Unified infrastructure verifier - verifies EVERYTHING

    Checks (in order):
    1. Billing status (FIRST - prevents cascade failures!)
    2. GCP resources (buckets, registries, service account)
    3. W&B resources (queue, project)
    4. HuggingFace repo
    5. Local credentials (key file)

    Args:
        helper: WandBHelper instance
        config: Training configuration dict
        status: Optional callback for status messages

    Returns:
        Comprehensive infrastructure dict:
        {
            "billing": {
                "enabled": bool | None,
                "error": str,
                "note": str
            },
            "gcp": {
                "buckets": {"count": int, "buckets": List[Dict]},
                "registry": {"exists": bool, "name": str},
                "persistent_registry": {"exists": bool, "name": str},
                "service_account": {"exists": bool, "email": str},
                "wandb_secret": {"exists": bool, "name": str}
            },
            "wandb": {
                "queue": {"exists": bool, "name": str},
                "project": {"exists": bool, "name": str}
            },
            "hf": {
                "repo": {"exists": bool, "id": str}
            },
            "local": {
                "key_file_exists": bool,
                "key_path": str
            }
        }
    """
    from .api_helpers import GCloudAccumulator, GeneralAccumulator
    from .stevens_dance import stevens_log

    stevens_log("infra", "🍞 BREADCRUMB 3.1: verify_all_infrastructure START")

    def _status(msg: str):
        """Helper to call status callback if provided"""
        if status:
            status(msg)

    # Load manifest (canonical infrastructure requirements)
    manifest = _load_manifest()
    stevens_log("infra", "🍞 BREADCRUMB 3.2: Manifest loaded")

    info = {
        "billing": {},
        "gcp": {},
        "wandb": {},
        "hf": {},
        "local": {}
    }

    # Extract config values
    project_id = config.get("GCP_PROJECT_ID", "")
    entity = config.get("WANDB_ENTITY", "")
    project_name = config.get("WANDB_PROJECT", "arr-coc-0-1")
    queue_name = config.get("WANDB_LAUNCH_QUEUE_NAME", "vertex-ai-queue")
    hf_repo = config.get("HF_HUB_REPO_ID", "")
    region = config.get("GCP_ROOT_RESOURCE_REGION", "us-central1")

    # ═══════════════════════════════════════════════════════════════════════
    # 🚨 STEP 1: CHECK BILLING FIRST (prevents cascade failures!)
    # ═══════════════════════════════════════════════════════════════════════

    stevens_log("infra", "🍞 BREADCRUMB 3.3: About to check billing")
    _status("Checking billing status...")

    billing_enabled = False
    billing_check_error = None

    try:
        result = subprocess.run(
            ["gcloud", "billing", "projects", "describe", project_id],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # Check if billing is enabled
            if "billingEnabled: true" in result.stdout:
                billing_enabled = True
                _status("  ✓ Billing enabled")
            elif "billingEnabled: false" in result.stdout:
                billing_enabled = False
                billing_check_error = "Billing is disabled"
                _status("  ✗ Billing disabled")
        else:
            # Command failed - might be permissions or project doesn't exist
            billing_check_error = f"Failed to check billing: {result.stderr.strip()}"
            _status(f"  ? {billing_check_error}")
    except Exception as e:
        # Can't check billing - proceed but note the error
        billing_check_error = f"Exception checking billing: {str(e)}"
        _status(f"  ? {billing_check_error}")

    info["billing"] = {
        "enabled": billing_enabled if billing_enabled is not None else None,
        "error": billing_check_error or "",
        "note": "⚠️ Enable billing to use GCP services" if billing_enabled is False else ""
    }

    stevens_log("infra", f"🍞 BREADCRUMB 3.4: Billing check done, enabled={billing_enabled}")

    # If billing is definitively disabled, HARD FAIL - return immediately!
    if billing_enabled is False:
        stevens_log("infra", "🚨 BREADCRUMB 3.4.1: Billing DISABLED - RETURNING EARLY!")
        _status("⚠️ Billing disabled - cannot check GCP infrastructure")
        info["gcp"] = {
            "buckets": {"count": 0, "buckets": [], "note": "Billing required"},
            "registry": {"exists": False, "name": "arr-coc-registry", "note": "Billing required"},
            "persistent_registry": {"exists": False, "name": "arr-coc-registry-persistent", "note": "Billing required"},
            "service_account": {"exists": False, "email": "", "note": "Billing required"}
        }
        info["wandb"] = {"queue": {"exists": False}, "project": {"exists": False}}
        info["hf"] = {"repo": {"exists": False}}
        info["local"] = {"key_file_exists": False}
        stevens_log("infra", "🚨 BREADCRUMB 3.4.2: Returning info with billing disabled")
        return info

    stevens_log("infra", "🍞 BREADCRUMB 3.4.3: Billing enabled, proceeding with GCP checks")

    # Billing is enabled, proceed with GCP checks
    if True:
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 2: CHECK GCP RESOURCES (parallel with GCloudAccumulator)
        # ═══════════════════════════════════════════════════════════════════════

        _status("Checking GCP resources...")

        gcloud_acc = GCloudAccumulator(max_workers=20)

        registry_name = "arr-coc-registry"
        persistent_registry_name = "arr-coc-registry-persistent"
        sa_name = "arr-coc-sa"
        sa_email = f"{sa_name}@{project_id}.iam.gserviceaccount.com"

        # Start all gcloud checks (non-blocking!)
        gcloud_acc.start(
            key="buckets",
            cmd=[
                "gcloud", "storage", "buckets", "list",
                f"--project={project_id}",
                "--format=csv[no-heading](name,location)",
            ],
            max_retries=3,
            timeout=10,
            operation_name="list GCS buckets",
        )

        gcloud_acc.start(
            key="registry",
            cmd=[
                "gcloud", "artifacts", "repositories", "describe",
                registry_name,
                "--location=us-central1",
                f"--project={project_id}",
            ],
            max_retries=3,
            timeout=10,
            operation_name="check Artifact Registry",
        )

        gcloud_acc.start(
            key="persistent_registry",
            cmd=[
                "gcloud", "artifacts", "repositories", "describe",
                persistent_registry_name,
                "--location=us-central1",
                f"--project={project_id}",
            ],
            max_retries=3,
            timeout=10,
            operation_name="check Persistent Artifact Registry",
        )

        gcloud_acc.start(
            key="sa",
            cmd=["gcloud", "iam", "service-accounts", "describe", sa_email],
            max_retries=3,
            timeout=10,
            operation_name="check Service Account",
        )

        gcloud_acc.start(
            key="wandb_secret",
            cmd=["gcloud", "secrets", "describe", "wandb-api-key"],
            max_retries=3,
            timeout=10,
            operation_name="check W&B API Key secret",
        )

        # Check GCP APIs (8 required APIs)
        gcloud_acc.start(
            key="apis",
            cmd=[
                "gcloud", "services", "list",
                "--enabled",
                f"--project={project_id}",
                "--format=value(config.name)"
            ],
            max_retries=3,
            timeout=10,
            operation_name="check enabled APIs",
        )

        # Check Cloud Build IAM (need project number first)
        gcloud_acc.start(
            key="project_number",
            cmd=[
                "gcloud", "projects", "describe", project_id,
                "--format=value(projectNumber)"
            ],
            max_retries=3,
            timeout=10,
            operation_name="get project number",
        )

        # Check VPC Peering (optional - for private worker pools)
        gcloud_acc.start(
            key="vpc_peering",
            cmd=[
                "gcloud", "compute", "networks", "peerings", "list",
                f"--project={project_id}",
                "--format=value(name)"
            ],
            max_retries=3,
            timeout=10,
            operation_name="check VPC peering",
        )

        # Get all results (blocks until ready)
        stevens_log("infra", "🍞 BREADCRUMB 3.5: About to call gcloud_acc.get_all() - THIS BLOCKS!")
        all_results = gcloud_acc.get_all()
        stevens_log("infra", "🍞 BREADCRUMB 3.6: gcloud_acc.get_all() returned!")
        gcloud_acc.shutdown()

        # Parse GCS buckets
        buckets_result = all_results.get("buckets")
        if buckets_result and buckets_result.returncode == 0:
            # Parse CSV output
            lines = [line.strip() for line in buckets_result.stdout.strip().split("\n") if line.strip()]
            arr_buckets = [
                {"name": parts[0], "location": parts[1]}
                for line in lines
                if (parts := line.split(",")) and "arr-coc-0-1" in parts[0]
            ]
            info["gcp"]["buckets"] = {
                "count": len(arr_buckets),
                "buckets": arr_buckets
            }
            _status(f"  ✓ Found {len(arr_buckets)} arr-coc-0-1 buckets")
        else:
            info["gcp"]["buckets"] = {"count": 0, "buckets": []}
            _status("  ○ No buckets found")

        # Parse Artifact Registries
        registry_result = all_results.get("registry")
        info["gcp"]["registry"] = {
            "exists": registry_result and registry_result.returncode == 0,
            "name": registry_name
        }
        _status(f"  {'✓' if info['gcp']['registry']['exists'] else '✗'} Registry: {registry_name}")

        persistent_registry_result = all_results.get("persistent_registry")
        info["gcp"]["persistent_registry"] = {
            "exists": persistent_registry_result and persistent_registry_result.returncode == 0,
            "name": persistent_registry_name
        }
        _status(f"  {'✓' if info['gcp']['persistent_registry']['exists'] else '✗'} Persistent Registry: {persistent_registry_name}")

        # Parse Service Account
        sa_result = all_results.get("sa")
        info["gcp"]["service_account"] = {
            "exists": sa_result and sa_result.returncode == 0,
            "email": sa_email if sa_result and sa_result.returncode == 0 else ""
        }
        _status(f"  {'✓' if info['gcp']['service_account']['exists'] else '✗'} Service Account: {sa_name}")

        # Parse W&B Secret
        secret_result = all_results.get("wandb_secret")
        info["gcp"]["wandb_secret"] = {
            "exists": secret_result and secret_result.returncode == 0,
            "name": "wandb-api-key"
        }
        _status(f"  {'✓' if info['gcp']['wandb_secret']['exists'] else '✗'} W&B Secret: wandb-api-key")

        # Parse APIs (from manifest)
        required_apis = [api["name"] for api in manifest["apis"]["required"]]

        apis_result = all_results.get("apis")
        if apis_result and apis_result.returncode == 0:
            enabled_apis = set(apis_result.stdout.strip().split("\n"))
            missing_apis = [api for api in required_apis if api not in enabled_apis]

            info["gcp"]["apis"] = {
                "all_enabled": len(missing_apis) == 0,
                "count": f"{len(required_apis) - len(missing_apis)}/{len(required_apis)}",
                "enabled": len(required_apis) - len(missing_apis),
                "total": len(required_apis),
                "missing": missing_apis
            }
            _status(f"  {'✓' if info['gcp']['apis']['all_enabled'] else '✗'} APIs: {info['gcp']['apis']['count']} enabled")
        else:
            info["gcp"]["apis"] = {
                "all_enabled": False,
                "count": "0/8",
                "enabled": 0,
                "total": 8,
                "missing": required_apis
            }
            _status("  ✗ APIs: Unable to check")

        # Parse Cloud Build IAM
        project_number_result = all_results.get("project_number")
        if project_number_result and project_number_result.returncode == 0:
            project_number = project_number_result.stdout.strip()
            cloudbuild_sa = f"{project_number}@cloudbuild.gserviceaccount.com"

            # Now check IAM policy for Cloud Build SA
            try:
                iam_result = subprocess.run(
                    [
                        "gcloud", "projects", "get-iam-policy", project_id,
                        "--flatten=bindings[].members",
                        f"--filter=bindings.members:serviceAccount:{cloudbuild_sa}",
                        "--format=value(bindings.role)"
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                if iam_result.returncode == 0:
                    granted_roles = set(iam_result.stdout.strip().split("\n"))
                    # Get required roles from manifest
                    required_roles = set([
                        role["role"]
                        for role in manifest["iam_roles"]["cloudbuild_service_account"]["required"]
                    ])
                    has_all_roles = required_roles.issubset(granted_roles)

                    info["gcp"]["cloudbuild_iam"] = {
                        "granted": has_all_roles,
                        "count": f"{len(required_roles & granted_roles)}/{len(required_roles)}",
                        "roles": list(granted_roles)
                    }
                    _status(f"  {'✓' if has_all_roles else '✗'} Cloud Build IAM: {info['gcp']['cloudbuild_iam']['count']} roles")
                else:
                    info["gcp"]["cloudbuild_iam"] = {
                        "granted": False,
                        "count": "0/2",
                        "roles": []
                    }
                    _status("  ✗ Cloud Build IAM: Unable to check")
            except Exception:
                info["gcp"]["cloudbuild_iam"] = {
                    "granted": False,
                    "count": "0/2",
                    "roles": []
                }
                _status("  ✗ Cloud Build IAM: Check failed")
        else:
            info["gcp"]["cloudbuild_iam"] = {
                "granted": False,
                "count": "0/2",
                "roles": []
            }
            _status("  ✗ Cloud Build IAM: Project number not found")

        # Parse VPC Peering (optional)
        vpc_peering_result = all_results.get("vpc_peering")
        if vpc_peering_result and vpc_peering_result.returncode == 0:
            peerings = [line.strip() for line in vpc_peering_result.stdout.strip().split("\n") if line.strip()]
            has_peering = len(peerings) > 0

            info["gcp"]["vpc_peering"] = {
                "exists": has_peering,
                "count": len(peerings),
                "names": peerings
            }
            if has_peering:
                _status(f"  ✓ VPC Peering: {len(peerings)} configured")
            else:
                _status("  ○ VPC Peering: Not configured (using public egress)")
        else:
            info["gcp"]["vpc_peering"] = {
                "exists": False,
                "count": 0,
                "names": []
            }
            _status("  ○ VPC Peering: Not configured")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3: CHECK W&B RESOURCES
    # ═══════════════════════════════════════════════════════════════════════

    stevens_log("infra", "🍞 BREADCRUMB 3.7: Starting W&B checks")
    _status("Checking W&B resources...")

    try:
        import wandb
        stevens_log("infra", "🍞 BREADCRUMB 3.7.1: About to call wandb.Api()")
        api = wandb.Api()
        stevens_log("infra", "🍞 BREADCRUMB 3.7.2: wandb.Api() created")

        # Check queue
        queue_exists = False
        try:
            stevens_log("infra", "🍞 BREADCRUMB 3.7.3: About to call api.run_queue() - NETWORK CALL!")
            queue = api.run_queue(entity, queue_name)
            queue_exists = queue is not None
            stevens_log("infra", f"🍞 BREADCRUMB 3.7.4: api.run_queue() returned, exists={queue_exists}")
        except Exception:
            pass

        info["wandb"]["queue"] = {
            "exists": queue_exists,
            "name": queue_name
        }
        _status(f"  {'✓' if queue_exists else '✗'} Queue: {queue_name}")

        # Check project
        project_exists = False
        try:
            stevens_log("infra", "🍞 BREADCRUMB 3.7.5: About to call api.project() - NETWORK CALL!")
            proj = api.project(f"{entity}/{project_name}")
            project_exists = proj is not None
            stevens_log("infra", f"🍞 BREADCRUMB 3.7.6: api.project() returned, exists={project_exists}")
        except Exception:
            pass

        info["wandb"]["project"] = {
            "exists": project_exists,
            "name": project_name
        }
        _status(f"  {'✓' if project_exists else '○'} Project: {project_name}")
        stevens_log("infra", "🍞 BREADCRUMB 3.7.7: W&B checks complete!")

    except Exception as e:
        _status(f"  ? W&B check error: {str(e)[:50]}")
        info["wandb"] = {
            "queue": {"exists": False, "name": queue_name},
            "project": {"exists": False, "name": project_name}
        }

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4: CHECK HUGGINGFACE REPO
    # ═══════════════════════════════════════════════════════════════════════

    stevens_log("infra", "🍞 BREADCRUMB 3.8: Starting HuggingFace checks")
    if hf_repo:
        _status("Checking HuggingFace repo...")

        try:
            from huggingface_hub import HfApi

            stevens_log("infra", "🍞 BREADCRUMB 3.8.1: About to create HfApi()")
            api = HfApi()
            stevens_log("infra", "🍞 BREADCRUMB 3.8.2: HfApi() created")

            try:
                stevens_log("infra", "🍞 BREADCRUMB 3.8.3: About to call api.repo_info() - NETWORK CALL!")
                api.repo_info(repo_id=hf_repo, repo_type="model")
                repo_exists = True
                stevens_log("infra", "🍞 BREADCRUMB 3.8.4: api.repo_info() returned, exists=True")
                _status(f"  ✓ Repo: {hf_repo}")
            except Exception as e:
                repo_exists = False
                stevens_log("infra", f"🚨 BREADCRUMB 3.8.4: Exception: {str(e)[:100]}")
                _status(f"  ○ Repo not found: {hf_repo}")

            info["hf"]["repo"] = {
                "exists": repo_exists,
                "id": hf_repo
            }
            stevens_log("infra", "🍞 BREADCRUMB 3.8.5: HF checks complete!")
        except Exception as e:
            _status(f"  ? HF check error: {str(e)[:50]}")
            info["hf"]["repo"] = {
                "exists": False,
                "id": hf_repo
            }
    else:
        info["hf"]["repo"] = {
            "exists": False,
            "id": ""
        }

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 5: CHECK LOCAL CREDENTIALS
    # ═══════════════════════════════════════════════════════════════════════

    _status("Checking local credentials...")

    # Check if service account key file exists
    key_dir = Path.home() / ".gcp-keys"
    key_path = key_dir / f"{sa_name}.json"

    key_exists = key_path.exists()
    info["local"] = {
        "key_file_exists": key_exists,
        "key_path": str(key_path)
    }
    _status(f"  {'✓' if key_exists else '○'} Key file: {key_path.name}")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 6: CHECK CLOUD BUILD WORKER POOLS (MECHA)
    # ═══════════════════════════════════════════════════════════════════════

    stevens_log("infra", "🍞 BREADCRUMB 3.10: Starting Worker Pool checks")
    _status("Checking Cloud Build worker pools...")

    # Skip if billing disabled
    if billing_enabled is False:
        _status("  Skipping (billing disabled)")
        info["cloud_build"] = {"worker_pools": {}}
    else:
        # Check MECHA regions (from MECHA_C3_MANIFEST source of truth)
        mecha_regions = [r["code"] for r in MECHA_C3_MANIFEST["regions"]]
        pool_name = "pytorch-mecha-pool"  # Standard pool name
        machine_type = MECHA_C3_MANIFEST["machine_type_default"]

        pool_acc = GCloudAccumulator(max_workers=10)

        # Start checks for each region
        for region in mecha_regions:
            pool_acc.start(
                key=region,
                cmd=[
                    "gcloud", "builds", "worker-pools", "list",
                    f"--region={region}",
                    f"--project={project_id}",
                    "--format=value(name)"
                ],
                max_retries=3,
                timeout=10,
                operation_name=f"check worker pool in {region}",
            )

        # Get all pool results
        stevens_log("infra", "🍞 BREADCRUMB 3.10.1: About to call pool_acc.get_all() - THIS BLOCKS!")
        pool_results = pool_acc.get_all()
        stevens_log("infra", "🍞 BREADCRUMB 3.10.2: pool_acc.get_all() returned!")
        pool_acc.shutdown()

        # Parse worker pools per region
        worker_pools = {}
        for region in mecha_regions:
            result = pool_results.get(region)
            if result and result.returncode == 0:
                pools = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
                has_pool = pool_name in pools

                worker_pools[region] = {
                    "exists": has_pool,
                    "name": pool_name if has_pool else None,
                    "machine_type": machine_type if has_pool else None
                }
                _status(f"  {'✓' if has_pool else '✗'} {region}: {pool_name if has_pool else 'Not found'}")
            else:
                worker_pools[region] = {
                    "exists": False,
                    "name": None,
                    "machine_type": None
                }
                _status(f"  ✗ {region}: Unable to check")

        info["cloud_build"] = {"worker_pools": worker_pools}

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 7: CHECK GPU & C3 QUOTAS (30-min cache - infra_verify ONLY!)
    # ═══════════════════════════════════════════════════════════════════════
    # NOTE: This cache is for TUI display/status checks ONLY
    # Launch-time checks (launch/core.py) are ALWAYS FRESH - no cache!

    stevens_log("infra", "🍞 BREADCRUMB 3.11: Starting GPU quota checks")
    _status("Checking GPU quotas...")

    # 🔥 STEVEN'S TIMING - Start quota check timer
    quota_start_time = time.time()

    # Skip if billing disabled
    if billing_enabled is False:
        _status("  Skipping (billing disabled)")
        info["quotas"] = {"vertex_gpu": {}, "c3_build": {}}
    else:
        from .quota import get_vertex_gpu_quotas, get_cloud_build_c3_region_quota
        from .log_paths import get_log_path
        from datetime import datetime

        # ─────────────────────────────────────────────────────────────────────
        # GPU QUOTA CHECK (with 30-min cache + partial cache support!)
        # ─────────────────────────────────────────────────────────────────────
        gpu_start = time.time()

        # Debug: Check what's in the cache
        stevens_log("infra", f"🔍 DEBUG PRE-GET: _quota_cache keys: {list(_quota_cache.keys())}")
        stevens_log("infra", f"🔍 DEBUG PRE-GET: _quota_cache id: {id(_quota_cache)}")  # Memory address!
        cached_gpu = _get_cached_quotas("gpu")
        stevens_log("infra", f"🔍 DEBUG POST-GET: cached_gpu is None? {cached_gpu is None}, type: {type(cached_gpu)}")

        # Start with cached data (even if partial!)
        if cached_gpu is not None:
            stevens_log("infra", "🍞 BREADCRUMB 3.11.1: PARTIAL/FULL CACHE! Starting with cached GPU data")
            vertex_gpu = cached_gpu.get("vertex_gpu", {})
            all_gpu_found = cached_gpu.get("all_gpu_found", [])
        else:
            stevens_log("infra", "🍞 BREADCRUMB 3.11.1: NO CACHE! Starting from empty")
            vertex_gpu = {}
            all_gpu_found = []

        # Build set of already-checked GPU+region combos
        checked_combos = set()
        for gpu_entry in all_gpu_found:
            # all_gpu_found has format: {"gpu": "T4", "region": "us-central1", ...}
            checked_combos.add((gpu_entry.get("gpu"), gpu_entry.get("region")))

        stevens_log("infra", f"🍞 BREADCRUMB 3.11.2: Found {len(checked_combos)} cached GPU+region combos, will check remaining")

        # Check only MISSING GPU+region combos (PARALLEL with GeneralAccumulator!)
        gpu_priority_order = [
            ("NVIDIA_TESLA_T4", "T4", "n1-standard-4"),
            ("NVIDIA_L4", "L4", "g2-standard-4"),
            ("NVIDIA_TESLA_A100", "A100", "a2-highgpu-1g"),
            ("NVIDIA_H100_80GB", "H100", "a3-highgpu-8g"),
            ("NVIDIA_H200", "H200", "a3-highgpu-8g"),
        ]
        gpu_check_regions = ["us-central1", "us-east1", "us-west1", "europe-west4"]

        gpu_skip_count = len(checked_combos)

        # 🦡⚡ BUILD list of checks to run in PARALLEL!
        gpu_checks = []  # List of (key, gpu_display, machine_type, region, gpu_internal)
        for gpu_internal, gpu_display, machine_type in gpu_priority_order:
            for region in gpu_check_regions:
                # Skip if already in cache!
                if (gpu_display, region) in checked_combos:
                    continue

                check_key = f"gpu_{gpu_display}_{region}"
                gpu_checks.append((check_key, gpu_display, machine_type, region, gpu_internal))

        gpu_check_count = len(gpu_checks)

        # 🦡⚡ LAUNCH ALL CHECKS IN PARALLEL WITH PROGRESSIVE TOASTS!
        from .api_helpers import GeneralAccumulator
        from .steven_toasts import steven_notify

        gpu_acc = GeneralAccumulator(max_workers=min(20, len(gpu_checks)))  # Up to 20 parallel!

        for check_key, gpu_display, machine_type, region, gpu_internal in gpu_checks:
            gpu_acc.start(check_key, lambda p=project_id, r=region, g=gpu_internal: get_vertex_gpu_quotas(p, r, g, use_spot=True))

        # ⏳ PROGRESSIVE RENDERING - Show toast as EACH check completes!
        # (time already imported at module level - line 36!)
        rendered = {check_key: False for check_key, _, _, _, _ in gpu_checks}
        gpu_fail_count = 0

        while not all(rendered.values()):
            for check_key, gpu_display, machine_type, region, gpu_internal in gpu_checks:
                if rendered[check_key]:
                    continue  # Already rendered

                if gpu_acc.is_done(check_key):
                    try:
                        quota_limit = gpu_acc.get(check_key)
                        if quota_limit and quota_limit > 0:
                            all_gpu_found.append({
                                "gpu": gpu_display,
                                "machine_type": machine_type,
                                "region": region,
                                "quota": quota_limit
                            })
                            if region not in vertex_gpu:
                                vertex_gpu[region] = {"granted": [], "pending": []}
                            vertex_gpu[region]["granted"].append({
                                "gpu_type": gpu_internal,
                                "quota_limit": quota_limit,
                                "region": region
                            })
                            # 💙 BLUE TOAST - Success! (via callback to main thread)
                            if gpu_progress_callback:
                                gpu_progress_callback(gpu_display, region, quota_limit, failed=False)
                        else:
                            # No quota (not an error, just 0)
                            pass
                    except Exception:
                        gpu_fail_count += 1
                        # ❤️ RED TOAST - Error! (via callback to main thread)
                        if gpu_progress_callback:
                            gpu_progress_callback(gpu_display, region, 0, failed=True)

                    rendered[check_key] = True

            time.sleep(0.05)  # Poll interval

        gpu_acc.shutdown()

        stevens_log("infra", f"🍞 BREADCRUMB 3.11.3: GPU checks complete! Cached: {gpu_skip_count}, Fresh: {gpu_check_count}, Total: {len(all_gpu_found)}")

        # Update cache with complete data (cached + new)
        _set_cached_quotas("gpu", {"vertex_gpu": vertex_gpu, "all_gpu_found": all_gpu_found})

        gpu_elapsed = time.time() - gpu_start

        # Display GPU results
        if all_gpu_found:
            first = all_gpu_found[0]
            _status(f"  ✓ Minimum GPU Quota met: {first['gpu']} ({first['machine_type']}) in {first['region']}")
            for gpu in all_gpu_found[1:]:
                _status(f"    + {gpu['gpu']} ({gpu['machine_type']}) in {gpu['region']}")
        else:
            _status("  ✗ No GPU quotas found (request quota for T4 or L4)")

        # ─────────────────────────────────────────────────────────────────────
        # C3 QUOTA CHECK (with 30-min cache + partial cache support!)
        # ─────────────────────────────────────────────────────────────────────
        stevens_log("infra", "🍞 BREADCRUMB 3.12: Starting C3 quota checks")
        c3_start = time.time()
        cached_c3 = _get_cached_quotas("c3")

        # Start with cached data (even if partial!)
        if cached_c3 is not None:
            stevens_log("infra", "🍞 BREADCRUMB 3.12.1: PARTIAL/FULL CACHE! Starting with cached C3 data")
            c3_build = cached_c3.get("c3_build", {})
            all_c3_found = cached_c3.get("all_c3_found", [])
        else:
            stevens_log("infra", "🍞 BREADCRUMB 3.12.1: NO CACHE! Starting from empty")
            c3_build = {}
            all_c3_found = []

        # Build set of already-checked regions
        checked_regions = set(c3_build.keys())
        stevens_log("infra", f"🍞 BREADCRUMB 3.12.2: Found {len(checked_regions)} cached C3 regions, will check remaining")

        # Check only MISSING regions (PARALLEL with GeneralAccumulator!)
        c3_check_regions = [r["code"] for r in MECHA_C3_MANIFEST["regions"]]
        c3_skip_count = len(checked_regions)

        # 🦡⚡ BUILD list of checks to run in PARALLEL!
        c3_checks = []  # List of (region)
        for region in c3_check_regions:
            # Skip if already in cache!
            if region in checked_regions:
                continue
            c3_checks.append(region)

        c3_check_count = len(c3_checks)

        # 🦡⚡ LAUNCH ALL CHECKS IN PARALLEL WITH PROGRESSIVE TOASTS!
        c3_acc = GeneralAccumulator(max_workers=min(20, len(c3_checks)))  # Up to 20 parallel!

        for region in c3_checks:
            c3_acc.start(f"c3_{region}", lambda p=project_id, r=region: get_cloud_build_c3_region_quota(p, r))

        # ⏳ PROGRESSIVE RENDERING - Show toast as EACH check completes!
        rendered_c3 = {f"c3_{region}": False for region in c3_checks}
        c3_fail_count = 0

        while not all(rendered_c3.values()):
            for region in c3_checks:
                check_key = f"c3_{region}"
                if rendered_c3[check_key]:
                    continue  # Already rendered

                if c3_acc.is_done(check_key):
                    try:
                        quota_info = c3_acc.get(check_key)
                        if quota_info and quota_info.get("vcpus", 0) > 0:
                            vcpus = quota_info.get("vcpus", 0)
                            machine = quota_info.get("machine_type", "unknown")
                            all_c3_found.append({
                                "region": region,
                                "vcpus": vcpus,
                                "machine": machine
                            })
                            c3_build[region] = quota_info
                            # 💙 BLUE TOAST - Success! (via callback to main thread)
                            if c3_progress_callback:
                                c3_progress_callback(region, vcpus, failed=False)
                        else:
                            # No quota (not an error, just 0)
                            pass
                    except Exception:
                        c3_fail_count += 1
                        # ❤️ RED TOAST - Error! (via callback to main thread)
                        if c3_progress_callback:
                            c3_progress_callback(region, 0, failed=True)

                    rendered_c3[check_key] = True

            time.sleep(0.05)  # Poll interval

        c3_acc.shutdown()

        stevens_log("infra", f"🍞 BREADCRUMB 3.12.3: C3 checks complete! Cached: {c3_skip_count}, Fresh: {c3_check_count}, Total: {len(all_c3_found)}")

        # Update cache with complete data (cached + new)
        _set_cached_quotas("c3", {"c3_build": c3_build, "all_c3_found": all_c3_found})

        c3_elapsed = time.time() - c3_start
        total_quota_elapsed = time.time() - quota_start_time

        # 🔥 STEVEN'S TIMING LOG
        if STEVEN_INFRA_VERIFY_DEBUG:
            log_file = get_log_path("infra_verify_timing.log")
            with open(log_file, "a") as f:
                f.write(f"{datetime.now().isoformat()} ⏱️ QUOTA_TIMING:\n")
                f.write(f"  GPU: {gpu_elapsed:.2f}s (cached={gpu_skip_count}, fresh={gpu_check_count}, fails={gpu_fail_count}, found={len(all_gpu_found)})\n")
                f.write(f"  C3:  {c3_elapsed:.2f}s (cached={c3_skip_count}, fresh={c3_check_count}, fails={c3_fail_count}, found={len(all_c3_found)})\n")
                f.write(f"  TOTAL: {total_quota_elapsed:.2f}s\n")

        # Display C3 results
        if all_c3_found:
            first = all_c3_found[0]
            _status(f"  ✓ Minimum C3 Quota met: {first['vcpus']} vCPUs ({first['machine']}) in {first['region']}")
            for c3 in all_c3_found[1:]:
                _status(f"    + {c3['vcpus']} vCPUs ({c3['machine']}) in {c3['region']}")
        else:
            _status("  ✗ No C3 quotas found (MECHA builds unavailable)")

        info["quotas"] = {
            "vertex_gpu": vertex_gpu,
            "c3_build": c3_build
        }

    # ═══════════════════════════════════════════════════════════════════════
    # DONE!
    # ═══════════════════════════════════════════════════════════════════════

    stevens_log("infra", "🍞 BREADCRUMB 3.9: All checks complete, about to return info")
    _status("Infrastructure check complete!")

    return info
