"""
Vertex AI GPU Quota Checking

Queries Vertex AI Custom Training quotas (NOT Compute Engine quotas!).
Different GPU types use different quota metrics.
"""

import subprocess
import json
from typing import Dict, List, Optional

# Import helpers for robust gcloud API calls
from ..api_helpers import run_gcloud_with_retry, run_gcloud_batch_parallel


# <claudes_code_comments>
# ** Function List **
# get_vertex_gpu_quotas(project_id, region, gpu_type, use_spot) - Get quota for ONE GPU type in ONE region
# get_all_vertex_gpu_quotas(project_id, region) - Get ALL GPU quotas for ONE region (all types)
# get_vertex_gpu_quota_metric(gpu_type, use_spot) - Convert GPU type to quota metric name
# has_vertex_gpu_quota(project_id, region, gpu_type, use_spot, required) - Check if region has sufficient quota
# get_all_gpu_quotas_for_type(project_id, gpu_type, use_spot) - Get quota for ONE GPU type across ALL regions (1 API call)
# get_all_gpu_quotas_multi_type(project_id, gpu_types, use_spot, max_workers) - Get quotas for MULTIPLE GPU types in PARALLEL (3-5× faster)
# _fetch_vertex_gpu_quota_from_api(project_id, region, metric_name) - Internal: Fetch specific quota from API
# _extract_gpu_name(quota_id) - Internal: Extract GPU name from quota ID string
# get_gpu_quota_display_data(project_id, region) - Get formatted quota data for TUI display
#
# ** Technical Review **
# Vertex AI GPU quota checking for Custom Training (NOT Compute Engine).
# Uses gcloud alpha services quota API with server-side filtering for performance.
#
# Single region/type queries:
# - get_vertex_gpu_quotas(): One region, one GPU type (specific lookup)
# - get_all_vertex_gpu_quotas(): One region, all GPU types (infra screen display)
#
# Multi-region/type queries (FAST):
# - get_all_gpu_quotas_for_type(): ONE GPU type, ALL regions, ONE API call (15× faster!)
# - get_all_gpu_quotas_multi_type(): MULTIPLE GPU types, ALL regions, PARALLEL (3-5× faster!)
#
# Parallel multi-type implementation (NEW):
# Uses run_gcloud_batch_parallel() to query multiple GPU types simultaneously.
# Default max_workers=10. Batching behavior: 15 types with max_workers=10 = 2 batches (10+5).
# Example: Query H100, L4, A100 quotas at once instead of sequentially.
# Returns: {gpu_type: {region: quota_limit}} for all requested types.
#
# GPU_QUOTA_METRICS maps GPU types to Vertex AI metric names (10 types supported).
# get_vertex_gpu_quota_metric() validates GPU type and builds full metric name with preemptible prefix.
# Server-side filtering (--filter=metric=...) reduces API response size dramatically.
# </claudes_code_comments>


# ═══════════════════════════════════════════════════════════════════════════════
# 🎮 GPU TYPES MANIFEST - SOURCE OF TRUTH FOR ALL GPU TYPES
# ═══════════════════════════════════════════════════════════════════════════════
# This is the canonical definition for ALL Vertex AI GPU types.
# Other modules should import from here, NOT from gcp-manifest.json
# gcp-manifest.json just points here as the source of truth.

GPU_TYPES_MANIFEST = {
    "description": "All Vertex AI GPU types for quota checking",
    "total_types": 10,
    "zeus_supported_count": 5,
    "non_zeus_count": 5,
    "types": [
        # ZEUS SUPPORTED (5 types - have thunder tiers)
        {"internal": "NVIDIA_TESLA_T4", "display": "T4", "metric": "nvidia_t4_gpus", "memory_gb": 16, "zeus_tier": "spark", "generation": "budget", "machine_type": "n1-standard-4"},
        {"internal": "NVIDIA_L4", "display": "L4", "metric": "nvidia_l4_gpus", "memory_gb": 24, "zeus_tier": "bolt", "generation": "budget", "machine_type": "g2-standard-4"},
        {"internal": "NVIDIA_TESLA_A100", "display": "A100", "metric": "nvidia_a100_gpus", "memory_gb": 40, "zeus_tier": "storm", "generation": "previous", "machine_type": "a2-highgpu-1g"},
        {"internal": "NVIDIA_H100_80GB", "display": "H100 80GB", "metric": "nvidia_h100_80gb_gpus", "memory_gb": 80, "zeus_tier": "tempest", "generation": "current", "machine_type": "a3-highgpu-8g"},
        {"internal": "NVIDIA_H200", "display": "H200", "metric": "nvidia_h200_gpus", "memory_gb": 141, "zeus_tier": "cataclysm", "generation": "current", "machine_type": "a3-highgpu-8g"},
        # NON-ZEUS (5 types - manual region selection)
        {"internal": "NVIDIA_A100_80GB", "display": "A100 80GB", "metric": "nvidia_a100_80gb_gpus", "memory_gb": 80, "zeus_tier": None, "generation": "previous", "machine_type": "a2-ultragpu-1g"},
        {"internal": "NVIDIA_H100", "display": "H100", "metric": "nvidia_h100_gpus", "memory_gb": 80, "zeus_tier": None, "generation": "current", "machine_type": "a3-highgpu-8g"},
        {"internal": "NVIDIA_TESLA_V100", "display": "V100", "metric": "nvidia_v100_gpus", "memory_gb": 16, "zeus_tier": None, "generation": "legacy", "machine_type": "n1-standard-8"},
        {"internal": "NVIDIA_TESLA_P4", "display": "P4", "metric": "nvidia_p4_gpus", "memory_gb": 8, "zeus_tier": None, "generation": "legacy", "machine_type": "n1-standard-4"},
        {"internal": "NVIDIA_TESLA_P100", "display": "P100", "metric": "nvidia_p100_gpus", "memory_gb": 16, "zeus_tier": None, "generation": "legacy", "machine_type": "n1-standard-8"},
    ],
}

# Runtime lookup dict (derived from manifest - kept for backward compatibility)
# GPU type → Vertex AI quota metric mapping
GPU_QUOTA_METRICS = {
    "NVIDIA_TESLA_T4": "nvidia_t4_gpus",
    "NVIDIA_L4": "nvidia_l4_gpus",
    "NVIDIA_TESLA_V100": "nvidia_v100_gpus",
    "NVIDIA_TESLA_P4": "nvidia_p4_gpus",
    "NVIDIA_TESLA_P100": "nvidia_p100_gpus",
    "NVIDIA_TESLA_A100": "nvidia_a100_gpus",
    "NVIDIA_A100_80GB": "nvidia_a100_80gb_gpus",
    "NVIDIA_H100": "nvidia_h100_gpus",
    "NVIDIA_H100_80GB": "nvidia_h100_80gb_gpus",
    "NVIDIA_H200": "nvidia_h200_gpus",
}


def get_vertex_gpu_quotas(project_id: str, region: str, gpu_type: str, use_spot: bool) -> int:
    """
    Get Vertex AI GPU quota for specific GPU type in region

    Args:
        project_id: GCP project ID
        region: GCP region (e.g., "us-central1")
        gpu_type: GPU type (e.g., "NVIDIA_TESLA_T4")
        use_spot: True for spot/preemptible, False for on-demand

    Returns:
        GPU quota limit (0 if no quota)

    Example:
        >>> get_vertex_gpu_quotas("my-project", "us-west2", "NVIDIA_TESLA_T4", True)
        1  # 1x T4 spot GPU available
    """
    metric_name = get_vertex_gpu_quota_metric(gpu_type, use_spot)
    quota = _fetch_vertex_gpu_quota_from_api(project_id, region, metric_name)
    return quota


def get_all_vertex_gpu_quotas(project_id: str, region: str) -> List[Dict]:
    """
    Get ALL Vertex AI GPU quotas for a region (all GPU types, spot + on-demand)

    Returns list of quota info dicts sorted by quota limit (highest first).
    Used by infra screen to show all available GPUs.

    Args:
        project_id: GCP project ID
        region: GCP region

    Returns:
        List of dicts with keys: gpu_name, quota_limit, metric_name, is_spot
        Example: [
            {"gpu_name": "T4", "quota_limit": 1, "metric_name": "..._nvidia_t4_gpus", "is_spot": True},
            {"gpu_name": "L4", "quota_limit": 0, "metric_name": "..._nvidia_l4_gpus", "is_spot": False},
        ]
    """
    all_quotas = []

    # Query all GPU quotas from Vertex AI API
    # PERFORMANCE: Filter server-side for GPU quotas only (not TPU/CPU/etc)
    try:
        result = run_gcloud_with_retry(
            [
                'gcloud', 'alpha', 'services', 'quota', 'list',
                '--service=aiplatform.googleapis.com',
                f'--consumer=projects/{project_id}',
                '--filter=metric:custom_model_training AND metric:nvidia',  # Server-side filter!
                '--format=json'
            ],
            max_retries=3,
            timeout=60,
            operation_name="fetch all Vertex AI GPU quotas",
        )

        if result.returncode != 0:
            return []

        data = json.loads(result.stdout)

        # Process GPU quotas (already filtered server-side to only GPU metrics)
        for item in data:
            metric = item.get('metric', '')

            # Extract quota_id (metric without service prefix)
            quota_id = metric.replace('aiplatform.googleapis.com/', '')

            # Determine GPU name and spot status
            is_spot = 'preemptible' in quota_id
            gpu_name = _extract_gpu_name(quota_id)

            # Find quota limit for the requested region
            quota_limit = 0
            for limit in item.get('consumerQuotaLimits', []):
                for bucket in limit.get('quotaBuckets', []):
                    dims = bucket.get('dimensions', {})
                    bucket_region = dims.get('region', 'global')

                    if bucket_region == region:
                        # Found the region - get effective limit
                        effective_str = bucket.get('effectiveLimit', bucket.get('defaultLimit', '0'))
                        try:
                            quota_limit = int(effective_str)
                        except (ValueError, TypeError):
                            quota_limit = 0
                        break
                if quota_limit > 0:  # Found quota for this region
                    break

            all_quotas.append({
                'gpu_name': gpu_name,
                'quota_limit': quota_limit,
                'metric_name': metric,
                'is_spot': is_spot,
                'quota_id': quota_id,
            })

        # Sort by quota limit (highest first)
        all_quotas.sort(key=lambda x: x['quota_limit'], reverse=True)

        return all_quotas

    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
        return []


def get_vertex_gpu_quota_metric(gpu_type: str, use_spot: bool) -> str:
    """
    Convert GPU type to Vertex AI quota metric name

    Args:
        gpu_type: GPU type (e.g., "NVIDIA_TESLA_T4")
        use_spot: True for spot/preemptible

    Returns:
        Full metric name
        Example: "aiplatform.googleapis.com/custom_model_training_preemptible_nvidia_t4_gpus"

    Raises:
        ValueError: If gpu_type is not in supported GPU types
    """
    # Check if GPU type is supported
    if gpu_type not in GPU_QUOTA_METRICS:
        supported_gpus = "\n  - ".join(sorted(GPU_QUOTA_METRICS.keys()))
        raise ValueError(
            f"Unsupported GPU type: {gpu_type}\n\n"
            f"Supported GPU types:\n  - {supported_gpus}\n\n"
            f"If this is a new GPU type from GCP, please update GPU_QUOTA_METRICS in:\n"
            f"  CLI/shared/quota/gpu_quota.py\n\n"
            f"Add the new GPU type mapping:\n"
            f'  "{gpu_type}": "nvidia_<type>_gpus",  # Example: nvidia_l5_gpus'
        )

    quota_suffix = GPU_QUOTA_METRICS[gpu_type]

    if use_spot:
        quota_suffix = f"preemptible_{quota_suffix}"

    return f"aiplatform.googleapis.com/custom_model_training_{quota_suffix}"


def has_vertex_gpu_quota(project_id: str, region: str, gpu_type: str, use_spot: bool, required: int = 1) -> bool:
    """
    Check if region has sufficient GPU quota

    Args:
        project_id: GCP project ID
        region: GCP region
        gpu_type: GPU type
        use_spot: Spot or on-demand
        required: Minimum quota needed (default: 1)

    Returns:
        True if quota >= required
    """
    quota = get_vertex_gpu_quotas(project_id, region, gpu_type, use_spot)
    return quota >= required


def get_all_gpu_quotas_for_type(project_id: str, gpu_type: str, use_spot: bool = True) -> Dict[str, int]:
    """
    Get GPU quota for specific GPU type across ALL regions (ONE API call)

    This is Zeus's equivalent of MECHA's get_cloud_build_c3_quotas()!
    Returns quota for EVERY region in a single API call.

    Args:
        project_id: GCP project ID
        gpu_type: GPU type (e.g., "NVIDIA_H100_80GB")
        use_spot: True for spot/preemptible

    Returns:
        Dict mapping region → quota_limit
        Example: {"us-central1": 8, "us-east4": 4, "europe-west4": 0}

    Performance:
        ONE API call returns ALL regions (vs 15 separate calls)
    """
    metric_name = get_vertex_gpu_quota_metric(gpu_type, use_spot)

    try:
        result = run_gcloud_with_retry(
            [
                'gcloud', 'alpha', 'services', 'quota', 'list',
                '--service=aiplatform.googleapis.com',
                f'--consumer=projects/{project_id}',
                f'--filter=metric={metric_name}',  # Server-side filter for this GPU type
                '--format=json'
            ],
            max_retries=3,
            timeout=60,
            operation_name="fetch GPU quotas for type",
        )

        if result.returncode != 0:
            return {}

        data = json.loads(result.stdout)

        # With --filter, we get exactly 1 metric (or 0 if not found)
        if not data or len(data) == 0:
            return {}

        item = data[0]  # Only one metric returned by filter

        # Extract quota for ALL regions from quota buckets
        quotas = {}
        for limit in item.get('consumerQuotaLimits', []):
            for bucket in limit.get('quotaBuckets', []):
                dims = bucket.get('dimensions', {})
                bucket_region = dims.get('region', 'global')

                if bucket_region != 'global':  # Skip global buckets
                    # Get effective limit
                    effective_str = bucket.get('effectiveLimit', bucket.get('defaultLimit', '0'))
                    try:
                        quota_limit = int(effective_str)
                        quotas[bucket_region] = quota_limit
                    except (ValueError, TypeError):
                        quotas[bucket_region] = 0

        return quotas

    except (subprocess.TimeoutExpired, ValueError, TypeError, Exception):
        return {}


def get_all_gpu_quotas_multi_type(
    project_id: str,
    gpu_types: List[str],
    use_spot: bool = True,
    max_workers: int = 10
) -> Dict[str, Dict[str, int]]:
    """
    Get GPU quotas for MULTIPLE GPU types in PARALLEL.

    Parallel version of get_all_gpu_quotas_for_type() for multiple types.
    Instead of querying one GPU type at a time, queries ALL types simultaneously.

    Perfect for:
    - Zeus refreshing multiple tier fleets at once
    - Initial Mount Olympus setup (scan all tiers)
    - Multi-tier price comparisons

    Args:
        project_id: GCP project ID
        gpu_types: List of GPU types (e.g., ["NVIDIA_H100_80GB", "NVIDIA_L4", "NVIDIA_TESLA_A100"])
        use_spot: True for spot/preemptible quotas
        max_workers: Max parallel jobs (default 10). If len(gpu_types) > max_workers,
                     queries run in batches of max_workers at a time.

    Returns:
        Dict mapping gpu_type → {region → quota_limit}
        Example: {
            "NVIDIA_H100_80GB": {"us-central1": 8, "us-east4": 0},
            "NVIDIA_L4": {"us-west1": 16, "europe-west4": 8},
            "NVIDIA_TESLA_A100": {"us-east4": 2, "asia-northeast1": 4}
        }

    Performance:
        3-5× faster than sequential queries (queries run in parallel!)

        Sequential (old way):
            H100: 5 sec + L4: 5 sec + A100: 5 sec = 15 seconds total

        Parallel (this function, 3 types, max_workers=10):
            H100 + L4 + A100 all at once = 5 seconds total! ⚡

        Parallel (15 types, max_workers=10):
            First batch: 10 types (5 sec) + Second batch: 5 types (5 sec) = 10 seconds
            Still much faster than 75 seconds sequential!
    """
    # Build parallel gcloud commands (one per GPU type)
    commands = []
    for gpu_type in gpu_types:
        try:
            metric_name = get_vertex_gpu_quota_metric(gpu_type, use_spot)

            commands.append({
                "cmd": [
                    'gcloud', 'alpha', 'services', 'quota', 'list',
                    '--service=aiplatform.googleapis.com',
                    f'--consumer=projects/{project_id}',
                    f'--filter=metric={metric_name}',
                    '--format=json'
                ],
                "operation_name": f"get {gpu_type} quotas",
                "timeout": 60,
            })
        except ValueError:
            # Invalid GPU type - skip it (get_vertex_gpu_quota_metric raises ValueError)
            continue

    # Run all quota queries IN PARALLEL! ⚡
    results = run_gcloud_batch_parallel(commands, max_workers=max_workers)

    # Parse results back into quota dictionaries
    all_quotas = {}

    for idx, result, error in results:
        gpu_type = gpu_types[idx]

        if error or not result:
            # API call failed - return empty dict for this GPU type
            all_quotas[gpu_type] = {}
            continue

        try:
            data = json.loads(result.stdout)

            # No quota data returned
            if not data or len(data) == 0:
                all_quotas[gpu_type] = {}
                continue

            item = data[0]  # Server-side filter returns exactly 1 metric

            # Extract quota for ALL regions from quota buckets
            quotas = {}
            for limit in item.get('consumerQuotaLimits', []):
                for bucket in limit.get('quotaBuckets', []):
                    dims = bucket.get('dimensions', {})
                    bucket_region = dims.get('region', 'global')

                    if bucket_region != 'global':  # Skip global buckets
                        # Get effective limit
                        effective_str = bucket.get('effectiveLimit', bucket.get('defaultLimit', '0'))
                        try:
                            quota_limit = int(effective_str)
                            quotas[bucket_region] = quota_limit
                        except (ValueError, TypeError):
                            quotas[bucket_region] = 0

            all_quotas[gpu_type] = quotas

        except (json.JSONDecodeError, KeyError, Exception):
            # Parsing failed - return empty dict for this GPU type
            all_quotas[gpu_type] = {}

    return all_quotas


def _fetch_vertex_gpu_quota_from_api(project_id: str, region: str, metric_name: str) -> int:
    """
    Internal: Fetch specific GPU quota from Vertex AI API

    Args:
        project_id: GCP project ID
        region: GCP region
        metric_name: Full metric name

    Returns:
        Quota value (0 if not found)
    """
    # Use correct gcloud command with --filter to fetch ONLY the specific metric
    # PERFORMANCE: Filtering server-side returns 1 quota instead of ~100+ quotas!
    try:
        result = run_gcloud_with_retry(
            [
                'gcloud', 'alpha', 'services', 'quota', 'list',
                '--service=aiplatform.googleapis.com',
                f'--consumer=projects/{project_id}',
                f'--filter=metric={metric_name}',  # ← Server-side filter (FAST!)
                '--format=json'
            ],
            max_retries=3,
            timeout=60,
            operation_name="fetch Vertex AI GPU quota",
        )

        if result.returncode != 0:
            return 0

        data = json.loads(result.stdout)

        # With --filter, we get exactly 1 metric (or 0 if not found)
        if not data or len(data) == 0:
            return 0

        item = data[0]  # Only one metric returned by filter

        # Extract quota for the requested region
        for limit in item.get('consumerQuotaLimits', []):
            for bucket in limit.get('quotaBuckets', []):
                dims = bucket.get('dimensions', {})
                bucket_region = dims.get('region', 'global')

                if bucket_region == region:
                    # Found the region - get effective limit
                    effective_str = bucket.get('effectiveLimit', bucket.get('defaultLimit', '0'))
                    try:
                        return int(effective_str)
                    except (ValueError, TypeError):
                        return 0

        # Region not found in quota buckets
        return 0

    except (subprocess.TimeoutExpired, ValueError, TypeError, Exception):
        return 0


def _extract_gpu_name(quota_id: str) -> str:
    """
    Internal: Extract display name from quota ID

    Example:
        "custom_model_training_preemptible_nvidia_t4_gpus" → "T4"
        "custom_model_training_nvidia_h100_80gb_gpus" → "H100 80GB"
    """
    # Remove prefix
    name = quota_id.replace('custom_model_training_', '')
    name = name.replace('preemptible_', '')
    name = name.replace('nvidia_', '')
    name = name.replace('_gpus', '')

    # Map to display names
    name_map = {
        't4': 'T4',
        'l4': 'L4',
        'v100': 'V100',
        'p4': 'P4',
        'p100': 'P100',
        'a100': 'A100',
        'a100_80gb': 'A100 80GB',
        'h100': 'H100',
        'h100_80gb': 'H100 80GB',
        'h200': 'H200',
    }

    return name_map.get(name, name.upper())


def get_gpu_quota_display_data(project_id: str, region: str) -> Dict:
    """
    Get GPU quota data formatted for display in infrastructure tree

    Returns dict with:
    - granted: List of GPU quotas > 0 (user has access)
    - pending: List of GPU quotas = 0 (need to request)
    - quota_request_url: URL to request quotas in GCP Console

    Each quota dict includes:
    - gpu_name: Display name (e.g., "T4", "H100 80GB")
    - quota_limit: How many GPUs available
    - is_spot: True if spot/preemptible
    - metric_name: Full Vertex AI metric name
    - request_url: Direct URL to request this specific quota

    Args:
        project_id: GCP project ID
        region: GCP region

    Returns:
        Dict with 'granted', 'pending', 'quota_request_url' keys
    """
    all_quotas = get_all_vertex_gpu_quotas(project_id, region)

    granted = [q for q in all_quotas if q['quota_limit'] > 0]
    pending = [q for q in all_quotas if q['quota_limit'] == 0]

    # Add request URLs to each quota
    base_url = f"https://console.cloud.google.com/iam-admin/quotas?project={project_id}"

    for quota in all_quotas:
        # Build search-friendly quota name for URL
        metric = quota.get('quota_id', '')
        quota['request_url'] = f"{base_url}&service=aiplatform.googleapis.com&metric={metric}"

    return {
        'granted': granted,
        'pending': pending,
        'quota_request_url': base_url,
        'region': region,
    }
