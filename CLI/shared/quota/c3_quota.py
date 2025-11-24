"""
Cloud Build C3 Quota Checking

Centralized quota checking for Cloud Build worker pools.
Returns confirmed Cloud Build C3 quotas for regions with approved custom quotas.

Uses in-memory caching (60-second TTL) to avoid redundant API calls when checking
multiple regions (e.g., MECHA checking 5 regions = 1 API call instead of 5).
"""

import subprocess
import json
import time
from typing import Dict

# Import retry helper for robust gcloud API calls
from ..api_helpers import run_gcloud_with_retry

# Module-level cache for C3 quotas
# Key: project_id, Value: {region: quota_vcpus}
_c3_quota_cache: Dict[str, Dict[str, int]] = {}
_c3_quota_cache_time: Dict[str, float] = {}
CACHE_TTL_SECONDS = 60  # 1 minute - covers command workflows, fresh enough for quota changes


def get_cloud_build_c3_quotas(project_id: str) -> Dict[str, int]:
    """
    Get Cloud Build C3 quota for all regions (with 60-second cache)

    Args:
        project_id: GCP project ID

    Returns:
        Dict mapping region → vCPUs quota
        Example: {"us-central1": 176, "asia-northeast1": 176}

    Note:
        Uses in-memory cache (60s TTL) to avoid redundant API calls.
        Checking 5 regions = 1 API call instead of 5!
    """
    cache_key = project_id
    now = time.time()

    # Check if cache exists and is still valid
    if cache_key in _c3_quota_cache:
        cache_age = now - _c3_quota_cache_time.get(cache_key, 0)
        if cache_age < CACHE_TTL_SECONDS:
            # Cache hit! Return cached data
            return _c3_quota_cache[cache_key].copy()

    # Cache miss or stale - fetch fresh data from API
    quotas = _fetch_c3_quotas_from_api(project_id)

    # Update cache
    _c3_quota_cache[cache_key] = quotas
    _c3_quota_cache_time[cache_key] = now

    return quotas.copy()


def get_cloud_build_c3_region_quota(project_id: str, region: str) -> int:
    """
    Get Cloud Build C3 quota for single region

    Args:
        project_id: GCP project ID
        region: GCP region (e.g., "us-central1")

    Returns:
        vCPUs quota for that region (0 if no quota)
    """
    quotas = get_cloud_build_c3_quotas(project_id)
    return quotas.get(region, 0)


def has_cloud_build_c3_quota(project_id: str, region: str, required_vcpus: int) -> bool:
    """
    Check if region has sufficient Cloud Build C3 quota

    Args:
        project_id: GCP project ID
        region: GCP region
        required_vcpus: Minimum vCPUs needed (e.g., 176)

    Returns:
        True if quota >= required_vcpus
    """
    quota = get_cloud_build_c3_region_quota(project_id, region)
    return quota >= required_vcpus


def _fetch_c3_quotas_from_api(project_id: str) -> Dict[str, int]:
    """
    Internal: Fetch quotas from gcloud API

    Returns only quotas with build_origin=default dimension

    PERFORMANCE: Uses server-side filter to fetch only C3 quotas (not all Cloud Build quotas)
    """
    try:
        result = run_gcloud_with_retry(
            [
                'gcloud', 'alpha', 'services', 'quota', 'list',
                '--service=cloudbuild.googleapis.com',
                f'--consumer=projects/{project_id}',
                '--filter=metric:concurrent_private_pool_c3_build_cpus',  # Server-side filter!
                '--format=json'
            ],
            max_retries=3,
            timeout=30,
            operation_name="fetch Cloud Build C3 quotas",
        )

        if result.returncode != 0:
            return {}

        data = json.loads(result.stdout)
        quotas = {}

        # Process C3 quotas (already filtered server-side)
        for service in data:
            for limit in service.get('consumerQuotaLimits', []):

                for bucket in limit.get('quotaBuckets', []):
                    dims = bucket.get('dimensions', {})

                    if 'build_origin' in dims and dims['build_origin'] == 'default':
                        if 'region' in dims:
                            region = dims['region']
                            effective_limit = bucket.get('effectiveLimit')

                            if effective_limit:
                                try:
                                    quotas[region] = int(effective_limit)
                                except (ValueError, TypeError):
                                    pass

        return quotas

    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
        return {}


def get_c3_quota_display_data(project_id: str) -> Dict:
    """
    Get Cloud Build C3 quota data formatted for display in infrastructure tree

    Returns dict with:
    - granted: List of regions with C3 quota > 0 (can build in these regions)
    - pending: List of regions with C3 quota = 0 (need to request)
    - quota_request_url: URL to request quotas in GCP Console

    Each region dict includes:
    - region: Region name (e.g., "us-central1")
    - vcpus: vCPU quota (e.g., 176)
    - machine_type: Best C3 machine for this quota (e.g., "c3-standard-176")
    - request_url: Direct URL to request quota for this region

    Args:
        project_id: GCP project ID

    Returns:
        Dict with 'granted', 'pending', 'quota_request_url' keys
    """
    all_quotas = get_cloud_build_c3_quotas(project_id)

    # Import machine selection here to avoid circular import
    from ..machine_selection import get_best_c3

    granted = []
    pending = []

    # Check all known GCP regions
    # (In practice, C3 quotas are only approved in specific regions)
    known_regions = [
        "us-central1", "us-east1", "us-west1", "us-west2", "us-west3", "us-west4",
        "us-east4", "us-east5", "us-south1",
        "europe-west1", "europe-west2", "europe-west3", "europe-west4",
        "asia-east1", "asia-northeast1", "asia-southeast1",
        "australia-southeast1",
    ]

    for region in known_regions:
        vcpus = all_quotas.get(region, 0)

        # Get best machine for this quota
        try:
            machine, machine_vcpus, quota_confirmed = get_best_c3(project_id, region)
        except Exception:
            machine = "c3-standard-4" if vcpus >= 4 else "none"
            machine_vcpus = vcpus

        region_data = {
            'region': region,
            'vcpus': vcpus,
            'machine_type': machine,
            'request_url': f"https://console.cloud.google.com/iam-admin/quotas?project={project_id}&service=cloudbuild.googleapis.com&metric=concurrent_private_pool_c3_build_cpus&region={region}"
        }

        if vcpus > 0:
            granted.append(region_data)
        else:
            pending.append(region_data)

    # Sort granted by vCPUs (highest first)
    granted.sort(key=lambda x: x['vcpus'], reverse=True)

    return {
        'granted': granted,
        'pending': pending,
        'quota_request_url': f"https://console.cloud.google.com/iam-admin/quotas?project={project_id}&service=cloudbuild.googleapis.com",
        'total_regions_with_quota': len(granted),
    }
