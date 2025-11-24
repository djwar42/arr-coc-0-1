"""
Quota Module - Canonical quota checking for Cloud Build and Vertex AI

Import everything from here:
    from CLI.shared.quota import get_cloud_build_c3_quotas, get_all_vertex_gpu_quotas, ...
"""

from .c3_quota import (
    get_cloud_build_c3_quotas,
    get_cloud_build_c3_region_quota,
    has_cloud_build_c3_quota,
    get_c3_quota_display_data,  # Helper for infrastructure tree display
)

from .gpu_quota import (
    get_vertex_gpu_quotas,
    get_all_vertex_gpu_quotas,
    get_vertex_gpu_quota_metric,
    has_vertex_gpu_quota,
    get_gpu_quota_display_data,  # Helper for infrastructure tree display
)

__all__ = [
    # C3 quotas
    'get_cloud_build_c3_quotas',
    'get_cloud_build_c3_region_quota',
    'has_cloud_build_c3_quota',
    'get_c3_quota_display_data',  # Display helper

    # GPU quotas
    'get_vertex_gpu_quotas',
    'get_all_vertex_gpu_quotas',
    'get_vertex_gpu_quota_metric',
    'has_vertex_gpu_quota',
    'get_gpu_quota_display_data',  # Display helper
]
