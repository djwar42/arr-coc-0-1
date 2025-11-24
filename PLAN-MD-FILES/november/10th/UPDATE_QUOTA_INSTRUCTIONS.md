# Quota Instructions Update Plan

## Current Locations That Need Updating

1. `training/cli/setup/core.py` lines 773-791 (upgrade to 88/176)
2. `training/cli/setup/core.py` lines 809-827 (upgrade to 88/176)
3. `training/cli/setup/core.py` lines 838-856 (initial request for 44+)
4. Any documentation files (C3_DIAGNOSIS.md, REQUEST_C3_QUOTA_CONSOLE.md)

## What Needs to Change

### OLD Instructions (Wrong - only mentions 1 quota):
```
Filter by:
• Service: Compute Engine API
• Search: C3 CPUs
• Region: us-central1
Select "C3 CPUs"
Request: 44+ CPUs
```

### NEW Instructions (Correct - mentions BOTH quotas):
```
⚠️  TWO separate quotas needed:

QUOTA 1: Compute Engine C3_CPUS (Resource Pool)
1. Clear filters, search: C3_CPUS
2. Select: Compute Engine API / C3_CPUS / us-central1
3. Request: 192 CPUs (max for C3 machines)

QUOTA 2: Cloud Build C3 CPUs (Permission Gate) ← CRITICAL!
1. Clear filters, search: Concurrent C3 Build CPUs Private Pool us-central1
2. Select: Cloud Build API / Concurrent C3 Build CPUs (Private Pool) / us-central1
3. Request: 192 CPUs (matches Compute Engine)

Justification (for both):
"Cloud Build worker pools for PyTorch compilation.
Requesting maximum C3 quota for optimal build performance.
Machine types: c3-highcpu-44 (44 vCPUs), c3-highcpu-88 (88 vCPUs), c3-highcpu-176 (176 vCPUs)."
```

## C3 Machine Type Levels

| Machine Type | vCPUs | Build Time | Notes |
|--------------|-------|------------|-------|
| c3-highcpu-44 | 44 | ~35 min | Default, good |
| c3-highcpu-88 | 88 | ~25 min | 30% faster |
| c3-highcpu-176 | 176 | ~15 min | 2× faster, MAX |
| c3-standard-192 | 192 | ~10 min | Absolute max vCPUs |

## Recommended Request Strategy

**Request Maximum (192) for BOTH quotas:**
- Gives room to upgrade from 44 → 88 → 176 → 192 without new requests
- Single approval covers all future upgrades
- No need to request again later

## Implementation Plan

1. Update setup/core.py lines 773-856 (3 sections)
2. Update any error messages mentioning quota
3. Update documentation files
4. Test the output text
5. Commit with clear message
