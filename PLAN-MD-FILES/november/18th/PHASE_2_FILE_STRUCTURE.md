# 📁 PHASE 2: File Structure & Function Names

**Where everything goes and what it's called!**

---

## 📂 **File Tree**

```
arr-coc-0-1/
├── training/
│   ├── cli/
│   │   ├── monitor/
│   │   │   ├── core.py ← ⭐ MAIN FILE - ALL Phase 2 code goes here!
│   │   │   ├── screen.py (no changes)
│   │   │   └── screen_old_good_file.py (ignore)
│   │   │
│   │   ├── launch/
│   │   │   └── core.py (no changes)
│   │   │
│   │   └── cli.py (no changes)
│   │
│   └── images/
│       └── arr-vertex-launcher/
│           └── entrypoint-wrapper.sh ← Referenced in inline comments (source of OUR patterns)
│
└── PHASE_2_REMEMBER_FAILED_ERRORS.md ← The plan (this file!)
└── PHASE_2_ERROR_PATTERN_INVESTIGATION.md ← Pattern documentation
└── PHASE_2_FILE_STRUCTURE.md ← This file!
```

---

## 📝 **training/cli/monitor/core.py Structure**

```python
"""
Monitor CLI - Core functionality for runner/execution monitoring
"""

# ═══════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════
import subprocess
from datetime import datetime
from typing import List, Dict, Optional
# ... other imports ...


# ═══════════════════════════════════════════════════════════════
# ⭐ PHASE 2: TERMINAL STATE MEMORY (MODULE-LEVEL)
# ═══════════════════════════════════════════════════════════════

# Remember error messages for FAILED executions (never change!)
_terminal_failures = {}  # exec_name → error message

# Remember success messages for FINISHED executions (never change!)
_terminal_successes = {}  # exec_name → success message (usually "—")


# ═══════════════════════════════════════════════════════════════
# EXISTING FUNCTIONS (no changes)
# ═══════════════════════════════════════════════════════════════

def run_gcloud_with_retry(...):
    """Execute gcloud commands with retry logic"""
    # Existing code - no changes!
    pass


def _parse_execution_status(...):
    """Parse execution status from GCP metadata"""
    # Existing code - no changes!
    pass


# ═══════════════════════════════════════════════════════════════
# ⭐ PHASE 2: NEW FUNCTION - ERROR EXTRACTION
# ═══════════════════════════════════════════════════════════════

def _fetch_and_extract_error(exec_name: str) -> str:
    """
    Fetch logs for a FAILED execution and extract concise error message.

    Uses ALL our complex error matching patterns (20+ patterns):
    - Wrapper bailout detection (🚨 FATAL ERROR, ❌ Killing agent)
    - Machine type errors
    - Permission errors (InvalidArgument, PermissionDenied, NotFound)
    - Quota errors (QuotaExceeded, ResourceExhausted)
    - Service errors (500, 503)
    - HTTP errors (400-503)
    - Container errors (ImagePullBackOff, ErrImagePull)
    - Python exceptions (Traceback, Exception, Error)
    - W&B agent errors (wandb: ERROR)
    - Info filtering (⏱️ ⏳ ℹ️ 🔍, "monitoring for")

    Args:
        exec_name: Cloud Run execution name (e.g., 'vertex-ai-launcher-f4hfv-0001')

    Returns:
        Concise error message for table display (max 200 chars)

    Example:
        >>> _fetch_and_extract_error('vertex-ai-launcher-f4hfv-0001')
        '❌ QuotaExceeded: Quota 'NVIDIA_L4_GPUS' exceeded. Limit: 0 in us-west2'
    """
    try:
        # ───────────────────────────────────────────────────────────
        # Step 1: Fetch logs from Cloud Logging
        # ───────────────────────────────────────────────────────────
        log_result = run_gcloud_with_retry(
            [
                "gcloud", "logging", "read",
                f'resource.type=cloud_run_job AND resource.labels.job_name=vertex-ai-launcher AND labels."run.googleapis.com/execution_name"={exec_name}',
                "--limit=300",
                "--format=value(textPayload)",
                "--project=weight-and-biases-476906",
            ],
            max_retries=1,
            timeout=10,
            operation_name="fetch Cloud Run job logs",
        )

        if log_result.returncode != 0 or not log_result.stdout:
            return "Error fetching logs"

        lines = log_result.stdout.strip().split('\n')

        # ───────────────────────────────────────────────────────────
        # Step 2: Extract error using ALL our complex patterns!
        # ───────────────────────────────────────────────────────────

        # ⭐ Priority 1: Wrapper bailout detection
        # ✅ MATCHING OUR OWN STUFF HERE: entrypoint-wrapper.sh prints these!
        # Source: training/images/arr-vertex-launcher/entrypoint-wrapper.sh lines 144-240
        # What: Wrapper detects fatal errors and prints bailout messages
        # Why: Fast bailout - we capture context around these markers
        for i, line in enumerate(lines):
            if '🚨 FATAL ERROR DETECTED' in line or '❌ Killing agent' in line:
                # Capture context (20 before + 80 after)
                start_idx = max(0, i - 20)
                end_idx = min(len(lines), i + 80)
                bailout_lines = lines[start_idx:end_idx]

                # Search for specific errors in context
                # ❌ NOT our output - REAL GCP/W&B/Python errors!
                for ctx_line in bailout_lines:
                    # ❌ GCP: Machine type incompatibility
                    # Source: GCP API (external)
                    # Example: "InvalidArgument: 400 Machine type 'n2-standard-4' is not supported for GPU 'NVIDIA_TESLA_T4'"
                    if 'Machine type' in ctx_line and 'is not supported' in ctx_line:
                        return ctx_line.split('wandb: ERROR')[-1].strip() if 'wandb: ERROR' in ctx_line else ctx_line.strip()

                    # ❌ GCP: Machine type (alternate phrasing)
                    # Source: GCP API (external)
                    # Example: "GPU 'NVIDIA_L4' is not supported for machine type 'n1-standard-4'"
                    elif 'is not supported for machine type' in ctx_line:
                        return ctx_line.split('wandb: ERROR')[-1].strip() if 'wandb: ERROR' in ctx_line else ctx_line.strip()

                    # ❌ GCP: Invalid args, permissions, missing resources
                    # Source: GCP API (external)
                    # Examples: "InvalidArgument: 400", "PermissionDenied: 403", "NotFound: 404"
                    elif 'InvalidArgument:' in ctx_line or 'PermissionDenied:' in ctx_line or 'NotFound:' in ctx_line:
                        return ctx_line.split('wandb: ERROR')[-1].strip() if 'wandb: ERROR' in ctx_line else ctx_line.strip()

                    # ❌ GCP: Quota exceeded
                    # Source: GCP Quota System (external)
                    # Example: "QuotaExceeded: Quota 'NVIDIA_L4_GPUS' exceeded. Limit: 0"
                    elif 'QuotaExceeded' in ctx_line or 'ResourceExhausted' in ctx_line:
                        return ctx_line.split('wandb: ERROR')[-1].strip() if 'wandb: ERROR' in ctx_line else ctx_line.strip()

                    # ❌ GCP: Service errors (500/503)
                    # Source: GCP APIs (external)
                    # Examples: "503 Service Unavailable", "500 Internal Error"
                    elif any(pattern in ctx_line for pattern in ['503', 'ServiceUnavailable', '500', 'Internal Error', 'Internal error']):
                        return ctx_line.split('wandb: ERROR')[-1].strip() if 'wandb: ERROR' in ctx_line else ctx_line.strip()

                    # ❌ GCP: HTTP error codes
                    # Source: GCP APIs (external)
                    # Example: "HttpError: <HttpError 429 ... 'Too Many Requests'>"
                    elif 'HttpError' in ctx_line and any(code in ctx_line for code in ['400', '401', '403', '404', '429', '500', '502', '503']):
                        return ctx_line.split('wandb: ERROR')[-1].strip() if 'wandb: ERROR' in ctx_line else ctx_line.strip()

                    # ❌ K8s/GCP: Image pull failures
                    # Source: Kubernetes/GCP (external)
                    # Examples: "ImagePullBackOff", "ErrImagePull"
                    elif 'ImagePullBackOff' in ctx_line or 'ErrImagePull' in ctx_line:
                        return ctx_line.split('wandb: ERROR')[-1].strip() if 'wandb: ERROR' in ctx_line else ctx_line.strip()

                    # ❌ Python: Exceptions
                    # Source: Python interpreter (external)
                    # Example: "Traceback (most recent call last)..."
                    elif 'Traceback' in ctx_line or 'Exception:' in ctx_line or 'Error:' in ctx_line:
                        return ctx_line.split('wandb: ERROR')[-1].strip() if 'wandb: ERROR' in ctx_line else ctx_line.strip()

                # Fallback: return wrapper's generic message
                # ✅ Our wrapper message if no specific error found
                return line.replace('🚨 FATAL ERROR DETECTED:', '').replace('❌', '').strip()

        # ⭐ Priority 2: W&B agent errors
        # ❌ NOT our output - W&B Launch agent's error prefix!
        # Source: W&B Launch agent (external)
        # What: W&B prefixes errors with "wandb: ERROR", often re-logging GCP errors
        for i, line in enumerate(lines):
            if 'wandb: ERROR' in line:
                if 'Machine type' in line and 'is not supported' in line:
                    return line.split('wandb: ERROR')[-1].strip()
                elif 'InvalidArgument' in line or 'PermissionDenied' in line or 'NotFound' in line:
                    return line.split('wandb: ERROR')[-1].strip()

        # ⭐ Priority 3: Generic error patterns (skip INFO messages)
        for line in lines:
            if any(keyword in line.lower() for keyword in ['error', 'exception', 'failed', 'notfound', '404', 'timeout']):
                # ✅ MATCHING OUR OWN STUFF HERE: Skip our info emojis!
                # Source: training/images/arr-vertex-launcher/entrypoint-wrapper.sh lines 35, 51
                # What: We print "⏱️ Idle timeout...", "⏳ Monitoring for..." - NOT errors!
                # Why: Filter false positives
                if any(info_indicator in line for info_indicator in ['⏱️', '⏳', 'ℹ️', '🔍']):
                    continue

                # ✅ MATCHING OUR OWN STUFF HERE: Skip our status messages!
                # Source: training/images/arr-vertex-launcher/entrypoint-wrapper.sh line 51
                # What: "⏳ Monitoring for fatal errors..." - NOT an error!
                # Why: Filter false positives
                if any(info_pattern in line.lower() for info_pattern in ['monitoring for', 'checking for', 'watching for', 'looking for']):
                    continue

                # ❌ Generic error from logs
                return line[:200]  # Truncate for table display

        # No error found
        return "Unknown error"

    except Exception as e:
        return f"Error parsing logs: {str(e)}"


# ═══════════════════════════════════════════════════════════════
# ⭐ PHASE 2: MODIFIED FUNCTION - ADD REMEMBERING LOGIC
# ═══════════════════════════════════════════════════════════════

def _fetch_runner_executions_all_regions(
    project_id: str,
    regions: List[str]
) -> List[Dict]:
    """
    Fetch runner executions from all regions with terminal state remembering.

    Auto-refreshes every 30s to check execution statuses:
    - RUNNING: Show "Running..." (check again next refresh)
    - FAILED: Fetch error once, remember forever!
    - FINISHED: Remember "—" forever!

    Terminal states (FAILED/FINISHED) are remembered in module-level dicts:
    - _terminal_failures: error messages
    - _terminal_successes: success messages

    Returns:
        List of top 5 newest executions with status + error/success messages
    """
    # ───────────────────────────────────────────────────────────
    # Step 1: Fetch metadata from all 18 regions (existing code)
    # ───────────────────────────────────────────────────────────
    all_execs = []

    for region in regions:
        # ... existing metadata fetch code (no changes!) ...
        result = run_gcloud_with_retry([
            "gcloud", "run", "jobs", "executions", "list",
            "--job=vertex-ai-launcher",
            f"--region={region}",
            "--format=json",
            # ... etc ...
        ])
        # ... parse and append to all_execs ...

    # ───────────────────────────────────────────────────────────
    # Step 2: Sort and get top 5 newest (existing code)
    # ───────────────────────────────────────────────────────────
    top_5 = sorted(all_execs, key=lambda x: x['created_at'], reverse=True)[:5]

    # ───────────────────────────────────────────────────────────
    # ⭐ Step 3: SMART LOG FETCHING - Only for NEW FAILED!
    # ───────────────────────────────────────────────────────────
    for exec in top_5:
        exec_name = exec['name']
        status = exec['status']  # RUNNING, FAILED, or FINISHED (from metadata)

        if status == "FAILED":
            # Terminal FAILED! Check if we already fetched error
            if exec_name in _terminal_failures:
                # Already fetched before - use remembered error!
                exec['error'] = _terminal_failures[exec_name]
            else:
                # New FAILED execution - fetch logs ONCE and remember!
                error_msg = _fetch_and_extract_error(exec_name)
                _terminal_failures[exec_name] = error_msg  # Remember forever!
                exec['error'] = error_msg

        elif status == "FINISHED":
            # Terminal FINISHED! Remember success message
            if exec_name in _terminal_successes:
                # Already know it succeeded
                exec['error'] = _terminal_successes[exec_name]
            else:
                # New FINISHED execution - remember success!
                _terminal_successes[exec_name] = "—"  # No log fetch needed!
                exec['error'] = "—"

        elif status == "RUNNING":
            # NOT terminal! Keep showing "Running..." and check again next refresh (30s)
            exec['error'] = "Running..."  # No log fetch, but we'll check status again in 30s!

    return top_5


# ═══════════════════════════════════════════════════════════════
# OTHER EXISTING FUNCTIONS (no changes)
# ═══════════════════════════════════════════════════════════════

def get_runner_stats(...):
    """Get runner statistics"""
    # Existing code - no changes!
    pass

# ... etc ...
```

---

## 🎯 **Summary**

### **File Modified:**
- `training/cli/monitor/core.py`

### **Module-Level Variables Added:**
- `_terminal_failures = {}` - Remembers FAILED error messages
- `_terminal_successes = {}` - Remembers FINISHED success messages

### **Functions Added:**
- `_fetch_and_extract_error(exec_name: str) -> str` - NEW! Fetches logs and extracts error

### **Functions Modified:**
- `_fetch_runner_executions_all_regions()` - MODIFIED! Adds remembering logic for terminal states

### **Total Lines:**
- ~150 lines of new code (vs 437 original adaptive code)
- All complex error matching preserved (20+ patterns)
- Every pattern has full inline comments!

---

## 🔗 **References**

All our error patterns come from:
- `training/images/arr-vertex-launcher/entrypoint-wrapper.sh` (lines 35, 51, 144-240)

Full pattern documentation:
- `PHASE_2_ERROR_PATTERN_INVESTIGATION.md` (416 lines)

Implementation plan:
- `PHASE_2_REMEMBER_FAILED_ERRORS.md` (640+ lines)

---

**ONE FILE, THREE CHANGES! Simple and clean! 🎯**
