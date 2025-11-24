# 🔗 Phase 2: Wrapper Integration Reference

**How monitoring code connects to arr-vertex-launcher wrapper**

---

## 📍 Component Locations

### 1. Runner Wrapper (Prints Logs)
**File**: `training/images/arr-vertex-launcher/entrypoint-wrapper.sh`
**Purpose**: Monitors W&B Launch agent, prints status logs to Cloud Logging

### 2. Monitoring Core (Parses Logs)
**File**: `training/cli/monitor/core.py`
**Purpose**: Fetches Cloud Run logs, extracts error/success messages

---

## 🔄 Data Flow

```
arr-vertex-launcher wrapper (Cloud Run)
    ↓ prints logs to Cloud Logging
    ↓
Cloud Logging API
    ↓ gcloud logging read (fetched by monitoring)
    ↓
training/cli/monitor/core.py
    ↓ parses logs using patterns
    ↓
TUI Display (screen.py)
```

---

## 📊 Success Messages (FINISHED Executions)

### Wrapper Prints "Runs: N"

**Source**: `training/images/arr-vertex-launcher/entrypoint-wrapper.sh`

```bash
# Line 36: Initial startup
echo "📊 Runs: 0"

# Line 59: Final stats (when runner exits)
echo "   • Runs: $JOBS_RUN"

# Line 111: After each job submission
echo "✅ Job submitted to Vertex AI! (Runs: $JOBS_RUN)"

# Line 128: Bailout stats
echo "   • Runs: $JOBS_RUN"

# Line 139: Alive heartbeat (periodic)
echo "[$(date '+%H:%M:%S')] Runner alive: ${LIFETIME}s lifetime, ${IDLE_TIME}s idle, Runs: $JOBS_RUN"
```

### Monitoring Parses "Runs: N"

**Source**: `training/cli/monitor/core.py` function `_fetch_and_extract_success()`

```python
# Lines 614-621: Extract highest "Runs: N" value
for line in lines:
    if 'Runs:' in line:
        runs_str = line.split('Runs:')[1].strip().split()[0]
        runs_count = int(runs_str)
        max_runs = max(max_runs, runs_count)

# Lines 624-626: Format success message
if max_runs > 0:
    return (f"✓ Completed: {max_runs} job{'s' if max_runs != 1 else ''}", max_runs)
```

**Result**: `"✓ Completed: 5 jobs"` (displayed in green in TUI)

---

## ❌ Error Messages (FAILED Executions)

### Wrapper Prints "🚨 FATAL ERROR DETECTED"

**Source**: `training/images/arr-vertex-launcher/entrypoint-wrapper.sh`

```bash
# Line 144: Quota error bailout
echo "🚨 FATAL ERROR DETECTED: Quota exceeded!"

# Line 153: Image pull error bailout
echo "🚨 FATAL ERROR DETECTED: Image pull failed!"

# Line 162: Machine type error bailout
echo "🚨 FATAL ERROR DETECTED: Machine type unsupported!"

# Lines 170, 180, 192, 202, 212, 228, 238: Other fatal errors
echo "🚨 FATAL ERROR DETECTED: [specific error]"

# Line 240: Killing agent
echo "❌ Killing agent - fatal error detected"
```

### Monitoring Parses Bailout Messages

**Source**: `training/cli/monitor/core.py` function `_fetch_and_extract_error()`

```python
# Lines 697-702: Find wrapper bailout marker
for i, line in enumerate(lines):
    if '🚨 FATAL ERROR DETECTED' in line or '❌ Killing agent' in line:
        # Capture 100-line context window (20 before + 80 after)
        start_idx = max(0, i - 20)
        end_idx = min(len(lines), i + 80)
        bailout_lines = lines[start_idx:end_idx]

# Lines 704-752: Extract REAL GCP/W&B/Python error from context
# Uses 20+ error patterns:
# - Machine type incompatibility (GCP)
# - Quota exceeded (GCP)
# - Permission denied (GCP)
# - ImagePullBackOff (K8s)
# - Python exceptions (Traceback)
# - W&B agent errors
# - HTTP error codes (400-503)
```

**Result**: `"❌ QuotaExceeded: Quota 'NVIDIA_L4_GPUS' exceeded"` (displayed in red in TUI)

---

## 🎯 Why This Architecture Works

### 1. Wrapper is the Source of Truth
- Wrapper detects errors in real-time (sitting inside Cloud Run job)
- Prints human-readable messages to logs
- Increments `JOBS_RUN` counter accurately

### 2. Monitoring Parses Wrapper Logs
- Fetches logs via `gcloud logging read`
- Searches for wrapper's bailout markers (`🚨`, `❌`)
- Extracts structured error details from surrounding context
- Finds highest "Runs: N" value for job count

### 3. Terminal State Caching
- Once fetched, messages are remembered forever
- FAILED → error message cached in `_terminal_failures`
- FINISHED → (success_msg, jobs_count) cached in `_terminal_successes`
- No re-fetching on subsequent refreshes!

---

## 📝 Pattern Dependencies

**If wrapper changes these patterns, monitoring MUST update:**

| Wrapper Pattern | Monitoring Matches | Update Required? |
|----------------|-------------------|-----------------|
| `🚨 FATAL ERROR DETECTED` | Line 697 | ✅ YES |
| `❌ Killing agent` | Line 697 | ✅ YES |
| `Runs: N` | Line 614 | ✅ YES |

**Critical**: These patterns are the "contract" between wrapper and monitoring!

---

## 🔍 Example Log Flow

### FINISHED Execution (Success)

```
Wrapper logs (in Cloud Logging):
───────────────────────────────────
🚀 Starting Semi-Persistent W&B Launch Agent...
📊 Runs: 0
✓ W&B agent started (PID: 1234)
⏳ Monitoring for fatal errors...
✅ Job submitted to Vertex AI! (Runs: 1)
✅ Job submitted to Vertex AI! (Runs: 2)
✅ Job submitted to Vertex AI! (Runs: 3)
⏱️  Idle timeout reached after 30 minutes
📊 Final runner stats:
   • Runs: 3
   • Lifetime: 35m 30s
───────────────────────────────────

Monitoring fetches logs:
gcloud logging read "resource.type=cloud_run_job AND ..."

Monitoring parses:
- Finds "Runs: 0" → max_runs = 0
- Finds "Runs: 1" → max_runs = 1
- Finds "Runs: 2" → max_runs = 2
- Finds "Runs: 3" → max_runs = 3
- Returns: ("✓ Completed: 3 jobs", 3)

TUI displays (in green):
✓ Completed: 3 jobs
```

### FAILED Execution (Error)

```
Wrapper logs (in Cloud Logging):
───────────────────────────────────
🚀 Starting Semi-Persistent W&B Launch Agent...
📊 Runs: 0
✓ W&B agent started (PID: 1234)
wandb: ERROR QuotaExceeded: Quota 'NVIDIA_L4_GPUS' exceeded. Limit: 0 in region us-west2
🚨 FATAL ERROR DETECTED: Quota exceeded!
❌ Killing agent - fatal error detected
📊 Final runner stats:
   • Runs: 0
   • Lifetime: 2m 15s
───────────────────────────────────

Monitoring fetches logs:
gcloud logging read "resource.type=cloud_run_job AND ..."

Monitoring parses:
- Finds "🚨 FATAL ERROR DETECTED" at line 50
- Captures context: lines 30-130 (100-line window)
- Searches context for GCP errors
- Finds "QuotaExceeded: Quota 'NVIDIA_L4_GPUS' exceeded"
- Returns: "QuotaExceeded: Quota 'NVIDIA_L4_GPUS' exceeded. Limit: 0 in region us-west2"

TUI displays (in red):
❌ QuotaExceeded: Quota 'NVIDIA_L4_GPUS' exceeded. Limit: 0 in region us-west2
```

---

## ✅ Integration Checklist

- [x] Wrapper prints "Runs: N" → Monitoring extracts job count
- [x] Wrapper prints "🚨 FATAL ERROR DETECTED" → Monitoring finds bailout marker
- [x] Wrapper prints "❌ Killing agent" → Monitoring finds bailout marker
- [x] Wrapper prints GCP errors → Monitoring extracts from context
- [x] Monitoring caches terminal states → No re-fetching!
- [x] Success messages show green → TUI screen.py displays properly
- [x] Error messages show red → TUI screen.py displays properly

---

**Last Updated**: 2025-11-19
**Files Referenced**:
- `training/images/arr-vertex-launcher/entrypoint-wrapper.sh` (wrapper)
- `training/cli/monitor/core.py` (monitoring)
- `training/cli/monitor/screen.py` (TUI display)
