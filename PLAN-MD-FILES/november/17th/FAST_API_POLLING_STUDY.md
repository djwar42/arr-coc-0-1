# Fast API Polling Study - Launch Stage Detection

**Date**: 2025-11-17
**Goal**: Replace slow Cloud Logging (3-5 min delay) with fast API polling for launch stage detection

---

## Current Problem

**Cloud Logging delay:**
- Logs take 3-5 minutes to appear in `gcloud logging read`
- User sees "⏳ Will pick up job and submit to Vertex AI (3-5 mins)..." with no updates
- Actual submission happens within 30-60s, but we don't see confirmation until logs propagate

**Result:** Confusing, slow, feels broken

---

## Solution Strategy

**Use fast APIs for SUCCESS detection, fall back to logs for ERROR details:**

```
✅ SUCCESS PATH (FAST - 10-20s total):
   1. Poll W&B API → Detect run appears (job picked up!)
   2. Poll Vertex AI API → Detect job appears (submitted!)
   3. Show success immediately

❌ FAILURE PATH (KEEP EXISTING - wait for detailed logs):
   1. Poll APIs → Detect failure state
   2. Fall back to Cloud Logging streaming
   3. Show full error details (as we do now)
```

**Brilliant insight:** If APIs show success quickly, who cares about logs! If APIs show failure, THEN engage log streaming for details!

---

## API Performance Testing

### W&B API

**Speed:**
```bash
$ time python -c "import wandb; api = wandb.Api(); runs = list(api.runs('newsofpeace2/arr-coc-0-1', filters={'state': 'running'})); print(f'Found {len(runs)} running jobs')"
Found 5 running jobs
1.185 seconds total
```

**What we can detect:**
```python
import wandb
api = wandb.Api()

# Get recent runs
runs = list(api.runs('newsofpeace2/arr-coc-0-1', order='-created_at'))

for r in runs:
    r.id          # Run ID
    r.name        # Display name
    r.state       # "running", "finished", "failed", "crashed"
    r.created_at  # Timestamp (when job picked up)
```

**States available:**
- `running` - Job picked up and executing
- `finished` - Completed successfully
- `failed` - Failed during execution
- `crashed` - Crashed/killed

**ERROR INFO AVAILABLE? ❌ LIMITED**
```python
# Tested on failed runs:
r.state        # "failed" or "crashed" (state only, no details)
r.exitcode     # N/A (not available via API)
r.summary      # Minimal info (just {'_wandb'})
r.history      # Empty for failed runs
```

**Conclusion:** W&B API can detect SUCCESS (running state) instantly, but NO detailed error messages!

---

### Vertex AI API

**Speed:**
```bash
$ time gcloud ai custom-jobs list --region=us-central1 --limit=1 --format="value(name.basename())"
2161270372761075712
1.096 seconds total
```

**What we can detect:**
```bash
$ gcloud ai custom-jobs list --region=us-central1 --limit=3 --format="table(name.basename(),displayName,state,createTime)"

NAME                 DISPLAY_NAME             STATE             CREATE_TIME
2161270372761075712  major-wind-1762750552    JOB_STATE_FAILED  2025-11-10T04:58:03Z
2339162558042210304  derpy-fog-1762749655     JOB_STATE_FAILED  2025-11-10T04:47:52Z
2557938983690567680  wild-horizon-1762743045  JOB_STATE_FAILED  2025-11-10T04:00:28Z
```

**States available:**
- `JOB_STATE_QUEUED` - Submitted, waiting for resources
- `JOB_STATE_PENDING` - Resources allocated, starting
- `JOB_STATE_RUNNING` - Training in progress
- `JOB_STATE_SUCCEEDED` - Completed successfully
- `JOB_STATE_FAILED` - Failed
- `JOB_STATE_CANCELLED` - Cancelled by user

**ERROR INFO AVAILABLE? ✅ YES! FULL DETAILS!**

```bash
$ time gcloud ai custom-jobs describe 2161270372761075712 --region=us-central1 --format="yaml(state,error)"
error:
  code: 8
  message: 'Replicas low on disk: workerpool0. Specify a larger bootDiskSizeGb https://cloud.google.com/vertex-ai/docs/reference/rest/v1/DiskSpec
    and try again.'
state: JOB_STATE_FAILED

1.309 seconds total
```

**Error details include:**
- `error.code` - Numeric error code
- `error.message` - FULL human-readable error message
- Specific guidance (e.g., "Specify a larger bootDiskSizeGb")

**Conclusion:** Vertex AI API can detect BOTH success AND failure with full error details in ~1.3s!

---

## Proposed Implementation

### Fast Path: Success Detection (10-20s total)

```python
def _stream_execution_logs_with_fast_polling(config, region, job_name, execution_name, status):
    """
    Enhanced log streaming with fast API polling for success detection.

    SUCCESS PATH (FAST):
    - Poll W&B API every 5s → Detect run appears
    - Poll Vertex AI API every 5s → Detect job appears with state=SUCCEEDED
    - Show stages immediately, return success

    FAILURE PATH (EXISTING):
    - APIs detect failure → Fall back to log streaming
    - Show full error details from logs (as we do now)
    """

    # Initialize
    last_wandb_check = 0
    last_vertex_check = 0
    shown_job_pickup = False
    shown_vertex_submission = False

    # Track baseline run count (to detect new runs)
    api = wandb.Api()
    initial_runs = list(api.runs("newsofpeace2/arr-coc-0-1", filters={"state": "running"}))
    baseline_count = len(initial_runs)

    while not execution_done and (time.time() - start_time) < agent_timeout:
        # STILL stream logs (for wrapper errors, startup issues)
        stream_logs()

        # Poll W&B API every 5 seconds (FAST JOB PICKUP DETECTION!)
        if not shown_job_pickup and (time.time() - last_wandb_check) > 5:
            try:
                current_runs = list(api.runs("newsofpeace2/arr-coc-0-1", filters={"state": "running"}))
                if len(current_runs) > baseline_count:
                    # New run appeared! Job picked up!
                    status("   [yellow]⚡ Job picked up from queue![/yellow]")
                    shown_job_pickup = True
                    baseline_count = len(current_runs)
            except Exception:
                pass  # Ignore API errors, keep trying
            last_wandb_check = time.time()

        # Poll Vertex AI API every 5 seconds (FAST SUBMISSION DETECTION!)
        if not shown_vertex_submission and (time.time() - last_vertex_check) > 5:
            try:
                # List latest job
                result = subprocess.run(
                    ["gcloud", "ai", "custom-jobs", "list",
                     "--region=us-central1", "--limit=1",
                     "--format=value(name.basename(),state)"],
                    capture_output=True, text=True, timeout=5
                )

                if result.returncode == 0 and result.stdout.strip():
                    job_id, state = result.stdout.strip().split()

                    if state == "JOB_STATE_SUCCEEDED":
                        # SUCCESS! Show immediately!
                        status("   [yellow]⚡ Submitting to Vertex AI...[/yellow]")
                        status("")
                        status("[bold cyan]✓ Job Invocation Submitted to the Cloud! ✓[/bold cyan]")
                        # [... ASCII art ...]
                        return True, "Job submitted successfully"

                    elif state == "JOB_STATE_FAILED":
                        # FAILURE! Get error details from API (FAST!)
                        error_result = subprocess.run(
                            ["gcloud", "ai", "custom-jobs", "describe", job_id,
                             "--region=us-central1", "--format=value(error.message)"],
                            capture_output=True, text=True, timeout=5
                        )

                        if error_result.returncode == 0:
                            error_msg = error_result.stdout.strip()
                            status(f"\n[red]❌ FATAL ERROR: Vertex AI job failed![/red]")
                            status(f"[dim]  {error_msg}[/dim]")
                            return False, error_msg
                        else:
                            # Can't get error from API, fall back to log streaming
                            status("\n[red]❌ Job failed, checking logs for details...[/red]")
                            # Continue log streaming to get error details

                    elif state in ["JOB_STATE_RUNNING", "JOB_STATE_PENDING", "JOB_STATE_QUEUED"]:
                        # Job submitted and starting!
                        if not shown_vertex_submission:
                            status("   [yellow]⚡ Submitting to Vertex AI...[/yellow]")
                            shown_vertex_submission = True

            except Exception:
                pass  # Ignore API errors, keep trying

            last_vertex_check = time.time()

        sleep(1)
```

---

## Comparison: Before vs After

### BEFORE (Current - Slow Log Streaming)

```
⏳ Will pick up job and submit to Vertex AI (3-5 mins)...

[3-5 minute silence]

✓ Job Invocation Submitted to the Cloud! ✓
```

**Total time:** 3-5 minutes
**User experience:** Confusing, feels stuck

---

### AFTER (Fast API Polling)

```
⏳ Will pick up job and submit to Vertex AI (3-5 mins)...

[5 seconds - W&B API poll]
⚡ Job picked up from queue!

[10 seconds - Vertex AI API poll]
⚡ Submitting to Vertex AI...

[15 seconds - Vertex AI API poll confirms]
✓ Job Invocation Submitted to the Cloud! ✓
```

**Total time:** 10-20 seconds
**User experience:** Fast, informative, reassuring

---

## Error Handling Strategy

### Scenario 1: Wrapper Errors (Agent startup fails)

**Detection:** Cloud Logging (keep existing code)
**Why:** These errors happen before any W&B/Vertex AI interaction
**Examples:** Permission denied, bucket creation failed, sitecustomize.py issues

**Flow:**
```
Agent starts → Wrapper error → Logs show error immediately → We catch it
```

---

### Scenario 2: Vertex AI Submission Errors (Job creation fails)

**Detection:** Vertex AI API (FAST - 1.3s!)
**Why:** API gives full error details immediately

**Flow:**
```
Poll Vertex AI → Job state = FAILED → Get error.message → Show immediately
```

**Example errors we can detect:**
- Machine type not supported
- Quota exceeded
- Disk size too small
- Permission denied
- Region mismatch

---

### Scenario 3: Training Runtime Errors (Job starts but fails)

**Detection:** W&B API + Vertex AI API
**Why:** Both APIs show failure state

**Flow:**
```
Poll APIs → state = "failed" or "crashed" → Job failed during training
```

**Note:** For runtime errors, logs may be more informative (check training logs, not wrapper logs)

---

## Fallback Strategy

**If APIs fail or timeout:**
- Keep log streaming as fallback
- Don't break if APIs are unreachable
- APIs are for SPEED, logs are for SAFETY

**Try/except around all API calls:**
```python
try:
    # Fast API check
    check_wandb_api()
except Exception:
    pass  # Keep going, logs will catch errors
```

---

## Benefits

✅ **10-15× FASTER** success detection (20s vs 3-5min)
✅ **Same error detail** (Vertex AI API has full messages)
✅ **More stages shown** (pickup, download, submit)
✅ **Better UX** (user sees progress, not silence)
✅ **No breaking changes** (keep logs as fallback)

---

## Risks

⚠️ **API rate limits** - W&B/GCP might throttle if we poll too aggressively
   **Mitigation:** Poll every 5-10s (not too fast)

⚠️ **API outages** - APIs might be unavailable
   **Mitigation:** Keep log streaming as fallback

⚠️ **Credential issues** - APIs need auth
   **Mitigation:** Already authenticated (same creds as gcloud/wandb CLI)

---

## Implementation Plan

### Phase 1: Add Vertex AI polling (Highest value!)
- Poll `gcloud ai custom-jobs list` every 5s
- Detect job appears → Show "⚡ Submitting to Vertex AI"
- Detect state=FAILED → Get error.message and show immediately

### Phase 2: Add W&B polling
- Poll `wandb.Api().runs()` every 5s
- Detect new run → Show "⚡ Job picked up from queue"

### Phase 3: Remove "3-5 mins" wait time estimate
- Replace with "10-20s" (more accurate with API polling)

---

## Code Location

**File to modify:** `training/cli/launch/core.py`
**Function:** `_stream_execution_logs()` (line ~3963)

**Changes:**
1. Add W&B API polling loop (every 5s)
2. Add Vertex AI API polling loop (every 5s)
3. Keep existing log streaming (for errors/fallback)
4. Add stage detection from APIs (not logs)

---

## Testing

**Success case:**
```bash
python training/cli.py launch
# Should show stages within 20s:
#   ⚡ Job picked up (~5s)
#   ⚡ Submitting to Vertex AI (~10s)
#   ✓ Job submitted! (~15s)
```

**Failure case:**
```bash
# Trigger Vertex AI error (e.g., machine type not supported)
# Should show error within 5-10s with full details from API
```

**Fallback case:**
```bash
# Disconnect network after agent starts
# Should fall back to log streaming (slower but works)
```

---

## Conclusion

**FAST API POLLING IS THE WAY! 🚀**

- ✅ Vertex AI API has FULL error details (better than logs!)
- ✅ W&B API detects job pickup instantly
- ✅ 10-15× faster than Cloud Logging
- ✅ Keep logs as safety fallback
- ✅ No breaking changes

**Recommendation:** Implement Phase 1 (Vertex AI polling) immediately!
