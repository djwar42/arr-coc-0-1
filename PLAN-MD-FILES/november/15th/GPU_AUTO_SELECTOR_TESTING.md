# GPU Auto-Selector Testing Methodology

**Date**: 2025-11-16
**Feature**: GPU machine type auto-selection + validation
**Test Champion**: CHAMPION #1: THE-AUTO-SELECTOR-PRIME

---

## Overview

This document describes the complete testing methodology for verifying the GPU auto-selection system works correctly. The system automatically selects the optimal machine type based on GPU choice and validates compatibility before job submission.

**What we're testing:**
- User specifies GPU only (NVIDIA_TESLA_T4)
- System auto-selects machine type (n1-standard-4)
- Validation confirms compatibility
- Vertex AI job creates with correct machine+GPU combo

**Success criteria:**
- No manual machine type selection needed
- No "Machine type not supported" errors
- Vertex AI job runs with T4 GPU on N1 machine

---

## Pre-Launch Setup

### 1. Verify .training Configuration

```bash
cd RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1
cat .training | grep -E "MACHINE_TYPE|ACCELERATOR"
```

**Expected configuration:**
```bash
# WANDB_LAUNCH_MACHINE_TYPE=""  # ← Should be commented/blank
WANDB_LAUNCH_ACCELERATOR_TYPE="NVIDIA_TESLA_T4"
WANDB_LAUNCH_ACCELERATOR_COUNT="1"
```

**✅ Correct**: Machine type blank/commented, GPU type set
**❌ Wrong**: Machine type manually specified (defeats auto-selection)

### 2. Verify Code Changes Are Present

```bash
# Check auto-selection function exists
grep -n "def get_best_gpu" training/cli/shared/machine_selection.py

# Check validation function exists
grep -n "def validate_gpu_machine_compatibility" training/cli/shared/machine_selection.py

# Check constants.py loads config with auto-selection
grep -n "get_best_gpu" training/cli/constants.py

# Check validation.py uses new validation
grep -n "validate_gpu_machine_compatibility" training/cli/launch/validation.py
```

**All should return line numbers** (not empty)

### 3. Check Current Vertex AI Jobs

```bash
# Clear any stuck jobs from previous launches
gcloud ai custom-jobs list --region=us-central1 --limit=5 --format="table(displayName,state,createTime)"
```

**Note**: Record job count BEFORE launch to verify new job appears

---

## Launch Command

### Terminal 1: Launch CHAMPION #1

```bash
cd RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1

# Cancel any existing launches
pkill -f "python -u training/cli.py launch"

# Launch with champion banner + log capture
cat << 'EOF' | python -u training/cli.py launch 2>&1 | tee /tmp/champion-1-launch.log
💥 CHAMPION #1: THE-AUTO-SELECTOR-PRIME! 💥

              🤖 FIRST TEST OF AUTO-SELECTION 🤖
             ═════════════════════════════════════
              GPU ONLY → MACHINE AUTO-SELECTED!
             ═════════════════════════════════════

EOF
```

**Log saved to**: `/tmp/champion-1-launch.log` (for analysis later)

---

## Monitoring Stations (5 Parallel Checks)

### Station 1: Launch Output (Terminal 1 - Immediate)

**Timeline**: 0-5 minutes

**Watch for (in order):**

#### Check 1.1: Auto-Selection Message (5-10 seconds in)
```
🤖 Auto-selected machine type: n1-standard-4 (for GPU: NVIDIA_TESLA_T4)
```

**✅ SUCCESS**: Message appears with correct machine type
**❌ FAILURE**: No message → constants.py didn't trigger
**Debug**: Check .training file has blank WANDB_LAUNCH_MACHINE_TYPE

---

#### Check 1.2: Validation Output (10-15 seconds in)
```
⏳ Validating launch configuration...
✓ GPU type valid: NVIDIA_TESLA_T4
✓ GPU count valid: 1
✓ Machine type valid: n1-standard-4
✓ GPU+Machine compatibility: ✅ n1-standard-4 supports T4
✓ Validation passed
```

**✅ SUCCESS**: All ✓ checks pass, especially GPU+Machine compatibility
**❌ FAILURE**: Compatibility check fails → validation.py bug
**Debug**: Check validation.py line 70-90 for validate_gpu_machine_compatibility() call

---

#### Check 1.3: Launch Summary (2-4 minutes in, after builds)
```
🚀 Launching training job to Vertex AI...
   Machine: n1-standard-4
   GPU: 1x NVIDIA_TESLA_T4
   Region: us-central1
```

**✅ SUCCESS**: Machine type shows n1-standard-4 (not blank/None)
**❌ FAILURE**: Machine type missing → config didn't propagate
**Debug**: Check constants.py line 74-124 for config injection

---

#### Check 1.4: W&B Queue Submission (4-5 minutes in)
```
✅ Job queued to W&B Launch!
   Queue: arr-coc-queue
   Run ID: abc123xyz
```

**✅ SUCCESS**: Job queued successfully
**❌ FAILURE**: Queue submission error → W&B config issue
**Debug**: Not related to GPU auto-selection

---

### Station 2: Cloud Run Launcher (Terminal 2 - 5-7 minutes)

**Timeline**: 30 seconds - 2 minutes after W&B queue submission

```bash
# Terminal 2: Watch Cloud Run executions
watch -n 5 'gcloud run jobs executions list \
  --job=vertex-ai-launcher \
  --region=us-central1 \
  --limit=3 \
  --format="table(name,status,startTime)"'
```

**Or stream logs live:**
```bash
# Get latest execution name
LATEST_EXEC=$(gcloud run jobs executions list \
  --job=vertex-ai-launcher \
  --region=us-central1 \
  --limit=1 \
  --format="value(name)")

# Stream logs
gcloud logging read \
  "resource.type=cloud_run_job \
   resource.labels.job_name=vertex-ai-launcher \
   labels.\"run.googleapis.com/execution_name\"=$LATEST_EXEC" \
  --limit=100 \
  --format="table(timestamp,textPayload)" \
  --freshness=5m
```

**Watch for:**

#### Check 2.1: Agent Startup
```
🚀 Starting W&B Launch Agent with fatal error detection...
✓ W&B agent started (PID: 7)
⏳ Monitoring for fatal errors...
```

**✅ SUCCESS**: Agent starts monitoring
**❌ FAILURE**: Agent crashes on startup → W&B auth issue

---

#### Check 2.2: Job Pickup
```
Connected to W&B queue: arr-coc-queue
Picked up job: abc123xyz
Downloading training code from W&B...
```

**✅ SUCCESS**: Agent picks up our job
**❌ FAILURE**: No job pickup → queue connection issue

---

#### Check 2.3: Vertex AI Submission (CRITICAL!)
```
Submitting to Vertex AI...
  Project: arr-coc-ovis
  Region: us-central1
  Machine: n1-standard-4          ← VERIFY THIS!
  GPU: NVIDIA_TESLA_T4 (count: 1) ← VERIFY THIS!
  Image: us-central1-docker.pkg.dev/.../arr-trainer:latest
```

**✅ SUCCESS**: Shows n1-standard-4 + NVIDIA_TESLA_T4
**❌ FAILURE**: Shows wrong machine type → config bug

---

#### Check 2.4: Vertex AI Job ID
```
✅ Submitted to Vertex AI!
   Job ID: 1234567890123456789
   View: https://console.cloud.google.com/vertex-ai/...
```

**✅ SUCCESS**: Job ID returned (Vertex accepted submission!)
**❌ FAILURE**: Error message appears → see failure scenarios below

---

#### Check 2.5: Wrapper Bailout (Should NOT happen!)
```
🚨 FATAL ERROR DETECTED: Machine type not supported!
━━━ Error Context (10 lines before error) ━━━
[error details]
❌ Killing agent (PID: 7) - this error will not self-resolve
```

**✅ SUCCESS**: This message NEVER appears
**❌ FAILURE**: Wrapper caught error → validation MISSED invalid combo!
**Debug**: Update validation.py to catch this GPU+Machine pair

---

### Station 3: Vertex AI Job Creation (Terminal 3 - 7-10 minutes)

**Timeline**: 2-3 minutes after Cloud Run submission

```bash
# Terminal 3: Watch Vertex AI jobs appear
watch -n 5 'gcloud ai custom-jobs list \
  --region=us-central1 \
  --limit=5 \
  --format="table(name.basename(),displayName,state,createTime)"'
```

**Watch for:**

#### Check 3.1: Job Appears in List
```
NAME              DISPLAY_NAME                    STATE              CREATE_TIME
1234567890123     arr-coc-training-20251116-...   JOB_STATE_QUEUED   2025-11-16T...
```

**✅ SUCCESS**: New job appears at TOP of list
**❌ FAILURE**: No new job → submission failed (check Cloud Run logs)

---

#### Check 3.2: Job State Progression
```
QUEUED → PENDING → RUNNING
```

**Timeline**:
- QUEUED: Immediate (0-30 seconds)
- PENDING: 30 seconds - 2 minutes (allocating resources)
- RUNNING: 2-5 minutes (container starting)

**✅ SUCCESS**: State progresses smoothly
**❌ FAILURE**: Stuck in PENDING > 5 minutes → quota/resource issue

---

#### Check 3.3: Job Spec Verification (CRITICAL!)
```bash
# Get latest job ID
LATEST_JOB=$(gcloud ai custom-jobs list \
  --region=us-central1 \
  --limit=1 \
  --format="value(name)")

# Show complete job spec
gcloud ai custom-jobs describe $LATEST_JOB \
  --region=us-central1 \
  --format="yaml(jobSpec.workerPoolSpecs)"
```

**Expected output:**
```yaml
workerPoolSpecs:
- machineSpec:
    machineType: n1-standard-4          ← VERIFY!
    acceleratorType: NVIDIA_TESLA_T4    ← VERIFY!
    acceleratorCount: 1                 ← VERIFY!
  replicaCount: 1
  containerSpec:
    imageUri: us-central1-docker.pkg.dev/.../arr-trainer:latest
```

**✅ SUCCESS**: All 3 values match (n1-standard-4, NVIDIA_TESLA_T4, count=1)
**❌ FAILURE**: Wrong machine type → auto-selection bug
**❌ FAILURE**: Wrong GPU type → config propagation bug
**❌ FAILURE**: Wrong count → config parsing bug

---

### Station 4: Vertex AI Job Logs (Terminal 4 - 10+ minutes)

**Timeline**: 3-5 minutes after job enters RUNNING state

```bash
# Terminal 4: Stream live job logs
gcloud ai custom-jobs stream-logs $LATEST_JOB --region=us-central1
```

**Watch for (first 2 minutes of logs):**

#### Check 4.1: Container Startup
```
Pulling image: us-central1-docker.pkg.dev/.../arr-trainer:latest
Successfully pulled image
Starting container...
```

**✅ SUCCESS**: Image pulls and container starts
**❌ FAILURE**: Image pull error → registry permissions

---

#### Check 4.2: GPU Detection (CRITICAL PROOF!)
```
Detected GPU: Tesla T4
GPU Memory: 15360 MiB
CUDA Version: 12.x
cuDNN Version: 8.x
```

**✅ SUCCESS**: Detects Tesla T4 GPU (PROOF machine+GPU combo worked!)
**❌ FAILURE**: "No GPU detected" → wrong machine family (validation bug!)
**❌ FAILURE**: "CUDA device not available" → GPU not attached (GCP bug or wrong machine)

---

#### Check 4.3: Training Initialization
```
Initializing W&B run...
Loading ARR-COC model...
W&B run: https://wandb.ai/newsofpeace2/arr-coc-0-1/runs/xyz123
Starting training loop...
```

**✅ SUCCESS**: Training begins
**❌ FAILURE**: Model loading error → not GPU-related

---

#### Check 4.4: Training Progress
```
Epoch 1/10: loss=0.523, acc=0.712, gpu_util=87%
```

**✅ SUCCESS**: Training progresses, GPU utilized
**❌ FAILURE**: Training crashes → model/data issue (not GPU-related)

---

### Station 5: CLI Monitor (Terminal 5 - Continuous)

**Timeline**: Parallel with all above checks

```bash
# Terminal 5: Run monitor
cd RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1

# Full dashboard
python training/cli.py monitor

# Or just Vertex AI table
python training/cli.py monitor --vertex
```

**Watch for:**

#### Check 5.1: Vertex AI Jobs Table
```
============================================================
VERTEX AI JOBS (Last 24h)
============================================================

Job ID: 1234567890123 (last 12 digits)
  Name:    arr-coc-training-20251116-123456
  Region:  us-central1        ← NEW COLUMN! (verify present)
  State:   JOB_STATE_RUNNING
  Runtime: 4m 23s
  Note:    —
```

**✅ SUCCESS**:
- Job appears in table
- Region column shows us-central1
- State shows RUNNING (not stuck)
- Note column empty (no errors)

**❌ FAILURE**: Error in Note column → check full logs

---

#### Check 5.2: Active W&B Runs Table
```
============================================================
ACTIVE W&B RUNS
============================================================

Run: xyz123
  Name:    arr-coc-training-20251116-123456
  State:   running
  Runtime: 2m 15s
  Tags:    vertex-ai, t4-gpu
```

**✅ SUCCESS**: W&B run appears and shows running
**❌ FAILURE**: No W&B run → training script issue

---

## Complete Verification Checklist

### ✅ Phase 1: Launch (0-5 minutes)

- [ ] **1.1**: Auto-selection message appears: "🤖 Auto-selected machine type: n1-standard-4"
- [ ] **1.2**: Validation passes all checks (especially GPU+Machine compatibility)
- [ ] **1.3**: Launch summary shows machine type (not blank/None)
- [ ] **1.4**: W&B queue submission succeeds

### ✅ Phase 2: Cloud Run Launcher (5-7 minutes)

- [ ] **2.1**: Agent starts and monitors for errors
- [ ] **2.2**: Agent picks up job from W&B queue
- [ ] **2.3**: Vertex AI submission shows n1-standard-4 + NVIDIA_TESLA_T4
- [ ] **2.4**: Vertex AI job ID returned (no errors)
- [ ] **2.5**: NO wrapper bailout errors (wrapper stays silent)

### ✅ Phase 3: Vertex AI Job Creation (7-10 minutes)

- [ ] **3.1**: Job appears in `gcloud ai custom-jobs list`
- [ ] **3.2**: State progresses: QUEUED → PENDING → RUNNING
- [ ] **3.3**: Job spec YAML shows correct machine+GPU+count

### ✅ Phase 4: Vertex AI Job Logs (10-15 minutes)

- [ ] **4.1**: Container starts successfully
- [ ] **4.2**: GPU detected in logs (Tesla T4, 15360 MiB)
- [ ] **4.3**: Training initializes (W&B run starts)
- [ ] **4.4**: Training progresses (loss/acc logged, GPU utilized)

### ✅ Phase 5: CLI Monitor (Continuous)

- [ ] **5.1**: Job appears in Vertex AI table with Region column
- [ ] **5.2**: W&B run appears in Active Runs table

---

## Success Criteria Summary

**CHAMPION #1 succeeds if ALL of these are true:**

1. ✅ Auto-selection message appears with n1-standard-4
2. ✅ Validation passes (no GPU+Machine compatibility errors)
3. ✅ Vertex AI accepts job submission (job ID returned)
4. ✅ Vertex AI job spec shows n1-standard-4 + NVIDIA_TESLA_T4
5. ✅ Container detects Tesla T4 GPU in logs
6. ✅ Training begins and progresses
7. ✅ NO wrapper bailout errors at any stage

**If all ✅**: GPU auto-selection system works perfectly!

**If any ❌**: Note which check failed, launch CHAMPION #2 with fix.

---

## Failure Scenarios & Debugging

| Check Failed | Symptom | Root Cause | Fix |
|--------------|---------|------------|-----|
| **1.1** | No auto-selection message | constants.py didn't trigger | Verify WANDB_LAUNCH_MACHINE_TYPE blank in .training |
| **1.2** | Validation fails | GPU+Machine incompatible | Check validation.py validate_gpu_machine_compatibility() |
| **2.3** | Wrong machine in submission | Config didn't propagate | Check constants.py config injection (line 74-124) |
| **2.5** | Wrapper bailout | Validation missed error | Update validation.py with new GPU+Machine rule |
| **3.3** | Wrong machine in job spec | W&B config override | Check W&B Launch queue config |
| **4.2** | No GPU detected | Wrong machine family | Critical validation bug! Check machine_selection.py |
| **5.1** | Job stuck PENDING | Quota/resource issue | Run `python training/cli.py infra` to check quotas |

---

## Post-Test Analysis

### 1. Review Launch Log
```bash
# Check full launch output
less /tmp/champion-1-launch.log

# Search for key messages
grep "Auto-selected" /tmp/champion-1-launch.log
grep "Validation" /tmp/champion-1-launch.log
grep "Launching training job" /tmp/champion-1-launch.log
```

### 2. Review Cloud Run Logs
```bash
# Get full launcher logs
gcloud logging read \
  "resource.type=cloud_run_job \
   resource.labels.job_name=vertex-ai-launcher" \
  --limit=200 \
  --format="table(timestamp,textPayload)" \
  --freshness=30m > /tmp/champion-1-launcher.log

# Search for submission details
grep -A5 "Submitting to Vertex AI" /tmp/champion-1-launcher.log
```

### 3. Review Vertex AI Job Details
```bash
# Get complete job description
gcloud ai custom-jobs describe $LATEST_JOB \
  --region=us-central1 \
  --format=yaml > /tmp/champion-1-job-spec.yaml

# Verify machine+GPU in spec
grep -A10 "machineSpec:" /tmp/champion-1-job-spec.yaml
```

### 4. Check Training Logs
```bash
# Save first 500 lines of training logs
gcloud ai custom-jobs stream-logs $LATEST_JOB \
  --region=us-central1 2>&1 | head -500 > /tmp/champion-1-training.log

# Verify GPU detection
grep -i "gpu\|cuda\|t4" /tmp/champion-1-training.log
```

---

## Next Steps

### If CHAMPION #1 Succeeds (All ✅)
1. Document success in GPU_VALIDATION_FIX_PLAN.md
2. Test with different GPU types (L4, V100, A100)
3. Test user override validation (invalid combos)
4. Update CLAUDE.md with production patterns

### If CHAMPION #1 Fails (Any ❌)
1. Note exact failure point and error message
2. Identify root cause from failure scenarios table
3. Apply fix to relevant file (constants.py, validation.py, machine_selection.py)
4. Launch CHAMPION #2 with descriptive name (e.g., "THE-CONFIG-PROPAGATOR" if 2.3 failed)
5. Repeat testing methodology

---

## Monitoring Commands Quick Reference

```bash
# Terminal 1: Launch
cat << 'EOF' | python -u training/cli.py launch 2>&1 | tee /tmp/champion-1-launch.log
💥 CHAMPION #1: THE-AUTO-SELECTOR-PRIME! 💥
EOF

# Terminal 2: Cloud Run
watch -n 5 'gcloud run jobs executions list --job=vertex-ai-launcher --region=us-central1 --limit=3'

# Terminal 3: Vertex AI
watch -n 5 'gcloud ai custom-jobs list --region=us-central1 --limit=5'

# Terminal 4: Job logs (once $LATEST_JOB set)
gcloud ai custom-jobs stream-logs $LATEST_JOB --region=us-central1

# Terminal 5: Monitor
python training/cli.py monitor --vertex
```

---

**READY TO LAUNCH CHAMPION #1!** 🚀

Follow this methodology step-by-step to verify GPU auto-selection works end-to-end.
