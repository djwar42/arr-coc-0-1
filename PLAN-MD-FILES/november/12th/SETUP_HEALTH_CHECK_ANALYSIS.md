# Setup Health Check Opportunities Analysis

## Current Setup Components (10 Resources)

### ✅ ALREADY HAS HEALTH CHECK:
- Cloud Function (DONE) - Verifies ACTIVE state + config

---

## HIGH BENEFIT - SIMPLE (DO THESE!)

### 1. Cloud Scheduler ⭐ PRIORITY 1
**Why:** Companion to Cloud Function, can silently fail
**States:** Job can exist but have wrong schedule/target/auth
**Complexity:** SIMPLE (1 gcloud call)
**Benefit:** HIGH (scheduler broken = no pricing updates)

**Check:**
```python
# gcloud scheduler jobs describe arr-coc-pricing-scheduler
# Verify:
- Job exists
- Schedule = "*/20 * * * *" (every 20 min)
- Target URI = function URL
- Auth header present
```

**Recovery:**
- If broken: DELETE + recreate (like we did for function)
- If missing: CREATE
- If healthy: SKIP

**Why This Matters:**
- Job can exist but point to wrong function URL
- Schedule can be wrong (every 1 hour vs 20 min)
- Auth header can be missing (function fails)

---

### 2. Artifact Registry ⭐ PRIORITY 2
**Why:** Critical for ALL builds (Docker images, pricing data)
**States:** Can exist with wrong format (docker vs generic)
**Complexity:** SIMPLE (1 gcloud call)
**Benefit:** HIGH (builds fail if registry broken)

**Check:**
```python
# gcloud artifacts repositories describe arr-coc-registry
# Verify:
- Repository exists
- Format = "GENERIC" (for pricing) or "DOCKER" (for images)
- Location = us-central1
- State = ACTIVE
```

**Recovery:**
- If wrong format: WARN user (can't fix automatically)
- If wrong location: WARN user (can't migrate)
- If missing: CREATE

**Why This Matters:**
- Wrong format = builds fail with cryptic errors
- Wrong location = high egress costs
- Can exist from previous setup attempt with wrong settings

---

## MEDIUM BENEFIT - MODERATE COMPLEXITY

### 3. VPC Peering
**Why:** Required for Vertex AI, affects other GCP services
**States:** Can be ACTIVE, INACTIVE, or partially configured
**Complexity:** MODERATE (network API calls)
**Benefit:** MEDIUM (failures are rare, but hard to debug)

**Check:**
```python
# gcloud compute networks peerings list
# Verify:
- Peering exists: google-managed-services-default
- State = ACTIVE
- Network = default
```

**Recovery:**
- If INACTIVE: Attempt reactivation
- If missing: CREATE
- If healthy: SKIP

**Why This Matters:**
- Peering can be inactive (broken connection)
- Affects Vertex AI training jobs
- Failures are silent until training starts

---

### 4. Service Account IAM Permissions
**Why:** SA can exist but lack permissions
**States:** SA exists, but IAM bindings failed
**Complexity:** MODERATE (need to test actual API access)
**Benefit:** MEDIUM (failures show up immediately during builds)

**Check:**
```python
# Test actual permissions (not just list bindings)
# Try:
- List GCS buckets (Storage Admin)
- List Artifact Registry repos (AR Writer)
- Describe Cloud Build (Cloud Build Editor)
```

**Recovery:**
- If missing: Re-apply IAM bindings
- If denied: WARN user (org policy issue)

**Why This Matters:**
- Bindings can exist but org policies override them
- Testing actual API access catches org-level restrictions
- Failures are immediate but hard to diagnose

---

## LOW BENEFIT - SKIP THESE

### 5. GCS Buckets (3 buckets)
**Why:** Simple binary state (exist or not)
**Complexity:** TRIVIAL
**Benefit:** LOW (failures are obvious immediately)

**Skip Because:**
- `gsutil ls` is instant and reliable
- No partial states possible
- Failures are immediate and obvious

---

### 6. Service Account Key File
**Why:** Simple file existence check
**Complexity:** TRIVIAL
**Benefit:** LOW (failures are immediate)

**Skip Because:**
- File either exists or doesn't (no partial state)
- Failures show up immediately when launch tries to use it

---

### 7. IAM Role Bindings (6 bindings)
**Why:** gcloud handles idempotency automatically
**Complexity:** TRIVIAL
**Benefit:** LOW (handled by gcloud)

**Skip Because:**
- `add-iam-policy-binding` is inherently idempotent
- Calling it twice is safe (no-op if already bound)
- No verification needed

---

## RECOMMENDATION PRIORITY

### Implement Now (High ROI):
1. ⭐ **Cloud Scheduler** (simple + high benefit)
2. ⭐ **Artifact Registry** (simple + high benefit)

### Nice to Have (Medium ROI):
3. VPC Peering (moderate complexity, medium benefit)
4. Service Account IAM test (moderate complexity, medium benefit)

### Skip (Low ROI):
- GCS Buckets (too simple)
- SA Key File (too simple)
- IAM Bindings (handled by gcloud)

---

## ESTIMATED EFFORT

**Cloud Scheduler:** ~30 min
- 1 gcloud describe call
- Parse JSON (schedule, uri, state)
- DELETE if broken + recreate

**Artifact Registry:** ~20 min
- 1 gcloud describe call
- Parse JSON (format, location)
- WARN if broken (can't auto-fix)

**Total:** ~50 min for both high-priority checks

---

## IMPACT ANALYSIS

### Without Health Checks:
```
User runs setup
Scheduler exists but points to old function URL
Setup says "✓ Complete"
20 minutes later: scheduler fires
ERROR: function not found
User confused: "setup said it worked!"
```

### With Health Checks:
```
User runs setup
Health check detects scheduler misconfiguration
Deletes broken scheduler
Recreates with correct config
Setup says "✓ Complete"
20 minutes later: scheduler fires
✓ Pricing updated successfully
```

**User Experience:** Goes from "why is this broken?" to "it just works"

