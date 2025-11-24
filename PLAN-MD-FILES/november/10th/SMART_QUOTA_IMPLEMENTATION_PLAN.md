# Smart Auto-Quota System Implementation Plan

**Goal:** Implement intelligent quota detection, auto-request, and upgrade suggestions for Cloud Build Worker Pools

---

## Phase 1: Quota Detection & Checking

- [✓] **Task 1.1:** Create helper function to check current CPU quota
  - Function: `check_c3_cpu_quota(project_id, region)` ✅ IMPLEMENTED
  - Returns: (limit, usage) tuple
  - Uses: `gcloud compute regions describe us-central1 --format=json`
  - Parses C3_CPUS quota metric from JSON

- [✓] **Task 1.2:** Create ordered machine availability list
  - Function: `get_available_c3_machines(c3_quota_limit)` ✅ IMPLEMENTED
  - List C3 machines: 176, 88, 44, 22, 8, 4 (largest first)
  - Filters to machines that fit within quota
  - Returns list of (machine_type, vcpus) tuples

- [✓] **Task 1.3:** Determine best available machine
  - Function: `get_best_available_c3_machine(project_id, region)` ✅ IMPLEMENTED
  - Check quota against ordered list (largest to smallest)
  - Returns (machine_type, vcpus, quota_limit) tuple
  - Fallback: c3-highcpu-4 if no quota

---

## Phase 2: Auto-Request Quota ✅ COMPLETE (Manual guidance approach)

- [✓] **Task 2.1:** ~~Create quota request tracking file~~ SKIPPED
  - Not needed - user requests quota manually via console
  - Simpler approach: comprehensive manual instructions

- [✓] **Task 2.2:** Create comprehensive quota error messages ✅ IMPLEMENTED
  - Detects quota errors during worker pool creation
  - Shows 2 options: Request quota OR use smaller machine
  - Direct console link with exact steps and filters
  - Suggests next smaller machine if available

- [✓] **Task 2.3:** Integrate smart quota detection into setup ✅ IMPLEMENTED
  - Automatically detects best C3 machine within quota
  - Shows detected quota limit and selected machine
  - Comprehensive error handling with actionable guidance
  - Displays upgrade suggestions when successful

---

## Phase 3: Quota Upgrade Detection

- [ ] **Task 3.1:** ~~Create upgrade detector function~~ DEFERRED
  - Can be added later if needed
  - Current approach: Show upgrade suggestions during setup success

- [ ] **Task 3.2:** ~~Integrate upgrade detection into launch~~ DEFERRED
  - User can manually re-run setup after quota approved
  - Simpler workflow than auto-detection

- [ ] **Task 3.3:** ~~Create upgrade helper command~~ DEFERRED
  - Not needed - setup automatically detects best machine
  - User just needs to run: python training/cli.py teardown && setup

---

## Phase 4: Nice Messages & UX ✅ COMPLETE

- [✓] **Task 4.1:** Create comprehensive quota message template ✅ IMPLEMENTED
  - Shows: Current quota, attempted machine, needed CPUs
  - Shows: 2 options (request quota OR use smaller machine)
  - Shows: Exact console link with step-by-step instructions

- [✓] **Task 4.2:** Add region-specific guidance ✅ IMPLEMENTED
  - Shows region (us-central1) in error messages
  - Directs user to filter by region in console
  - Region-specific quota detection

- [✓] **Task 4.3:** Update setup output ✅ IMPLEMENTED
  - Shows detected quota limit during pool creation
  - Shows selected machine and vCPUs
  - Shows upgrade suggestions if quota allows better machines
  - Estimates build times based on machine size

---

## Phase 5: Testing & Validation

- [ ] **Task 5.1:** Test quota detection
  - Verify correct quota values returned
  - Verify machine availability detection works

- [ ] **Task 5.2:** Test auto-request flow
  - Verify request tracking file created
  - Verify request command executed (or fallback shown)
  - Verify no duplicate requests

- [ ] **Task 5.3:** Test upgrade detection
  - Simulate quota increase
  - Verify upgrade suggestion shown on launch
  - Verify upgrade command works (if implemented)

- [ ] **Task 5.4:** Test end-to-end flow
  - Run setup with no quota
  - Verify auto-request happens
  - Simulate quota approval
  - Verify upgrade detection triggers
  - Verify upgrade works

---

## Implementation Notes

**Quota Detection Method:**
```bash
# Check CPUs in region
gcloud compute regions describe us-central1 --project=PROJECT_ID \
  --format="value(quotas[name='CPUS'].limit)"

# Or use Services Quota API
gcloud services quota list --service=compute.googleapis.com \
  --project=PROJECT_ID --consumer=projects/PROJECT_ID
```

**Quota Request Method:**
```bash
# Method 1: Alpha quotas API (preferred)
gcloud alpha quotas preferences create \
  --service=compute.googleapis.com \
  --project=PROJECT_ID \
  --quota-id=CPUS-per-project-region \
  --preferred-value=88 \
  --dimensions=region=us-central1 \
  --justification="Cloud Build worker pools for PyTorch compilation"

# Method 2: Fallback - manual instructions only
# (If alpha command doesn't work or quota ID is wrong)
```

**Machine Ordered List (fastest → slowest):**
1. c3-highcpu-176 (176 vCPUs) - ~15-20 min builds
2. c3-highcpu-88 (88 vCPUs) - ~25-30 min builds
3. c3-highcpu-44 (44 vCPUs) - ~35-40 min builds ← DEFAULT
4. c3-highcpu-22 (22 vCPUs) - ~50-60 min builds
5. c3-highcpu-8 (8 vCPUs) - ~80-90 min builds
6. c3-highcpu-4 (4 vCPUs) - ~120+ min builds

---

## Success Criteria

✅ User never has to manually request quota (auto-requested)
✅ User sees helpful messages about quota status
✅ User notified when quota upgrades are available
✅ Manual instructions always shown as fallback
✅ No duplicate quota requests
✅ Region-specific requests (us-central1)
✅ System detects and suggests upgrades automatically

---

## Files to Modify

- `training/cli/setup/core.py` - Add quota detection, request, upgrade logic
- `training/cli/launch/core.py` - Add upgrade detection on each launch
- `.gitignore` - Add `.quota-requests.json`
- `SPOT_WORKER_POOL_IMPLEMENTATION.md` - Update with smart quota info
