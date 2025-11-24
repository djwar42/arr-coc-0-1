# MECHA Lingering System - Post-Victory Adulation Mode

**Status**: CONCEPT → DESIGN → IMPLEMENTATION
**Date**: 2025-11-14
**Motivation**: Skip worker pool spin-up (2-3 min) + leverage warm ccache (10-100× faster rebuilds) for quick iteration cycles

---

## 🎯 The Core Idea

After a MECHA completes a job (victory or defeat), it **LINGERS for 5 minutes** in "RECEIVING_ADULATION" state, ready for immediate redeployment. If user relaunches during this window:

**Normal Launch** (cold):
```
Launch CLI → MECHA Battle (select region, spin up worker) → Queue (2-3 min) → Build (30 min)
Total: ~33 minutes
```

**Lingering Launch** (hot):
```
Launch CLI → Detect Lingering MECHA → Skip Battle → Build immediately (ccache warm!)
Total: ~3-8 minutes (10× faster on cache hits!)
```

---

## 🦾 MECHA State Machine

```
        ┌──────────────┐
        │   INACTIVE   │ ← Default (no worker pool active)
        └──────┬───────┘
               │ launch
               ↓
        ┌──────────────┐
        │   SELECTING  │ ← MECHA Battle (price comparison, region selection)
        └──────┬───────┘
               │ selected
               ↓
        ┌──────────────┐
        │    QUEUED    │ ← Worker pool spinning up
        └──────┬───────┘
               │ started
               ↓
        ┌──────────────┐
        │   WORKING    │ ← Build running
        └──────┬───────┘
               │ completed/failed
               ↓
   ┌───────────────────────────┐
   │  RECEIVING_ADULATION      │ ← NEW! Lingering for 5 min
   │  (Warm ccache ready!)     │
   └───────┬───────────────────┘
           │
           ├─ relaunch (< 5 min) → QUEUED (skip MECHA Battle!)
           │
           └─ timeout (5 min) → INACTIVE (worker pool destroyed)
```

**Key States:**
- **RECEIVING_ADULATION**: Post-victory linger period (5 min)
  - Worker pool still alive
  - ccache warm
  - Region locked in
  - Quick-ready for relaunch

---

## 📊 Data Structures

### mecha_hangar.json Extensions

```json
{
  "regions": {
    "us-west2": {
      "operational_status": "RECEIVING_ADULATION",
      "current_build_id": "d0ee27e8-...",
      "last_build_status": "SUCCESS",
      "lingering_until": "2025-11-14T12:45:00Z",  // NEW!
      "linger_reason": "POST_VICTORY",             // NEW!
      "ccache_warm": true,                         // NEW!
      "worker_pool_ready": true,                   // NEW!
      "quick_relaunch_available": true             // NEW!
    }
  }
}
```

**New Fields:**
- `lingering_until`: ISO timestamp when linger expires (5 min after completion)
- `linger_reason`: Why lingering (POST_VICTORY, POST_FAILURE)
- `ccache_warm`: Boolean flag (true = warm cache available)
- `worker_pool_ready`: Boolean (true = no spin-up needed)
- `quick_relaunch_available`: Boolean (true = can skip MECHA Battle)

---

## 🧠 Adaptive Lingering Time - Smart Duration Tuning

**Problem**: Fixed 5-minute linger wastes money if user never relaunches, but too short if they're actively iterating.

**Solution**: Start conservative (1 min), expand based on actual usage patterns, contract when unused.

### 🎯 Approach 1: Success-Based Expansion (Simple)

**Logic**: Every successful MECHA reuse increases linger duration.

```
Initial linger: 1 minute
1st catch → 2 minutes
2nd catch → 3 minutes
3rd catch → 4 minutes
...
Max: 10 minutes (safety cap)
```

**Warmdown**: Decrease -1 min every linger expiration without reuse.

```
Timeline:
10:00 - Build completes, linger 3 min (earned from previous catches)
10:03 - Linger expires (no relaunch) → next linger: 2 min
10:05 - User relaunches immediately, catches lingering MECHA!
10:10 - Build completes, linger 3 min (expanded from 2→3)
10:13 - User relaunches, catches again!
10:18 - Build completes, linger 4 min (expanded from 3→4)
10:22 - Linger expires (no relaunch) → next linger: 3 min
```

**Data Structure Extensions** (mecha_hangar.json):
```json
{
  "regions": {
    "us-west2": {
      "linger_duration_minutes": 3,       // Current linger time (adaptive)
      "linger_catches": 5,                 // Total successful reuses
      "linger_misses": 2,                  // Total linger expirations without reuse
      "linger_catch_streak": 2,            // Current catch streak
      "last_linger_catch": "2025-11-14T10:13:00Z",
      "last_linger_miss": "2025-11-14T09:22:00Z"
    }
  }
}
```

**Pseudocode**:
```python
def calculate_next_linger_duration(region_stats):
    current_duration = region_stats.get("linger_duration_minutes", 1)  # Default: 1 min

    if reuse_happened:
        # Expand: Caught the lingering MECHA!
        new_duration = min(current_duration + 1, 10)  # Cap at 10 min
        region_stats["linger_catches"] += 1
        region_stats["linger_catch_streak"] += 1
        region_stats["last_linger_catch"] = now()
    else:
        # Contract: Linger expired unused
        new_duration = max(current_duration - 1, 1)  # Floor at 1 min
        region_stats["linger_misses"] += 1
        region_stats["linger_catch_streak"] = 0
        region_stats["last_linger_miss"] = now()

    region_stats["linger_duration_minutes"] = new_duration
    return new_duration
```

**Pros**:
- Simple logic
- Self-tuning based on actual user behavior
- Fast adaptation (1 min changes per cycle)
- Works for any usage pattern

**Cons**:
- Slow to detect rapid iteration (takes 5 catches to reach 5 min)
- Doesn't anticipate - only reacts

---

### 🎯 Approach 2: Pattern Detection (Smart)

**Logic**: Detect rapid iteration sessions and proactively increase linger.

**Pattern Signals**:
1. **Rapid launches**: 2+ launches within 10 minutes
2. **Short build duration**: Builds completing in <10 min (ccache warm!)
3. **Build frequency**: 3+ builds in 30 min window

**Adaptive Tiers**:
```
Tier 0 (Cold):     1 min linger  - Default, no recent activity
Tier 1 (Warm):     3 min linger  - 2 launches in 10 min detected
Tier 2 (Hot):      5 min linger  - 3 launches in 15 min detected
Tier 3 (Blazing):  10 min linger - 4+ launches in 20 min detected
```

**Warmdown Decay**: Drop 1 tier every 15 minutes of inactivity.

```
Timeline:
09:00 - Launch #1 (Tier 0 → 1 min linger after completion)
09:05 - Launch #2 (within 10 min!) → Detect rapid iteration → Tier 1 (3 min)
09:12 - Launch #3 (within 15 min!) → Tier 2 (5 min)
09:20 - Launch #4 → Tier 3 (10 min)
09:35 - Build completes, linger 10 min (Tier 3)
09:45 - Linger expires, no activity for 15 min → Tier 2 (5 min)
10:00 - Still no activity → Tier 1 (3 min)
10:15 - Still no activity → Tier 0 (1 min)
```

**Data Structure Extensions**:
```json
{
  "regions": {
    "us-west2": {
      "linger_tier": 2,                       // Current adaptive tier (0-3)
      "recent_launches": [                     // Last 10 launches (timestamps)
        "2025-11-14T09:20:00Z",
        "2025-11-14T09:12:00Z",
        "2025-11-14T09:05:00Z"
      ],
      "last_activity": "2025-11-14T09:45:00Z",
      "iteration_session_active": true,        // Rapid iteration detected
      "avg_build_duration_min": 7.5            // Rolling average (last 5 builds)
    }
  }
}
```

**Pseudocode**:
```python
def detect_iteration_pattern(region_stats):
    now = datetime.now()
    recent_launches = region_stats.get("recent_launches", [])

    # Count launches in time windows
    launches_10min = count_launches_since(recent_launches, now - timedelta(minutes=10))
    launches_15min = count_launches_since(recent_launches, now - timedelta(minutes=15))
    launches_20min = count_launches_since(recent_launches, now - timedelta(minutes=20))

    # Determine tier
    if launches_20min >= 4:
        tier = 3  # Blazing (10 min linger)
    elif launches_15min >= 3:
        tier = 2  # Hot (5 min linger)
    elif launches_10min >= 2:
        tier = 1  # Warm (3 min linger)
    else:
        tier = 0  # Cold (1 min linger)

    region_stats["linger_tier"] = tier
    region_stats["iteration_session_active"] = (tier > 0)

    return TIER_DURATIONS[tier]  # [1, 3, 5, 10]

def apply_warmdown_decay(region_stats):
    now = datetime.now()
    last_activity = region_stats.get("last_activity")

    if not last_activity:
        return

    minutes_idle = (now - last_activity).total_seconds() / 60

    # Drop 1 tier every 15 min of inactivity
    if minutes_idle >= 15:
        tiers_to_drop = int(minutes_idle / 15)
        current_tier = region_stats.get("linger_tier", 0)
        new_tier = max(0, current_tier - tiers_to_drop)
        region_stats["linger_tier"] = new_tier
```

**Pros**:
- Proactive (anticipates continued iteration)
- Fast tier escalation (3 launches → 5 min linger)
- Graceful warmdown (gradual decay)
- Detects "deep work" sessions

**Cons**:
- More complex logic
- Requires tracking launch history
- Could over-linger if pattern breaks

---

### 🎯 Hybrid Approach (Best of Both)

**Combine success-based expansion with pattern detection**:

1. **Pattern detection sets baseline tier** (0-3)
2. **Successful catches add bonus minutes** (+1 min per catch, up to +3)
3. **Total linger = tier_duration + catch_bonus**

```
Example:
- Tier 2 (Hot) = 5 min base
- 2 successful catches = +2 min bonus
- Total linger = 7 min
```

**Benefits**:
- Fast adaptation (pattern detection)
- Rewards actual usage (catch bonus)
- Self-correcting (catch bonus decays if not caught)

**Data Structure**:
```json
{
  "linger_tier": 2,              // Pattern-based tier (0-3)
  "linger_catch_bonus": 2,       // Extra minutes from catches (0-3)
  "linger_duration_minutes": 7   // Total: tier + bonus
}
```

---

### 📊 General Considerations

**Cost Impact**:
```
Fixed 5 min:     $0.63-1.25 per linger
Adaptive 1 min:  $0.13-0.25 (initial)
Adaptive 10 min: $1.25-2.50 (max)
```

**Savings from Correct Prediction**:
- Catch within linger → Save 2-3 min spin-up ($0.25-0.40)
- Missed linger → Waste linger cost ($0.13-2.50)

**Hit Rate Optimization**:
- Goal: >50% catch rate (reuse > waste)
- Adaptive helps: linger long when needed, short when not

**Safety Caps**:
- Max linger: 10 minutes (prevent runaway costs)
- Min linger: 1 minute (always give a chance)
- Absolute timeout: Worker pool auto-destroys at max linger

**Edge Cases**:
1. **First launch of day**: Tier 0 (1 min) - no pattern yet
2. **Long lunch break**: Warmdown to Tier 0 after 45 min idle
3. **Alternating regions**: Each region tracks independently
4. **Build failure during iteration**: Don't penalize - keep tier

**Monitoring Metrics**:
- `linger_catch_rate`: catches / (catches + misses)
- `avg_linger_duration`: Average linger time used
- `total_linger_cost`: Sum of all linger costs
- `savings_from_catches`: Spin-up time avoided

**UI Indicators**:
```
Launch output when catching:
✅ Caught lingering MECHA! (Tier 2: Hot - 5 min earned from pattern)
→ Skipped 2.5 min spin-up
→ Next linger: 6 min (expanded +1 from catch)

Launch output when linger expires:
⏰ Linger expired (unused for 5 min)
→ Next linger: 4 min (reduced -1)
```

---

### 🚀 Recommended Implementation Path

**Phase 1: Start Simple (Approach 1)**
- Implement success-based expansion
- 1 min → expand on catch → contract on miss
- Get baseline data on catch rates

**Phase 2: Add Pattern Detection (Approach 2)**
- Detect rapid iteration (2+ launches in 10 min)
- Bump to Tier 1 (3 min) when detected
- Monitor improvement in catch rate

**Phase 3: Hybrid Refinement**
- Combine tier + catch bonus
- Fine-tune decay rates based on metrics
- Optimize for >60% catch rate

---

### Phase 1: Linger State Management

**File**: `training/cli/launch/mecha/mecha_hangar.py`

```python
from datetime import datetime, timedelta, timezone

def set_lingering_state(region: str, build_status: str, build_id: str):
    """
    Set MECHA to RECEIVING_ADULATION state after build completion.

    Args:
        region: GCP region (e.g., "us-west2")
        build_status: Final build status (SUCCESS/FAILURE)
        build_id: CloudBuild ID
    """
    hangar = load_mecha_hangar()

    linger_until = datetime.now(timezone.utc) + timedelta(minutes=5)

    hangar["regions"][region].update({
        "operational_status": "RECEIVING_ADULATION",
        "last_build_status": build_status,
        "current_build_id": build_id,
        "lingering_until": linger_until.isoformat(),
        "linger_reason": f"POST_{build_status}",
        "ccache_warm": True,
        "worker_pool_ready": True,
        "quick_relaunch_available": True
    })

    save_mecha_hangar(hangar)

    print(f"🦾 MECHA {region} now RECEIVING_ADULATION (lingering until {linger_until.strftime('%H:%M:%S')})")
    print(f"   Quick relaunch available for next 5 minutes! ⚡")


def check_lingering_mecha(region: str = None) -> dict:
    """
    Check if a lingering MECHA is available for quick relaunch.

    Returns:
        {
            "available": bool,
            "region": str,
            "lingering_until": str,
            "ccache_warm": bool,
            "time_remaining_seconds": int
        }
    """
    hangar = load_mecha_hangar()
    now = datetime.now(timezone.utc)

    # If region specified, check that region only
    regions_to_check = [region] if region else hangar["regions"].keys()

    for reg in regions_to_check:
        mecha = hangar["regions"].get(reg, {})

        if mecha.get("operational_status") != "RECEIVING_ADULATION":
            continue

        linger_until = datetime.fromisoformat(mecha.get("lingering_until", ""))

        # Check if still within linger window
        if now < linger_until:
            time_remaining = (linger_until - now).total_seconds()

            return {
                "available": True,
                "region": reg,
                "lingering_until": mecha["lingering_until"],
                "ccache_warm": mecha.get("ccache_warm", False),
                "time_remaining_seconds": int(time_remaining),
                "last_build_status": mecha.get("last_build_status", "UNKNOWN")
            }
        else:
            # Linger expired - mark as inactive
            expire_lingering_mecha(reg)

    return {"available": False}


def expire_lingering_mecha(region: str):
    """
    Expire a lingering MECHA (linger timeout reached).

    Sets state back to INACTIVE and clears linger flags.
    """
    hangar = load_mecha_hangar()

    hangar["regions"][region].update({
        "operational_status": "INACTIVE",
        "current_build_id": "NONE",
        "lingering_until": None,
        "linger_reason": None,
        "ccache_warm": False,
        "worker_pool_ready": False,
        "quick_relaunch_available": False
    })

    save_mecha_hangar(hangar)

    print(f"⏱️  MECHA {region} linger expired - now INACTIVE")
```

---

### Phase 2: Launch Flow Integration

**File**: `training/cli/launch/core.py`

**At launch start (before MECHA Battle):**

```python
def launch_training_job(base_image_tag: str, training_image_tag: str, ...):
    """Launch training job with lingering MECHA detection."""

    # NEW: Check for lingering MECHA first!
    lingering = check_lingering_mecha(region=None)  # Check all regions

    if lingering["available"]:
        region = lingering["region"]
        time_left = lingering["time_remaining_seconds"]

        status("[bold green]⚡ LINGERING MECHA DETECTED![/bold green]")
        status(f"   Region: [cyan]{region}[/cyan]")
        status(f"   Status: [yellow]RECEIVING_ADULATION (Post-{lingering['last_build_status']})[/yellow]")
        status(f"   Time remaining: [magenta]{time_left}s[/magenta]")
        status(f"   ccache: [green]WARM ♨️[/green] (10-100× faster rebuilds!)")
        status("")
        status("[bold yellow]🦾 MECHA BATTLE SKIPPED - Using lingering MECHA![/bold yellow]")
        status("")

        # Use lingering region directly - NO MECHA BATTLE!
        selected_region = region
        skip_worker_spinup = True

        # Update state: RECEIVING_ADULATION → QUEUED
        from cli.launch.mecha.mecha_hangar import update_mecha_status
        update_mecha_status(region, "QUEUED", build_id="pending")

    else:
        # Normal flow: MECHA Battle
        status("[bold cyan]⚔️  MECHA BATTLE COMMENCING![/bold cyan]")
        status("   Selecting optimal region based on price and fatigue...")

        selected_region = select_mecha_region()  # Normal selection
        skip_worker_spinup = False

    # Continue with build...
    build_id = submit_cloud_build(...)

    # After build completes (at end of launch):
    final_status = wait_for_build_completion(build_id)

    # NEW: Set lingering state after completion!
    from cli.launch.mecha.mecha_hangar import set_lingering_state
    set_lingering_state(selected_region, final_status, build_id)
```

---

### Phase 3: Linger Expiration Cleanup

**Background task or next-launch check:**

```python
def cleanup_expired_lingering_mechas():
    """
    Check all lingering MECHAs and expire any past their window.

    Called at:
    - Start of each launch (cleanup before battle)
    - Periodic background task (optional)
    """
    hangar = load_mecha_hangar()
    now = datetime.now(timezone.utc)

    for region, mecha in hangar["regions"].items():
        if mecha.get("operational_status") != "RECEIVING_ADULATION":
            continue

        linger_until_str = mecha.get("lingering_until")
        if not linger_until_str:
            continue

        linger_until = datetime.fromisoformat(linger_until_str)

        if now >= linger_until:
            print(f"⏱️  Expiring lingering MECHA in {region} (timeout)")
            expire_lingering_mecha(region)
```

---

## 🎨 UI/UX Enhancements

### Launch Output (Lingering MECHA Detected)

```
============================================================
ARR-COC Training Job Submission
============================================================

⚡ LINGERING MECHA DETECTED! ⚡

   Region: us-west2
   Status: RECEIVING_ADULATION (Post-SUCCESS)
   Time remaining: 247s
   ccache: WARM ♨️ (10-100× faster rebuilds!)

🦾 MECHA BATTLE SKIPPED - Using lingering MECHA!

   Why: Worker pool still active from previous victory!
   Benefit: No 2-3 min spin-up delay + warm ccache!
   Expected: 3-8 min build time (vs 30+ min cold)

⏳ Submitting build to lingering MECHA...
```

### Launch Output (No Lingering MECHA)

```
============================================================
ARR-COC Training Job Submission
============================================================

⏱️  No lingering MECHAs available (all expired or inactive)

⚔️  MECHA BATTLE COMMENCING!
   Selecting optimal region based on price and fatigue...
```

### Post-Build Output (Linger Activated)

```
✅ Build Complete!

🦾 MECHA us-west2 now RECEIVING_ADULATION!

   Lingering for 5 minutes (until 12:45:00)
   Quick relaunch available! ⚡

   If you run `python training/cli.py launch` within 5 minutes:
   - MECHA Battle skipped
   - Worker pool ready
   - ccache warm (10-100× faster!)

   Perfect for quick iterations! 🚀
```

---

## 🔍 Edge Cases & Fallbacks

### 1. Lingering MECHA Build Fails

**Problem**: What if relaunch on lingering MECHA fails immediately?

**Solution**: Fall back to normal MECHA Battle

```python
if lingering_used and build_failed_early:  # e.g., <2 min into build
    status("⚠️  Lingering MECHA build failed - falling back to MECHA Battle!")
    expire_lingering_mecha(region)
    selected_region = select_mecha_region()  # Normal battle
    # Retry with fresh MECHA
```

### 2. Multiple Concurrent Launches

**Problem**: What if user launches twice during linger window?

**Solution**: First launch "claims" the lingering MECHA, subsequent launches do normal battle

```python
def claim_lingering_mecha(region: str, build_id: str) -> bool:
    """
    Atomically claim a lingering MECHA for use.

    Returns False if already claimed by another launch.
    """
    hangar = load_mecha_hangar()

    if hangar["regions"][region]["operational_status"] != "RECEIVING_ADULATION":
        return False  # Already claimed or expired

    # Claim by setting to QUEUED with new build ID
    update_mecha_status(region, "QUEUED", build_id)
    return True
```

### 3. Linger Expiration During Build Submission

**Problem**: Lingering MECHA expires WHILE we're submitting the build

**Solution**: Check timestamp BEFORE claim, fail gracefully

```python
lingering = check_lingering_mecha()

if lingering["available"]:
    if lingering["time_remaining_seconds"] < 30:  # Less than 30s left
        status("⚠️  Lingering MECHA expires too soon - doing normal battle")
        lingering["available"] = False
    else:
        # Proceed with linger
        ...
```

### 4. Worker Pool Preempted During Linger

**Problem**: GCP preempts worker pool during 5-min linger window

**Solution**: Worker pool monitoring + fallback

```python
def verify_worker_pool_alive(region: str) -> bool:
    """Check if worker pool is actually still alive."""
    result = subprocess.run(
        ["gcloud", "builds", "worker-pools", "describe", f"pytorch-mecha-pool",
         f"--region={region}", "--format=value(state)"],
        capture_output=True, text=True, timeout=10
    )

    return result.returncode == 0 and "RUNNING" in result.stdout

# Before using lingering MECHA:
if lingering["available"]:
    if not verify_worker_pool_alive(lingering["region"]):
        status("⚠️  Lingering MECHA worker pool no longer alive - doing normal battle")
        expire_lingering_mecha(lingering["region"])
        lingering["available"] = False
```

---

## 💰 Cost Analysis

### Linger Cost

**Worker Pool Idle Cost** (5 min linger):
- C3-176: ~$15/hour → $1.25 for 5 min
- C3-88: ~$7.50/hour → $0.625 for 5 min

**When Linger Pays Off:**
- If relaunch happens: Save 2-3 min spin-up ($0.75-1.12) + ccache speedup (potentially 20+ min saved!)
- If no relaunch: Lose $0.63-1.25

**ROI Scenarios:**
- 20% relaunch rate: Break even
- 50% relaunch rate: 2× ROI
- 80% relaunch rate: 5× ROI (massive time savings from warm ccache)

**Recommendation**: Enable by default, add flag to disable:

```python
# In launch command
python training/cli.py launch --no-linger  # Disable linger (immediate shutdown)
```

---

## 🚀 Implementation Phases

### Phase 1: Core Linger State (1-2 hours)
- [ ] Add linger fields to mecha_hangar.json schema
- [ ] Implement `set_lingering_state()`
- [ ] Implement `check_lingering_mecha()`
- [ ] Implement `expire_lingering_mecha()`
- [ ] Add linger expiration cleanup at launch start

### Phase 2: Launch Integration (1-2 hours)
- [ ] Detect lingering MECHA at launch start
- [ ] Skip MECHA Battle if lingering available
- [ ] Update UI/status messages for linger
- [ ] Set linger state after build completion
- [ ] Test: Launch → complete → relaunch (within 5 min)

### Phase 3: Edge Cases & Fallbacks (1 hour)
- [ ] Implement early build failure fallback
- [ ] Add linger claim atomicity (concurrent launches)
- [ ] Add time-remaining check (< 30s = skip linger)
- [ ] Verify worker pool alive before using linger
- [ ] Test: Linger expiration, concurrent launches, worker pool preemption

### Phase 4: Monitoring & Tuning (30 min)
- [ ] Add linger hit rate to campaign_stats.json
- [ ] Track ccache speedup on linger relaunches
- [ ] Add `--linger-duration` flag (default: 5 min, configurable)
- [ ] Add `--no-linger` flag (disable feature)

**Total Estimate**: 4-6 hours implementation

---

## 🎯 Success Metrics

**After implementing linger system, track:**

1. **Linger Hit Rate**: % of launches using lingering MECHA
   - Target: > 30% for active development

2. **Time Savings**: Average time saved per linger hit
   - Target: > 5 min (spin-up + warm ccache)

3. **ccache Hit Rate**: On linger relaunches vs cold
   - Target: > 80% cache hits on linger

4. **Cost Efficiency**: Net cost savings from linger
   - Track: Linger idle cost vs spin-up + cold build savings

---

## 🔮 Future Enhancements

### Smart Linger Duration

Adjust linger time based on user behavior:

```python
# If user frequently relaunches within 2 min: shorten linger to 2-3 min
# If user rarely relaunches: increase linger to 10 min for rare but valuable hits

linger_duration = calculate_optimal_linger_duration(
    recent_relaunch_intervals=[2.5, 1.8, 4.2, ...]  # minutes
)
```

### Multi-Region Linger

Allow lingering MECHAs in multiple regions simultaneously:

```python
# Limit: Max 2-3 lingering MECHAs across all regions
# Priority: Keep lingering in regions with most recent activity
```

### Linger Prediction

Use ML to predict if user will relaunch:

```python
# Features: time of day, code change size, test failures, etc.
# If predicted relaunch probability > 60%: enable linger
# If < 20%: skip linger, shutdown immediately
```

---

## 🦾 MECHA LINGERING PHILOSOPHY

**The Concept**: A victorious MECHA doesn't immediately return to the hangar - it LINGERS on the battlefield, basking in the crowd's adulation, ready for an immediate encore performance if called upon!

**Key Insight**: The most expensive part of a CloudBuild isn't the compute - it's the LATENCY. Worker pool spin-up (2-3 min) + cold ccache (10× slower) dominates user experience for quick iterations.

**The Trade-Off**: Pay a small idle cost ($0.63-1.25 for 5 min) for MASSIVE time savings (5-25 min) if relaunch happens.

**When It Shines**:
- Debugging build failures (fix → relaunch cycle)
- Tweaking Dockerfiles (change → test cycle)
- CHONK monitoring development (today's work!)
- Any rapid iteration workflow

**The Name**: RECEIVING_ADULATION - because our MECHAs are celebrities, and celebrities linger after performances! 🦾💎🎤

---

## 🧠 Adaptive Lingering Duration - Smart Timing Strategy

**Problem**: Fixed 5-minute linger wastes money if never relaunched, but too short if actively iterating.

**Solution**: Start conservative (1 min), expand based on usage patterns, contract when unused.

### 🎯 Approach 1: Success-Based Expansion (Simple & Reactive)

**Core Logic**: Every successful MECHA reuse increases linger duration by 1 minute.

```
Initial linger: 1 minute
1st catch within linger → expand to 2 minutes
2nd catch within linger → expand to 3 minutes
3rd catch within linger → expand to 4 minutes
...
Max cap: 10 minutes
```

**Warmdown Logic**: Every linger expiration without reuse decreases duration by 1 minute (floor at 1 min).

**Example Timeline**:
```
10:00 - Build completes, linger 3 min (earned from previous catches)
10:03 - Linger expires unused → next linger: 2 min
10:05 - User relaunches, catches lingering MECHA! ✅
10:10 - Build completes, linger 3 min (expanded from 2→3)
10:13 - User relaunches, catches again! ✅
10:18 - Build completes, linger 4 min (expanded from 3→4)
10:22 - Linger expires unused → next linger: 3 min
10:30 - User relaunches (new MECHA, old one expired)
10:35 - Build completes, linger 3 min (no change, missed catch)
```

**Data Structure Extensions** (mecha_hangar.json):
```json
{
  "regions": {
    "us-west2": {
      "linger_duration_minutes": 3,
      "linger_catches": 5,
      "linger_misses": 2,
      "linger_catch_streak": 2,
      "last_linger_catch": "2025-11-14T10:13:00Z",
      "last_linger_miss": "2025-11-14T09:22:00Z"
    }
  }
}
```

**Pseudocode**:
```python
def calculate_next_linger_duration(region_stats, reuse_happened):
    current = region_stats.get("linger_duration_minutes", 1)  # Default: 1 min

    if reuse_happened:
        # Expand: Caught the lingering MECHA!
        new = min(current + 1, 10)  # Cap at 10 min
        region_stats["linger_catches"] += 1
        region_stats["linger_catch_streak"] += 1
        region_stats["last_linger_catch"] = now()
    else:
        # Contract: Linger expired unused
        new = max(current - 1, 1)  # Floor at 1 min
        region_stats["linger_misses"] += 1
        region_stats["linger_catch_streak"] = 0
        region_stats["last_linger_miss"] = now()

    region_stats["linger_duration_minutes"] = new
    return new
```

**✅ Pros**:
- Simple logic, easy to implement
- Self-tuning based on actual user behavior
- Fast adaptation (1 min changes per cycle)
- Works for any usage pattern

**❌ Cons**:
- Slow to detect rapid iteration (takes 5 catches to reach 5 min)
- Doesn't anticipate - only reacts
- Wastes money during warmup phase

---

### 🎯 Approach 2: Pattern Detection (Smart & Proactive)

**Core Logic**: Detect rapid iteration sessions and proactively increase linger.

**Pattern Signals**:
1. **Rapid launches**: 2+ launches within 10 minutes
2. **Short build duration**: Builds completing in <10 min (ccache warm!)
3. **Build frequency**: 3+ builds in 30 min window

**Adaptive Tiers**:
```
Tier 0 (Cold):     1 min linger  - Default, no recent activity
Tier 1 (Warm):     3 min linger  - 2 launches in 10 min detected
Tier 2 (Hot):      5 min linger  - 3 launches in 15 min detected
Tier 3 (Blazing):  10 min linger - 4+ launches in 20 min detected
```

**Warmdown Decay**: Drop 1 tier every 15 minutes of inactivity.

**Example Timeline**:
```
09:00 - Launch #1 (Tier 0 → 1 min linger after completion)
09:05 - Launch #2 (within 10 min!) → Detect rapid iteration → Tier 1 (3 min)
09:12 - Launch #3 (within 15 min!) → Tier 2 (5 min)
09:20 - Launch #4 → Tier 3 (10 min)
09:35 - Build completes, linger 10 min (Tier 3)
09:45 - Linger expires, no activity for 15 min → Tier 2 (5 min)
10:00 - Still no activity → Tier 1 (3 min)
10:15 - Still no activity → Tier 0 (1 min)
```

**Data Structure Extensions**:
```json
{
  "regions": {
    "us-west2": {
      "linger_tier": 2,
      "recent_launches": [
        "2025-11-14T09:20:00Z",
        "2025-11-14T09:12:00Z",
        "2025-11-14T09:05:00Z"
      ],
      "last_activity": "2025-11-14T09:45:00Z",
      "iteration_session_active": true,
      "avg_build_duration_min": 7.5
    }
  }
}
```

**Pseudocode**:
```python
TIER_DURATIONS = [1, 3, 5, 10]  # minutes

def detect_iteration_pattern(region_stats):
    now = datetime.now()
    recent = region_stats.get("recent_launches", [])

    # Count launches in time windows
    launches_10min = count_since(recent, now - timedelta(minutes=10))
    launches_15min = count_since(recent, now - timedelta(minutes=15))
    launches_20min = count_since(recent, now - timedelta(minutes=20))

    # Determine tier
    if launches_20min >= 4:
        tier = 3  # Blazing (10 min)
    elif launches_15min >= 3:
        tier = 2  # Hot (5 min)
    elif launches_10min >= 2:
        tier = 1  # Warm (3 min)
    else:
        tier = 0  # Cold (1 min)

    region_stats["linger_tier"] = tier
    region_stats["iteration_session_active"] = (tier > 0)

    return TIER_DURATIONS[tier]

def apply_warmdown_decay(region_stats):
    now = datetime.now()
    last = region_stats.get("last_activity")

    if not last:
        return

    idle_minutes = (now - last).total_seconds() / 60

    # Drop 1 tier every 15 min of inactivity
    if idle_minutes >= 15:
        tiers_to_drop = int(idle_minutes / 15)
        current = region_stats.get("linger_tier", 0)
        new = max(0, current - tiers_to_drop)
        region_stats["linger_tier"] = new
```

**✅ Pros**:
- Proactive (anticipates continued iteration)
- Fast tier escalation (3 launches → 5 min linger)
- Graceful warmdown (gradual decay)
- Detects "deep work" sessions

**❌ Cons**:
- More complex logic
- Requires tracking launch history
- Could over-linger if pattern breaks suddenly

---

### 🎯 Approach 3: Hybrid (Best of Both Worlds) ⭐ **RECOMMENDED**

**Combine success-based expansion with pattern detection**:

1. **Pattern detection sets baseline tier** (0-3) → fast adaptation
2. **Successful catches add bonus minutes** (+1 min per catch, up to +3)
3. **Total linger = tier_duration + catch_bonus**

**Example**:
```
- Tier 2 (Hot) = 5 min base
- 2 successful catches = +2 min bonus
- Total linger = 7 min
```

**Benefits**:
- Fast adaptation to rapid iteration (pattern detection)
- Rewards actual usage (catch bonus)
- Self-correcting (catch bonus decays if not caught)
- Best ROI (spend where it matters)

**Data Structure**:
```json
{
  "linger_tier": 2,              // Pattern-based tier (0-3)
  "linger_catch_bonus": 2,       // Extra minutes from catches (0-3)
  "linger_duration_minutes": 7   // Total: tier + bonus
}
```

**Pseudocode**:
```python
def calculate_hybrid_linger(region_stats, reuse_happened):
    # 1. Pattern detection (baseline tier)
    tier = detect_iteration_pattern(region_stats)
    base_duration = TIER_DURATIONS[tier]

    # 2. Catch bonus (reactive adjustment)
    catch_bonus = region_stats.get("linger_catch_bonus", 0)

    if reuse_happened:
        catch_bonus = min(catch_bonus + 1, 3)  # Cap at +3 min
    else:
        catch_bonus = max(catch_bonus - 1, 0)  # Decay on miss

    region_stats["linger_catch_bonus"] = catch_bonus

    # 3. Total duration
    total = base_duration + catch_bonus
    region_stats["linger_duration_minutes"] = total

    return total
```

---

### 📊 General Considerations

**Cost Impact**:
```
Fixed 5 min:      $0.63-1.25 per linger (always)
Adaptive 1 min:   $0.13-0.25 (cold start)
Adaptive 10 min:  $1.25-2.50 (hot session, max)
Hybrid 7 min:     $0.88-1.75 (typical hot session)
```

**Savings from Correct Prediction**:
- Catch within linger → Save 2-3 min spin-up ($0.25-0.40)
- Missed linger → Waste linger cost ($0.13-2.50)
- **Goal**: >50% catch rate (reuse > waste)

**Hit Rate Optimization**:
- Adaptive helps: linger long when needed, short when not
- Hybrid best: proactive + reactive = highest hit rate

**Safety Caps**:
- Max linger: 10 minutes (prevent runaway costs)
- Min linger: 1 minute (always give a chance)
- Absolute timeout: Worker pool auto-destroys at max linger

**Edge Cases**:
1. **First launch of day**: Tier 0 (1 min) - no pattern yet
2. **Long lunch break**: Warmdown to Tier 0 after 45 min idle
3. **Alternating regions**: Each region tracks independently
4. **Build failure during iteration**: Don't penalize - keep tier
5. **User switches projects**: Pattern resets per campaign

**Monitoring Metrics**:
- `linger_catch_rate`: catches / (catches + misses)
- `avg_linger_duration`: Average linger time used
- `total_linger_cost`: Sum of all linger costs
- `savings_from_catches`: Spin-up time avoided
- `tier_distribution`: Time spent in each tier

**UI Indicators**:
```
Launch output when catching:
✅ Caught lingering MECHA! (Tier 2: Hot - 5 min + 2 min bonus = 7 min earned)
→ Skipped 2.5 min spin-up
→ Next linger: 8 min (tier unchanged, bonus +1)

Launch output when linger expires:
⏰ Linger expired unused (7 min)
→ Next linger: 5 min (tier unchanged, bonus -1)

Launch output tier change:
🔥 Rapid iteration detected! Tier 1→2 (3 min → 5 min base)
```

**Cost Tracking**:
```
infra screen shows:
MECHA Linger Stats (Last 24 Hours)
Catches:        8 (67% hit rate)
Misses:         4
Avg Duration:   4.2 min
Total Cost:     $6.72
Savings:        $2.40 (spin-up avoided)
Net Benefit:    -$4.32 (still learning pattern)
```

---

### 🚀 Recommended Implementation Path

**Phase 1: Start Simple (Approach 1)** - 1-2 hours
- Implement success-based expansion
- 1 min → expand on catch → contract on miss
- Get baseline data on catch rates
- Validate cost tracking

**Phase 2: Add Pattern Detection (Approach 2)** - 2-3 hours
- Detect rapid iteration (2+ launches in 10 min)
- Bump to Tier 1 (3 min) when detected
- Monitor improvement in catch rate
- Tune tier thresholds based on data

**Phase 3: Hybrid Refinement (Approach 3)** - 1 hour
- Combine tier + catch bonus
- Fine-tune decay rates based on metrics
- Optimize for >60% catch rate
- Polish UI indicators

**Total Estimated Time**: 4-6 hours for full adaptive system

---

## 🔬 Research Findings - GCP Worker Pool Lifecycle & Persistence

**Date**: 2025-11-14
**Searches**: 4 comprehensive Bright Data searches on worker pool lifecycle, ccache persistence, VM reuse, timeout configuration

### Key Finding #1: No Built-in "Keep Alive" for Worker Pools

**Search**: "Cloud Build private worker pool prevent shutdown idle timeout keep alive"

**Findings**:
- GCP Cloud Build **does not provide** built-in "idle timeout" or "keep alive" configuration
- Worker pools spin up VMs for each build, then **auto-destroy immediately** after build completes
- No GCP-native way to keep worker pool VMs lingering after job completion

**Implications for MECHA LINGERING**:
- Must implement at **CLI level** (our code manages teardown timing)
- Cannot rely on GCP worker pool config for lingering
- Need to **delay teardown command** ourselves (don't call `gcloud builds worker-pools delete` until linger expires)

**Relevant GCP Docs**:
- https://docs.cloud.google.com/build/docs/private-pools/run-builds-in-private-pool
- https://docs.cloud.google.com/build/docs/private-pools/private-pools-overview
- Worker pools are **stateless** between builds by default

---

### Key Finding #2: ccache Persistence via Volume Mounts

**Search**: "ccache Docker Cloud Build persistent storage volume mount keep between builds"

**Findings**:
- **Docker cache mounts** (`RUN --mount=type=cache`) can persist ccache between builds
- **GCS buckets** can store ccache, mounted at build time
- **Persistent volumes** attached to worker pool VMs can keep ccache warm

**Best Practices**:
1. Use `RUN --mount=type=cache,target=/root/.ccache` in Dockerfile
2. Mount GCS bucket to worker pool for persistent cache
3. Cache mount survives **within same worker VM** (perfect for LINGERING!)

**Example from Search Results**:
```dockerfile
# Dockerfile with cache mount
RUN --mount=type=cache,target=/root/.ccache \
    ccache -s && \
    cd pytorch && python setup.py install && \
    ccache -s
```

**Implications for MECHA LINGERING**:
- **Warm ccache is the PRIZE** - this is WHY lingering matters!
- If worker VM lingers 5 min, ccache stays warm
- Next build on **same VM** hits cache → 10-100× faster
- Our LINGERING system **preserves the ccache warmth**

**Relevant Docs**:
- https://docs.docker.com/build/cache/optimize/
- https://stackoverflow.com/questions/39650056/using-ccache-when-building-inside-of-docker
- https://cloud.google.com/build/docs/optimize-builds/speeding-up-builds

---

### Key Finding #3: No VM Reuse Between Builds (By Default)

**Search**: "GCP Cloud Build private worker pool lifecycle VM reuse same instance multiple builds"

**Findings**:
- Each build gets **fresh worker VM** by default
- No GCP-native "VM pooling" or "warm instances"
- Worker pool **creates new VM**, runs build, **destroys VM** immediately

**Quote from GCP Docs**:
> "Each build runs on its own worker and is isolated from other workloads."

**Implications for MECHA LINGERING**:
- This is **exactly the problem** we're solving!
- GCP's default = no reuse = cold ccache every time
- MECHA LINGERING = **prevent worker pool deletion** for X minutes
- Keep **same VM alive** = warm ccache = fast rebuilds

**How MECHA LINGERING Works Around This**:
1. Build completes → GCP wants to destroy worker pool
2. **We don't call teardown yet** - delay for linger duration
3. User relaunches within linger → **reuses same worker pool** → warm ccache!
4. If linger expires → normal teardown

---

### Key Finding #4: Timeout Configuration (Build-Level, Not Pool-Level)

**Search**: "Cloud Build worker pool auto destroy timeout configuration prevent deletion"

**Findings**:
- Can configure **build timeout** (how long build can run)
- Can configure **queue timeout** (how long build waits for worker)
- **Cannot configure** "idle timeout" or "pool linger timeout"

**GCP Timeout Settings**:
```yaml
# cloudbuild.yaml
timeout: 3600s  # Build timeout (how long build runs)
queueTtl: 300s  # Queue timeout (how long waits for worker)
```

**What's NOT Available**:
```yaml
# ❌ This doesn't exist in GCP:
workerPoolIdleTimeout: 300s  # Keep pool alive 5 min after build
```

**Implications for MECHA LINGERING**:
- Must implement **application-level timeout** (our CLI tracks linger expiration)
- Use **datetime tracking** in mecha_hangar.json
- Worker pool **will not auto-destroy** while it exists (only when we call delete)
- Our CLI is **gatekeeper** for teardown timing

**Implementation Approach**:
```python
# In mecha_hangar.json:
{
  "linger_expires_at": "2025-11-14T10:18:00Z",  # 5 min from build complete
  "state": "RECEIVING_ADULATION"
}

# In CLI launch flow:
if now() < linger_expires_at:
    print("✅ Caught lingering MECHA!")
    # Skip setup, use existing worker pool
else:
    print("⏰ Linger expired, spinning up fresh MECHA")
    # Normal setup flow
```

---

### Key Finding #5: Cost & Practical Limits

**From Search Results & GCP Pricing**:
- c3-standard-176 costs **~$12.50/hour** = **$0.21/min**
- 5 min linger = **$1.05 cost**
- Spin-up time saved = **2-3 min** = **$0.42-0.63 value**
- **Break-even**: Need ~50% catch rate to justify lingering

**Practical Constraints**:
- Max linger: **10 minutes** (diminishing returns + cost control)
- Min linger: **1 minute** (always give user a chance)
- Recommended start: **1 minute** (adaptive expansion based on usage)

**Cost Optimization Strategy**:
- Start conservative (1 min)
- Expand on successful catches (user is iterating!)
- Contract on missed lingers (user stopped iterating)
- Cap at 10 min (prevent runaway costs)

---

### Summary: MECHA LINGERING is Application-Level Feature

**What GCP Provides**:
- Worker pools (spin up, run build, auto-destroy)
- Cache mounts (persist within same VM)
- Timeout configs (build duration only)

**What GCP Does NOT Provide**:
- ❌ Idle timeout / keep alive for worker pools
- ❌ VM reuse between builds
- ❌ Automatic pool lingering

**What WE Must Build**:
- ✅ Track linger expiration in mecha_hangar.json
- ✅ Delay teardown command for linger duration
- ✅ Check linger status before setup (reuse if still alive)
- ✅ Adaptive linger duration (1-10 min based on usage)
- ✅ Cost tracking and hit rate metrics

**The MECHA LINGERING Value Proposition**:
1. User triggers build → worker pool spins up (2-3 min)
2. Build completes → **instead of immediate teardown**, MECHA lingers (1-10 min)
3. User makes quick code fix, relaunches within linger
4. **Reuses same worker pool** → ccache warm → build 10-100× faster!
5. If no relaunch → linger expires → normal teardown

**This is a CLI-level feature, not a GCP feature. We own the timing!** 🦾💎

---

**End of Plan**

Ready to implement Phase 1? Let's give our MECHAs the adulation they deserve! 🦾✨
