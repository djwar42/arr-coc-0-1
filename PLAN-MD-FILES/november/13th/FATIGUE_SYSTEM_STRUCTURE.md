# MECHA Fatigue System Structure - Complete Design

## Core Principle

**Two Levels of Fatigue Tracking:**

1. **Build Level** (individual build) → `fatigues: []` array with MULTIPLE fatigue events
2. **Region Level** (aggregate stats) → `last_fatigue_*` fields with MOST RECENT fatigue

---

## Complete JSON Structure

```json
{
  "campaign_start": 1763026000.0,
  "last_updated": 1763030000.0,
  "total_builds_all_regions": 47,

  "regions": {
    "us-west2": {
      // Build Counts
      "total_builds": 45,
      "successes": 42,
      "failures": 3,
      "success_rate": 93.3,

      // Timing Stats
      "total_duration_minutes": 562.5,
      "total_queue_wait_minutes": 135.0,
      "avg_duration_minutes": 12.5,
      "avg_queue_wait_minutes": 3.0,
      "fastest_minutes": 8.2,
      "slowest_minutes": 18.7,

      // Last Build Error (aggregate)
      "last_error": "Docker layer pull timeout (step 45/72)",
      "last_error_time": 1763028000.0,

      // Last Fatigue Event (aggregate) - REGION LEVEL
      "fatigue_incidents": 2,
      "last_fatigue_reason": "Queue timeout - 45min QUEUED",
      "last_fatigue_reason_code": "queue-timeout",
      "last_fatigue_time": 1763026500.0,

      // Performance
      "current_streak": 12,
      "last_used": 1763029950.0,

      // Recent Builds (rich detail) - BUILD LEVEL
      "recent_builds": [
        {
          "timestamp": 1763029950.0,
          "build_id": "abc123-xyz789",
          "build_type": "pytorch-clean",
          "machine_type": "c3-standard-176",

          "success": false,
          "status": "TIMEOUT",

          "duration_seconds": 2700,
          "duration_minutes": 45.0,
          "queue_wait_seconds": 2700,
          "queue_wait_minutes": 45.0,
          "working_seconds": 0,

          "total_steps": 72,
          "failed_step": null,

          "spot_price_per_hour": 8.53,
          "cost_usd": 6.40,

          "error": "Build stuck in QUEUED for 45 minutes",
          "timeout_reason": "QUEUED",

          // FATIGUES ARRAY - Multiple fatigue events in THIS build!
          "fatigues": [
            {
              "fatigue_reason": "Queue timeout - 45min QUEUED",
              "fatigue_reason_code": "queue-timeout",
              "fatigue_time": 1763029950.0,
              "fatigue_duration_hours": 4,
              "fatigue_type": "FATIGUED"
            }
          ]
        },
        {
          "timestamp": 1763026000.0,
          "build_id": "def456-uvw123",
          "build_type": "pytorch-clean",
          "machine_type": "c3-standard-176",

          "success": false,
          "status": "FAILURE",

          "duration_seconds": 1800,
          "duration_minutes": 30.0,
          "queue_wait_seconds": 300,
          "queue_wait_minutes": 5.0,
          "working_seconds": 1500,

          "total_steps": 72,
          "failed_step": 45,

          "spot_price_per_hour": 8.53,
          "cost_usd": 4.27,

          "error": "Docker layer pull timeout",
          "timeout_reason": null,

          // This build had NO fatigue events (normal failure)
          "fatigues": []
        }
      ]
    }
  }
}
```

---

## Field Naming Conventions

### Build-Level Fatigue Object (inside `fatigues[]` array)

**Plain names** (no "last_" prefix):

```python
{
  "fatigue_reason": str,           # Human-readable: "Queue timeout - 45min QUEUED"
  "fatigue_reason_code": str,      # Machine-readable: "queue-timeout"
  "fatigue_time": float,           # Unix timestamp when fatigue occurred
  "fatigue_duration_hours": int,   # How long fatigued: 4 or 24
  "fatigue_type": str             # "FATIGUED" (4h) or "EXHAUSTED" (24h)
}
```

**Why plain names?**
- It's ONE fatigue event among potentially many
- Not the "last" one - just "a" fatigue
- Clear context: inside `fatigues[]` array

### Region-Level Fatigue Fields (aggregate)

**"last_" prefix** (matches `last_error`, `last_used` pattern):

```python
{
  "fatigue_incidents": int,              # Total count across ALL builds
  "last_fatigue_reason": str,            # Most recent fatigue reason
  "last_fatigue_reason_code": str,       # Most recent fatigue code
  "last_fatigue_time": float            # When most recent fatigue occurred
}
```

**Why "last_" prefix?**
- Aggregate summary across MANY builds
- Tracks MOST RECENT fatigue
- Matches existing pattern: `last_error`, `last_error_time`, `last_used`

---

## When Fatigues Get Recorded

### Scenario 1: Build Times Out (Queue Godzilla)

**Build Monitor detects 45-min QUEUED timeout:**

1. **Record to build** (via `record_build_result()`):
   ```python
   build_record["fatigues"] = [
       {
           "fatigue_reason": "Queue timeout - 45min QUEUED",
           "fatigue_reason_code": "queue-timeout",
           "fatigue_time": now,
           "fatigue_duration_hours": 4,
           "fatigue_type": "FATIGUED"
       }
   ]
   ```

2. **Update region stats** (via `record_fatigue_event()`):
   ```python
   region_stats["fatigue_incidents"] += 1
   region_stats["last_fatigue_reason"] = "Queue timeout - 45min QUEUED"
   region_stats["last_fatigue_reason_code"] = "queue-timeout"
   region_stats["last_fatigue_time"] = now
   ```

### Scenario 2: Beacon Timeout (Pool Creation Fails)

**MECHA acquire fails during Fleet Blast:**

1. **No build record** (pool never created, no build happened)

2. **Only update region stats** (via `record_fatigue_event()`):
   ```python
   region_stats["fatigue_incidents"] += 1
   region_stats["last_fatigue_reason"] = "Pool creation timeout"
   region_stats["last_fatigue_reason_code"] = "beacon-timeout"
   region_stats["last_fatigue_time"] = now
   ```

### Scenario 3: Normal Build Failure (No Fatigue)

**Build fails but no fatigue triggered:**

```python
build_record["fatigues"] = []  # Empty array - no fatigue events
```

Region stats unchanged (no fatigue event).

---

## Implementation Changes Required

### 1. Update `record_build_result()` signature

**Add fatigue parameter:**

```python
def record_build_result(
    region: str,
    success: bool,
    build_id: str,
    # ... existing params ...
    fatigues: List[Dict] = None  # NEW! Optional fatigue events for this build
):
```

### 2. Update build record creation

```python
build_record = {
    "timestamp": now,
    "build_id": build_id,
    # ... all existing fields ...
    "fatigues": fatigues or []  # NEW! Empty array if no fatigues
}
```

### 3. Fix region initialization (both functions)

**Change from:**
```python
"fatigue_reason": None,
"fatigue_time": None,
```

**To:**
```python
"last_fatigue_reason": None,
"last_fatigue_reason_code": None,
"last_fatigue_time": None,
```

### 4. Fix `record_fatigue_event()` writes

**Change from:**
```python
region_stats["fatigue_reason"] = ...
region_stats["fatigue_reason_code"] = ...
region_stats["fatigue_time"] = ...
```

**To:**
```python
region_stats["last_fatigue_reason"] = ...
region_stats["last_fatigue_reason_code"] = ...
region_stats["last_fatigue_time"] = ...
```

### 5. Update CAMPAIGN_STATS_STRUCTURE.md

Add `fatigues[]` to build record example.
Fix region field names to use `last_` prefix.

---

## Call Sites That Need Updates

### In `core.py` (BuildQueueMonitor callback)

**When queue timeout detected:**

```python
# Current: Only calls record_fatigue_event()
record_mecha_timeout(registry, region, reason, reason_code, error_msg, build_id)

# New: ALSO pass fatigue to record_build_result()
fatigue_event = {
    "fatigue_reason": "Queue timeout - 45min QUEUED",
    "fatigue_reason_code": "queue-timeout",
    "fatigue_time": time.time(),
    "fatigue_duration_hours": 4,
    "fatigue_type": "FATIGUED"
}

record_build_result(
    region=region,
    success=False,
    # ... other params ...
    fatigues=[fatigue_event]  # NEW!
)

# Still call this for MECHA registry + godzilla log
record_fatigue_event(region, reason, reason_code, error_msg)
```

### In `mecha_acquire.py` (Beacon timeouts)

**No change needed!** Beacon timeouts have no build, so only call `record_fatigue_event()`.

---

## Migration Strategy

**For existing JSON files:**

Old structure will have:
```json
"fatigue_reason": "...",
"fatigue_time": 123456
```

Code reads these and auto-migrates to:
```json
"last_fatigue_reason": "...",
"last_fatigue_reason_code": null,
"last_fatigue_time": 123456
```

**No data loss!** Just field rename on first load.

---

## Benefits of New System

1. **Build-level detail**: See WHICH builds had fatigue events
2. **Multiple fatigues per build**: Future-proof for complex scenarios
3. **Consistent naming**: `last_*` at region level, plain names in arrays
4. **Complete history**: Last 100 builds with fatigue details
5. **Easy windowing**: "How many fatigues in last 7 days?" → scan `recent_builds`

---

## Questions to Resolve

1. **Fatigue event structure**: Include `fatigue_duration_hours` and `fatigue_type`?
2. **Build without build_id**: Beacon timeouts have no build - handle gracefully?
3. **Retroactive fatigues**: Mark past builds in `recent_builds` with fatigues?

---

**Status**: Design complete, ready for implementation! 🚀
