# Enhanced Build Stats Implementation Plan

**Date**: 2025-11-14
**Goal**: Store full resource samples + calculated anomaly metrics in campaign_stats.json

## 🎯 Executive Summary

We're enhancing campaign stats to include:
1. **Full resource timeline** (all 21 samples from build-stats.json)
2. **Spike/trough detection** (anomaly detection)
3. **Consistency scoring** (build stability metrics)

**Key Insight**: The ending file (`/build-stats.json`) ALREADY contains all the data we need! We just need to grab it from CloudBuild logs and store it alongside our calculated aggregates.

---

## 📊 Current State Analysis

### What We Have Now

**Dockerfile** (training/images/pytorch-clean/Dockerfile):
```dockerfile
# Step 30: Resource monitoring during PyTorch build
# Collects samples every 60 seconds → /build-stats.jsonl
# Converts to JSON → /build-stats.json
# Outputs to logs between BUILD_STATS_JSON_START/END markers
```

**campaign_stats.py** (`parse_build_stats_from_logs()`):
- ✅ Fetches CloudBuild logs
- ✅ Finds BUILD_STATS_JSON_START/END markers
- ✅ Parses the samples JSON
- ✅ Calculates aggregates (avg, peak, totals)
- ❌ **THROWS AWAY the full samples array!**

### What Gets Lost

From the logs we get:
```json
{
  "samples": [
    {"ts":"2025-11-14T03:10:14+00:00","cpu":0,"mem_used_gb":1,...},
    {"ts":"2025-11-14T03:11:14+00:00","cpu":6,"mem_used_gb":13,...},
    ... 21 total samples ...
  ]
}
```

But we only store:
```json
{
  "cpu_utilization_avg": 48,
  "cpu_utilization_peak": 57,
  "memory_used_avg_gb": 52,
  "memory_used_peak_gb": 137
}
```

**The samples themselves are discarded!** We need to KEEP them!

---

## 🎯 Implementation Strategy

### The Simple Solution

**Instead of throwing away samples after calculating aggregates, KEEP THEM!**

Current flow:
```
samples → calculate aggregates → return aggregates (samples discarded!)
```

New flow:
```
samples → calculate aggregates + detect anomalies + calculate consistency
        → return {aggregates, samples, anomalies, consistency}
```

---

## 📋 Implementation Checklist

### Phase 1: Update parse_build_stats_from_logs()

**File**: `training/cli/launch/mecha/campaign_stats.py`

- [ ] **Step 1.1**: Store the full samples array (don't discard it!)
  ```python
  # After parsing samples from logs:
  metrics["resource_samples"] = samples  # NEW: Keep them!
  metrics["sample_count"] = len(samples)
  ```

  **Reasoning**: The samples are already parsed from logs - we just need to include them in the return dict instead of throwing them away!

- [ ] **Step 1.2**: Calculate spike/trough anomalies
  ```python
  # NEW FUNCTION: detect_anomalies(samples)
  # Returns: {"spikes": [...], "troughs": [...]}
  ```

  **Logic**:
  - Loop through samples with index
  - For each metric (cpu, mem_used_gb):
    - Compare sample[i] to sample[i-1]
    - If delta > threshold (e.g., 50% change), record as spike/trough
    - Store: from_value, to_value, delta, percent_change, timestamp, severity

  **Thresholds**:
  - CPU spike: >30% increase in one sample
  - Memory spike: >40GB or >50% increase
  - Severity: HIGH (>100% change), MEDIUM (>50%), LOW (>30%)

- [ ] **Step 1.3**: Calculate consistency score
  ```python
  # NEW FUNCTION: calculate_consistency_score(samples)
  # Returns: {"cpu": {...}, "memory": {...}, "overall": ...}
  ```

  **Logic**:
  - Extract all CPU values → calculate variance, std_dev, CV
  - Extract all memory values → calculate variance, std_dev, CV
  - Consistency score = 1 - normalized(CV)
    - CV (coefficient of variation) = std_dev / mean
    - Lower CV = more consistent = higher score
    - Score range: 0.0 (chaos) to 1.0 (perfect consistency)

- [ ] **Step 1.4**: Return enhanced metrics dict
  ```python
  metrics = {
      # Existing aggregates (keep these!)
      "cpu_utilization_avg": ...,
      "cpu_utilization_peak": ...,
      "memory_used_avg_gb": ...,
      "memory_used_peak_gb": ...,
      "io_read_total_mb": ...,
      "io_write_total_mb": ...,

      # NEW: Full timeline
      "resource_samples": samples,
      "sample_count": len(samples),

      # NEW: Anomaly detection
      "anomalies": {
          "spikes": [...],
          "troughs": [...]
      },

      # NEW: Consistency scoring
      "consistency_score": {
          "cpu": {"score": 0.72, "variance": ..., "std_dev": ..., "cv": ...},
          "memory": {"score": 0.45, ...},
          "overall": 0.58
      }
  }
  ```

### Phase 2: Update record_build_result() to store enhanced metrics

**File**: `training/cli/launch/mecha/campaign_stats.py`

- [ ] **Step 2.1**: Add new fields to build_record
  ```python
  build_record = {
      # ... existing fields ...

      # NEW: Resource timeline
      "resource_samples": resource_metrics.get("resource_samples", []),
      "sample_count": resource_metrics.get("sample_count", 0),

      # NEW: Anomaly detection
      "anomalies": resource_metrics.get("anomalies", {"spikes": [], "troughs": []}),

      # NEW: Consistency scoring
      "consistency_score": resource_metrics.get("consistency_score", {
          "cpu": {"score": 0, "variance": 0, "std_dev": 0, "cv": 0},
          "memory": {"score": 0, "variance": 0, "std_dev": 0, "cv": 0},
          "overall": 0
      })
  }
  ```

### Phase 3: Update update_build_completion() (straggly bit!)

**File**: `training/cli/launch/mecha/campaign_stats.py`

- [ ] **Step 3.1**: Same as Step 2.1 - add new fields when updating build record

  **Reasoning**: This is the alternate code path that also updates build records. We found this straggly bit earlier - it needs the same enhancements!

### Phase 4: Implement helper functions

**File**: `training/cli/launch/mecha/campaign_stats.py`

- [ ] **Step 4.1**: Implement `detect_anomalies(samples)`
  ```python
  def detect_anomalies(samples: List[dict]) -> dict:
      """
      Detect spikes and troughs in resource usage.

      Spike: Sudden increase (>50% or >40GB for memory)
      Trough: Sudden decrease (>30% drop)

      Returns:
          {
              "spikes": [
                  {
                      "type": "memory",
                      "sample_index": 10,
                      "from_value": 52,
                      "to_value": 137,
                      "delta": 85,
                      "percent_change": 163.5,
                      "timestamp": "2025-11-14T03:19:15+00:00",
                      "severity": "HIGH"
                  },
                  ...
              ],
              "troughs": [...]
          }
      """
      spikes = []
      troughs = []

      for i in range(1, len(samples)):
          prev = samples[i-1]
          curr = samples[i]

          # Check CPU
          cpu_delta = curr["cpu"] - prev["cpu"]
          cpu_pct_change = (cpu_delta / prev["cpu"] * 100) if prev["cpu"] > 0 else 0

          if cpu_pct_change > 30:  # Spike threshold
              severity = "HIGH" if cpu_pct_change > 100 else "MEDIUM" if cpu_pct_change > 50 else "LOW"
              spikes.append({
                  "type": "cpu",
                  "sample_index": i,
                  "from_value": prev["cpu"],
                  "to_value": curr["cpu"],
                  "delta": cpu_delta,
                  "percent_change": round(cpu_pct_change, 1),
                  "timestamp": curr["ts"],
                  "severity": severity
              })

          # Check Memory (similar logic)
          mem_delta = curr["mem_used_gb"] - prev["mem_used_gb"]
          mem_pct_change = (mem_delta / prev["mem_used_gb"] * 100) if prev["mem_used_gb"] > 0 else 0

          if mem_pct_change > 50 or mem_delta > 40:  # Spike threshold
              severity = "HIGH" if mem_pct_change > 100 else "MEDIUM" if mem_pct_change > 50 else "LOW"
              spikes.append({
                  "type": "memory",
                  "sample_index": i,
                  "from_value": prev["mem_used_gb"],
                  "to_value": curr["mem_used_gb"],
                  "delta": mem_delta,
                  "percent_change": round(mem_pct_change, 1),
                  "timestamp": curr["ts"],
                  "severity": severity
              })

          # Check for troughs (negative deltas)
          if cpu_pct_change < -30:
              troughs.append({
                  "type": "cpu",
                  "sample_index": i,
                  "from_value": prev["cpu"],
                  "to_value": curr["cpu"],
                  "delta": cpu_delta,
                  "percent_change": round(cpu_pct_change, 1),
                  "timestamp": curr["ts"]
              })

          if mem_pct_change < -30:
              troughs.append({
                  "type": "memory",
                  "sample_index": i,
                  "from_value": prev["mem_used_gb"],
                  "to_value": curr["mem_used_gb"],
                  "delta": mem_delta,
                  "percent_change": round(mem_pct_change, 1),
                  "timestamp": curr["ts"]
              })

      return {
          "spikes": spikes,
          "troughs": troughs
      }
  ```

- [ ] **Step 4.2**: Implement `calculate_consistency_score(samples)`
  ```python
  def calculate_consistency_score(samples: List[dict]) -> dict:
      """
      Calculate consistency scores for resource usage.

      Consistency score = 1 - normalized(CV)
      where CV (coefficient of variation) = std_dev / mean

      Score interpretation:
      - 0.9-1.0: Extremely consistent
      - 0.7-0.9: Consistent
      - 0.5-0.7: Moderate variance
      - 0.0-0.5: High variance

      Returns:
          {
              "cpu": {
                  "score": 0.72,
                  "variance": 234.5,
                  "std_dev": 15.3,
                  "cv": 0.32
              },
              "memory": {...},
              "overall": 0.58
          }
      """
      import statistics

      cpu_values = [s["cpu"] for s in samples if "cpu" in s and s["cpu"] > 0]
      mem_values = [s["mem_used_gb"] for s in samples if "mem_used_gb" in s and s["mem_used_gb"] > 0]

      def calculate_metrics(values):
          if len(values) < 2:
              return {"score": 0, "variance": 0, "std_dev": 0, "cv": 0}

          mean = statistics.mean(values)
          variance = statistics.variance(values)
          std_dev = statistics.stdev(values)
          cv = std_dev / mean if mean > 0 else 0

          # Consistency score: inverse of CV, clamped to 0-1
          # CV of 0 = score 1.0 (perfect)
          # CV of 1 = score 0.0 (chaos)
          score = max(0, min(1, 1 - cv))

          return {
              "score": round(score, 2),
              "variance": round(variance, 1),
              "std_dev": round(std_dev, 1),
              "cv": round(cv, 2)
          }

      cpu_metrics = calculate_metrics(cpu_values)
      memory_metrics = calculate_metrics(mem_values)
      overall_score = round((cpu_metrics["score"] + memory_metrics["score"]) / 2, 2)

      return {
          "cpu": cpu_metrics,
          "memory": memory_metrics,
          "overall": overall_score
      }
  ```

- [ ] **Step 4.3**: Update `parse_build_stats_from_logs()` to use helper functions
  ```python
  # Inside parse_build_stats_from_logs(), after parsing samples:

  # Calculate anomalies
  anomalies = detect_anomalies(samples)

  # Calculate consistency
  consistency = calculate_consistency_score(samples)

  # Add to metrics dict
  metrics["resource_samples"] = samples
  metrics["sample_count"] = len(samples)
  metrics["anomalies"] = anomalies
  metrics["consistency_score"] = consistency
  ```

### Phase 5: Update DEBUG output

**File**: `training/cli/launch/mecha/campaign_stats.py`

- [ ] **Step 5.1**: Update DEBUG print for resource metrics
  ```python
  # After line 365 (existing debug output):
  print(f"DEBUG: RESOURCE_METRICS_JSON: {json.dumps(metrics, indent=2)}")

  # Add additional debug for anomalies:
  if "anomalies" in metrics:
      spike_count = len(metrics["anomalies"]["spikes"])
      trough_count = len(metrics["anomalies"]["troughs"])
      print(f"DEBUG: Detected {spike_count} spikes and {trough_count} troughs")

  if "consistency_score" in metrics:
      overall = metrics["consistency_score"]["overall"]
      print(f"DEBUG: Overall consistency score: {overall}")
  ```

---

## 🧪 Testing Strategy

**Note**: No checkboxes for testing, just descriptions!

### Test 1: Verify samples are stored

**What to test**:
1. Run a build that completes (success or failure)
2. Check campaign_stats.json for latest build
3. Verify `resource_samples` field exists and contains array
4. Verify `sample_count` matches array length

**Expected result**:
```json
{
  "build_id": "abc123...",
  "resource_samples": [
    {"ts": "...", "cpu": 0, "mem_used_gb": 1, ...},
    {"ts": "...", "cpu": 6, "mem_used_gb": 13, ...},
    ... 21 total ...
  ],
  "sample_count": 21
}
```

### Test 2: Verify anomaly detection

**What to test**:
1. Look at build 84fbeb34 (known to have 163% memory spike)
2. Check if spike was detected at sample 10
3. Verify severity = "HIGH"

**Expected result**:
```json
{
  "anomalies": {
    "spikes": [
      {
        "type": "memory",
        "sample_index": 10,
        "from_value": 52,
        "to_value": 137,
        "delta": 85,
        "percent_change": 163.5,
        "severity": "HIGH"
      }
    ]
  }
}
```

### Test 3: Verify consistency scoring

**What to test**:
1. Check consistency_score field exists
2. Verify all sub-scores present (cpu, memory, overall)
3. Check scores are in 0-1 range

**Expected result**:
```json
{
  "consistency_score": {
    "cpu": {
      "score": 0.72,
      "variance": 234.5,
      "std_dev": 15.3,
      "cv": 0.32
    },
    "memory": {
      "score": 0.45,
      "variance": 1156.7,
      "std_dev": 34.0,
      "cv": 0.65
    },
    "overall": 0.58
  }
}
```

### Test 4: Verify DEBUG output appears

**What to test**:
1. Trigger a build completion
2. Check logs for DEBUG messages
3. Verify RESOURCE_METRICS_JSON shows full structure
4. Verify spike/trough counts are printed
5. Verify consistency score is printed

**Expected output**:
```
DEBUG: Parsing resource stats from logs for 84fbeb34-276...
DEBUG: Found BUILD_STATS_JSON markers in logs! Parsing...
DEBUG: Parsed 21 resource samples from logs
DEBUG: ✅ Resource metrics parsed! CPU avg: 48.5%, Memory peak: 137GB
DEBUG: RESOURCE_METRICS_JSON: {full JSON here}
DEBUG: Detected 2 spikes and 0 troughs
DEBUG: Overall consistency score: 0.58
```

### Test 5: Verify no errors on builds without samples

**What to test**:
1. Look at old builds (before monitoring was added)
2. Verify parse_build_stats_from_logs() handles missing markers gracefully
3. Verify empty arrays/zero scores returned

**Expected result**:
```json
{
  "resource_samples": [],
  "sample_count": 0,
  "anomalies": {"spikes": [], "troughs": []},
  "consistency_score": {
    "cpu": {"score": 0, "variance": 0, "std_dev": 0, "cv": 0},
    "memory": {"score": 0, "variance": 0, "std_dev": 0, "cv": 0},
    "overall": 0
  }
}
```

---

## 📏 Size Impact Analysis

### JSON Size Estimates

**Per build**:
- Current aggregates: ~200 bytes
- 21 samples: ~2,100 bytes (21 × 100 bytes each)
- Anomalies: ~200 bytes (2-4 anomalies typical)
- Consistency: ~150 bytes
- **Total per build**: ~2,650 bytes (~2.6 KB)

**Campaign stats file**:
- Keeps last 100 builds per region
- 3 regions typical
- Current size: ~60 KB
- New size: ~60 KB + (2.6 KB × 100 × 3) = ~840 KB

**Verdict**: Acceptable! 840 KB is small for rich analytics!

---

## 🎯 Success Criteria

Implementation is complete when:

1. ✅ Full samples array stored in campaign_stats.json
2. ✅ Spike/trough detection working (verified on build 84fbeb34)
3. ✅ Consistency scores calculated and stored
4. ✅ DEBUG output shows enhanced metrics
5. ✅ No errors on old builds without samples
6. ✅ All existing functionality still works (no regressions)

---

## 🚀 Deployment Notes

**This is a backwards-compatible change!**

- Old builds without samples → get empty arrays/zero scores
- New builds with samples → get full enhanced metrics
- No migration needed!
- No schema breaking changes!

**Next build after deployment will show full metrics!** 📊

---

## 📝 Implementation Order

Execute in this order:

1. Phase 4 first (implement helper functions)
2. Phase 1 (update parse_build_stats_from_logs)
3. Phase 5 (update DEBUG output)
4. Phase 2 (update record_build_result)
5. Phase 3 (update update_build_completion)
6. Test!

---

## 🎉 Expected Output After Implementation

When the next build completes, you'll see:

```
DEBUG: Fetching CloudBuild metrics for acaba721-ab8... (region: us-west2)
DEBUG: ✅ CloudBuild metrics fetched! Queue: 645s, Working: 628s
DEBUG: CLOUDBUILD_METRICS_JSON: {...}

DEBUG: Parsing resource stats from logs for acaba721-ab8...
DEBUG: Found BUILD_STATS_JSON markers in logs! Parsing...
DEBUG: Parsed 21 resource samples from logs
DEBUG: ✅ Resource metrics parsed! CPU avg: 48.5%, Memory peak: 137GB
DEBUG: RESOURCE_METRICS_JSON: {
  "cpu_utilization_avg": 48.5,
  "cpu_utilization_peak": 57,
  "memory_used_avg_gb": 52.3,
  "memory_used_peak_gb": 137,
  "io_read_total_mb": 0,
  "io_write_total_mb": 140539,
  "resource_samples": [... 21 samples ...],
  "sample_count": 21,
  "anomalies": {
    "spikes": [
      {"type": "memory", "sample_index": 10, "delta": 85, "percent_change": 163.5, "severity": "HIGH"},
      {"type": "cpu", "sample_index": 12, "delta": 16, "percent_change": 38.1, "severity": "LOW"}
    ],
    "troughs": []
  },
  "consistency_score": {
    "cpu": {"score": 0.72, "variance": 234.5, "std_dev": 15.3, "cv": 0.32},
    "memory": {"score": 0.45, "variance": 1156.7, "std_dev": 34.0, "cv": 0.65},
    "overall": 0.58
  }
}
DEBUG: Detected 2 spikes and 0 troughs
DEBUG: Overall consistency score: 0.58
```

And in campaign_stats.json:

```json
{
  "regions": {
    "us-west2": {
      "recent_builds": [
        {
          "build_id": "acaba721-ab81-430c-8818-a67a53aac01b",
          "status": "SUCCESS",
          "cpu_utilization_avg": 48.5,
          "cpu_utilization_peak": 57,
          "memory_used_avg_gb": 52.3,
          "memory_used_peak_gb": 137,
          "resource_samples": [
            {"ts": "2025-11-14T03:10:14+00:00", "cpu": 0, "mem_used_gb": 1, ...},
            {"ts": "2025-11-14T03:11:14+00:00", "cpu": 6, "mem_used_gb": 13, ...},
            ... all 21 samples ...
          ],
          "sample_count": 21,
          "anomalies": {
            "spikes": [...],
            "troughs": []
          },
          "consistency_score": {
            "cpu": {"score": 0.72, ...},
            "memory": {"score": 0.45, ...},
            "overall": 0.58
          }
        }
      ]
    }
  }
}
```

**FULL RESOURCE TIMELINE CAPTURED!** 🎉📊
