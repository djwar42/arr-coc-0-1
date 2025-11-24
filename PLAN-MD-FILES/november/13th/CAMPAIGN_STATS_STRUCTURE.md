# Campaign Stats JSON Structure

**File**: `training/cli/launch/mecha/data/campaign_stats.json`

Tracks MECHA build performance and fatigue incidents across all regions.

## Complete Structure Example

```json
{
  "campaign_start": 1763026000.0,
  "last_updated": 1763030000.0,
  "total_builds_all_regions": 47,

  "regions": {
    "us-west2": {
      "total_builds": 45,
      "successes": 42,
      "failures": 3,
      "success_rate": 93.3,

      "total_duration_minutes": 562.5,
      "total_queue_wait_minutes": 135.0,
      "avg_duration_minutes": 12.5,
      "avg_queue_wait_minutes": 3.0,
      "fastest_minutes": 8.2,
      "slowest_minutes": 18.7,

      "last_error": "Docker layer pull timeout (step 45/72)",
      "last_error_time": 1763028000.0,

      "fatigue_incidents": 2,
      "last_fatigue_reason": "Queue timeout - 45min QUEUED",
      "last_fatigue_time": 1763026500.0,

      "current_streak": 12,
      "last_used": 1763029950.0
    },

    "asia-northeast1": {
      "total_builds": 2,
      "successes": 0,
      "failures": 2,
      "success_rate": 0.0,

      "total_duration_minutes": 90.0,
      "total_queue_wait_minutes": 90.0,
      "avg_duration_minutes": 45.0,
      "avg_queue_wait_minutes": 45.0,
      "fastest_minutes": 45.0,
      "slowest_minutes": 45.0,

      "last_error": "Worker pool timeout - 45min QUEUED",
      "last_error_time": 1763023288.0,

      "fatigue_incidents": 1,
      "last_fatigue_reason": "Worker pool timeout",
      "last_fatigue_time": 1763023288.0,

      "current_streak": -2,
      "last_used": 1763023288.0
    }
  }
}
```

## Field Descriptions

### Global Fields
- `campaign_start` - Unix timestamp when first build was recorded
- `last_updated` - Unix timestamp of last stats update
- `total_builds_all_regions` - Total builds across ALL regions

### Per-Region Stats

**Build Counts:**
- `total_builds` - How many times this region was used
- `successes` - Successful builds
- `failures` - Failed builds
- `success_rate` - Success percentage (0-100)

**Timing Stats:**
- `total_duration_minutes` - Sum of all build durations
- `total_queue_wait_minutes` - Sum of all queue wait times
- `avg_duration_minutes` - Average total build time
- `avg_queue_wait_minutes` - Average queue wait time
- `fastest_minutes` - Fastest build ever
- `slowest_minutes` - Slowest build ever

**Error Tracking:**
- `last_error` - Concise last build error (max 80 chars)
- `last_error_time` - When last error occurred

**Fatigue/Godzilla Tracking:**
- `fatigue_incidents` - Total fatigue events (queue timeouts, etc.)
- `last_fatigue_reason` - Concise last fatigue reason (max 60 chars)
- `last_fatigue_time` - When last fatigue occurred

**Performance:**
- `current_streak` - Positive = consecutive wins, Negative = consecutive losses
- `last_used` - When region was last used

## Key Insights Derived

**Best Regions:**
```
us-west2: 93.3% success (42/45), 12-streak, 12.5min avg
europe-west1: 88.0% success (22/25), 5-streak, 11.8min avg
```

**Problematic Regions:**
```
asia-northeast1: 0% success (0/2), -2-streak, 1 fatigue event, OUTLAWED
us-central1: 70% success (14/20), 2 fatigue events
```

**Speed Champions:**
```
europe-north1: 9.5min avg (fastest)
us-west4: 10.2min avg
us-west2: 12.5min avg
```

**Fatigue Incidents:**
```
asia-northeast1: 1 incident (Worker pool timeout)
us-central1: 2 incidents (Queue timeouts)
```

## Integration Points

**When Fatigue Occurs** (`mecha_hangar.py:record_mecha_timeout`):
```python
from cli.launch.mecha.campaign_stats import record_fatigue_event
record_fatigue_event(region, reason, error_message)
```

**When Build Completes** (TODO - integrate into `core.py`):
```python
from cli.launch.mecha.campaign_stats import record_build_result
record_build_result(
    region="us-west2",
    success=True,
    duration_minutes=12.3,
    queue_wait_minutes=2.1,
    error_message=None
)
```

## Future Use Cases

1. **Intelligent MECHA Selection**: Weight price + success_rate + speed
2. **Auto-Outlaw**: Regions with >3 fatigue incidents + <50% success
3. **Champion Badges**: Award "Speed King", "Reliability Champion", "Cost Leader"
4. **Historical Trends**: Track if regions are improving/declining over time
5. **Cost Optimization**: Calculate actual $/success for each region

## File Size

- ~30-40 lines per region
- 18 regions × 40 lines = ~720 lines total
- ~35KB file size (tiny!)
