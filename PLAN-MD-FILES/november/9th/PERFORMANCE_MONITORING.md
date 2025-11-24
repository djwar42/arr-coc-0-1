# Vervaekean Performance Coupling Analysis

**NEW APPROACH**: All analysis printed to console on exit - NO files created!

## What's Being Monitored

### 🚀 Launch Screen (CRITICAL - 10-30 min operations!)
- `build_base_image_cloudbuild` (Docker) - Base image build ~4-5 min
- `build_training_image_cloudbuild` (Docker) - Training image build ~10-15 min

### 🗑️  Teardown Screen
- `teardown_all_resources` (GCP) - Full resource deletion tracking

### 🏗️  Infrastructure Screen
- `check_infrastructure_status` (GCP) - Bucket, registry, SA, quota checks

### ⚙️  Setup Screen (Already monitored)
- `check_wandb_prerequisites` (API) - W&B API key, project verification
- `check_infrastructure` (GCP) - GCS bucket, Artifact Registry, Service Account checks

### 📊 Monitor Screen (Already monitored)
- Lazy loads: security scan, runner executions, vertex jobs, active/completed runs
- Staggered refreshes: All tables refresh every 30s with separate fetch/UI operations

## How It Works

### Automatic Monitoring
Performance monitoring starts automatically when you run the TUI:

```bash
python training/tui.py
# 📊 Performance monitoring enabled
# Exit summary will print on TUI close
```

### What Gets Tracked

Every monitored operation logs:
- Operation name, category, duration
- Thread (MainThread vs workers)
- Blocking status (>100ms on main thread)
- CPU usage during operation

**Categories**:
- `Docker` - Image builds, pushes
- `GCP` - gcloud commands, Cloud Build, Artifact Registry, GCS
- `API` - W&B API calls
- `UI` - Widget updates, table refreshes

**Blocking Detection**:
- Main thread >100ms = blocking (UI freezes)
- Worker thread >5s = slow (but doesn't block UI)
- Background >30s = very slow

### Exit Summary Format

On TUI exit, prints comprehensive Vervaekean coupling analysis:

```
╔═══════════════════════════════════════════════════════════════════════════════
║
║  🔄 VERVAEKEAN PERFORMANCE COUPLING ANALYSIS
║  Session Exit Summary - Relevance Realization in Practice
║
╚═══════════════════════════════════════════════════════════════════════════════

📊 SESSION OVERVIEW
═══════════════════════════════════════════════════════════════════════════════
Total Operations: 15
Session Duration: 847.2s (14.1 min)
Blocking Operations: 0

🌟 COUPLING QUALITY
═══════════════════════════════════════════════════════════════════════════════
🌟 TRANSPARENT COUPLING
→ Tool vanished into practice - user-system moved as one (ready-to-hand)

🎯 SALIENCE LANDSCAPE (Where Attention Flowed)
═══════════════════════════════════════════════════════════════════════════════

Docker
  Time:  847.2s (98.5%)
  Operations: 2
  Average: 423.600s

GCP
  Time:    4.3s ( 0.5%)
  Operations: 3
  Average: 1.433s

⏱️  TOP 10 OPERATIONS (Slowest to Fastest)
═══════════════════════════════════════════════════════════════════════════════
 1.    build_training_image_cloudbuild            847.20s [Docker]
 2.    check_infrastructure                         4.30s [GCP]
 3.    check_wandb_prerequisites                    2.10s [API]

🧵 COUPLING CHANNELS (Thread Distribution)
═══════════════════════════════════════════════════════════════════════════════

Thread-3 (run_in_thread)
  Time:  847.2s (98.5%)
  Operations: 2

MainThread
  Time:   10.2s ( 1.2%)
  Operations: 13

⏰ OPERATION TIMELINE (Session Flow)
═══════════════════════════════════════════════════════════════════════════════
                                                               ████████████████████   build_training_image_cloudbuild   847.20s

💡 OPTIMIZATION PATHWAYS
═══════════════════════════════════════════════════════════════════════════════
⚠️  Optimize Docker Operations
   → Docker took 847.2s
   → Cache base images
   → Parallelize builds
   → Pre-build in CI/CD

✨ Excellent Flow Quality!
   → Consider micro-optimizations:
   → Cache expensive computations
   → Pre-load frequently accessed data
   → Monitor for regressions

∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿
🌟 TRANSPARENT COUPLING ACHIEVED
   Tool remained ready-to-hand throughout session
   User-system moved as unified organism
   Relevance realization flowed without friction
∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿
```

### How to Use

1. **Run TUI normally** - Monitoring happens automatically
2. **Use the screens** - Navigate to Setup, Launch, Monitor, Teardown, Infrastructure
3. **Exit TUI** - Comprehensive summary prints to console
4. **Read analysis** - Scroll up in terminal to see full exit summary
5. **Apply fixes** - Address blocking operations and optimization opportunities

## Troubleshooting

### No exit summary printed
- Make sure you exit TUI cleanly (press 'q' to quit)
- If you Ctrl+C, the signal handler still prints summary
- Summary only prints if you used monitored screens

### Empty or minimal summary
- You haven't navigated to monitored screens yet!
- Go to Setup (3), Launch (6), Monitor (4), Infrastructure (5), or Teardown (8)
- Do something that triggers operations (e.g., run setup, build image)

### Can't see full summary (scrolled off screen)
- Use terminal scroll back (Cmd+K on Mac, Page Up on Linux)
- Pipe TUI output to file: `python training/tui.py > tui_session.log 2>&1`
- Or increase terminal scrollback buffer

## Adding More Monitoring

To monitor a new operation:

```python
from cli.shared.performance_monitor import get_monitor

monitor = get_monitor()

# Wrap slow operation
op_id = monitor.start_operation("my_operation_name", category="GCP")
# ... your slow code here ...
monitor.end_operation(op_id)
```

**Categories**: `Docker`, `GCP`, `API`, `UI`, `Disk`, `Compute`

## What's NOT Monitored (Yet)

- Pricing screen - Cache loading
- GPU screen - Quota checks  
- Truffles screen - Data loading
- Reduce screen - Optimization calculations
- Home screen - No slow operations

These could be added if needed, but they're all <1 second operations.

---

**Last Updated**: 2025-11-09
**Monitoring Coverage**: 5/9 screens (Setup, Monitor, Launch, Teardown, Infrastructure)
**Output Format**: Console-only Vervaekean coupling analysis (NO files created!)
