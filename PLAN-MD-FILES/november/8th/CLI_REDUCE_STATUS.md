# GPU Cost Reduction Feature - Status Update

## ✅ What's Complete

### CLI Command
```bash
python training/cli.py reduce 'H100 80GB'
python training/cli.py reduce 'A100 40GB'
python training/cli.py reduce 'L4'
```

**Features:**
- Searches all GCP regions for a specific GPU type
- Finds cheapest region with that GPU
- Shows spot pricing and discount percentage
- Calculates training cost estimates
- Compares pricing across regions
- Displays savings vs other regions

### Core Module
- `training/cli/reduce/core.py` - UI-agnostic reduce logic
- `run_reduce_core()` - Main function called by CLI
- `find_cheapest_region_for_gpu()` - Region search logic
- `list_available_gpus()` - List all available GPU types

### Pricing Page Dropdown Fix
**Issue:** Dropdown wasn't updating content when region changed  
**Fix:** 
- Clear all child widgets before reloading
- Run worker with new region
- Properly handle region change events

**Now Works:**
- ✅ Dropdown shows all valid GCP regions
- ✅ Changing region clears old content
- ✅ Shows loading message during fetch
- ✅ Loads new pricing data for selected region
- ✅ Updates status label with GPU count

## 🚧 What's Next

### TUI Page 7 (Reduce)
- Create `training/cli/reduce/screen.py` - TUI screen for reduce
- Add to main TUI app with keyboard shortcut `7`
- Add "Reduce (7)" button to home screen
- GPU type dropdown/input field
- "Find Cheapest Region" button
- Display results in scrollable panel

### Structure (same as other pages)
```python
class ReduceScreen(BaseScreen):
    - compose() - GPU input + results panel + buttons
    - initialize_content() - Run reduce analysis
    - finish_loading() - Update results
    - action_back() - Return to home
```

### Expected Usage
1. User launches TUI
2. Presses `7` or clicks "Reduce (7)"
3. Enters/selects GPU type (e.g., "H100 80GB")
4. Clicks "Find Cheapest Region"
5. Sees results: cheapest region, price, comparison table
6. Can update GPU type and search again

## 🔧 Files Modified
- `training/cli.py` - Added reduce command
- `training/cli/reduce/core.py` - Core reduce logic (CREATED)
- `training/cli/reduce/__init__.py` - Module init (CREATED)
- `training/cli/pricing/screen.py` - Fixed dropdown handler

## 📝 Example Output
```
====================================================================================================
ARR-COC GPU Cost Reduction - Finding Cheapest Region for: H100 80GB
====================================================================================================

🔬 Finding cheapest region for: H100 80GB
====================================================================================================
Searching 17 regions...
✅ Found in us-central1: $2.25/hr (64% off)
✅ Found in europe-west4: $2.38/hr (62% off)
...

====================================================================================================
✅ CHEAPEST REGION FOUND!
====================================================================================================

🎯 GPU Type: H100 80GB
🌍 Cheapest Region: us-central1
💰 Spot Price: $2.25/hour (64% off)
🚀 On-Demand Price: $6.30/hour
💾 VRAM: 80 GB
⚡ TFLOPS (FP16): 989,000

📊 TRAINING ESTIMATES:
  Training Time: 5.2 hours (~0.2 days)
  Total Cost (with 20% buffer): $13.98
  Throughput: ~256,000 samples/hour

====================================================================================================
📋 COMPARISON WITH OTHER REGIONS:
====================================================================================================
Region                    Spot $/hr       Discount %      Savings        
----------------------------------------------------------------------------------------------------
us-central1               $2.25           64%             $0.00
europe-west4              $2.38           62%             $0.13
...

✅ Reduce analysis complete!
💡 Update GCP_REGION='us-central1' in your .training config to use the cheapest region.
```
