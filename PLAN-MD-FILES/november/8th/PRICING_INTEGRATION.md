# GPU Pricing Integration - CLI Page 6 / TUI Page 6

**Comprehensive GPU pricing analysis integrated into ARR-COC training CLI/TUI**

## ✅ What We Built

### CLI Command: `python training/cli.py pricing`

Verbose output with comprehensive metrics for decision-making.

### Output Sections (8 Total):

1. **Top 10 by Total Cost** - Cheapest training cost (with 20% preemption buffer)
2. **Top 10 by Speed** - Fastest wall-clock training time
3. **Top 10 by TFLOPS/$** - Best compute value (FP16 performance per dollar)
4. **Top 10 by Samples/$** - Best training throughput per dollar
5. **Top 10 by VRAM/$** - Best memory value per dollar
6. **Spot Savings Analysis** - How much you save vs on-demand
7. **Final Recommendations** - Best options for different use cases
8. **Key Insights** - Decision-making guidance

---

## 📊 Example Output

```
🏆 TOP 10 CHEAPEST GPU OPTIONS (by Total Cost)

#1. H100 80GB
  💰 PRICING:
     Spot (preemptible):  $2.25/hour
     On-Demand (regular): $6.18/hour
     Spot Discount:       63.6% cheaper! 🔥

  📊 TRAINING ESTIMATES:
     Throughput:          30,000 samples/hour
     Training Time:       44.3 hours (~1.8 days)
     Total Cost (spot):   $119.61 ⭐
     Per Epoch:           $39.87

  🔧 HARDWARE SPECS:
     FP16 Performance:    1,979 TFLOPS
     VRAM:                80 GB

  💎 VALUE METRICS:
     TFLOPS per $/hr:     879.6
     VRAM GB per $/hr:    35.6
     Samples per $/hr:    13,333
```

---

## 🎯 Key Findings

### H100 80GB is the Clear Winner

**Cheapest total cost:** $119.61 (despite $2.25/hr vs T4's $0.22/hr)
**Why?** 3× faster training (44 hours vs 665 hours)

### Top 5 Recommendations:

1. **Production Training:** H100 spot - $119.61 total, 44.3 hours
2. **Maximum Speed:** H200 spot - $169.50 total, 38.0 hours (fastest)
3. **Budget Option:** L4 spot - $159.48 total, 221.5 hours
4. **Development/Testing:** T4 spot - $175.43 total, 665 hours
5. **Large Models (>40GB):** A100 80GB spot - $250.38 total

---

## 📁 File Structure

```
training/
├── cli.py                          # Added: pricing command
├── cli/
│   └── pricing/
│       ├── __init__.py
│       └── core.py                 # Core pricing logic (CLI/TUI shared)
├── scrape_gpu_pricing_mcp.py      # Production scraper (MCP + fallback)
├── scrape_gpu_pricing.py          # Simple scraper
├── analyze_gpu_pricing2.py        # Multi-source validator
└── GPU_PRICING_TOOLS.md           # Documentation
```

---

## 🚀 Usage

### CLI (Terminal)
```bash
# Full pricing analysis
python training/cli.py pricing

# Output is verbose - pipe to less for paging
python training/cli.py pricing | less
```

### TUI (Future - Page 6)
```bash
# Navigate to pricing screen
python training/tui.py
# Press '6' for GPU Pricing
```

---

## 🔧 Technical Details

### Architecture: TUI/CLI Shared Core Pattern

```
User Command
    ↓
┌────────────────────────────────────┐
│ CLI: python training/cli.py pricing│
│ TUI: Page 6 (PricingScreen)       │
└────────────────────────────────────┘
    ↓
┌────────────────────────────────────┐
│ cli/pricing/core.py                │
│ run_pricing_core(config, callback) │
│                                    │
│ • UI-agnostic business logic      │
│ • Calls scrape_gpu_pricing_mcp.py │
│ • Formats comprehensive output    │
└────────────────────────────────────┘
    ↓
┌────────────────────────────────────┐
│ scrape_gpu_pricing_mcp.py          │
│                                    │
│ • Scrapes GCP pricing pages       │
│ • Calculates costs & metrics      │
│ • Returns structured data         │
└────────────────────────────────────┘
    ↓
┌────────────────────────────────────┐
│ Callback (PrintCallback or TUI)   │
│                                    │
│ • CLI: Strips Rich → terminal     │
│ • TUI: Rich markup → widgets      │
└────────────────────────────────────┘
```

### Data Flow

1. **User runs:** `python training/cli.py pricing`
2. **CLI loads:** Training config from `.training`
3. **Core scrapes:** Live pricing from GCP (or static fallback)
4. **Core calculates:**
   - Training costs for 443K samples × 3 epochs
   - TFLOPS/$ metrics
   - VRAM/$ metrics
   - Samples/$ metrics
   - Spot savings
5. **Core ranks:** GPUs by different factors (cost, speed, value)
6. **Core outputs:** 8 comprehensive sections via callback
7. **User sees:** Verbose analysis for decision-making

---

## 📈 Metrics Tracked

### Pricing Metrics
- **Spot Price** - Preemptible hourly rate
- **On-Demand Price** - Regular hourly rate
- **Spot Discount %** - Savings vs on-demand (60-91%)

### Training Metrics
- **Throughput** - Estimated samples/hour
- **Training Time** - Total hours for full training
- **Total Cost** - Spot price × hours + 20% preemption buffer
- **Cost per Epoch** - Total ÷ 3

### Hardware Specs
- **FP16 TFLOPS** - GPU compute performance (mixed precision)
- **VRAM** - GPU memory capacity (GB)

### Value Metrics
- **TFLOPS per $** - Compute performance per dollar/hour
- **VRAM per $** - Memory capacity per dollar/hour
- **Samples per $** - Training throughput per dollar/hour

---

## 🎯 Design Principles

### Verbose Output for Decision-Making

Unlike typical pricing tools that show minimal data, this gives you:
- ✅ **Raw data** - All specs, all metrics
- ✅ **Multiple rankings** - See best option by different factors
- ✅ **Comprehensive comparison** - Spot vs on-demand for all GPUs
- ✅ **Context** - Why faster GPUs are cheaper total cost
- ✅ **Guidance** - Use case recommendations

### Why Verbose?

GPU selection is a **critical decision** that affects:
- Training cost ($120-$300)
- Training time (38-665 hours)
- Success/failure of training run

Better to show too much data than too little!

---

## 🔄 TUI Integration (Next Step)

To add Page 6 to TUI, create `cli/pricing/screen.py`:

```python
from textual.screen import Screen
from textual.widgets import Static
from .core import run_pricing_core

class PricingScreen(Screen):
    """GPU Pricing Analysis (Page 6)"""

    def compose(self):
        yield Static("Loading pricing data...", id="pricing-output")

    async def on_mount(self):
        # Run pricing core with TUICallback
        from cli.shared.callbacks import TUICallback

        output_widget = self.query_one("#pricing-output")
        callback = TUICallback(output_widget)

        config = load_training_config()
        pricing_data = run_pricing_core(config, callback)
```

Then add to `tui.py`:
```python
from cli.pricing.screen import PricingScreen

# In TuiApp:
def action_show_pricing(self):
    """Show GPU pricing (Page 6)"""
    self.push_screen(PricingScreen())

# Add keybinding: BINDINGS = [("6", "show_pricing", "Pricing")]
```

---

## ✅ Testing

```bash
# Test CLI (works now)
python training/cli.py pricing

# Test TUI (after screen.py created)
python training/tui.py
# Press '6' for pricing
```

---

## 📝 Summary

**Status:** CLI integration ✅ COMPLETE

**Files Created/Modified:**
- ✅ `cli.py` - Added `pricing` command
- ✅ `cli/pricing/core.py` - Comprehensive analysis logic
- ✅ `cli/pricing/__init__.py` - Module init
- ✅ Integration with `scrape_gpu_pricing_mcp.py`

**Output:** 8 sections, top 10 lists, verbose metrics for decision-making

**Next:** Add `cli/pricing/screen.py` for TUI Page 6 integration

**Result:** Users can now run `python training/cli.py pricing` for comprehensive GPU analysis!

---

**Built for ARR-COC-VIS Training Infrastructure 🤖**
*Karpathy Deep Oracle + Bright Data MCP Integration*
