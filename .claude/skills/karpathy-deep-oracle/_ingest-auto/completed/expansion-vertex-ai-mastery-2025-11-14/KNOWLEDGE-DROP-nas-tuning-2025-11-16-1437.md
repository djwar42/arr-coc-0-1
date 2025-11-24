# KNOWLEDGE DROP: Vertex AI Vizier & Hyperparameter Tuning

**Date**: 2025-11-16 14:37
**Runner**: PART 20
**File Created**: `gcp-vertex/19-nas-hyperparameter-tuning.md` (~730 lines)

## What Was Created

Comprehensive guide to Vertex AI Vizier covering:

### 1. Search Algorithms (4 major approaches)
- **Grid Search**: Exhaustive search for small discrete spaces
- **Random Search**: Surprisingly effective baseline (matches Bayesian on many tasks)
- **Bayesian Optimization**: Gaussian Process-based intelligent search (default)
- **Hyperband**: Revolutionary adaptive early stopping (parameter-free, adapts to unknown convergence)

### 2. Trial Configuration
- Parameter specifications (continuous, integer, categorical, discrete)
- Conditional parameters (momentum only for SGD optimizer)
- Metric specification (single and multi-objective)
- Measurement reporting from training scripts

### 3. Early Stopping Strategies
- **Median Stopping Rule**: Stop if worse than median performance at step t
- **Performance Curve Prediction**: Fit learning curves to predict final performance
- **Hyperband Successive Halving**: Geometric resource allocation (81 configs × 1 epoch → 1 config × 81 epochs)

### 4. Multi-Objective Optimization
- Pareto frontier discovery (accuracy vs latency vs model size)
- Scalarization approaches
- Deployment configuration selection from Pareto set

### 5. Cost Optimization
- Preemptible trials (80% cost reduction with checkpointing)
- Parallel execution limits (balance wall-clock time vs hourly cost)
- Warm starting from previous tuning jobs
- Subset data for early exploration trials

### 6. ARR-COC-0-1 Hyperparameter Search
Complete practical example with:
- 11-dimensional search space (token budget, LOD ranges, relevance weights, opponent balances, learning rates)
- Multi-objective: VQA accuracy + avg tokens per image + inference latency
- Training script integration with Vizier metric reporting
- Pareto frontier visualization and deployment config selection

## Key Insights

### Hyperband's Revolutionary Approach

From [Hyperband research](https://homes.cs.washington.edu/~jamieson/hyperband.html):
> "In contrast to treating the problem as a configuration selection problem, we pose the problem as a configuration evaluation problem and select configurations randomly. By computing more efficiently, we look at more hyperparameter configurations."

**Why it works:**
- Adapts to unknown convergence rates (no need to model learning curves)
- Parameter-free (only requires max_iter)
- Provably near-optimal resource allocation
- Considers 256 configs while Bayesian methods consider 5 in same time

### Random Search Effectiveness

Surprising finding from benchmarks:
- On 117 hyperparameter datasets, random search × 2 speed beats state-of-art Bayesian methods
- Bayesian optimization advantage is often negligible
- High-dimensional spaces favor random sampling

### Multi-Objective Trade-offs

Real-world deployment requires balancing:
- Production: 92% accuracy, 120ms latency, 50M params (optimize for accuracy)
- Balanced API: 90% accuracy, 80ms latency, 30M params (production deployment)
- Mobile/Edge: 88% accuracy, 50ms latency, 15M params (optimize for speed)

## Technical Highlights

### Hyperband Bracket Strategy

Example with max_iter=81, eta=3:

| Bracket | Initial Configs | Initial Epochs | Final | Total Budget |
|---------|----------------|----------------|-------|--------------|
| s=4 (aggressive) | 81 configs | 1 epoch | 1 × 81 | 405 epochs |
| s=3 (balanced) | 27 configs | 3 epochs | 1 × 81 | 405 epochs |
| s=2 (moderate) | 9 configs | 9 epochs | 1 × 81 | 405 epochs |
| s=1 (conservative) | 6 configs | 27 epochs | 2 × 81 | 405 epochs |
| s=0 (baseline) | 5 configs | 81 epochs | 5 × 81 | 405 epochs |

All brackets use same total budget but explore different breadth/depth trade-offs.

### Cost Optimization Math

**Preemptible savings:**
- Regular VM: $1.00/hour × 100 hours = $100
- Preemptible: $0.20/hour × 110 hours (10% restart overhead) = $22
- **Savings: 78%**

**Parallel execution trade-off:**
- 1 trial: $100 total, 50 hours wall-clock, $2/hour
- 10 trials: $100 total, 5 hours wall-clock, $20/hour
- **Same total cost, 10x faster results, 10x higher hourly spend**

## Code Quality

**Complete runnable examples:**
- Full Vertex AI HyperparameterTuningJob configuration
- Training script with Vizier metric reporting integration
- Pareto frontier visualization and analysis
- ARR-COC-0-1 specific hyperparameter search

**Real-world patterns:**
- Checkpoint resuming for preemptible instances
- Conditional parameters (optimizer-specific hyperparameters)
- Warm starting from coarse to fine search
- Multi-objective deployment config selection

## Sources Cited

**Google Cloud Documentation:**
- Vertex AI Vizier overview (accessed 2025-11-16)
- Hyperparameter tuning overview (accessed 2025-11-16)

**Research Papers:**
- Google Vizier: A Service for Black-Box Optimization (Golovin et al., 2017)
- Hyperband: A Novel Bandit-Based Approach (Li, Jamieson et al., 2016)
- Neural Architecture Search: Insights from 1000 Papers (White et al., 2023)

**Web Resources:**
- Hyperband demo by kgjamieson (accessed 2025-11-16)
- Medium tutorial on Vertex AI Vizier (Nicholas Ord, 2024)

## Stats

- **File**: gcp-vertex/19-nas-hyperparameter-tuning.md
- **Lines**: ~730 lines
- **Sections**: 8 major sections
- **Code Examples**: 15+ complete examples
- **Tables**: 4 comparison tables
- **External Sources**: 8 cited sources with access dates

## Next Steps

PART 20 complete. Ready for:
- Batch 5 continuation (PARTs 17-20)
- Or Batch 6 execution (PARTs 21-24)
- Oracle consolidation after all batches complete
