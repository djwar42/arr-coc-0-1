# KNOWLEDGE DROP: GCP Preemptible & Spot GPU Strategies

**Created**: 2025-11-16 16:25
**PART**: 7 of 24 (Batch 2: Multi-GPU & Distributed)
**File**: gcp-gpu/06-preemptible-spot-gpu-strategies.md
**Size**: ~720 lines

---

## What Was Created

Comprehensive guide to cost-optimized GPU training using preemptible and Spot VMs on GCP, achieving 60-91% cost savings through fault-tolerant architectures.

**7 Major Sections:**

1. **Spot GPU Pricing Analysis** (~120 lines)
   - Current 2025 pricing: H100 ($2.25/hr Spot vs $7.50 on-demand)
   - A100 80GB pricing: $1.57/hr Spot (57% savings)
   - Regional pricing variations (us-central1 cheapest)
   - Multi-GPU cost comparison (8×A100: $1,382 Spot vs $2,936 on-demand)

2. **Preemption Patterns and Availability** (~100 lines)
   - 30-second termination notice via ACPI G2 + metadata API
   - MTBI (Mean Time Between Interruptions): 4-8 hours for A100
   - Availability patterns: weekday mornings = highest preemption
   - Cross-region strategies for better availability

3. **Checkpoint Strategies** (~150 lines)
   - Three-tier storage: Local SSD (10min) → PD (30min) → GCS (2hr)
   - Optimal checkpoint interval formula
   - Emergency checkpoint on preemption signal
   - Distributed checkpoint for multi-GPU

4. **Automatic Restart and Recovery** (~120 lines)
   - Managed Instance Groups with health checks
   - Cloud Functions for automatic restart
   - PyTorch training resume logic
   - Uptime tracking and goodput monitoring

5. **Hybrid On-Demand + Spot Architectures** (~150 lines)
   - Master-worker hybrid: 1 on-demand + 7 Spot = 50% savings + 95% reliability
   - Elastic training with dynamic workers (torchrun)
   - Cost analysis: Hybrid vs All-Spot vs All-On-Demand
   - Rolling restart patterns

6. **Production Best Practices** (~100 lines)
   - Multi-day training checkpoint cadence
   - Weights & Biases integration for Spot monitoring
   - Quota management (Spot quotas separate from on-demand)
   - arr-coc-0-1 specific configuration

7. **Cost-Performance Tradeoff** (~60 lines)
   - Goodput optimization (85-90% for optimal Spot, 92-96% for hybrid)
   - Cost per training step metrics
   - Decision matrix: when to use each configuration

---

## Key Insights

**Spot vs Preemptible VMs:**
- Spot VMs have **no 24-hour limit** (vs Preemptible's hard limit)
- Variable 60-91% discounts vs fixed ~79%
- Spot is the current recommended approach

**Cost Savings:**
- A100 80GB Spot: $1.57/hr vs $3.67 on-demand (57% savings)
- H100 Spot: $2.25/hr vs $7.50 on-demand (70% savings)
- 8×A100 training: $1,382 Spot vs $2,936 on-demand over 100 hours

**Hybrid Architecture Benefits:**
- 1 on-demand master + 7 Spot workers = $14.66/hr (50% savings)
- 95% goodput vs 85% for all-Spot
- Master stores authoritative checkpoints (never preempted)
- Best balance of cost and reliability for production

**Checkpoint Strategy:**
- Three-tier approach minimizes overhead while ensuring safety
- Local SSD every 10min (fast), PD every 30min (restart-safe), GCS every 2hr (permanent)
- Emergency checkpoint on 30-second preemption signal
- Optimal interval = sqrt(2 × MTBI × save_overhead / load_time)

**Preemption Patterns:**
- MTBI: 4-8 hours for A100, 2-6 hours for H100
- Off-peak hours (nights/weekends) have lower preemption rates
- us-central1 cheapest but higher preemption vs other regions

---

## Integration Points

**Builds on existing knowledge:**
- [38-gcp-spot-fundamentals.md](../../karpathy/practical-implementation/38-gcp-spot-fundamentals.md) - Spot architecture
- [43-gcp-spot-checkpoint-strategies.md](../../karpathy/practical-implementation/43-gcp-spot-checkpoint-strategies.md) - Checkpoint patterns
- [45-gcp-spot-production-patterns.md](../../karpathy/practical-implementation/45-gcp-spot-production-patterns.md) - Production strategies

**Connects to:**
- Multi-GPU training (PART 5): Distributed checkpointing for multi-GPU Spot
- Multi-node training (PART 6): Hybrid architectures across nodes
- Cost optimization (PART 17): Spot savings as primary cost reduction strategy

**arr-coc-0-1 application:**
- Recommended: 1 on-demand A100 master + 7 Spot A100 workers
- $14.66/hr vs $29.36 on-demand (50% savings)
- 95% goodput with robust checkpoint strategy
- Deployment script provided in production section

---

## Research Quality

**Sources:**
- ✅ Google Cloud official documentation (Spot pricing, preemptible VMs)
- ✅ Academic research (ETH Zurich distributed ML, arXiv RLBoost)
- ✅ Existing oracle knowledge (3 Spot files from practical-implementation)
- ✅ Industry analysis (Cast AI, Economize Cloud pricing reports)

**Citations:**
- All pricing data from official GCP pricing pages (accessed 2025-11-16)
- Preemption mechanics from Google Cloud docs
- Hybrid architecture insights from arXiv research paper
- Checkpoint strategies from existing Karpathy oracle knowledge

**Completeness:**
- Covers pricing, availability, checkpointing, automation, hybrid patterns
- Production-ready code examples (PyTorch, bash scripts, Cloud Functions)
- arr-coc-0-1 specific configuration and deployment
- Decision framework for choosing configuration

---

## Statistics

- **Lines**: ~720
- **Sections**: 7 major sections
- **Code examples**: 15+ (Python, bash, YAML)
- **Tables**: 8 comparison tables
- **Sources**: 13 cited (3 internal + 10 web)
- **arr-coc-0-1 content**: Production configuration + deployment script
