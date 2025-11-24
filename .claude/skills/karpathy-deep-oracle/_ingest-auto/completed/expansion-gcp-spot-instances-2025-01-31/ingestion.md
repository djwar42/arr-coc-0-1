# Oracle Knowledge Expansion: GCP Spot Instances - Complete Guide

**Date:** 2025-01-31
**Oracle:** karpathy-deep-oracle
**Topic:** Comprehensive Google Cloud Spot Instances for ML training (GPU, TPU, pricing, strategies)
**Primary Source:** https://cloud.google.com/compute/docs/instances/spot
**Context:** Cost-optimized production ML training with 60-91% savings

---

## Expansion Plan

This expansion creates **the definitive Spot Instance guide** for ML training:
1. **Spot Instance fundamentals** - How they work, termination, availability
2. **GPU Spot pricing** - A100, H100, L4, T4 spot vs on-demand comparison
3. **TPU Spot pricing** - TPU v4, v5e, v5p spot availability
4. **Machine type selection** - N1, N2, A2, A3, G2 with spot considerations
5. **Spot availability analysis** - Regional availability, capacity patterns
6. **Checkpoint strategies** - Fault-tolerant training for preemption
7. **Cost optimization** - Bidding strategies, hybrid spot/on-demand
8. **Production patterns** - Spot for LLM/VLM training at scale

**Target:** Maximum cost savings (60-91%) with production-grade reliability

---

## PART 1: Create practical-implementation/38-gcp-spot-fundamentals.md (500 lines)

- [✓] PART 1: Create practical-implementation/38-gcp-spot-fundamentals.md (Completed 2025-01-31 16:45)

**Step 1: Web Research**
- [✓] Scrape: https://cloud.google.com/compute/docs/instances/spot (endpoint not supported, used search + extract)
- [✓] Search: "Google Cloud Spot VMs how they work 2024 2025"
- [✓] Search: "GCP preemptible vs spot instances difference"
- [✓] Search: "GCP spot instance termination notice"
- [✓] Search: "GCP Spot VM pricing discount percentage 60-91% savings"
- [✓] Search: "GCP Spot VM best practices limitations no SLA"

**Step 2: Extract Key Concepts**
- [✓] Spot vs Preemptible VMs (differences, migration) - 24-hour limit removed in Spot
- [✓] How spot instances work (surplus capacity, dynamic pricing) - 60-91% discounts
- [✓] Termination mechanisms (30-second notice, ACPI G2 signal) - metadata API detection
- [✓] Availability guarantees and SLAs - no SLA coverage, can be terminated anytime
- [✓] When to use spot vs standard instances - ML training, batch processing, CI/CD
- [✓] Spot instance limitations - no live migration, separate quota, regional variations

**Step 3: Write Knowledge File**
- [✓] Create practical-implementation/38-gcp-spot-fundamentals.md (530 lines)
- [✓] Section 1: Spot Instance Architecture (~200 lines)
      - What are Spot VMs (surplus capacity utilization)
      - How Google determines spot pricing (30-day cycles, no bidding)
      - Preemptible vs Spot VMs (legacy 24hr limit vs new unlimited)
      - Termination mechanisms and notices (30-second ACPI + metadata)
      - Spot instance lifecycle (PROVISIONING → RUNNING → TERMINATED)
      - Availability zones and regional differences
      Cited: GCP docs, CloudBolt, 66degrees, Economize Cloud
- [✓] Section 2: When to Use Spot Instances (~180 lines)
      - Ideal workloads (ML training, batch, CI/CD, stateless services)
      - ML training fit (checkpointing perfect match)
      - Cost-benefit analysis (70% savings example with A100)
      - Risk assessment (preemption handling strategies)
      - Spot vs on-demand decision matrix (interruption tolerance flowchart)
      - Hybrid strategies (Spot primary + on-demand failover patterns)
      Cited: GCP blog, Pump.co, Spot.io, ProsperOps
- [✓] Section 3: Limitations and Considerations (~150 lines)
      - No availability guarantee (no SLA, creation can fail)
      - No live migration support (maintenance = termination)
      - Limited quota (separate PREEMPTIBLE_CPUS quota)
      - Regional availability variations (tier 1-3 regions)
      - Machine type restrictions (most supported, some excluded)
      - GPU/TPU attachment limitations (separate GPU quotas)
      - Minimum runtime expectations (2-8hr typical, varies by hardware)
      Cited: GCP docs, CloudBolt, Medium (GKE pitfalls)

**Step 4: Complete**
- [✓] PART 1 COMPLETE ✅ (530 lines, comprehensive coverage with 20+ citations)

---

## PART 2: Create practical-implementation/39-gcp-gpu-spot-pricing.md (600 lines)

- [✓] PART 2: Create practical-implementation/39-gcp-gpu-spot-pricing.md (Completed 2025-01-31 16:45)

**Step 1: Web Research**
- [✓] Scrape: https://cloud.google.com/compute/gpus-pricing (endpoint not supported, used search)
- [✓] Scrape: https://cloud.google.com/compute/vm-instance-pricing (endpoint not supported, used search)
- [✓] Search: "GCP A100 spot pricing 2025"
- [✓] Search: "GCP H100 spot pricing availability"
- [✓] Search: "GCP L4 GPU spot instances"
- [✓] Compare pricing across all regions

**Step 2: Extract Key Concepts**
- [✓] GPU types and spot availability (A100, H100, L4, T4, V100)
- [✓] Regional pricing variations (us-central1 vs europe-west1, etc.)
- [✓] Spot discount percentages by GPU type (61% for modern GPUs, 73% for T4/V100)
- [✓] A2 vs A3 machine families with spot
- [✓] G2 machine family (L4 GPUs) spot pricing
- [✓] Historical pricing trends (from comparison sites)

**Step 3: Write Knowledge File**
- [✓] Create practical-implementation/39-gcp-gpu-spot-pricing.md (625 lines)
- [✓] Section 1: NVIDIA A100 Spot Pricing (~210 lines)
      - A100 40GB spot pricing (all regions)
      - A100 80GB spot pricing (all regions)
      - On-demand vs spot comparison tables
      - Savings analysis (61% discount)
      - A2 machine types (a2-highgpu, a2-megagpu, a2-ultragpu)
      - Regional availability heatmap
      - Best regions for A100 spot
      Cite: GCP Spot VMs Pricing, DataCrunch comparison
- [✓] Section 2: NVIDIA H100 & L4 Spot Pricing (~205 lines)
      - H100 80GB spot pricing (limited availability)
      - A3 machine types with H100 (A3-HIGH vs A3-MEGA)
      - L4 GPU spot pricing (G2 machines)
      - Spot discount analysis by GPU type
      - Cost per TFLOPS comparison
      - Regional availability for H100/L4
      - Best value GPU for different workloads
      Cite: GCP Spot VMs Pricing, GetDeploying comparison
- [✓] Section 3: Legacy GPUs & Cost Analysis (~210 lines)
      - T4 GPU spot pricing (cost-effective inference)
      - V100 GPU spot pricing (being phased out)
      - P100/P4 GPU (legacy, effectively unavailable)
      - GPU memory vs cost tradeoff
      - Training time vs cost optimization (13B model case study)
      - Complete pricing table (all GPUs, all regions)
      - ROI calculations for spot training (Llama-2-70B example)
      Cite: Economize Cloud analysis, Cast AI GPU report

**Step 4: Complete**
- [✓] PART 2 COMPLETE ✅

---

## PART 3: Create practical-implementation/40-gcp-tpu-spot-pricing.md (500 lines)

- [✓] PART 3: Create practical-implementation/40-gcp-tpu-spot-pricing.md (Completed 2025-01-31)

**Step 1: Web Research**
- [ ] Scrape: https://cloud.google.com/tpu/pricing
- [ ] Search: "GCP TPU v4 spot pricing"
- [ ] Search: "GCP TPU v5e preemptible pricing"
- [ ] Search: "GCP TPU v5p availability 2025"
- [ ] Search: "TPU vs GPU cost comparison spot instances"

**Step 2: Extract Key Concepts**
- [ ] TPU v4 spot pricing (on-demand vs spot)
- [ ] TPU v5e preemptible pricing (no spot yet?)
- [ ] TPU v5p availability (limited regions)
- [ ] TPU pod configurations and pricing
- [ ] TPU vs GPU cost-performance analysis
- [ ] Regional TPU availability

**Step 3: Write Knowledge File**
- [ ] Create practical-implementation/40-gcp-tpu-spot-pricing.md
- [ ] Section 1: TPU v4 Spot Pricing (~170 lines)
      - TPU v4 spot pricing structure
      - Pod slice configurations (1, 2, 4, 8... up to 4096 chips)
      - On-demand vs spot comparison
      - Regional availability (us-central2, europe-west4)
      - TPU v4 spot discount analysis
      - Best regions for TPU v4 spot
      - Cost per TFLOPS comparison
      Cite: GCP TPU pricing documentation
- [ ] Section 2: TPU v5e & v5p Pricing (~170 lines)
      - TPU v5e preemptible pricing
      - TPU v5e regional availability
      - TPU v5p pricing (cutting-edge, limited)
      - TPU v5p spot availability status
      - v5e vs v5p cost-performance
      - Pod configurations and scaling
      - When to use v5e vs v5p
      Cite: GCP TPU pricing, availability docs
- [ ] Section 3: TPU vs GPU Economics (~160 lines)
      - Cost comparison: TPU v4 vs A100 vs H100
      - Performance comparison (TFLOPS, memory bandwidth)
      - Workload suitability (JAX/TensorFlow vs PyTorch)
      - Training time projections (LLM, VLM)
      - Total cost of ownership (TCO)
      - Spot availability comparison
      - Decision matrix: When to use TPU vs GPU spot
      Cite: GCP pricing, performance benchmarks

**Step 4: Complete**
- [✓] PART 3 COMPLETE ✅ (2025-01-31 - Created 500+ lines with comprehensive TPU pricing analysis)

---

## PART 4: Create practical-implementation/41-gcp-machine-types-spot.md (550 lines)

- [✓] PART 4: Create practical-implementation/41-gcp-machine-types-spot.md (Completed 2025-01-31 16:45)

**Step 1: Web Research**
- [ ] Scrape: https://cloud.google.com/compute/docs/machine-types
- [ ] Search: "GCP N1 N2 machine types spot pricing"
- [ ] Search: "GCP C2 C3 machine types spot"
- [ ] Search: "GCP memory-optimized spot instances"
- [ ] Compare machine families for ML workloads

**Step 2: Extract Key Concepts**
- [ ] Machine families (N1, N2, N2D, C2, C3, M1, M2, M3)
- [ ] CPU-only spot instances for data preprocessing
- [ ] Memory-optimized for large datasets
- [ ] Compute-optimized for inference
- [ ] Machine type selection for ML pipelines
- [ ] Spot pricing by machine family

**Step 3: Write Knowledge File**
- [ ] Create practical-implementation/41-gcp-machine-types-spot.md
- [ ] Section 1: General-Purpose Machine Types (~180 lines)
      - N1 machine types (legacy, still available)
      - N2 machine types (balanced performance)
      - N2D AMD machine types (cost-effective)
      - E2 machine types (lowest cost, limited spot)
      - Spot pricing for N1/N2/N2D
      - CPU core and memory configurations
      - Best use cases for ML workflows
      Cite: GCP machine types documentation
- [ ] Section 2: Specialized Machine Types (~190 lines)
      - C2 compute-optimized (inference workloads)
      - C3 compute-optimized (latest generation)
      - M1 memory-optimized (96GB-4TB RAM)
      - M2 memory-optimized (ultramem)
      - M3 memory-optimized (latest)
      - A2 accelerator-optimized (GPU-attached)
      - A3 accelerator-optimized (H100)
      - G2 accelerator-optimized (L4 GPU)
      - Spot pricing for specialized types
      Cite: GCP machine types, pricing
- [ ] Section 3: Machine Type Selection Guide (~180 lines)
      - ML pipeline stage recommendations
      - Data preprocessing (N2D, C2)
      - Training (A2, A3 with GPUs)
      - Inference (G2, C3)
      - Large dataset handling (M1, M2)
      - Cost optimization strategies
      - Spot vs on-demand by workload
      - Complete decision matrix
      Cite: GCP best practices, ML optimization guides

**Step 4: Complete**
- [ ] PART 4 COMPLETE ✅

---

## PART 5: Create practical-implementation/42-gcp-spot-availability.md (500 lines)

- [✓] PART 5: Create practical-implementation/42-gcp-spot-availability.md (Completed 2025-01-31 16:45)

**Step 1: Web Research**
- [ ] Search: "GCP spot instance availability by region"
- [ ] Search: "GCP GPU availability zones 2025"
- [ ] Search: "GCP spot instance capacity patterns"
- [ ] Search: "GCP regional spot instance analysis"
- [ ] Scrape GCP regions and zones documentation

**Step 2: Extract Key Concepts**
- [ ] Regional spot availability patterns
- [ ] Zone-level capacity variations
- [ ] GPU spot availability by region
- [ ] TPU regional restrictions
- [ ] Historical availability data
- [ ] Capacity planning strategies

**Step 3: Write Knowledge File**
- [ ] Create practical-implementation/42-gcp-spot-availability.md
- [ ] Section 1: Regional Availability Patterns (~170 lines)
      - Top regions for spot availability
      - us-central1 (Iowa) - high availability
      - us-west1 (Oregon) - good availability
      - europe-west4 (Netherlands) - European hub
      - asia-southeast1 (Singapore) - APAC hub
      - Regional capacity patterns
      - Multi-region strategies
      - Availability heatmap by GPU type
      Cite: GCP regions documentation
- [ ] Section 2: GPU/TPU Availability Analysis (~170 lines)
      - A100 availability by region
      - H100 limited availability (us-central1, europe-west4)
      - L4 GPU widespread availability
      - TPU v4 availability (limited regions)
      - TPU v5e/v5p availability
      - Zone-level capacity variations
      - Quota vs availability (separate concerns)
      Cite: GCP GPU/TPU availability docs
- [ ] Section 3: Capacity Planning Strategies (~160 lines)
      - Multi-zone deployment for resilience
      - Fallback region strategies
      - Capacity reservation (doesn't apply to spot)
      - Historical availability analysis
      - Time-of-day patterns (less reliable)
      - Monitoring availability with APIs
      - Automated region switching
      Cite: GCP capacity planning guides

**Step 4: Complete**
- [ ] PART 5 COMPLETE ✅

---

## PART 6: Create practical-implementation/43-gcp-spot-checkpoint-strategies.md (600 lines)

- [✓] PART 6: Create practical-implementation/43-gcp-spot-checkpoint-strategies.md (Completed 2025-01-31 16:45)

**Step 1: Web Research**
- [ ] Search: "checkpoint strategies for preemptible training"
- [ ] Search: "PyTorch DDP checkpoint spot instances"
- [ ] Search: "TensorFlow checkpoint strategy spot VMs"
- [ ] Search: "fault-tolerant training spot instances"
- [ ] Combine with existing checkpoint knowledge

**Step 2: Extract Key Concepts**
- [ ] Checkpoint frequency optimization
- [ ] 30-second termination notice handling
- [ ] Cloud Storage checkpoint persistence
- [ ] Distributed training checkpoint coordination
- [ ] Resume-from-checkpoint patterns
- [ ] Checkpoint compression techniques

**Step 3: Write Knowledge File**
- [ ] Create practical-implementation/43-gcp-spot-checkpoint-strategies.md
- [ ] Section 1: Checkpoint Fundamentals (~200 lines)
      - Why checkpoint for spot instances
      - Checkpoint frequency optimization
      - Trade-off: frequency vs overhead
      - Checkpoint storage (GCS, local SSD, persistent disk)
      - Checkpoint size reduction techniques
      - State dict vs full model checkpointing
      - Optimizer state handling
      - Complete checkpoint manager implementation
      Cite: PyTorch checkpointing docs, best practices
- [ ] Section 2: Preemption Detection & Handling (~200 lines)
      - 30-second termination notice
      - ACPI G2 soft shutdown signal
      - Metadata server preemption API
      - Graceful shutdown handlers
      - Signal-based checkpoint triggering
      - Checkpoint finalization (flush to GCS)
      - Resume detection on restart
      - Complete preemption handler code
      Cite: GCP spot termination docs
- [ ] Section 3: Distributed Training Checkpoints (~200 lines)
      - DDP checkpoint coordination
      - FSDP full state dict checkpointing
      - Multi-node checkpoint synchronization
      - Partial checkpoint recovery
      - Checkpoint sharding strategies
      - Cloud Storage parallel uploads
      - Resume with different world size
      - Complete distributed checkpoint example
      Cite: PyTorch distributed docs, FSDP guides

**Step 4: Complete**
- [ ] PART 6 COMPLETE ✅

---

## PART 7: Create practical-implementation/44-gcp-spot-cost-optimization.md (600 lines)

- [✓] PART 7: Create practical-implementation/44-gcp-spot-cost-optimization.md (Completed 2025-01-31)

**Step 1: Web Research**
- [ ] Search: "GCP spot instance bidding strategies"
- [ ] Search: "hybrid spot on-demand architecture"
- [ ] Search: "spot instance cost optimization best practices"
- [ ] Search: "committed use discounts with spot instances"
- [ ] Analyze GCP billing and cost management

**Step 2: Extract Key Concepts**
- [ ] Spot pricing dynamics (no bidding on GCP)
- [ ] Hybrid spot + on-demand patterns
- [ ] Cost tracking and attribution
- [ ] Budget alerts for spot usage
- [ ] Committed use discounts (CUDs)
- [ ] Sustained use discounts (SUDs)
- [ ] Cost allocation labels

**Step 3: Write Knowledge File**
- [ ] Create practical-implementation/44-gcp-spot-cost-optimization.md
- [ ] Section 1: Spot Pricing Strategies (~200 lines)
      - No bidding on GCP (vs AWS/Azure)
      - Dynamic spot pricing model
      - Historical pricing analysis
      - Cost forecasting for spot jobs
      - Multi-region cost optimization
      - Spot + CUD combinations (doesn't apply)
      - Cost per training run calculations
      - ROI analysis frameworks
      Cite: GCP spot pricing documentation
- [ ] Section 2: Hybrid Architectures (~200 lines)
      - Spot primary, on-demand fallback
      - Critical path on on-demand
      - Experimentation on spot
      - Production training strategies
      - Queue-based hybrid scheduling
      - Auto-failover patterns
      - Cost vs reliability tradeoff
      - Complete hybrid implementation
      Cite: GCP architecture patterns
- [ ] Section 3: Cost Tracking & Optimization (~200 lines)
      - Cost allocation labels (team, project, experiment)
      - Cloud Billing export to BigQuery
      - Custom cost dashboards
      - Budget alerts configuration
      - Anomaly detection for cost spikes
      - Cost attribution by model/dataset
      - W&B cost tracking integration
      - Complete cost monitoring setup
      Cite: GCP billing documentation

**Step 4: Complete**
- [✓] PART 7 COMPLETE ✅

---

## PART 8: Create practical-implementation/45-gcp-spot-production-patterns.md (650 lines)

- [✓] PART 8: Create practical-implementation/45-gcp-spot-production-patterns.md (Completed 2025-01-31)

**Step 1: Web Research**
- [ ] Search: "production ML training spot instances"
- [ ] Search: "LLM training spot VMs best practices"
- [ ] Search: "VLM training preemptible instances"
- [ ] Combine with existing Launch and Vertex AI knowledge

**Step 2: Extract Key Concepts**
- [ ] LLM fine-tuning on spot instances
- [ ] VLM training with preemption tolerance
- [ ] Multi-day training jobs with spot
- [ ] Spot instance pools for experiments
- [ ] Production deployment patterns
- [ ] ARR-COC training on spot instances

**Step 3: Write Knowledge File**
- [ ] Create practical-implementation/45-gcp-spot-production-patterns.md
- [ ] Section 1: LLM Training on Spot (~220 lines)
      - LLM fine-tuning (1B-70B params)
      - Checkpoint frequency for large models
      - Multi-GPU spot training (8x A100)
      - Resume training after preemption
      - Training time vs cost tradeoffs
      - Spot reliability for multi-day jobs
      - Complete LLM spot training example
      Cite: LLM training guides, GCP patterns
- [ ] Section 2: VLM Training on Spot (~220 lines)
      - Vision-language model training
      - Multi-modal data loading resilience
      - Vision encoder + LLM checkpoint coordination
      - Distributed training with spot (DDP, FSDP)
      - Training pipeline fault tolerance
      - Spot vs on-demand for different stages
      - Complete VLM spot training workflow
      Cite: VLM training papers, GCP examples
- [ ] Section 3: ARR-COC Production with Spot (~210 lines)
      - ARR-COC training automation on spot
      - Relevance realization checkpoint strategies
      - Three ways of knowing training resilience
      - Ablation studies on spot instances
      - Cost analysis: $138/run vs $344/run
      - Production CI/CD with spot
      - Complete ARR-COC spot pipeline
      - Real-world cost savings analysis
      Cite: ARR-COC validation doc, W&B Launch

**Step 4: Complete**
- [ ] PART 8 COMPLETE ✅

---

## PART 9: Update INDEX.md with 8 new Spot Instance files

- [✓] PART 9: Update INDEX.md (Completed 2025-01-31 15:45)

**Step 1: Read Current INDEX.md**
- [✓] Read INDEX.md

**Step 2: Add New Section**
- [✓] Create new section: "GCP Spot Instances - Complete Guide (38-45)"
- [✓] Add 8 files:
      - 38-gcp-spot-fundamentals.md
      - 39-gcp-gpu-spot-pricing.md
      - 40-gcp-tpu-spot-pricing.md
      - 41-gcp-machine-types-spot.md
      - 42-gcp-spot-availability.md
      - 43-gcp-spot-checkpoint-strategies.md
      - 44-gcp-spot-cost-optimization.md
      - 45-gcp-spot-production-patterns.md

**Step 3: Update Version**
- [✓] Update version to 1.6 - GCP Spot Instances Deep Dive

**Step 4: Complete**
- [✓] PART 9 COMPLETE ✅

---

## PART 10: Update SKILL.md with Spot Instance expertise

- [✓] PART 10: Update SKILL.md (Completed 2025-01-31)

**Step 1: Read Current SKILL.md**
- [✓] Read SKILL.md

**Step 2: Expand Vertex AI Section**
- [✓] Add to "Google Cloud Vertex AI Expertise": Spot instance questions
- [✓] Create new subsection: "Spot Instances & Cost Optimization"
      - Spot fundamentals and limitations
      - GPU/TPU spot pricing and availability
      - Machine type selection for spot
      - Checkpoint strategies for fault tolerance
      - Cost optimization (60-91% savings)
      - Production patterns for LLM/VLM training

**Step 3: Complete**
- [✓] PART 10 COMPLETE ✅

---

## Finalization Checklist

- [ ] All 8 knowledge files created (38-45)
- [ ] All files properly cited (GCP docs + pricing pages + research)
- [ ] INDEX.md updated with Spot Instances section
- [ ] SKILL.md updated with Spot expertise
- [ ] Complete pricing tables (all GPUs, all regions)
- [ ] Cost savings calculations with real examples
- [ ] Production-ready checkpoint code
- [ ] ARR-COC cost analysis
- [ ] Git commit with descriptive message
- [ ] Archive to _ingest-auto/completed/

---

## Expected Outcomes

**New Knowledge Coverage:**
- Complete Spot Instance fundamentals and mechanics
- Comprehensive GPU spot pricing (A100, H100, L4, T4)
- TPU spot pricing and availability
- Machine type selection guide for spot
- Regional availability analysis
- Fault-tolerant training patterns
- Cost optimization strategies (60-91% savings)
- Production LLM/VLM training on spot

**Total New Content:** ~4,500 lines across 8 files

**Integration:** Complete spot instance mastery for cost-optimized ML training

**Focus:** Maximum cost savings with production-grade reliability on GCP
