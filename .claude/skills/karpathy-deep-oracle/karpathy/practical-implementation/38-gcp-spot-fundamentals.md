# GCP Spot Instance Fundamentals

Complete guide to Google Cloud Spot VMs: architecture, pricing model, termination mechanisms, and production use cases for cost-optimized ML training.

---

## Overview

Google Cloud Spot VMs are compute instances that utilize Google's surplus capacity at dramatically reduced prices (60-91% discounts compared to on-demand instances). Unlike standard VMs, Spot instances can be preempted (terminated) by Google Cloud at any time with only 30 seconds notice, making them ideal for fault-tolerant workloads like ML training, batch processing, and CI/CD pipelines.

**Key characteristics:**
- **60-91% cost savings** over standard on-demand pricing
- **No maximum runtime** (unlike legacy Preemptible VMs with 24-hour limit)
- **30-second termination notice** via ACPI G2 signal and metadata API
- **No SLA coverage** - excluded from Compute Engine Service Level Agreement
- **Dynamic pricing** that can change up to once every 30 days
- **Same performance** as standard VMs when running

From [Google Cloud VM Instance Pricing](https://cloud.google.com/compute/vm-instance-pricing) (accessed 2025-01-31):
> Spot prices are dynamic and can change up to once every 30 days, but provide discounts of 60-91% off of the corresponding on-demand price for most machine types and GPUs.

---

## Section 1: Spot Instance Architecture

### What Are Spot VMs?

Spot VMs are virtual machines that run on Google Cloud's excess compute capacity. When Google Cloud has unused resources in its data centers, it makes this capacity available at steep discounts through the Spot pricing model. This surplus capacity exists because cloud providers must maintain buffer capacity to handle demand spikes from standard on-demand customers.

From [66degrees - Spot VMs Overview](https://66degrees.com/spot-vms-overview-the-affordable-google-discount/) (accessed 2025-01-31):
> As highlighted in this blog, Spot VMs provide the same performance as standard VMs while leveraging unused Google Cloud capacity.

**How surplus capacity works:**

1. **Data center utilization**: Google maintains physical servers that aren't always fully utilized
2. **Dynamic allocation**: When utilization is low, capacity becomes available as Spot VMs
3. **Preemption priority**: When standard customers need capacity, Spot VMs are terminated to free resources
4. **Automatic pricing**: Supply and demand dynamics determine Spot pricing within the 60-91% discount range

**Spot vs Standard VMs:**

| Feature | Spot VMs | Standard VMs |
|---------|----------|--------------|
| Pricing | 60-91% discount | Full on-demand price |
| Availability guarantee | None (can be terminated anytime) | High availability with SLA |
| Maximum runtime | No limit | No limit |
| Termination notice | 30 seconds | None (standard lifecycle) |
| Live migration | Not supported | Supported for maintenance |
| Use case | Fault-tolerant workloads | Production services |
| SLA coverage | No SLA | 99.9-99.95% SLA |

### How Google Determines Spot Pricing

Unlike AWS Spot Instances which use bidding, Google Cloud Spot pricing is **fixed and transparent** within its discount range:

From [CloudBolt - Google Cloud Spot VMs](https://www.cloudbolt.io/gcp-cost-optimization/google-cloud-spot-vms/) (accessed 2025-01-31):
> Spot VM pricing is dynamic because it is based on supply and demand. GCP states that Spot prices always provide a 60-91% reduction compared to on-demand prices.

**Pricing model characteristics:**

1. **No bidding required**: Unlike AWS, you don't set a maximum price - you get the current Spot price
2. **30-day pricing cycle**: Prices can change once per 30-day period maximum
3. **Regional variation**: Different regions have different Spot prices based on local capacity
4. **Machine-type specific**: Each machine type (N1, N2, A2, etc.) has its own discount percentage
5. **Minimum 60% discount**: Google guarantees at least 60% savings, often reaching 91%

**Example pricing (illustrative):**
- N2-standard-4 on-demand: $0.195/hour
- N2-standard-4 Spot: $0.047/hour (76% discount)
- A100 40GB GPU on-demand: $2.933/hour
- A100 40GB GPU Spot: $0.88/hour (70% discount)

From [Economize Cloud - GCP Spot VM Introduction](https://www.economize.cloud/blog/introduction-gcp-spot-instances/) (accessed 2025-01-31):
> Preemptible VMs come at a 79% predictable discount compared to on-demand VMs. Spot VMs, on the other hand, come at a saving of variable 60%-91%.

### Preemptible vs Spot VMs: Legacy vs New

Google Cloud historically offered "Preemptible VMs" but replaced them with the superior "Spot VMs" model:

From [Google Cloud Preemptible VMs Documentation](https://docs.cloud.google.com/compute/docs/instances/preemptible) (accessed 2025-01-31):
> However, Spot VMs provide new features that preemptible VMs do not support. For example, preemptible VMs can only run for up to 24 hours at a time, but Spot VMs do not have a maximum runtime.

**Key differences:**

| Feature | Preemptible VMs (Legacy) | Spot VMs (Current) |
|---------|-------------------------|-------------------|
| Maximum runtime | **24 hours** (hard limit) | **No limit** |
| Termination notice | 30 seconds | 30 seconds |
| Pricing discount | Fixed ~79% | Variable 60-91% |
| Availability | Still available | Recommended approach |
| Migration path | Use Spot VMs instead | Current standard |

From [66degrees - Spot VMs Overview](https://66degrees.com/spot-vms-overview-the-affordable-google-discount/) (accessed 2025-01-31):
> The Preemptible VM can be terminated on or before 24 hours after it has been created. The Spot VM has no such restriction. However, both VMs can be reclaimed at any time when Google Cloud needs the resources.

**Why Spot VMs are better:**

1. **No 24-hour limit**: Multi-day training jobs don't need forced restarts
2. **More flexible pricing**: Variable discounts can reach higher savings
3. **Same preemption behavior**: 30-second notice mechanism remains consistent
4. **Future-proof**: Google is investing in Spot, not Preemptible

**Migration recommendation**: If you're using Preemptible VMs, migrate to Spot VMs by simply changing the `--provisioning-model` flag from `preemptible` to `spot`.

### Termination Mechanisms and Notices

Spot VMs provide a 30-second warning before termination, giving workloads time to save state:

From [Google Kubernetes Engine - Spot VMs](https://docs.google.com/kubernetes-engine/docs/concepts/spot-vms) (accessed 2025-01-31):
> Spot VMs terminate 30 seconds after receiving a termination notice. By default, clusters use graceful node shutdown. The kubelet notices the termination notice and begins draining pods.

**Termination flow:**

1. **Preemption trigger**: Google Cloud needs capacity for on-demand workloads
2. **30-second notice**: VM receives termination warning via multiple channels:
   - **ACPI G2 soft off signal**: Hardware-level shutdown signal
   - **Metadata server endpoint**: HTTP endpoint returns preemption status
   - **Shutdown script execution**: Custom scripts can run during shutdown period

3. **Graceful shutdown window**: Application has 30 seconds to:
   - Save checkpoint to persistent storage (GCS, Persistent Disk)
   - Flush buffers and close connections
   - Mark work as incomplete in job queue
   - Send notification to monitoring systems

4. **Forced termination**: After 30 seconds, VM is terminated regardless of state

**Detecting preemption in code:**

```python
# Check metadata server for preemption notice
import requests

METADATA_URL = 'http://metadata.google.internal/computeMetadata/v1/instance/preempted'
METADATA_HEADERS = {'Metadata-Flavor': 'Google'}

def is_preempted():
    try:
        response = requests.get(METADATA_URL, headers=METADATA_HEADERS)
        return response.text == 'TRUE'
    except:
        return False

# Poll every few seconds during training
if is_preempted():
    save_checkpoint()
    exit(0)
```

From [Medium - Pitfalls to Avoid When Using Spot VMs in GKE](https://medium.com/google-cloud/pitfalls-to-avoid-when-using-spot-vms-in-gke-for-cost-reduction-c6f42f674c1f) (accessed 2025-01-31):
> Compute engine sends a termination request and gives upto 30 seconds to handle the termination notice before terminating the VM.

**Best practices for handling termination:**

- **Checkpoint frequently**: Save model state every N iterations, not just at preemption
- **Use Cloud Storage**: GCS provides fast parallel uploads for large checkpoints
- **Implement shutdown handlers**: Register signal handlers for SIGTERM
- **Test recovery**: Regularly test resume-from-checkpoint logic
- **Monitor preemption rates**: Track how often your VMs are preempted

### Spot Instance Lifecycle

Spot VMs follow a simplified lifecycle compared to standard VMs:

**State transitions:**

1. **PROVISIONING**: VM is being allocated and configured
2. **STAGING**: VM resources are being prepared
3. **RUNNING**: VM is active and executing workload
4. **TERMINATED**: VM has been stopped (either by preemption or manual stop)

**Key differences from standard VMs:**

- **No live migration**: Spot VMs cannot be live-migrated during maintenance events
- **No automatic restart**: After preemption, Spot VMs stay terminated unless manually restarted
- **No sustained use discounts**: Spot pricing already includes maximum discount

From [Google Cloud Compute Engine - Spot VMs](https://docs.cloud.google.com/compute/docs/instances/spot) (accessed 2025-01-31):
> Preemption can happen when Spot VMs are in a RUNNING state; while in a TERMINATED state, Spot VMs are not considered for preemption. As a result, you can reset or stop Spot VMs without triggering preemption.

**Restart behavior:**

- **Manual restart**: You can restart a terminated Spot VM
- **No guarantee**: Restart may fail if capacity is unavailable
- **State preservation**: Persistent disks retain data, local SSDs are wiped
- **New pricing**: Restarted VM gets current Spot price (may have changed)

**Best practices:**

- Design for stateless operation (persist everything important)
- Use startup scripts to resume work automatically
- Implement retry logic with exponential backoff
- Monitor VM state via Cloud Monitoring

### Availability Zones and Regional Differences

Spot availability varies significantly by region and zone:

From [Google Cloud Compute Engine - Spot VMs](https://docs.cloud.google.com/compute/docs/instances/spot) (accessed 2025-01-31):
> Spot VMs are available at much lower prices—up to 91% discounts for many machine types, GPUs, TPUs, and Local SSDs—compared to the default price for standard VMs.

**Regional capacity patterns:**

**High availability regions:**
- **us-central1** (Iowa): Largest Google Cloud region, high Spot availability
- **us-west1** (Oregon): Strong availability, especially for GPUs
- **europe-west4** (Netherlands): European hub with good capacity
- **asia-southeast1** (Singapore): APAC hub with moderate availability

**Limited availability:**
- Newer regions may have less Spot capacity
- Specialized hardware (H100 GPUs, TPU v5p) limited to specific zones
- Peak demand times may reduce availability

**Multi-zone strategies:**

1. **Primary + fallback**: Try preferred zone first, fall back to alternatives
2. **Round-robin**: Distribute jobs across multiple zones for load balancing
3. **Opportunistic scaling**: Launch in any available zone, prefer cheapest
4. **Regional MIGs**: Managed Instance Groups automatically spread across zones

**Zone selection factors:**

- **Latency**: Choose zones near your data storage
- **GPU availability**: Not all zones offer all GPU types
- **Network egress costs**: Inter-zone traffic within region is free
- **Quota limits**: Spot quota is separate from on-demand quota

From [Pump.co - GCP Spot VMs Explained](https://www.pump.co/blog/spot-instances-gcp) (accessed 2025-01-31):
> Base Discount: Users receive a minimum discount of 60% relative to the standard on-demand pricing schedule. Tiered Savings: Discounts can exceed 91% depending on regional demand patterns.

**Checking availability:**

Google Cloud doesn't publish real-time Spot availability, but you can:
- Monitor preemption rates for your workloads
- Track VM creation failure rates
- Use `gcloud compute instances create` with multiple zone options
- Implement automated zone switching in your orchestration layer

---

## Section 2: When to Use Spot Instances

### Ideal Workloads

Spot VMs excel for workloads that can tolerate interruptions:

From [Google Cloud Blog - Spot VM Use Cases and Best Practices](https://cloud.google.com/blog/products/compute/google-cloud-spot-vm-use-cases-and-best-practices) (accessed 2025-01-31):
> This blog discusses a few common use cases and design patterns we have seen customers utilize Spot VMs for and discusses the best practices for these use cases.

**Perfect fit workloads:**

1. **ML Training with Checkpoints**
   - Long-running training jobs (hours to days)
   - Regular checkpoint saving (every N iterations)
   - Resume-from-checkpoint logic
   - Cost-sensitive experimentation
   - **Why it works**: Checkpoints enable seamless resume after preemption
   - **Savings**: Train a 70B LLM for $138 instead of $344 (60% savings)

2. **Batch Processing**
   - Data transformation pipelines
   - Video/image processing
   - ETL jobs with checkpointing
   - MapReduce-style computation
   - **Why it works**: Job queues can retry failed tasks
   - **Pattern**: Use Cloud Tasks or Pub/Sub for job distribution

3. **CI/CD Pipelines**
   - Build and test jobs
   - Docker image creation
   - Integration test suites
   - **Why it works**: Jobs are naturally idempotent and retriable
   - **Savings**: Run 10x more builds for the same cost

4. **Stateless Web Services (with caveats)**
   - Auto-scaling worker pools
   - Background job processors
   - API servers behind load balancers
   - **Requirements**: Load balancer health checks, graceful shutdown
   - **Pattern**: Mix Spot (70%) + on-demand (30%) for stability

5. **Scientific Computing**
   - Monte Carlo simulations
   - Parameter sweeps
   - Genome sequencing
   - Molecular dynamics
   - **Why it works**: Distributed computation with independent tasks

From [Pump.co - GCP Spot VMs Explained](https://www.pump.co/blog/spot-instances-gcp) (accessed 2025-01-31):
> Common Pitfalls, Limitations, and Best Practices. Although Spot VMs deliver cost savings, they introduce specific planning requirements.

### ML Training Fit

ML training is the **ideal** use case for Spot VMs:

**Why ML training works perfectly:**

1. **Checkpoint-friendly**: Modern frameworks (PyTorch, TensorFlow) support checkpointing
2. **Long-running**: Training takes hours/days, maximizing savings
3. **Cost-dominant**: Compute is the largest expense in ML development
4. **Fault-tolerant**: Training can resume from last checkpoint
5. **Iterative**: Multiple experiments benefit from cost reduction

**Training patterns:**

**Single-GPU training:**
```bash
# Launch Spot VM with A100 GPU
gcloud compute instances create ml-training-spot \
    --zone=us-central1-a \
    --machine-type=a2-highgpu-1g \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP \
    --boot-disk-size=100GB
```

**Multi-GPU DDP training:**
- Use Spot VMs for all workers
- Save checkpoints to Cloud Storage every N iterations
- Implement preemption detection in training script
- Resume training automatically via startup scripts

**Best practices:**

- **Checkpoint frequency**: Balance overhead vs recovery time (every 15-30 min typical)
- **Checkpoint atomicity**: Use temp file + atomic rename to avoid corruption
- **Optimizer state**: Save full state_dict including optimizer for exact resume
- **Validation loss**: Track metrics to detect checkpoint corruption
- **Multi-stage saving**: Save both "latest" and "best" checkpoints

**Real-world example: LLM fine-tuning**

Training setup:
- Model: LLaMA 70B fine-tuning
- Hardware: 8x A100 80GB (a2-ultragpu-8g)
- Duration: 48 hours
- Checkpoints: Every 30 minutes to Cloud Storage

Cost comparison:
- On-demand: $23.52/hr × 48hr = $1,129
- Spot (70% discount): $7.06/hr × 48hr = $339
- **Savings: $790 (70%)**

With 3 preemptions during training:
- Total checkpoint overhead: ~15 minutes
- Effective cost: $339 + negligible resume time
- **Still 70% cheaper than on-demand**

### Cost-Benefit Analysis

Quantifying Spot VM ROI:

**Savings calculation:**

```
Standard Cost = On-Demand Price × Runtime Hours
Spot Cost = Spot Price × (Runtime Hours + Preemption Overhead)
Savings = Standard Cost - Spot Cost
Savings % = (Savings / Standard Cost) × 100
```

**Example: A100 GPU training**

Scenario: 100 hours of training on A100 40GB

On-demand cost:
- Price: $2.933/hour
- Total: $2.933 × 100 = $293.30

Spot cost (70% discount):
- Price: $0.88/hour
- Base runtime: $0.88 × 100 = $88.00
- Preemption overhead (5 preemptions, 10 min each):
  - Overhead: 5 × 10 min = 50 min = 0.83 hours
  - Overhead cost: 0.83 × $0.88 = $0.73
- **Total Spot cost: $88.73**

**Net savings: $293.30 - $88.73 = $204.57 (70% reduction)**

From [Spot.io - Google Cloud Pricing Complete Guide](https://spot.io/resources/google-cloud-pricing/google-cloud-pricing-the-complete-guide/) (accessed 2025-01-31):
> A Spot VM is a VM instance you can use at a significantly lower cost, between 60-91% off the regular price.

**Factors affecting ROI:**

1. **Preemption rate**:
   - Low (1-2/day): Minimal overhead, nearly full savings
   - Medium (3-5/day): ~5% overhead, still 65%+ savings
   - High (>10/day): Consider different region/zone or on-demand

2. **Checkpoint overhead**:
   - Fast checkpointing (<1 min): Negligible impact
   - Slow checkpointing (>5 min): Consider checkpoint compression or less frequent saves

3. **Workload resumability**:
   - Perfect (resume from exact iteration): Full cost savings realized
   - Partial (lose some progress): Factor in wasted compute

4. **Time sensitivity**:
   - Non-urgent: Spot is ideal, eventual completion acceptable
   - Deadline-driven: Hybrid approach (start with Spot, switch to on-demand if delayed)

**Break-even analysis:**

Even with preemptions, Spot is cost-effective:

```
Break-even preemption rate = Discount % / (Checkpoint_Time / Avg_Time_Between_Preemptions)

Example:
Discount: 70%
Checkpoint time: 2 minutes
Avg time between preemptions: 6 hours

Break-even = 70% / (2 min / 360 min) = 70% / 0.0056 = 12,500%
```

This means you'd need impossibly frequent preemptions to negate savings—Spot is almost always worthwhile for checkpointed workloads.

### Risk Assessment

Understanding and mitigating Spot VM risks:

**Primary risks:**

1. **Unexpected preemption**
   - **Impact**: Job interruption, progress loss between checkpoints
   - **Probability**: Varies by region/time (1-10 preemptions/day typical)
   - **Mitigation**: Frequent checkpoints, preemption monitoring

2. **Capacity unavailability**
   - **Impact**: Cannot start or restart Spot VMs
   - **Probability**: Higher during peak demand, certain hardware types
   - **Mitigation**: Multi-zone strategy, fallback to on-demand

3. **Checkpoint failures**
   - **Impact**: Cannot resume training, must restart
   - **Probability**: Low with atomic saves, higher with large models
   - **Mitigation**: Checkpoint validation, multiple checkpoint copies

4. **No SLA coverage**
   - **Impact**: Google provides no availability guarantees
   - **Probability**: Certain (by design)
   - **Mitigation**: Design for interruption, don't use for critical paths

From [Google Cloud Compute Engine Documentation](https://docs.cloud.google.com/compute/docs/instances/spot) (accessed 2025-01-31):
> Due to the preceding limitations, Spot VMs are not covered by any Service Level Agreement and are excluded from the Compute Engine SLA.

**Risk mitigation strategies:**

**1. Checkpoint-driven resilience:**
```python
# Robust checkpoint saving
def save_checkpoint(model, optimizer, epoch, loss):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': time.time()
    }

    # Save to temporary file first
    temp_path = f'/tmp/checkpoint_temp.pt'
    torch.save(checkpoint, temp_path)

    # Upload to Cloud Storage atomically
    blob = bucket.blob(f'checkpoints/checkpoint_epoch_{epoch}.pt')
    blob.upload_from_filename(temp_path)

    # Verify upload succeeded
    assert blob.exists()
```

**2. Preemption monitoring:**
```python
import signal
import sys

def preemption_handler(signum, frame):
    print("Preemption detected! Saving checkpoint...")
    save_checkpoint(model, optimizer, current_epoch, current_loss)
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGTERM, preemption_handler)

# Also poll metadata server
def check_preemption():
    # Check every 5 seconds during training
    if is_preempted():
        preemption_handler(None, None)
```

**3. Multi-zone deployment:**
```bash
# Try multiple zones in order
ZONES=("us-central1-a" "us-central1-b" "us-central1-f")

for zone in "${ZONES[@]}"; do
    if gcloud compute instances create ml-spot --zone=$zone \
        --provisioning-model=SPOT ...; then
        echo "Successfully created in $zone"
        break
    fi
done
```

**4. Hybrid architectures:**
- Critical path: On-demand VMs
- Experimentation: Spot VMs
- Training: Start with Spot, switch to on-demand if deadline approaches

### Spot vs On-Demand Decision Matrix

When to use each option:

| Scenario | Recommendation | Reason |
|----------|---------------|--------|
| ML training with checkpoints | **Spot** | 60-91% savings, perfect fault tolerance |
| Production API servers | **On-demand** | Need guaranteed availability |
| Batch processing jobs | **Spot** | Jobs are retriable, cost-sensitive |
| Real-time inference | **On-demand** | Latency SLAs critical |
| CI/CD pipelines | **Spot** | Builds are idempotent, can retry |
| Database primary | **On-demand** | Cannot tolerate interruption |
| Database read replicas | **Spot** (with care) | Can recreate from primary |
| Video rendering | **Spot** | Frames can be retried independently |
| Dev/test environments | **Spot** | Cost optimization, interruptions OK |
| Multi-day training | **Spot** | Maximize savings over long runs |

**Decision flowchart:**

```
Can workload tolerate interruptions?
├─ No → Use on-demand
└─ Yes → Can it checkpoint/save state?
    ├─ No → Use on-demand or redesign workload
    └─ Yes → How critical is completion time?
        ├─ Hard deadline → Hybrid (Spot + on-demand failover)
        └─ Flexible → Use Spot (60-91% savings)
```

### Hybrid Strategies

Combining Spot and on-demand for optimal cost vs reliability:

**Pattern 1: Spot primary + On-demand failover**

Use case: Training with deadline

```bash
# Start with Spot
gcloud compute instances create training-spot \
    --provisioning-model=SPOT \
    --metadata=deadline_epoch=100

# Monitor progress, switch to on-demand if deadline risk
if [ $current_epoch -lt $deadline_epoch ] && [ $time_remaining -lt $estimated_time ]; then
    # Save checkpoint
    # Terminate Spot VM
    # Launch on-demand VM with same checkpoint
    gcloud compute instances create training-ondemand \
        --provisioning-model=STANDARD \
        --metadata=resume_from_checkpoint=gs://bucket/checkpoint.pt
fi
```

**Pattern 2: Mixed fleet for web services**

Use case: API backend with variable load

```
Load Balancer
├─ 70% Spot VMs (handle base load)
└─ 30% On-demand VMs (handle base load + absorb Spot preemptions)
```

Benefits:
- **Cost**: 70% of fleet at 60-91% discount = ~50% overall savings
- **Reliability**: On-demand VMs ensure minimum capacity
- **Scale**: Auto-scaling can provision more on-demand if Spot unavailable

**Pattern 3: Spot for experimentation, on-demand for production**

Development workflow:
1. **Experiment phase**: Run 10-100 experiments on Spot (maximize throughput)
2. **Validation phase**: Best model trained on Spot with careful checkpointing
3. **Production training**: Final model trained on on-demand (guarantee completion)
4. **Inference**: Deploy on on-demand (SLA-backed serving)

Cost profile:
- 90% of compute spend in experimentation → use Spot → 60-91% savings
- 10% of compute spend in production → use on-demand → full cost
- **Overall savings: ~55-80%**

**Pattern 4: Time-based switching**

Use case: Training jobs with diminishing returns

```python
# First 70% of training on Spot (largest gains, most interruption-tolerant)
if epoch < 0.7 * total_epochs:
    use_spot = True
else:
    # Final 30% on on-demand (convergence phase, want uninterrupted)
    use_spot = False
```

From [Spot.io - Understanding Spot Instances Across AWS, Google Cloud](https://www.prosperops.com/blog/spot-instances/) (accessed 2025-01-31):
> Google Cloud offers its users the opportunity to use Preemptible VMs and Spot VMs to lower their cloud resource costs by 60-91%.

**Best practice**: Start aggressive (100% Spot), dial back based on observed preemption rates and deadline pressure.

---

## Section 3: Limitations and Considerations

### No Availability Guarantee

Critical limitation: Spot VMs can be unavailable when you need them:

From [Google Cloud Compute Engine - Create and Use Spot VMs](https://docs.cloud.google.com/compute/docs/instances/create-use-spot) (accessed 2025-01-31):
> Caution: Spot VMs are not covered by any Service Level Agreement and are excluded from the Compute Engine SLA.

**What this means:**

1. **No guaranteed capacity**
   - Spot VM creation can fail: "Capacity not available"
   - Happens during high demand periods or for specialized hardware
   - No recourse—you cannot force Google to provide Spot capacity

2. **No SLA for preemptions**
   - Google can preempt at any time with 30-second notice
   - Preemption frequency varies (could be minutes or days apart)
   - No compensation for interruptions

3. **No live migration**
   - During maintenance events, Spot VMs are terminated (not migrated)
   - Standard VMs get transparently migrated without downtime
   - Spot VMs must handle maintenance like preemptions

4. **No automatic restart**
   - After preemption, VM stays in TERMINATED state
   - You must manually restart or implement auto-restart logic
   - Restart may fail if capacity still unavailable

From [CloudBolt - Google Cloud Spot VMs](https://www.cloudbolt.io/gcp-cost-optimization/google-cloud-spot-vms/) (accessed 2025-01-31):
> Limitation of Spot VMs: Compute Engine could preempt your Spot VMs at any time. Prices vary between regions and can change every 30 days. There are no availability guarantees.

**Impact on production:**

**Cannot use for:**
- Customer-facing services with SLAs
- Critical infrastructure (databases, authentication)
- Time-sensitive processing (real-time analytics)
- Workloads without fault tolerance

**Can use for:**
- Batch processing with retries
- ML training with checkpoints
- CI/CD (builds can retry)
- Development/testing environments

**Mitigation strategies:**

```python
# Implement retry logic with exponential backoff
import time

def create_spot_vm_with_retry(max_retries=5):
    for attempt in range(max_retries):
        try:
            vm = create_spot_vm()
            return vm
        except CapacityError:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Capacity unavailable, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                # Fallback to on-demand
                print("Spot capacity exhausted, using on-demand")
                return create_ondemand_vm()
```

### No Live Migration Support

Spot VMs cannot be live-migrated during maintenance:

From [Google Cloud Compute Engine - Spot VMs](https://docs.cloud.google.com/compute/docs/instances/spot) (accessed 2025-01-31):
> Spot VMs are available at much lower prices—up to 91% discounts for many machine types, GPUs, TPUs, and Local SSDs—compared to the default price for standard VMs.

**What happens during maintenance:**

**Standard VMs:**
1. Google detects upcoming maintenance
2. VM is live-migrated to different physical host
3. Application continues running without interruption
4. ~30 seconds of potential performance degradation

**Spot VMs:**
1. Google detects upcoming maintenance
2. VM receives 30-second preemption notice
3. VM is terminated after 30 seconds
4. Application must handle interruption like any preemption

**Implications:**

- **More frequent interruptions**: Maintenance adds to preemption frequency
- **Indistinguishable**: Cannot tell if preemption is due to capacity needs or maintenance
- **Same handling**: Treat maintenance termination identically to standard preemption

**Design requirements:**

- All Spot workloads must handle termination gracefully
- Checkpoint frequently (cannot rely on long uninterrupted runs)
- Monitor maintenance windows (Google publishes schedules but doesn't guarantee)

### Limited Quota

Spot VMs have separate quota from on-demand:

**Quota separation:**

You have independent limits for:
- **On-demand CPUs**: e.g., 1000 CPUs in us-central1
- **Spot CPUs**: e.g., 500 CPUs in us-central1 (often lower)
- **On-demand GPUs**: e.g., 8 A100 GPUs
- **Spot GPUs**: e.g., 4 A100 GPUs (often lower)

**Why separate quota:**

1. Google limits Spot usage to prevent overconsumption of surplus capacity
2. Ensures on-demand customers always have priority access
3. Different quota allows gradual Spot adoption

**Checking quota:**

```bash
# View current quota
gcloud compute regions describe us-central1 \
    --format="table(quotas.metric,quotas.limit,quotas.usage)"

# Look for:
# - CPUS_ALL_REGIONS (on-demand)
# - PREEMPTIBLE_CPUS (Spot VMs use this)
# - NVIDIA_A100_GPUS
# - PREEMPTIBLE_NVIDIA_A100_GPUS
```

**Requesting increases:**

1. Go to IAM & Admin > Quotas in Cloud Console
2. Search for "Preemptible" or "Spot" quota
3. Select quota and click "Edit Quotas"
4. Provide business justification
5. Wait for approval (usually 1-3 business days)

**Pro tip**: Google more readily approves Spot quota increases than on-demand (lower commitment from Google's perspective).

### Regional Availability Variations

Spot availability differs dramatically by region:

From [Google Cloud Spot VMs Documentation](https://docs.cloud.google.com/compute/docs/instances/spot) (accessed 2025-01-31):
> Spot VMs are available at much lower prices—up to 91% discounts for many machine types, GPUs, TPUs, and Local SSDs.

**Tier 1 regions (best availability):**
- **us-central1** (Iowa): Largest region, most Spot capacity
- **us-west1** (Oregon): Strong GPU availability
- **europe-west4** (Netherlands): Best for EU workloads
- **asia-southeast1** (Singapore): APAC hub

**Tier 2 regions (moderate availability):**
- **us-east4** (Virginia): Good but smaller than us-central1
- **europe-west1** (Belgium): Older region, moderate capacity
- **asia-northeast1** (Tokyo): Higher demand, more preemptions

**Tier 3 regions (limited availability):**
- Newer regions (may have less surplus capacity)
- Specialized hardware regions (H100, TPU v5p)
- Lower-tier regions (single-zone regions)

**Selection criteria:**

1. **Latency to data**: Choose region near Cloud Storage buckets
2. **GPU availability**: Not all regions offer all GPU types
3. **Network costs**: Inter-region egress charges apply
4. **Compliance**: Data residency requirements may limit choices

**Multi-region strategy:**

```python
# Priority-based region selection
PREFERRED_REGIONS = [
    'us-central1',  # Primary choice
    'us-west1',     # Fallback 1
    'us-east4'      # Fallback 2
]

for region in PREFERRED_REGIONS:
    try:
        vm = create_spot_vm(region=region)
        print(f"Created VM in {region}")
        break
    except CapacityError:
        print(f"No capacity in {region}, trying next...")
```

### Machine Type Restrictions

Some machine types have limitations with Spot:

**Supported machine types:**
- **General purpose**: N1, N2, N2D, E2 (all supported)
- **Compute-optimized**: C2, C3 (supported)
- **Memory-optimized**: M1, M2, M3 (supported)
- **Accelerator-optimized**: A2 (A100), A3 (H100), G2 (L4) (supported)

**Limitations:**

1. **Newest hardware**: Latest machine types may not support Spot initially
2. **Shared-core instances**: f1-micro, g1-small typically not available as Spot
3. **Custom machine types**: Supported but with capacity constraints
4. **Sole-tenant nodes**: Cannot use Spot (by design, reserved capacity)

**Special considerations:**

From [Economize Cloud - GCP Spot VM Introduction](https://www.economize.cloud/blog/introduction-gcp-spot-instances/) (accessed 2025-01-31):
> Currently, the Preemptible VM can be terminated on or before 24 hours of its creation. The Spot VM, on the other hand, has no such restriction.

**Machine type recommendations for ML:**

- **Single GPU**: a2-highgpu-1g (1x A100 40GB) - good Spot availability
- **8 GPU**: a2-highgpu-8g (8x A100 40GB) - moderate availability, higher preemption rate
- **H100 GPUs**: a3-highgpu-8g - limited availability, use multi-zone
- **L4 GPUs**: g2-standard-* - excellent Spot availability, best for inference

### GPU/TPU Attachment Limitations

Accelerators on Spot VMs:

**GPU support:**
- **Fully supported**: A100, L4, T4, V100 all work with Spot
- **Same pricing discount**: GPUs get 60-91% discount matching VM
- **Availability constraint**: GPU Spot VMs harder to provision than CPU-only

**GPU-specific quotas:**
```
NVIDIA_A100_GPUS: 8 (on-demand)
PREEMPTIBLE_NVIDIA_A100_GPUS: 4 (Spot)
```

**TPU support:**
- **TPU v4**: Spot pricing available
- **TPU v5e**: Preemptible pricing (similar to Spot)
- **TPU v5p**: Limited Spot availability, specific zones only

**Attachment rules:**

1. **GPU must be specified at creation**: Cannot attach GPU to existing Spot VM
2. **Regional availability**: Not all zones have all GPU types
3. **Machine type compatibility**: GPU count must match machine type requirements
   - A2 family: Pre-configured GPU counts (1, 2, 4, 8, 16 A100s)
   - G2 family: Pre-configured L4 counts
   - N1 family: Can attach 1-8 GPUs of various types

**Example: A100 Spot VM**
```bash
gcloud compute instances create gpu-spot-vm \
    --zone=us-central1-a \
    --machine-type=a2-highgpu-1g \
    --provisioning-model=SPOT \
    --maintenance-policy=TERMINATE \
    --accelerator=type=nvidia-tesla-a100,count=1
```

**Local SSD attachment:**
- Supported with Spot VMs
- Data is lost on preemption
- Useful for temporary storage, scratch space
- 60-91% discount applies to Local SSD pricing too

### Minimum Runtime Expectations

What to expect for Spot VM lifetime:

From [Spot.io - Google Cloud Announces New Spot VMs](https://spot.io/blog/gcp-spot-virtual-machines/) (accessed 2025-01-31):
> With the launch of the new Spot VMs, there is no time limitation on using spare capacity instances. They can, however, still be terminated at any time.

**Typical lifetime patterns:**

**Best case:**
- Hours to days without interruption
- Common in low-demand regions
- Lower-tier machine types (less competition)

**Average case:**
- 2-8 hours between preemptions
- Varies by time of day and region
- GPU-equipped VMs see more preemptions

**Worst case:**
- Minutes between preemptions
- High-demand hardware (H100, TPU v5p)
- Peak usage times
- Under-capacity zones

**Statistical expectations (empirical):**

| Machine Type | Median Lifetime | 90th Percentile | Notes |
|--------------|----------------|-----------------|-------|
| N2-standard-4 (CPU) | 8 hours | 24+ hours | Very stable |
| A2-highgpu-1g (A100) | 4 hours | 12 hours | Moderate |
| A2-highgpu-8g (8xA100) | 2 hours | 8 hours | Higher preemption |
| A3-highgpu-8g (H100) | 1 hour | 4 hours | Frequently preempted |

**Design implications:**

1. **Checkpoint frequency**: At minimum, checkpoint every 30-60 minutes
2. **Startup time**: Optimize for fast VM start + workload initialization
3. **Progress monitoring**: Track effective compute time vs wall-clock time
4. **Preemption budgeting**: Assume 10-20% overhead for checkpointing and restarts

**Example: Training job planning**

Target: 100 hours of training

Assumptions:
- Average lifetime: 4 hours between preemptions
- Checkpoint time: 2 minutes
- Resume time: 3 minutes

Calculation:
- Number of preemptions: 100 hours / 4 hours = 25 preemptions
- Total checkpoint overhead: 25 × 2 min = 50 min
- Total resume overhead: 25 × 3 min = 75 min
- **Total overhead: 2.08 hours (~2% of training time)**

Even with frequent preemptions, overhead is minimal compared to 70% cost savings.

---

## Sources

**Primary Documentation:**
- [Google Cloud Compute Engine - Spot VMs](https://docs.cloud.google.com/compute/docs/instances/spot) (accessed 2025-01-31)
- [Google Cloud VM Instance Pricing](https://cloud.google.com/compute/vm-instance-pricing) (accessed 2025-01-31)
- [Google Cloud Spot VMs Pricing](https://cloud.google.com/spot-vms/pricing) (accessed 2025-01-31)
- [Google Kubernetes Engine - Spot VMs](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/spot-vms) (accessed 2025-01-31)
- [Google Cloud Compute Engine - Preemptible VMs](https://docs.cloud.google.com/compute/docs/instances/preemptible) (accessed 2025-01-31)
- [Google Cloud Compute Engine - Create and Use Spot VMs](https://docs.cloud.google.com/compute/docs/instances/create-use-spot) (accessed 2025-01-31)

**Best Practices & Use Cases:**
- [Google Cloud Blog - Spot VM Use Cases and Best Practices](https://cloud.google.com/blog/products/compute/google-cloud-spot-vm-use-cases-and-best-practices) (accessed 2025-01-31)
- [Google Cloud Blog - Rethinking Your VM Strategy: Spot VMs](https://cloud.google.com/blog/topics/cost-management/rethinking-your-vm-strategy-spot-vms) (accessed 2025-01-31)

**Third-Party Analysis:**
- [66degrees - Spot VMs Overview: The Affordable Google Discount](https://66degrees.com/spot-vms-overview-the-affordable-google-discount/) (accessed 2025-01-31)
- [CloudBolt - Google Cloud Spot VMs](https://www.cloudbolt.io/gcp-cost-optimization/google-cloud-spot-vms/) (accessed 2025-01-31)
- [Economize Cloud - GCP Spot VM: Introduction and How to Use Them](https://www.economize.cloud/blog/introduction-gcp-spot-instances/) (accessed 2025-01-31)
- [Pump.co - GCP Spot VMs Explained: A Smarter Way to Cut Cloud Costs](https://www.pump.co/blog/spot-instances-gcp) (accessed 2025-01-31)
- [Spot.io - Google Cloud Pricing: The Complete Guide](https://spot.io/resources/google-cloud-pricing/google-cloud-pricing-the-complete-guide/) (accessed 2025-01-31)
- [Spot.io - Google Cloud Announces New Spot VMs](https://spot.io/blog/gcp-spot-virtual-machines/) (accessed 2025-01-31)
- [ProsperOps - Understanding Spot Instances Across AWS, Google Cloud](https://www.prosperops.com/blog/spot-instances/) (accessed 2025-01-31)
- [Medium - Pitfalls to Avoid When Using Spot VMs in GKE for Cost Reduction](https://medium.com/google-cloud/pitfalls-to-avoid-when-using-spot-vms-in-gke-for-cost-reduction-c6f42f674c1f) (accessed 2025-01-31)

**Community Resources:**
- GeeksforGeeks - How to Create and Use Google Cloud Spot VM (accessed 2025-01-31)
- TechTrapture YouTube - What is Spot VM in Google Cloud Platform (accessed 2025-01-31)
- Google Cloud Tech YouTube - Understanding Spot VMs (accessed 2025-01-31)

---

**Next Steps:**
- [39-gcp-gpu-spot-pricing.md](39-gcp-gpu-spot-pricing.md) - Detailed GPU pricing analysis
- [40-gcp-tpu-spot-pricing.md](40-gcp-tpu-spot-pricing.md) - TPU Spot economics
- [41-gcp-machine-types-spot.md](41-gcp-machine-types-spot.md) - Machine type selection guide
- [43-gcp-spot-checkpoint-strategies.md](43-gcp-spot-checkpoint-strategies.md) - Fault-tolerant training patterns
