# CUDA Advanced Multi-GPU Troubleshooting & NCCL Debugging (Expert)

**Expert-level guide to debugging multi-GPU training failures, NCCL errors, DDP synchronization issues, and hard edge cases**

**Target audience**: ML engineers debugging production multi-GPU training, researchers scaling to large clusters

---

## Overview

Multi-GPU training introduces complex failure modes beyond single-GPU development:
- **NCCL communication errors** (timeouts, aborted communicators, peer access failures)
- **DDP gradient synchronization bugs** (out-of-sync gradients, bucket issues)
- **Hard edge cases** (ECC errors, thermal throttling, driver crashes, checkpoint corruption)

This guide covers expert-level troubleshooting for these scenarios with real error messages, diagnostic workflows, and recovery strategies.

---

## Section 1: Multi-GPU Initialization & Peer Access Failures (~100 lines)

### 1.1 NCCL Initialization Errors

**Error: `Socket Timeout` during NCCL setup**

```
torch.distributed.DistBackendError: is setting up NCCL communicator and retrieving
ncclUniqueId from via c10d key-value store by key '1', but store->get('1') got error:
Socket Timeout
```

**What's happening:**
- Rank 0 (coordinator) failed to share `ncclUniqueId` with other ranks
- TCP key-value store communication blocked or timed out
- Network instability between nodes

**Diagnosis workflow:**

```bash
# 1. Verify all nodes can communicate
ping <node_ip>

# 2. Check network interface configuration
ifconfig
# Identify correct interface (e.g., eth0, ib0, enp0s31f6)

# 3. Test NCCL directly with nccl-tests
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make MPI=1 MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi
mpirun -np 4 --allow-run-as-root ./build/all_reduce_perf -b 8 -e 256M -f 2 -g 1

# 4. Enable NCCL debug logging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
```

**Solutions:**

```bash
# Specify correct network interface
export NCCL_SOCKET_IFNAME=eth0  # or ib0 for InfiniBand

# Increase timeout for slow networks
export NCCL_TIMEOUT=3600  # seconds
# In Python:
import torch
from datetime import timedelta
torch.distributed.init_process_group(
    backend='nccl',
    timeout=timedelta(seconds=3600)
)

# Monitor rank 0 stability
nvidia-smi dmon -i 0 -s pucvmet -c 10
# Watch for GPU 0 crashes during init
```

**Prevention:**
- Use dedicated network for distributed training (separate from management network)
- Verify firewall rules allow communication on all ports
- Test network stability with `iperf` before training

From [Medium - Debugging NCCL Errors](https://medium.com/@devaru.ai/debugging-nccl-errors-in-distributed-training-a-comprehensive-guide-28df87512a34) (accessed 2025-11-13):
> "Rank 0 acts as the coordinator that initializes communication between GPUs. If rank 0 crashes or fails to respond, other ranks cannot retrieve the ncclUniqueId."

### 1.2 Peer-to-Peer Access Failures

**Error: `peer access already enabled` or `peer access not supported`**

```
RuntimeError: CUDA error: peer access already enabled
RuntimeError: CUDA error: peer access is not supported between these two devices
```

**What's happening:**
- GPU topology doesn't support P2P (e.g., consumer GPUs without NVLink)
- Attempting to enable P2P twice on same device pair
- PCIe topology limits direct peer access

**Check P2P support:**

```python
import torch

def check_p2p_access():
    """Comprehensive P2P capability check"""
    n_gpus = torch.cuda.device_count()
    print(f"Found {n_gpus} GPUs")

    for i in range(n_gpus):
        for j in range(n_gpus):
            if i != j:
                can_access = torch.cuda.can_device_access_peer(i, j)
                print(f"GPU {i} -> GPU {j}: {can_access}")

                if can_access:
                    # Get P2P attribute (0=no P2P, 1=P2P supported)
                    p2p_attr = torch.cuda.get_device_properties(i).multi_processor_count
                    print(f"  Link type: {p2p_attr}")

check_p2p_access()

# Expected output for NVLink GPUs:
# GPU 0 -> GPU 1: True
#   Link type: 108 (A100 SMs)
# GPU 1 -> GPU 0: True
```

**Workarounds for non-P2P systems:**

```bash
# Disable P2P in NCCL (force through CPU)
export NCCL_P2P_DISABLE=1

# Use NCCL's SHM (shared memory) path instead
export NCCL_SHM_DISABLE=0

# For debugging, enable verbose NCCL topology logging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=GRAPH,TUNING
```

**DDP without P2P:**

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize without assuming P2P
dist.init_process_group(backend='nccl')

model = MyModel().cuda()
# DDP will use NCCL's fallback path (SHM or network)
ddp_model = DDP(model, device_ids=[local_rank])

# Verify no P2P required
print(f"NCCL P2P disabled: {os.environ.get('NCCL_P2P_DISABLE', 'not set')}")
```

From [NVIDIA Forums - Multi-GPU P2P](https://forums.developer.nvidia.com/t/multi-gpu-peer-to-peer-access-cuda-sdk-example-not-working-why/35958) (accessed 2025-11-13):
> "cudaMemcpyPeer does not work if P2P access is not supported. Use cudaDeviceCanAccessPeer to check before enabling P2P."

### 1.3 Multi-GPU Device Ordering Issues

**Problem: GPU IDs don't match expected order**

```python
# Wrong: Assumes CUDA_VISIBLE_DEVICES ordering
device = torch.device(f"cuda:{rank}")  # May not match physical GPU!

# Correct: Use local_rank from torch.distributed
import torch.distributed as dist

local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

print(f"Rank {dist.get_rank()}, Local Rank {local_rank}, Device {device}")
# Rank 0, Local Rank 0, Device cuda:0
# Rank 1, Local Rank 1, Device cuda:1
```

**Diagnostic commands:**

```bash
# Check GPU bus topology
nvidia-smi topo -m

# Expected output shows NVLink connections:
#         GPU0    GPU1    GPU2    GPU3
# GPU0     X      NV12    NV12    NV12
# GPU1    NV12     X      NV12    NV12
# GPU2    NV12    NV12     X      NV12
# GPU3    NV12    NV12    NV12     X

# Verify CUDA_VISIBLE_DEVICES mapping
echo $CUDA_VISIBLE_DEVICES
python -c "import torch; print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
```

---

## Section 2: DDP/FSDP Gradient Synchronization Failures (~100 lines)

### 2.1 Out-of-Sync Gradients Across Ranks

**Error: Gradients differ across GPUs after backward()**

```python
# Symptom: Different loss values on each rank
# Rank 0: loss=0.523
# Rank 1: loss=0.891  # Should match rank 0!
```

**Root cause: Gradient synchronization disabled or broken**

From [PyTorch Forums - DDP Gradient Sync](https://discuss.pytorch.org/t/ddp-and-gradient-sync/206096) (accessed 2025-11-13):
> "I found that my gradients were not being synced after loss.backward() call. The reason was using no_sync() context manager incorrectly."

**Diagnosis:**

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Check if gradients are synchronized
def check_gradient_sync(model):
    """Verify gradients match across all ranks"""
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Gather gradients from all ranks
            grad_list = [torch.zeros_like(param.grad) for _ in range(dist.get_world_size())]
            dist.all_gather(grad_list, param.grad)

            # Compare gradients
            for i, grad in enumerate(grad_list[1:], 1):
                if not torch.allclose(grad_list[0], grad, rtol=1e-5):
                    print(f"MISMATCH in {name}: rank 0 vs rank {i}")
                    print(f"  Max diff: {(grad_list[0] - grad).abs().max()}")
                    return False
    return True

# After backward pass:
loss.backward()
if check_gradient_sync(ddp_model.module):
    print("✓ Gradients synchronized correctly")
else:
    print("✗ Gradient synchronization FAILED")
```

**Common causes and fixes:**

```python
# 1. Using no_sync() incorrectly
# WRONG:
with ddp_model.no_sync():
    loss = model(x)
    loss.backward()  # Gradients NOT synced!
optimizer.step()  # Ranks diverge!

# CORRECT:
for i, (x, y) in enumerate(dataloader):
    if i % accumulation_steps != 0:
        # Accumulate gradients without sync
        with ddp_model.no_sync():
            loss = model(x)
            loss.backward()
    else:
        # Final step: sync gradients
        loss = model(x)
        loss.backward()  # DDP syncs here

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

```python
# 2. Unused parameters breaking DDP
# DDP requires all parameters to be used in forward pass
ddp_model = DDP(
    model,
    device_ids=[local_rank],
    find_unused_parameters=True  # Enable if some params unused
)

# Check for unused parameters warning:
# UserWarning: find_unused_parameters=True was specified in DDP constructor,
# but did not find any unused parameters...
```

```python
# 3. Gradient clipping BEFORE DDP sync
# WRONG:
loss.backward()  # DDP syncs gradients here
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Too late!
optimizer.step()

# CORRECT:
loss.backward()  # DDP syncs
# Clipping happens on already-synced gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

### 2.2 DDP Bucket Synchronization Failures

**Error: Gradient all-reduce hangs or times out**

```
RuntimeError: Detected that not all parameters in the model have been used...
This can happen if not all inputs are used to compute the loss...
```

**DDP buckets explained:**
- DDP groups gradients into "buckets" (~25MB default)
- All-reduce starts when a bucket is ready (before backward() completes)
- Overlaps communication with computation

**Debug bucket issues:**

```python
import torch.distributed as dist

# Enable DDP debugging
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

ddp_model = DDP(
    model,
    device_ids=[local_rank],
    bucket_cap_mb=25,  # Default bucket size
    gradient_as_bucket_view=True,  # More efficient
    broadcast_buffers=True  # Sync batch norm stats
)

# DDP will print detailed bucket info:
# DDP bucket 0: [param0, param1, param2] size=24.5MB
# DDP bucket 1: [param3, param4] size=10.2MB
```

**Adjust bucket size for your model:**

```python
# Smaller buckets = more overlap, but more overhead
ddp_model = DDP(model, bucket_cap_mb=10)  # For many small all-reduces

# Larger buckets = less overhead, but less overlap
ddp_model = DDP(model, bucket_cap_mb=100)  # For fewer large all-reduces

# Disable bucketing entirely (debugging only)
ddp_model = DDP(model, bucket_cap_mb=float('inf'))
```

### 2.3 DDP Hanging During Training

**Symptom: Training stalls, no error message**

**Diagnosis workflow:**

```bash
# 1. Check if all ranks are alive
# On each node:
ps aux | grep python
nvidia-smi

# 2. Attach debugger to hung process
gdb -p <pid>
# (gdb) thread apply all bt  # Backtrace of all threads

# 3. Enable NCCL watchdog
export NCCL_BLOCKING_WAIT=1  # NCCL will timeout instead of hanging
export NCCL_ASYNC_ERROR_HANDLING=1  # Async error reporting

# 4. Use torch.distributed debugging
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_CPP_LOG_LEVEL=INFO
```

**Common causes:**

```python
# 1. Deadlock from mismatched collectives
# Rank 0:
dist.all_reduce(tensor)  # Waiting for rank 1
# Rank 1:
dist.broadcast(tensor, src=0)  # Waiting for rank 0
# DEADLOCK!

# Solution: Ensure all ranks call same collective
if dist.get_rank() == 0:
    dist.broadcast(tensor, src=0)
else:
    dist.broadcast(tensor, src=0)  # Same call on all ranks
```

```python
# 2. DataLoader workers mismatch
# WRONG: Different num_workers on different ranks
if dist.get_rank() == 0:
    dataloader = DataLoader(dataset, num_workers=4)  # 4 workers
else:
    dataloader = DataLoader(dataset, num_workers=8)  # 8 workers - MISMATCH!

# CORRECT: Same configuration on all ranks
dataloader = DataLoader(dataset, num_workers=4)
```

---

## Section 3: NCCL Timeout & Communication Errors (~100 lines)

### 3.1 NCCL Watchdog Timeout

**Error: `Watchdog caught collective operation timeout`**

```
torch.distributed.DistBackendError: Watchdog caught collective operation timeout:
WorkNCCL(OpType=ALLREDUCE, Timeout(ms)=1800000) ran for 1800123 milliseconds before timing out.
```

From [Medium - NCCL Timeout Debugging](https://medium.com/@devaru.ai/debugging-nccl-errors-in-distributed-training-a-comprehensive-guide-28df87512a34) (accessed 2025-11-13):
> "This error occurs when a collective operation (e.g., ALLREDUCE) exceeds its allowed timeout limit during distributed training."

**Root causes:**
- Large tensor sizes (e.g., 10GB+ model weights)
- Network bandwidth saturation
- Slow GPUs causing imbalance
- NCCL blocked on failed rank

**Solutions:**

```bash
# 1. Increase NCCL timeout
export NCCL_TIMEOUT=7200  # 2 hours for very large models

# 2. Enable blocking wait (better error messages)
export NCCL_BLOCKING_WAIT=1

# 3. Increase IB timeout (InfiniBand networks)
export NCCL_IB_TIMEOUT=23  # Default is 18 (18 * 50ms = 900ms)
# Higher values: 23 (1150ms), 30 (1500ms)

# 4. Adjust NCCL buffer sizes
export NCCL_BUFFSIZE=8388608  # 8MB (default 4MB)
```

**Python-side timeout configuration:**

```python
import torch.distributed as dist
from datetime import timedelta

# Set timeout during init_process_group
dist.init_process_group(
    backend='nccl',
    timeout=timedelta(minutes=60)  # 1 hour timeout
)

# For very large models (GPT-3 scale):
dist.init_process_group(
    backend='nccl',
    timeout=timedelta(hours=2)  # 2 hour timeout
)
```

**Optimize data distribution to reduce timeout risk:**

```python
from torch.utils.data import DistributedSampler, DataLoader

# Ensure even data distribution
sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
    drop_last=True  # Drop incomplete batches
)

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=sampler,
    num_workers=4,
    pin_memory=True
)

# Verify even distribution
print(f"Rank {rank}: {len(dataloader)} batches")
# All ranks should print same number!
```

### 3.2 NCCL Communicator Aborted

**Error: `NCCL communicator was aborted on rank X`**

```
torch.distributed.DistBackendError: NCCL communicator was aborted on rank 2.
Original reason for failure was: [Rank 2] Watchdog caught collective operation timeout
```

**What's happening:**
- One rank crashed or hung during collective operation
- NCCL aborted communicator to prevent other ranks from hanging
- Need to identify which rank failed and why

**Diagnosis workflow:**

```bash
# 1. Check logs from ALL ranks (not just rank 0)
# Rank 0 log:
tail -f rank_0.log

# Rank 2 log (the one that aborted):
tail -f rank_2.log
# Look for: OutOfMemoryError, CUDA errors, segfaults

# 2. Enable detailed NCCL logging on all ranks
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL  # Very verbose!

# 3. Monitor GPU health on the faulted rank
nvidia-smi -i 2 -q -d MEMORY,UTILIZATION,TEMPERATURE,POWER
```

**Recovery strategies:**

From [NVIDIA Developer Blog - Fault-Tolerant NCCL](https://developer.nvidia.com/blog/building-scalable-and-fault-tolerant-nccl-applications/) (accessed 2025-11-13):
> "NCCL communicators can be dynamically resized after a fault allowing recovery within the application without fully restarting the workload."

```python
import torch.distributed as dist

# 1. Detect abort and clean up
try:
    dist.all_reduce(tensor)
except dist.DistBackendError as e:
    print(f"NCCL error: {e}")

    # Abort the communicator
    if dist.is_initialized():
        dist.destroy_process_group()

    # Option A: Re-initialize with fewer ranks
    # Remove faulted rank and continue
    healthy_ranks = [0, 1, 3, 4]  # Exclude rank 2
    # ... re-initialize with new world_size=4

    # Option B: Full restart with checkpointing
    # Load latest checkpoint and restart training
```

**NCCL 2.27+ shrink feature:**

```python
# Dynamically remove faulted ranks without full restart
import torch.distributed as dist

# Before NCCL 2.27:
# - Detect fault
# - Abort all communicators
# - Re-initialize from scratch

# NCCL 2.27+:
# - Call ncclCommShrink to remove faulted ranks
# - Continue training with remaining healthy ranks

# Example usage (C++ API):
# ncclCommShrink(oldComm, excludedRanks, numExcluded, &newComm, &config, NCCL_SHRINK_ABORT);
```

### 3.3 NCCL Network Tuning for High Latency

**Symptom: NCCL operations are slow on high-latency networks**

```bash
# Benchmark NCCL performance
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make MPI=1

# Test all-reduce performance across nodes
mpirun -np 8 -H node1:4,node2:4 \
    ./build/all_reduce_perf -b 8 -e 1G -f 2 -g 1

# Expected output:
#       size         time    algbw   busbw
#         8B       50.21us    0.00    0.00
#       256B       51.43us    0.00    0.01
#         1KB      52.89us    0.02    0.02
# ...
#         1GB      15.23ms   67.20  125.25  # GB/s
```

**NCCL tuning for cross-datacenter training:**

```bash
# 1. Use TCP instead of InfiniBand for WAN
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0

# 2. Increase buffer sizes for high bandwidth-delay product
export NCCL_BUFFSIZE=16777216  # 16MB buffers

# 3. Adjust number of channels
export NCCL_NCHANNELS_PER_NET_PEER=4  # More channels = better bandwidth

# 4. Enable NCCL tuning for specific network topology
export NCCL_TUNER_PLUGIN=libnccl-net.so
export NCCL_TUNER_CONFIG_PATH=/path/to/nccl_tuner_config.json
```

**Python-side optimizations:**

```python
# Gradient compression for WAN training
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

ddp_model = DDP(model, device_ids=[local_rank])

# Apply FP16 compression to gradients
ddp_model.register_comm_hook(
    state=None,
    hook=default_hooks.fp16_compress_hook
)

# Or use PowerSGD for better compression (2-10x reduction)
from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook
state = powerSGD_hook.PowerSGDState(
    process_group=None,
    matrix_approximation_rank=2,  # Lower = more compression
    start_powerSGD_iter=10
)
ddp_model.register_comm_hook(state=state, hook=powerSGD_hook.powerSGD_hook)
```

---

## Section 4: Hard Edge Cases & Recovery Strategies (~100 lines)

### 4.1 ECC Errors & GPU Memory Corruption

**Error: `CUDA error: uncorrectable ECC error encountered`**

```
RuntimeError: CUDA error: an uncorrectable ECC error was encountered
Device-side assert triggered
```

**What's happening:**
- GPU DRAM experienced bit flip that ECC couldn't correct
- Often caused by hardware degradation or cosmic rays
- GPU may need RMA

**Diagnosis:**

```bash
# 1. Check ECC error counters
nvidia-smi -i 0 --query-gpu=ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total --format=csv

# Example output:
# ecc.errors.corrected.volatile.total, ecc.errors.uncorrected.volatile.total
# 143, 2  # 2 uncorrectable errors - GPU likely failing!

# 2. Check ECC mode enabled
nvidia-smi -i 0 -q -d ECC

# 3. Reset ECC counters and test
nvidia-smi -i 0 --reset-ecc-errors

# 4. Run GPU memory test
git clone https://github.com/ComputationalRadiationPhysics/cuda_memtest.git
cd cuda_memtest
make
./cuda_memtest

# If errors found: GPU hardware failure
```

**Recovery workflow:**

```python
import torch

def safe_gpu_operation(gpu_id, operation):
    """Wrap operations with ECC error detection"""
    try:
        with torch.cuda.device(gpu_id):
            result = operation()
        return result
    except RuntimeError as e:
        if "ECC error" in str(e):
            print(f"ECC error on GPU {gpu_id}, excluding from training")
            # Remove GPU from training
            return None
        else:
            raise

# Example: Remove failing GPU from multi-GPU training
healthy_gpus = []
for i in range(torch.cuda.device_count()):
    test_tensor = torch.randn(1000, 1000, device=f'cuda:{i}')
    result = safe_gpu_operation(i, lambda: test_tensor @ test_tensor)
    if result is not None:
        healthy_gpus.append(i)

print(f"Healthy GPUs: {healthy_gpus}")
# Continue training with remaining GPUs
```

### 4.2 GPU Thermal Throttling

**Symptom: Training slows down over time, gradients take longer**

```bash
# Monitor GPU temperature and clock throttling
nvidia-smi dmon -i 0 -s pucvmet -c 100

# Example output showing throttling:
# gpu   pwr  gtemp  mtemp     sm    mem    enc    dec   mclk   pclk
#   0    75     82     -      0      0      0      0   9501    300  # Throttled!
#   0   350     78     -     98     45      0      0   9501   1410  # Normal
```

**Causes:**
- Insufficient cooling (datacenter AC failure)
- Blocked air intakes
- Thermal paste degradation
- Prolonged 100% GPU utilization

**Solutions:**

```bash
# 1. Check throttle reasons
nvidia-smi -i 0 -q -d PERFORMANCE

# Look for:
# Clocks Throttle Reasons
#   Idle                        : Not Active
#   Applications Clocks Setting : Not Active
#   SW Power Cap                : Not Active
#   HW Slowdown                 : Active  ← Thermal throttling!
#   Sync Boost                  : Not Active

# 2. Reduce power limit to reduce heat
nvidia-smi -i 0 -pl 300  # Limit to 300W (from 400W)

# 3. Enable persistence mode (reduces GPU init thermal spikes)
nvidia-smi -i 0 -pm 1
```

**Python-side thermal management:**

```python
import torch
import pynvml

pynvml.nvmlInit()

def check_thermal_throttling(gpu_id):
    """Monitor GPU throttling status"""
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

    # Get throttle reasons
    throttle_reasons = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)

    if throttle_reasons & pynvml.nvmlClocksThrottleReasonThermal:
        print(f"GPU {gpu_id} thermally throttled! Temp: {temp}°C")
        return True

    return False

# In training loop:
if check_thermal_throttling(local_rank):
    # Reduce batch size or wait for cooldown
    time.sleep(60)  # Wait 1 minute
```

### 4.3 Checkpoint Corruption & Model Divergence

**Problem: Multi-GPU checkpoint doesn't load correctly**

```python
# Error when loading checkpoint:
RuntimeError: Error(s) in loading state_dict for DistributedDataParallel:
    Missing key(s) in state_dict: "module.layer1.weight", ...
    Unexpected key(s) in state_dict: "layer1.weight", ...
```

**Root cause: `module.` prefix inconsistency**

```python
# Saving with DDP (has "module." prefix):
torch.save(ddp_model.state_dict(), 'checkpoint.pt')
# Keys: ["module.layer1.weight", "module.layer2.weight", ...]

# Loading without DDP (no "module." prefix):
model.load_state_dict(torch.load('checkpoint.pt'))
# ERROR: Keys don't match!

# Solution 1: Save unwrapped model
torch.save(ddp_model.module.state_dict(), 'checkpoint.pt')

# Solution 2: Strip "module." prefix when loading
checkpoint = torch.load('checkpoint.pt')
state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
model.load_state_dict(state_dict)
```

**Robust checkpoint saving/loading:**

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def save_checkpoint(model, optimizer, epoch, path, rank=0):
    """Save checkpoint from rank 0 only"""
    if rank == 0:
        # Unwrap DDP to get clean state dict
        if isinstance(model, DDP):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    # Synchronize all ranks
    if dist.is_initialized():
        dist.barrier()

def load_checkpoint(model, optimizer, path, device):
    """Load checkpoint with flexible key handling"""
    checkpoint = torch.load(path, map_location=device)

    # Handle "module." prefix
    state_dict = checkpoint['model_state_dict']

    # Try loading directly
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        # Strip or add "module." prefix as needed
        if list(state_dict.keys())[0].startswith('module.'):
            # Checkpoint has "module.", model doesn't
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        else:
            # Model has "module.", checkpoint doesn't
            state_dict = {f'module.{k}': v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=True)

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']
```

### 4.4 Driver Crashes & GPU Hangs

**Symptom: GPU becomes unresponsive, nvidia-smi hangs**

```bash
# nvidia-smi command hangs indefinitely
nvidia-smi
# <no output, process stuck>

# Kernel logs show Xid errors
dmesg | grep -i nvidia
# NVRM: Xid (PCI:0000:03:00): 79, GPU has fallen off the bus.
# NVRM: Xid (PCI:0000:03:00): 94, Channel exception
```

**Common Xid error codes:**
- **Xid 13**: Graphics Engine Exception
- **Xid 31**: GPU memory page fault
- **Xid 43**: GPU stopped responding
- **Xid 79**: GPU fallen off the bus (catastrophic)
- **Xid 94**: Contained ECC error

**Recovery workflow:**

```bash
# 1. Try soft reset (often fails with Xid 79)
nvidia-smi -i 0 -r

# 2. Unload and reload driver
sudo rmmod nvidia_uvm
sudo rmmod nvidia_drm
sudo rmmod nvidia_modeset
sudo rmmod nvidia
sudo modprobe nvidia
sudo modprobe nvidia_modeset
sudo modprobe nvidia_drm
sudo modprobe nvidia_uvm

# 3. If still failed, reboot required
sudo reboot

# 4. After reboot, check for hardware issues
nvidia-smi --query-gpu=gpu_name,pci.bus_id,driver_version --format=csv
```

**Prevention strategies:**

```bash
# Enable persistence mode (keeps driver loaded)
sudo nvidia-smi -pm 1

# Set aggressive power management
sudo nvidia-smi -acp 0  # Disable aggressive power save

# Monitor for Xid errors proactively
watch -n 1 'dmesg | grep -i nvidia | tail -20'
```

### 4.5 Production Multi-GPU Debugging Workflow

**Complete diagnostic checklist for production failures:**

```bash
#!/bin/bash
# multi_gpu_diagnostic.sh - Run on all nodes

echo "=== Multi-GPU Health Check ==="
echo "Node: $(hostname)"
echo "Date: $(date)"

# 1. GPU Inventory
echo -e "\n1. GPU Inventory:"
nvidia-smi --query-gpu=index,name,driver_version,pci.bus_id --format=csv

# 2. ECC Errors
echo -e "\n2. ECC Error Check:"
nvidia-smi --query-gpu=index,ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total --format=csv

# 3. Thermal Status
echo -e "\n3. Thermal Status:"
nvidia-smi --query-gpu=index,temperature.gpu,clocks_throttle_reasons.hw_slowdown --format=csv

# 4. P2P Topology
echo -e "\n4. P2P Topology:"
nvidia-smi topo -m

# 5. NCCL Test
echo -e "\n5. NCCL Test (4 GPUs):"
timeout 30 /path/to/nccl-tests/build/all_reduce_perf -b 8 -e 256M -f 2 -g 4 || echo "NCCL test FAILED or TIMEOUT"

# 6. Recent Kernel Errors
echo -e "\n6. Recent GPU Kernel Errors:"
dmesg | grep -i nvidia | tail -10

# 7. Driver/CUDA Versions
echo -e "\n7. Driver/CUDA Versions:"
nvidia-smi | grep -E "Driver Version|CUDA Version"

echo -e "\n=== Diagnostic Complete ==="
```

**Run diagnostics on all nodes:**

```bash
# Execute on all training nodes
parallel-ssh -h nodes.txt -i 'bash /path/to/multi_gpu_diagnostic.sh'

# Compare outputs to identify failing nodes
```

---

## Sources

**Source Documents:**
- None (web research only)

**Web Research:**
- [Medium - Debugging NCCL Errors in Distributed Training](https://medium.com/@devaru.ai/debugging-nccl-errors-in-distributed-training-a-comprehensive-guide-28df87512a34) - NCCL timeout errors, socket timeout, watchdog debugging (accessed 2025-11-13)
- [Massed Compute - Multi-GPU Training Issues](https://massedcompute.com/faq-answers/?question=What%20are%20the%20common%20issues%20encountered%20during%20multi-GPU%20training%20with%20PyTorch%20and%20how%20to%20troubleshoot%20them?) - Common PyTorch multi-GPU problems (accessed 2025-11-13)
- [NVIDIA Developer Blog - Fault-Tolerant NCCL Applications](https://developer.nvidia.com/blog/building-scalable-and-fault-tolerant-nccl-applications/) - ncclCommShrink, dynamic scaling, fault recovery (accessed 2025-11-13)
- [PyTorch Forums - DDP Gradient Sync](https://discuss.pytorch.org/t/ddp-and-gradient-sync/206096) - DDP synchronization failures (accessed 2025-11-13)
- [PyTorch Forums - Multi-GPU Out of Sync](https://discuss.pytorch.org/t/multi-gpu-training-is-out-of-sync/205185) - Gradient synchronization debugging (accessed 2025-11-13)
- [NVIDIA Forums - Multi-GPU P2P](https://forums.developer.nvidia.com/t/multi-gpu-peer-to-peer-access-cuda-sdk-example-not-working-why/35958) - Peer-to-peer access troubleshooting (accessed 2025-11-13)

**Additional References:**
- PyTorch DDP documentation - gradient synchronization internals
- NCCL documentation - communication primitives, error codes
- NVIDIA GPU diagnostic tools - nvidia-smi, nccl-tests
