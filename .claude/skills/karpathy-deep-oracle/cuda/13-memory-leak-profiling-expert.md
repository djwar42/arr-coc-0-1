# CUDA Memory Leak Detection & Advanced Profiling: Expert-Level Debugging

## Overview

Memory leaks and persistent allocations are among the hardest GPU debugging challenges. Unlike CPU memory leaks that can be detected with valgrind, GPU memory leaks require specialized tooling that understands CUDA's caching allocator, PyTorch's memory management, and the complex interactions between host and device memory. This document covers expert-level techniques for detecting memory leaks, analyzing fragmentation patterns, tracking persistent allocations, and production monitoring.

**Why Memory Leak Profiling is Critical:**
- **Silent failures**: Leaks accumulate over time, causing OOM days after deployment
- **Caching allocator complexity**: PyTorch reserves more memory than actually allocated
- **Fragmentation issues**: Memory becomes unusable even when technically "free"
- **Production impact**: Small leaks compound in long-running training jobs
- **Multi-GPU complexity**: Leaks can manifest differently across devices

From [Understanding GPU Memory 1: Visualizing All Allocations over Time](https://pytorch.org/blog/understanding-gpu-memory-1/) (PyTorch Blog, accessed 2025-11-13):
> "During your time with PyTorch on GPUs, you may be familiar with this common error message: `torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 512.00 MiB. GPU 0 has a total capacity of 79.32 GiB of which 401.56 MiB is free.` The Memory Snapshot tool provides a fine-grained GPU memory visualization for debugging GPU OOMs."

**Related Knowledge:**
- See [cuda/01-memory-management-unified.md](01-memory-management-unified.md) for memory allocation fundamentals
- See [cuda/09-runtime-errors-debugging-expert.md](09-runtime-errors-debugging-expert.md) for OOM debugging basics
- See [cuda/10-performance-debugging-profiling-expert.md](10-performance-debugging-profiling-expert.md) for performance profiling

---

## Section 1: Memory Leak Detection Tools (~125 lines)

### NVIDIA Compute Sanitizer Memcheck

**Compute Sanitizer** is the successor to cuda-memcheck, providing comprehensive memory error and leak detection for CUDA applications.

From [Efficient CUDA Debugging: Using NVIDIA Compute Sanitizer](https://developer.nvidia.com/blog/efficient-cuda-debugging-using-compute-sanitizer-with-nvtx-and-creating-custom-tools/) (NVIDIA Developer Blog, accessed 2025-11-13):
> "NVIDIA Compute Sanitizer is a powerful tool that can save you time and effort while improving the reliability and performance of your CUDA applications. Compute Sanitizer is a suite of tools that can perform different types of checks on the functional correctness of your code. There are four main tools: `memcheck`: Memory access error and leak detection."

**Basic Memcheck Usage:**
```bash
# Run memcheck on CUDA application
compute-sanitizer --tool memcheck ./my_cuda_program

# Run memcheck on PyTorch script
compute-sanitizer --tool memcheck python train.py

# Enable leak checking specifically
compute-sanitizer --tool memcheck --leak-check full python train.py

# Save report to file
compute-sanitizer --tool memcheck --log-file memcheck.log python train.py
```

**Memcheck Output Example:**
```
========= COMPUTE-SANITIZER
========= Program hit CUDA API error 11 on CUDA API call to cudaMalloc
========= Saved host backtrace up to driver entry point at cudaMalloc time
=========     #0 0x7f3e4c2f4567 in libcuda.so.1
=========     #1 0x5634a8f1c234 in my_allocator
=========     #2 0x5634a8f1c567 in train_loop
=========
========= Invalid __global__ write of size 4 bytes
=========     at 0x70 in kernel
=========     by thread (127,0,0) in block (0,0,0)
=========     Address 0x7f2a98000400 is out of bounds
=========
========= Program leaked 5242880 bytes on CUDA device 0
=========     Saved host backtrace at cudaMalloc time
=========     #0 0x7f3e4c2f4567 in libcuda.so.1
=========     #1 0x5634a8f1c234 in my_allocator
```

**Key Memcheck Features:**
- **Memory access errors**: Out-of-bounds reads/writes, invalid addresses
- **Leak detection**: Identifies allocations never freed
- **Stack traces**: Shows where leaks originated
- **Device-side errors**: Catches errors missed by host-only tools

### PyTorch Memory Profiler with torch.cuda.memory

**PyTorch Memory Snapshot** provides fine-grained visualization of GPU memory allocations over time.

From [Understanding GPU Memory 1](https://pytorch.org/blog/understanding-gpu-memory-1/) (PyTorch Blog, accessed 2025-11-13):
> "The Memory Snapshot tool provides a fine-grained GPU memory visualization for debugging GPU OOMs. Captured memory snapshots will show memory events including allocations, frees and OOMs, along with their stack traces."

**Memory Snapshot Workflow:**

**1. Start recording memory history:**
```python
import torch

# Start recording with buffer for 100,000 memory events
torch.cuda.memory._record_memory_history(max_entries=100000)
```

**2. Run your model:**
```python
# Your training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

**3. Save snapshot to file:**
```python
try:
    torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
except Exception as e:
    print(f"Failed to capture snapshot: {e}")
```

**4. Stop recording:**
```python
torch.cuda.memory._record_memory_history(enabled=None)
```

**Visualizing Snapshots:**

**Option 1: PyTorch Memory Viz (Online)**
- Visit https://pytorch.org/memory_viz
- Drag and drop your `.pickle` file
- Interactive timeline shows allocations over time
- **Privacy Note**: Tool does not save your snapshot

**Option 2: Generate HTML Locally**
```bash
# Generate HTML visualization
python torch/cuda/_memory_viz.py trace_plot memory_snapshot.pickle -o snapshot.html

# Open in browser
open snapshot.html
```

**Snapshot Features:**
- **Color-coded allocations**: Each tensor gets unique color
- **Time axis**: Shows memory events chronologically
- **Stack traces**: Mouse over allocations to see where they came from
- **Peak detection**: Easily identify memory spikes
- **Iteration patterns**: Visualize memory behavior across training iterations

### PyTorch Memory Summary and Statistics

**Memory Summary:**
```python
# Detailed memory breakdown
print(torch.cuda.memory_summary(device=None, abbreviated=False))

# Output shows:
# - Allocated memory by tensors
# - Reserved memory by caching allocator
# - Free memory available
# - Peak memory usage
# - Allocation counts
```

**Memory Statistics:**
```python
# Current memory usage
allocated = torch.cuda.memory_allocated()  # Bytes in active tensors
reserved = torch.cuda.memory_reserved()     # Bytes reserved by allocator
free = torch.cuda.mem_get_info()[0]        # Free GPU memory

print(f"Allocated: {allocated / 1e9:.2f} GB")
print(f"Reserved: {reserved / 1e9:.2f} GB")
print(f"Free: {free / 1e9:.2f} GB")

# Peak memory tracking
torch.cuda.reset_peak_memory_stats()
# ... run model ...
peak = torch.cuda.max_memory_allocated()
print(f"Peak memory: {peak / 1e9:.2f} GB")
```

### NSight Systems Memory Tracking

**NSight Systems** can track CUDA memory allocations and transfers over time.

**Capturing Memory Trace:**
```bash
# Profile with GPU memory tracking
nsys profile --trace=cuda,nvtx --gpu-metrics-device=all \
    --cuda-memory-usage=true python train.py

# Generate report
nsys stats --report cuda_gpu_mem_size report.nsys-rep
```

**NSight Systems Memory View:**
- **Memory timeline**: Visualizes allocations/deallocations
- **Transfer tracking**: Shows H2D and D2H memory copies
- **Bandwidth analysis**: Memory transfer throughput
- **Correlation**: Links memory events to kernel launches

---

## Section 2: PyTorch-Specific Memory Issues (~125 lines)

### Understanding PyTorch Caching Allocator

**The Caching Allocator Problem:**

From [cuda/09-runtime-errors-debugging-expert.md](09-runtime-errors-debugging-expert.md):
> "PyTorch uses a caching allocator that reserves more memory than actively allocated. The error message shows: **Total capacity**: GPU physical memory (10.76 GiB), **Already allocated**: Active PyTorch tensors (4.29 GiB), **Free**: Available for allocation (10.12 MiB) ← actual bottleneck, **Reserved**: Cached by PyTorch allocator (4.46 GiB)"

**Key Insight**: PyTorch's allocator caches freed memory to avoid expensive cudaMalloc calls. This means:
- `torch.cuda.memory_allocated()` shows **active tensor memory**
- `torch.cuda.memory_reserved()` shows **cached + active memory**
- OOM can occur when `free < requested`, even if `reserved - allocated >> requested`

### Common PyTorch Memory Leaks

**1. Retaining computation graphs:**
```python
# BAD: Accumulating losses keeps computation graphs
losses = []
for batch in dataloader:
    loss = model(batch)
    losses.append(loss)  # Keeps entire graph!

# GOOD: Detach values from graph
losses = []
for batch in dataloader:
    loss = model(batch)
    losses.append(loss.item())  # Just the scalar value
```

**2. Not clearing gradients:**
```python
# BAD: Gradients accumulate indefinitely
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
    # Missing: optimizer.zero_grad()

# GOOD: Clear gradients before backward
for batch in dataloader:
    optimizer.zero_grad()  # Clear before backward
    loss = model(batch)
    loss.backward()
    optimizer.step()

# BETTER: Set to None (more memory efficient)
for batch in dataloader:
    optimizer.zero_grad(set_to_none=True)
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

**3. Gradient accumulation leaks:**
```python
# BAD: Not properly normalizing and clearing
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss.backward()  # Gradients keep accumulating
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        # Missing gradient clear!

# GOOD: Normalize and clear properly
accumulation_steps = 4
optimizer.zero_grad()
for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss = loss / accumulation_steps  # Normalize
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

**4. Caching tensors in hooks:**
```python
# BAD: Hook retains tensor references
saved_activations = []
def hook_fn(module, input, output):
    saved_activations.append(output)  # Leak!

model.register_forward_hook(hook_fn)

# GOOD: Detach or clone if needed
saved_activations = []
def hook_fn(module, input, output):
    saved_activations.append(output.detach().clone())

model.register_forward_hook(hook_fn)

# BETTER: Clear when done
saved_activations = []
def hook_fn(module, input, output):
    saved_activations.append(output.detach())

# ... use activations ...
saved_activations.clear()  # Clear when done
```

### Detecting PyTorch Memory Leaks

**Using gc module to find tensor references:**
```python
import gc
import torch

# Find all tensor objects
tensors = [obj for obj in gc.get_objects() if torch.is_tensor(obj)]

# Print tensor details
for obj in tensors:
    if torch.is_tensor(obj):
        print(f"{type(obj)} {obj.size()} {obj.device}")

# Find tensors with gradients
grad_tensors = [obj for obj in gc.get_objects()
                if torch.is_tensor(obj) and obj.requires_grad]
print(f"Tensors with gradients: {len(grad_tensors)}")
```

**Using weak references to track lifetime:**
```python
import weakref
import torch

# Create weak reference to tensor
x = torch.randn(100, 100, device='cuda')
weak_x = weakref.ref(x)

# Tensor is alive
print(f"Alive: {weak_x() is not None}")

# Delete tensor
del x
torch.cuda.empty_cache()

# Check if collected
print(f"Alive: {weak_x() is not None}")  # Should be False
```

### Reference Counting Issues

**Python reference cycles:**
```python
# BAD: Circular reference prevents garbage collection
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.activations = []

    def forward(self, x):
        output = self.layer(x)
        self.activations.append(output)  # Circular ref!
        return output

# GOOD: Break circular references
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        output = self.layer(x)
        return output

# Store activations externally
activations = []
output = model(x)
activations.append(output.detach())
```

**Debugging reference counting:**
```python
import sys
import torch

x = torch.randn(100, 100, device='cuda')
print(f"Reference count: {sys.getrefcount(x)}")

# Add reference
y = x
print(f"Reference count: {sys.getrefcount(x)}")

# Remove reference
del y
print(f"Reference count: {sys.getrefcount(x)}")
```

---

## Section 3: Fragmentation Analysis (~125 lines)

### Understanding Memory Fragmentation

**Memory Fragmentation** occurs when free memory is split into non-contiguous blocks, making it impossible to allocate large contiguous regions even though total free memory is sufficient.

From [PyTorch Forums: Systematically debugging out-of-memory issue](https://discuss.pytorch.org/t/systematically-debugging-out-of-memory-issue/175034) (accessed 2025-11-13):
> "I'm having a recurring out-of-memory issue that seems to be caused by memory fragmentation: `torch.cuda.OutOfMemoryError: CUDA out of memory.`"

**Fragmentation Example:**
```
Memory State (16GB GPU):
┌─────────────────────────────────────┐
│ Used  │ Free │ Used │ Free │ Used  │
│ 4GB   │ 2GB  │ 4GB  │ 2GB  │ 4GB   │
└─────────────────────────────────────┘

Total Free: 4GB (2GB + 2GB)
Largest Block: 2GB
Cannot allocate: 3GB (no contiguous block)
```

### Detecting Fragmentation with torch.cuda.memory_stats()

**Memory Statistics:**
```python
import torch

# Get detailed memory statistics
stats = torch.cuda.memory_stats()

# Key fragmentation metrics
num_alloc_retries = stats.get('num_alloc_retries', 0)
num_ooms = stats.get('num_ooms', 0)
max_split_size = stats.get('max_split_size', 0)

print(f"Allocation retries: {num_alloc_retries}")
print(f"OOM errors: {num_ooms}")
print(f"Max split size: {max_split_size / 1e9:.2f} GB")

# Reserved vs allocated ratio (indicator of fragmentation)
reserved = torch.cuda.memory_reserved()
allocated = torch.cuda.memory_allocated()
fragmentation_ratio = (reserved - allocated) / reserved

print(f"Fragmentation ratio: {fragmentation_ratio:.2%}")
# High ratio (>30%) suggests fragmentation
```

### Visualizing Memory Blocks

**PyTorch Memory Snapshot shows fragmentation:**

From [Understanding GPU Memory 1](https://pytorch.org/blog/understanding-gpu-memory-1/) (PyTorch Blog, accessed 2025-11-13):
> "In this snapshot, there are 3 peaks showing the memory allocations over 3 training iterations. When looking at the peaks, it is easy to see the rise of memory in the forward pass and the fall during the backward pass as the gradients are computed. One thing that stands out is the many tiny spikes in memory, by mousing over them, we see that they are buffers used temporarily by convolution operators."

**Fragmentation Patterns in Snapshot:**
- **Sawtooth pattern**: Many small allocations/deallocations
- **Persistent gaps**: Free memory blocks never reused
- **Increasing baseline**: Reserved memory grows but allocated stays constant
- **Uneven peaks**: Allocation patterns change over iterations

### Mitigating Fragmentation

**1. PyTorch CUDA Allocator Configuration:**

```python
import os

# Enable expandable segments (PyTorch 2.0+)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Set max split size to reduce fragmentation
# Forces allocator to split larger blocks more aggressively
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Combine multiple settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
```

**Expandable Segments Explanation:**

From [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html):
> "The allocator attempts to reduce fragmentation by requesting larger allocations from cudaMalloc, splitting them into blocks, and reusing those blocks."

Expandable segments allow PyTorch's allocator to dynamically grow reserved memory regions, reducing fragmentation by:
- Avoiding fixed-size pools that leave gaps
- Allowing better packing of variable-sized allocations
- Reducing number of cudaMalloc/cudaFree calls

**2. Manual Cache Management:**

```python
# Empty cache to consolidate free blocks
torch.cuda.empty_cache()

# Note: empty_cache() doesn't free allocated memory!
# It only releases reserved but unused memory back to CUDA
```

**When to call empty_cache():**
- After large temporary allocations
- Between training and validation
- When switching between models
- NOT in tight loops (overhead!)

**3. Allocation Strategy Changes:**

```python
# BAD: Many small allocations
for i in range(1000):
    x = torch.randn(10, 10, device='cuda')
    y = model(x)
    del x, y  # Fragments memory

# GOOD: Preallocate and reuse
x = torch.zeros(10, 10, device='cuda')
for i in range(1000):
    x.copy_(torch.randn(10, 10))
    y = model(x)

# BETTER: Batch allocations
batch_size = 100
x = torch.zeros(batch_size, 10, 10, device='cuda')
for i in range(0, 1000, batch_size):
    x.copy_(torch.randn(batch_size, 10, 10))
    y = model(x)
```

**4. Consistent Tensor Shapes:**

```python
# BAD: Variable shapes cause fragmentation
for batch in dataloader:  # Batches have different sizes
    output = model(batch)  # Allocates different sizes each time

# GOOD: Pad to consistent shape
from torch.nn.utils.rnn import pad_sequence

for batch in dataloader:
    padded_batch = pad_sequence(batch, batch_first=True)
    output = model(padded_batch)
```

### Fragmentation Metrics and Thresholds

**Key Metrics:**
```python
def check_fragmentation():
    stats = torch.cuda.memory_stats()
    reserved = torch.cuda.memory_reserved()
    allocated = torch.cuda.memory_allocated()

    # Fragmentation ratio
    frag_ratio = (reserved - allocated) / reserved if reserved > 0 else 0

    # Allocation retries (high = fragmentation)
    retries = stats.get('num_alloc_retries', 0)

    # Number of segments
    num_segments = stats.get('num_alloc_retries', 0)

    print(f"Fragmentation: {frag_ratio:.2%}")
    print(f"Alloc retries: {retries}")

    # Thresholds
    if frag_ratio > 0.3:
        print("⚠️ High fragmentation detected!")
    if retries > 100:
        print("⚠️ Many allocation retries!")

check_fragmentation()
```

**Production Monitoring:**
```python
import logging

class FragmentationMonitor:
    def __init__(self, threshold=0.3):
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)

    def check(self):
        reserved = torch.cuda.memory_reserved()
        allocated = torch.cuda.memory_allocated()
        frag = (reserved - allocated) / reserved if reserved > 0 else 0

        if frag > self.threshold:
            self.logger.warning(
                f"High fragmentation: {frag:.2%} "
                f"(reserved: {reserved/1e9:.2f}GB, "
                f"allocated: {allocated/1e9:.2f}GB)"
            )
            return True
        return False

monitor = FragmentationMonitor()
if monitor.check():
    torch.cuda.empty_cache()
```

---

## Section 4: Production Memory Monitoring (~125 lines)

### Real-Time Memory Tracking

**Memory Monitoring Loop:**
```python
import time
import torch
from dataclasses import dataclass
from typing import List

@dataclass
class MemorySnapshot:
    timestamp: float
    allocated: int
    reserved: int
    free: int
    peak: int

class MemoryTracker:
    def __init__(self, device='cuda:0'):
        self.device = device
        self.snapshots: List[MemorySnapshot] = []
        torch.cuda.reset_peak_memory_stats(device)

    def capture(self):
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            allocated=torch.cuda.memory_allocated(self.device),
            reserved=torch.cuda.memory_reserved(self.device),
            free=torch.cuda.mem_get_info(self.device)[0],
            peak=torch.cuda.max_memory_allocated(self.device)
        )
        self.snapshots.append(snapshot)
        return snapshot

    def report(self):
        if not self.snapshots:
            return "No snapshots captured"

        latest = self.snapshots[-1]
        return (
            f"Memory Usage:\n"
            f"  Allocated: {latest.allocated / 1e9:.2f} GB\n"
            f"  Reserved:  {latest.reserved / 1e9:.2f} GB\n"
            f"  Free:      {latest.free / 1e9:.2f} GB\n"
            f"  Peak:      {latest.peak / 1e9:.2f} GB"
        )

# Usage in training loop
tracker = MemoryTracker()

for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(dataloader):
        outputs = model(batch)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Track every 100 batches
        if batch_idx % 100 == 0:
            tracker.capture()
            print(tracker.report())
```

### Prometheus Metrics Integration

**Exposing memory metrics for Prometheus:**
```python
from prometheus_client import Gauge, start_http_server
import torch

# Define Prometheus metrics
gpu_memory_allocated = Gauge('gpu_memory_allocated_bytes',
                             'GPU memory allocated by PyTorch',
                             ['device'])
gpu_memory_reserved = Gauge('gpu_memory_reserved_bytes',
                            'GPU memory reserved by PyTorch',
                            ['device'])
gpu_memory_free = Gauge('gpu_memory_free_bytes',
                        'Free GPU memory',
                        ['device'])

def update_memory_metrics(device='cuda:0'):
    """Update Prometheus metrics with current GPU memory usage"""
    gpu_memory_allocated.labels(device=device).set(
        torch.cuda.memory_allocated(device)
    )
    gpu_memory_reserved.labels(device=device).set(
        torch.cuda.memory_reserved(device)
    )
    gpu_memory_free.labels(device=device).set(
        torch.cuda.mem_get_info(device)[0]
    )

# Start Prometheus HTTP server
start_http_server(8000)

# Training loop with metrics
for epoch in range(num_epochs):
    for batch in dataloader:
        # Training code
        outputs = model(batch)
        loss.backward()
        optimizer.step()

        # Update metrics periodically
        update_memory_metrics()
```

### Memory Alerts and Emergency Procedures

**Alert System:**
```python
import logging
import torch

class MemoryAlertSystem:
    def __init__(self,
                 warning_threshold_gb=70.0,
                 critical_threshold_gb=75.0,
                 device='cuda:0'):
        self.warning_threshold = warning_threshold_gb * 1e9
        self.critical_threshold = critical_threshold_gb * 1e9
        self.device = device
        self.logger = logging.getLogger(__name__)

    def check_memory(self):
        allocated = torch.cuda.memory_allocated(self.device)
        total = torch.cuda.get_device_properties(self.device).total_memory
        free = torch.cuda.mem_get_info(self.device)[0]

        # Check thresholds
        if allocated > self.critical_threshold:
            self.logger.critical(
                f"CRITICAL: GPU memory at {allocated/1e9:.2f}GB! "
                f"Free: {free/1e9:.2f}GB"
            )
            self.emergency_cleanup()
            return 'critical'

        elif allocated > self.warning_threshold:
            self.logger.warning(
                f"WARNING: GPU memory at {allocated/1e9:.2f}GB. "
                f"Free: {free/1e9:.2f}GB"
            )
            return 'warning'

        return 'ok'

    def emergency_cleanup(self):
        """Emergency memory cleanup procedures"""
        self.logger.info("Running emergency memory cleanup...")

        # 1. Empty PyTorch cache
        torch.cuda.empty_cache()

        # 2. Force garbage collection
        import gc
        gc.collect()

        # 3. Log final state
        allocated = torch.cuda.memory_allocated(self.device)
        free = torch.cuda.mem_get_info(self.device)[0]
        self.logger.info(
            f"After cleanup: Allocated {allocated/1e9:.2f}GB, "
            f"Free {free/1e9:.2f}GB"
        )

# Usage
alert_system = MemoryAlertSystem()

for batch in dataloader:
    # Check memory before batch
    status = alert_system.check_memory()

    if status == 'critical':
        # Skip batch or reduce batch size
        continue

    # Normal training
    outputs = model(batch)
    loss.backward()
    optimizer.step()
```

### Production Memory Debugging Workflow

**Complete Production Debugging Script:**
```python
#!/usr/bin/env python3
"""
Production CUDA Memory Debugging Script
Captures memory snapshots, tracks metrics, and provides alerts
"""

import argparse
import logging
import signal
import sys
import time
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProductionMemoryMonitor:
    def __init__(self,
                 snapshot_interval=300,  # 5 minutes
                 max_snapshots=10):
        self.snapshot_interval = snapshot_interval
        self.max_snapshots = max_snapshots
        self.snapshots = []
        self.running = True

        # Setup signal handlers
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

    def shutdown(self, signum, frame):
        logger.info("Shutdown signal received. Saving final snapshot...")
        self.save_snapshot()
        sys.exit(0)

    def save_snapshot(self):
        """Save current memory snapshot"""
        try:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"memory_snapshot_{timestamp}.pickle"
            torch.cuda.memory._dump_snapshot(filename)
            logger.info(f"Saved snapshot: {filename}")

            # Maintain max snapshot limit
            self.snapshots.append(filename)
            if len(self.snapshots) > self.max_snapshots:
                old_snapshot = self.snapshots.pop(0)
                # Optionally delete old snapshot
                logger.info(f"Oldest snapshot: {old_snapshot}")

        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")

    def check_memory_health(self):
        """Check for memory issues"""
        stats = torch.cuda.memory_stats()
        reserved = torch.cuda.memory_reserved()
        allocated = torch.cuda.memory_allocated()

        # Check fragmentation
        frag_ratio = (reserved - allocated) / reserved if reserved > 0 else 0
        if frag_ratio > 0.3:
            logger.warning(f"High fragmentation: {frag_ratio:.2%}")

        # Check allocation retries
        retries = stats.get('num_alloc_retries', 0)
        if retries > 100:
            logger.warning(f"Many allocation retries: {retries}")

        # Check for OOMs
        num_ooms = stats.get('num_ooms', 0)
        if num_ooms > 0:
            logger.error(f"OOM errors detected: {num_ooms}")
            self.save_snapshot()

    def monitor_loop(self):
        """Main monitoring loop"""
        logger.info("Starting production memory monitoring...")

        # Enable memory history recording
        torch.cuda.memory._record_memory_history(max_entries=100000)

        last_snapshot_time = time.time()

        while self.running:
            # Check memory health
            self.check_memory_health()

            # Periodic snapshots
            current_time = time.time()
            if current_time - last_snapshot_time >= self.snapshot_interval:
                self.save_snapshot()
                last_snapshot_time = current_time

            # Sleep before next check
            time.sleep(10)

def main():
    parser = argparse.ArgumentParser(
        description='Production CUDA Memory Monitoring'
    )
    parser.add_argument('--interval', type=int, default=300,
                       help='Snapshot interval in seconds (default: 300)')
    parser.add_argument('--max-snapshots', type=int, default=10,
                       help='Maximum number of snapshots to keep')

    args = parser.parse_args()

    monitor = ProductionMemoryMonitor(
        snapshot_interval=args.interval,
        max_snapshots=args.max_snapshots
    )

    try:
        monitor.monitor_loop()
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    finally:
        # Save final snapshot
        monitor.save_snapshot()
        # Stop recording
        torch.cuda.memory._record_memory_history(enabled=None)

if __name__ == '__main__':
    main()
```

**Usage:**
```bash
# Run with default settings (5 min intervals, 10 max snapshots)
python production_memory_monitor.py

# Custom intervals
python production_memory_monitor.py --interval 60 --max-snapshots 20

# Run in background
nohup python production_memory_monitor.py > memory_monitor.log 2>&1 &
```

---

## Sources

**NVIDIA Docs:**
- [Efficient CUDA Debugging: Using NVIDIA Compute Sanitizer](https://developer.nvidia.com/blog/efficient-cuda-debugging-using-compute-sanitizer-with-nvtx-and-creating-custom-tools/) (accessed 2025-11-13)
- [Efficient CUDA Debugging: How to Hunt Bugs with NVIDIA Compute Sanitizer](https://developer.nvidia.com/blog/debugging-cuda-more-efficiently-with-nvidia-compute-sanitizer/) (accessed 2025-11-13)
- [NVIDIA Compute Sanitizer User Manual](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)

**PyTorch Docs:**
- [Understanding GPU Memory 1: Visualizing All Allocations over Time](https://pytorch.org/blog/understanding-gpu-memory-1/) (PyTorch Blog, accessed 2025-11-13)
- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html) (accessed 2025-11-13)
- [Understanding CUDA Memory Usage](https://pytorch.org/docs/stable/torch_cuda_memory.html) (accessed 2025-11-13)

**Community:**
- [PyTorch Forums: Systematically debugging out-of-memory issue](https://discuss.pytorch.org/t/systematically-debugging-out-of-memory-issue/175034) (accessed 2025-11-13)
- [PyTorch Forums: Memory Leak Debugging and Common Causes](https://discuss.pytorch.org/t/memory-leak-debugging-and-common-causes/67339) (accessed 2025-11-13)
- [Stack Overflow: How to avoid "CUDA out of memory" in PyTorch](https://stackoverflow.com/questions/59129812/how-to-avoid-cuda-out-of-memory-in-pytorch) (accessed 2025-11-13)

**Additional References:**
- [Medium: Mastering Memory Profiling in PyTorch](https://medium.com/biased-algorithms/mastering-memory-profiling-in-pytorch-40007ced2e46) (accessed 2025-11-13)
- [Medium: Memory Management using PYTORCH_CUDA_ALLOC_CONF](https://iamholumeedey007.medium.com/memory-management-using-pytorch-cuda-alloc-conf-dabe7adec130) (accessed 2025-11-13)
- [arXiv: Reducing GPU Memory Fragmentation](https://arxiv.org/html/2507.16274v1) (accessed 2025-11-13)

**Coverage**: Memory leak detection (compute-sanitizer memcheck, PyTorch memory profiler), persistent allocations (torch.cuda.memory API, memory snapshots), fragmentation analysis (caching allocator, expandable segments, metrics), production monitoring (real-time tracking, Prometheus integration, alert systems, emergency cleanup procedures)
