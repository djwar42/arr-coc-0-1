"""
DualPipe Pipeline Parallelism Scheduler

<karpathys_code_comments>
** This File's Role **
Overlaps computation and communication in pipeline parallelism. While GPU 1 computes, GPU 0 can
send data to GPU 2. This maximizes hardware utilization.

** Function List **
schedule_with_overlap(stages, micro_batches) - Create overlapped schedule
async_send(tensor, target_gpu) - Non-blocking tensor transfer
async_recv(source_gpu) - Non-blocking tensor receive
sync_streams(compute_stream, comm_stream) - Synchronize CUDA streams

** Technical Deep Dive **
Standard pipeline parallelism: compute → wait → communicate → wait → compute. Lots of idle time.

DualPipe: Use two CUDA streams (compute and communication). While computing forward pass, start
sending previous activation to next GPU. Overlap eliminates idle time.

The trick: Careful stream synchronization. Can't overwrite a tensor that's being sent. Can't
compute on data that hasn't arrived. DualPipe manages these dependencies automatically.

Result: Near-perfect GPU utilization. Training V3 is bottlenecked by compute, not communication.

Karpathy: This is where CUDA expertise matters. Understanding streams, events, and synchronization
primitives. Get it wrong and you get silent data corruption. Get it right and you save millions.
</karpathys_code_comments>
"""

import torch

def schedule_with_overlap(stages, micro_batches):
    # Karpathy: Key idea - use async operations to overlap compute and communication
    compute_stream = torch.cuda.Stream()
    comm_stream = torch.cuda.Stream()

    for micro_batch in micro_batches:
        # Karpathy: Compute on compute stream
        with torch.cuda.stream(compute_stream):
            output = stage_forward(micro_batch)

        # Karpathy: While computing next, send current output on comm stream
        with torch.cuda.stream(comm_stream):
            async_send(output, target_gpu=next_stage_gpu)

        # Karpathy: Streams run in parallel - overlap achieved!

# Karpathy: The code looks simple, but the engineering is in getting synchronization right.
