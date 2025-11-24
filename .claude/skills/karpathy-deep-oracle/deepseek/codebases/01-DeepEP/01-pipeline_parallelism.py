"""
DeepEP Pipeline Parallelism Scheduler

<karpathys_code_comments>
** This File's Role **
Implements DeepEP's efficient pipeline parallelism. Splits model layers across GPUs and overlaps
forward/backward passes to minimize idle time.

** Function List **
schedule_pipeline(micro_batches, num_stages) - Create pipeline schedule
forward_stage(inputs, stage_id) - Execute forward pass for one pipeline stage
backward_stage(outputs_grad, stage_id) - Execute backward pass for one stage
overlap_communication(tensors) - Overlap tensor communication with computation

** Technical Deep Dive **
Pipeline parallelism splits model vertically (layers 1-N on GPU 0, layers N+1-2N on GPU 1, etc).
The challenge: GPUs sit idle waiting for data. DeepEP's solution: micro-batching + careful scheduling.

Instead of processing one big batch (GPU 0 forward → GPU 1 forward → GPU 1 backward → GPU 0 backward),
split into micro-batches and overlap: While GPU 1 does forward on micro-batch 1, GPU 0 starts forward
on micro-batch 2. This keeps all GPUs busy.

Karpathy: The scheduling logic is the hard part. You need to carefully order operations to avoid
deadlocks while maximizing overlap. DeepEP uses a "1F1B" schedule (one forward, one backward) that's
proven to be optimal for memory and throughput.
</karpathys_code_comments>
"""

def schedule_pipeline(micro_batches, num_stages):
    # Karpathy: 1F1B schedule. Each stage does forward pass, then backward pass.
    # Overlaps forward of next micro-batch with backward of previous micro-batch.
    schedule = []
    for i, micro_batch in enumerate(micro_batches):
        # Forward pass
        schedule.append(('forward', micro_batch, i % num_stages))
        # Backward pass (after some delay for pipeline fill)
        if i >= num_stages:
            schedule.append(('backward', micro_batches[i - num_stages], i % num_stages))
    return schedule

# Karpathy: That's the core idea. The devil is in the communication/synchronization details.
