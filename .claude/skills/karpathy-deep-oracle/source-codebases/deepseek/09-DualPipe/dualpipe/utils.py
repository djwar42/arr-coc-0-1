"""Utility functions for DualPipe - Zero-bubble optimization and tensor operations"""

# <claudes_code_comments>
# ** Function List **
# WeightGradStore.put(func) - Queue weight gradient computation function
# WeightGradStore.flush() - Move queued functions to processing queue
# WeightGradStore.pop() - Execute and remove oldest batch of weight gradient functions
# WeightGradStore.clear() - Reset all queues and cache
# run_backward(tensors, grad_tensors) - Execute backward pass with custom settings
# chunk_tensor(x, chunks, dim) - Split tensor into micro-batches along dimension
# cat_tensor(x, dim) - Concatenate list of tensors along dimension
# scatter(inputs, chunks, dim) - Split inputs into micro-batches
# gather(micro_outputs, dim) - Combine micro-batch outputs into full batch
#
# ** Technical Review **
# This module provides critical utilities for DualPipe's zero-bubble optimization and data handling.
#
# Zero-Bubble Optimization (WeightGradStore):
# The key innovation for reducing pipeline bubbles. During backward pass, weight gradient computation
# (the "W" in F+B+W) is deferred and queued instead of executed immediately. This allows the backward
# pass to complete faster, enabling it to overlap with subsequent forward/backward chunks.
#
# How it works:
# 1. When enable_zb=True in backward pass, WeightGradStore.enabled is set
# 2. PyTorch's autograd hooks detect this and call put() to queue weight grad functions
# 3. flush() moves queued functions to FIFO processing queue
# 4. Later, _weight_chunk() calls pop() to execute deferred weight gradient computations
# 5. This overlaps weight computation with forward/backward of other micro-batches
#
# Why this reduces bubbles:
# Traditional pipeline: Each chunk must complete F+B+W before next chunk
# Zero-bubble: Chunk completes F+B, allowing next chunk to start while W executes later
# Result: W time is "hidden" behind other computation, reducing bubble from (PP-1)(F+B) to (PP/2-1)(F&B+B-3W)
#
# FIFO Queue Semantics:
# Weight gradients must be applied in same order as backward passes to maintain correctness.
# Queue ensures chronological execution matching the original backward pass order.
#
# Micro-batch Utilities (scatter/gather/chunk_tensor/cat_tensor):
# Pipeline parallelism requires splitting full batch into micro-batches for pipelined execution.
# - scatter(): Splits inputs/labels into num_chunks micro-batches along batch dimension
# - gather(): Recombines micro-batch outputs back into full batch
# - chunk_tensor(): Low-level tensor splitting with None handling
# - cat_tensor(): Low-level tensor concatenation with None handling
#
# run_backward():
# Custom backward execution using PyTorch's internal engine. Parameters chosen for pipeline:
# - keep_graph=False: Don't need graph after backward (memory optimization)
# - create_graph=False: Not doing higher-order gradients
# - allow_unreachable=True: Some tensors may not have gradients in pipeline
# - accumulate_grad=True: Gradients accumulate across micro-batches (critical for correctness)
#
# The accumulate_grad=True is essential: each micro-batch's gradients accumulate in the same
# weight tensors, producing the correct full-batch gradient at the end.
# </claudes_code_comments>

import queue
from typing import List, Callable

import torch
from torch.autograd import Variable


class WeightGradStore:

    enabled: bool = False
    cache: List[Callable] = []
    funcs_queue = queue.Queue()

    @classmethod
    def put(cls, func: Callable) -> None:
        cls.cache.append(func)

    @classmethod
    def flush(cls) -> None:
        cls.funcs_queue.put(cls.cache)
        cls.cache = []

    @classmethod
    def pop(cls) -> None:
        assert not cls.funcs_queue.empty(), "Pop empty queue."
        funcs = cls.funcs_queue.get()
        for func in funcs:
            func()

    @classmethod
    def clear(cls) -> None:
        cls.cache = []
        cls.funcs_queue = queue.Queue()


def run_backward(tensors: List[torch.Tensor], grad_tensors: List[torch.Tensor]) -> None:
    kwargs = dict(
        keep_graph=False,
        create_graph=False,
        allow_unreachable=True,
        accumulate_grad=True,
    )
    Variable._execution_engine.run_backward(tensors, grad_tensors, **kwargs)


def chunk_tensor(x, chunks, dim):
    if x is None:
        return [None for _ in range(chunks)]
    return x.tensor_split(chunks, dim=dim)


def cat_tensor(x, dim):
    if (isinstance(x, tuple) or isinstance(x, list)):
        if len(x) == 1:
            return x[0]
        elif x[0] is None:
            assert all(y is None for y in x)
            return None
    return torch.cat(x, dim=dim)


def scatter(inputs, chunks, dim):
    assert isinstance(inputs, (torch.Tensor, tuple, list))
    if isinstance(inputs, torch.Tensor):
        inputs = (inputs,)
    assert all(x is None or isinstance(x, torch.Tensor) for x in inputs)
    inputs = [chunk_tensor(x, chunks, dim) for x in inputs]
    microbatches = [microbatch for microbatch in zip(*inputs)]
    if len(microbatches) == 0:
        microbatches = [() for _ in range(chunks)]
    return microbatches


def gather(micro_outputs, dim):
    assert isinstance(micro_outputs[0], (torch.Tensor, tuple, list))
    if isinstance(micro_outputs[0], torch.Tensor):
        micro_outputs = [(x,) for x in micro_outputs]
    outputs = [x for x in zip(*micro_outputs)]
    outputs = tuple(cat_tensor(x, dim=dim) for x in outputs)
    return outputs
