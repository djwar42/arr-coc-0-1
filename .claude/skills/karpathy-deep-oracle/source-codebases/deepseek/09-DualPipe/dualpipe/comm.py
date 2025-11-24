"""Point-to-point communication utilities for DualPipe"""

# <claudes_code_comments>
# ** Function List **
# set_p2p_tensor_shapes(shapes) - Configure shapes of tensors for P2P communication
# set_p2p_tensor_dtype(dtype) - Configure data type of tensors for P2P communication
# build_from_tensor_shapes() - Allocate empty tensors matching configured shapes/dtype
# append_irecv(ops, src, group) - Queue async receive operation for tensors
# append_isend(ops, tensors, dst, group) - Queue async send operation for tensors
#
# ** Technical Review **
# This module handles point-to-point (P2P) communication between pipeline ranks in DualPipe.
# Enables efficient async data transfer for activations (forward) and gradients (backward).
#
# Pre-allocation Strategy:
# DualPipe requires knowing tensor shapes before communication to pre-allocate receive buffers.
# Users must call set_p2p_tensor_shapes() and set_p2p_tensor_dtype() before step() execution.
# This is necessary because async receives need to allocate destination buffers before data arrives.
#
# Global Configuration:
# TENSOR_SHAPES and TENSOR_DTYPE are module-level globals storing communication metadata.
# All pipeline stages must use identical shapes/dtype for correct P2P matching.
# Shapes typically correspond to activation tensor shapes after one forward pass.
#
# Batched Communication Pattern:
# Instead of immediate send/recv, operations are queued in ops list (List[dist.P2POp]).
# DualPipe calls append_isend/append_irecv multiple times to build queue, then executes
# all operations together with dist.batch_isend_irecv(). This batching enables:
# 1. Communication-computation overlap (compute while transfers in progress)
# 2. Better utilization of network bandwidth (multiple transfers in flight)
# 3. Reduced synchronization overhead (one barrier instead of many)
#
# append_irecv():
# Creates empty tensors on GPU with requires_grad=True for receiving activations/gradients.
# Queues irecv P2POp for each tensor. Returns tensor list to caller (DualPipe) which stores
# them in input_chunks or output_grad_chunks for later computation use.
#
# append_isend():
# Queues isend P2POp for each tensor in the list. Tensors must already exist (computed).
# Used to send forward activations (output_chunks) or backward gradients (input_grad_chunks).
#
# Rank Mapping:
# get_global_rank() converts process group ranks to global ranks for P2P operations.
# This allows DualPipe to use custom rank mappings while still using PyTorch distributed.
#
# None Handling:
# Supports None tensors in shapes list (for optional/conditional tensors).
# Skips P2P ops for None tensors, allowing flexible activation patterns.
#
# GPU Allocation:
# All tensors allocated on current CUDA device with requires_grad=True.
# Gradients flow automatically through PyTorch autograd for received tensors.
# </claudes_code_comments>

from typing import List, Tuple

import torch
import torch.distributed as dist


TENSOR_SHAPES: List[Tuple[int]] = None
TENSOR_DTYPE: torch.dtype = None


def set_p2p_tensor_shapes(shapes: List[Tuple[int]]):
    global TENSOR_SHAPES
    TENSOR_SHAPES = shapes


def set_p2p_tensor_dtype(dtype: torch.dtype):
    global TENSOR_DTYPE
    TENSOR_DTYPE = dtype


def build_from_tensor_shapes():
    return [torch.empty(s, dtype=TENSOR_DTYPE, device="cuda", requires_grad=True) for s in TENSOR_SHAPES]


def append_irecv(ops: List[dist.P2POp], src: int, group: dist.ProcessGroup) -> List[torch.Tensor]:
    tensors = build_from_tensor_shapes()
    src = dist.distributed_c10d.get_global_rank(group, src)
    for tensor in tensors:
        if tensor is not None:
            ops.append(dist.P2POp(dist.irecv, tensor, src))
    return tensors


def append_isend(ops: List[dist.P2POp], tensors: List[torch.Tensor], dst: int, group: dist.ProcessGroup) -> None:
    dst = dist.distributed_c10d.get_global_rank(group, dst)
    for tensor in tensors:
        if tensor is not None:
            ops.append(dist.P2POp(dist.isend, tensor, dst))
