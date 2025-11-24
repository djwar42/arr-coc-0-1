# <claudes_code_comments>
# ** Function List **
# (Module exports only - no functions defined in this file)
#
# ** Technical Review **
# This is the main DeepEP package initialization file that exports the core API.
# Provides three categories of functionality:
#
# 1. **Python utilities** (from .utils):
#    - EventOverlap: CUDA event wrapper for communication-computation overlap
#
# 2. **High-level API** (from .buffer):
#    - Buffer: Main class for expert-parallel dispatch/combine operations
#      Supports both normal mode (NVLink-based intranode) and low-latency mode
#      (RDMA+NVLink hybrid with NVSHMEM+IBGDA for ultra-low-latency inference)
#
# 3. **C++ bindings** (from deep_ep_cpp):
#    - Config: Kernel configuration (num_sms, nvl_chunk_size, rdma_chunk_size, etc.)
#    - topk_idx_t: Type alias for top-k expert indices (typically int16 or int32)
#
# Architecture: DeepEP enables MoE expert parallelism with asymmetric-domain bandwidth
# forwarding. Normal mode achieves ~160 GB/s via NVLink for intranode all-to-all.
# Low-latency mode combines ~50 GB/s RDMA + ~160 GB/s NVLink for multi-node inference
# with sub-microsecond expert dispatch/combine latencies (critical for DeepSeek-V3).
# </claudes_code_comments>

import torch

from .utils import EventOverlap
from .buffer import Buffer

# noinspection PyUnresolvedReferences
from deep_ep_cpp import Config, topk_idx_t
