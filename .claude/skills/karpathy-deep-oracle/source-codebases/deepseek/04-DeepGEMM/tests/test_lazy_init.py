# <claudes_code_comments>
# ** Function List **
# main(local_rank) - Initialize CUDA device for process rank
#
# ** Technical Review **
# Fork-safety validation test for DeepGEMM multiprocessing initialization.
#
# Tests that importing deep_gemm doesn't trigger CUDA initialization before fork.
# Critical requirement: DeepGEMM uses custom _find_cuda_home() in __init__.py that
# avoids PyTorch's CUDA initialization, enabling fork-based multiprocessing.
#
# Test strategy: Import deep_gemm in parent process → fork 8 child processes →
# each child sets CUDA device independently. If deep_gemm incorrectly initialized
# CUDA runtime before fork, this would fail with CUDA errors (forking after CUDA
# init is unsafe due to driver state corruption).
#
# Success criteria: All 8 processes complete without CUDA initialization errors.
# This validates JIT compilation can occur independently in each forked worker.
# </claudes_code_comments>

import torch
import torch.multiprocessing as mp
import deep_gemm


def main(local_rank: int):
    torch.cuda.set_device(local_rank)


if __name__ == '__main__':
    procs = [mp.Process(target=main, args=(i, ), ) for i in range(8)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
