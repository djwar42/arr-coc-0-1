---
sourceFile: "Welcome to CUTLASS - NVIDIA Docs Hub"
exportedBy: "Kortex"
exportDate: "2025-10-29T02:37:52.708Z"
---

# Welcome to CUTLASS - NVIDIA Docs Hub

1da0df53-363b-4418-a42a-81ae0636ce0a

Welcome to CUTLASS - NVIDIA Docs Hub

eeb92504-f1af-48a6-82d6-e2c0992b3ca8

https://docs.nvidia.com/cutlass/index.html

## Skip to main content

## NVIDIA CUTLASS Documentation

## NVIDIA CUTLASS Documentation

## Welcome to CUTLASS

https://docs.nvidia.com#welcome-to-cutlass

What is CUTLASS?

https://docs.nvidia.com#what-is-cutlass

CUTLASS is a collection of CUDA C++ template abstractions and Python domain-specific languages (DSLs) designed to enable high-performance matrix-matrix multiplication (GEMM) and related computations across all levels within CUDA. It incorporates hierarchical decomposition and data movement strategies similar to those used in libraries like cuBLAS and cuDNN.

C++ and Python Integration

https://docs.nvidia.com#c-and-python-integration

The C++ implementation modularizes the computational building blocks or “modular parts” into reusable software components using template classes. The Python DSLs, initially introduced as the CuTe DSL, provide an intuitive interface for rapid kernel development with fine-grained control over hardware behavior.

These abstractions allow developers to specialize and tune computation primitives for different layers of the parallelization hierarchy using configurable custom tiling sizes, data types, and other algorithmic policies through C++ template metaprogramming or dynamic Python APIs.

Extensive Mixed-Precision Data Type Support

https://docs.nvidia.com#extensive-mixed-precision-data-type-support

To support a broad range of applications, CUTLASS offers comprehensive support for mixed-precision computations via both its C++ templates and Python interfaces. Supported data types include:

Floating-point types

: FP64, FP32, TF32, FP16, BF16

Tensor Core-emulated FP32

8-bit floating-point formats

: e5m2 and e4m3

## Block scaled types

: NVIDIA NVFP4 and OCP standard MXFP4, MXFP6, MXFP8

## Narrow integer types

: 4-bit and 8-bit signed/unsigned integers

## Binary types

: 1-bit data types

The Python DSL extends this support by enabling experimentation with optimal data type combinations using a NumPy-style API.

## Performance Across NVIDIA Architectures

https://docs.nvidia.com#performance-across-nvidia-architectures

CUTLASS delivers peak-performance matrix multiplication on NVIDIA Tensor Core architectures, from the

## NVIDIA Ampere Architecture

through the

## NVIDIA Blackwell Architecture

, using both optimized C++ templates and Python-generated kernels.

## The Python DSL is optimized for the

## NVIDIA Blackwell Architecture

and achieves performance within 2% of handwritten C++ implementations, while reducing development time through just-in-time (JIT) compilation and interactive debugging.

## Python support for the

## NVIDIA Hopper Architecture

## NVIDIA Ampere Architecture

will be available as an experimental feature at launch, with improvements planned in future releases.

Beyond GEMM: Convolution Support

https://docs.nvidia.com#beyond-gemm-convolution-support

Beyond GEMM, CUTLASS supports high-performance convolution operations through the

implicit GEMM algorithm

. Implicit GEMM reformulates convolution operations as matrix multiplications (GEMM), enabling CUTLASS to leverage its modular and highly optimized GEMM pipeline. This approach allows CUTLASS to construct efficient convolutions by reusing highly optimized GEMM components.

## On this page

Copyright © 2025, NVIDIA Corporation.

Last updated on Oct 22, 2025.

