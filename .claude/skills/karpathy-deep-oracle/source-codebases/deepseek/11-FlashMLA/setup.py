"""FlashMLA Setup - CUDA Extension Build Configuration"""

# <claudes_code_comments>
# ** Function List **
# is_flag_set(flag) - Checks environment variable boolean flags
# get_features_args() - Returns feature-specific compilation flags
# get_arch_flags() - Generates GPU architecture flags (SM90/SM100)
# get_nvcc_thread_args() - Returns NVCC parallel compilation thread count
#
# ** Technical Review **
# This setup script builds FlashMLA's CUDA kernels as a PyTorch C++/CUDA extension.
#
# Core Build Process:
# 1. Detects NVCC version and validates compatibility (12.9+ for SM100, 12.8+ for SM90)
# 2. Configures architecture-specific compilation flags
# 3. Compiles separate kernel implementations for SM90 (Hopper H800) and SM100 (Blackwell B200)
# 4. Links with CUTLASS library (git submodule) for optimized GEMM operations
# 5. Generates flash_mla.cuda PyBind11 module
#
# GPU Architecture Support:
# - SM90 (Hopper, H800): Dense/sparse decoding, sparse prefill
# - SM100 (Blackwell, B200): All kernels including dense prefill with forward/backward
# - Environment flags: FLASH_MLA_DISABLE_SM90, FLASH_MLA_DISABLE_SM100
#
# Kernel Source Files (9 total):
# - csrc/pybind.cpp: Python bindings
# - csrc/smxx/*: Architecture-agnostic utilities (metadata, combine)
# - csrc/sm90/decode/*: H800 decoding kernels (dense + FP8 sparse)
# - csrc/sm90/prefill/sparse/*: H800 sparse prefill
# - csrc/sm100/decode/sparse_fp8/*: B200 FP8 sparse decoding
# - csrc/sm100/prefill/dense/*: B200 dense MHA forward/backward
# - csrc/sm100/prefill/sparse/*: B200 sparse prefill
#
# Compilation Flags:
# - Fast math, relaxed constexpr, extended lambda for performance
# - Register usage reporting via --ptxas-options
# - O3 optimization, C++17 standard
# - CUTLASS include paths for template metaprogramming GEMM
#
# Dependencies:
# - PyTorch with CUDA support
# - NVCC 12.8+ (12.9+ for SM100)
# - CUTLASS (via git submodule at csrc/cutlass)
# - Build parallelism controlled via NVCC_THREADS env var (default 32)
# </claudes_code_comments>

import os
from pathlib import Path
from datetime import datetime
import subprocess

from setuptools import setup, find_packages

from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    IS_WINDOWS,
    CUDA_HOME
)


def is_flag_set(flag: str) -> bool:
    return os.getenv(flag, "FALSE").lower() in ["true", "1", "y", "yes"]

def get_features_args():
    features_args = []
    if is_flag_set("FLASH_MLA_DISABLE_FP16"):
        features_args.append("-DFLASH_MLA_DISABLE_FP16")
    return features_args

def get_arch_flags():
    # Check NVCC Version
    # NOTE The "CUDA_HOME" here is not necessarily from the `CUDA_HOME` environment variable. For more details, see `torch/utils/cpp_extension.py`
    assert CUDA_HOME is not None, "PyTorch must be compiled with CUDA support"
    nvcc_version = subprocess.check_output(
        [os.path.join(CUDA_HOME, "bin", "nvcc"), '--version'], stderr=subprocess.STDOUT
    ).decode('utf-8')
    nvcc_version_number = nvcc_version.split('release ')[1].split(',')[0].strip()
    major, minor = map(int, nvcc_version_number.split('.'))
    print(f'Compiling using NVCC {major}.{minor}')

    DISABLE_SM100 = is_flag_set("FLASH_MLA_DISABLE_SM100")
    DISABLE_SM90 = is_flag_set("FLASH_MLA_DISABLE_SM90")
    if major < 12 or (major == 12 and minor <= 8):
        assert DISABLE_SM100, "sm100 compilation for Flash MLA requires NVCC 12.9 or higher. Please set FLASH_MLA_DISABLE_SM100=1 to disable sm100 compilation, or update your environment."

    arch_flags = []
    if not DISABLE_SM100:
        arch_flags.extend(["-gencode", "arch=compute_100a,code=sm_100a"])
    if not DISABLE_SM90:
        arch_flags.extend(["-gencode", "arch=compute_90a,code=sm_90a"])
    return arch_flags

def get_nvcc_thread_args():
    nvcc_threads = os.getenv("NVCC_THREADS") or "32"
    return ["--threads", nvcc_threads]

subprocess.run(["git", "submodule", "update", "--init", "csrc/cutlass"])

this_dir = os.path.dirname(os.path.abspath(__file__))

if IS_WINDOWS:
    cxx_args = ["/O2", "/std:c++17", "/DNDEBUG", "/W0"]
else:
    cxx_args = ["-O3", "-std=c++17", "-DNDEBUG", "-Wno-deprecated-declarations"]

ext_modules = []
ext_modules.append(
    CUDAExtension(
        name="flash_mla.cuda",
        sources=[
            "csrc/pybind.cpp",
            "csrc/smxx/get_mla_metadata.cu",
            "csrc/smxx/mla_combine.cu",
            "csrc/sm90/decode/dense/splitkv_mla.cu",
            "csrc/sm90/decode/sparse_fp8/splitkv_mla.cu",
            "csrc/sm90/prefill/sparse/fwd.cu",
            "csrc/sm100/decode/sparse_fp8/splitkv_mla.cu",
            "csrc/sm100/prefill/dense/fmha_cutlass_fwd_sm100.cu",
            "csrc/sm100/prefill/dense/fmha_cutlass_bwd_sm100.cu",
            "csrc/sm100/prefill/sparse/fwd.cu",
        ],
        extra_compile_args={
            "cxx": cxx_args + get_features_args(),
            "nvcc": [
                "-O3",
                "-std=c++17",
                "-DNDEBUG",
                "-D_USE_MATH_DEFINES",
                "-Wno-deprecated-declarations",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--use_fast_math",
                "--ptxas-options=-v,--register-usage-level=10"
            ] + get_features_args() + get_arch_flags() + get_nvcc_thread_args(),
        },
        include_dirs=[
            Path(this_dir) / "csrc",
            Path(this_dir) / "csrc" / "sm90",
            Path(this_dir) / "csrc" / "cutlass" / "include",
            Path(this_dir) / "csrc" / "cutlass" / "tools" / "util" / "include",
        ],
    )
)

try:
    cmd = ['git', 'rev-parse', '--short', 'HEAD']
    rev = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
except Exception as _:
    now = datetime.now()
    date_time_str = now.strftime("%Y-%m-%d-%H-%M-%S")
    rev = '+' + date_time_str


setup(
    name="flash_mla",
    version="1.0.0" + rev,
    packages=find_packages(include=['flash_mla']),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
