# <claudes_code_comments>
# ** Function List **
# get_nvshmem_host_lib_name() - Find NVSHMEM host library soname in directory
#
# ** Technical Review **
# Build configuration for DeepEP C++/CUDA extension module with PyTorch integration.
# Handles architecture-specific compilation (SM80/Ampere vs SM90/Hopper), NVSHMEM linking,
# and conditional feature enablement for internode/low-latency modes.
#
# Build flow:
# 1. NVSHMEM detection: Check NVSHMEM_DIR env var → fallback to nvidia.nvshmem module →
#    disable internode features if unavailable. Supports both system-installed NVSHMEM
#    and pip-installed nvidia-nvshmem wheels.
# 2. Architecture selection: DISABLE_SM90_FEATURES=1 → SM80 (A100), else SM90 (H800/H100)
# 3. Feature flags: -DDISABLE_NVSHMEM, -DDISABLE_SM90_FEATURES, -DDISABLE_AGGRESSIVE_PTX_INSTRS
# 4. CUDA source compilation: deep_ep.cpp + runtime.cu + layout.cu + intranode.cu +
#    optionally internode.cu + internode_ll.cu (if NVSHMEM available)
# 5. Linking: PyTorch extension → libnvshmem_host.so + libnvshmem_device.a with rpath
#
# Key environment variables:
# - NVSHMEM_DIR: Path to NVSHMEM installation (required for internode/low-latency)
# - DISABLE_SM90_FEATURES: 1 = SM80 only (A100), 0 = SM90 (H800/H100, default)
# - TORCH_CUDA_ARCH_LIST: Target GPU architectures (default: 9.0 for Hopper, 8.0 for Ampere)
# - DISABLE_AGGRESSIVE_PTX_INSTRS: 1 = safe PTX, 0 = aggressive ld.global.nc.L1::no_allocate
# - TOPK_IDX_BITS: 32 or 64 bits for topk_idx dtype (default: 64 for int64)
#
# Optimization features:
# - SM90 features: FP8 support, TMA (Tensor Memory Accelerator), cluster launch methods
# - Aggressive PTX: Uses ld.global.nc.L1::no_allocate for volatile reads on Hopper,
#   achieving significant performance gains but undefined behavior per PTX spec. Tested
#   correct on Hopper due to L1/non-coherent cache unification. Auto-disabled for non-9.0.
# - RDC (Relocatable Device Code): Required for NVSHMEM device linking (-rdc=true)
# - Register pressure control: --register-usage-level=10 for better occupancy
#
# NVSHMEM integration details:
# - Requires device-side library (libnvshmem_device.a) for GPU-initiated RDMA
# - Requires host-side library (libnvshmem_host.so) for initialization/control
# - Uses -dlink for separate device linking phase (NVSHMEM symbols resolution)
# - Wheel compatibility: get_nvshmem_host_lib_name() finds versioned soname (e.g., .so.2)
#
# Conditional compilation matrix:
# | NVSHMEM | SM90 | Features Available |
# |---------|------|--------------------|
# | No      | No   | Intranode only (A100 PCIe) |
# | No      | Yes  | Invalid (SM90 needs NVSHMEM for internode) |
# | Yes     | No   | Intranode only (A100 NVLink) |
# | Yes     | Yes  | Full (intranode + internode + low-latency) |
#
# Build artifacts: deep_ep_cpp.so (Python extension) contains CUDA kernels + PyTorch bindings
# Version: 1.2.1 + git short commit hash for traceability
# </claudes_code_comments>

import os
import subprocess
import setuptools
import importlib

from pathlib import Path
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


# Wheel specific: the wheels only include the soname of the host library `libnvshmem_host.so.X`
def get_nvshmem_host_lib_name(base_dir):
    path = Path(base_dir).joinpath('lib')
    for file in path.rglob('libnvshmem_host.so.*'):
        return file.name
    raise ModuleNotFoundError('libnvshmem_host.so not found')


if __name__ == '__main__':
    disable_nvshmem = False
    nvshmem_dir = os.getenv('NVSHMEM_DIR', None)
    nvshmem_host_lib = 'libnvshmem_host.so'
    if nvshmem_dir is None:
        try:
            nvshmem_dir = importlib.util.find_spec("nvidia.nvshmem").submodule_search_locations[0]
            nvshmem_host_lib = get_nvshmem_host_lib_name(nvshmem_dir)
            import nvidia.nvshmem as nvshmem  # noqa: F401
        except (ModuleNotFoundError, AttributeError, IndexError):
            print(
                'Warning: `NVSHMEM_DIR` is not specified, and the NVSHMEM module is not installed. All internode and low-latency features are disabled\n'
            )
            disable_nvshmem = True
    else:
        disable_nvshmem = False

    if not disable_nvshmem:
        assert os.path.exists(nvshmem_dir), f'The specified NVSHMEM directory does not exist: {nvshmem_dir}'

    cxx_flags = ['-O3', '-Wno-deprecated-declarations', '-Wno-unused-variable', '-Wno-sign-compare', '-Wno-reorder', '-Wno-attributes']
    nvcc_flags = ['-O3', '-Xcompiler', '-O3']
    sources = ['csrc/deep_ep.cpp', 'csrc/kernels/runtime.cu', 'csrc/kernels/layout.cu', 'csrc/kernels/intranode.cu']
    include_dirs = ['csrc/']
    library_dirs = []
    nvcc_dlink = []
    extra_link_args = []

    # NVSHMEM flags
    if disable_nvshmem:
        cxx_flags.append('-DDISABLE_NVSHMEM')
        nvcc_flags.append('-DDISABLE_NVSHMEM')
    else:
        sources.extend(['csrc/kernels/internode.cu', 'csrc/kernels/internode_ll.cu'])
        include_dirs.extend([f'{nvshmem_dir}/include'])
        library_dirs.extend([f'{nvshmem_dir}/lib'])
        nvcc_dlink.extend(['-dlink', f'-L{nvshmem_dir}/lib', '-lnvshmem_device'])
        extra_link_args.extend([f'-l:{nvshmem_host_lib}', '-l:libnvshmem_device.a', f'-Wl,-rpath,{nvshmem_dir}/lib'])

    if int(os.getenv('DISABLE_SM90_FEATURES', 0)):
        # Prefer A100
        os.environ['TORCH_CUDA_ARCH_LIST'] = os.getenv('TORCH_CUDA_ARCH_LIST', '8.0')

        # Disable some SM90 features: FP8, launch methods, and TMA
        cxx_flags.append('-DDISABLE_SM90_FEATURES')
        nvcc_flags.append('-DDISABLE_SM90_FEATURES')

        # Disable internode and low-latency kernels
        assert disable_nvshmem
    else:
        # Prefer H800 series
        os.environ['TORCH_CUDA_ARCH_LIST'] = os.getenv('TORCH_CUDA_ARCH_LIST', '9.0')

        # CUDA 12 flags
        nvcc_flags.extend(['-rdc=true', '--ptxas-options=--register-usage-level=10'])

    # Disable LD/ST tricks, as some CUDA version does not support `.L1::no_allocate`
    if os.environ['TORCH_CUDA_ARCH_LIST'].strip() != '9.0':
        assert int(os.getenv('DISABLE_AGGRESSIVE_PTX_INSTRS', 1)) == 1
        os.environ['DISABLE_AGGRESSIVE_PTX_INSTRS'] = '1'

    # Disable aggressive PTX instructions
    if int(os.getenv('DISABLE_AGGRESSIVE_PTX_INSTRS', '1')):
        cxx_flags.append('-DDISABLE_AGGRESSIVE_PTX_INSTRS')
        nvcc_flags.append('-DDISABLE_AGGRESSIVE_PTX_INSTRS')

    # Bits of `topk_idx.dtype`, choices are 32 and 64
    if "TOPK_IDX_BITS" in os.environ:
        topk_idx_bits = int(os.environ['TOPK_IDX_BITS'])
        cxx_flags.append(f'-DTOPK_IDX_BITS={topk_idx_bits}')
        nvcc_flags.append(f'-DTOPK_IDX_BITS={topk_idx_bits}')

    # Put them together
    extra_compile_args = {
        'cxx': cxx_flags,
        'nvcc': nvcc_flags,
    }
    if len(nvcc_dlink) > 0:
        extra_compile_args['nvcc_dlink'] = nvcc_dlink

    # Summary
    print('Build summary:')
    print(f' > Sources: {sources}')
    print(f' > Includes: {include_dirs}')
    print(f' > Libraries: {library_dirs}')
    print(f' > Compilation flags: {extra_compile_args}')
    print(f' > Link flags: {extra_link_args}')
    print(f' > Arch list: {os.environ["TORCH_CUDA_ARCH_LIST"]}')
    print(f' > NVSHMEM path: {nvshmem_dir}')
    print()

    # noinspection PyBroadException
    try:
        cmd = ['git', 'rev-parse', '--short', 'HEAD']
        revision = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
    except Exception as _:
        revision = ''

    setuptools.setup(name='deep_ep',
                     version='1.2.1' + revision,
                     packages=setuptools.find_packages(include=['deep_ep']),
                     ext_modules=[
                         CUDAExtension(name='deep_ep_cpp',
                                       include_dirs=include_dirs,
                                       library_dirs=library_dirs,
                                       sources=sources,
                                       extra_compile_args=extra_compile_args,
                                       extra_link_args=extra_link_args)
                     ],
                     cmdclass={'build_ext': BuildExtension})
