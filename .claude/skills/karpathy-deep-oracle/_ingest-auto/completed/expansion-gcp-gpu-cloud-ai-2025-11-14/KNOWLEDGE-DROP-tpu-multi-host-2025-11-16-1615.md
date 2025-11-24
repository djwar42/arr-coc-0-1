# KNOWLEDGE DROP: TPU Multi-Host Distributed Training

**Created**: 2025-11-16 16:15
**PART**: 14
**File**: `gcp-gpu/13-tpu-multi-host-distributed.md`
**Lines**: ~700

## What Was Created

Comprehensive guide to TPU multi-host distributed training covering:

1. **TPU Pod Slice Configurations** (v5p-8 through v5p-1024)
2. **JAX Multi-Process Programming** (pjit, GSPMD, mesh sharding)
3. **Data Parallelism Patterns** (process-spanning arrays)
4. **Model Parallelism** (tensor parallelism, FSDP-style sharding)
5. **PyTorch/XLA Multi-Host** (FSDP on TPU, SPMD)
6. **Checkpoint Sharding** (JAX Orbax, PyTorch distributed checkpointing)
7. **Inter-Host Communication** (collectives, network optimization)
8. **Profiling Multi-Host Training**
9. **Fault Tolerance** (checkpoint-resume, elastic training)
10. **Performance Optimization** (gradient accumulation, ZeRO, pipeline parallelism)
11. **Common Pitfalls** (deadlocks, printing distributed arrays)
12. **arr-coc-0-1 Migration Path**

## Key Insights

### Multi-Controller Architecture

**Critical concept**: Each host runs the same Python program as a separate **process** (controller). JAX coordinates these to form one logical cluster.

- **Local devices**: TPU chips attached to this host (via `jax.local_devices()`)
- **Global devices**: All TPU chips across all hosts (via `jax.devices()`)
- **Process-spanning arrays**: Single `jax.Array` sharded across all processes

### TPU Pod Slices

**v5p-128** example:
- 128 TensorCores (64 chips)
- 8 hosts × 16 devices per host
- 64 GB HBM per chip = 4 TB total memory
- Can train 100B+ parameter models with FSDP

### GSPMD (General and Scalable Parallelization)

JAX's compiler-level sharding system:
- Automatic collective communication insertion
- Data + model + pipeline parallelism
- Migrating to **Shardy** (new system as of 2025)

### Critical Rules

1. **Initialization First**: Must call `jax.distributed.initialize()` before accessing devices
2. **Same Code Everywhere**: All processes must run same computation in same order (or deadlocks)
3. **Can't Print Distributed Arrays**: Must replicate first or access local shards only
4. **Network Topology Matters**: ICI (600 GB/s intra-host) >> Ethernet (inter-host)

## Three Ways to Create Process-Spanning Arrays

From JAX docs:

1. **`jax.device_put()`**: Load full array on all processes, then shard (simple but expensive)
2. **`jax.make_array_from_process_local_data()`**: Each process loads its shard (for data loading)
3. **`jax.make_array_from_single_device_arrays()`**: Most control, per-device data

## PyTorch/XLA on TPU

**FSDP on TPU v4-64**:
- 16B parameter GPT-2: 39% hardware utilization
- 128B models on v5p-1024 feasible

**SPMD**: PyTorch equivalent to JAX's GSPMD for sharding

## arr-coc-0-1 Migration Strategy

**Current**: Single A100 GPU, ~5B parameters, batch=32

**TPU v5p-128 Target**:
1. Port JAX code to multi-host (add `jax.distributed.initialize()`)
2. Update data loading for per-process sharding
3. Configure 2D mesh: `(8, 16)` for data + model parallelism
4. Test on v5p-8 (single host) first
5. Scale to v5p-128 (8 hosts)

**Expected**: Can train models 20× larger (100B+ parameters)

## Sources Summary

- **JAX Official**: Multi-process programming, explicit sharding, scaling book
- **Google Cloud**: TPU v5p documentation, performance specs
- **PyTorch**: FSDP on TPU blog, SPMD blog, XLA GitHub
- **Community**: Medium (JAX sharding patterns), GitHub discussions (checkpointing)

All sources accessed 2025-11-16.

## Integration Notes

**Related files**:
- Will reference distributed-training files (DeepSpeed, Megatron-LM, FSDP) when created
- Complements `12-cloud-tpu-architecture-programming.md` (PART 13)
- Builds on GCP GPU multi-node patterns from PART 6

**arr-coc-0-1 connection**:
- Clear migration path from single GPU to TPU pod
- Example configuration provided for v5p-128 training
- 2D mesh setup (data + model parallelism)

**Next steps** (for oracle):
- Update INDEX.md with new file
- Note multi-host patterns for GPU comparison (PART 16)
- Reference in cost analysis (PART 17) for TCO calculation
