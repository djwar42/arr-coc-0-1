# <claudes_code_comments>
# ** Function List **
# solve_model(runtime_task, chain_table_type, num_nodes, group_size, ...) - Single data placement optimization task
# solve_loop(runtime_ctx, input_tables, init_timelimit, max_timelimit, ...) - Batch solve multiple configurations
# search_data_placement_plans(chain_table_type, num_nodes, group_size, ...) - Distributed search pipeline builder
# main() - CLI entry point with argument parsing
#
# ** Technical Review **
# Smallpond-based distributed execution framework for running data placement optimizations at scale.
# Wraps DataPlacementModel into parallel compute jobs for multi-configuration batch solving.
#
# Architecture:
# - Built on smallpond: Distributed query execution framework (similar to Spark/Dask)
# - Uses Apache Arrow for zero-copy data interchange between workers
# - Partitions parameter grid across cluster nodes for parallel solving
#
# solve_model() - Worker Function:
# - Runs on individual executor nodes
# - Instantiates DataPlacementModel for given (chain_table_type, num_nodes, group_size)
# - Calls model.run() with auto-relaxation enabled
# - Returns solved instance with incidence matrix (disk→group assignments)
# - Captures timing via runtime_task.add_elapsed_time()
# - Suppresses Pyomo warnings (pyomo_logger.setLevel(WARNING))
#
# solve_loop() - Batch Worker:
# - Processes list of parameter configurations
# - Input: Arrow table with columns [chain_table_type, num_nodes, group_size, min_targets_per_disk]
# - Output: Arrow table with columns [chain_table_type, num_nodes, group_size, disks, groups]
# - Loops through input rows, calls solve_model(), concatenates results
# - disks/groups are list fields: List[uint32] containing incidence matrix pairs
#
# search_data_placement_plans() - Pipeline Builder:
# 1. Parameter Grid Generation:
#    - Cartesian product: num_nodes × group_size (filtered: v >= k)
#    - Pandas DataFrame → Arrow Table conversion
# 2. Logical Plan Construction:
#    - DataSourceNode: Wraps parameter table
#    - DataSetPartitionNode: Splits into N partitions (1 per config)
#    - ArrowComputeNode: Maps solve_loop() over partitions
#      * cpu_limit=solver_threads: Threads per solver instance
#      * process_func=functools.partial(solve_loop, ...): Captures solver config
# 3. Returns LogicalPlan for Driver.run()
#
# Data Flow:
# ```
# Parameter Grid (Pandas)
#   → Arrow Table
#     → DataSourceNode
#       → DataSetPartitionNode (1 partition/config)
#         → ArrowComputeNode (solve_loop per partition)
#           → solve_model (per row)
#             → DataPlacementModel.run()
#               → Pyomo solver (HiGHS/CBC/SCIP)
#   → Consolidated Arrow Table (results)
# ```
#
# main() - CLI Interface:
# Arguments:
# - --chain_table_type: "EC" (Erasure Coding) or "CR" (Chain Replication)
# - --num_nodes: List of node counts to test (e.g., -v 100 150 200)
# - --group_size: Single replication factor (default: 3)
# - --min_targets_per_disk: Min targets per node (default: 1)
# - --solver_threads: Threads per solver instance (default: 32)
# - --init_timelimit: Initial solver timeout (default: 1800s)
# - --max_timelimit: Max solver timeout (default: 10800s)
# - --pyomo_solver: Solver backend (default: "appsi_highs", options: cbc, scip)
#
# Execution Flow:
# 1. Driver parses CLI args
# 2. search_data_placement_plans() builds LogicalPlan
# 3. driver.run(plan) executes on cluster
# 4. Results written to output directory
#
# Solver Configuration:
# - qlinearize=True: Quadratic linearization for HiGHS (no QP support)
# - relax_lb=1, relax_ub=0: Auto-relaxation bounds
# - auto_relax=True: Retry with relaxed constraints if infeasible
# - Timeout progression: init_timelimit → max_timelimit over iterations
#
# Output Schema:
# - chain_table_type: string (EC/CR)
# - num_nodes: uint32 (v)
# - group_size: uint32 (k)
# - disks: List[uint32] (node IDs in incidence matrix)
# - groups: List[uint32] (group IDs in incidence matrix)
#
# Use Cases:
# - Hyperparameter sweep: Test many (v, k) combinations
# - Cluster sizing: Find optimal configs for deployment
# - Failure analysis: Compare EC vs CR recovery characteristics
#
# Example:
# ```bash
# python data_placement_job.py \
#   --chain_table_type EC \
#   --num_nodes 100 150 200 \
#   --group_size 5 \
#   --min_targets_per_disk 10 \
#   --solver_threads 64 \
#   --init_timelimit 3600
# ```
#
# Integration:
# - Requires smallpond cluster setup
# - Output consumed by gen_chain_table.py for chain table generation
# - Timing data captured in runtime_output_abspath logs
# </claudes_code_comments>

# local test
# pytest test/test_plan.py -v -x
# production setup
import functools
import socket
import sys
import os.path
import itertools
import pandas as pd
import pyarrow as arrow
from typing import List, Literal
from loguru import logger
from smallpond.common import pytest_running
from smallpond.logical.dataset import ArrowTableDataSet
from smallpond.logical.node import Context, ConsolidateNode, DataSetPartitionNode, DataSourceNode, ArrowComputeNode, LogicalPlan, SqlEngineNode
from smallpond.execution.driver import Driver
from smallpond.execution.task import RuntimeContext, ArrowComputeTask


def solve_model(runtime_task: ArrowComputeTask,
                chain_table_type, num_nodes, group_size, min_targets_per_disk,
                init_timelimit, max_timelimit,
                pyomo_solver="appsi_highs"):
  import logging
  pyomo_logger = logging.getLogger('pyomo')
  pyomo_logger.setLevel(logging.WARNING)

  try:
    from src.model.data_placement import DataPlacementModel
  except:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from src.model.data_placement import DataPlacementModel

  model = DataPlacementModel(chain_table_type, num_nodes, group_size, min_targets_per_disk=min_targets_per_disk, bibd_only=False, qlinearize=True, relax_lb=1, relax_ub=0)
  runtime_task.add_elapsed_time("build model time")

  instance = model.run(
    pyomo_solver=pyomo_solver,
    threads=runtime_task.cpu_limit,
    init_timelimit=init_timelimit,
    max_timelimit=max_timelimit,
    auto_relax=True,
    output_root=runtime_task.runtime_output_abspath,
    add_elapsed_time=runtime_task.add_elapsed_time)
  return model, instance

def solve_loop(runtime_ctx: RuntimeContext, input_tables: List[arrow.Table],
               init_timelimit, max_timelimit,
               pyomo_solver="appsi_highs") -> arrow.Table:
  runtime_task = runtime_ctx.task
  model_params, = input_tables

  output_table = None
  schema = arrow.schema([
      arrow.field("chain_table_type", arrow.string()),
      arrow.field("num_nodes", arrow.uint32()),
      arrow.field("group_size", arrow.uint32()),
      arrow.field("disks", arrow.list_(arrow.uint32())),
      arrow.field("groups", arrow.list_(arrow.uint32())),
  ])

  for chain_table_type, num_nodes, group_size, min_targets_per_disk in zip(*model_params.to_pydict().values()):
    model, instance = solve_model(runtime_task, chain_table_type, num_nodes, group_size, min_targets_per_disk, init_timelimit, max_timelimit, pyomo_solver)
    incidence_matrix = model.get_incidence_matrix(instance)
    disks, groups = zip(*incidence_matrix.keys())
    sol_table = arrow.Table.from_arrays([[chain_table_type], [num_nodes], [group_size], [disks], [groups]], schema=schema)
    output_table = sol_table if output_table is None else arrow.concat_tables((output_table, sol_table))
  return output_table


def search_data_placement_plans(
    chain_table_type: Literal["EC", "CR"],
    num_nodes: List[int], group_size: List[int], min_targets_per_disk=1,
    init_timelimit=1800, max_timelimit=3600*3,
    solver_threads: int=64,
    pyomo_solver="appsi_highs"):
  params = pd.DataFrame([(chain_table_type, v, k, min_targets_per_disk)
                         for v, k in itertools.product(num_nodes, group_size) if v >= k],
                         columns=["chain_table_type", "num_nodes", "group_size", "min_targets_per_disk"])
  logger.warning(f"params: {params}")

  ctx = Context()
  params_source = DataSourceNode(ctx, ArrowTableDataSet(arrow.Table.from_pandas(params)))
  params_partitions = DataSetPartitionNode(ctx, (params_source,), npartitions=len(params), partition_by_rows=True)

  data_placement_sols = ArrowComputeNode(
    ctx, (params_partitions,),
    process_func=functools.partial(solve_loop, init_timelimit=init_timelimit, max_timelimit=max_timelimit, pyomo_solver=pyomo_solver),
    cpu_limit=solver_threads)
  return LogicalPlan(ctx, data_placement_sols)


def main():
  driver = Driver()
  driver.add_argument("-pyomo", "--pyomo_solver", default="appsi_highs", choices=["appsi_highs", "cbc",  "scip"], help="Solver used by Pyomo")
  driver.add_argument("-type", "--chain_table_type",  type=str, required=True, choices=["EC", "CR"], help="CR - Chain Replication; EC - Erasure Coding")
  driver.add_argument("-v", "--num_nodes", nargs="+", type=int, required=True, help="Number of storage nodes")
  driver.add_argument("-k", "--replication_factor", "--group_size", dest="group_size", type=int, default=3, help="Replication factor or erasure coding group size")
  driver.add_argument("-min_r", "--min_targets_per_disk", type=int, default=1, help="Min number of storage targets on each disk")
  driver.add_argument("-j", "--solver_threads", type=int, default=32, help="Number of solver threads")
  driver.add_argument("-t", "--init_timelimit", type=int, default=1800, help="Initial timeout for solver")
  driver.add_argument("-T", "--max_timelimit", type=int, default=3600*3, help="Max timeout for solver")
  plan = search_data_placement_plans(num_executors=driver.num_executors, **driver.get_arguments())
  driver.run(plan)


if __name__ == "__main__":
  main()
