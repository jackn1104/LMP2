# HPC use

LMP2 uses one MPI rank per node by default and OpenMP within each rank. This
layout avoids replicated localized OVOV storage and gives each rank the full
node-local memory allowance.

## Generic Slurm allocation

Request CPU nodes according to the site policy, then run inside the allocation:

```bash
./LMP2 geometry.xyz \
  --basis avdz \
  --mode full \
  --quadrature minimax \
  --points 7 \
  --ranks "$SLURM_JOB_NUM_NODES" \
  --threads 128
```

The launcher detects Slurm and creates an `srun` step with one task per node.
It uses `SLURM_CPUS_ON_NODE` for the step's logical CPU allowance when that
variable is available. Override `--threads` with the number of physical cores
available to each rank.

## Cray compiler wrappers

The supplied cache enables MPI/OpenMP and uses BLAS/LAPACK from the Cray
compiler wrappers:

```bash
cmake -S . -B build/hpc -G Ninja \
  -C configs/perlmutter.cmake \
  -DLibint2_DIR=/path/to/libint2/lib/cmake/libint2 \
  -DHDF5_ROOT="$HDF5_DIR" \
  -DHDF5_PREFER_PARALLEL=ON \
  -DBUILD_TESTING=OFF

cmake --build build/hpc --target lmp2_engine -j 8
export LMP2_ENGINE="$PWD/build/hpc/lmp2_engine"
```

Module names and dependency paths are site-specific and intentionally not
embedded in the scientific source.

## Memory

For `n_o` active occupied and `n_v` virtual orbitals, the logical cached
localized tensor contains `(n_o n_v)^2` doubles. MPI ranks own disjoint
right-pair slices, but the AO-to-MO build and full-mode scaled tensor require
additional rank-local workspaces. The terminal prints the smart planner's
usable memory, transform peak, and total peak before the expensive build.

Do not set a memory value larger than a node's physical memory divided by the
number of ranks on that node. The planner reserves 20 percent for the
operating system, MPI, integral engines, RHF, localization, and estimation
uncertainty.

## Reproducibility checklist

Preserve the complete run directory and record:

- source revision;
- compiler and dependency versions;
- node count, MPI ranks, and OpenMP threads;
- basis, quadrature rule/order, and both Schwarz thresholds;
- RHF and localization restart status;
- canonical/localized storage ownership audit;
- low-confidence orbital counts;
- 1M--4M sum error;
- wall-time components and maximum resident memory.

