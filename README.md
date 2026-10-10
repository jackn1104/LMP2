# LMP2-1M2M

Paper: [DOI: 10.1021/acs.jctc.6c01287](https://doi.org/10.1021/acs.jctc.6c01287).

LMP2-1M2M calculates localized Laplace-transform MP2 energies for neutral water
clusters. It performs RHF, localizes the complete active occupied and virtual
spaces with Pipek--Mezey localization, assigns orbitals to water monomers with
symmetric Lowdin populations, builds one distributed four-index OVOV tensor
directly in the localized basis, and reports the 1M, 2M, 3M, and 4M
correlation contributions.

The command is:

```bash
./LMP2-1M2M examples/water_tetramer.xyz
```

The included geometry gives the following seven point minimax result with the
default settings (wall time is machine-dependent):

```text
LMP2-1M2M completed
  Basis:        aug-cc-pVDZ
  RHF:          -304.192978726043748 Eh
  1M:           -0.790063577123332 Eh
  2M:           -0.104506137924530 Eh
  3M:           -0.001868096155842 Eh
  4M:           -0.000016320785323 Eh
  Correlation:  -0.896454131989028 Eh
  Total MP2:    -305.089432858032751 Eh
  Wall time:     10.038 s
  Local OVOV:    0.040 GiB
```

The complete machine-readable record is written to `energy.csv`; the terminal
summary above is only the most important subset. The sanitized complete
example summary is preserved in
[`examples/water_tetramer_aug_cc_pVDZ_output.txt`](examples/water_tetramer_aug_cc_pVDZ_output.txt).

The terminal shows each stage as it runs and prints the final RHF energy,
the localized MP2 correlation energy, the total MP2 energy, and wall time.
The 1M--4M labels tell how many distinct water monomers are spanned by the
four orbital indices of a double excitation. No knowledge of those labels is
needed to run the default complete calculation.

The default is the exact `localized_cached_ovov` representation with a
seven-point finite-interval minimax Laplace rule, an aug-cc-pVDZ basis, full
1M--4M decomposition, and automatic memory/tile planning. No canonical OVOV
tensor, density fitting, RI, Cholesky decomposition, or three-index ERI
representation is used.

## Supported scope

- Neutral, closed-shell water clusters in XYZ format.
- Contiguous `O H H` monomer ordering.
- Spherical aug-cc-pVDZ, aug-cc-pVTZ, and aug-cc-pVQZ bases.
- Frozen oxygen 1s core orbitals.
- Complete active occupied and complete virtual orbital spaces.
- Serial, OpenMP, MPI, and hybrid CPU execution.
- HDF5 RHF and localization restart files.

Input coordinates are interpreted as angstrom. Energies are written in
hartree.

## Build

LMP2-1M2M requires a C++20 compiler, CMake 3.25 or newer, Ninja, BLAS/LAPACK,
Eigen3, Libint2, HDF5, MPI, and OpenMP.

On macOS with Homebrew:

```bash
brew install cmake ninja eigen libint hdf5 open-mpi openblas libomp
cmake --preset release
cmake --build --preset release
```

On Ubuntu or Debian systems with the required development packages available:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake ninja-build \
  libeigen3-dev libint2-dev libhdf5-dev \
  libopenmpi-dev openmpi-bin libopenblas-dev liblapack-dev

cmake --preset release
cmake --build --preset release
```

Verify the launcher without starting a calculation:

```bash
./LMP2-1M2M examples/water_monomer.xyz --dry-run
```

Optionally install the built command under a chosen prefix:

```bash
cmake --install build/release --prefix "$HOME/.local"
export PATH="$HOME/.local/bin:$PATH"
LMP2-1M2M examples/water_monomer.xyz
```

### Container build

If Docker is available, it can supply the compiler and libraries:

```bash
docker build -t lmp2_1m2m .
docker run --rm -v "$PWD:/work" -w /work lmp2_1m2m \
  examples/water_monomer.xyz --output results/container-test
```

## Basic use

Water monomer, default settings:

```bash
./LMP2-1M2M examples/water_monomer.xyz
```

The included cyclic water tetramer, using the default aug-cc-pVDZ basis and
full 1M--4M decomposition:

```bash
./LMP2-1M2M examples/water_tetramer.xyz
```

The same tetramer with aug-cc-pVTZ:

```bash
./LMP2-1M2M examples/water_tetramer.xyz --basis avtz
```

Approximate tetramer 1M+2M truncation:

```bash
./LMP2-1M2M examples/water_tetramer.xyz --mode 1m2m
```

The `1m2m` mode deliberately omits every 3M and 4M excitation contribution.
Use the default `full` mode for the complete localized MP2 correlation
energy. The terminal prints 1M--4M for `full`; it prints only 1M, 2M, and
their sum for `1m2m`. The CSV retains explicit zero-valued 3M and 4M columns
in truncated mode so automated analyses keep a fixed schema.

Every run creates a new output directory containing:

```text
geometry.xyz   exact input geometry copy
energy.csv     energies, timings, memory, screening, and ownership audit
progress.err   live progress and failure diagnostics
metadata.json  method settings, thresholds, input hash, and exit status
rhf.h5         reusable RHF checkpoint for a new calculation
rhf.h5.pm-pyscf-ciah.localization.h5
               reusable occupied/virtual PM localization sidecar
```

Existing nonempty output directories are never overwritten.

## Restart

Reuse a completed RHF checkpoint and its localization sidecar:

```bash
./LMP2-1M2M examples/water_tetramer.xyz \
  --checkpoint results/LMP2-1M2M-PREVIOUS/rhf.h5 \
  --output results/water-dimer-restart
```

The geometry and basis must match. Incompatible checkpoints fail instead of
silently changing the orbital space.

## Parallel execution

On a workstation:

```bash
./LMP2-1M2M examples/water_dimer.xyz --ranks 2 --threads 8
```

Inside a Slurm allocation, the launcher uses one MPI rank per requested node:

```bash
./LMP2-1M2M cluster.xyz --ranks 4 --threads 128
```

The program sets unrelated BLAS thread counts to one and uses OpenMP inside
the LMP2-1M2M kernels. Request physical cores and enough memory for the localized
OVOV tensor. The smart planner leaves a node-local memory safety margin and
fails before energy evaluation if the explicit localized cache cannot fit.

## Numerical settings

The defaults are:

```text
Laplace quadrature             minimax, 7 points
post-HF AO Schwarz threshold   1e-10
RHF ERI Schwarz threshold      1e-12
localized energy screens       disabled
PM/Löwdin assignment gate      0.8
```

The AO Schwarz threshold is an approximation. For a new chemical regime,
compare the target threshold with `--ao-schwarz 0` or a tighter value.
Quadrature convergence is independent of integral screening. Compare six and
seven minimax points before treating a new system class or basis as
converged:

```bash
./LMP2-1M2M cluster.xyz --points 6 --output results/q6
./LMP2-1M2M cluster.xyz --points 7 --output results/q7
```

Do not weaken the `0.8` assignment gate merely to finish a calculation. A
failed gate means at least one localized orbital is insufficiently assigned
under the documented PM/Löwdin convention.

## Reference checks

The included water-monomer aug-cc-pVDZ minimax calculation is used as the
small automated installation check. The tetramer result above exercises all
four nM buckets. A completed full W20/aVDZ seven-point calculation gave:

```text
1M   -3.727615804890495 Eh
2M   -0.763111759675327 Eh
3M   -0.037222989762694 Eh
4M   -0.000553953402291 Eh
sum  -4.528504507730807 Eh
```

The four buckets had zero reported partition error. The result differed from
the corresponding supplied full-correlation reference by
`+1.27338193e-7 Eh`. This single validation does not establish convergence
for every geometry or basis.

Build the focused tests with:

```bash
cmake --preset debug -DLMP2_1M2M_FETCH_TEST_DEPENDENCIES=ON
cmake --build --preset debug
ctest --preset debug
```

For a fast local resource-planner check that skips molecular integral
calculations:

```bash
ctest --preset debug -L fast
```

The full command additionally runs the known W1/aVDZ energy check and the
full-space localized-tensor equivalence test.

## Method and limitations

See [docs/METHOD.md](docs/METHOD.md) for equations, index conventions,
parallel ownership, memory behavior, and validation requirements.
See [VALIDATION.md](VALIDATION.md) for completed checks and untested limits of
this package snapshot.

This is research software. Verify quadrature, screening, SCF convergence,
orbital assignment, and energy sums before using results in scientific work.

## License

BSD 3-Clause. See [LICENSE](LICENSE).
