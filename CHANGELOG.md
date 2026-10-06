# Changelog

## 0.1.0 - 2026-08-10

- Added the `LMP2-1M2M` command for validated water-cluster XYZ input, automatic
  workstation or Slurm launch, live progress, restart handling, and
  non-overwriting result directories.
- Made the exact four-center `localized_cached_ovov` representation the only
  public energy backend.
- Added full 1M--4M decomposition and explicit truncated 1M+2M mode.
- Added minimax, Gauss--Laguerre, and continuous-fit Laplace quadratures.
- Added MPI-distributed localized OVOV ownership with OpenMP rank-local work.
- Added fail-fast PM/Löwdin assignment and output ownership audits.
- Added focused C++ tests and a W1/aVDZ public-command reference test.
- Added a cyclic water-tetramer geometry as the primary decomposition example.
- Made the terminal energy summary follow the selected full or 1M+2M mode.
- Printed the full orbital-basis name first and added the tetramer aVDZ output.
