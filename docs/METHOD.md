# Method

## Energy convention

For real, closed-shell spatial orbitals, LMP2 uses

\[
E_{\mathrm{MP2}}^{\mathrm{corr}}
=-
\sum_{ijab}
\frac{(ia|jb)\left[2(ia|jb)-(ib|ja)\right]}
{\epsilon_a+\epsilon_b-\epsilon_i-\epsilon_j}.
\]

Occupied indices are `i,j`, virtual indices are `a,b`, and the denominator is
strictly positive. The reported correlation energy is therefore normally
negative. The total MP2 energy is the RHF energy plus this correlation
energy. All energies are in hartree.

The oxygen 1s core orbital on each water is frozen. Localization retains all
remaining occupied orbitals and the complete virtual space.

## Laplace transformation

The denominator is approximated by

\[
\frac{1}{x}\approx\sum_{\alpha=1}^{N_t}
w_\alpha e^{-t_\alpha x}.
\]

Canonical orbital-energy half factors are

\[
D_o(t_\alpha)_{II}=e^{+\epsilon_I t_\alpha/2},\qquad
D_v(t_\alpha)_{AA}=e^{-\epsilon_A t_\alpha/2}.
\]

The implementation uses a common energy shift to avoid overflow; the shift
cancels exactly between occupied and virtual factors.

## Localization and assignment

The RHF occupied and virtual spaces are localized independently with the
Pipek--Mezey objective and a CIAH optimizer. The PM objective uses
ANO-preorthogonalized meta-Lowdin atomic populations. Monomer labels are a
separate post-localization analysis using ordinary symmetric-Lowdin atomic
populations summed over the O and two H atoms belonging to each water.

An orbital is assigned to the monomer with the largest summed population. If
the maximum population is below the confidence threshold, the calculation
stops before the expensive energy stage. There is no unassigned energy
bucket.

## Localized cached OVOV representation

Let canonical and localized coefficients be related by

\[
C_o^L=C_o^C U_o,\qquad C_v^L=C_v^C U_v.
\]

The code transforms four-center AO ERIs directly into one unscaled localized
four-index tensor

\[
G^L_{ia,jb}=(ia|jb).
\]

It does not construct a canonical OVOV tensor first. At each Laplace point,
exact canonical-energy scaling is represented in the localized basis by

\[
S_o(t)=U_o^T D_o(t)U_o,\qquad
S_v(t)=U_v^T D_v(t)U_v.
\]

These generally dense propagators are applied to the cached localized tensor.
The resulting scaled direct and exchange values enter

\[
E_\alpha=-w_\alpha\sum_{ijab}
g_{ia,jb}(t_\alpha)
\left[2g_{ia,jb}(t_\alpha)-g_{ib,ja}(t_\alpha)\right].
\]

Elementwise scaling with diagonal localized Fock elements is not used.

## Monomer decomposition

For each ordered excitation `ijab`, the code counts the distinct monomer
labels spanned by the four orbitals. Contributions are accumulated into 1M,
2M, 3M, or 4M buckets. In `full` mode,

\[
E_{\mathrm{corr}}=E_{1\mathrm{M}}+E_{2\mathrm{M}}
+E_{3\mathrm{M}}+E_{4\mathrm{M}}
\]

must hold within the stated floating-point reduction tolerance. In `1m2m`
mode, all terms spanning three or four monomers are intentionally omitted.

## Parallel ownership

The logical localized OVOV tensor has

\[
(n_o n_v)^2
\]

double-precision elements. MPI ranks own disjoint right-pair slices, so the
global storage replication factor is one. The AO-to-MO transformation is also
distributed by left occupied index. During each Laplace point, ranks exchange
bounded pair slices so direct and exchange values meet exactly once. MPI is
not called from the innermost integral-contraction loop. OpenMP and BLAS handle
rank-local contractions.

The principal memory limitation remains the four-index localized OVOV tensor
and one distributed scaled working slice in full mode. No density fitting,
resolution of the identity, Cholesky decomposition, or three-index factor is
used.

## Screening and error controls

Two independent AO shell-quartet Schwarz thresholds are exposed:

- the RHF ERI threshold used during direct Fock construction;
- the post-HF threshold used while building localized OVOV.

The public launcher disables localized energy screens because their bounds
are representation-specific. AO Schwarz screening and Laplace quadrature
error are separate approximations and must be validated separately.

The CSV output records evaluated and screened shell quartets, quadrature
reciprocal error, residual SCF metrics, localization confidence, logical and
rank-local tensor sizes, global storage, timing components, and energy
buckets.

## Required checks for new applications

1. Confirm RHF convergence and finite residual metrics.
2. Require zero low-confidence occupied and virtual orbitals.
3. Compare adjacent quadrature orders.
4. Compare the chosen AO Schwarz threshold against zero or a tighter value.
5. Require zero canonical OVOV bytes for this method.
6. Require localized global storage to equal one logical localized tensor.
7. Require the 1M--4M sum rule in full mode.
8. Preserve the raw CSV, progress log, geometry copy, and metadata.

