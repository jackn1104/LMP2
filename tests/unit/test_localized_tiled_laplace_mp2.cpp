#include "modernqc/integrals/integral_provider.hpp"
#include "modernqc/linalg/matrix.hpp"
#include "modernqc/molecule/molecule.hpp"
#include "modernqc/mp2/canonical_mp2.hpp"
#include "modernqc/mp2/laplace_fit.hpp"
#include "modernqc/mp2/localized_tiled_laplace_mp2.hpp"
#include "modernqc/mp2/nm_decomposition.hpp"
#include "modernqc/mp2/tiled_laplace_mp2.hpp"
#include "modernqc/scf/rhf.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {

[[nodiscard]] modernqc::molecule::Molecule h2_for_localized_lmp2() {
  using modernqc::molecule::Atom;
  return modernqc::molecule::Molecule{
      std::vector<Atom>{
          Atom{.symbol = "H",
               .atomic_number = 1,
               .position_bohr = {0.0, 0.0, -0.7}},
          Atom{.symbol = "H",
               .atomic_number = 1,
               .position_bohr = {0.0, 0.0, 0.7}},
      },
      0,
      1};
}

[[nodiscard]] modernqc::molecule::Molecule h4_for_localized_lmp2() {
  using modernqc::molecule::Atom;
  return modernqc::molecule::Molecule{
      std::vector<Atom>{
          Atom{.symbol = "H",
               .atomic_number = 1,
               .position_bohr = {-3.0, 0.0, -0.7}},
          Atom{.symbol = "H",
               .atomic_number = 1,
               .position_bohr = {-3.0, 0.0, 0.7}},
          Atom{.symbol = "H",
               .atomic_number = 1,
               .position_bohr = {3.0, 0.0, -0.7}},
          Atom{.symbol = "H",
               .atomic_number = 1,
               .position_bohr = {3.0, 0.0, 0.7}},
      },
      0,
      1};
}

[[nodiscard]] modernqc::linalg::Matrix planar_rotation(double angle) {
  modernqc::linalg::Matrix result{2, 2};
  result(0, 0) = std::cos(angle);
  result(1, 0) = std::sin(angle);
  result(0, 1) = -std::sin(angle);
  result(1, 1) = std::cos(angle);
  return result;
}

[[nodiscard]] modernqc::linalg::Matrix paired_planar_rotation(
    double first_angle, double second_angle) {
  modernqc::linalg::Matrix result =
      modernqc::linalg::Matrix::identity(4);
  const auto first = planar_rotation(first_angle);
  const auto second = planar_rotation(second_angle);
  for(std::size_t row = 0; row < 2; ++row) {
    for(std::size_t column = 0; column < 2; ++column) {
      result(row, column) = first(row, column);
      result(row + 2, column + 2) = second(row, column);
    }
  }
  return result;
}

}  // namespace

TEST_CASE("localized tiled Laplace MP2 preserves the full-space energy") {
  const auto molecule = h2_for_localized_lmp2();
  modernqc::integrals::Libint2IntegralProvider provider{
      molecule, "avdz", true};
  const auto rhf = modernqc::scf::run_rhf(
      molecule, provider,
      modernqc::scf::RhfOptions{
          .maximum_iterations = 80,
          .energy_tolerance = 1.0e-11,
          .density_rms_tolerance = 1.0e-9,
          .commutator_rms_tolerance = 1.0e-9,
          .linear_dependency_tolerance = 1.0e-8,
          .diis = true,
          .diis_start = 2,
          .diis_subspace = 8,
          .damping = 0.0,
          .level_shift = 0.0,
      });
  std::vector<std::size_t> virtuals;
  for(std::size_t orbital = 1; orbital < rhf.coefficients.columns();
      ++orbital) {
    virtuals.push_back(orbital);
  }
  const auto fit = modernqc::mp2::build_and_fit_laplace_quadrature(
      rhf.orbital_energies, {0}, virtuals,
      modernqc::mp2::LaplaceFitOptions{
          .histogram_bins = 128,
          .number_of_points = 8,
          .gap_block_size = 32,
          .validation_grid_size = 512,
          .maximum_optimizer_iterations = 6000,
          .threads = 1,
          .mpi_rank = 0,
          .mpi_ranks = 1,
      });
  const auto canonical = modernqc::mp2::compute_tiled_laplace_mp2(
      provider, rhf.coefficients, rhf.orbital_energies, {0}, virtuals,
      fit,
      modernqc::mp2::TiledLaplaceMp2Options{
          .occupied_tile_size = 1,
          .virtual_tile_size = 8,
          .maximum_additional_memory_bytes = 32U * 1024U * 1024U,
          .checkpoint_interval_tiles = 4,
          .threads = 1,
          .mpi_rank = 0,
          .mpi_ranks = 1,
      });
  const modernqc::linalg::Matrix occupied_rotation =
      modernqc::linalg::Matrix::identity(1);
  const modernqc::linalg::Matrix virtual_rotation =
      modernqc::linalg::Matrix::identity(virtuals.size());
  const std::vector<modernqc::mp2::MonomerAssignment> occupied_assignments{
      {.monomer_id = 0, .confidence = 1.0}};
  std::vector<modernqc::mp2::MonomerAssignment> virtual_assignments(
      virtuals.size(), {.monomer_id = 0, .confidence = 1.0});
  const auto localized =
      modernqc::mp2::compute_localized_tiled_laplace_mp2(
          provider, rhf.coefficients, rhf.orbital_energies, {0}, virtuals,
          occupied_rotation, virtual_rotation, occupied_assignments,
          virtual_assignments, fit,
          modernqc::mp2::LocalizedTiledLaplaceMp2Options{
              .occupied_tile_size = 1,
              .virtual_tile_size = 8,
              .maximum_additional_memory_bytes = 32U * 1024U * 1024U,
              .threads = 1,
              .mpi_rank = 0,
              .mpi_ranks = 1,
              .execution_mode = modernqc::mp2::NmExecutionMode::full,
              .backend = modernqc::mp2::LocalizedLaplaceMp2Backend::
                  cached_canonical_ovov,
          });
  CHECK(std::abs(localized.correlation_energy_hartree -
                 canonical.correlation_energy) <= 3.0e-12);
  CHECK(localized.decomposition.energy_hartree[1] ==
        localized.correlation_energy_hartree);
  CHECK(localized.decomposition.energy_hartree[2] == 0.0);
  CHECK(localized.decomposition.energy_hartree[3] == 0.0);
  CHECK(localized.decomposition.energy_hartree[4] == 0.0);
  CHECK(localized.omitted_excitation_terms == 0);
  CHECK(localized.backend == modernqc::mp2::LocalizedLaplaceMp2Backend::
                                 cached_canonical_ovov);
  CHECK(localized.direct_transform_count == 1);
  CHECK(localized.exchange_transform_count == 0);
  CHECK(localized.canonical_ovov_bytes > 0);
  CHECK(localized.canonical_ovov_max_local_bytes ==
        localized.canonical_ovov_bytes);
  CHECK(localized.canonical_ovov_build_seconds >= 0.0);
  CHECK(localized.canonical_ovov_broadcast_seconds == 0.0);
  CHECK(localized.direct_ovov_transform_seconds == 0.0);
  CHECK(localized.localized_rotation_seconds >= 0.0);
  CHECK(localized.pair_transpose_seconds >= 0.0);
  CHECK(localized.energy_accumulation_seconds >= 0.0);
  CHECK(localized.distributed_communication_seconds == 0.0);
  CHECK(localized.computed_excitation_terms ==
        virtuals.size() * virtuals.size());

  const auto exact_canonical = modernqc::mp2::compute_canonical_mp2(
      provider, rhf.coefficients, rhf.orbital_energies, {0}, virtuals);
  const auto laguerre = modernqc::mp2::build_and_fit_laplace_quadrature(
      rhf.orbital_energies, {0}, virtuals,
      modernqc::mp2::LaplaceFitOptions{
          .histogram_bins = 128,
          .number_of_points = 50,
          .gap_block_size = 32,
          .validation_grid_size = 2048,
          .maximum_optimizer_iterations = 6000,
          .threads = 1,
          .mpi_rank = 0,
          .mpi_ranks = 1,
          .scheme = modernqc::mp2::LaplaceQuadratureScheme::gauss_laguerre,
      });
  const auto laguerre_localized =
      modernqc::mp2::compute_localized_tiled_laplace_mp2(
          provider, rhf.coefficients, rhf.orbital_energies, {0}, virtuals,
          occupied_rotation, virtual_rotation, occupied_assignments,
          virtual_assignments, laguerre,
          modernqc::mp2::LocalizedTiledLaplaceMp2Options{
              .occupied_tile_size = 1,
              .virtual_tile_size = 8,
              .maximum_additional_memory_bytes = 32U * 1024U * 1024U,
              .threads = 1,
              .mpi_rank = 0,
              .mpi_ranks = 1,
              .execution_mode = modernqc::mp2::NmExecutionMode::full,
              .backend = modernqc::mp2::LocalizedLaplaceMp2Backend::
                  cached_canonical_ovov,
          });
  CHECK(std::abs(laguerre_localized.correlation_energy_hartree -
                 exact_canonical.correlation_energy) <= 1.0e-7);

  const auto automatic_small_memory =
      modernqc::mp2::compute_localized_tiled_laplace_mp2(
          provider, rhf.coefficients, rhf.orbital_energies, {0}, virtuals,
          occupied_rotation, virtual_rotation, occupied_assignments,
          virtual_assignments, fit,
          modernqc::mp2::LocalizedTiledLaplaceMp2Options{
              .occupied_tile_size = 1,
              .virtual_tile_size = 1,
              .maximum_additional_memory_bytes = 64U * 1024U,
              .threads = 1,
              .mpi_rank = 0,
              .mpi_ranks = 1,
              .execution_mode = modernqc::mp2::NmExecutionMode::full,
              .backend = modernqc::mp2::LocalizedLaplaceMp2Backend::automatic,
          });
  CHECK(automatic_small_memory.backend ==
        modernqc::mp2::LocalizedLaplaceMp2Backend::direct_tiled);
  CHECK(automatic_small_memory.canonical_ovov_bytes == 0);
  CHECK(std::abs(automatic_small_memory.correlation_energy_hartree -
                 localized.correlation_energy_hartree) <= 3.0e-12);

  for(std::size_t orbital = 0; orbital < virtual_assignments.size();
      ++orbital) {
    virtual_assignments[orbital].monomer_id =
        1 + static_cast<int>(orbital % 2);
  }
  const auto full_partition =
      modernqc::mp2::compute_localized_tiled_laplace_mp2(
          provider, rhf.coefficients, rhf.orbital_energies, {0}, virtuals,
          occupied_rotation, virtual_rotation, occupied_assignments,
          virtual_assignments, fit,
          modernqc::mp2::LocalizedTiledLaplaceMp2Options{
              .occupied_tile_size = 1,
              .virtual_tile_size = 8,
              .maximum_additional_memory_bytes = 32U * 1024U * 1024U,
              .threads = 1,
              .mpi_rank = 0,
              .mpi_ranks = 1,
              .execution_mode = modernqc::mp2::NmExecutionMode::full,
              .backend = modernqc::mp2::LocalizedLaplaceMp2Backend::
                  direct_tiled,
          });
  CHECK(std::abs(full_partition.correlation_energy_hartree -
                 canonical.correlation_energy) <= 3.0e-12);
  CHECK(full_partition.decomposition.energy_hartree[3] != 0.0);

  const auto one_two =
      modernqc::mp2::compute_localized_tiled_laplace_mp2(
          provider, rhf.coefficients, rhf.orbital_energies, {0}, virtuals,
          occupied_rotation, virtual_rotation, occupied_assignments,
          virtual_assignments, fit,
          modernqc::mp2::LocalizedTiledLaplaceMp2Options{
              .occupied_tile_size = 1,
              .virtual_tile_size = 8,
              .maximum_additional_memory_bytes = 32U * 1024U * 1024U,
              .threads = 1,
              .mpi_rank = 0,
              .mpi_ranks = 1,
              .execution_mode =
                  modernqc::mp2::NmExecutionMode::one_two_monomer,
              .backend = modernqc::mp2::LocalizedLaplaceMp2Backend::
                  direct_tiled,
          });
  CHECK(one_two.decomposition.energy_hartree[3] == 0.0);
  CHECK(one_two.decomposition.energy_hartree[4] == 0.0);
  CHECK(one_two.omitted_excitation_terms > 0);
  CHECK(one_two.computed_excitation_terms +
            one_two.omitted_excitation_terms ==
        virtuals.size() * virtuals.size());
  CHECK(one_two.computed_point_tasks <
        full_partition.computed_point_tasks);

  const auto localized_direct_selected =
      modernqc::mp2::compute_localized_tiled_laplace_mp2(
          provider, rhf.coefficients, rhf.orbital_energies, {0}, virtuals,
          occupied_rotation, virtual_rotation, occupied_assignments,
          virtual_assignments, fit,
          modernqc::mp2::LocalizedTiledLaplaceMp2Options{
              .occupied_tile_size = 1,
              .virtual_tile_size = 8,
              .maximum_additional_memory_bytes = 32U * 1024U * 1024U,
              .threads = 1,
              .mpi_rank = 0,
              .mpi_ranks = 1,
              .execution_mode =
                  modernqc::mp2::NmExecutionMode::one_two_monomer,
              .backend = modernqc::mp2::LocalizedLaplaceMp2Backend::
                  localized_direct_selected,
          });
  CHECK(localized_direct_selected.backend ==
        modernqc::mp2::LocalizedLaplaceMp2Backend::
            localized_direct_selected);
  CHECK(std::abs(localized_direct_selected.correlation_energy_hartree -
                 one_two.correlation_energy_hartree) <= 3.0e-12);
  CHECK(std::abs(
            localized_direct_selected.decomposition.energy_hartree[1] -
            one_two.decomposition.energy_hartree[1]) <= 3.0e-12);
  CHECK(std::abs(
            localized_direct_selected.decomposition.energy_hartree[2] -
            one_two.decomposition.energy_hartree[2]) <= 3.0e-12);
  CHECK(localized_direct_selected.decomposition.energy_hartree[3] == 0.0);
  CHECK(localized_direct_selected.decomposition.energy_hartree[4] == 0.0);
  CHECK(localized_direct_selected.computed_excitation_terms ==
        one_two.computed_excitation_terms);
  CHECK(localized_direct_selected.omitted_excitation_terms ==
        one_two.omitted_excitation_terms);
  CHECK(localized_direct_selected.direct_transform_count ==
        fit.nodes.size());
  CHECK(localized_direct_selected.direct_ovov_transform_seconds > 0.0);
  CHECK(localized_direct_selected.canonical_ovov_bytes == 0);
  CHECK(localized_direct_selected.canonical_ovov_max_local_bytes == 0);
  CHECK(localized_direct_selected.canonical_ovov_stored_bytes_global == 0);
  CHECK(localized_direct_selected.distributed_communication_seconds == 0.0);

  const auto batched_localized_direct_selected =
      modernqc::mp2::compute_localized_tiled_laplace_mp2(
          provider, rhf.coefficients, rhf.orbital_energies, {0}, virtuals,
          occupied_rotation, virtual_rotation, occupied_assignments,
          virtual_assignments, fit,
          modernqc::mp2::LocalizedTiledLaplaceMp2Options{
              .occupied_tile_size = 1,
              .virtual_tile_size = 8,
              .quadrature_batch_size = fit.nodes.size(),
              .lambda_ao_block_size = 2,
              .maximum_additional_memory_bytes = 32U * 1024U * 1024U,
              .threads = 1,
              .mpi_rank = 0,
              .mpi_ranks = 1,
              .execution_mode =
                  modernqc::mp2::NmExecutionMode::one_two_monomer,
              .backend = modernqc::mp2::LocalizedLaplaceMp2Backend::
                  localized_direct_selected,
          });
  CHECK(std::abs(
            batched_localized_direct_selected.correlation_energy_hartree -
            localized_direct_selected.correlation_energy_hartree) <=
        3.0e-12);
  CHECK(std::abs(
            batched_localized_direct_selected.decomposition
                    .energy_hartree[1] -
            localized_direct_selected.decomposition.energy_hartree[1]) <=
        3.0e-12);
  CHECK(std::abs(
            batched_localized_direct_selected.decomposition
                    .energy_hartree[2] -
            localized_direct_selected.decomposition.energy_hartree[2]) <=
        3.0e-12);
  CHECK(batched_localized_direct_selected.direct_transform_count == 1);
  CHECK(batched_localized_direct_selected.canonical_ovov_bytes == 0);
  CHECK(localized_direct_selected.evaluated_shell_quartets ==
        batched_localized_direct_selected.evaluated_shell_quartets *
            fit.nodes.size());
  CHECK(localized_direct_selected.schwarz_screened_shell_quartets ==
        batched_localized_direct_selected.schwarz_screened_shell_quartets *
            fit.nodes.size());

  CHECK_THROWS_AS(
      modernqc::mp2::compute_localized_tiled_laplace_mp2(
          provider, rhf.coefficients, rhf.orbital_energies, {0}, virtuals,
          occupied_rotation, virtual_rotation, occupied_assignments,
          virtual_assignments, fit,
          modernqc::mp2::LocalizedTiledLaplaceMp2Options{
              .occupied_tile_size = 1,
              .virtual_tile_size = 8,
              .maximum_additional_memory_bytes = 32U * 1024U * 1024U,
              .threads = 1,
              .mpi_rank = 0,
              .mpi_ranks = 1,
              .execution_mode = modernqc::mp2::NmExecutionMode::full,
              .backend = modernqc::mp2::LocalizedLaplaceMp2Backend::
                  localized_direct_selected,
          }),
      std::invalid_argument);

  const auto automatic_one_two =
      modernqc::mp2::compute_localized_tiled_laplace_mp2(
          provider, rhf.coefficients, rhf.orbital_energies, {0}, virtuals,
          occupied_rotation, virtual_rotation, occupied_assignments,
          virtual_assignments, fit,
          modernqc::mp2::LocalizedTiledLaplaceMp2Options{
              .occupied_tile_size = 1,
              .virtual_tile_size = 8,
              .maximum_additional_memory_bytes = 32U * 1024U * 1024U,
              .threads = 2,
              .mpi_rank = 0,
              .mpi_ranks = 1,
              .execution_mode =
                  modernqc::mp2::NmExecutionMode::one_two_monomer,
              .backend =
                  modernqc::mp2::LocalizedLaplaceMp2Backend::automatic,
          });
  CHECK(std::abs(automatic_one_two.correlation_energy_hartree -
                 one_two.correlation_energy_hartree) <= 3.0e-12);
  CHECK(std::abs(automatic_one_two.decomposition.energy_hartree[1] -
                 one_two.decomposition.energy_hartree[1]) <= 3.0e-12);
  CHECK(std::abs(automatic_one_two.decomposition.energy_hartree[2] -
                 one_two.decomposition.energy_hartree[2]) <= 3.0e-12);
  CHECK(automatic_one_two.decomposition.energy_hartree[3] == 0.0);
  CHECK(automatic_one_two.decomposition.energy_hartree[4] == 0.0);
  CHECK(automatic_one_two.omitted_excitation_terms ==
        one_two.omitted_excitation_terms);
  CHECK(automatic_one_two.backend ==
        modernqc::mp2::LocalizedLaplaceMp2Backend::
            localized_cached_ovov);
  CHECK(automatic_one_two.canonical_ovov_bytes == 0);
  CHECK(automatic_one_two.localized_ovov_bytes > 0);
  CHECK(automatic_one_two.localized_ovov_max_local_bytes ==
        automatic_one_two.localized_ovov_bytes);
  CHECK(automatic_one_two.pair_transpose_seconds == 0.0);
  CHECK(automatic_one_two.distributed_communication_seconds == 0.0);
  CHECK(automatic_one_two.computed_point_tasks > 0);
  CHECK(std::abs(localized_direct_selected.correlation_energy_hartree -
                 automatic_one_two.correlation_energy_hartree) <= 3.0e-12);
  CHECK(localized_direct_selected.canonical_ovov_bytes == 0);
  CHECK(automatic_one_two.localized_ovov_bytes > 0);

  auto fully_screened_options =
      modernqc::mp2::LocalizedTiledLaplaceMp2Options{
          .occupied_tile_size = 1,
          .virtual_tile_size = 8,
          .maximum_additional_memory_bytes = 32U * 1024U * 1024U,
          .threads = 2,
          .mpi_rank = 0,
          .mpi_ranks = 1,
          .localized_energy_schwarz_threshold_hartree = 1.0e100,
          .execution_mode =
              modernqc::mp2::NmExecutionMode::one_two_monomer,
          .backend = modernqc::mp2::LocalizedLaplaceMp2Backend::automatic,
      };
  const auto fully_screened =
      modernqc::mp2::compute_localized_tiled_laplace_mp2(
          provider, rhf.coefficients, rhf.orbital_energies, {0}, virtuals,
          occupied_rotation, virtual_rotation, occupied_assignments,
          virtual_assignments, fit, fully_screened_options);
  CHECK(fully_screened.correlation_energy_hartree == 0.0);
  CHECK(fully_screened.computed_excitation_terms == 0);
  CHECK(fully_screened.localized_screened_excitation_terms ==
        automatic_one_two.computed_excitation_terms);
  CHECK(fully_screened.localized_screened_block_tasks > 0);
  CHECK(fully_screened.omitted_excitation_terms ==
        automatic_one_two.computed_excitation_terms +
            automatic_one_two.omitted_excitation_terms);
  CHECK(fully_screened.localized_screening_omitted_energy_bound_hartree >=
        std::abs(automatic_one_two.correlation_energy_hartree));
  CHECK(fully_screened.localized_screening_seconds >= 0.0);
  CHECK(fully_screened.computed_point_tasks == 0);

  auto point_screened_options =
      modernqc::mp2::LocalizedTiledLaplaceMp2Options{
          .occupied_tile_size = 1,
          .virtual_tile_size = 8,
          .maximum_additional_memory_bytes = 32U * 1024U * 1024U,
          .threads = 2,
          .mpi_rank = 0,
          .mpi_ranks = 1,
          .localized_point_energy_threshold_hartree = 1.0e-8,
          .execution_mode =
              modernqc::mp2::NmExecutionMode::one_two_monomer,
          .backend = modernqc::mp2::LocalizedLaplaceMp2Backend::automatic,
      };
  const auto partially_point_screened =
      modernqc::mp2::compute_localized_tiled_laplace_mp2(
          provider, rhf.coefficients, rhf.orbital_energies, {0}, virtuals,
          occupied_rotation, virtual_rotation, occupied_assignments,
          virtual_assignments, fit, point_screened_options);
  const std::uint64_t point_contribution_count =
      automatic_one_two.computed_excitation_terms * fit.nodes.size();
  CHECK(partially_point_screened.localized_screened_point_contributions > 0);
  CHECK(partially_point_screened.localized_screened_point_contributions <
        point_contribution_count);
  CHECK(std::abs(partially_point_screened.correlation_energy_hartree -
                 automatic_one_two.correlation_energy_hartree) <=
        partially_point_screened
                .localized_point_screening_omitted_energy_bound_hartree +
            2.0e-15);

  fully_screened_options.localized_energy_schwarz_threshold_hartree = -1.0;
  CHECK_THROWS_AS(
      modernqc::mp2::compute_localized_tiled_laplace_mp2(
          provider, rhf.coefficients, rhf.orbital_energies, {0}, virtuals,
          occupied_rotation, virtual_rotation, occupied_assignments,
          virtual_assignments, fit, fully_screened_options),
      std::invalid_argument);
  point_screened_options.localized_point_energy_threshold_hartree = -1.0;
  CHECK_THROWS_AS(
      modernqc::mp2::compute_localized_tiled_laplace_mp2(
          provider, rhf.coefficients, rhf.orbital_energies, {0}, virtuals,
          occupied_rotation, virtual_rotation, occupied_assignments,
          virtual_assignments, fit, point_screened_options),
      std::invalid_argument);
}

TEST_CASE("localized block-pair symmetry preserves ordered MP2 terms") {
  const auto molecule = h4_for_localized_lmp2();
  modernqc::integrals::Libint2IntegralProvider provider{
      molecule, "avdz", true};
  const auto rhf = modernqc::scf::run_rhf(
      molecule, provider,
      modernqc::scf::RhfOptions{
          .maximum_iterations = 80,
          .energy_tolerance = 1.0e-11,
          .density_rms_tolerance = 1.0e-9,
          .commutator_rms_tolerance = 1.0e-9,
          .linear_dependency_tolerance = 1.0e-8,
          .diis = true,
          .diis_start = 2,
          .diis_subspace = 8,
          .damping = 0.0,
          .level_shift = 0.0,
      });
  const std::vector<std::size_t> occupied{0, 1};
  std::vector<std::size_t> virtuals;
  for(std::size_t orbital = occupied.size();
      orbital < rhf.coefficients.columns() && virtuals.size() < 4;
      ++orbital) {
    virtuals.push_back(orbital);
  }
  REQUIRE(virtuals.size() == 4);
  const auto fit = modernqc::mp2::build_and_fit_laplace_quadrature(
      rhf.orbital_energies, occupied, virtuals,
      modernqc::mp2::LaplaceFitOptions{
          .histogram_bins = 96,
          .number_of_points = 6,
          .gap_block_size = 32,
          .validation_grid_size = 384,
          .maximum_optimizer_iterations = 4000,
          .threads = 1,
          .mpi_rank = 0,
          .mpi_ranks = 1,
      });
  const auto canonical = modernqc::mp2::compute_tiled_laplace_mp2(
      provider, rhf.coefficients, rhf.orbital_energies, occupied, virtuals,
      fit,
      modernqc::mp2::TiledLaplaceMp2Options{
          .occupied_tile_size = 1,
          .virtual_tile_size = 1,
          .maximum_additional_memory_bytes = 32U * 1024U * 1024U,
          .checkpoint_interval_tiles = 4,
          .threads = 1,
          .mpi_rank = 0,
          .mpi_ranks = 1,
      });
  const std::vector<modernqc::mp2::MonomerAssignment> occupied_assignments{
      {.monomer_id = 0, .confidence = 1.0},
      {.monomer_id = 1, .confidence = 1.0}};
  const std::vector<modernqc::mp2::MonomerAssignment> virtual_assignments{
      {.monomer_id = 0, .confidence = 1.0},
      {.monomer_id = 1, .confidence = 1.0},
      {.monomer_id = 2, .confidence = 1.0},
      {.monomer_id = 3, .confidence = 1.0}};
  const modernqc::linalg::Matrix occupied_rotation = planar_rotation(0.31);
  const modernqc::linalg::Matrix virtual_rotation =
      paired_planar_rotation(-0.47, 0.22);
  const auto localized =
      modernqc::mp2::compute_localized_tiled_laplace_mp2(
          provider, rhf.coefficients, rhf.orbital_energies, occupied,
          virtuals, occupied_rotation, virtual_rotation, occupied_assignments,
          virtual_assignments, fit,
          modernqc::mp2::LocalizedTiledLaplaceMp2Options{
              .occupied_tile_size = 1,
              .virtual_tile_size = 1,
              .maximum_additional_memory_bytes = 32U * 1024U * 1024U,
              .threads = 1,
              .mpi_rank = 0,
              .mpi_ranks = 1,
              .execution_mode = modernqc::mp2::NmExecutionMode::full,
              .backend = modernqc::mp2::LocalizedLaplaceMp2Backend::
                  direct_tiled,
          });
  const auto cached =
      modernqc::mp2::compute_localized_tiled_laplace_mp2(
          provider, rhf.coefficients, rhf.orbital_energies, occupied,
          virtuals, occupied_rotation, virtual_rotation, occupied_assignments,
          virtual_assignments, fit,
          modernqc::mp2::LocalizedTiledLaplaceMp2Options{
              .occupied_tile_size = 1,
              .virtual_tile_size = 1,
              .maximum_additional_memory_bytes = 32U * 1024U * 1024U,
              .threads = 2,
              .mpi_rank = 0,
              .mpi_ranks = 1,
              .execution_mode = modernqc::mp2::NmExecutionMode::full,
              .backend = modernqc::mp2::LocalizedLaplaceMp2Backend::
                  cached_canonical_ovov,
          });
  const auto selected =
      modernqc::mp2::compute_localized_tiled_laplace_mp2(
          provider, rhf.coefficients, rhf.orbital_energies, occupied,
          virtuals, occupied_rotation, virtual_rotation, occupied_assignments,
          virtual_assignments, fit,
          modernqc::mp2::LocalizedTiledLaplaceMp2Options{
              .occupied_tile_size = 1,
              .virtual_tile_size = 1,
              .maximum_additional_memory_bytes = 32U * 1024U * 1024U,
              .threads = 2,
              .mpi_rank = 0,
              .mpi_ranks = 1,
              .execution_mode =
                  modernqc::mp2::NmExecutionMode::one_two_monomer,
              .backend = modernqc::mp2::LocalizedLaplaceMp2Backend::
                  distributed_selected_ovov,
          });
  const auto localized_cached =
      modernqc::mp2::compute_localized_tiled_laplace_mp2(
          provider, rhf.coefficients, rhf.orbital_energies, occupied,
          virtuals, occupied_rotation, virtual_rotation, occupied_assignments,
          virtual_assignments, fit,
          modernqc::mp2::LocalizedTiledLaplaceMp2Options{
              .occupied_tile_size = 1,
              .virtual_tile_size = 1,
              .maximum_additional_memory_bytes = 32U * 1024U * 1024U,
              .threads = 2,
              .mpi_rank = 0,
              .mpi_ranks = 1,
              .execution_mode =
                  modernqc::mp2::NmExecutionMode::one_two_monomer,
              .backend = modernqc::mp2::LocalizedLaplaceMp2Backend::
                  localized_cached_ovov,
          });
  const auto localized_cached_full =
      modernqc::mp2::compute_localized_tiled_laplace_mp2(
          provider, rhf.coefficients, rhf.orbital_energies, occupied,
          virtuals, occupied_rotation, virtual_rotation,
          occupied_assignments, virtual_assignments, fit,
          modernqc::mp2::LocalizedTiledLaplaceMp2Options{
              .occupied_tile_size = 1,
              .virtual_tile_size = 1,
              .maximum_additional_memory_bytes = 32U * 1024U * 1024U,
              .threads = 2,
              .mpi_rank = 0,
              .mpi_ranks = 1,
              .execution_mode = modernqc::mp2::NmExecutionMode::full,
              .backend = modernqc::mp2::LocalizedLaplaceMp2Backend::
                  localized_cached_ovov,
          });
  CHECK(std::abs(localized.correlation_energy_hartree -
                 canonical.correlation_energy) <= 3.0e-12);
  CHECK(std::abs(cached.correlation_energy_hartree -
                 localized.correlation_energy_hartree) <= 3.0e-12);
  CHECK(std::abs(cached.decomposition.energy_hartree[1] -
                 localized.decomposition.energy_hartree[1]) <= 3.0e-12);
  CHECK(std::abs(selected.correlation_energy_hartree -
                 localized.decomposition.energy_hartree[1] -
                 localized.decomposition.energy_hartree[2]) <= 3.0e-12);
  CHECK(std::abs(selected.decomposition.energy_hartree[1] -
                 localized.decomposition.energy_hartree[1]) <= 3.0e-12);
  CHECK(std::abs(selected.decomposition.energy_hartree[2] -
                 localized.decomposition.energy_hartree[2]) <= 3.0e-12);
  CHECK(std::abs(localized_cached.correlation_energy_hartree -
                 selected.correlation_energy_hartree) <= 3.0e-12);
  CHECK(std::abs(localized_cached.decomposition.energy_hartree[1] -
                 selected.decomposition.energy_hartree[1]) <= 3.0e-12);
  CHECK(std::abs(localized_cached.decomposition.energy_hartree[2] -
                 selected.decomposition.energy_hartree[2]) <= 3.0e-12);
  CHECK(std::abs(localized_cached_full.correlation_energy_hartree -
                 cached.correlation_energy_hartree) <= 3.0e-12);
  for(std::size_t body = 1;
      body <= modernqc::mp2::maximum_mp2_body_order; ++body) {
    CHECK(std::abs(
              localized_cached_full.decomposition.energy_hartree[body] -
              cached.decomposition.energy_hartree[body]) <= 3.0e-12);
  }
  CHECK(cached.decomposition.energy_hartree[3] != 0.0);
  CHECK(cached.decomposition.energy_hartree[4] != 0.0);
  CHECK(localized_cached_full.canonical_ovov_bytes == 0);
  CHECK(localized_cached_full.localized_ovov_bytes > 0);
  CHECK(localized_cached_full.localized_ovov_stored_bytes_global ==
        localized_cached_full.localized_ovov_bytes);
  CHECK(localized_cached.canonical_ovov_bytes == 0);
  CHECK(localized_cached.canonical_ovov_max_local_bytes == 0);
  CHECK(localized_cached.canonical_ovov_stored_bytes_global == 0);
  CHECK(localized_cached.localized_ovov_bytes > 0);
  CHECK(localized_cached.localized_ovov_max_local_bytes ==
        localized_cached.localized_ovov_bytes);
  CHECK(localized_cached.localized_ovov_stored_bytes_global ==
        localized_cached.localized_ovov_bytes);
  CHECK(localized_cached.canonical_ovov_build_seconds == 0.0);
  CHECK(localized_cached.localized_ovov_build_seconds >= 0.0);
  CHECK(selected.pair_transpose_seconds == 0.0);
  CHECK(selected.canonical_ovov_stored_bytes_global ==
        selected.canonical_ovov_bytes);
  CHECK(localized.computed_excitation_terms ==
        occupied.size() * occupied.size() * virtuals.size() *
            virtuals.size());
  CHECK(localized.computed_point_tasks ==
        fit.nodes.size() * (occupied.size() * (occupied.size() + 1) / 2) *
            (virtuals.size() * (virtuals.size() + 1) / 2));
}
