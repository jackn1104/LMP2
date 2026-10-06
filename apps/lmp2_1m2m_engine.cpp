#include "lmp2_1m2m/integrals/integral_provider.hpp"
#include "lmp2_1m2m/io/rhf_checkpoint.hpp"
#include "lmp2_1m2m/localization/orbital_localization.hpp"
#include "lmp2_1m2m/localization/pipek_mezey.hpp"
#include "lmp2_1m2m/molecule/molecule.hpp"
#include "lmp2_1m2m/molecule/xyz_reader.hpp"
#include "lmp2_1m2m/mp2/laplace_fit.hpp"
#include "lmp2_1m2m/mp2/localized_tiled_laplace_mp2.hpp"
#include "lmp2_1m2m/mp2/nm_decomposition.hpp"
#include "lmp2_1m2m/mp2/resource_planner.hpp"
#include "lmp2_1m2m/scf/rhf.hpp"
#include "lmp2_1m2m/scf/rhf_guess.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <unistd.h>

#ifdef LMP2_1M2M_HAS_MPI
#include <mpi.h>
#endif

namespace {

[[nodiscard]] double seconds_since(
    std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now() - start)
      .count();
}

[[nodiscard]] std::size_t bytes_from_gib(double gib) {
  if(!std::isfinite(gib) || gib <= 0.0) {
    throw std::invalid_argument("memory GiB must be finite and positive");
  }
  const long double bytes =
      static_cast<long double>(gib) * 1024.0L * 1024.0L * 1024.0L;
  if(bytes >= static_cast<long double>(
                  std::numeric_limits<std::size_t>::max())) {
    return std::numeric_limits<std::size_t>::max();
  }
  return static_cast<std::size_t>(bytes);
}

[[nodiscard]] std::optional<std::size_t> optional_positive_size(
    std::string_view text, const char* description) {
  if(text == "auto" || text == "smart") {
    return std::nullopt;
  }
  const std::size_t value =
      static_cast<std::size_t>(std::stoull(std::string{text}));
  if(value == 0) {
    throw std::invalid_argument(std::string{description} +
                                " must be positive or auto");
  }
  return value;
}

[[nodiscard]] std::optional<std::size_t> optional_memory_bytes(
    std::string_view text) {
  if(text == "auto" || text == "smart") {
    return std::nullopt;
  }
  return bytes_from_gib(std::stod(std::string{text}));
}

[[nodiscard]] std::optional<std::size_t> environment_size(
    const char* name) {
  const char* value = std::getenv(name);
  if(value == nullptr || *value == '\0') {
    return std::nullopt;
  }
  try {
    const unsigned long long parsed = std::stoull(value);
    if(parsed == 0 ||
       parsed > static_cast<unsigned long long>(
                    std::numeric_limits<std::size_t>::max())) {
      return std::nullopt;
    }
    return static_cast<std::size_t>(parsed);
  } catch(const std::exception&) {
    return std::nullopt;
  }
}

struct DetectedMemory {
  std::size_t bytes_per_node{0};
  const char* source{"unknown"};
};

[[nodiscard]] DetectedMemory detect_memory_per_node() {
  if(const auto slurm_megabytes = environment_size("SLURM_MEM_PER_NODE")) {
    if(*slurm_megabytes <=
       std::numeric_limits<std::size_t>::max() / (1024U * 1024U)) {
      return DetectedMemory{
          .bytes_per_node = *slurm_megabytes * 1024U * 1024U,
          .source = "SLURM_MEM_PER_NODE",
      };
    }
  }
  if(const auto memory_per_cpu = environment_size("SLURM_MEM_PER_CPU")) {
    if(const auto cpus_per_node = environment_size("SLURM_CPUS_ON_NODE")) {
      if(*memory_per_cpu <=
             std::numeric_limits<std::size_t>::max() / *cpus_per_node &&
         *memory_per_cpu * *cpus_per_node <=
             std::numeric_limits<std::size_t>::max() /
                 (1024U * 1024U)) {
        return DetectedMemory{
            .bytes_per_node = *memory_per_cpu * *cpus_per_node *
                              1024U * 1024U,
            .source = "SLURM_MEM_PER_CPU*SLURM_CPUS_ON_NODE",
        };
      }
    }
  }
  const long pages = sysconf(_SC_PHYS_PAGES);
  const long page_size = sysconf(_SC_PAGESIZE);
  if(pages <= 0 || page_size <= 0 ||
     static_cast<unsigned long long>(pages) >
         std::numeric_limits<std::size_t>::max() /
             static_cast<unsigned long long>(page_size)) {
    throw std::runtime_error(
        "unable to detect node memory; provide MEMORY_GIB explicitly");
  }
  return DetectedMemory{
      .bytes_per_node = static_cast<std::size_t>(pages) *
                        static_cast<std::size_t>(page_size),
      .source = "sysconf physical memory",
  };
}

[[nodiscard]] double gibibytes(std::size_t bytes) {
  return static_cast<double>(bytes) /
         (1024.0 * 1024.0 * 1024.0);
}

[[nodiscard]] std::vector<lmp2_1m2m::mp2::MonomerAssignment>
assignments(const lmp2_1m2m::localization::PipekMezeyResult& localized,
            const lmp2_1m2m::molecule::Molecule& molecule) {
  std::vector<lmp2_1m2m::mp2::MonomerAssignment> result;
  result.reserve(localized.assignments.size());
  for(const lmp2_1m2m::localization::OrbitalAssignment& assignment :
      localized.assignments) {
    if(!std::isfinite(assignment.monomer_population) ||
       assignment.monomer_population < -2.0e-10 ||
       assignment.monomer_population > 1.0 + 2.0e-10) {
      throw std::runtime_error(
          "localized monomer population lies outside [0,1]");
    }
    result.push_back(lmp2_1m2m::mp2::MonomerAssignment{
        .monomer_id = molecule.monomer_ids().at(assignment.atom_index),
        .confidence = std::clamp(assignment.monomer_population, 0.0, 1.0),
    });
  }
  return result;
}

struct LocalizedSpace {
  lmp2_1m2m::linalg::Matrix rotation;
  std::vector<lmp2_1m2m::mp2::MonomerAssignment> assignments;
  int iterations{0};
  std::optional<lmp2_1m2m::localization::PipekMezeyResult>
      pipek_mezey_diagnostics;
};

[[nodiscard]] LocalizedSpace localized_space_from_checkpoint(
    lmp2_1m2m::localization::PipekMezeyResult localized,
    const lmp2_1m2m::molecule::Molecule& molecule) {
  return LocalizedSpace{
      .rotation = localized.rotation,
      .assignments = assignments(localized, molecule),
      .iterations = localized.completed_sweeps,
      .pipek_mezey_diagnostics = std::move(localized),
  };
}

class CollectiveAssignmentGateFailure final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

[[nodiscard]] lmp2_1m2m::mp2::MonomerAssignmentQuality
require_confident_localization(
    std::span<const lmp2_1m2m::mp2::MonomerAssignment> assignments,
    double threshold, std::string_view space, std::string_view metric,
    int rank) {
  auto quality = lmp2_1m2m::mp2::summarize_assignment_quality(
      assignments, threshold);

#ifdef LMP2_1M2M_HAS_MPI
  const auto local_low =
      static_cast<unsigned long long>(quality.low_confidence_count);
  unsigned long long global_low = 0;
  double global_minimum = quality.minimum_confidence;
  if(MPI_Allreduce(&local_low, &global_low, 1, MPI_UNSIGNED_LONG_LONG,
                   MPI_MAX, MPI_COMM_WORLD) != MPI_SUCCESS ||
     MPI_Allreduce(&quality.minimum_confidence, &global_minimum, 1,
                   MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD) != MPI_SUCCESS) {
    throw std::runtime_error(
        "unable to reduce monomer-assignment confidence gate");
  }
  quality.low_confidence_count = static_cast<std::size_t>(global_low);
  quality.minimum_confidence = global_minimum;
#endif

  if(quality.low_confidence_count == 0) {
    return quality;
  }

  std::ostringstream message;
  message << std::setprecision(17)
          << space
          << " localization failed the hard monomer-assignment confidence "
             "gate: "
          << quality.low_confidence_count << '/' << quality.orbital_count
          << " orbitals have " << metric << " below " << threshold
          << "; minimum " << metric << '=' << quality.minimum_confidence;
  if(space == "occupied") {
    message << "; virtual localization, quadrature construction, and "
               "energy evaluation were not started";
  } else {
    message << "; quadrature construction and energy evaluation were not "
               "started";
  }
  if(rank == 0) {
    std::cerr << "fatal " << message.str() << '\n';
  }
  throw CollectiveAssignmentGateFailure(message.str());
}

[[nodiscard]] std::vector<lmp2_1m2m::mp2::MonomerAssignment>
assign_centroids_to_nearest_oxygen_monomers(
    const lmp2_1m2m::molecule::Molecule& molecule,
    const std::vector<lmp2_1m2m::localization::OrbitalSpatialDiagnostic>&
        diagnostics) {
  std::vector<std::array<double, 3>> centroids;
  centroids.reserve(diagnostics.size());
  for(const auto& diagnostic : diagnostics) {
    centroids.push_back(diagnostic.centroid_bohr);
  }
  std::vector<std::array<double, 3>> oxygen_positions;
  std::vector<int> oxygen_monomers;
  for(std::size_t atom = 0; atom < molecule.atoms().size(); ++atom) {
    if(molecule.atoms()[atom].atomic_number == 8) {
      oxygen_positions.push_back(molecule.atoms()[atom].position_bohr);
      oxygen_monomers.push_back(molecule.monomer_ids()[atom]);
    }
  }
  if(oxygen_positions.size() != molecule.atoms().size() / 3) {
    throw std::runtime_error(
        "FM water-monomer assignment requires one oxygen per water");
  }
  return lmp2_1m2m::mp2::assign_centroids_to_nearest_monomer_references(
      centroids, oxygen_positions, oxygen_monomers);
}

[[nodiscard]] lmp2_1m2m::linalg::Matrix select_columns(
    const lmp2_1m2m::linalg::Matrix& coefficients,
    const std::vector<std::size_t>& indices) {
  lmp2_1m2m::linalg::Matrix selected{coefficients.rows(), indices.size()};
  for(std::size_t column = 0; column < indices.size(); ++column) {
    if(indices[column] >= coefficients.columns()) {
      throw std::invalid_argument("localization orbital index is invalid");
    }
    for(std::size_t row = 0; row < coefficients.rows(); ++row) {
      selected(row, column) = coefficients(row, indices[column]);
    }
  }
  return selected;
}

[[nodiscard]] lmp2_1m2m::localization::LocalizationParallelOptions
localization_parallel_options(int rank, int ranks) {
  lmp2_1m2m::localization::LocalizationParallelOptions result{
      .rank = static_cast<std::size_t>(rank),
      .ranks = static_cast<std::size_t>(ranks),
  };
#ifdef LMP2_1M2M_HAS_MPI
  if(ranks > 1) {
    result.sum_in_place = [](std::vector<double>& values) {
      std::size_t offset = 0;
      while(offset < values.size()) {
        const int chunk = static_cast<int>(std::min(
            values.size() - offset,
            static_cast<std::size_t>(std::numeric_limits<int>::max())));
        if(MPI_Allreduce(MPI_IN_PLACE, values.data() + offset, chunk,
                         MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) != MPI_SUCCESS) {
          throw std::runtime_error(
              "MPI reduction failed during orbital localization");
        }
        offset += static_cast<std::size_t>(chunk);
      }
    };
    result.broadcast_from_root = [](std::vector<double>& values) {
      std::size_t offset = 0;
      while(offset < values.size()) {
        const int chunk = static_cast<int>(std::min(
            values.size() - offset,
            static_cast<std::size_t>(std::numeric_limits<int>::max())));
        if(MPI_Bcast(values.data() + offset, chunk, MPI_DOUBLE, 0,
                     MPI_COMM_WORLD) != MPI_SUCCESS) {
          throw std::runtime_error(
              "MPI broadcast failed during orbital localization");
        }
        offset += static_cast<std::size_t>(chunk);
      }
    };
  }
#else
  (void)rank;
#endif
  return result;
}

[[nodiscard]] LocalizedSpace localize_space(
    const std::string& method, const lmp2_1m2m::molecule::Molecule& molecule,
    const lmp2_1m2m::basis::BasisSet& basis,
    const lmp2_1m2m::linalg::Matrix& overlap,
    const lmp2_1m2m::linalg::Matrix& canonical_coefficients,
    const std::vector<std::size_t>& orbital_indices,
    const std::optional<lmp2_1m2m::localization::CartesianMomentIntegrals>&
        moments,
    const lmp2_1m2m::linalg::Matrix* meta_lowdin_orthogonal_ao,
    int rank, int ranks, int maximum_iterations,
    bool retain_pipek_mezey_diagnostics, const char* space_label) {
  if(method == "pm" || method == "pm-lowdin") {
    const bool meta_lowdin = method == "pm";
    if(meta_lowdin && meta_lowdin_orthogonal_ao == nullptr) {
      throw std::logic_error("PM meta-Lowdin AO basis is unavailable");
    }
    auto localized = lmp2_1m2m::localization::localize_pipek_mezey(
        molecule, basis, overlap, canonical_coefficients, orbital_indices,
        lmp2_1m2m::localization::PipekMezeyOptions{
            .maximum_sweeps = 100,
            .objective_tolerance = 1.0e-6,
            .optimizer = lmp2_1m2m::localization::
                PipekMezeyOptimizer::pyscf_ciah,
            .gradient_tolerance = 0.0,
            .ambiguity_population_margin = 0.05,
            .population_method =
                meta_lowdin
                    ? lmp2_1m2m::localization::
                          PipekMezeyPopulationMethod::meta_lowdin
                    : lmp2_1m2m::localization::
                          PipekMezeyPopulationMethod::lowdin,
            .meta_lowdin_orthogonal_ao =
                meta_lowdin ? *meta_lowdin_orthogonal_ao
                            : lmp2_1m2m::linalg::Matrix{},
            .atomic_initial_guess = meta_lowdin,
            .progress = [rank, space_label](
                const lmp2_1m2m::localization::PipekMezeySweep& cycle,
                const lmp2_1m2m::linalg::Matrix&,
                const std::vector<double>&) {
              if(rank == 0 && cycle.completed_sweeps > 0) {
                std::cerr << std::setprecision(10)
                          << "progress " << space_label
                          << " pm-ciah macro="
                          << cycle.completed_sweeps
                          << " objective=" << cycle.objective
                          << " delta=" << cycle.sweep_gain
                          << " gradient=" << cycle.gradient_norm
                          << " keyframes=" << cycle.keyframes
                          << " hessian_actions="
                          << cycle.hessian_actions << '\n';
                }
            },
            .parallel = localization_parallel_options(rank, ranks),
        });
    LocalizedSpace result{
        .rotation = localized.rotation,
        .assignments = assignments(localized, molecule),
        .iterations = localized.completed_sweeps,
    };
    if(retain_pipek_mezey_diagnostics) {
      result.pipek_mezey_diagnostics = std::move(localized);
    }
    return result;
  }
  if(method != "fm2" && method != "fm4") {
    throw std::invalid_argument(
        "localizer must be pm, pm-lowdin, fm2, or fm4");
  }
  if(!moments.has_value()) {
    throw std::logic_error("FM localization requires AO moment integrals");
  }
  const lmp2_1m2m::linalg::Matrix canonical_space =
      select_columns(canonical_coefficients, orbital_indices);
  const auto localized = lmp2_1m2m::localization::localize_fourth_moment(
      canonical_space, overlap, *moments, method == "fm2" ? 2 : 4,
      lmp2_1m2m::localization::TrustRegionLocalizationOptions{
          .maximum_iterations = maximum_iterations,
          .gradient_tolerance = 1.0e-8,
          .objective_tolerance = 1.0e-12,
          .parallel = localization_parallel_options(rank, ranks),
      });
  if(!localized.converged) {
    throw std::runtime_error(method +
                             " localization did not converge: " +
                             localized.convergence_reason);
  }
  return LocalizedSpace{
      .rotation = localized.rotation,
      .assignments =
          assign_centroids_to_nearest_oxygen_monomers(
              molecule, localized.diagnostics),
      .iterations = localized.iterations,
  };
}

void write_pipek_mezey_diagnostics_collectively(
    const LocalizedSpace& occupied,
    const std::vector<std::size_t>& occupied_indices,
    const LocalizedSpace& virtual_space,
    const std::vector<std::size_t>& virtual_indices,
    const lmp2_1m2m::molecule::Molecule& molecule,
    const std::filesystem::path& prefix, int rank) {
  int write_status = 0;
  std::string write_error;
  if(rank == 0) {
    try {
      if(!occupied.pipek_mezey_diagnostics ||
         !virtual_space.pipek_mezey_diagnostics) {
        throw std::logic_error(
            "Pipek-Mezey diagnostics were not retained");
      }
      const std::filesystem::path occupied_prefix{
          prefix.string() + "_occupied"};
      const std::filesystem::path virtual_prefix{
          prefix.string() + "_virtual"};
      static_cast<void>(
          lmp2_1m2m::localization::write_pipek_mezey_diagnostics(
              *occupied.pipek_mezey_diagnostics, molecule,
              occupied_indices, occupied_prefix));
      static_cast<void>(
          lmp2_1m2m::localization::write_pipek_mezey_diagnostics(
              *virtual_space.pipek_mezey_diagnostics, molecule,
              virtual_indices, virtual_prefix));
      std::cerr << "progress Pipek-Mezey diagnostics written; prefix="
                << prefix.string() << '\n';
    } catch(const std::exception& error) {
      write_status = 1;
      write_error = error.what();
    }
  }
#ifdef LMP2_1M2M_HAS_MPI
  if(MPI_Bcast(&write_status, 1, MPI_INT, 0, MPI_COMM_WORLD) !=
     MPI_SUCCESS) {
    throw std::runtime_error(
        "unable to broadcast Pipek-Mezey diagnostic status");
  }
#endif
  if(write_status != 0) {
    throw std::runtime_error(
        rank == 0 ? write_error
                  : "rank zero failed to write Pipek-Mezey diagnostics");
  }
}

[[nodiscard]] double maximum_rank_time(double local) {
#ifdef LMP2_1M2M_HAS_MPI
  double maximum = local;
  if(MPI_Allreduce(MPI_IN_PLACE, &maximum, 1, MPI_DOUBLE, MPI_MAX,
                   MPI_COMM_WORLD) != MPI_SUCCESS) {
    throw std::runtime_error("unable to reduce localized LMP2-1M2M wall time");
  }
  return maximum;
#else
  return local;
#endif
}

void broadcast_matrix_from_rank_zero(lmp2_1m2m::linalg::Matrix& matrix,
                                     int rank, int ranks) {
#ifdef LMP2_1M2M_HAS_MPI
  if(ranks <= 1) {
    return;
  }
  std::array<unsigned long long, 2> dimensions{
      rank == 0 ? static_cast<unsigned long long>(matrix.rows()) : 0ULL,
      rank == 0 ? static_cast<unsigned long long>(matrix.columns()) : 0ULL,
  };
  if(MPI_Bcast(dimensions.data(), static_cast<int>(dimensions.size()),
               MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD) != MPI_SUCCESS) {
    throw std::runtime_error(
        "unable to broadcast localization matrix dimensions");
  }
  if(dimensions[0] == 0 || dimensions[1] == 0 ||
     dimensions[0] > static_cast<unsigned long long>(
                         std::numeric_limits<std::size_t>::max()) ||
     dimensions[1] > static_cast<unsigned long long>(
                         std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error(
        "broadcast localization matrix dimensions are invalid");
  }
  if(rank != 0) {
    matrix = lmp2_1m2m::linalg::Matrix{
        static_cast<std::size_t>(dimensions[0]),
        static_cast<std::size_t>(dimensions[1])};
  }
  std::size_t offset = 0;
  while(offset < matrix.size()) {
    const int chunk = static_cast<int>(std::min(
        matrix.size() - offset,
        static_cast<std::size_t>(std::numeric_limits<int>::max())));
    if(MPI_Bcast(matrix.data() + offset, chunk, MPI_DOUBLE, 0,
                 MPI_COMM_WORLD) != MPI_SUCCESS) {
      throw std::runtime_error(
          "unable to broadcast localization matrix values");
    }
    offset += static_cast<std::size_t>(chunk);
  }
#else
  (void)matrix;
  (void)rank;
  (void)ranks;
#endif
}

[[nodiscard]] lmp2_1m2m::mp2::LocalizedLaplaceMp2Backend parse_backend(
    const std::string& name) {
  if(name == "auto") {
    return lmp2_1m2m::mp2::LocalizedLaplaceMp2Backend::automatic;
  }
  if(name == "direct") {
    return lmp2_1m2m::mp2::LocalizedLaplaceMp2Backend::direct_tiled;
  }
  if(name == "localized-direct-selected" || name == "direct-selected") {
    return lmp2_1m2m::mp2::LocalizedLaplaceMp2Backend::
        localized_direct_selected;
  }
  if(name == "localized-cached" || name == "localized-cached-ovov") {
    return lmp2_1m2m::mp2::LocalizedLaplaceMp2Backend::
        localized_cached_ovov;
  }
  if(name == "cached") {
    return lmp2_1m2m::mp2::LocalizedLaplaceMp2Backend::cached_canonical_ovov;
  }
  if(name == "shared-selected") {
    return lmp2_1m2m::mp2::LocalizedLaplaceMp2Backend::shared_selected_ovov;
  }
  if(name == "replicated-selected") {
    return lmp2_1m2m::mp2::LocalizedLaplaceMp2Backend::
        replicated_selected_ovov;
  }
  if(name == "distributed-selected" || name == "selected") {
    return lmp2_1m2m::mp2::LocalizedLaplaceMp2Backend::
        distributed_selected_ovov;
  }
  throw std::invalid_argument(
      "localized LMP2-1M2M backend must be auto, direct, cached, "
      "localized-direct-selected, localized-cached-ovov, shared-selected, "
      "replicated-selected, or selected");
}

[[nodiscard]] const char* backend_name(
    lmp2_1m2m::mp2::LocalizedLaplaceMp2Backend backend) {
  switch(backend) {
    case lmp2_1m2m::mp2::LocalizedLaplaceMp2Backend::automatic:
      return "auto";
    case lmp2_1m2m::mp2::LocalizedLaplaceMp2Backend::direct_tiled:
      return "direct_tiled";
    case lmp2_1m2m::mp2::LocalizedLaplaceMp2Backend::
        localized_direct_selected:
      return "localized_direct_selected";
    case lmp2_1m2m::mp2::LocalizedLaplaceMp2Backend::localized_cached_ovov:
      return "localized_cached_ovov";
    case lmp2_1m2m::mp2::LocalizedLaplaceMp2Backend::cached_canonical_ovov:
      return "cached_canonical_ovov";
    case lmp2_1m2m::mp2::LocalizedLaplaceMp2Backend::shared_selected_ovov:
      return "shared_selected_ovov";
    case lmp2_1m2m::mp2::LocalizedLaplaceMp2Backend::
        replicated_selected_ovov:
      return "replicated_selected_ovov";
    case lmp2_1m2m::mp2::LocalizedLaplaceMp2Backend::
        distributed_selected_ovov:
      return "distributed_selected_ovov";
  }
  throw std::logic_error("unknown localized LMP2-1M2M backend");
}

[[nodiscard]] const char* energy_kernel_name(
    lmp2_1m2m::mp2::LocalizedLaplaceMp2Backend backend) {
  switch(backend) {
    case lmp2_1m2m::mp2::LocalizedLaplaceMp2Backend::cached_canonical_ovov:
      return "threaded_blas_separable";
    case lmp2_1m2m::mp2::LocalizedLaplaceMp2Backend::shared_selected_ovov:
      return "openmp_blocked_separable_four_pass_selected_panels";
    case lmp2_1m2m::mp2::LocalizedLaplaceMp2Backend::
        replicated_selected_ovov:
      return "mpi_quadrature_openmp_row_blocked_selected_panels";
    case lmp2_1m2m::mp2::LocalizedLaplaceMp2Backend::
        distributed_selected_ovov:
      return "distributed_selected_pair_panels";
    case lmp2_1m2m::mp2::LocalizedLaplaceMp2Backend::localized_cached_ovov:
      return "distributed_localized_propagator_panels";
    case lmp2_1m2m::mp2::LocalizedLaplaceMp2Backend::automatic:
    case lmp2_1m2m::mp2::LocalizedLaplaceMp2Backend::direct_tiled:
      return "not_applicable";
    case lmp2_1m2m::mp2::LocalizedLaplaceMp2Backend::
        localized_direct_selected:
      return "mpi_point_semidirect_selected_tiles";
  }
  throw std::logic_error("unknown localized LMP2-1M2M backend");
}

[[nodiscard]] lmp2_1m2m::mp2::LaplaceQuadratureScheme
parse_quadrature_scheme(const std::string& name) {
  if(name == "wa-continuous") {
    return lmp2_1m2m::mp2::LaplaceQuadratureScheme::wa_continuous;
  }
  if(name == "laguerre" || name == "gauss-laguerre") {
    return lmp2_1m2m::mp2::LaplaceQuadratureScheme::gauss_laguerre;
  }
  if(name == "minimax" || name == "bh-minimax" ||
     name == "braess-hackbusch") {
    return lmp2_1m2m::mp2::LaplaceQuadratureScheme::minimax;
  }
  throw std::invalid_argument(
      "quadrature must be wa-continuous, laguerre, or minimax");
}

[[nodiscard]] const char* quadrature_scheme_name(
    lmp2_1m2m::mp2::LaplaceQuadratureScheme scheme) {
  switch(scheme) {
    case lmp2_1m2m::mp2::LaplaceQuadratureScheme::wa_continuous:
      return "wa-continuous";
    case lmp2_1m2m::mp2::LaplaceQuadratureScheme::gauss_laguerre:
      return "gauss-laguerre";
    case lmp2_1m2m::mp2::LaplaceQuadratureScheme::minimax:
      return "minimax";
  }
  throw std::logic_error("unknown Laplace quadrature scheme");
}

}  // namespace

int main(int argc, char** argv) {
  int rank = 0;
  int ranks = 1;
  int ranks_per_node = 1;
#ifdef LMP2_1M2M_HAS_MPI
  int provided = MPI_THREAD_SINGLE;
  if(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) !=
         MPI_SUCCESS ||
     provided < MPI_THREAD_FUNNELED) {
    std::cerr << "MPI_THREAD_FUNNELED is unavailable\n";
    return 2;
  }
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  MPI_Comm node_communicator = MPI_COMM_NULL;
  if(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                         MPI_INFO_NULL, &node_communicator) != MPI_SUCCESS ||
     MPI_Comm_size(node_communicator, &ranks_per_node) != MPI_SUCCESS ||
     MPI_Comm_free(&node_communicator) != MPI_SUCCESS) {
    std::cerr << "unable to determine node-local MPI topology\n";
    MPI_Abort(MPI_COMM_WORLD, 2);
    return 2;
  }
#endif
  int local_status = 0;
  try {
    if(argc < 11 || argc > 23) {
      throw std::invalid_argument(
          "usage: lmp2_1m2m_localized_lmp2_1m2m GEOMETRY.xyz BASIS POINTS "
          "SCHWARZ_THRESHOLD THREADS OCC_TILE|auto VIR_TILE|auto "
          "MEMORY_GIB|auto "
          "MODE(full|1m2m) ASSIGNMENT_CONFIDENCE "
          "[LOCALIZER(pm|pm-lowdin|fm2|fm4)] "
          "[BACKEND(auto|direct|localized-direct-selected|cached|"
          "localized-cached-ovov|shared-selected|replicated-selected|"
          "selected)] "
          "[QUADRATURE(wa-continuous|laguerre|minimax)] "
          "[PM_DIAGNOSTIC_PREFIX|none] "
          "[LOCALIZED_ENERGY_SCHWARZ_THRESHOLD_HARTREE] "
          "[LOCALIZED_POINT_ENERGY_THRESHOLD_HARTREE] "
          "[RHF_ERI_SCHWARZ_THRESHOLD] "
          "[RHF_FOCK_CONTRIBUTION_THRESHOLD] "
          "[RHF_CHECKPOINT_OUTPUT|none] "
          "[RHF_INITIAL_GUESS_CHECKPOINT|none] "
          "[RHF_INITIAL_GUESS_BASIS|same] "
          "[RESOURCE_POLICY(auto|smart|manual)]");
    }
    const std::filesystem::path geometry = argv[1];
    const std::string basis_name = argv[2];
    const std::size_t points =
        static_cast<std::size_t>(std::stoull(argv[3]));
    const double schwarz = std::stod(argv[4]);
    const std::size_t threads =
        static_cast<std::size_t>(std::stoull(argv[5]));
    const std::optional<std::size_t> occupied_tile_input =
        optional_positive_size(argv[6], "occupied tile");
    const std::optional<std::size_t> virtual_tile_input =
        optional_positive_size(argv[7], "virtual tile");
    const std::optional<std::size_t> memory_input =
        optional_memory_bytes(argv[8]);
    const std::string mode_name = argv[9];
    const double confidence = std::stod(argv[10]);
    const std::string localizer = argc >= 12 ? argv[11] : "fm2";
    const lmp2_1m2m::mp2::LocalizedLaplaceMp2Backend requested_backend_input =
        argc >= 13 ? parse_backend(argv[12])
                   : lmp2_1m2m::mp2::LocalizedLaplaceMp2Backend::automatic;
    const lmp2_1m2m::mp2::LaplaceQuadratureScheme quadrature_scheme =
        argc >= 14
            ? parse_quadrature_scheme(argv[13])
            : lmp2_1m2m::mp2::LaplaceQuadratureScheme::wa_continuous;
    const std::optional<std::filesystem::path> pm_diagnostic_prefix =
        argc >= 15 && std::string_view{argv[14]} != "none"
            ? std::optional<std::filesystem::path>{argv[14]}
            : std::nullopt;
    const lmp2_1m2m::mp2::NmExecutionMode mode =
        mode_name == "full"
            ? lmp2_1m2m::mp2::NmExecutionMode::full
            : mode_name == "1m2m"
                  ? lmp2_1m2m::mp2::NmExecutionMode::one_two_monomer
                  : throw std::invalid_argument(
                        "localized LMP2-1M2M mode must be 'full' or '1m2m'");
    const bool localized_direct_selected_requested =
        requested_backend_input ==
        lmp2_1m2m::mp2::LocalizedLaplaceMp2Backend::
            localized_direct_selected;
    const double localized_energy_schwarz =
        argc >= 16
            ? std::stod(argv[15])
            : mode == lmp2_1m2m::mp2::NmExecutionMode::one_two_monomer &&
                      !localized_direct_selected_requested &&
                      requested_backend_input !=
                          lmp2_1m2m::mp2::LocalizedLaplaceMp2Backend::
                              automatic &&
                      requested_backend_input !=
                          lmp2_1m2m::mp2::LocalizedLaplaceMp2Backend::
                              localized_cached_ovov
                  ? 1.0e-12
                  : 0.0;
    const double localized_point_energy_threshold =
        argc >= 17 ? std::stod(argv[16]) : 0.0;
    const double rhf_eri_schwarz_threshold =
        argc >= 18 ? std::stod(argv[17]) : 0.0;
    const double rhf_fock_contribution_threshold =
        argc >= 19 ? std::stod(argv[18]) : 0.0;
    const std::optional<std::filesystem::path> rhf_checkpoint_output =
        argc >= 20 && std::string_view{argv[19]} != "none"
            ? std::optional<std::filesystem::path>{argv[19]}
            : std::nullopt;
    std::optional<std::filesystem::path> rhf_initial_guess_checkpoint =
        argc >= 21 && std::string_view{argv[20]} != "none"
            ? std::optional<std::filesystem::path>{argv[20]}
            : std::nullopt;
    const std::string rhf_initial_guess_basis =
        argc >= 22 ? argv[21] : "same";
    const std::string resource_policy =
        argc >= 23 ? argv[22] : "auto";
    if(points == 0 ||
       points > std::numeric_limits<std::size_t>::max() / 200U ||
       threads == 0 || !std::isfinite(schwarz) || schwarz < 0.0 ||
       !std::isfinite(localized_energy_schwarz) ||
       localized_energy_schwarz < 0.0 ||
       !std::isfinite(localized_point_energy_threshold) ||
       localized_point_energy_threshold < 0.0 ||
       !std::isfinite(rhf_eri_schwarz_threshold) ||
       rhf_eri_schwarz_threshold < 0.0 ||
       !std::isfinite(rhf_fock_contribution_threshold) ||
       rhf_fock_contribution_threshold < 0.0 ||
       !std::isfinite(confidence) || confidence < 0.0 || confidence > 1.0) {
      throw std::invalid_argument("localized LMP2-1M2M options are invalid");
    }
    if(resource_policy != "auto" && resource_policy != "smart" &&
       resource_policy != "manual") {
      throw std::invalid_argument(
          "localized LMP2-1M2M resource policy must be auto, smart, or manual");
    }
    if(localizer != "pm" && localizer != "pm-lowdin" &&
       localizer != "fm2" && localizer != "fm4") {
      throw std::invalid_argument(
          "localizer must be pm, pm-lowdin, fm2, or fm4");
    }
    if(pm_diagnostic_prefix && localizer != "pm" &&
       localizer != "pm-lowdin") {
      throw std::invalid_argument(
          "PM_DIAGNOSTIC_PREFIX is available only for pm or pm-lowdin");
    }

    const auto total_start = std::chrono::steady_clock::now();
    const auto atoms = lmp2_1m2m::molecule::read_xyz_file(
        geometry, lmp2_1m2m::core::CoordinateUnit::angstrom);
    const std::vector<int> monomers =
        lmp2_1m2m::molecule::contiguous_water_monomers(atoms, 3);
    const lmp2_1m2m::molecule::Molecule molecule{atoms, 0, 1, monomers};
    lmp2_1m2m::integrals::Libint2IntegralProvider provider{
        molecule, basis_name, true};

    if(rhf_checkpoint_output &&
       std::filesystem::exists(*rhf_checkpoint_output)) {
      if(rhf_initial_guess_checkpoint &&
         *rhf_initial_guess_checkpoint != *rhf_checkpoint_output) {
        throw std::invalid_argument(
            "existing RHF output checkpoint and a different initial-guess "
            "checkpoint are ambiguous");
      }
      rhf_initial_guess_checkpoint = rhf_checkpoint_output;
    }
    std::optional<std::filesystem::path> localization_checkpoint;
    if(localizer == "pm" || localizer == "pm-lowdin") {
      const std::optional<std::filesystem::path> checkpoint_base =
          rhf_checkpoint_output
              ? rhf_checkpoint_output
              : (rhf_initial_guess_basis == "same" ||
                         rhf_initial_guess_basis == basis_name
                     ? rhf_initial_guess_checkpoint
                     : std::nullopt);
      if(checkpoint_base) {
        localization_checkpoint = std::filesystem::path{
            checkpoint_base->string() + "." + localizer +
            "-pyscf-ciah.localization.h5"};
      }
    }

    std::optional<lmp2_1m2m::linalg::Matrix> rhf_initial_density;
    std::optional<lmp2_1m2m::scf::ProjectedRhfGuess> projected_rhf_guess;
    std::optional<lmp2_1m2m::scf::RhfResult> restored_rhf;
    std::string rhf_guess_name = "core_hamiltonian";
    if(rhf_initial_guess_checkpoint) {
#ifdef LMP2_1M2M_HAS_HDF5
      if(!std::filesystem::is_regular_file(*rhf_initial_guess_checkpoint)) {
        throw std::runtime_error(
            "RHF initial-guess checkpoint is missing: " +
            rhf_initial_guess_checkpoint->string());
      }
      if(rhf_initial_guess_basis == "same" ||
         rhf_initial_guess_basis == basis_name) {
        restored_rhf = lmp2_1m2m::io::read_rhf_result_checkpoint(
            *rhf_initial_guess_checkpoint, molecule,
            provider.basis_metadata());
        rhf_guess_name = "same_basis_checkpoint_exact";
      } else {
        lmp2_1m2m::integrals::Libint2IntegralProvider source_provider{
            molecule, rhf_initial_guess_basis, true};
        const lmp2_1m2m::io::CanonicalRhfCheckpoint source_checkpoint =
            lmp2_1m2m::io::read_canonical_rhf_checkpoint(
                *rhf_initial_guess_checkpoint, molecule,
                source_provider.basis_metadata());
        const lmp2_1m2m::integrals::OneElectronIntegrals target_one_electron =
            provider.compute_one_electron();
        projected_rhf_guess = lmp2_1m2m::scf::project_occupied_rhf_guess(
            source_checkpoint.coefficients,
            molecule.occupied_orbitals_rhf(), target_one_electron.overlap,
            provider.compute_cross_overlap(source_provider), 1.0e-8,
            1.0e-10);
        rhf_initial_density = projected_rhf_guess->density;
        rhf_guess_name = "cross_basis_projected_" +
                         rhf_initial_guess_basis + "_to_" + basis_name;
      }
#else
      throw std::runtime_error(
          "RHF checkpoint restart requires an HDF5-enabled build");
#endif
    }

    if(rank == 0) {
      std::cerr << std::setprecision(17)
                << (restored_rhf ? "progress RHF checkpoint restore started; "
                                 : "progress RHF started; ")
                << "Schwarz threshold="
                << rhf_eri_schwarz_threshold
                << "; density-aware Fock threshold="
                << rhf_fock_contribution_threshold
                << "; guess=" << rhf_guess_name << '\n';
      if(projected_rhf_guess) {
        std::cerr
            << "progress RHF projected guess; target metric rank="
            << projected_rhf_guess->target_metric_rank
            << "; minimum occupied Gram eigenvalue="
            << projected_rhf_guess->minimum_occupied_gram_eigenvalue
            << "; S-orthonormality error="
            << projected_rhf_guess
                   ->occupied_orthonormality_maximum_error
            << "; electron count=" << projected_rhf_guess->electron_count
            << '\n';
      }
    }
    const auto rhf_start = std::chrono::steady_clock::now();
    auto rhf_iteration_start = rhf_start;
    const lmp2_1m2m::scf::RhfObserver rhf_observer =
        [rank, &rhf_iteration_start](
            const lmp2_1m2m::scf::RhfIteration& iteration) {
          if(rank != 0) {
            return;
          }
          const auto now = std::chrono::steady_clock::now();
          const double iteration_seconds =
              std::chrono::duration<double>(now - rhf_iteration_start)
                  .count();
          rhf_iteration_start = now;
          std::cerr << std::setprecision(17)
                    << "progress RHF iteration=" << iteration.iteration
                    << " total_energy_hartree=" << iteration.total_energy
                    << " energy_change_hartree="
                    << iteration.energy_change
                    << " density_rms=" << iteration.density_rms
                    << " commutator_rms=" << iteration.commutator_rms
                    << " iteration_seconds=" << iteration_seconds << '\n';
        };
    const bool rhf_checkpoint_restored = restored_rhf.has_value();
    const auto rhf = [&]() -> lmp2_1m2m::scf::RhfResult {
      if(restored_rhf) {
        return std::move(*restored_rhf);
      }
      return lmp2_1m2m::scf::run_rhf(
          molecule, provider,
          lmp2_1m2m::scf::RhfOptions{
              .maximum_iterations = 100,
              .energy_tolerance = 1.0e-10,
              .density_rms_tolerance = 1.0e-8,
              .commutator_rms_tolerance = 1.0e-7,
              .linear_dependency_tolerance = 1.0e-8,
              .diis = true,
              .diis_start = 2,
              .diis_subspace = 8,
              .damping = 0.0,
              .level_shift = 0.0,
              .eri_schwarz_threshold = rhf_eri_schwarz_threshold,
              .density_screen = rhf_fock_contribution_threshold > 0.0,
              .fock_contribution_threshold =
                  rhf_fock_contribution_threshold,
              .mpi_rank = static_cast<std::size_t>(rank),
              .mpi_ranks = static_cast<std::size_t>(ranks),
              .fock_threads = threads,
          },
          rhf_initial_density, rhf_observer);
    }();
    const double rhf_seconds =
        maximum_rank_time(seconds_since(rhf_start));
    if(rank == 0) {
      if(rhf_checkpoint_restored) {
        std::cerr << "progress RHF restored exactly from checkpoint; path="
                  << rhf_initial_guess_checkpoint->string()
                  << "; stored_iterations=" << rhf.iteration_count
                  << "; iterations_this_run=0; wall_seconds="
                  << rhf_seconds << '\n';
      } else {
        std::cerr << "progress RHF complete; iterations="
                  << rhf.iteration_count
                  << "; iterations_this_run=" << rhf.iteration_count
                  << "; wall_seconds=" << rhf_seconds << '\n';
      }
    }

    if(rhf_checkpoint_output &&
       !std::filesystem::exists(*rhf_checkpoint_output)) {
#ifdef LMP2_1M2M_HAS_HDF5
      if(threads >
         static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            "RHF checkpoint OpenMP thread count exceeds int");
      }
      std::exception_ptr checkpoint_error;
      if(rank == 0) {
        try {
          lmp2_1m2m::io::write_rhf_checkpoint(
              *rhf_checkpoint_output, molecule, provider.basis_metadata(),
              rhf, true,
              lmp2_1m2m::io::CheckpointRuntime{
                  .mpi_size = ranks,
                  .openmp_threads = static_cast<int>(threads),
              },
              false);
          std::cerr << "progress RHF checkpoint written; path="
                    << rhf_checkpoint_output->string() << '\n';
        } catch(...) {
          checkpoint_error = std::current_exception();
        }
      }
#ifdef LMP2_1M2M_HAS_MPI
      if(ranks > 1) {
        int checkpoint_failed = checkpoint_error ? 1 : 0;
        if(MPI_Allreduce(MPI_IN_PLACE, &checkpoint_failed, 1, MPI_INT,
                         MPI_MAX, MPI_COMM_WORLD) != MPI_SUCCESS) {
          throw std::runtime_error(
              "MPI reduction failed after RHF checkpoint write");
        }
        if(checkpoint_failed != 0 && !checkpoint_error) {
          throw std::runtime_error(
              "RHF checkpoint write failed on rank zero");
        }
      }
#endif
      if(checkpoint_error) {
        std::rethrow_exception(checkpoint_error);
      }
#else
      throw std::runtime_error(
          "RHF checkpoint output requires an HDF5-enabled build");
#endif
    }

    const std::size_t occupied_count = molecule.occupied_orbitals_rhf();
    const std::size_t frozen_core = atoms.size() / 3;
    if(frozen_core >= occupied_count) {
      throw std::runtime_error("frozen core removes all occupied orbitals");
    }
    std::vector<std::size_t> active_occupied;
    std::vector<std::size_t> virtuals;
    for(std::size_t orbital = frozen_core; orbital < occupied_count;
        ++orbital) {
      active_occupied.push_back(orbital);
    }
    for(std::size_t orbital = occupied_count;
        orbital < rhf.coefficients.columns(); ++orbital) {
      virtuals.push_back(orbital);
    }

    const bool incomplete_manual_resources =
        !occupied_tile_input.has_value() ||
        !virtual_tile_input.has_value() || !memory_input.has_value();
    if(resource_policy == "manual" && incomplete_manual_resources) {
      throw std::invalid_argument(
          "manual resource policy requires numeric occupied tile, virtual "
          "tile, and memory values");
    }
    const bool smart_resources =
        resource_policy == "smart" ||
        (resource_policy == "auto" && incomplete_manual_resources);
    std::size_t occupied_tile = occupied_tile_input.value_or(1U);
    std::size_t virtual_tile = virtual_tile_input.value_or(1U);
    std::size_t quadrature_batch = 1U;
    std::size_t lambda_ao_block = virtual_tile_input.value_or(16U);
    std::size_t memory = memory_input.value_or(1U);
    auto requested_backend = requested_backend_input;
    std::optional<lmp2_1m2m::mp2::LocalizedMp2ResourcePlan> resource_plan;
    std::string memory_source = "explicit command line";
    if(smart_resources) {
      DetectedMemory detected_memory = detect_memory_per_node();
#ifdef LMP2_1M2M_HAS_MPI
      if(ranks > 1) {
        unsigned long long node_bytes =
            static_cast<unsigned long long>(detected_memory.bytes_per_node);
        if(MPI_Allreduce(MPI_IN_PLACE, &node_bytes, 1,
                         MPI_UNSIGNED_LONG_LONG, MPI_MIN,
                         MPI_COMM_WORLD) != MPI_SUCCESS ||
           node_bytes > static_cast<unsigned long long>(
                            std::numeric_limits<std::size_t>::max())) {
          throw std::runtime_error(
              "unable to reduce detected node memory");
        }
        detected_memory.bytes_per_node =
            static_cast<std::size_t>(node_bytes);
      }
#endif
      memory_source = detected_memory.source;
      auto planner_backend = requested_backend_input;
      const bool localized_screen_requested =
          localized_energy_schwarz > 0.0 ||
          localized_point_energy_threshold > 0.0;
      if(planner_backend ==
             lmp2_1m2m::mp2::LocalizedLaplaceMp2Backend::automatic &&
         localized_screen_requested &&
         mode == lmp2_1m2m::mp2::NmExecutionMode::one_two_monomer) {
        planner_backend =
            ranks == 1
                ? lmp2_1m2m::mp2::LocalizedLaplaceMp2Backend::
                      shared_selected_ovov
                : lmp2_1m2m::mp2::LocalizedLaplaceMp2Backend::
                      distributed_selected_ovov;
      }
      resource_plan = lmp2_1m2m::mp2::plan_localized_mp2_resources(
          lmp2_1m2m::mp2::LocalizedMp2MachineResources{
              .mpi_ranks = static_cast<std::size_t>(ranks),
              .ranks_per_node =
                  static_cast<std::size_t>(ranks_per_node),
              .threads_per_rank = threads,
              .available_memory_bytes_per_node =
                  detected_memory.bytes_per_node,
          },
          lmp2_1m2m::mp2::LocalizedMp2Workload{
              .atom_count = atoms.size(),
              .ao_count = provider.basis_metadata().number_of_aos,
              .active_occupied_count = active_occupied.size(),
              .virtual_count = virtuals.size(),
              .quadrature_points = points,
              .execution_mode = mode,
          },
          lmp2_1m2m::mp2::LocalizedMp2ResourceOverrides{
              .occupied_tile_size = occupied_tile_input,
              .virtual_tile_size = virtual_tile_input,
              .memory_limit_bytes_per_rank = memory_input,
              .backend = planner_backend,
          });
      occupied_tile = resource_plan->occupied_tile_size;
      virtual_tile = resource_plan->virtual_tile_size;
      quadrature_batch = resource_plan->quadrature_batch_size;
      lambda_ao_block = resource_plan->lambda_ao_block_size;
      memory = resource_plan->usable_memory_bytes_per_rank;
      requested_backend = resource_plan->backend;
      if(rank == 0) {
        std::cerr << std::setprecision(6)
                  << "resource plan policy=smart atoms=" << atoms.size()
                  << " aos=" << provider.basis_metadata().number_of_aos
                  << " active_occ=" << active_occupied.size()
                  << " virtual=" << virtuals.size()
                  << " quadrature_points=" << points << '\n'
                  << "resource plan topology nodes="
                  << resource_plan->nodes << " mpi_ranks=" << ranks
                  << " ranks_per_node=" << ranks_per_node
                  << " threads_per_rank=" << threads
                  << " memory_source=" << memory_source << '\n'
                  << "resource plan selected backend="
                  << backend_name(requested_backend)
                  << " occupied_tile=" << occupied_tile
                  << " occupied_batches="
                  << resource_plan->occupied_batches
                  << " distributed_transform_left_occupied="
                  << resource_plan
                         ->distributed_transform_occupied_count
                  << " virtual_tile=" << virtual_tile
                  << " virtual_batches="
                  << resource_plan->virtual_batches
                  << " quadrature_batch=" << quadrature_batch
                  << " lambda_ao_block=" << lambda_ao_block << '\n'
                  << "resource plan memory usable_gib_per_rank="
                  << gibibytes(memory)
                  << " canonical_ovov_gib="
                  << gibibytes(resource_plan->canonical_ovov_bytes)
                  << " localized_ovov_gib="
                  << gibibytes(resource_plan->localized_ovov_bytes)
                  << " transform_peak_gib_per_rank="
                  << gibibytes(
                         resource_plan
                             ->estimated_transform_peak_bytes_per_rank)
                  << " estimated_total_peak_gib_per_rank="
                  << gibibytes(
                         resource_plan->estimated_total_peak_bytes_per_rank)
                  << '\n';
        if(resource_plan->memory_was_clamped_to_machine) {
          std::cerr
              << "resource plan warning=explicit memory exceeded the "
                 "safe node-local share and was clamped\n";
        }
        if(static_cast<std::size_t>(ranks) > points) {
          std::cerr
              << "resource plan warning=more MPI ranks than quadrature "
                 "points; some ranks cannot own a Laplace point\n";
        }
      }
    }

    const auto localization_start = std::chrono::steady_clock::now();
    bool localization_checkpoint_restored = false;
    std::optional<LocalizedSpace> occupied_localized_storage;
    std::optional<LocalizedSpace> virtual_localized_storage;
#ifdef LMP2_1M2M_HAS_HDF5
    if(localization_checkpoint &&
       std::filesystem::exists(*localization_checkpoint)) {
      lmp2_1m2m::io::LocalizedRhfCheckpoint saved =
          lmp2_1m2m::io::read_localized_rhf_checkpoint(
              *localization_checkpoint, molecule,
              provider.basis_metadata());
      const auto expected_population_method =
          localizer == "pm"
              ? lmp2_1m2m::localization::PipekMezeyPopulationMethod::
                    meta_lowdin
              : lmp2_1m2m::localization::PipekMezeyPopulationMethod::lowdin;
      if(saved.occupied.population_method != expected_population_method ||
         saved.virtual_space.population_method !=
             expected_population_method ||
         saved.occupied.optimizer !=
             lmp2_1m2m::localization::PipekMezeyOptimizer::pyscf_ciah ||
         saved.virtual_space.optimizer !=
             lmp2_1m2m::localization::PipekMezeyOptimizer::pyscf_ciah ||
         saved.canonical.total_energy != rhf.total_energy ||
         lmp2_1m2m::linalg::maximum_absolute_value(
             lmp2_1m2m::linalg::subtract(saved.canonical.coefficients,
                                         rhf.coefficients)) != 0.0) {
        throw std::runtime_error(
            "localization checkpoint does not match the current RHF state "
            "or PM/CIAH convention");
      }
      occupied_localized_storage = localized_space_from_checkpoint(
          std::move(saved.occupied), molecule);
      virtual_localized_storage = localized_space_from_checkpoint(
          std::move(saved.virtual_space), molecule);
      localization_checkpoint_restored = true;
      if(rank == 0) {
        std::cerr << "progress occupied and virtual localization restored; "
                     "path="
                  << localization_checkpoint->string() << '\n';
      }
    }
#endif
    std::optional<lmp2_1m2m::localization::CartesianMomentIntegrals> moments;
    std::optional<lmp2_1m2m::linalg::Matrix> meta_lowdin_orthogonal_ao;
    if(!localization_checkpoint_restored) {
      if(rank == 0) {
        std::cerr << "progress full occupied " << localizer
                  << " localization started; MPI ranks=" << ranks
                  << "; atom ownership=cyclic\n";
      }
      if(localizer == "fm2" || localizer == "fm4") {
        moments = provider.compute_cartesian_moments_through_fourth();
      }
      if(localizer == "pm") {
        if(rank == 0) {
          std::cerr << "progress rank-zero ANO meta-Lowdin AO "
                       "construction started\n";
        }
        int construction_status = 0;
        std::string construction_error;
        if(rank == 0) {
          try {
            const auto reference =
                provider.compute_ano_reference_overlap();
            meta_lowdin_orthogonal_ao =
                lmp2_1m2m::localization::build_meta_lowdin_orthogonal_ao(
                    molecule, provider.basis_metadata(), rhf.overlap,
                    reference.source_basis,
                    reference.target_source_overlap);
          } catch(const std::exception& error) {
            construction_status = 1;
            construction_error = error.what();
          }
        }
#ifdef LMP2_1M2M_HAS_MPI
        if(ranks > 1 &&
           MPI_Bcast(&construction_status, 1, MPI_INT, 0,
                     MPI_COMM_WORLD) != MPI_SUCCESS) {
          throw std::runtime_error(
              "unable to broadcast meta-Lowdin construction status");
        }
#endif
        if(construction_status != 0) {
          throw std::runtime_error(
              rank == 0
                  ? construction_error
                  : "rank zero failed to construct the meta-Lowdin AO basis");
        }
        if(rank != 0) {
          meta_lowdin_orthogonal_ao.emplace();
        }
        broadcast_matrix_from_rank_zero(
            *meta_lowdin_orthogonal_ao, rank, ranks);
        if(rank == 0) {
          std::cerr << "progress ANO meta-Lowdin AO construction complete; "
                       "broadcast to MPI ranks\n";
        }
      }
      occupied_localized_storage = localize_space(
          localizer, molecule, provider.basis_metadata(), rhf.overlap,
          rhf.coefficients, active_occupied, moments,
          meta_lowdin_orthogonal_ao
              ? &*meta_lowdin_orthogonal_ao
              : nullptr,
          rank, ranks, 200,
          pm_diagnostic_prefix.has_value() ||
              localization_checkpoint.has_value(),
          "occupied");
    }
    const LocalizedSpace& occupied_localized =
        *occupied_localized_storage;
    const std::string_view assignment_metric =
        localizer == "pm" || localizer == "pm-lowdin"
            ? "assigned-monomer population"
            : "oxygen-separation confidence";
    const auto occupied_assignment_quality = require_confident_localization(
        occupied_localized.assignments, confidence, "occupied",
        assignment_metric, rank);
    if(rank == 0) {
      std::cerr << std::setprecision(17)
                << "progress occupied monomer-assignment gate passed; "
                   "minimum "
                << assignment_metric << '='
                << occupied_assignment_quality.minimum_confidence << '\n';
    }
    if(!localization_checkpoint_restored) {
      if(rank == 0) {
        std::cerr << "progress full virtual " << localizer
                  << " localization started; MPI ranks=" << ranks
                  << "; atom ownership=cyclic\n";
      }
      virtual_localized_storage = localize_space(
          localizer, molecule, provider.basis_metadata(), rhf.overlap,
          rhf.coefficients, virtuals, moments,
          meta_lowdin_orthogonal_ao
              ? &*meta_lowdin_orthogonal_ao
              : nullptr,
          rank, ranks, 300,
          pm_diagnostic_prefix.has_value() ||
              localization_checkpoint.has_value(),
          "virtual");
    }
    const LocalizedSpace& virtual_localized = *virtual_localized_storage;
    const auto virtual_assignment_quality = require_confident_localization(
        virtual_localized.assignments, confidence, "virtual",
        assignment_metric, rank);
    if(rank == 0) {
      std::cerr << std::setprecision(17)
                << "progress virtual monomer-assignment gate passed; "
                   "minimum "
                << assignment_metric << '='
                << virtual_assignment_quality.minimum_confidence << '\n';
    }
    if(!localization_checkpoint_restored && localization_checkpoint) {
#ifdef LMP2_1M2M_HAS_HDF5
      int write_status = 0;
      std::string write_error;
      if(rank == 0) {
        try {
          if(threads > static_cast<std::size_t>(
                           std::numeric_limits<int>::max())) {
            throw std::overflow_error(
                "localization checkpoint OpenMP thread count exceeds int");
          }
          if(!occupied_localized.pipek_mezey_diagnostics ||
             !virtual_localized.pipek_mezey_diagnostics) {
            throw std::logic_error(
                "PM localization records were not retained");
          }
          lmp2_1m2m::io::write_rhf_checkpoint(
              *localization_checkpoint, molecule,
              provider.basis_metadata(), rhf, true,
              lmp2_1m2m::io::CheckpointRuntime{
                  .mpi_size = ranks,
                  .openmp_threads = static_cast<int>(threads),
              },
              false, &*occupied_localized.pipek_mezey_diagnostics,
              &*virtual_localized.pipek_mezey_diagnostics);
          std::cerr << "progress occupied and virtual localization "
                       "checkpoint written; path="
                    << localization_checkpoint->string() << '\n';
        } catch(const std::exception& error) {
          write_status = 1;
          write_error = error.what();
        }
      }
#ifdef LMP2_1M2M_HAS_MPI
      if(MPI_Bcast(&write_status, 1, MPI_INT, 0, MPI_COMM_WORLD) !=
         MPI_SUCCESS) {
        throw std::runtime_error(
            "unable to broadcast localization checkpoint status");
      }
#endif
      if(write_status != 0) {
        throw std::runtime_error(
            rank == 0
                ? write_error
                : "rank zero failed to write localization checkpoint");
      }
#else
      throw std::runtime_error(
          "localization checkpoint requires an HDF5-enabled build");
#endif
    }
    if(pm_diagnostic_prefix) {
      write_pipek_mezey_diagnostics_collectively(
          occupied_localized, active_occupied, virtual_localized, virtuals,
          molecule, *pm_diagnostic_prefix, rank);
    }
    const double localization_seconds =
        maximum_rank_time(seconds_since(localization_start));

    if(rank == 0) {
      std::cerr << "progress rank-zero "
                << quadrature_scheme_name(quadrature_scheme)
                << " quadrature construction started\n";
    }
    const auto fit_start = std::chrono::steady_clock::now();
    const auto fit = lmp2_1m2m::mp2::build_and_fit_laplace_quadrature(
        rhf.orbital_energies, active_occupied, virtuals,
        lmp2_1m2m::mp2::LaplaceFitOptions{
            .histogram_bins = 1200,
            .number_of_points = points,
            .gap_block_size = 256,
            .validation_grid_size = 4800,
            .maximum_optimizer_iterations =
                std::max<std::size_t>(1000, 200 * points),
            .threads = threads,
            .mpi_rank = static_cast<std::size_t>(rank),
            .mpi_ranks = static_cast<std::size_t>(ranks),
            .logarithmic_node_tolerance = 1.0e-10,
            .objective_tolerance = 1.0e-14,
            .scheme = quadrature_scheme,
        });
    const double fit_seconds = maximum_rank_time(seconds_since(fit_start));

    if(rank == 0) {
      std::cerr << "progress localized Laplace energy started\n";
      if(mode == lmp2_1m2m::mp2::NmExecutionMode::one_two_monomer) {
        std::cerr
            << "notice 1m2m is an approximate truncated energy; "
               "3M and 4M excitation blocks are omitted before localized "
               "pair materialization and energy accumulation\n";
      }
    }
    const auto energy_start = std::chrono::steady_clock::now();
    const auto energy =
        lmp2_1m2m::mp2::compute_localized_tiled_laplace_mp2(
            provider, rhf.coefficients, rhf.orbital_energies,
            active_occupied, virtuals, occupied_localized.rotation,
            virtual_localized.rotation, occupied_localized.assignments,
            virtual_localized.assignments, fit,
            lmp2_1m2m::mp2::LocalizedTiledLaplaceMp2Options{
                .occupied_tile_size = occupied_tile,
                .virtual_tile_size = virtual_tile,
                .quadrature_batch_size = quadrature_batch,
                .lambda_ao_block_size = lambda_ao_block,
                .maximum_additional_memory_bytes = memory,
                .threads = threads,
                .mpi_rank = static_cast<std::size_t>(rank),
                .mpi_ranks = static_cast<std::size_t>(ranks),
                .eri_schwarz_threshold = schwarz,
                .localized_energy_schwarz_threshold_hartree =
                    localized_energy_schwarz,
                .localized_point_energy_threshold_hartree =
                    localized_point_energy_threshold,
                .report_laplace_point_progress = true,
                .assignment_confidence_threshold = confidence,
                .execution_mode = mode,
                .backend = requested_backend,
            });
    const double energy_seconds =
        maximum_rank_time(seconds_since(energy_start));
    const double total_seconds =
        maximum_rank_time(seconds_since(total_start));

    if(rank == 0) {
      const double canonical_storage_replication_factor =
          energy.canonical_ovov_bytes == 0
              ? 0.0
              : static_cast<double>(
                    energy.canonical_ovov_stored_bytes_global) /
                    static_cast<double>(energy.canonical_ovov_bytes);
      const double localized_storage_replication_factor =
          energy.localized_ovov_bytes == 0
              ? 0.0
              : static_cast<double>(
                    energy.localized_ovov_stored_bytes_global) /
                    static_cast<double>(energy.localized_ovov_bytes);
      std::cerr << std::setprecision(17)
                << "progress canonical OVOV ownership audit; logical_bytes="
                << energy.canonical_ovov_bytes
                << "; maximum_local_bytes="
                << energy.canonical_ovov_max_local_bytes
                << "; globally_stored_bytes="
                << energy.canonical_ovov_stored_bytes_global
                << "; replication_factor="
                << canonical_storage_replication_factor << '\n';
      std::cerr << std::setprecision(17)
                << "progress localized OVOV ownership audit; logical_bytes="
                << energy.localized_ovov_bytes
                << "; maximum_local_bytes="
                << energy.localized_ovov_max_local_bytes
                << "; globally_stored_bytes="
                << energy.localized_ovov_stored_bytes_global
                << "; replication_factor="
                << localized_storage_replication_factor << '\n';
      std::cout
          << "geometry,basis,waters,aos,active_occ,virtual,localizer,"
             "localization_optimizer,"
             "localization_parallel_strategy,"
             "localization_checkpoint_restored,localization_checkpoint,"
             "occupied_localization_iterations,"
             "virtual_localization_iterations,assignment_scheme,"
             "assignment_confidence_definition,occupied_min_confidence,"
             "virtual_min_confidence,energy_scope,points,"
             "quadrature_scheme,schwarz_threshold,mode,energy_backend,"
             "localized_energy_schwarz_threshold_hartree,"
             "localized_point_energy_threshold_hartree,"
             "cached_rotation_kernel,"
             "ranks,threads,"
             "occupied_tile,virtual_tile,quadrature_batch,"
             "lambda_ao_block,"
             "rhf_seconds,rhf_checkpoint_restored,"
             "rhf_iterations_this_run,rhf_guess,rhf_checkpoint_output,"
             "rhf_initial_guess_checkpoint,rhf_initial_guess_basis,"
             "rhf_projection_target_metric_rank,"
             "rhf_projection_minimum_occupied_gram_eigenvalue,"
             "rhf_projection_occupied_orthonormality_maximum_error,"
             "rhf_projection_electron_count,"
             "rhf_iterations,rhf_final_energy_change_hartree,"
             "rhf_final_density_rms,rhf_final_commutator_rms,"
             "rhf_removed_overlap_vectors,rhf_eri_schwarz_threshold,"
             "rhf_density_screen,rhf_fock_contribution_threshold,"
             "rhf_unique_shell_quartets_per_fock,"
             "rhf_evaluated_shell_quartets_total,"
             "rhf_screened_shell_quartets_total,"
             "rhf_schwarz_screened_shell_quartets_total,"
             "rhf_density_screened_shell_quartets_total,"
             "rhf_final_screened_fock_maximum_error_bound,"
             "rhf_direct_fock_wall_seconds_total,"
             "rhf_minimum_direct_fock_wall_seconds_last,"
             "rhf_maximum_direct_fock_wall_seconds_last,"
             "localization_seconds,fit_seconds,lmp2_1m2m_seconds,"
             "total_seconds,rhf_energy_hartree,lmp2_1m2m_1m_hartree,"
             "lmp2_1m2m_2m_hartree,lmp2_1m2m_3m_hartree,lmp2_1m2m_4m_hartree,"
             "lmp2_1m2m_correlation_hartree,total_mp2_hartree,"
             "fit_max_abs_inverse_hartree,"
             "computed_terms,omitted_terms,computed_point_tasks,"
             "omitted_point_tasks,direct_transforms,exchange_transforms,"
             "reused_exchange_tiles,screened_shell_quartets,"
             "evaluated_shell_quartets,low_confidence_occ,"
             "low_confidence_vir,peak_additional_bytes,"
             "localized_screened_terms,localized_screened_block_tasks,"
             "localized_screening_omitted_bound_hartree,"
             "localized_screened_point_contributions,"
             "localized_point_screening_omitted_bound_hartree,"
             "canonical_ovov_bytes,canonical_ovov_max_local_bytes,"
             "canonical_ovov_stored_bytes_global,"
             "canonical_ovov_storage_replication_factor,"
             "localized_ovov_bytes,localized_ovov_max_local_bytes,"
             "localized_ovov_stored_bytes_global,"
             "localized_ovov_storage_replication_factor,"
             "canonical_ovov_build_seconds,"
             "canonical_ovov_broadcast_seconds,"
             "localized_ovov_build_seconds,"
             "direct_ovov_transform_seconds,"
             "localized_rotation_seconds,pair_transpose_seconds,"
             "energy_accumulation_seconds,"
             "distributed_communication_seconds,"
             "localized_screening_seconds,"
             "resource_policy,resource_memory_source,resource_nodes,"
             "resource_ranks_per_node,resource_usable_memory_bytes_per_rank,"
             "resource_estimated_transform_peak_bytes_per_rank,"
             "resource_estimated_total_peak_bytes_per_rank,"
             "resource_distributed_transform_occupied_count,"
             "resource_occupied_batches,resource_virtual_batches,"
             "localization_population_definition\n"
          << std::setprecision(17) << geometry.filename().string() << ','
          << basis_name << ',' << atoms.size() / 3 << ','
          << provider.basis_metadata().number_of_aos << ','
          << active_occupied.size() << ',' << virtuals.size() << ','
          << localizer << ','
          << (localizer == "pm" || localizer == "pm-lowdin"
                  ? "pyscf_2_11_ciah"
                  : "fourth_moment_trust_region")
          << ','
          << (localizer == "pm" || localizer == "pm-lowdin"
                  ? "mpi_cyclic_atoms_root_global_matrix_broadcast_"
                    "openmp_row_blocked_blas"
                  : "mpi_distributed_orbitals")
          << ',' << (localization_checkpoint_restored ? 1 : 0) << ','
          << (localization_checkpoint ? localization_checkpoint->string()
                                      : "none")
          << ',' << occupied_localized.iterations << ','
          << virtual_localized.iterations << ','
          << (localizer == "pm" || localizer == "pm-lowdin"
                  ? "lowdin_atom_then_monomer"
                  : "nearest_oxygen_centroid_to_monomer")
          << ','
          << (localizer == "pm" || localizer == "pm-lowdin"
                  ? "assigned_monomer_lowdin_population"
                  : "one_minus_nearest_over_second_oxygen_distance")
          << ',' << occupied_assignment_quality.minimum_confidence << ','
          << virtual_assignment_quality.minimum_confidence << ','
          << (mode == lmp2_1m2m::mp2::NmExecutionMode::full
                  ? "full_1m_through_4m"
                  : "truncated_1m2m")
          << ',' << points << ',' << quadrature_scheme_name(quadrature_scheme)
          << ','
          << schwarz << ',' << mode_name << ',' << backend_name(energy.backend)
          << ',' << localized_energy_schwarz << ','
          << localized_point_energy_threshold << ','
          << energy_kernel_name(energy.backend)
          << ',' << ranks
          << ',' << threads << ',' << occupied_tile << ',' << virtual_tile
          << ',' << quadrature_batch << ',' << lambda_ao_block
          << ',' << rhf_seconds << ',' << (rhf_checkpoint_restored ? 1 : 0)
          << ',' << (rhf_checkpoint_restored ? 0 : rhf.iteration_count)
          << ',' << rhf_guess_name << ','
          << (rhf_checkpoint_output
                  ? rhf_checkpoint_output->string()
                  : "none")
          << ','
          << (rhf_initial_guess_checkpoint
                  ? rhf_initial_guess_checkpoint->string()
                  : "none")
          << ',' << rhf_initial_guess_basis << ','
          << (projected_rhf_guess
                  ? projected_rhf_guess->target_metric_rank
                  : 0)
          << ','
          << (projected_rhf_guess
                  ? projected_rhf_guess
                        ->minimum_occupied_gram_eigenvalue
                  : 0.0)
          << ','
          << (projected_rhf_guess
                  ? projected_rhf_guess
                        ->occupied_orthonormality_maximum_error
                  : 0.0)
          << ','
          << (projected_rhf_guess ? projected_rhf_guess->electron_count
                                  : 0.0)
          << ',' << rhf.iteration_count << ','
          << rhf.final_energy_change << ',' << rhf.final_density_rms << ','
          << rhf.final_commutator_rms << ','
          << rhf.removed_overlap_vectors << ','
          << rhf.eri_schwarz_threshold << ','
          << (rhf.density_screen ? 1 : 0) << ','
          << rhf.fock_contribution_threshold << ','
          << rhf.unique_shell_quartets_per_fock << ','
          << rhf.evaluated_shell_quartets_total << ','
          << rhf.screened_shell_quartets_total << ','
          << rhf.schwarz_screened_shell_quartets_total << ','
          << rhf.density_screened_shell_quartets_total << ','
          << rhf.final_screened_fock_maximum_error_bound << ','
          << rhf.direct_fock_wall_seconds_total << ','
          << rhf.minimum_direct_fock_wall_seconds_last << ','
          << rhf.maximum_direct_fock_wall_seconds_last << ','
          << localization_seconds << ','
          << fit_seconds << ',' << energy_seconds << ',' << total_seconds
          << ',' << rhf.total_energy << ','
          << energy.decomposition.energy_hartree[1] << ','
          << energy.decomposition.energy_hartree[2] << ','
          << energy.decomposition.energy_hartree[3] << ','
          << energy.decomposition.energy_hartree[4] << ','
          << energy.correlation_energy_hartree << ','
          << rhf.total_energy + energy.correlation_energy_hartree << ','
          << fit.validation_maximum_absolute_error << ','
          << energy.computed_excitation_terms << ','
          << energy.omitted_excitation_terms << ','
          << energy.computed_point_tasks << ','
          << energy.omitted_point_tasks << ','
          << energy.direct_transform_count << ','
          << energy.exchange_transform_count << ','
          << energy.reused_exchange_tile_count << ','
          << energy.schwarz_screened_shell_quartets << ','
          << energy.evaluated_shell_quartets << ','
          << energy.low_confidence_occupied_orbitals << ','
          << energy.low_confidence_virtual_orbitals << ','
          << energy.estimated_peak_additional_bytes << ','
          << energy.localized_screened_excitation_terms << ','
          << energy.localized_screened_block_tasks << ','
          << energy.localized_screening_omitted_energy_bound_hartree << ','
          << energy.localized_screened_point_contributions << ','
          << energy.localized_point_screening_omitted_energy_bound_hartree
          << ','
          << energy.canonical_ovov_bytes << ','
          << energy.canonical_ovov_max_local_bytes << ','
          << energy.canonical_ovov_stored_bytes_global << ','
          << canonical_storage_replication_factor << ','
          << energy.localized_ovov_bytes << ','
          << energy.localized_ovov_max_local_bytes << ','
          << energy.localized_ovov_stored_bytes_global << ','
          << localized_storage_replication_factor << ','
          << energy.canonical_ovov_build_seconds << ','
          << energy.canonical_ovov_broadcast_seconds << ','
          << energy.localized_ovov_build_seconds << ','
          << energy.direct_ovov_transform_seconds << ','
          << energy.localized_rotation_seconds << ','
          << energy.pair_transpose_seconds << ','
          << energy.energy_accumulation_seconds << ','
          << energy.distributed_communication_seconds << ','
          << energy.localized_screening_seconds << ','
          << (smart_resources ? "smart" : "manual") << ','
          << memory_source << ','
          << (resource_plan ? resource_plan->nodes : 0U) << ','
          << ranks_per_node << ',' << memory << ','
          << (resource_plan
                  ? resource_plan
                        ->estimated_transform_peak_bytes_per_rank
                  : 0U)
          << ','
          << (resource_plan
                  ? resource_plan->estimated_total_peak_bytes_per_rank
                  : 0U)
          << ','
          << (resource_plan
                  ? resource_plan
                        ->distributed_transform_occupied_count
                  : 0U)
          << ','
          << (resource_plan ? resource_plan->occupied_batches : 0U) << ','
          << (resource_plan ? resource_plan->virtual_batches : 0U) << ','
          << (localizer == "pm"
                  ? "ano_preorthogonalized_meta_lowdin"
                  : localizer == "pm-lowdin"
                        ? "symmetric_lowdin"
                        : "fourth_central_moment")
          << '\n';
    }
  } catch(const CollectiveAssignmentGateFailure&) {
    local_status = 1;
  } catch(const std::exception& error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n';
    local_status = 1;
  }
#ifdef LMP2_1M2M_HAS_MPI
  int global_status = 0;
  MPI_Allreduce(&local_status, &global_status, 1, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);
  MPI_Finalize();
  return global_status;
#else
  return local_status;
#endif
}
