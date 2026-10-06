#include "lmp2_1m2m/mp2/tiled_laplace_mp2.hpp"

#include "lmp2_1m2m/integrals/integral_provider.hpp"
#include "lmp2_1m2m/linalg/matrix.hpp"

#ifdef LMP2_1M2M_HAS_MPI
#include <mpi.h>
#endif

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lmp2_1m2m::mp2 {
namespace {

constexpr std::uint64_t fnv_offset = UINT64_C(1469598103934665603);
constexpr std::uint64_t fnv_prime = UINT64_C(1099511628211);

[[nodiscard]] std::size_t checked_product(
    std::initializer_list<std::size_t> factors, const char* name) {
  std::size_t result = 1;
  for(const std::size_t factor : factors) {
    if(factor == 0 ||
       result > std::numeric_limits<std::size_t>::max() / factor) {
      throw std::overflow_error(std::string{name} +
                                " element count overflows size_t");
    }
    result *= factor;
  }
  return result;
}

[[nodiscard]] std::size_t checked_add(std::size_t left,
                                      std::size_t right,
                                      const char* name) {
  if(left > std::numeric_limits<std::size_t>::max() - right) {
    throw std::overflow_error(std::string{name} +
                              " byte count overflows size_t");
  }
  return left + right;
}

[[nodiscard]] std::size_t checked_bytes(std::size_t count,
                                        std::size_t element_size,
                                        const char* name) {
  if(count > std::numeric_limits<std::size_t>::max() / element_size) {
    throw std::overflow_error(std::string{name} +
                              " byte count overflows size_t");
  }
  return count * element_size;
}

void hash_bytes(std::uint64_t& hash, const void* data,
                std::size_t byte_count) {
  const auto* bytes = static_cast<const unsigned char*>(data);
  for(std::size_t index = 0; index < byte_count; ++index) {
    hash ^= static_cast<std::uint64_t>(bytes[index]);
    hash *= fnv_prime;
  }
}

template <typename T>
void hash_value(std::uint64_t& hash, const T& value) {
  hash_bytes(hash, &value, sizeof(T));
}

void hash_double(std::uint64_t& hash, double value) {
  const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
  hash_value(hash, bits);
}

[[nodiscard]] std::uint64_t calculation_fingerprint(
    const integrals::Libint2IntegralProvider& provider,
    const linalg::Matrix& coefficients,
    const std::vector<double>& energies,
    const std::vector<std::size_t>& occupied,
    const std::vector<std::size_t>& virtuals,
    const LaplaceFitResult& fit,
    const TiledLaplaceMp2Options& options) {
  std::uint64_t hash = fnv_offset;
  constexpr std::uint64_t format_version = 1;
  hash_value(hash, format_version);
  const auto& basis = provider.basis_metadata();
  hash_bytes(hash, basis.canonical_name.data(),
             basis.canonical_name.size());
  hash_value(hash, basis.spherical);
  hash_value(hash, basis.number_of_aos);
  hash_value(hash, coefficients.rows());
  hash_value(hash, coefficients.columns());
  for(const double value : coefficients.values()) {
    hash_double(hash, value);
  }
  for(const double value : energies) {
    hash_double(hash, value);
  }
  for(const std::size_t value : occupied) {
    hash_value(hash, value);
  }
  for(const std::size_t value : virtuals) {
    hash_value(hash, value);
  }
  for(const double value : fit.nodes) {
    hash_double(hash, value);
  }
  for(const double value : fit.weights) {
    hash_double(hash, value);
  }
  hash_value(hash, options.occupied_tile_size);
  hash_value(hash, options.virtual_tile_size);
  hash_double(hash, options.minimum_positive_denominator);
  hash_double(hash, options.eri_schwarz_threshold);
  return hash;
}

void validate_indices(const std::vector<std::size_t>& indices,
                      std::size_t orbital_count, const char* name,
                      std::set<std::size_t>& used) {
  if(indices.empty()) {
    throw std::invalid_argument(std::string{name} +
                                " orbital space may not be empty");
  }
  for(const std::size_t index : indices) {
    if(index >= orbital_count) {
      throw std::invalid_argument(std::string{name} +
                                  " orbital index is out of range");
    }
    if(!used.insert(index).second) {
      throw std::invalid_argument(
          "tiled Laplace MP2 orbital indices overlap or repeat");
    }
  }
}

void validate_topology(const TiledLaplaceMp2Options& options) {
  if(options.mpi_ranks == 0 || options.mpi_rank >= options.mpi_ranks) {
    throw std::invalid_argument(
        "tiled Laplace MP2 MPI topology is invalid");
  }
#ifdef LMP2_1M2M_HAS_MPI
  if(options.mpi_ranks > 1) {
    int initialized = 0;
    if(MPI_Initialized(&initialized) != MPI_SUCCESS ||
       initialized == 0) {
      throw std::runtime_error(
          "multi-rank tiled Laplace MP2 requires initialized MPI");
    }
    int rank = 0;
    int ranks = 1;
    if(MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS ||
       MPI_Comm_size(MPI_COMM_WORLD, &ranks) != MPI_SUCCESS) {
      throw std::runtime_error(
          "unable to query MPI_COMM_WORLD for tiled Laplace MP2");
    }
    if(rank < 0 || ranks <= 0 ||
       static_cast<std::size_t>(rank) != options.mpi_rank ||
       static_cast<std::size_t>(ranks) != options.mpi_ranks) {
      throw std::invalid_argument(
          "tiled Laplace MP2 topology does not match MPI_COMM_WORLD");
    }
  }
#else
  if(options.mpi_rank != 0 || options.mpi_ranks != 1) {
    throw std::invalid_argument(
        "multi-rank tiled Laplace MP2 requires MPI support");
  }
#endif
}

#ifdef LMP2_1M2M_HAS_MPI
void check_mpi(int status, const char* operation) {
  if(status != MPI_SUCCESS) {
    throw std::runtime_error(std::string{"MPI failure while "} + operation);
  }
}

void propagate_failure(bool local_failure, const char* operation) {
  int failed = local_failure ? 1 : 0;
  check_mpi(MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX,
                          MPI_COMM_WORLD),
            operation);
  if(failed != 0) {
    throw std::runtime_error(
        local_failure
            ? "tiled Laplace MP2 failed on this MPI rank"
            : "tiled Laplace MP2 failed on another MPI rank");
  }
}

void allreduce_sum(std::vector<double>& values) {
  std::size_t offset = 0;
  while(offset < values.size()) {
    const int count = static_cast<int>(std::min(
        values.size() - offset,
        static_cast<std::size_t>(std::numeric_limits<int>::max())));
    check_mpi(MPI_Allreduce(MPI_IN_PLACE, values.data() + offset, count,
                            MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD),
              "reducing tiled Laplace MP2 energy contributions");
    offset += static_cast<std::size_t>(count);
  }
}

[[nodiscard]] std::pair<std::uint64_t, std::uint64_t>
reduce_work_range(std::uint64_t local_count) {
  unsigned long long minimum =
      static_cast<unsigned long long>(local_count);
  unsigned long long maximum = minimum;
  check_mpi(MPI_Allreduce(MPI_IN_PLACE, &minimum, 1,
                          MPI_UNSIGNED_LONG_LONG, MPI_MIN,
                          MPI_COMM_WORLD),
            "reducing minimum tiled Laplace MP2 workload");
  check_mpi(MPI_Allreduce(MPI_IN_PLACE, &maximum, 1,
                          MPI_UNSIGNED_LONG_LONG, MPI_MAX,
                          MPI_COMM_WORLD),
            "reducing maximum tiled Laplace MP2 workload");
  return {static_cast<std::uint64_t>(minimum),
          static_cast<std::uint64_t>(maximum)};
}

void require_identical_hash(std::uint64_t hash, const char* name) {
  unsigned long long minimum =
      static_cast<unsigned long long>(hash);
  unsigned long long maximum = minimum;
  check_mpi(MPI_Allreduce(MPI_IN_PLACE, &minimum, 1,
                          MPI_UNSIGNED_LONG_LONG, MPI_MIN,
                          MPI_COMM_WORLD),
            "reducing tiled Laplace MP2 hash minimum");
  check_mpi(MPI_Allreduce(MPI_IN_PLACE, &maximum, 1,
                          MPI_UNSIGNED_LONG_LONG, MPI_MAX,
                          MPI_COMM_WORLD),
            "reducing tiled Laplace MP2 hash maximum");
  if(minimum != maximum) {
    throw std::runtime_error(
        std::string{"tiled Laplace MP2 "} + name +
        " differs across MPI ranks");
  }
}

void reduce_transform_diagnostics(
    std::uint64_t& direct, std::uint64_t& exchange,
    std::uint64_t& reused, std::uint64_t& quartets,
    std::uint64_t& evaluated, std::uint64_t& screened,
    std::size_t& peak_bytes) {
  std::array<unsigned long long, 5> counts{
      static_cast<unsigned long long>(direct),
      static_cast<unsigned long long>(exchange),
      static_cast<unsigned long long>(reused),
      static_cast<unsigned long long>(evaluated),
      static_cast<unsigned long long>(screened),
  };
  check_mpi(MPI_Allreduce(MPI_IN_PLACE, counts.data(),
                          static_cast<int>(counts.size()),
                          MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                          MPI_COMM_WORLD),
            "reducing tiled Laplace MP2 transform counts");
  unsigned long long maximum_quartets =
      static_cast<unsigned long long>(quartets);
  unsigned long long maximum_peak =
      static_cast<unsigned long long>(peak_bytes);
  check_mpi(MPI_Allreduce(MPI_IN_PLACE, &maximum_quartets, 1,
                          MPI_UNSIGNED_LONG_LONG, MPI_MAX,
                          MPI_COMM_WORLD),
            "reducing tiled Laplace MP2 shell-quartet count");
  check_mpi(MPI_Allreduce(MPI_IN_PLACE, &maximum_peak, 1,
                          MPI_UNSIGNED_LONG_LONG, MPI_MAX,
                          MPI_COMM_WORLD),
            "reducing tiled Laplace MP2 peak memory");
  direct = static_cast<std::uint64_t>(counts[0]);
  exchange = static_cast<std::uint64_t>(counts[1]);
  reused = static_cast<std::uint64_t>(counts[2]);
  evaluated = static_cast<std::uint64_t>(counts[3]);
  screened = static_cast<std::uint64_t>(counts[4]);
  quartets = static_cast<std::uint64_t>(maximum_quartets);
  if(maximum_peak >
     static_cast<unsigned long long>(
         std::numeric_limits<std::size_t>::max())) {
    throw std::overflow_error(
        "reduced tiled Laplace MP2 peak memory exceeds size_t");
  }
  peak_bytes = static_cast<std::size_t>(maximum_peak);
}
#endif

struct Tile {
  std::size_t occupied_1_first;
  std::size_t occupied_1_count;
  std::size_t occupied_2_first;
  std::size_t occupied_2_count;
  std::size_t virtual_1_first;
  std::size_t virtual_1_count;
  std::size_t virtual_2_first;
  std::size_t virtual_2_count;
};

[[nodiscard]] Tile decode_tile(std::size_t ordinal,
                               std::size_t occupied_count,
                               std::size_t virtual_count,
                               std::size_t occupied_tile_size,
                               std::size_t virtual_tile_size) {
  const std::size_t occupied_tiles =
      (occupied_count + occupied_tile_size - 1) / occupied_tile_size;
  const std::size_t virtual_tiles =
      (virtual_count + virtual_tile_size - 1) / virtual_tile_size;
  const std::size_t virtual_2_tile = ordinal % virtual_tiles;
  ordinal /= virtual_tiles;
  const std::size_t virtual_1_tile = ordinal % virtual_tiles;
  ordinal /= virtual_tiles;
  const std::size_t occupied_2_tile = ordinal % occupied_tiles;
  const std::size_t occupied_1_tile = ordinal / occupied_tiles;
  const std::size_t occupied_1_first =
      occupied_1_tile * occupied_tile_size;
  const std::size_t occupied_2_first =
      occupied_2_tile * occupied_tile_size;
  const std::size_t virtual_1_first =
      virtual_1_tile * virtual_tile_size;
  const std::size_t virtual_2_first =
      virtual_2_tile * virtual_tile_size;
  return Tile{
      .occupied_1_first = occupied_1_first,
      .occupied_1_count =
          std::min(occupied_tile_size,
                   occupied_count - occupied_1_first),
      .occupied_2_first = occupied_2_first,
      .occupied_2_count =
          std::min(occupied_tile_size,
                   occupied_count - occupied_2_first),
      .virtual_1_first = virtual_1_first,
      .virtual_1_count =
          std::min(virtual_tile_size,
                   virtual_count - virtual_1_first),
      .virtual_2_first = virtual_2_first,
      .virtual_2_count =
          std::min(virtual_tile_size,
                   virtual_count - virtual_2_first),
  };
}

[[nodiscard]] linalg::Matrix select_columns(
    const linalg::Matrix& coefficients,
    const std::vector<std::size_t>& indices) {
  linalg::Matrix selected{coefficients.rows(), indices.size()};
  for(std::size_t column = 0; column < indices.size(); ++column) {
    for(std::size_t row = 0; row < coefficients.rows(); ++row) {
      selected(row, column) = coefficients(row, indices[column]);
    }
  }
  return selected;
}

}  // namespace

TiledLaplaceMp2Result compute_tiled_laplace_mp2(
    const integrals::Libint2IntegralProvider& provider,
    const linalg::Matrix& canonical_coefficients,
    const std::vector<double>& orbital_energies,
    const std::vector<std::size_t>& active_occupied_indices,
    const std::vector<std::size_t>& virtual_indices,
    const LaplaceFitResult& fit,
    const TiledLaplaceMp2Options& options,
    const TiledLaplaceMp2Progress* restart,
    const TiledLaplaceMp2Observer& observer) {
  if(options.occupied_tile_size == 0 ||
     options.virtual_tile_size == 0 ||
     options.maximum_additional_memory_bytes == 0 ||
     options.checkpoint_interval_tiles == 0 ||
     options.threads == 0 ||
     !std::isfinite(options.minimum_positive_denominator) ||
     options.minimum_positive_denominator <= 0.0 ||
     !std::isfinite(options.eri_schwarz_threshold) ||
     options.eri_schwarz_threshold < 0.0) {
    throw std::invalid_argument(
        "tiled Laplace MP2 options are invalid");
  }
  validate_topology(options);
  const std::size_t ao_count =
      provider.basis_metadata().number_of_aos;
  const std::size_t orbital_count = canonical_coefficients.columns();
  if(canonical_coefficients.rows() != ao_count ||
     orbital_count == 0 ||
     orbital_energies.size() != orbital_count) {
    throw std::invalid_argument(
        "tiled Laplace MP2 coefficient/energy dimensions are incompatible");
  }
  linalg::require_finite(canonical_coefficients,
                         "tiled Laplace MP2 coefficients");
  for(const double energy : orbital_energies) {
    if(!std::isfinite(energy)) {
      throw std::invalid_argument(
          "tiled Laplace MP2 orbital energy is nonfinite");
    }
  }
  std::set<std::size_t> used_indices;
  validate_indices(active_occupied_indices, orbital_count,
                   "active occupied", used_indices);
  validate_indices(virtual_indices, orbital_count, "virtual",
                   used_indices);
  if(fit.nodes.empty() || fit.nodes.size() != fit.weights.size()) {
    throw std::invalid_argument(
        "tiled Laplace MP2 quadrature is empty or inconsistent");
  }
  for(std::size_t point = 0; point < fit.nodes.size(); ++point) {
    if(!std::isfinite(fit.nodes[point]) || fit.nodes[point] <= 0.0 ||
       !std::isfinite(fit.weights[point])) {
      throw std::invalid_argument(
          "tiled Laplace MP2 quadrature contains an invalid node or weight");
    }
  }

  double minimum_denominator =
      std::numeric_limits<double>::infinity();
  double maximum_denominator = 0.0;
  for(const std::size_t i : active_occupied_indices) {
    for(const std::size_t j : active_occupied_indices) {
      for(const std::size_t a : virtual_indices) {
        for(const std::size_t b : virtual_indices) {
          const double denominator =
              orbital_energies[a] + orbital_energies[b] -
              orbital_energies[i] - orbital_energies[j];
          if(!std::isfinite(denominator) ||
             denominator <= options.minimum_positive_denominator) {
            throw std::runtime_error(
                "tiled Laplace MP2 denominator is nonpositive or below "
                "the configured minimum");
          }
          minimum_denominator =
              std::min(minimum_denominator, denominator);
          maximum_denominator =
              std::max(maximum_denominator, denominator);
        }
      }
    }
  }
  const double interval_tolerance =
      128.0 * std::numeric_limits<double>::epsilon() *
      std::max(1.0, maximum_denominator);
  if(std::abs(fit.histogram.denominator_minimum -
              minimum_denominator) > interval_tolerance ||
     std::abs(fit.histogram.denominator_maximum -
              maximum_denominator) > interval_tolerance) {
    throw std::invalid_argument(
        "Laplace fit denominator interval does not match the tiled orbital "
        "spaces");
  }

  const std::size_t occupied_count =
      active_occupied_indices.size();
  const std::size_t virtual_count = virtual_indices.size();
  const std::size_t occupied_tiles =
      (occupied_count + options.occupied_tile_size - 1) /
      options.occupied_tile_size;
  const std::size_t virtual_tiles =
      (virtual_count + options.virtual_tile_size - 1) /
      options.virtual_tile_size;
  const std::size_t total_tiles = checked_product(
      {occupied_tiles, occupied_tiles, virtual_tiles, virtual_tiles},
      "tiled Laplace MP2 tile space");
  const std::uint64_t fingerprint =
      calculation_fingerprint(provider, canonical_coefficients,
                              orbital_energies,
                              active_occupied_indices, virtual_indices,
                              fit, options);
#ifdef LMP2_1M2M_HAS_MPI
  if(options.mpi_ranks > 1) {
    require_identical_hash(fingerprint, "calculation fingerprint");
  }
#endif
  TiledLaplaceMp2Progress progress{
      .fingerprint = fingerprint,
      .completed = std::vector<std::uint8_t>(total_tiles, 0),
      .tile_energy_contributions =
          std::vector<double>(total_tiles, 0.0),
  };
  if(restart != nullptr) {
    if(restart->fingerprint != fingerprint ||
       restart->completed.size() != total_tiles ||
       restart->tile_energy_contributions.size() != total_tiles) {
      throw std::invalid_argument(
          "tiled Laplace MP2 restart fingerprint or tile dimensions do not "
          "match this calculation");
    }
    progress = *restart;
    for(std::size_t tile = 0; tile < total_tiles; ++tile) {
      if(progress.completed[tile] > 1 ||
         !std::isfinite(progress.tile_energy_contributions[tile]) ||
         (progress.completed[tile] == 0 &&
          progress.tile_energy_contributions[tile] != 0.0)) {
        throw std::invalid_argument(
            "tiled Laplace MP2 restart contains invalid tile state");
      }
    }
  }
#ifdef LMP2_1M2M_HAS_MPI
  if(options.mpi_ranks > 1) {
    std::uint64_t progress_hash = fnv_offset;
    hash_value(progress_hash, progress.fingerprint);
    hash_bytes(progress_hash, progress.completed.data(),
               progress.completed.size());
    for(const double value : progress.tile_energy_contributions) {
      hash_double(progress_hash, value);
    }
    require_identical_hash(progress_hash, "restart state");
  }
#endif
  const std::size_t initially_completed = static_cast<std::size_t>(
      std::count(progress.completed.begin(), progress.completed.end(),
                 static_cast<std::uint8_t>(1)));

  linalg::Matrix occupied_coefficients =
      select_columns(canonical_coefficients,
                     active_occupied_indices);
  linalg::Matrix virtual_coefficients =
      select_columns(canonical_coefficients, virtual_indices);
  std::size_t persistent_bytes = checked_bytes(
      progress.completed.size(), sizeof(std::uint8_t),
      "tiled Laplace MP2 completion mask");
  persistent_bytes = checked_add(
      persistent_bytes,
      checked_bytes(progress.tile_energy_contributions.size(),
                    sizeof(double),
                    "tiled Laplace MP2 energy contributions"),
      "tiled Laplace MP2 persistent storage");
  persistent_bytes = checked_add(
      persistent_bytes,
      checked_bytes(occupied_coefficients.size() +
                        virtual_coefficients.size(),
                    sizeof(double),
                    "tiled Laplace MP2 coefficient subsets"),
      "tiled Laplace MP2 persistent storage");
  persistent_bytes = checked_add(
      persistent_bytes,
      checked_bytes(total_tiles, sizeof(double),
                    "tiled Laplace MP2 reduction batch"),
      "tiled Laplace MP2 persistent storage");
  if(persistent_bytes >=
     options.maximum_additional_memory_bytes) {
    throw std::runtime_error(
        "tiled Laplace MP2 persistent storage exceeds the explicit "
        "additional-memory limit");
  }

  std::size_t peak_bytes = persistent_bytes;
  std::uint64_t local_tiles = 0;
  std::uint64_t direct_transforms = 0;
  std::uint64_t exchange_transforms = 0;
  std::uint64_t reused_exchange_tiles = 0;
  std::uint64_t ordered_shell_quartets = 0;
  std::uint64_t evaluated_shell_quartets = 0;
  std::uint64_t screened_shell_quartets = 0;
  const double minimum_exponent =
      -maximum_denominator *
      *std::max_element(fit.nodes.begin(), fit.nodes.end());
  const double maximum_exponent =
      -minimum_denominator *
      *std::min_element(fit.nodes.begin(), fit.nodes.end());

  for(std::size_t batch_first = 0; batch_first < total_tiles;
      batch_first += options.checkpoint_interval_tiles) {
    const std::size_t batch_last =
        std::min(total_tiles,
                 batch_first + options.checkpoint_interval_tiles);
    std::vector<double> batch_values(total_tiles, 0.0);
    std::exception_ptr local_error;
    for(std::size_t ordinal = batch_first;
        ordinal < batch_last && !local_error; ++ordinal) {
      if(progress.completed[ordinal] != 0 ||
         ordinal % options.mpi_ranks != options.mpi_rank) {
        continue;
      }
      try {
        const Tile tile = decode_tile(
            ordinal, occupied_count, virtual_count,
            options.occupied_tile_size, options.virtual_tile_size);
        const std::size_t transform_budget =
            options.maximum_additional_memory_bytes - persistent_bytes;
        const integrals::OvovTileResult direct =
            provider.compute_ovov_tile(
                occupied_coefficients, virtual_coefficients,
                integrals::OvovTileOptions{
                    .left_occupied_first =
                        tile.occupied_1_first,
                    .left_occupied_count =
                        tile.occupied_1_count,
                    .left_virtual_first =
                        tile.virtual_1_first,
                    .left_virtual_count =
                        tile.virtual_1_count,
                    .right_occupied_first =
                        tile.occupied_2_first,
                    .right_occupied_count =
                        tile.occupied_2_count,
                    .right_virtual_first =
                        tile.virtual_2_first,
                    .right_virtual_count =
                        tile.virtual_2_count,
                    .threads = options.threads,
                    .schwarz_threshold =
                        options.eri_schwarz_threshold,
                    .maximum_additional_memory_bytes =
                        transform_budget,
                });
        ++direct_transforms;
        ordered_shell_quartets = direct.ordered_shell_quartets;
        evaluated_shell_quartets +=
            direct.evaluated_shell_quartets;
        screened_shell_quartets +=
            direct.schwarz_screened_shell_quartets;
        peak_bytes = std::max(
            peak_bytes,
            checked_add(persistent_bytes,
                        direct.estimated_peak_additional_bytes,
                        "tiled Laplace MP2 peak storage"));
        const bool reuse_exchange =
            tile.virtual_1_first == tile.virtual_2_first &&
            tile.virtual_1_count == tile.virtual_2_count;
        integrals::OvovTileResult exchange;
        if(reuse_exchange) {
          ++reused_exchange_tiles;
        } else {
          const std::size_t direct_result_bytes =
              checked_bytes(direct.values.size(), sizeof(double),
                            "tiled Laplace MP2 retained direct tile");
          if(checked_add(persistent_bytes, direct_result_bytes,
                         "tiled Laplace MP2 retained storage") >=
             options.maximum_additional_memory_bytes) {
            throw std::runtime_error(
                "tiled Laplace MP2 has insufficient memory for the exchange "
                "tile");
          }
          exchange = provider.compute_ovov_tile(
              occupied_coefficients, virtual_coefficients,
              integrals::OvovTileOptions{
                  .left_occupied_first =
                      tile.occupied_1_first,
                  .left_occupied_count =
                      tile.occupied_1_count,
                  .left_virtual_first =
                      tile.virtual_2_first,
                  .left_virtual_count =
                      tile.virtual_2_count,
                  .right_occupied_first =
                      tile.occupied_2_first,
                  .right_occupied_count =
                      tile.occupied_2_count,
                  .right_virtual_first =
                      tile.virtual_1_first,
                  .right_virtual_count =
                      tile.virtual_1_count,
                  .threads = options.threads,
                  .schwarz_threshold =
                      options.eri_schwarz_threshold,
                  .maximum_additional_memory_bytes =
                      options.maximum_additional_memory_bytes -
                      persistent_bytes - direct_result_bytes,
              });
          ++exchange_transforms;
          evaluated_shell_quartets +=
              exchange.evaluated_shell_quartets;
          screened_shell_quartets +=
              exchange.schwarz_screened_shell_quartets;
          peak_bytes = std::max(
              peak_bytes,
              checked_add(
                  checked_add(persistent_bytes, direct_result_bytes,
                              "tiled Laplace MP2 peak storage"),
                  exchange.estimated_peak_additional_bytes,
                  "tiled Laplace MP2 peak storage"));
        }

        long double tile_energy = 0.0L;
        for(std::size_t i = 0; i < tile.occupied_1_count; ++i) {
          const double energy_i = orbital_energies[
              active_occupied_indices[tile.occupied_1_first + i]];
          for(std::size_t a = 0; a < tile.virtual_1_count; ++a) {
            const double energy_a = orbital_energies[
                virtual_indices[tile.virtual_1_first + a]];
            for(std::size_t j = 0; j < tile.occupied_2_count; ++j) {
              const double energy_j = orbital_energies[
                  active_occupied_indices[
                      tile.occupied_2_first + j]];
              for(std::size_t b = 0; b < tile.virtual_2_count; ++b) {
                const double energy_b = orbital_energies[
                    virtual_indices[tile.virtual_2_first + b]];
                const double denominator =
                    energy_a + energy_b - energy_i - energy_j;
                long double reciprocal = 0.0L;
                for(std::size_t point = 0;
                    point < fit.nodes.size(); ++point) {
                  reciprocal +=
                      static_cast<long double>(fit.weights[point]) *
                      std::exp(-denominator * fit.nodes[point]);
                }
                const double direct_value = direct(i, a, j, b);
                const double exchange_value =
                    reuse_exchange
                        ? direct(i, b, j, a)
                        : exchange(i, b, j, a);
                tile_energy -=
                    static_cast<long double>(direct_value) *
                    static_cast<long double>(
                        2.0 * direct_value - exchange_value) *
                    reciprocal;
              }
            }
          }
        }
        batch_values[ordinal] = static_cast<double>(tile_energy);
        if(!std::isfinite(batch_values[ordinal])) {
          throw std::runtime_error(
              "tiled Laplace MP2 produced a nonfinite tile energy");
        }
        ++local_tiles;
      } catch(...) {
        local_error = std::current_exception();
      }
    }
#ifdef LMP2_1M2M_HAS_MPI
    if(options.mpi_ranks > 1) {
      propagate_failure(local_error != nullptr,
                        "propagating a tiled Laplace MP2 transform failure");
      allreduce_sum(batch_values);
    }
#endif
    if(local_error) {
      std::rethrow_exception(local_error);
    }
    for(std::size_t ordinal = batch_first; ordinal < batch_last;
        ++ordinal) {
      if(progress.completed[ordinal] == 0) {
        if(!std::isfinite(batch_values[ordinal])) {
          throw std::runtime_error(
              "tiled Laplace MP2 reduced a nonfinite tile energy");
        }
        progress.tile_energy_contributions[ordinal] =
            batch_values[ordinal];
        progress.completed[ordinal] = 1;
      }
    }
    std::exception_ptr observer_error;
    if(observer && options.mpi_rank == 0) {
      try {
        observer(progress);
      } catch(...) {
        observer_error = std::current_exception();
      }
    }
#ifdef LMP2_1M2M_HAS_MPI
    if(options.mpi_ranks > 1) {
      propagate_failure(observer_error != nullptr,
                        "propagating a tiled Laplace MP2 checkpoint failure");
    }
#endif
    if(observer_error) {
      std::rethrow_exception(observer_error);
    }
  }

  long double energy_sum = 0.0L;
  for(std::size_t ordinal = 0; ordinal < total_tiles; ++ordinal) {
    if(progress.completed[ordinal] != 1) {
      throw std::runtime_error(
          "tiled Laplace MP2 ended with an incomplete tile");
    }
    energy_sum += static_cast<long double>(
        progress.tile_energy_contributions[ordinal]);
  }
  const double energy = static_cast<double>(energy_sum);
  if(!std::isfinite(energy) || energy > 1.0e-12) {
    throw std::runtime_error(
        "tiled Laplace MP2 correlation energy is nonfinite or unexpectedly "
        "positive");
  }
  std::pair<std::uint64_t, std::uint64_t> work_range{
      local_tiles, local_tiles};
#ifdef LMP2_1M2M_HAS_MPI
  if(options.mpi_ranks > 1) {
    work_range = reduce_work_range(local_tiles);
    reduce_transform_diagnostics(
        direct_transforms, exchange_transforms,
        reused_exchange_tiles, ordered_shell_quartets,
        evaluated_shell_quartets, screened_shell_quartets,
        peak_bytes);
  }
#endif
  return TiledLaplaceMp2Result{
      .correlation_energy = energy,
      .minimum_denominator = minimum_denominator,
      .maximum_denominator = maximum_denominator,
      .minimum_scaling_exponent = minimum_exponent,
      .maximum_scaling_exponent = maximum_exponent,
      .quadrature_points = fit.nodes.size(),
      .total_tiles = total_tiles,
      .initially_completed_tiles = initially_completed,
      .locally_computed_tiles =
          static_cast<std::size_t>(local_tiles),
      .minimum_tiles_per_rank = work_range.first,
      .maximum_tiles_per_rank = work_range.second,
      .direct_transform_count = direct_transforms,
      .exchange_transform_count = exchange_transforms,
      .reused_exchange_tile_count = reused_exchange_tiles,
      .ordered_shell_quartets_per_transform =
          ordered_shell_quartets,
      .evaluated_shell_quartets = evaluated_shell_quartets,
      .schwarz_screened_shell_quartets =
          screened_shell_quartets,
      .eri_schwarz_threshold = options.eri_schwarz_threshold,
      .estimated_peak_additional_bytes = peak_bytes,
      .fingerprint = fingerprint,
      .progress = std::move(progress),
  };
}

}  // namespace lmp2_1m2m::mp2
