#pragma once

#include "lmp2_1m2m/integrals/integral_provider.hpp"
#include "lmp2_1m2m/linalg/matrix.hpp"

#include <cstddef>
#include <vector>

namespace lmp2_1m2m::mp2 {

struct CanonicalMp2Options {
  std::size_t maximum_memory_bytes{512U * 1024U * 1024U};
  double minimum_positive_denominator{1.0e-10};
};

struct CanonicalMp2Result {
  double correlation_energy;
  double minimum_denominator;
  double maximum_denominator;
  std::size_t active_occupied_count;
  std::size_t virtual_count;
  std::size_t estimated_peak_bytes;
  std::size_t ao_eri_bytes;
  // (i a | j b), chemists' notation; b is the fastest index.
  std::vector<double> ovov_integrals;

  [[nodiscard]] double integral(std::size_t i, std::size_t a,
                                std::size_t j, std::size_t b) const {
    return ovov_integrals.at(
        (((i * virtual_count) + a) * active_occupied_count + j) *
            virtual_count +
        b);
  }
};

[[nodiscard]] CanonicalMp2Result compute_canonical_mp2(
    const integrals::Libint2IntegralProvider& provider,
    const linalg::Matrix& canonical_coefficients,
    const std::vector<double>& orbital_energies,
    const std::vector<std::size_t>& active_occupied_indices,
    const std::vector<std::size_t>& virtual_indices,
    const CanonicalMp2Options& options = {});

}  // namespace lmp2_1m2m::mp2
