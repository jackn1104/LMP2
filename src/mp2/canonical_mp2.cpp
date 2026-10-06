#include "lmp2_1m2m/mp2/canonical_mp2.hpp"

#include "lmp2_1m2m/integrals/integral_provider.hpp"
#include "lmp2_1m2m/linalg/matrix.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lmp2_1m2m::mp2 {
namespace {

[[nodiscard]] std::size_t checked_product(
    std::initializer_list<std::size_t> factors, const char* name) {
  std::size_t result = 1;
  for(const std::size_t factor : factors) {
    if(factor == 0) {
      throw std::invalid_argument(std::string{name} +
                                  " contains a zero dimension");
    }
    if(result > std::numeric_limits<std::size_t>::max() / factor) {
      throw std::overflow_error(std::string{name} +
                                " element count overflows size_t");
    }
    result *= factor;
  }
  return result;
}

[[nodiscard]] std::size_t checked_bytes(std::size_t elements,
                                        const char* name) {
  if(elements >
     std::numeric_limits<std::size_t>::max() / sizeof(double)) {
    throw std::overflow_error(std::string{name} +
                              " byte count overflows size_t");
  }
  return elements * sizeof(double);
}

[[nodiscard]] std::size_t checked_sum(std::size_t left,
                                      std::size_t right,
                                      const char* name) {
  if(left > std::numeric_limits<std::size_t>::max() - right) {
    throw std::overflow_error(std::string{name} +
                              " byte count overflows size_t");
  }
  return left + right;
}

void validate_indices(const std::vector<std::size_t>& indices,
                      std::size_t orbital_count, const char* name,
                      std::set<std::size_t>& all_indices) {
  if(indices.empty()) {
    throw std::invalid_argument(std::string{name} +
                                " orbital space may not be empty");
  }
  for(const std::size_t index : indices) {
    if(index >= orbital_count) {
      throw std::invalid_argument(std::string{name} +
                                  " orbital index is out of range");
    }
    if(!all_indices.insert(index).second) {
      throw std::invalid_argument(
          "MP2 occupied/virtual orbital indices overlap or repeat");
    }
  }
}

}  // namespace

CanonicalMp2Result compute_canonical_mp2(
    const integrals::Libint2IntegralProvider& provider,
    const linalg::Matrix& canonical_coefficients,
    const std::vector<double>& orbital_energies,
    const std::vector<std::size_t>& active_occupied_indices,
    const std::vector<std::size_t>& virtual_indices,
    const CanonicalMp2Options& options) {
  if(options.maximum_memory_bytes == 0 ||
     !std::isfinite(options.minimum_positive_denominator) ||
     options.minimum_positive_denominator <= 0.0) {
    throw std::invalid_argument(
        "canonical MP2 memory/denominator options are invalid");
  }
  const std::size_t ao_count =
      provider.basis_metadata().number_of_aos;
  const std::size_t orbital_count = canonical_coefficients.columns();
  if(canonical_coefficients.rows() != ao_count || orbital_count == 0 ||
     orbital_energies.size() != orbital_count) {
    throw std::invalid_argument(
        "canonical MP2 coefficient/energy dimensions are incompatible");
  }
  linalg::require_finite(canonical_coefficients,
                         "canonical MP2 coefficients");
  for(const double energy : orbital_energies) {
    if(!std::isfinite(energy)) {
      throw std::invalid_argument(
          "canonical MP2 orbital energy is nonfinite");
    }
  }
  std::set<std::size_t> all_indices;
  validate_indices(active_occupied_indices, orbital_count,
                   "active occupied", all_indices);
  validate_indices(virtual_indices, orbital_count, "virtual", all_indices);

  const std::size_t occupied_count = active_occupied_indices.size();
  const std::size_t virtual_count = virtual_indices.size();
  const std::size_t ao_elements =
      checked_product({ao_count, ao_count, ao_count, ao_count},
                      "AO ERI tensor");
  const std::size_t stage_1_elements =
      checked_product({occupied_count, ao_count, ao_count, ao_count},
                      "MP2 first transform");
  const std::size_t stage_2_elements =
      checked_product({occupied_count, virtual_count, ao_count, ao_count},
                      "MP2 second transform");
  const std::size_t stage_3_elements =
      checked_product(
          {occupied_count, virtual_count, occupied_count, ao_count},
          "MP2 third transform");
  const std::size_t stage_4_elements =
      checked_product(
          {occupied_count, virtual_count, occupied_count, virtual_count},
          "MP2 OVOV tensor");
  const std::size_t ao_bytes = checked_bytes(ao_elements, "AO ERI tensor");
  const std::size_t stage_1_bytes =
      checked_bytes(stage_1_elements, "MP2 first transform");
  const std::size_t stage_2_bytes =
      checked_bytes(stage_2_elements, "MP2 second transform");
  const std::size_t stage_3_bytes =
      checked_bytes(stage_3_elements, "MP2 third transform");
  const std::size_t stage_4_bytes =
      checked_bytes(stage_4_elements, "MP2 OVOV tensor");
  const std::size_t maximum_adjacent =
      std::max({checked_sum(stage_1_bytes, stage_2_bytes,
                            "MP2 first/second transform"),
                checked_sum(stage_2_bytes, stage_3_bytes,
                            "MP2 second/third transform"),
                checked_sum(stage_3_bytes, stage_4_bytes,
                            "MP2 third/fourth transform")});
  const std::size_t estimated_peak_bytes =
      checked_sum(ao_bytes, maximum_adjacent,
                  "canonical MP2 peak-memory estimate");
  if(estimated_peak_bytes > options.maximum_memory_bytes) {
    throw std::runtime_error(
        "canonical MP2 validation path estimates " +
        std::to_string(estimated_peak_bytes) +
        " peak bytes, exceeding the explicit limit of " +
        std::to_string(options.maximum_memory_bytes));
  }

  integrals::ValidationAoEriTensor ao_eris =
      provider.compute_validation_ao_eris(options.maximum_memory_bytes);
  std::vector<double> stage_1(stage_1_elements, 0.0);
  const auto stage_1_index =
      [=](std::size_t i, std::size_t nu, std::size_t lambda,
          std::size_t sigma) {
        return (((i * ao_count) + nu) * ao_count + lambda) * ao_count +
               sigma;
      };
  for(std::size_t i = 0; i < occupied_count; ++i) {
    const std::size_t canonical_i = active_occupied_indices[i];
    for(std::size_t nu = 0; nu < ao_count; ++nu) {
      for(std::size_t lambda = 0; lambda < ao_count; ++lambda) {
        for(std::size_t sigma = 0; sigma < ao_count; ++sigma) {
          double value = 0.0;
          for(std::size_t mu = 0; mu < ao_count; ++mu) {
            value += canonical_coefficients(mu, canonical_i) *
                     ao_eris(mu, nu, lambda, sigma);
          }
          stage_1[stage_1_index(i, nu, lambda, sigma)] = value;
        }
      }
    }
  }

  std::vector<double> stage_2(stage_2_elements, 0.0);
  const auto stage_2_index =
      [=](std::size_t i, std::size_t a, std::size_t lambda,
          std::size_t sigma) {
        return (((i * virtual_count) + a) * ao_count + lambda) *
                   ao_count +
               sigma;
      };
  for(std::size_t i = 0; i < occupied_count; ++i) {
    for(std::size_t a = 0; a < virtual_count; ++a) {
      const std::size_t canonical_a = virtual_indices[a];
      for(std::size_t lambda = 0; lambda < ao_count; ++lambda) {
        for(std::size_t sigma = 0; sigma < ao_count; ++sigma) {
          double value = 0.0;
          for(std::size_t nu = 0; nu < ao_count; ++nu) {
            value += canonical_coefficients(nu, canonical_a) *
                     stage_1[stage_1_index(i, nu, lambda, sigma)];
          }
          stage_2[stage_2_index(i, a, lambda, sigma)] = value;
        }
      }
    }
  }
  std::vector<double>{}.swap(stage_1);

  std::vector<double> stage_3(stage_3_elements, 0.0);
  const auto stage_3_index =
      [=](std::size_t i, std::size_t a, std::size_t j,
          std::size_t sigma) {
        return (((i * virtual_count) + a) * occupied_count + j) *
                   ao_count +
               sigma;
      };
  for(std::size_t i = 0; i < occupied_count; ++i) {
    for(std::size_t a = 0; a < virtual_count; ++a) {
      for(std::size_t j = 0; j < occupied_count; ++j) {
        const std::size_t canonical_j = active_occupied_indices[j];
        for(std::size_t sigma = 0; sigma < ao_count; ++sigma) {
          double value = 0.0;
          for(std::size_t lambda = 0; lambda < ao_count; ++lambda) {
            value += canonical_coefficients(lambda, canonical_j) *
                     stage_2[stage_2_index(i, a, lambda, sigma)];
          }
          stage_3[stage_3_index(i, a, j, sigma)] = value;
        }
      }
    }
  }
  std::vector<double>{}.swap(stage_2);

  std::vector<double> ovov(stage_4_elements, 0.0);
  const auto ovov_index =
      [=](std::size_t i, std::size_t a, std::size_t j,
          std::size_t b) {
        return (((i * virtual_count) + a) * occupied_count + j) *
                   virtual_count +
               b;
      };
  for(std::size_t i = 0; i < occupied_count; ++i) {
    for(std::size_t a = 0; a < virtual_count; ++a) {
      for(std::size_t j = 0; j < occupied_count; ++j) {
        for(std::size_t b = 0; b < virtual_count; ++b) {
          const std::size_t canonical_b = virtual_indices[b];
          double value = 0.0;
          for(std::size_t sigma = 0; sigma < ao_count; ++sigma) {
            value += canonical_coefficients(sigma, canonical_b) *
                     stage_3[stage_3_index(i, a, j, sigma)];
          }
          ovov[ovov_index(i, a, j, b)] = value;
        }
      }
    }
  }
  std::vector<double>{}.swap(stage_3);

  double minimum_denominator = std::numeric_limits<double>::infinity();
  double maximum_denominator = 0.0;
  long double correlation_energy = 0.0L;
  for(std::size_t i = 0; i < occupied_count; ++i) {
    const double energy_i =
        orbital_energies[active_occupied_indices[i]];
    for(std::size_t j = 0; j < occupied_count; ++j) {
      const double energy_j =
          orbital_energies[active_occupied_indices[j]];
      for(std::size_t a = 0; a < virtual_count; ++a) {
        const double energy_a = orbital_energies[virtual_indices[a]];
        for(std::size_t b = 0; b < virtual_count; ++b) {
          const double energy_b = orbital_energies[virtual_indices[b]];
          const double denominator =
              energy_a + energy_b - energy_i - energy_j;
          if(!std::isfinite(denominator) ||
             denominator <= options.minimum_positive_denominator) {
            throw std::runtime_error(
                "canonical MP2 denominator is nonpositive or below the "
                "configured minimum");
          }
          minimum_denominator =
              std::min(minimum_denominator, denominator);
          maximum_denominator =
              std::max(maximum_denominator, denominator);
          const double direct = ovov[ovov_index(i, a, j, b)];
          const double exchange = ovov[ovov_index(i, b, j, a)];
          correlation_energy -=
              static_cast<long double>(direct) *
              static_cast<long double>(2.0 * direct - exchange) /
              static_cast<long double>(denominator);
        }
      }
    }
  }
  const double energy = static_cast<double>(correlation_energy);
  if(!std::isfinite(energy) || energy > 1.0e-12) {
    throw std::runtime_error(
        "canonical MP2 correlation energy is nonfinite or unexpectedly positive");
  }
  return CanonicalMp2Result{
      .correlation_energy = energy,
      .minimum_denominator = minimum_denominator,
      .maximum_denominator = maximum_denominator,
      .active_occupied_count = occupied_count,
      .virtual_count = virtual_count,
      .estimated_peak_bytes = estimated_peak_bytes,
      .ao_eri_bytes = ao_bytes,
      .ovov_integrals = std::move(ovov),
  };
}

}  // namespace lmp2_1m2m::mp2
