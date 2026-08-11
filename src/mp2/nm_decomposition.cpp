#include "modernqc/mp2/nm_decomposition.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <string>

namespace modernqc::mp2 {
namespace {

[[nodiscard]] std::size_t checked_term_count(std::size_t occupied,
                                             std::size_t virtual_count) {
  std::size_t count = 1;
  for(const std::size_t factor :
      {occupied, occupied, virtual_count, virtual_count}) {
    if(factor == 0) {
      throw std::invalid_argument(
          "nM decomposition requires nonempty occupied and virtual spaces");
    }
    if(count > std::numeric_limits<std::size_t>::max() / factor) {
      throw std::overflow_error("nM excitation count overflows size_t");
    }
    count *= factor;
  }
  return count;
}

void validate_assignments(std::span<const MonomerAssignment> assignments,
                          const char* space) {
  for(const MonomerAssignment& assignment : assignments) {
    if(assignment.monomer_id < 0 ||
       !std::isfinite(assignment.confidence) ||
       assignment.confidence < 0.0 || assignment.confidence > 1.0) {
      throw std::invalid_argument(
          std::string{"invalid hard monomer assignment in "} + space);
    }
  }
}

[[nodiscard]] std::size_t term_index(std::size_t occupied_count,
                                     std::size_t virtual_count,
                                     std::size_t i, std::size_t j,
                                     std::size_t a, std::size_t b) {
  return (((i * occupied_count) + j) * virtual_count + a) *
             virtual_count +
         b;
}

}  // namespace

double NmEnergyDecomposition::total_energy_hartree() const noexcept {
  long double total = 0.0L;
  for(std::size_t body = 1; body <= maximum_mp2_body_order; ++body) {
    total += static_cast<long double>(energy_hartree[body]);
  }
  return static_cast<double>(total);
}

std::uint64_t NmEnergyDecomposition::total_terms() const noexcept {
  std::uint64_t total = 0;
  for(std::size_t body = 1; body <= maximum_mp2_body_order; ++body) {
    total += term_count[body];
  }
  return total;
}

std::vector<MonomerAssignment>
assign_centroids_to_nearest_monomer_references(
    std::span<const std::array<double, 3>> orbital_centroids_bohr,
    std::span<const std::array<double, 3>> reference_positions_bohr,
    std::span<const int> reference_monomer_ids) {
  if(reference_positions_bohr.empty()) {
    throw std::invalid_argument(
        "centroid assignment requires at least one monomer reference");
  }
  if(reference_positions_bohr.size() != reference_monomer_ids.size()) {
    throw std::invalid_argument(
        "reference positions and monomer labels have different sizes");
  }
  for(std::size_t reference = 0;
      reference < reference_positions_bohr.size(); ++reference) {
    if(reference_monomer_ids[reference] < 0 ||
       !std::ranges::all_of(reference_positions_bohr[reference],
                           [](double value) { return std::isfinite(value); })) {
      throw std::invalid_argument(
          "centroid assignment requires finite reference positions and "
          "nonnegative monomer labels");
    }
  }

  std::vector<MonomerAssignment> assignments;
  assignments.reserve(orbital_centroids_bohr.size());
  for(const auto& centroid : orbital_centroids_bohr) {
    if(!std::ranges::all_of(
           centroid, [](double value) { return std::isfinite(value); })) {
      throw std::invalid_argument(
          "centroid assignment received a nonfinite orbital centroid");
    }
    std::size_t nearest_reference = 0;
    double nearest_squared = std::numeric_limits<double>::infinity();
    for(std::size_t reference = 0;
        reference < reference_positions_bohr.size(); ++reference) {
      double distance_squared = 0.0;
      for(std::size_t axis = 0; axis < 3; ++axis) {
        const double displacement =
            centroid[axis] - reference_positions_bohr[reference][axis];
        distance_squared += displacement * displacement;
      }
      // A strict comparison makes exact ties deterministic: the reference
      // with the lowest input index remains selected.
      if(distance_squared < nearest_squared) {
        nearest_squared = distance_squared;
        nearest_reference = reference;
      }
    }
    double competing_monomer_squared =
        std::numeric_limits<double>::infinity();
    for(std::size_t reference = 0;
        reference < reference_positions_bohr.size(); ++reference) {
      if(reference_monomer_ids[reference] ==
         reference_monomer_ids[nearest_reference]) {
        continue;
      }
      double distance_squared = 0.0;
      for(std::size_t axis = 0; axis < 3; ++axis) {
        const double displacement =
            centroid[axis] - reference_positions_bohr[reference][axis];
        distance_squared += displacement * displacement;
      }
      competing_monomer_squared =
          std::min(competing_monomer_squared, distance_squared);
    }
    const double confidence =
        std::isfinite(competing_monomer_squared) &&
                competing_monomer_squared > 0.0
            ? std::clamp(
                  1.0 - std::sqrt(nearest_squared /
                                  competing_monomer_squared),
                  0.0, 1.0)
            : 1.0;
    assignments.push_back(
        {.monomer_id = reference_monomer_ids[nearest_reference],
         .confidence = confidence});
  }
  return assignments;
}

MonomerAssignmentQuality summarize_assignment_quality(
    std::span<const MonomerAssignment> assignments,
    double confidence_threshold) {
  if(assignments.empty()) {
    throw std::invalid_argument(
        "monomer-assignment quality requires at least one orbital");
  }
  if(!std::isfinite(confidence_threshold) || confidence_threshold < 0.0 ||
     confidence_threshold > 1.0) {
    throw std::invalid_argument(
        "monomer-assignment confidence threshold must lie in [0,1]");
  }
  validate_assignments(assignments, "assignment-quality input");
  MonomerAssignmentQuality result{
      .orbital_count = assignments.size(),
      .low_confidence_count = 0,
      .minimum_confidence = 1.0,
  };
  for(const MonomerAssignment& assignment : assignments) {
    result.minimum_confidence =
        std::min(result.minimum_confidence, assignment.confidence);
    if(assignment.confidence < confidence_threshold) {
      ++result.low_confidence_count;
    }
  }
  return result;
}

std::size_t excitation_body_order(int occupied_i_monomer,
                                  int occupied_j_monomer,
                                  int virtual_a_monomer,
                                  int virtual_b_monomer) {
  std::array<int, maximum_mp2_body_order> labels{
      occupied_i_monomer, occupied_j_monomer,
      virtual_a_monomer, virtual_b_monomer};
  if(std::ranges::any_of(labels,
                         [](int label) { return label < 0; })) {
    throw std::invalid_argument(
        "nM body order requires four hard nonnegative monomer labels");
  }
  std::ranges::sort(labels);
  return static_cast<std::size_t>(
      1 + std::ranges::count_if(
              std::views::iota(std::size_t{1}, labels.size()),
              [&](std::size_t index) {
                return labels[index] != labels[index - 1];
              }));
}

bool should_compute_excitation(NmExecutionMode mode,
                               int occupied_i_monomer,
                               int occupied_j_monomer,
                               int virtual_a_monomer,
                               int virtual_b_monomer) {
  const std::size_t body = excitation_body_order(
      occupied_i_monomer, occupied_j_monomer,
      virtual_a_monomer, virtual_b_monomer);
  return mode == NmExecutionMode::full || body <= 2;
}

NmEnergyDecomposition decompose_localized_mp2_terms(
    std::span<const MonomerAssignment> occupied_assignments,
    std::span<const MonomerAssignment> virtual_assignments,
    std::span<const double> term_energy_hartree,
    double confidence_threshold) {
  if(!std::isfinite(confidence_threshold) || confidence_threshold < 0.0 ||
     confidence_threshold > 1.0) {
    throw std::invalid_argument(
        "nM confidence threshold must lie in [0,1]");
  }
  validate_assignments(occupied_assignments, "occupied space");
  validate_assignments(virtual_assignments, "virtual space");
  const std::size_t expected = checked_term_count(
      occupied_assignments.size(), virtual_assignments.size());
  if(term_energy_hartree.size() != expected) {
    throw std::invalid_argument(
        "nM term-energy tensor has incompatible dimensions");
  }

  NmEnergyDecomposition result;
  result.confidence_threshold = confidence_threshold;
  std::array<long double, maximum_mp2_body_order + 1> sums{};
  for(std::size_t i = 0; i < occupied_assignments.size(); ++i) {
    for(std::size_t j = 0; j < occupied_assignments.size(); ++j) {
      for(std::size_t a = 0; a < virtual_assignments.size(); ++a) {
        for(std::size_t b = 0; b < virtual_assignments.size(); ++b) {
          const double energy = term_energy_hartree[term_index(
              occupied_assignments.size(), virtual_assignments.size(),
              i, j, a, b)];
          if(!std::isfinite(energy)) {
            throw std::invalid_argument(
                "nM term-energy tensor contains a nonfinite value");
          }
          const std::size_t body = excitation_body_order(
              occupied_assignments[i].monomer_id,
              occupied_assignments[j].monomer_id,
              virtual_assignments[a].monomer_id,
              virtual_assignments[b].monomer_id);
          sums[body] += static_cast<long double>(energy);
          ++result.term_count[body];
          if(occupied_assignments[i].confidence < confidence_threshold ||
             occupied_assignments[j].confidence < confidence_threshold ||
             virtual_assignments[a].confidence < confidence_threshold ||
             virtual_assignments[b].confidence < confidence_threshold) {
            ++result.low_confidence_terms;
          }
        }
      }
    }
  }
  for(std::size_t body = 1; body <= maximum_mp2_body_order; ++body) {
    result.energy_hartree[body] = static_cast<double>(sums[body]);
  }
  return result;
}

}  // namespace modernqc::mp2
