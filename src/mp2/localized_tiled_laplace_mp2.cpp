#include "lmp2_1m2m/mp2/localized_tiled_laplace_mp2.hpp"

#include "lmp2_1m2m/integrals/integral_provider.hpp"
#include "lmp2_1m2m/linalg/matrix.hpp"
#include "lmp2_1m2m/mp2/nm_decomposition.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <optional>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef LMP2_1M2M_HAS_OPENMP
#include <omp.h>
#endif

#ifdef LMP2_1M2M_HAS_MPI
#include <mpi.h>
#endif

namespace lmp2_1m2m::mp2 {
namespace {

constexpr long double log_minimum_normal =
    -708.39641853226410622L;

extern "C" {
void dgemm_(const char* transa, const char* transb, const int* m,
            const int* n, const int* k, const double* alpha,
            const double* a, const int* lda, const double* b,
            const int* ldb, const double* beta, double* c,
            const int* ldc);
}

struct OrbitalBlock {
  std::size_t first;
  std::size_t count;
  int monomer;
};

struct LocalizedTask {
  std::size_t occupied_i_block;
  std::size_t occupied_j_block;
  std::size_t virtual_a_block;
  std::size_t virtual_b_block;
  std::size_t body_order;
  std::uint64_t term_count;
  bool distinct_occupied_blocks;
  bool distinct_virtual_blocks;
};

struct IndexRange {
  std::size_t first;
  std::size_t count;
};

struct LocalizedRightPanel {
  std::size_t owner;
  std::size_t occupied_block;
  std::size_t virtual_block;
  std::vector<std::size_t> pair_indices;
};

struct OrderedBlockMask {
  std::size_t occupied_blocks;
  std::size_t virtual_blocks;
  std::vector<unsigned char> active;

  [[nodiscard]] std::size_t index(
      std::size_t occupied_i_block, std::size_t occupied_j_block,
      std::size_t virtual_a_block,
      std::size_t virtual_b_block) const {
    if(occupied_i_block >= occupied_blocks ||
       occupied_j_block >= occupied_blocks ||
       virtual_a_block >= virtual_blocks ||
       virtual_b_block >= virtual_blocks) {
      throw std::out_of_range("localized screening block index is invalid");
    }
    return (((occupied_i_block * occupied_blocks + occupied_j_block) *
             virtual_blocks + virtual_a_block) *
            virtual_blocks + virtual_b_block);
  }

  [[nodiscard]] bool operator()(
      std::size_t occupied_i_block, std::size_t occupied_j_block,
      std::size_t virtual_a_block,
      std::size_t virtual_b_block) const {
    return active.at(index(occupied_i_block, occupied_j_block,
                           virtual_a_block, virtual_b_block)) != 0U;
  }

  void enable(std::size_t occupied_i_block,
              std::size_t occupied_j_block,
              std::size_t virtual_a_block,
              std::size_t virtual_b_block) {
    active.at(index(occupied_i_block, occupied_j_block,
                    virtual_a_block, virtual_b_block)) = 1U;
  }
};

struct SelectedIntegral {
  std::uint64_t key;
  double value;
  std::size_t body_order;
  bool accumulate_energy{true};
};

struct PointScreeningData {
  double threshold_hartree;
  double quadrature_weight;
  const std::vector<double>* pair_vector_norms;
  const std::vector<double>* one_sided_integral_norms;
};

struct SelectedRotationResult {
  std::vector<SelectedIntegral> entries;
  std::uint64_t retained_point_contributions{0};
  std::uint64_t screened_point_contributions{0};
  long double omitted_energy_bound_hartree{0.0L};
};

struct BatchedOwnedSelectedTask {
  LocalizedTask task;
  std::size_t left_occupied_first;
  std::size_t left_occupied_count;
  std::vector<std::vector<double>> direct_by_point;
  std::vector<std::vector<double>> exchange_by_point;
};

struct BatchedSelectedOperation {
  std::size_t owned_task;
  bool exchange;
};

[[nodiscard]] double elapsed_seconds(
    std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now() - start)
      .count();
}

[[nodiscard]] std::size_t checked_multiply(
    std::size_t left, std::size_t right, const char* description);

[[nodiscard]] std::vector<double> transform_batched_left_virtual_block(
    const integrals::BatchedFirstIndexAoBlock& block,
    std::size_t point, const linalg::Matrix& virtual_coefficients,
    std::size_t virtual_first, std::size_t virtual_count,
    std::size_t threads) {
  if(point >= block.point_values.size() || block.ao_count == 0 ||
     block.lambda_ao_count == 0 || block.occupied_count == 0 ||
     virtual_coefficients.rows() != block.ao_count ||
     virtual_first > virtual_coefficients.columns() ||
     virtual_count == 0 ||
     virtual_count > virtual_coefficients.columns() - virtual_first ||
     threads == 0) {
    throw std::invalid_argument(
        "batched selected virtual transform dimensions are invalid");
  }
  const std::size_t rows = checked_multiply(
      checked_multiply(block.lambda_ao_count, block.ao_count,
                       "batched selected lambda-sigma rows"),
      block.occupied_count, "batched selected occupied rows");
  const std::size_t expected = checked_multiply(
      rows, block.ao_count, "batched selected first-index block size");
  if(block.point_values[point].size() != expected) {
    throw std::invalid_argument(
        "batched selected first-index block has an invalid size");
  }
  std::vector<double> result(
      checked_multiply(rows, virtual_count,
                       "batched selected virtual block size"),
      0.0);
  std::vector<std::exception_ptr> errors(threads);
  const auto transform_rows =
      [&](std::size_t thread, std::size_t actual_threads) {
        const std::vector<double>& input = block.point_values[point];
        for(std::size_t row = thread; row < rows;
            row += actual_threads) {
          const double* source =
              input.data() + row * block.ao_count;
          double* destination = result.data() + row * virtual_count;
          for(std::size_t local_a = 0; local_a < virtual_count;
              ++local_a) {
            double sum = 0.0;
#if defined(LMP2_1M2M_HAS_OPENMP) && defined(__GNUC__) && \
    !defined(__clang__)
#pragma omp simd reduction(+ : sum)
#endif
            for(std::size_t nu = 0; nu < block.ao_count; ++nu) {
              sum += source[nu] *
                     virtual_coefficients(
                         nu, virtual_first + local_a);
            }
            destination[local_a] = sum;
          }
        }
      };
#ifdef LMP2_1M2M_HAS_OPENMP
#pragma omp parallel num_threads(static_cast<int>(threads))
  {
    const std::size_t thread =
        static_cast<std::size_t>(omp_get_thread_num());
    const std::size_t actual_threads =
        static_cast<std::size_t>(omp_get_num_threads());
    try {
      transform_rows(thread, actual_threads);
    } catch(...) {
      errors[thread] = std::current_exception();
    }
  }
#else
  try {
    transform_rows(0U, 1U);
  } catch(...) {
    errors[0] = std::current_exception();
  }
#endif
  for(const auto& error : errors) {
    if(error) {
      std::rethrow_exception(error);
    }
  }
  return result;
}

void accumulate_batched_selected_operation(
    const integrals::BatchedFirstIndexAoBlock& block,
    const std::vector<double>& left_virtual_block,
    std::size_t left_virtual_count,
    const linalg::Matrix& occupied_coefficients,
    const linalg::Matrix& virtual_coefficients,
    const OrbitalBlock& right_occupied,
    const OrbitalBlock& right_virtual,
    const BatchedOwnedSelectedTask& owned_task,
    std::vector<double>& output) {
  const std::size_t ao_count = block.ao_count;
  const std::size_t left_pair_count = checked_multiply(
      owned_task.left_occupied_count, left_virtual_count,
      "batched selected left-pair count");
  const std::size_t right_pair_count = checked_multiply(
      right_occupied.count, right_virtual.count,
      "batched selected right-pair count");
  if(occupied_coefficients.rows() != ao_count ||
     virtual_coefficients.rows() != ao_count ||
     right_occupied.first > occupied_coefficients.columns() ||
     right_occupied.count >
         occupied_coefficients.columns() - right_occupied.first ||
     right_virtual.first > virtual_coefficients.columns() ||
     right_virtual.count >
         virtual_coefficients.columns() - right_virtual.first ||
     output.size() !=
         checked_multiply(left_pair_count, right_pair_count,
                          "batched selected output size")) {
    throw std::invalid_argument(
        "batched selected operation dimensions are invalid");
  }
  const std::size_t expected_left = checked_multiply(
      checked_multiply(
          checked_multiply(block.lambda_ao_count, ao_count,
                           "batched selected T lambda-sigma"),
          block.occupied_count, "batched selected T occupied"),
      left_virtual_count, "batched selected T virtual");
  if(left_virtual_block.size() != expected_left) {
    throw std::invalid_argument(
        "batched selected T block has an invalid size");
  }
  const std::size_t owned_offset =
      owned_task.left_occupied_first;
  std::vector<double> third(
      checked_multiply(ao_count, right_occupied.count,
                       "batched selected third-stage workspace"),
      0.0);
  for(std::size_t local_i = 0;
      local_i < owned_task.left_occupied_count; ++local_i) {
    const std::size_t owned_i = owned_offset + local_i;
    for(std::size_t local_a = 0; local_a < left_virtual_count;
        ++local_a) {
      const std::size_t left_pair =
          local_i * left_virtual_count + local_a;
      for(std::size_t local_j = 0; local_j < right_occupied.count;
          ++local_j) {
        double* third_j = third.data() + local_j * ao_count;
        for(std::size_t sigma = 0; sigma < ao_count; ++sigma) {
          double sum = 0.0;
          for(std::size_t lambda_local = 0;
              lambda_local < block.lambda_ao_count; ++lambda_local) {
            const std::size_t row =
                (lambda_local * ao_count + sigma) *
                    block.occupied_count +
                owned_i;
            sum += left_virtual_block[
                       row * left_virtual_count + local_a] *
                   occupied_coefficients(
                       block.lambda_ao_first + lambda_local,
                       right_occupied.first + local_j);
          }
          third_j[sigma] = sum;
        }
        for(std::size_t local_b = 0; local_b < right_virtual.count;
            ++local_b) {
          double sum = 0.0;
#if defined(LMP2_1M2M_HAS_OPENMP) && defined(__GNUC__) && \
    !defined(__clang__)
#pragma omp simd reduction(+ : sum)
#endif
          for(std::size_t sigma = 0; sigma < ao_count; ++sigma) {
            sum += third_j[sigma] *
                   virtual_coefficients(
                       sigma, right_virtual.first + local_b);
          }
          const std::size_t right_pair =
              local_j * right_virtual.count + local_b;
          output[right_pair * left_pair_count + left_pair] += sum;
        }
      }
    }
  }
}

[[nodiscard]] IndexRange balanced_range(std::size_t extent,
                                        std::size_t rank,
                                        std::size_t ranks) {
  if(ranks == 0 || rank >= ranks) {
    throw std::invalid_argument("balanced range topology is invalid");
  }
  const std::size_t base = extent / ranks;
  const std::size_t remainder = extent % ranks;
  return IndexRange{
      .first = rank * base + std::min(rank, remainder),
      .count = base + (rank < remainder ? 1U : 0U),
  };
}

[[nodiscard]] std::size_t balanced_owner(std::size_t index,
                                         std::size_t extent,
                                         std::size_t ranks) {
  if(index >= extent || ranks == 0) {
    throw std::invalid_argument("balanced owner index is invalid");
  }
  const std::size_t base = extent / ranks;
  const std::size_t remainder = extent % ranks;
  const std::size_t enlarged_extent = remainder * (base + 1U);
  if(index < enlarged_extent) {
    return index / (base + 1U);
  }
  if(base == 0) {
    throw std::logic_error("balanced owner has an empty tail");
  }
  return remainder + (index - enlarged_extent) / base;
}

[[nodiscard]] std::uint64_t checked_u64_add(
    std::uint64_t left, std::uint64_t right, const char* description) {
  if(left > std::numeric_limits<std::uint64_t>::max() - right) {
    throw std::overflow_error(std::string{description} +
                              " overflows uint64");
  }
  return left + right;
}

[[nodiscard]] std::uint64_t checked_u64_multiply(
    std::uint64_t left, std::uint64_t right, const char* description) {
  if(left != 0 &&
     right > std::numeric_limits<std::uint64_t>::max() / left) {
    throw std::overflow_error(std::string{description} +
                              " overflows uint64");
  }
  return left * right;
}

[[nodiscard]] std::size_t checked_add(std::size_t left, std::size_t right,
                                      const char* description) {
  if(left > std::numeric_limits<std::size_t>::max() - right) {
    throw std::overflow_error(std::string{description} + " overflows size_t");
  }
  return left + right;
}

[[nodiscard]] std::size_t checked_multiply(
    std::size_t left, std::size_t right, const char* description) {
  if(left != 0 &&
     right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::overflow_error(std::string{description} +
                              " overflows size_t");
  }
  return left * right;
}

[[nodiscard]] std::size_t checked_bytes(std::size_t elements,
                                        const char* description) {
  if(elements > std::numeric_limits<std::size_t>::max() / sizeof(double)) {
    throw std::overflow_error(std::string{description} + " byte count overflows");
  }
  return elements * sizeof(double);
}

[[nodiscard]] int blas_dimension(std::size_t value,
                                 const char* description) {
  if(value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error(std::string{description} +
                              " exceeds the BLAS integer range");
  }
  return static_cast<int>(value);
}

[[nodiscard]] std::uint64_t checked_terms(const OrbitalBlock& i,
                                          const OrbitalBlock& j,
                                          const OrbitalBlock& a,
                                          const OrbitalBlock& b) {
  std::uint64_t count = 1;
  for(const std::size_t factor : {i.count, j.count, a.count, b.count}) {
    if(factor > static_cast<std::size_t>(
                     std::numeric_limits<std::uint64_t>::max() / count)) {
      throw std::overflow_error("localized excitation count overflows uint64");
    }
    count *= static_cast<std::uint64_t>(factor);
  }
  return count;
}

void validate_topology(const LocalizedTiledLaplaceMp2Options& options) {
  if(options.mpi_ranks == 0 || options.mpi_rank >= options.mpi_ranks) {
    throw std::invalid_argument("localized LMP2-1M2M MPI topology is invalid");
  }
#ifdef LMP2_1M2M_HAS_MPI
  if(options.mpi_ranks > 1) {
    int initialized = 0;
    int rank = 0;
    int ranks = 0;
    if(MPI_Initialized(&initialized) != MPI_SUCCESS || initialized == 0 ||
       MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS ||
       MPI_Comm_size(MPI_COMM_WORLD, &ranks) != MPI_SUCCESS || rank < 0 ||
       ranks < 1 || static_cast<std::size_t>(rank) != options.mpi_rank ||
       static_cast<std::size_t>(ranks) != options.mpi_ranks) {
      throw std::runtime_error(
          "localized LMP2-1M2M topology does not match MPI_COMM_WORLD");
    }
  }
#else
  if(options.mpi_ranks != 1) {
    throw std::runtime_error("multiple localized LMP2-1M2M ranks require MPI");
  }
#endif
}

[[nodiscard]] double orthogonality_error(const linalg::Matrix& rotation) {
  if(rotation.empty() || rotation.rows() != rotation.columns()) {
    throw std::invalid_argument(
        "localized LMP2-1M2M rotations must be nonempty and square");
  }
  linalg::require_finite(rotation, "localized LMP2-1M2M rotation");
  const linalg::Matrix metric =
      linalg::multiply(linalg::transpose(rotation), rotation);
  double error = 0.0;
  for(std::size_t column = 0; column < metric.columns(); ++column) {
    for(std::size_t row = 0; row < metric.rows(); ++row) {
      error = std::max(error, std::abs(
          metric(row, column) - (row == column ? 1.0 : 0.0)));
    }
  }
  return error;
}

void validate_assignments(const std::vector<MonomerAssignment>& assignments,
                          std::size_t expected, const char* space) {
  if(assignments.size() != expected) {
    throw std::invalid_argument(std::string{"localized LMP2-1M2M "} + space +
                                " assignment count is incompatible");
  }
  for(const MonomerAssignment& assignment : assignments) {
    if(assignment.monomer_id < 0 ||
       !std::isfinite(assignment.confidence) ||
       assignment.confidence < 0.0 || assignment.confidence > 1.0) {
      throw std::invalid_argument(std::string{"localized LMP2-1M2M "} + space +
                                  " assignment is invalid");
    }
  }
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

struct GroupedSpace {
  linalg::Matrix rotation;
  std::vector<MonomerAssignment> assignments;
  std::vector<OrbitalBlock> blocks;
};

[[nodiscard]] GroupedSpace group_space(
    const linalg::Matrix& rotation,
    const std::vector<MonomerAssignment>& assignments,
    std::size_t tile_size) {
  std::vector<std::size_t> order(assignments.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::ranges::stable_sort(order, [&](std::size_t left, std::size_t right) {
    return assignments[left].monomer_id < assignments[right].monomer_id;
  });
  GroupedSpace result{
      .rotation = linalg::Matrix{rotation.rows(), rotation.columns()},
      .assignments = {},
      .blocks = {},
  };
  result.assignments.reserve(assignments.size());
  for(std::size_t column = 0; column < order.size(); ++column) {
    for(std::size_t row = 0; row < rotation.rows(); ++row) {
      result.rotation(row, column) = rotation(row, order[column]);
    }
    result.assignments.push_back(assignments[order[column]]);
  }
  for(std::size_t first = 0; first < result.assignments.size();) {
    const int monomer = result.assignments[first].monomer_id;
    std::size_t monomer_last = first + 1;
    while(monomer_last < result.assignments.size() &&
          result.assignments[monomer_last].monomer_id == monomer) {
      ++monomer_last;
    }
    for(std::size_t block_first = first; block_first < monomer_last;
        block_first += tile_size) {
      result.blocks.push_back(OrbitalBlock{
          .first = block_first,
          .count = std::min(tile_size, monomer_last - block_first),
          .monomer = monomer,
      });
    }
    first = monomer_last;
  }
  return result;
}

[[nodiscard]] std::vector<LocalizedRightPanel> localized_right_panels(
    const GroupedSpace& occupied, const GroupedSpace& virtuals,
    std::size_t ranks, std::size_t maximum_panel_size) {
  if(maximum_panel_size == 0) {
    throw std::invalid_argument("localized right-panel size must be positive");
  }
  const std::size_t occupied_count = occupied.assignments.size();
  const std::size_t virtual_count = virtuals.assignments.size();
  if(occupied_count != 0 &&
     virtual_count >
         std::numeric_limits<std::size_t>::max() / occupied_count) {
    throw std::overflow_error("localized right-pair count overflows size_t");
  }
  const std::size_t pair_count = occupied_count * virtual_count;
  std::vector<LocalizedRightPanel> result;
  for(std::size_t occupied_block = 0;
      occupied_block < occupied.blocks.size(); ++occupied_block) {
    const OrbitalBlock& ob = occupied.blocks[occupied_block];
    for(std::size_t virtual_block = 0;
        virtual_block < virtuals.blocks.size(); ++virtual_block) {
      const OrbitalBlock& vb = virtuals.blocks[virtual_block];
      LocalizedRightPanel panel{
          .owner = 0,
          .occupied_block = occupied_block,
          .virtual_block = virtual_block,
          .pair_indices = {},
      };
      panel.pair_indices.reserve(maximum_panel_size);
      for(std::size_t local_occupied = 0;
          local_occupied < ob.count; ++local_occupied) {
        const std::size_t occupied_index = ob.first + local_occupied;
        for(std::size_t local_virtual = 0;
            local_virtual < vb.count; ++local_virtual) {
          const std::size_t virtual_index = vb.first + local_virtual;
          const std::size_t pair =
              occupied_index * virtual_count + virtual_index;
          const std::size_t owner = balanced_owner(pair, pair_count, ranks);
          if(!panel.pair_indices.empty() &&
             (panel.owner != owner ||
              panel.pair_indices.size() == maximum_panel_size)) {
            result.push_back(std::move(panel));
            panel = LocalizedRightPanel{
                .owner = owner,
                .occupied_block = occupied_block,
                .virtual_block = virtual_block,
                .pair_indices = {},
            };
            panel.pair_indices.reserve(maximum_panel_size);
          }
          if(panel.pair_indices.empty()) {
            panel.owner = owner;
          }
          panel.pair_indices.push_back(pair);
        }
      }
      if(!panel.pair_indices.empty()) {
        result.push_back(std::move(panel));
      }
    }
  }
  return result;
}

[[nodiscard]] std::uint64_t selected_integral_key(
    std::size_t left_pair, std::size_t right_pair,
    std::size_t pair_count) {
  if(left_pair >= pair_count || right_pair >= pair_count ||
     pair_count > static_cast<std::size_t>(
                      std::numeric_limits<std::uint64_t>::max()) ||
     right_pair >
         static_cast<std::size_t>(
             std::numeric_limits<std::uint64_t>::max() /
             static_cast<std::uint64_t>(pair_count))) {
    throw std::overflow_error("selected OVOV key overflows uint64");
  }
  return static_cast<std::uint64_t>(right_pair) *
             static_cast<std::uint64_t>(pair_count) +
         static_cast<std::uint64_t>(left_pair);
}

[[nodiscard]] std::vector<LocalizedTask> build_tasks(
    const std::vector<OrbitalBlock>& occupied,
    const std::vector<OrbitalBlock>& virtuals,
    NmExecutionMode mode, std::uint64_t& omitted_terms,
    std::uint64_t& omitted_tasks) {
  std::vector<LocalizedTask> result;
  omitted_terms = 0;
  omitted_tasks = 0;
  for(std::size_t ib = 0; ib < occupied.size(); ++ib) {
    for(std::size_t jb = ib; jb < occupied.size(); ++jb) {
      for(std::size_t ab = 0; ab < virtuals.size(); ++ab) {
        for(std::size_t bb = ab; bb < virtuals.size(); ++bb) {
          const std::size_t body = excitation_body_order(
              occupied[ib].monomer, occupied[jb].monomer,
              virtuals[ab].monomer, virtuals[bb].monomer);
          const std::uint64_t block_terms = checked_terms(
              occupied[ib], occupied[jb], virtuals[ab], virtuals[bb]);
          const std::uint64_t multiplicity =
              static_cast<std::uint64_t>(ib == jb ? 1 : 2) *
              static_cast<std::uint64_t>(ab == bb ? 1 : 2);
          const std::uint64_t terms = checked_u64_multiply(
              block_terms, multiplicity,
              "symmetry-completed localized excitation count");
          if(mode == NmExecutionMode::one_two_monomer && body > 2) {
            omitted_terms = checked_u64_add(
                omitted_terms, terms,
                "omitted localized excitation count");
            omitted_tasks = checked_u64_add(
                omitted_tasks, 1,
                "omitted localized block-task count");
          } else {
            result.push_back(LocalizedTask{
                .occupied_i_block = ib,
                .occupied_j_block = jb,
                .virtual_a_block = ab,
                .virtual_b_block = bb,
                .body_order = body,
                .term_count = terms,
                .distinct_occupied_blocks = ib != jb,
                .distinct_virtual_blocks = ab != bb,
            });
          }
        }
      }
    }
  }
  return result;
}

[[nodiscard]] OrderedBlockMask build_ordered_block_mask(
    std::size_t occupied_blocks, std::size_t virtual_blocks,
    const std::vector<LocalizedTask>& tasks,
    const std::vector<unsigned char>& task_is_active) {
  if(tasks.size() != task_is_active.size()) {
    throw std::invalid_argument(
        "localized screening task mask has an invalid size");
  }
  const std::size_t occupied_pairs = checked_multiply(
      occupied_blocks, occupied_blocks,
      "localized screening occupied-block pairs");
  const std::size_t virtual_pairs = checked_multiply(
      virtual_blocks, virtual_blocks,
      "localized screening virtual-block pairs");
  OrderedBlockMask result{
      .occupied_blocks = occupied_blocks,
      .virtual_blocks = virtual_blocks,
      .active = std::vector<unsigned char>(
          checked_multiply(occupied_pairs, virtual_pairs,
                           "localized screening ordered block mask"),
          static_cast<unsigned char>(0)),
  };
  for(std::size_t task_index = 0; task_index < tasks.size(); ++task_index) {
    if(task_is_active[task_index] == 0U) {
      continue;
    }
    const LocalizedTask& task = tasks[task_index];
    for(const bool swap_occupied : {false, true}) {
      if(swap_occupied && !task.distinct_occupied_blocks) {
        continue;
      }
      const std::size_t ib = swap_occupied
          ? task.occupied_j_block : task.occupied_i_block;
      const std::size_t jb = swap_occupied
          ? task.occupied_i_block : task.occupied_j_block;
      for(const bool swap_virtual : {false, true}) {
        if(swap_virtual && !task.distinct_virtual_blocks) {
          continue;
        }
        const std::size_t ab = swap_virtual
            ? task.virtual_b_block : task.virtual_a_block;
        const std::size_t bb = swap_virtual
            ? task.virtual_a_block : task.virtual_b_block;
        result.enable(ib, jb, ab, bb);
      }
    }
  }
  return result;
}

[[nodiscard]] std::vector<double> scaled_pair_vector_norms(
    const linalg::Matrix& occupied_transform,
    const linalg::Matrix& virtual_transform) {
  std::vector<double> occupied_norms(occupied_transform.columns(), 0.0);
  std::vector<double> virtual_norms(virtual_transform.columns(), 0.0);
  for(std::size_t localized = 0;
      localized < occupied_transform.columns(); ++localized) {
    long double squared = 0.0L;
    for(std::size_t canonical = 0;
        canonical < occupied_transform.rows(); ++canonical) {
      const long double value = occupied_transform(canonical, localized);
      squared += value * value;
    }
    occupied_norms[localized] = std::sqrt(static_cast<double>(squared));
  }
  for(std::size_t localized = 0;
      localized < virtual_transform.columns(); ++localized) {
    long double squared = 0.0L;
    for(std::size_t canonical = 0;
        canonical < virtual_transform.rows(); ++canonical) {
      const long double value = virtual_transform(canonical, localized);
      squared += value * value;
    }
    virtual_norms[localized] = std::sqrt(static_cast<double>(squared));
  }
  const std::size_t pair_count = checked_multiply(
      occupied_norms.size(), virtual_norms.size(),
      "localized screening scaled-pair norms");
  std::vector<double> result(pair_count, 0.0);
  for(std::size_t occupied = 0; occupied < occupied_norms.size();
      ++occupied) {
    for(std::size_t virtual_orbital = 0;
        virtual_orbital < virtual_norms.size(); ++virtual_orbital) {
      result[occupied * virtual_norms.size() + virtual_orbital] =
          occupied_norms[occupied] * virtual_norms[virtual_orbital];
    }
  }
  return result;
}

[[nodiscard]] double maximum_pair_value(
    const std::vector<double>& values, const OrbitalBlock& occupied,
    const OrbitalBlock& virtuals, std::size_t virtual_count) {
  double result = 0.0;
  for(std::size_t local_occupied = 0;
      local_occupied < occupied.count; ++local_occupied) {
    const std::size_t occupied_index = occupied.first + local_occupied;
    for(std::size_t local_virtual = 0;
        local_virtual < virtuals.count; ++local_virtual) {
      const std::size_t pair =
          occupied_index * virtual_count + virtuals.first + local_virtual;
      result = std::max(result, values.at(pair));
    }
  }
  return result;
}

[[nodiscard]] long double localized_task_point_energy_bound(
    const LocalizedTask& task, const GroupedSpace& occupied,
    const GroupedSpace& virtuals,
    const std::vector<double>& right_vector_norms,
    const std::vector<double>& one_sided_integral_norms,
    double quadrature_weight) {
  const std::size_t virtual_count = virtuals.assignments.size();
  const OrbitalBlock& ib = occupied.blocks.at(task.occupied_i_block);
  const OrbitalBlock& jb = occupied.blocks.at(task.occupied_j_block);
  const OrbitalBlock& ab = virtuals.blocks.at(task.virtual_a_block);
  const OrbitalBlock& bb = virtuals.blocks.at(task.virtual_b_block);
  const auto integral_bound = [&](const OrbitalBlock& left_occupied,
                                  const OrbitalBlock& left_virtual,
                                  const OrbitalBlock& right_occupied,
                                  const OrbitalBlock& right_virtual) {
    const double left_norm = maximum_pair_value(
        right_vector_norms, left_occupied, left_virtual, virtual_count);
    const double transformed_right_norm = maximum_pair_value(
        one_sided_integral_norms, right_occupied, right_virtual,
        virtual_count);
    return static_cast<long double>(left_norm) *
           static_cast<long double>(transformed_right_norm);
  };
  const long double direct_ij = integral_bound(ib, ab, jb, bb);
  const long double exchange_ij = integral_bound(ib, bb, jb, ab);
  long double maximum =
      direct_ij * (2.0L * direct_ij + exchange_ij);
  if(task.distinct_virtual_blocks) {
    maximum = std::max(
        maximum,
        exchange_ij * (2.0L * exchange_ij + direct_ij));
  }
  if(task.distinct_occupied_blocks) {
    const long double direct_ji = integral_bound(jb, ab, ib, bb);
    const long double exchange_ji = integral_bound(jb, bb, ib, ab);
    maximum = std::max(
        maximum,
        direct_ji * (2.0L * direct_ji + exchange_ji));
    if(task.distinct_virtual_blocks) {
      maximum = std::max(
          maximum,
          exchange_ji * (2.0L * exchange_ji + direct_ji));
    }
  }
  return std::abs(static_cast<long double>(quadrature_weight)) * maximum;
}

[[nodiscard]] bool panel_has_active_left_blocks(
    const LocalizedRightPanel& panel,
    const OrderedBlockMask& active_blocks) {
  for(std::size_t occupied_block = 0;
      occupied_block < active_blocks.occupied_blocks; ++occupied_block) {
    for(std::size_t virtual_block = 0;
        virtual_block < active_blocks.virtual_blocks; ++virtual_block) {
      if(active_blocks(occupied_block, panel.occupied_block,
                       virtual_block, panel.virtual_block)) {
        return true;
      }
    }
  }
  return false;
}

[[nodiscard]] double safe_scale(long double exponent) {
  if(!std::isfinite(exponent) || exponent > 0.0L) {
    throw std::runtime_error("localized LMP2-1M2M scaling exponent is invalid");
  }
  return exponent < log_minimum_normal
             ? 0.0
             : static_cast<double>(std::exp(exponent));
}

[[nodiscard]] linalg::Matrix scaled_localized_coefficients(
    const linalg::Matrix& canonical_space,
    const std::vector<double>& energies, double energy_shift,
    double node, bool occupied, const linalg::Matrix& grouped_rotation) {
  linalg::Matrix scaled = canonical_space;
  for(std::size_t column = 0; column < scaled.columns(); ++column) {
    const long double exponent =
        (occupied ? 0.5L : -0.5L) *
        (static_cast<long double>(energies[column]) -
         static_cast<long double>(energy_shift)) *
        static_cast<long double>(node);
    const double factor = safe_scale(exponent);
    for(std::size_t row = 0; row < scaled.rows(); ++row) {
      scaled(row, column) *= factor;
    }
  }
  return linalg::multiply(scaled, grouped_rotation);
}

[[nodiscard]] linalg::Matrix scaled_rotation(
    const std::vector<double>& energies, double energy_shift, double node,
    bool occupied, const linalg::Matrix& rotation) {
  if(rotation.rows() != energies.size() ||
     rotation.columns() != energies.size()) {
    throw std::invalid_argument(
        "cached localized LMP2-1M2M rotation dimensions are invalid");
  }
  linalg::Matrix result{rotation.rows(), rotation.columns()};
  for(std::size_t canonical = 0; canonical < rotation.rows(); ++canonical) {
    const long double exponent =
        (occupied ? 0.5L : -0.5L) *
        (static_cast<long double>(energies[canonical]) -
         static_cast<long double>(energy_shift)) *
        static_cast<long double>(node);
    const double factor = safe_scale(exponent);
    for(std::size_t localized = 0; localized < rotation.columns();
        ++localized) {
      result(canonical, localized) =
          factor * rotation(canonical, localized);
    }
  }
  return result;
}

// Expresses exact canonical half scaling in the fixed localized basis. If
// C_L = C_C U, then the scaled localized orbitals are C_L S(t), where
// S(t) = U^T D(t) U and D contains the canonical orbital-energy factors.
[[nodiscard]] linalg::Matrix localized_propagator(
    const std::vector<double>& energies, double energy_shift, double node,
    bool occupied, const linalg::Matrix& rotation) {
  const linalg::Matrix canonical_to_scaled_local = scaled_rotation(
      energies, energy_shift, node, occupied, rotation);
  linalg::Matrix propagator = linalg::multiply(
      linalg::transpose(rotation), canonical_to_scaled_local);
  linalg::require_finite(propagator,
                         "localized LMP2-1M2M dense propagator");
  return propagator;
}

#ifndef NDEBUG
[[nodiscard]] std::vector<double> rotate_cached_ovov_scalar_reference(
    const std::vector<double>& canonical, std::size_t occupied_count,
    std::size_t virtual_count, const linalg::Matrix& occupied_transform,
    const linalg::Matrix& virtual_transform) {
  const std::size_t pair_count = occupied_count * virtual_count;
  std::vector<double> first(canonical.size(), 0.0);
  std::vector<double> second(canonical.size(), 0.0);
  for(std::size_t i = 0; i < occupied_count; ++i) {
    for(std::size_t right = 0; right < pair_count; ++right) {
      for(std::size_t A = 0; A < virtual_count; ++A) {
        for(std::size_t a = 0; a < virtual_count; ++a) {
          first[right * pair_count + i * virtual_count + A] +=
              canonical[right * pair_count + i * virtual_count + a] *
              virtual_transform(a, A);
        }
      }
    }
  }
  for(std::size_t right = 0; right < pair_count; ++right) {
    for(std::size_t I = 0; I < occupied_count; ++I) {
      for(std::size_t A = 0; A < virtual_count; ++A) {
        for(std::size_t i = 0; i < occupied_count; ++i) {
          second[right * pair_count + I * virtual_count + A] +=
              first[right * pair_count + i * virtual_count + A] *
              occupied_transform(i, I);
        }
      }
    }
  }
  std::fill(first.begin(), first.end(), 0.0);
  for(std::size_t left = 0; left < pair_count; ++left) {
    for(std::size_t j = 0; j < occupied_count; ++j) {
      for(std::size_t B = 0; B < virtual_count; ++B) {
        for(std::size_t b = 0; b < virtual_count; ++b) {
          first[(j * virtual_count + B) * pair_count + left] +=
              second[(j * virtual_count + b) * pair_count + left] *
              virtual_transform(b, B);
        }
      }
    }
  }
  std::fill(second.begin(), second.end(), 0.0);
  for(std::size_t left = 0; left < pair_count; ++left) {
    for(std::size_t J = 0; J < occupied_count; ++J) {
      for(std::size_t B = 0; B < virtual_count; ++B) {
        for(std::size_t j = 0; j < occupied_count; ++j) {
          second[(J * virtual_count + B) * pair_count + left] +=
              first[(j * virtual_count + B) * pair_count + left] *
              occupied_transform(j, J);
        }
      }
    }
  }
  return second;
}
#endif

struct CachedRotationResult {
  std::vector<double> values;
  double contraction_seconds;
  double transpose_seconds;
};

[[nodiscard]] CachedRotationResult rotate_cached_ovov(
    const std::vector<double>& canonical, std::size_t occupied_count,
    std::size_t virtual_count, const linalg::Matrix& occupied_transform,
    const linalg::Matrix& virtual_transform, std::size_t threads) {
  if(occupied_count != 0 &&
     virtual_count >
         std::numeric_limits<std::size_t>::max() / occupied_count) {
    throw std::overflow_error(
        "cached localized LMP2-1M2M pair count overflows size_t");
  }
  const std::size_t pair_count = occupied_count * virtual_count;
  if(pair_count != 0 &&
     pair_count > std::numeric_limits<std::size_t>::max() / pair_count) {
    throw std::overflow_error(
        "cached localized LMP2-1M2M tensor size overflows size_t");
  }
  if(canonical.size() != pair_count * pair_count ||
     occupied_transform.rows() != occupied_count ||
     occupied_transform.columns() != occupied_count ||
     virtual_transform.rows() != virtual_count ||
     virtual_transform.columns() != virtual_count) {
    throw std::invalid_argument(
        "cached localized LMP2-1M2M tensor/rotation dimensions are invalid");
  }
  std::vector<double> first(canonical.size(), 0.0);
  std::vector<double> second(canonical.size(), 0.0);

  const auto blas_dimension = [](std::size_t value, const char* name) {
    if(value >
       static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      throw std::overflow_error(std::string{name} +
                                " exceeds the BLAS integer range");
    }
    return static_cast<int>(value);
  };
  const int virtual_dimension =
      blas_dimension(virtual_count, "cached OVOV virtual dimension");
  const int occupied_dimension =
      blas_dimension(occupied_count, "cached OVOV occupied dimension");
  constexpr char no_transpose = 'N';
  constexpr char transpose = 'T';
  constexpr double alpha = 1.0;
  constexpr double beta = 0.0;

  const auto rotate_virtual_rows = [&](const std::vector<double>& input,
                                       std::vector<double>& output) {
    if(pair_count >
       std::numeric_limits<std::size_t>::max() / occupied_count) {
      throw std::overflow_error(
          "cached OVOV virtual rotation column count overflows size_t");
    }
    const std::size_t columns = pair_count * occupied_count;
    const int all_column_dimension = blas_dimension(
        columns, "cached OVOV virtual rotation column count");
#ifdef LMP2_1M2M_HAS_OPENMP
    (void)all_column_dimension;
#pragma omp parallel num_threads(threads)
    {
      const std::size_t worker =
          static_cast<std::size_t>(omp_get_thread_num());
      const std::size_t workers =
          static_cast<std::size_t>(omp_get_num_threads());
      const std::size_t base = columns / workers;
      const std::size_t remainder = columns % workers;
      const std::size_t first_column =
          worker * base + std::min(worker, remainder);
      const std::size_t local_columns =
          base + (worker < remainder ? 1U : 0U);
      if(local_columns != 0) {
        const int column_dimension = static_cast<int>(local_columns);
        dgemm_(&transpose, &no_transpose, &virtual_dimension,
               &column_dimension, &virtual_dimension, &alpha,
               virtual_transform.data(), &virtual_dimension,
               input.data() + first_column * virtual_count,
               &virtual_dimension, &beta,
               output.data() + first_column * virtual_count,
               &virtual_dimension);
      }
    }
#else
    dgemm_(&transpose, &no_transpose, &virtual_dimension,
           &all_column_dimension, &virtual_dimension, &alpha,
           virtual_transform.data(), &virtual_dimension, input.data(),
           &virtual_dimension, &beta, output.data(), &virtual_dimension);
#endif
  };

  const auto rotate_occupied_rows = [&](const std::vector<double>& input,
                                        std::vector<double>& output) {
#ifdef LMP2_1M2M_HAS_OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for(std::size_t other = 0; other < pair_count; ++other) {
      dgemm_(&no_transpose, &no_transpose, &virtual_dimension,
             &occupied_dimension, &occupied_dimension, &alpha,
             input.data() + other * pair_count, &virtual_dimension,
             occupied_transform.data(), &occupied_dimension, &beta,
             output.data() + other * pair_count, &virtual_dimension);
    }
  };

  const auto transpose_pair_matrix = [&](const std::vector<double>& input,
                                         std::vector<double>& output) {
    constexpr std::size_t block_size = 32;
    const std::size_t blocks =
        (pair_count + block_size - 1U) / block_size;
#ifdef LMP2_1M2M_HAS_OPENMP
#pragma omp parallel for collapse(2) schedule(static) num_threads(threads)
#endif
    for(std::size_t column_block = 0; column_block < blocks;
        ++column_block) {
      for(std::size_t row_block = 0; row_block < blocks; ++row_block) {
        const std::size_t first_column = column_block * block_size;
        const std::size_t last_column =
            std::min(pair_count, first_column + block_size);
        const std::size_t first_row = row_block * block_size;
        const std::size_t last_row =
            std::min(pair_count, first_row + block_size);
        for(std::size_t column = first_column; column < last_column;
            ++column) {
          for(std::size_t row = first_row; row < last_row; ++row) {
            output[row * pair_count + column] =
                input[column * pair_count + row];
          }
        }
      }
    }
  };

  // Apply the separable occupied/virtual transformation to the left pair,
  // transpose, and repeat for the right pair. BLAS stays serial inside each
  // OpenMP worker, matching the Perlmutter outer-threading policy.
  double contraction_seconds = 0.0;
  double transpose_seconds = 0.0;
  auto stage_start = std::chrono::steady_clock::now();
  rotate_virtual_rows(canonical, first);
  rotate_occupied_rows(first, second);
  contraction_seconds += elapsed_seconds(stage_start);
  stage_start = std::chrono::steady_clock::now();
  transpose_pair_matrix(second, first);
  transpose_seconds += elapsed_seconds(stage_start);
  stage_start = std::chrono::steady_clock::now();
  rotate_virtual_rows(first, second);
  rotate_occupied_rows(second, first);
  contraction_seconds += elapsed_seconds(stage_start);
  stage_start = std::chrono::steady_clock::now();
  transpose_pair_matrix(first, second);
  transpose_seconds += elapsed_seconds(stage_start);
#ifndef NDEBUG
  if(pair_count <= 64U) {
    const std::vector<double> reference =
        rotate_cached_ovov_scalar_reference(
            canonical, occupied_count, virtual_count, occupied_transform,
            virtual_transform);
    double maximum_scaled_error = 0.0;
    for(std::size_t index = 0; index < second.size(); ++index) {
      maximum_scaled_error = std::max(
          maximum_scaled_error,
          std::abs(second[index] - reference[index]) /
              std::max(1.0, std::abs(reference[index])));
    }
    if(maximum_scaled_error > 5.0e-12) {
      throw std::runtime_error(
          "BLAS cached OVOV rotation disagrees with the scalar reference");
    }
  }
#endif
  return CachedRotationResult{
      .values = std::move(second),
      .contraction_seconds = contraction_seconds,
      .transpose_seconds = transpose_seconds,
  };
}

struct CachedOvovMemory {
  std::size_t tensor_bytes;
  std::size_t build_peak_bytes;
  std::size_t rotation_peak_bytes;
};

[[nodiscard]] CachedOvovMemory cached_ovov_memory(
    std::size_t ao_count, std::size_t occupied_count,
    std::size_t virtual_count, std::size_t persistent_bytes) {
  const auto multiply = [](std::size_t left, std::size_t right,
                           const char* description) {
    if(left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
      throw std::overflow_error(std::string{description} + " overflows size_t");
    }
    return left * right;
  };
  const std::size_t ao_pairs = multiply(ao_count, ao_count,
                                        "cached OVOV AO-pair count");
  const std::size_t orbital_pairs = multiply(
      occupied_count, virtual_count, "cached OVOV orbital-pair count");
  const std::size_t tensor_elements = multiply(
      orbital_pairs, orbital_pairs, "cached OVOV tensor size");
  const std::size_t tensor_bytes = checked_bytes(
      tensor_elements, "cached OVOV tensor storage");
  const std::size_t pair_ao_elements = multiply(
      orbital_pairs, ao_pairs, "cached OVOV pair-AO workspace");
  const std::size_t pair_ao_bytes = checked_bytes(
      pair_ao_elements, "cached OVOV pair-AO workspace");
  std::size_t build_peak = persistent_bytes;
  build_peak = checked_add(build_peak, tensor_bytes,
                           "cached OVOV build peak");
  for(int copy = 0; copy < 3; ++copy) {
    build_peak = checked_add(build_peak, pair_ao_bytes,
                             "cached OVOV build peak");
  }
  std::size_t rotation_peak = persistent_bytes;
  for(int copy = 0; copy < 3; ++copy) {
    rotation_peak = checked_add(rotation_peak, tensor_bytes,
                                "cached OVOV rotation peak");
  }
  return CachedOvovMemory{
      .tensor_bytes = tensor_bytes,
      .build_peak_bytes = build_peak,
      .rotation_peak_bytes = rotation_peak,
  };
}

#ifdef LMP2_1M2M_HAS_MPI
void broadcast_doubles(std::vector<double>& values) {
  std::size_t offset = 0;
  while(offset < values.size()) {
    const int count = static_cast<int>(std::min(
        values.size() - offset,
        static_cast<std::size_t>(std::numeric_limits<int>::max())));
    if(MPI_Bcast(values.data() + offset, count, MPI_DOUBLE, 0,
                 MPI_COMM_WORLD) != MPI_SUCCESS) {
      throw std::runtime_error("cached OVOV MPI broadcast failed");
    }
    offset += static_cast<std::size_t>(count);
  }
}
#endif

#ifdef LMP2_1M2M_HAS_MPI
void reduce_results(std::array<double, maximum_mp2_body_order + 1>& energy,
                    std::array<std::uint64_t, 5>& counters,
                    std::size_t& peak_bytes) {
  if(MPI_Allreduce(MPI_IN_PLACE, energy.data(),
                   static_cast<int>(energy.size()), MPI_DOUBLE, MPI_SUM,
                   MPI_COMM_WORLD) != MPI_SUCCESS ||
     MPI_Allreduce(MPI_IN_PLACE, counters.data(),
                   static_cast<int>(counters.size()), MPI_UINT64_T, MPI_SUM,
                   MPI_COMM_WORLD) != MPI_SUCCESS) {
    throw std::runtime_error("localized LMP2-1M2M MPI reduction failed");
  }
  std::uint64_t peak = static_cast<std::uint64_t>(peak_bytes);
  if(MPI_Allreduce(MPI_IN_PLACE, &peak, 1, MPI_UINT64_T, MPI_MAX,
                   MPI_COMM_WORLD) != MPI_SUCCESS) {
    throw std::runtime_error("localized LMP2-1M2M peak-memory reduction failed");
  }
  if(peak > static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max())) {
    throw std::overflow_error("localized LMP2-1M2M peak memory exceeds size_t");
  }
  peak_bytes = static_cast<std::size_t>(peak);
}
#endif

struct DistributedCanonicalOvov {
  IndexRange canonical_right_pairs;
  std::vector<double> values;
  std::array<std::uint64_t, 5> counters{};
  std::size_t peak_bytes{0};
  double build_seconds{0.0};
  double communication_seconds{0.0};
};

#ifdef LMP2_1M2M_HAS_MPI
[[nodiscard]] DistributedCanonicalOvov
build_mpi_distributed_separable_canonical_ovov(
    const integrals::Libint2IntegralProvider& provider,
    const linalg::Matrix& canonical_occupied,
    const linalg::Matrix& canonical_virtual,
    const LocalizedTiledLaplaceMp2Options& options,
    std::size_t persistent_bytes, bool report_progress) {
  const std::size_t occupied_count = canonical_occupied.columns();
  const std::size_t virtual_count = canonical_virtual.columns();
  const std::size_t pair_count = checked_multiply(
      occupied_count, virtual_count,
      "distributed separable OVOV pair count");
  const IndexRange owned_right =
      balanced_range(pair_count, options.mpi_rank, options.mpi_ranks);
  const IndexRange owned_left_occupied =
      balanced_range(occupied_count, options.mpi_rank, options.mpi_ranks);
  const std::size_t owned_left_pairs = checked_multiply(
      owned_left_occupied.count, virtual_count,
      "distributed separable OVOV left-pair count");
  const std::size_t local_elements = checked_multiply(
      pair_count, owned_right.count,
      "distributed separable OVOV local slice");
  const std::size_t local_bytes = checked_bytes(
      local_elements, "distributed separable OVOV local slice");
  const std::size_t persistent_local_bytes = checked_add(
      persistent_bytes, local_bytes,
      "distributed separable OVOV persistent storage");
  if(persistent_local_bytes >= options.maximum_additional_memory_bytes) {
    throw std::runtime_error(
        "distributed separable OVOV local slice exceeds the memory limit");
  }
  if(options.mpi_rank >
         static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
     options.mpi_ranks >
         static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error(
        "distributed separable OVOV MPI topology exceeds int");
  }

  DistributedCanonicalOvov result{
      .canonical_right_pairs = owned_right,
      .values = std::vector<double>(local_elements, 0.0),
      .counters = {},
      .peak_bytes = persistent_local_bytes,
      .build_seconds = 0.0,
      .communication_seconds = 0.0,
  };
  const auto build_start = std::chrono::steady_clock::now();
  if(report_progress) {
    const std::size_t maximum_left_occupied =
        (occupied_count + options.mpi_ranks - 1U) / options.mpi_ranks;
    std::cerr
        << "progress distributed separable four-pass OVOV build started; "
        << "ranks=" << options.mpi_ranks
        << " source_pairs=" << pair_count
        << " rank_balanced_left_occupied_maximum="
        << maximum_left_occupied
        << " rank0_left_occupied_first=" << owned_left_occupied.first
        << " rank0_left_occupied_count=" << owned_left_occupied.count
        << " rank0_right_pair_first=" << owned_right.first
        << " rank0_right_pair_count=" << owned_right.count << '\n';
  }

  std::vector<double> transformed_values;
  std::size_t transformed_peak_bytes = 0;
  std::exception_ptr transform_error;
  const auto transform_start = std::chrono::steady_clock::now();
  try {
    if(owned_left_occupied.count != 0) {
      const std::size_t transform_budget =
          options.maximum_additional_memory_bytes - persistent_local_bytes;
      auto transformed = provider.compute_ovov_tile(
          canonical_occupied, canonical_virtual,
          integrals::OvovTileOptions{
              .left_occupied_first = owned_left_occupied.first,
              .left_occupied_count = owned_left_occupied.count,
              .left_virtual_first = 0,
              .left_virtual_count = virtual_count,
              .right_occupied_first = 0,
              .right_occupied_count = occupied_count,
              .right_virtual_first = 0,
              .right_virtual_count = virtual_count,
              .threads = options.threads,
              .schwarz_threshold = options.eri_schwarz_threshold,
              .require_separable_four_pass = true,
              .report_progress = report_progress,
              .maximum_additional_memory_bytes = transform_budget,
          });
      const std::size_t expected_elements = checked_multiply(
          pair_count, owned_left_pairs,
          "distributed separable OVOV transformed slice");
      if(transformed.values.size() != expected_elements) {
        throw std::runtime_error(
            "distributed separable OVOV transformed slice is inconsistent");
      }
      transformed_peak_bytes = transformed.estimated_peak_additional_bytes;
      transformed_values = std::move(transformed.values);
      result.counters[0] = 1U;
      result.counters[3] = transformed.evaluated_shell_quartets;
      result.counters[4] =
          transformed.schwarz_screened_shell_quartets;
    }
  } catch(...) {
    transform_error = std::current_exception();
  }

  int transform_failed = transform_error ? 1 : 0;
  if(MPI_Allreduce(MPI_IN_PLACE, &transform_failed, 1, MPI_INT, MPI_MAX,
                   MPI_COMM_WORLD) != MPI_SUCCESS) {
    throw std::runtime_error(
        "distributed separable OVOV transform failure reduction failed");
  }
  if(transform_failed != 0) {
    if(transform_error) {
      std::rethrow_exception(transform_error);
    }
    throw std::runtime_error(
        "another rank failed its distributed separable OVOV transform");
  }

  double maximum_transform_seconds = elapsed_seconds(transform_start);
  if(MPI_Allreduce(MPI_IN_PLACE, &maximum_transform_seconds, 1, MPI_DOUBLE,
                   MPI_MAX, MPI_COMM_WORLD) != MPI_SUCCESS) {
    throw std::runtime_error(
        "distributed separable OVOV transform-time reduction failed");
  }
  if(report_progress) {
    std::cerr
        << "progress distributed separable four-pass rank-local transforms "
           "complete; maximum_seconds="
        << maximum_transform_seconds << '\n';
  }

  // Each rank computed one unique left-occupied slice against all canonical
  // right pairs. Redistribute those columns into the final disjoint
  // right-pair ownership. Messages are bounded so neither MPI's int count nor
  // a temporary full tensor is required.
  constexpr std::size_t exchange_buffer_bytes =
      64U * 1024U * 1024U;
  constexpr std::size_t exchange_buffer_doubles =
      exchange_buffer_bytes / sizeof(double);
  static_assert(exchange_buffer_doubles <=
                static_cast<std::size_t>(
                    std::numeric_limits<int>::max()));
  const auto communication_start = std::chrono::steady_clock::now();
  for(std::size_t shift = 0; shift < options.mpi_ranks; ++shift) {
    const std::size_t destination =
        (options.mpi_rank + shift) % options.mpi_ranks;
    const std::size_t source =
        (options.mpi_rank + options.mpi_ranks - shift) %
        options.mpi_ranks;
    const IndexRange destination_right =
        balanced_range(pair_count, destination, options.mpi_ranks);
    const IndexRange source_left_occupied =
        balanced_range(occupied_count, source, options.mpi_ranks);
    const std::size_t source_left_pairs = checked_multiply(
        source_left_occupied.count, virtual_count,
        "distributed separable OVOV source left pairs");

    if(shift == 0) {
      if(owned_left_pairs != 0) {
        for(std::size_t right = 0; right < owned_right.count; ++right) {
          const double* source_values =
              transformed_values.data() +
              (owned_right.first + right) * owned_left_pairs;
          double* destination_values =
              result.values.data() + right * pair_count +
              owned_left_occupied.first * virtual_count;
          std::copy_n(source_values, owned_left_pairs,
                      destination_values);
        }
      }
      continue;
    }

    const std::size_t send_right_per_message =
        owned_left_pairs == 0
            ? destination_right.count
            : std::max<std::size_t>(
                  1U, exchange_buffer_doubles / owned_left_pairs);
    const std::size_t receive_right_per_message =
        source_left_pairs == 0
            ? owned_right.count
            : std::max<std::size_t>(
                  1U, exchange_buffer_doubles / source_left_pairs);
    const std::size_t send_messages =
        send_right_per_message == 0
            ? 0U
            : (destination_right.count + send_right_per_message - 1U) /
                  send_right_per_message;
    const std::size_t receive_messages =
        receive_right_per_message == 0
            ? 0U
            : (owned_right.count + receive_right_per_message - 1U) /
                  receive_right_per_message;
    const std::size_t messages =
        std::max(send_messages, receive_messages);
    std::vector<double> receive_buffer;
    for(std::size_t message = 0; message < messages; ++message) {
      const std::size_t send_right_first =
          message * send_right_per_message;
      const std::size_t receive_right_first =
          message * receive_right_per_message;
      const std::size_t send_right_count =
          send_right_first < destination_right.count
              ? std::min(send_right_per_message,
                         destination_right.count - send_right_first)
              : 0U;
      const std::size_t receive_right_count =
          receive_right_first < owned_right.count
              ? std::min(receive_right_per_message,
                         owned_right.count - receive_right_first)
              : 0U;
      const std::size_t send_count = checked_multiply(
          send_right_count, owned_left_pairs,
          "distributed separable OVOV send count");
      const std::size_t receive_count = checked_multiply(
          receive_right_count, source_left_pairs,
          "distributed separable OVOV receive count");
      if(send_count >
             static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
         receive_count >
             static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            "distributed separable OVOV message exceeds MPI int count");
      }
      const double* send_values =
          send_count == 0
              ? nullptr
              : transformed_values.data() +
                    (destination_right.first + send_right_first) *
                        owned_left_pairs;
      receive_buffer.resize(receive_count);
      if(MPI_Sendrecv(
             send_values, static_cast<int>(send_count), MPI_DOUBLE,
             static_cast<int>(destination), 0,
             receive_count == 0 ? nullptr : receive_buffer.data(),
             static_cast<int>(receive_count), MPI_DOUBLE,
             static_cast<int>(source), 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE) != MPI_SUCCESS) {
        throw std::runtime_error(
            "distributed separable OVOV bounded exchange failed");
      }
      if(source_left_pairs != 0) {
        for(std::size_t right = 0; right < receive_right_count; ++right) {
          const double* source_values =
              receive_buffer.data() + right * source_left_pairs;
          double* destination_values =
              result.values.data() +
              (receive_right_first + right) * pair_count +
              source_left_occupied.first * virtual_count;
          std::copy_n(source_values, source_left_pairs,
                      destination_values);
        }
      }
    }
    if(report_progress) {
      std::cerr
          << "progress distributed separable four-pass OVOV redistribution "
          << "shift=" << shift << '/' << (options.mpi_ranks - 1U)
          << " complete\n";
    }
  }
  result.communication_seconds = elapsed_seconds(communication_start);
  const std::size_t transformed_bytes = checked_bytes(
      transformed_values.size(),
      "distributed separable OVOV transformed storage");
  result.peak_bytes = std::max(
      result.peak_bytes,
      checked_add(
          persistent_local_bytes,
          std::max(transformed_peak_bytes,
                   checked_add(transformed_bytes, exchange_buffer_bytes,
                               "distributed separable OVOV exchange peak")),
          "distributed separable OVOV peak"));
  result.build_seconds = elapsed_seconds(build_start);
  if(report_progress) {
    std::cerr
        << "progress distributed separable four-pass OVOV build complete; "
        << "build_seconds=" << result.build_seconds
        << " communication_seconds=" << result.communication_seconds
        << " rank0_stored_bytes=" << local_bytes << '\n';
  }
  return result;
}
#endif

[[nodiscard]] DistributedCanonicalOvov build_distributed_canonical_ovov(
    const integrals::Libint2IntegralProvider& provider,
    const linalg::Matrix& canonical_occupied,
    const linalg::Matrix& canonical_virtual,
    const LocalizedTiledLaplaceMp2Options& options,
    std::size_t persistent_bytes, std::size_t left_occupied_batch_size,
    bool report_progress) {
#ifdef LMP2_1M2M_HAS_MPI
  if(options.mpi_ranks > 1) {
    return build_mpi_distributed_separable_canonical_ovov(
        provider, canonical_occupied, canonical_virtual, options,
        persistent_bytes, report_progress);
  }
#endif
  const std::size_t occupied_count = canonical_occupied.columns();
  const std::size_t virtual_count = canonical_virtual.columns();
  if(left_occupied_batch_size == 0) {
    throw std::invalid_argument(
        "canonical OVOV left occupied batch size is zero");
  }
  if(virtual_count >
     std::numeric_limits<std::size_t>::max() / occupied_count) {
    throw std::overflow_error("distributed canonical OVOV pair count overflows");
  }
  const std::size_t pair_count = occupied_count * virtual_count;
  const IndexRange owned =
      balanced_range(pair_count, options.mpi_rank, options.mpi_ranks);
  if(owned.count != 0 &&
     pair_count >
         std::numeric_limits<std::size_t>::max() / owned.count) {
    throw std::overflow_error(
        "distributed canonical OVOV local slice overflows size_t");
  }
  const std::size_t local_elements = pair_count * owned.count;
  const std::size_t local_bytes = checked_bytes(
      local_elements, "distributed canonical OVOV local slice");
  if(checked_add(persistent_bytes, local_bytes,
                 "distributed canonical OVOV persistent storage") >=
     options.maximum_additional_memory_bytes) {
    throw std::runtime_error(
        "distributed canonical OVOV local slice exceeds the memory limit");
  }

  DistributedCanonicalOvov result{
      .canonical_right_pairs = owned,
      .values = std::vector<double>(local_elements, 0.0),
      .counters = {},
      .peak_bytes = checked_add(persistent_bytes, local_bytes,
                                "distributed canonical OVOV storage"),
      .build_seconds = 0.0,
      .communication_seconds = 0.0,
  };
  std::exception_ptr build_error;
  const auto build_start = std::chrono::steady_clock::now();
  try {
    std::size_t global_pair = owned.first;
    std::size_t remaining = owned.count;
    std::size_t local_column = 0;
    while(remaining != 0) {
      const std::size_t occupied_first = global_pair / virtual_count;
      const std::size_t virtual_first = global_pair % virtual_count;
      std::size_t occupied_block = 1;
      std::size_t virtual_block =
          std::min(remaining, virtual_count - virtual_first);
      if(virtual_first == 0 && remaining >= virtual_count) {
        occupied_block = remaining / virtual_count;
        virtual_block = virtual_count;
      }
      const std::size_t right_pair_block =
          occupied_block * virtual_block;
      for(std::size_t left_occupied_first = 0;
          left_occupied_first < occupied_count;
          left_occupied_first += left_occupied_batch_size) {
        const std::size_t left_occupied_count = std::min(
            left_occupied_batch_size,
            occupied_count - left_occupied_first);
        const std::size_t left_pair_count = checked_multiply(
            left_occupied_count, virtual_count,
            "canonical OVOV left occupied batch");
        const std::size_t transform_budget =
            options.maximum_additional_memory_bytes - persistent_bytes -
            local_bytes;
        const auto transformed = provider.compute_ovov_tile(
            canonical_occupied, canonical_virtual,
            integrals::OvovTileOptions{
                .left_occupied_first = left_occupied_first,
                .left_occupied_count = left_occupied_count,
                .left_virtual_first = 0,
                .left_virtual_count = virtual_count,
                .right_occupied_first = occupied_first,
                .right_occupied_count = occupied_block,
                .right_virtual_first = virtual_first,
                .right_virtual_count = virtual_block,
                .threads = options.threads,
                .schwarz_threshold = options.eri_schwarz_threshold,
                .require_separable_four_pass = options.mpi_ranks == 1,
                .report_progress = report_progress,
                .maximum_additional_memory_bytes = transform_budget,
            });
        if(transformed.values.size() !=
           left_pair_count * right_pair_block) {
          throw std::runtime_error(
              "distributed canonical OVOV tile size is inconsistent");
        }
        for(std::size_t right = 0; right < right_pair_block; ++right) {
          const double* source = transformed.values.data() +
                                 right * left_pair_count;
          double* destination = result.values.data() +
                                (local_column + right) * pair_count +
                                left_occupied_first * virtual_count;
          std::copy_n(source, left_pair_count, destination);
        }
        result.counters[0] = checked_u64_add(
            result.counters[0], 1,
            "distributed canonical OVOV transform count");
        result.counters[3] = checked_u64_add(
            result.counters[3], transformed.evaluated_shell_quartets,
            "distributed canonical OVOV evaluated-quartet count");
        result.counters[4] = checked_u64_add(
            result.counters[4], transformed.schwarz_screened_shell_quartets,
            "distributed canonical OVOV screened-quartet count");
        result.peak_bytes = std::max(
            result.peak_bytes,
            checked_add(
                checked_add(persistent_bytes, local_bytes,
                            "distributed canonical OVOV build peak"),
                transformed.estimated_peak_additional_bytes,
                "distributed canonical OVOV build peak"));
      }
      global_pair += right_pair_block;
      local_column += right_pair_block;
      remaining -= right_pair_block;
    }
  } catch(...) {
    build_error = std::current_exception();
  }
  result.build_seconds = elapsed_seconds(build_start);

#ifdef LMP2_1M2M_HAS_MPI
  if(options.mpi_ranks > 1) {
    int failed = build_error ? 1 : 0;
    if(MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX,
                     MPI_COMM_WORLD) != MPI_SUCCESS) {
      throw std::runtime_error(
          "distributed canonical OVOV failure reduction failed");
    }
    if(failed != 0) {
      if(build_error) {
        std::rethrow_exception(build_error);
      }
      throw std::runtime_error(
          "another rank failed to build its distributed canonical OVOV slice");
    }
  }
#endif
  if(build_error) {
    std::rethrow_exception(build_error);
  }
  return result;
}

void reduce_panel_to_owner(const std::vector<double>& partial,
                           std::vector<double>& owned,
                           std::size_t owner,
                           const LocalizedTiledLaplaceMp2Options& options) {
  if(owner >= options.mpi_ranks) {
    throw std::invalid_argument("distributed OVOV panel owner is invalid");
  }
#ifdef LMP2_1M2M_HAS_MPI
  if(options.mpi_ranks > 1) {
    if(owner > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      throw std::overflow_error("distributed OVOV panel owner exceeds int");
    }
    if(options.mpi_rank == owner) {
      owned.assign(partial.size(), 0.0);
    } else {
      owned.clear();
    }
    std::size_t offset = 0;
    while(offset < partial.size()) {
      const int count = static_cast<int>(std::min(
          partial.size() - offset,
          static_cast<std::size_t>(std::numeric_limits<int>::max())));
      double* receive = options.mpi_rank == owner
                            ? owned.data() + offset
                            : nullptr;
      if(MPI_Reduce(partial.data() + offset, receive, count, MPI_DOUBLE,
                    MPI_SUM, static_cast<int>(owner),
                    MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error(
            "distributed selected OVOV panel reduction failed");
      }
      offset += static_cast<std::size_t>(count);
    }
    return;
  }
#endif
  owned = partial;
}

[[nodiscard]] std::vector<double> rotate_full_left_pairs(
    const std::vector<double>& right_transformed,
    std::size_t occupied_count, std::size_t virtual_count,
    const linalg::Matrix& occupied_transform,
    const linalg::Matrix& virtual_transform,
    std::size_t panel_columns, std::size_t threads) {
  const std::size_t pair_count = checked_multiply(
      occupied_count, virtual_count,
      "full localized OVOV pair count");
  const std::size_t expected = checked_multiply(
      pair_count, panel_columns,
      "full localized OVOV panel storage");
  if(right_transformed.size() != expected || panel_columns == 0 ||
     threads == 0 ||
     occupied_transform.rows() != occupied_count ||
     occupied_transform.columns() != occupied_count ||
     virtual_transform.rows() != virtual_count ||
     virtual_transform.columns() != virtual_count) {
    throw std::invalid_argument(
        "full localized OVOV left-rotation dimensions are invalid");
  }
  std::vector<double> virtual_rotated(expected, 0.0);
  std::vector<double> result(expected, 0.0);
  const int virtual_dimension = blas_dimension(
      virtual_count, "full localized OVOV virtual dimension");
  const int occupied_dimension = blas_dimension(
      occupied_count, "full localized OVOV occupied dimension");
  const std::size_t virtual_columns = checked_multiply(
      occupied_count, panel_columns,
      "full localized OVOV virtual-rotation columns");
  constexpr char no_transpose = 'N';
  constexpr char transpose = 'T';
  constexpr double alpha = 1.0;
  constexpr double beta = 0.0;

#ifdef LMP2_1M2M_HAS_OPENMP
#pragma omp parallel num_threads(static_cast<int>(threads))
  {
    const std::size_t worker =
        static_cast<std::size_t>(omp_get_thread_num());
    const std::size_t workers =
        static_cast<std::size_t>(omp_get_num_threads());
    const std::size_t base = virtual_columns / workers;
    const std::size_t remainder = virtual_columns % workers;
    const std::size_t first =
        worker * base + std::min(worker, remainder);
    const std::size_t count =
        base + (worker < remainder ? 1U : 0U);
    if(count != 0) {
      const int column_count = blas_dimension(
          count, "full localized OVOV virtual thread columns");
      dgemm_(&transpose, &no_transpose, &virtual_dimension,
             &column_count, &virtual_dimension, &alpha,
             virtual_transform.data(), &virtual_dimension,
             right_transformed.data() + first * virtual_count,
             &virtual_dimension, &beta,
             virtual_rotated.data() + first * virtual_count,
             &virtual_dimension);
    }
  }
#else
  const int all_columns = blas_dimension(
      virtual_columns, "full localized OVOV virtual columns");
  dgemm_(&transpose, &no_transpose, &virtual_dimension,
         &all_columns, &virtual_dimension, &alpha,
         virtual_transform.data(), &virtual_dimension,
         right_transformed.data(), &virtual_dimension, &beta,
         virtual_rotated.data(), &virtual_dimension);
#endif

#ifdef LMP2_1M2M_HAS_OPENMP
#pragma omp parallel for schedule(static) num_threads(static_cast<int>(threads))
#endif
  for(std::size_t column = 0; column < panel_columns; ++column) {
    dgemm_(&no_transpose, &no_transpose, &virtual_dimension,
           &occupied_dimension, &occupied_dimension, &alpha,
           virtual_rotated.data() + column * pair_count,
           &virtual_dimension, occupied_transform.data(),
           &occupied_dimension, &beta,
           result.data() + column * pair_count,
           &virtual_dimension);
  }
  return result;
}

void accumulate_full_localized_exchange_slice(
    const std::vector<double>& direct_local,
    const IndexRange& direct_right_pairs,
    const std::vector<double>& exchange_slice,
    const IndexRange& exchange_right_pairs,
    std::size_t occupied_count, std::size_t virtual_count,
    const std::vector<MonomerAssignment>& occupied_assignments,
    const std::vector<MonomerAssignment>& virtual_assignments,
    double weight, std::size_t threads,
    std::array<long double, maximum_mp2_body_order + 1>& energy) {
  const std::size_t pair_count = checked_multiply(
      occupied_count, virtual_count,
      "full localized energy pair count");
  if(direct_local.size() !=
         checked_multiply(pair_count, direct_right_pairs.count,
                          "full localized direct slice") ||
     exchange_slice.size() !=
         checked_multiply(pair_count, exchange_right_pairs.count,
                          "full localized exchange slice") ||
     occupied_assignments.size() != occupied_count ||
     virtual_assignments.size() != virtual_count || threads == 0 ||
     !std::isfinite(weight)) {
    throw std::invalid_argument(
        "full localized energy slice dimensions are invalid");
  }
  std::vector<std::array<long double,
                         maximum_mp2_body_order + 1>>
      thread_energy(threads);
#ifdef LMP2_1M2M_HAS_OPENMP
#pragma omp parallel for schedule(static) num_threads(static_cast<int>(threads))
#endif
  for(std::size_t exchange_column = 0;
      exchange_column < exchange_right_pairs.count; ++exchange_column) {
#ifdef LMP2_1M2M_HAS_OPENMP
    const std::size_t worker =
        static_cast<std::size_t>(omp_get_thread_num());
#else
    const std::size_t worker = 0;
#endif
    const std::size_t exchange_pair =
        exchange_right_pairs.first + exchange_column;
    const std::size_t j = exchange_pair / virtual_count;
    const std::size_t a = exchange_pair % virtual_count;
    const std::size_t j_first_pair = j * virtual_count;
    const std::size_t direct_first = std::max(
        direct_right_pairs.first, j_first_pair);
    const std::size_t direct_last = std::min(
        checked_add(direct_right_pairs.first,
                    direct_right_pairs.count,
                    "full localized direct range end"),
        checked_add(j_first_pair, virtual_count,
                    "full localized occupied-pair end"));
    if(direct_first >= direct_last) {
      continue;
    }
    const double* exchange_column_values =
        exchange_slice.data() + exchange_column * pair_count;
    for(std::size_t direct_pair = direct_first;
        direct_pair < direct_last; ++direct_pair) {
      const std::size_t b = direct_pair - j_first_pair;
      const double* direct_column = direct_local.data() +
          (direct_pair - direct_right_pairs.first) * pair_count;
      for(std::size_t i = 0; i < occupied_count; ++i) {
        const std::size_t direct_left = i * virtual_count + a;
        const std::size_t exchange_left = i * virtual_count + b;
        const double direct = direct_column[direct_left];
        const double exchange = exchange_column_values[exchange_left];
        const std::size_t body = excitation_body_order(
            occupied_assignments[i].monomer_id,
            occupied_assignments[j].monomer_id,
            virtual_assignments[a].monomer_id,
            virtual_assignments[b].monomer_id);
        thread_energy[worker][body] -=
            static_cast<long double>(weight) *
            static_cast<long double>(direct) *
            static_cast<long double>(2.0 * direct - exchange);
      }
    }
  }
  for(const auto& partial : thread_energy) {
    for(std::size_t body = 1; body <= maximum_mp2_body_order; ++body) {
      energy[body] += partial[body];
    }
  }
}

[[nodiscard]] SelectedRotationResult rotate_selected_left_pairs(
    const std::vector<double>& right_transformed,
    const LocalizedRightPanel& panel,
    const GroupedSpace& occupied, const GroupedSpace& virtuals,
    const linalg::Matrix& occupied_transform,
    const linalg::Matrix& virtual_transform,
    const OrderedBlockMask& active_blocks,
    std::size_t threads, const PointScreeningData* screening) {
  const std::size_t occupied_count = occupied.assignments.size();
  const std::size_t virtual_count = virtuals.assignments.size();
  const std::size_t pair_count = occupied_count * virtual_count;
  if(right_transformed.size() !=
     pair_count * panel.pair_indices.size()) {
    throw std::invalid_argument(
        "distributed selected OVOV right panel has an invalid size");
  }
  if(panel.occupied_block >= occupied.blocks.size() ||
     panel.virtual_block >= virtuals.blocks.size()) {
    throw std::invalid_argument(
        "selected OVOV panel block index is invalid");
  }
  if(screening != nullptr &&
     (screening->threshold_hartree <= 0.0 ||
      !std::isfinite(screening->threshold_hartree) ||
      !std::isfinite(screening->quadrature_weight) ||
      screening->pair_vector_norms == nullptr ||
      screening->one_sided_integral_norms == nullptr ||
      screening->pair_vector_norms->size() != pair_count ||
      screening->one_sided_integral_norms->size() != pair_count)) {
    throw std::invalid_argument(
        "selected OVOV point-screening data are invalid");
  }
  std::vector<std::vector<SelectedIntegral>> column_entries(
      panel.pair_indices.size());
  std::vector<std::uint64_t> column_screened(
      panel.pair_indices.size(), 0U);
  std::vector<long double> column_omitted_bounds(
      panel.pair_indices.size(), 0.0L);
  const int canonical_virtual =
      blas_dimension(virtual_count, "selected OVOV canonical virtual count");
  const int canonical_occupied =
      blas_dimension(occupied_count, "selected OVOV canonical occupied count");
  constexpr char no_transpose = 'N';
  constexpr char transpose = 'T';
  constexpr double alpha = 1.0;
  constexpr double beta = 0.0;
#ifdef LMP2_1M2M_HAS_OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
  for(std::size_t panel_column = 0;
      panel_column < panel.pair_indices.size(); ++panel_column) {
    const std::size_t right_pair = panel.pair_indices[panel_column];
    const double* right_column =
        right_transformed.data() + panel_column * pair_count;
    auto& entries = column_entries[panel_column];
    const std::size_t right_occupied = right_pair / virtual_count;
    const std::size_t right_virtual = right_pair % virtual_count;
    for(std::size_t virtual_block = 0;
        virtual_block < virtuals.blocks.size(); ++virtual_block) {
      const OrbitalBlock& virtual_range = virtuals.blocks[virtual_block];
      bool virtual_range_is_needed = false;
      std::vector<std::vector<unsigned char>> retained_by_occupied_block(
          occupied.blocks.size());
      for(std::size_t occupied_block = 0;
          occupied_block < occupied.blocks.size(); ++occupied_block) {
        const OrbitalBlock& occupied_range = occupied.blocks[occupied_block];
        if(!active_blocks(occupied_block, panel.occupied_block,
                          virtual_block, panel.virtual_block)) {
          continue;
        }
        auto& retained = retained_by_occupied_block[occupied_block];
        retained.assign(
            checked_multiply(occupied_range.count, virtual_range.count,
                             "selected point-screening mask"),
            static_cast<unsigned char>(1));
        if(screening != nullptr) {
          for(std::size_t local_i = 0;
              local_i < occupied_range.count; ++local_i) {
            const std::size_t localized_i = occupied_range.first + local_i;
            for(std::size_t local_a = 0;
                local_a < virtual_range.count; ++local_a) {
              const std::size_t localized_a = virtual_range.first + local_a;
              const std::size_t left_pair =
                  localized_i * virtual_count + localized_a;
              const std::size_t exchange_left =
                  localized_i * virtual_count + right_virtual;
              const std::size_t exchange_right =
                  right_occupied * virtual_count + localized_a;
              const long double direct_bound =
                  static_cast<long double>(
                      screening->pair_vector_norms->at(left_pair)) *
                  static_cast<long double>(
                      screening->one_sided_integral_norms->at(right_pair));
              const long double exchange_bound =
                  static_cast<long double>(
                      screening->pair_vector_norms->at(exchange_left)) *
                  static_cast<long double>(
                      screening->one_sided_integral_norms->at(exchange_right));
              const long double energy_bound =
                  std::abs(static_cast<long double>(
                      screening->quadrature_weight)) *
                  direct_bound * (2.0L * direct_bound + exchange_bound);
              const long double exchange_energy_bound =
                  std::abs(static_cast<long double>(
                      screening->quadrature_weight)) *
                  exchange_bound * (2.0L * exchange_bound + direct_bound);
              if(!std::isfinite(energy_bound) ||
                 !std::isfinite(exchange_energy_bound)) {
                throw std::runtime_error(
                    "selected OVOV point-screening bound is nonfinite");
              }
              const std::size_t mask_index =
                  local_i * virtual_range.count + local_a;
              if(energy_bound < static_cast<long double>(
                                     screening->threshold_hartree)) {
                ++column_screened[panel_column];
                column_omitted_bounds[panel_column] += energy_bound;
                retained[mask_index] =
                    exchange_energy_bound >= static_cast<long double>(
                        screening->threshold_hartree)
                        ? 2U
                        : 0U;
              } else {
                retained[mask_index] = 1U;
                virtual_range_is_needed = true;
              }
              if(retained[mask_index] != 0U) {
                virtual_range_is_needed = true;
              }
            }
          }
        } else {
          virtual_range_is_needed = true;
        }
      }
      if(!virtual_range_is_needed) {
        continue;
      }
      const int localized_virtual = blas_dimension(
          virtual_range.count, "selected OVOV localized virtual block");
      linalg::Matrix virtual_partial{
          virtual_range.count, occupied_count};
      dgemm_(&transpose, &no_transpose, &localized_virtual,
             &canonical_occupied, &canonical_virtual, &alpha,
             virtual_transform.data() +
                 virtual_range.first * virtual_count,
             &canonical_virtual, right_column, &canonical_virtual,
             &beta, virtual_partial.data(), &localized_virtual);
      for(std::size_t occupied_block = 0;
          occupied_block < occupied.blocks.size(); ++occupied_block) {
        const OrbitalBlock& occupied_range =
            occupied.blocks[occupied_block];
        if(!active_blocks(occupied_block, panel.occupied_block,
                          virtual_block, panel.virtual_block)) {
          continue;
        }
        const auto& retained =
            retained_by_occupied_block[occupied_block];
        if(std::ranges::none_of(retained, [](unsigned char value) {
             return value != 0U;
           })) {
          continue;
        }
        const std::size_t body = excitation_body_order(
            occupied_range.monomer,
            occupied.blocks[panel.occupied_block].monomer,
            virtual_range.monomer,
            virtuals.blocks[panel.virtual_block].monomer);
        const int localized_occupied = blas_dimension(
            occupied_range.count,
            "selected OVOV localized occupied block");
        linalg::Matrix selected_block{
            virtual_range.count, occupied_range.count};
        dgemm_(&no_transpose, &no_transpose, &localized_virtual,
               &localized_occupied, &canonical_occupied, &alpha,
               virtual_partial.data(), &localized_virtual,
               occupied_transform.data() +
                   occupied_range.first * occupied_count,
               &canonical_occupied, &beta, selected_block.data(),
               &localized_virtual);
        for(std::size_t local_i = 0;
            local_i < occupied_range.count; ++local_i) {
          const std::size_t I = occupied_range.first + local_i;
          for(std::size_t local_a = 0;
              local_a < virtual_range.count; ++local_a) {
            const std::size_t A = virtual_range.first + local_a;
            const std::size_t left_pair = I * virtual_count + A;
            if(retained[local_i * virtual_range.count + local_a] == 0U) {
              continue;
            }
            entries.push_back(SelectedIntegral{
                .key = selected_integral_key(
                    left_pair, right_pair, pair_count),
                .value = selected_block(local_a, local_i),
                .body_order = body,
                .accumulate_energy =
                    retained[local_i * virtual_range.count + local_a] == 1U,
            });
          }
        }
      }
    }
  }
  std::size_t entry_count = 0;
  for(const auto& entries : column_entries) {
    entry_count = checked_add(entry_count, entries.size(),
                              "selected OVOV panel entry count");
  }
  SelectedRotationResult result;
  result.entries.reserve(entry_count);
  for(auto& entries : column_entries) {
    for(const SelectedIntegral& entry : entries) {
      if(entry.accumulate_energy) {
        result.retained_point_contributions = checked_u64_add(
            result.retained_point_contributions, 1U,
            "selected retained point-contribution count");
      }
    }
    result.entries.insert(result.entries.end(),
                          std::make_move_iterator(entries.begin()),
                          std::make_move_iterator(entries.end()));
  }
  for(std::size_t column = 0; column < column_screened.size(); ++column) {
    result.screened_point_contributions = checked_u64_add(
        result.screened_point_contributions, column_screened[column],
        "selected point-screening contribution count");
    result.omitted_energy_bound_hartree += column_omitted_bounds[column];
  }
  return result;
}

[[nodiscard]] double find_selected_integral(
    const std::vector<SelectedIntegral>& sorted,
    std::uint64_t key) {
  const auto found = std::lower_bound(
      sorted.begin(), sorted.end(), key,
      [](const SelectedIntegral& entry, std::uint64_t requested) {
        return entry.key < requested;
      });
  if(found == sorted.end() || found->key != key) {
    throw std::runtime_error(
        "distributed selected OVOV exchange integral is missing");
  }
  return found->value;
}

[[nodiscard]] std::vector<double> exchange_values(
    const std::vector<SelectedIntegral>& local_entries,
    std::size_t occupied_count, std::size_t virtual_count,
    const LocalizedTiledLaplaceMp2Options& options,
    double& communication_seconds) {
  const std::size_t pair_count = occupied_count * virtual_count;
  std::vector<double> result(local_entries.size(), 0.0);
  std::vector<std::vector<std::uint64_t>> keys_by_rank(options.mpi_ranks);
  std::vector<std::vector<std::size_t>> indices_by_rank(options.mpi_ranks);
  int initial_lookup_failed = 0;
  for(std::size_t index = 0; index < local_entries.size(); ++index) {
    const std::uint64_t key = local_entries[index].key;
    const std::size_t right_pair =
        static_cast<std::size_t>(key / pair_count);
    const std::size_t left_pair =
        static_cast<std::size_t>(key % pair_count);
    const std::size_t I = left_pair / virtual_count;
    const std::size_t A = left_pair % virtual_count;
    const std::size_t J = right_pair / virtual_count;
    const std::size_t B = right_pair % virtual_count;
    const std::size_t exchange_left = I * virtual_count + B;
    const std::size_t exchange_right = J * virtual_count + A;
    const std::uint64_t exchange_key = selected_integral_key(
        exchange_left, exchange_right, pair_count);
    const std::size_t owner = balanced_owner(
        exchange_right, pair_count, options.mpi_ranks);
    if(owner == options.mpi_rank) {
      try {
        result[index] = find_selected_integral(local_entries, exchange_key);
      } catch(...) {
        initial_lookup_failed = 1;
      }
    } else {
      keys_by_rank[owner].push_back(exchange_key);
      indices_by_rank[owner].push_back(index);
    }
  }

#ifdef LMP2_1M2M_HAS_MPI
  if(options.mpi_ranks > 1) {
    if(MPI_Allreduce(MPI_IN_PLACE, &initial_lookup_failed, 1, MPI_INT,
                     MPI_MAX, MPI_COMM_WORLD) != MPI_SUCCESS) {
      throw std::runtime_error(
          "distributed selected OVOV local-lookup reduction failed");
    }
    if(initial_lookup_failed != 0) {
      throw std::runtime_error(
          "distributed selected OVOV local exchange lookup failed");
    }
    if(options.mpi_ranks >
       static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      throw std::overflow_error(
          "distributed selected OVOV rank count exceeds int");
    }
    std::vector<int> send_counts(options.mpi_ranks, 0);
    std::vector<int> receive_counts(options.mpi_ranks, 0);
    for(std::size_t rank = 0; rank < options.mpi_ranks; ++rank) {
      if(keys_by_rank[rank].size() >
         static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            "distributed selected OVOV request count exceeds int");
      }
      send_counts[rank] = static_cast<int>(keys_by_rank[rank].size());
    }
    const auto communication_start = std::chrono::steady_clock::now();
    if(MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                    receive_counts.data(), 1, MPI_INT,
                    MPI_COMM_WORLD) != MPI_SUCCESS) {
      throw std::runtime_error(
          "distributed selected OVOV request-count exchange failed");
    }
    std::vector<int> send_displacements(options.mpi_ranks, 0);
    std::vector<int> receive_displacements(options.mpi_ranks, 0);
    for(std::size_t rank = 1; rank < options.mpi_ranks; ++rank) {
      if(send_displacements[rank - 1] >
             std::numeric_limits<int>::max() - send_counts[rank - 1] ||
         receive_displacements[rank - 1] >
             std::numeric_limits<int>::max() - receive_counts[rank - 1]) {
        throw std::overflow_error(
            "distributed selected OVOV request displacement exceeds int");
      }
      send_displacements[rank] =
          send_displacements[rank - 1] + send_counts[rank - 1];
      receive_displacements[rank] =
          receive_displacements[rank - 1] + receive_counts[rank - 1];
    }
    const std::size_t send_total = options.mpi_ranks == 0
        ? 0U
        : static_cast<std::size_t>(send_displacements.back()) +
              static_cast<std::size_t>(send_counts.back());
    const std::size_t receive_total = options.mpi_ranks == 0
        ? 0U
        : static_cast<std::size_t>(receive_displacements.back()) +
              static_cast<std::size_t>(receive_counts.back());
    std::vector<std::uint64_t> send_keys;
    std::vector<std::size_t> send_indices;
    send_keys.reserve(send_total);
    send_indices.reserve(send_total);
    for(std::size_t rank = 0; rank < options.mpi_ranks; ++rank) {
      send_keys.insert(send_keys.end(), keys_by_rank[rank].begin(),
                       keys_by_rank[rank].end());
      send_indices.insert(send_indices.end(), indices_by_rank[rank].begin(),
                          indices_by_rank[rank].end());
    }
    std::vector<std::uint64_t> receive_keys(receive_total);
    std::uint64_t dummy_key = 0;
    if(MPI_Alltoallv(send_keys.empty() ? &dummy_key : send_keys.data(),
                     send_counts.data(),
                     send_displacements.data(), MPI_UINT64_T,
                     receive_keys.empty() ? &dummy_key : receive_keys.data(),
                     receive_counts.data(),
                     receive_displacements.data(), MPI_UINT64_T,
                     MPI_COMM_WORLD) != MPI_SUCCESS) {
      throw std::runtime_error(
          "distributed selected OVOV key exchange failed");
    }
    std::vector<double> response_values(receive_total, 0.0);
    int local_lookup_failed = 0;
    try {
      for(std::size_t index = 0; index < receive_total; ++index) {
        response_values[index] =
            find_selected_integral(local_entries, receive_keys[index]);
      }
    } catch(...) {
      local_lookup_failed = 1;
    }
    if(MPI_Allreduce(MPI_IN_PLACE, &local_lookup_failed, 1, MPI_INT,
                     MPI_MAX, MPI_COMM_WORLD) != MPI_SUCCESS) {
      throw std::runtime_error(
          "distributed selected OVOV lookup failure reduction failed");
    }
    if(local_lookup_failed != 0) {
      throw std::runtime_error(
          "distributed selected OVOV exchange lookup failed");
    }
    std::vector<double> returned_values(send_total, 0.0);
    double dummy_value = 0.0;
    if(MPI_Alltoallv(
                     response_values.empty() ? &dummy_value
                                             : response_values.data(),
                     receive_counts.data(),
                     receive_displacements.data(), MPI_DOUBLE,
                     returned_values.empty() ? &dummy_value
                                             : returned_values.data(),
                     send_counts.data(),
                     send_displacements.data(), MPI_DOUBLE,
                     MPI_COMM_WORLD) != MPI_SUCCESS) {
      throw std::runtime_error(
          "distributed selected OVOV value exchange failed");
    }
    for(std::size_t index = 0; index < send_total; ++index) {
      result[send_indices[index]] = returned_values[index];
    }
    communication_seconds += elapsed_seconds(communication_start);
  }
#else
  (void)communication_seconds;
#endif
  if(initial_lookup_failed != 0) {
    throw std::runtime_error(
        "distributed selected OVOV local exchange lookup failed");
  }
  return result;
}

void reduce_maximum_timings(std::array<double, 6>& timings,
                            const LocalizedTiledLaplaceMp2Options& options) {
#ifdef LMP2_1M2M_HAS_MPI
  if(options.mpi_ranks > 1 &&
     MPI_Allreduce(MPI_IN_PLACE, timings.data(),
                   static_cast<int>(timings.size()), MPI_DOUBLE, MPI_MAX,
                   MPI_COMM_WORLD) != MPI_SUCCESS) {
    throw std::runtime_error("localized LMP2-1M2M timing reduction failed");
  }
#else
  (void)options;
#endif
}

void reduce_maximum_size(std::size_t& value,
                         const LocalizedTiledLaplaceMp2Options& options,
                         const char* description) {
#ifdef LMP2_1M2M_HAS_MPI
  if(options.mpi_ranks > 1) {
    std::uint64_t reduced = static_cast<std::uint64_t>(value);
    if(MPI_Allreduce(MPI_IN_PLACE, &reduced, 1, MPI_UINT64_T, MPI_MAX,
                     MPI_COMM_WORLD) != MPI_SUCCESS) {
      throw std::runtime_error(std::string{description} +
                               " reduction failed");
    }
    if(reduced > static_cast<std::uint64_t>(
                     std::numeric_limits<std::size_t>::max())) {
      throw std::overflow_error(std::string{description} +
                                " exceeds size_t");
    }
    value = static_cast<std::size_t>(reduced);
  }
#else
  (void)options;
  (void)description;
#endif
}

void reduce_sum_size(std::size_t& value,
                     const LocalizedTiledLaplaceMp2Options& options,
                     const char* description) {
#ifdef LMP2_1M2M_HAS_MPI
  if(options.mpi_ranks > 1) {
    std::uint64_t reduced = static_cast<std::uint64_t>(value);
    if(MPI_Allreduce(MPI_IN_PLACE, &reduced, 1, MPI_UINT64_T, MPI_SUM,
                     MPI_COMM_WORLD) != MPI_SUCCESS) {
      throw std::runtime_error(std::string{description} +
                               " reduction failed");
    }
    if(reduced > static_cast<std::uint64_t>(
                     std::numeric_limits<std::size_t>::max())) {
      throw std::overflow_error(std::string{description} +
                                " exceeds size_t");
    }
    value = static_cast<std::size_t>(reduced);
  }
#else
  (void)options;
  (void)description;
#endif
}

void reduce_maximum_double(double& value,
                           const LocalizedTiledLaplaceMp2Options& options,
                           const char* description) {
#ifdef LMP2_1M2M_HAS_MPI
  if(options.mpi_ranks > 1 &&
     MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_DOUBLE, MPI_MAX,
                   MPI_COMM_WORLD) != MPI_SUCCESS) {
    throw std::runtime_error(std::string{description} +
                             " reduction failed");
  }
#else
  (void)value;
  (void)options;
  (void)description;
#endif
}

}  // namespace

LocalizedTiledLaplaceMp2Result compute_localized_tiled_laplace_mp2(
    const integrals::Libint2IntegralProvider& provider,
    const linalg::Matrix& canonical_coefficients,
    const std::vector<double>& orbital_energies,
    const std::vector<std::size_t>& active_occupied_indices,
    const std::vector<std::size_t>& virtual_indices,
    const linalg::Matrix& occupied_rotation,
    const linalg::Matrix& virtual_rotation,
    const std::vector<MonomerAssignment>& occupied_assignments,
    const std::vector<MonomerAssignment>& virtual_assignments,
    const LaplaceFitResult& fit,
    const LocalizedTiledLaplaceMp2Options& options) {
  if(options.occupied_tile_size == 0 || options.virtual_tile_size == 0 ||
     options.quadrature_batch_size == 0 ||
     options.lambda_ao_block_size == 0 ||
     options.maximum_additional_memory_bytes == 0 || options.threads == 0 ||
     !std::isfinite(options.eri_schwarz_threshold) ||
     options.eri_schwarz_threshold < 0.0 ||
     !std::isfinite(
         options.localized_energy_schwarz_threshold_hartree) ||
     options.localized_energy_schwarz_threshold_hartree < 0.0 ||
     !std::isfinite(options.localized_point_energy_threshold_hartree) ||
     options.localized_point_energy_threshold_hartree < 0.0 ||
     !std::isfinite(options.assignment_confidence_threshold) ||
     options.assignment_confidence_threshold < 0.0 ||
     options.assignment_confidence_threshold > 1.0) {
    throw std::invalid_argument("localized tiled LMP2-1M2M options are invalid");
  }
  validate_topology(options);
  const std::size_t ao_count = provider.basis_metadata().number_of_aos;
  if(canonical_coefficients.rows() != ao_count ||
     canonical_coefficients.columns() != orbital_energies.size() ||
     active_occupied_indices.empty() || virtual_indices.empty() ||
     occupied_rotation.rows() != active_occupied_indices.size() ||
     virtual_rotation.rows() != virtual_indices.size() ||
     orthogonality_error(occupied_rotation) > 2.0e-10 ||
     orthogonality_error(virtual_rotation) > 2.0e-10 || fit.nodes.empty() ||
     fit.nodes.size() != fit.weights.size()) {
    throw std::invalid_argument(
        "localized tiled LMP2-1M2M dimensions, rotations, or quadrature are invalid");
  }
  linalg::require_finite(canonical_coefficients,
                         "localized tiled LMP2-1M2M canonical coefficients");
  validate_assignments(occupied_assignments,
                       active_occupied_indices.size(), "occupied");
  validate_assignments(virtual_assignments, virtual_indices.size(), "virtual");

  std::vector<double> occupied_energies;
  std::vector<double> virtual_energies;
  occupied_energies.reserve(active_occupied_indices.size());
  virtual_energies.reserve(virtual_indices.size());
  for(const std::size_t index : active_occupied_indices) {
    if(index >= orbital_energies.size() ||
       !std::isfinite(orbital_energies[index])) {
      throw std::invalid_argument("localized LMP2-1M2M occupied energy is invalid");
    }
    occupied_energies.push_back(orbital_energies[index]);
  }
  for(const std::size_t index : virtual_indices) {
    if(index >= orbital_energies.size() ||
       !std::isfinite(orbital_energies[index])) {
      throw std::invalid_argument("localized LMP2-1M2M virtual energy is invalid");
    }
    virtual_energies.push_back(orbital_energies[index]);
  }
  const double occupied_maximum =
      *std::max_element(occupied_energies.begin(), occupied_energies.end());
  const double virtual_minimum =
      *std::min_element(virtual_energies.begin(), virtual_energies.end());
  if(virtual_minimum <= occupied_maximum) {
    throw std::runtime_error("localized LMP2-1M2M orbital energy gap is nonpositive");
  }
  const double energy_shift = 0.5 * (occupied_maximum + virtual_minimum);

  const GroupedSpace occupied = group_space(
      occupied_rotation, occupied_assignments, options.occupied_tile_size);
  const GroupedSpace virtuals = group_space(
      virtual_rotation, virtual_assignments, options.virtual_tile_size);
  std::uint64_t omitted_terms = 0;
  std::uint64_t omitted_tasks_per_point = 0;
  const std::vector<LocalizedTask> tasks = build_tasks(
      occupied.blocks, virtuals.blocks, options.execution_mode,
      omitted_terms, omitted_tasks_per_point);
  if(tasks.empty()) {
    throw std::runtime_error("localized LMP2-1M2M task space is empty");
  }
  std::vector<unsigned char> task_is_active(
      tasks.size(), static_cast<unsigned char>(1));
  std::uint64_t localized_screened_terms = 0;
  std::uint64_t localized_screened_tasks = 0;
  long double localized_screening_omitted_bound = 0.0L;
  std::uint64_t localized_screened_point_contributions = 0;
  long double localized_point_screening_omitted_bound = 0.0L;
  double localized_screening_seconds = 0.0;

  const linalg::Matrix canonical_occupied =
      select_columns(canonical_coefficients, active_occupied_indices);
  const linalg::Matrix canonical_virtual =
      select_columns(canonical_coefficients, virtual_indices);
  const std::size_t coefficient_elements = checked_add(
      canonical_occupied.size(), canonical_virtual.size(),
      "localized LMP2-1M2M coefficient storage");
  if(coefficient_elements >
     std::numeric_limits<std::size_t>::max() / 3U) {
    throw std::overflow_error(
        "localized LMP2-1M2M coefficient element count overflows size_t");
  }
  const std::size_t persistent_bytes = checked_bytes(
      3U * coefficient_elements,
      "localized LMP2-1M2M canonical/scaled/localized coefficient storage");
  if(persistent_bytes >= options.maximum_additional_memory_bytes) {
    throw std::runtime_error(
        "localized LMP2-1M2M coefficient storage exceeds the memory limit");
  }

  CachedOvovMemory cached_memory{};
  bool cached_estimate_valid = false;
  if(options.backend != LocalizedLaplaceMp2Backend::direct_tiled &&
     options.backend !=
         LocalizedLaplaceMp2Backend::localized_direct_selected) {
    try {
      cached_memory = cached_ovov_memory(
          ao_count, active_occupied_indices.size(), virtual_indices.size(),
          persistent_bytes);
      cached_estimate_valid = true;
    } catch(const std::overflow_error&) {
      if(options.backend ==
         LocalizedLaplaceMp2Backend::cached_canonical_ovov) {
        throw;
      }
    }
  }
  const bool cached_fits =
      cached_estimate_valid &&
      cached_memory.build_peak_bytes <=
          options.maximum_additional_memory_bytes &&
      cached_memory.rotation_peak_bytes <=
          options.maximum_additional_memory_bytes;
  LocalizedLaplaceMp2Backend backend = options.backend;
  if(backend == LocalizedLaplaceMp2Backend::automatic) {
    const bool localized_screen_requested =
        options.localized_energy_schwarz_threshold_hartree > 0.0 ||
        options.localized_point_energy_threshold_hartree > 0.0;
    if(localized_screen_requested &&
       options.execution_mode == NmExecutionMode::one_two_monomer) {
      backend = options.mpi_ranks == 1
                    ? LocalizedLaplaceMp2Backend::shared_selected_ovov
                    : LocalizedLaplaceMp2Backend::distributed_selected_ovov;
    } else {
      backend = cached_fits
                    ? LocalizedLaplaceMp2Backend::localized_cached_ovov
                    : LocalizedLaplaceMp2Backend::direct_tiled;
    }
  }
  if(backend == LocalizedLaplaceMp2Backend::cached_canonical_ovov &&
     options.execution_mode == NmExecutionMode::one_two_monomer) {
    backend = options.mpi_ranks == 1
                  ? LocalizedLaplaceMp2Backend::shared_selected_ovov
                  : LocalizedLaplaceMp2Backend::distributed_selected_ovov;
  }
  const bool selected_ovov_backend =
      backend == LocalizedLaplaceMp2Backend::localized_cached_ovov ||
      backend == LocalizedLaplaceMp2Backend::shared_selected_ovov ||
      backend == LocalizedLaplaceMp2Backend::replicated_selected_ovov ||
      backend == LocalizedLaplaceMp2Backend::distributed_selected_ovov;
  const bool localized_cached_backend =
      backend == LocalizedLaplaceMp2Backend::localized_cached_ovov;
  const bool localized_direct_selected_backend =
      backend ==
      LocalizedLaplaceMp2Backend::localized_direct_selected;
  if(localized_direct_selected_backend &&
     options.execution_mode != NmExecutionMode::one_two_monomer) {
    throw std::invalid_argument(
        "localized direct selected requires the 1M+2M execution mode");
  }
  if(selected_ovov_backend && !localized_cached_backend &&
     options.execution_mode != NmExecutionMode::one_two_monomer) {
    throw std::invalid_argument(
        "selected OVOV requires the 1M+2M execution mode");
  }
  if(localized_cached_backend &&
     (options.localized_energy_schwarz_threshold_hartree > 0.0 ||
      options.localized_point_energy_threshold_hartree > 0.0)) {
    throw std::invalid_argument(
        "localized cached OVOV currently requires both localized energy "
        "screens to be disabled for an exact comparison");
  }
  if(!selected_ovov_backend &&
     (options.localized_energy_schwarz_threshold_hartree > 0.0 ||
      options.localized_point_energy_threshold_hartree > 0.0)) {
    throw std::invalid_argument(
        "localized pre-energy screening requires a canonical selected OVOV "
        "backend");
  }
  if(backend == LocalizedLaplaceMp2Backend::shared_selected_ovov &&
     options.mpi_ranks != 1) {
    throw std::invalid_argument(
        "shared selected OVOV requires exactly one MPI rank");
  }
  if(backend == LocalizedLaplaceMp2Backend::replicated_selected_ovov &&
     options.mpi_ranks < 2) {
    throw std::invalid_argument(
        "replicated selected OVOV requires at least two MPI ranks");
  }
  if(backend == LocalizedLaplaceMp2Backend::cached_canonical_ovov &&
     !cached_fits) {
    throw std::runtime_error(
        "cached canonical OVOV backend requires a peak estimate of " +
        std::to_string(std::max(cached_memory.build_peak_bytes,
                                cached_memory.rotation_peak_bytes)) +
        " bytes, exceeding the explicit limit of " +
        std::to_string(options.maximum_additional_memory_bytes));
  }

  std::array<long double, maximum_mp2_body_order + 1> local_energy{};
  std::array<std::uint64_t, 5> counters{};
  std::size_t peak_bytes = persistent_bytes;
  std::size_t canonical_ovov_bytes = 0;
  std::size_t canonical_ovov_max_local_bytes = 0;
  std::size_t canonical_ovov_stored_bytes_global = 0;
  std::size_t localized_ovov_bytes = 0;
  std::size_t localized_ovov_max_local_bytes = 0;
  std::size_t localized_ovov_stored_bytes_global = 0;
  std::size_t distributed_panel_count = 0;
  std::array<double, 6> timings{};
  double direct_ovov_transform_seconds = 0.0;
  if(backend == LocalizedLaplaceMp2Backend::cached_canonical_ovov) {
    const std::size_t occupied_count = active_occupied_indices.size();
    const std::size_t virtual_count = virtual_indices.size();
    if(virtual_count >
       std::numeric_limits<std::size_t>::max() / occupied_count) {
      throw std::overflow_error("cached canonical OVOV pair count overflows");
    }
    const std::size_t pair_count = occupied_count * virtual_count;
    std::vector<double> canonical_ovov(
        cached_memory.tensor_bytes / sizeof(double), 0.0);
    std::exception_ptr build_error;
    const auto build_start = std::chrono::steady_clock::now();
    if(options.mpi_rank == 0) {
      try {
        const auto transformed = provider.compute_ovov_tile(
            canonical_occupied, canonical_virtual,
            integrals::OvovTileOptions{
                .left_occupied_first = 0,
                .left_occupied_count = occupied_count,
                .left_virtual_first = 0,
                .left_virtual_count = virtual_count,
                .right_occupied_first = 0,
                .right_occupied_count = occupied_count,
                .right_virtual_first = 0,
                .right_virtual_count = virtual_count,
                .threads = options.threads,
                .schwarz_threshold = options.eri_schwarz_threshold,
                .maximum_additional_memory_bytes =
                    options.maximum_additional_memory_bytes -
                    persistent_bytes,
            });
        canonical_ovov = transformed.values;
        counters[0] = 1;
        counters[3] = transformed.evaluated_shell_quartets;
        counters[4] = transformed.schwarz_screened_shell_quartets;
        peak_bytes = std::max(
            peak_bytes,
            checked_add(persistent_bytes,
                        transformed.estimated_peak_additional_bytes,
                        "cached canonical OVOV build peak"));
      } catch(...) {
        build_error = std::current_exception();
      }
      timings[0] += elapsed_seconds(build_start);
    }
#ifdef LMP2_1M2M_HAS_MPI
    if(options.mpi_ranks > 1) {
      int failed = build_error ? 1 : 0;
      if(MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX,
                       MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error(
            "cached canonical OVOV failure reduction failed");
      }
      if(failed != 0) {
        if(build_error) {
          std::rethrow_exception(build_error);
        }
        throw std::runtime_error(
            "rank zero failed to build the cached canonical OVOV tensor");
      }
    }
#endif
    if(build_error) {
      std::rethrow_exception(build_error);
    }
    const auto broadcast_start = std::chrono::steady_clock::now();
#ifdef LMP2_1M2M_HAS_MPI
    if(options.mpi_ranks > 1) {
      broadcast_doubles(canonical_ovov);
    }
#endif
    if(options.mpi_ranks > 1) {
      timings[1] += elapsed_seconds(broadcast_start);
    }
    canonical_ovov_bytes = cached_memory.tensor_bytes;
    canonical_ovov_max_local_bytes = cached_memory.tensor_bytes;
    peak_bytes = std::max(peak_bytes, cached_memory.rotation_peak_bytes);

    for(std::size_t point = options.mpi_rank; point < fit.nodes.size();
        point += options.mpi_ranks) {
      const double node = fit.nodes[point];
      const double weight = fit.weights[point];
      if(!std::isfinite(node) || node <= 0.0 || !std::isfinite(weight)) {
        throw std::invalid_argument("localized LMP2-1M2M quadrature is invalid");
      }
      const linalg::Matrix occupied_transform = scaled_rotation(
          occupied_energies, energy_shift, node, true, occupied.rotation);
      const linalg::Matrix virtual_transform = scaled_rotation(
          virtual_energies, energy_shift, node, false, virtuals.rotation);
      CachedRotationResult rotated = rotate_cached_ovov(
          canonical_ovov, occupied_count, virtual_count,
          occupied_transform, virtual_transform, options.threads);
      timings[2] += rotated.contraction_seconds;
      timings[3] += rotated.transpose_seconds;
      const std::vector<double>& localized_ovov = rotated.values;

      const auto accumulation_start = std::chrono::steady_clock::now();
      for(std::size_t I = 0; I < occupied_count; ++I) {
        for(std::size_t A = 0; A < virtual_count; ++A) {
          const std::size_t left = I * virtual_count + A;
          for(std::size_t J = 0; J < occupied_count; ++J) {
            for(std::size_t B = 0; B < virtual_count; ++B) {
              const std::size_t body = excitation_body_order(
                  occupied.assignments[I].monomer_id,
                  occupied.assignments[J].monomer_id,
                  virtuals.assignments[A].monomer_id,
                  virtuals.assignments[B].monomer_id);
              if(options.execution_mode ==
                     NmExecutionMode::one_two_monomer &&
                 body > 2) {
                continue;
              }
              const std::size_t right = J * virtual_count + B;
              const std::size_t exchange_left = I * virtual_count + B;
              const std::size_t exchange_right = J * virtual_count + A;
              const double direct_value =
                  localized_ovov[right * pair_count + left];
              const double exchange_value =
                  localized_ovov[exchange_right * pair_count +
                                 exchange_left];
              local_energy[body] -=
                  static_cast<long double>(weight) *
                  static_cast<long double>(direct_value) *
                  static_cast<long double>(2.0 * direct_value -
                                           exchange_value);
            }
          }
        }
      }
      timings[4] += elapsed_seconds(accumulation_start);
    }
  } else if(selected_ovov_backend) {
    const bool replicated =
        backend == LocalizedLaplaceMp2Backend::replicated_selected_ovov;
    const std::size_t occupied_count = active_occupied_indices.size();
    const std::size_t virtual_count = virtual_indices.size();
    if(virtual_count >
       std::numeric_limits<std::size_t>::max() / occupied_count) {
      throw std::overflow_error(
          "distributed selected OVOV pair count overflows size_t");
    }
    const std::size_t pair_count = occupied_count * virtual_count;
    if(pair_count > std::numeric_limits<std::size_t>::max() / pair_count) {
      throw std::overflow_error(
          "distributed selected OVOV global size overflows size_t");
    }
    const std::size_t logical_ovov_bytes = checked_bytes(
        pair_count * pair_count,
        "distributed selected OVOV global storage reference");
    if(localized_cached_backend) {
      localized_ovov_bytes = logical_ovov_bytes;
    } else {
      canonical_ovov_bytes = logical_ovov_bytes;
    }
    LocalizedTiledLaplaceMp2Options canonical_options = options;
    if(replicated) {
      // One rank owns one complete node-local canonical tensor.  MPI is used
      // only to divide independent quadrature points and to reduce energies.
      canonical_options.mpi_rank = 0;
      canonical_options.mpi_ranks = 1;
    }
    std::optional<linalg::Matrix> localized_source_occupied;
    std::optional<linalg::Matrix> localized_source_virtual;
    if(localized_cached_backend) {
      localized_source_occupied.emplace(
          linalg::multiply(canonical_occupied, occupied.rotation));
      localized_source_virtual.emplace(
          linalg::multiply(canonical_virtual, virtuals.rotation));
    }
    const linalg::Matrix& source_occupied = localized_cached_backend
        ? *localized_source_occupied
        : canonical_occupied;
    const linalg::Matrix& source_virtual = localized_cached_backend
        ? *localized_source_virtual
        : canonical_virtual;
    DistributedCanonicalOvov canonical;
    std::exception_ptr replicated_build_error;
    if(localized_cached_backend && options.mpi_rank == 0) {
      std::cerr
          << "progress localized-cached OVOV build started; "
             "source_basis=localized canonical_ovov_storage_bytes=0\n";
    }
    try {
      canonical = build_distributed_canonical_ovov(
          provider, source_occupied, source_virtual,
          canonical_options, persistent_bytes,
          replicated ? options.occupied_tile_size : occupied_count,
          options.mpi_rank == 0);
    } catch(...) {
      replicated_build_error = std::current_exception();
    }
#ifdef LMP2_1M2M_HAS_MPI
    if(replicated) {
      int failed = replicated_build_error ? 1 : 0;
      if(MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX,
                       MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error(
            "replicated canonical OVOV failure reduction failed");
      }
      if(failed != 0) {
        if(replicated_build_error) {
          std::rethrow_exception(replicated_build_error);
        }
        throw std::runtime_error(
            "another rank failed to build its replicated canonical OVOV "
            "tensor");
      }
    }
#endif
    if(replicated_build_error) {
      std::rethrow_exception(replicated_build_error);
    }
    counters = canonical.counters;
    peak_bytes = std::max(peak_bytes, canonical.peak_bytes);
    const std::size_t local_ovov_bytes = checked_bytes(
        canonical.values.size(),
        "distributed selected OVOV local storage");
    if(localized_cached_backend) {
      localized_ovov_max_local_bytes = local_ovov_bytes;
    } else {
      canonical_ovov_max_local_bytes = local_ovov_bytes;
    }
    timings[0] += canonical.build_seconds;
    timings[5] += canonical.communication_seconds;
    if(localized_cached_backend && options.mpi_rank == 0) {
      std::cerr << std::setprecision(17)
                << "progress localized-cached OVOV rank-local build "
                   "complete; build_seconds="
                << canonical.build_seconds
                << " rank0_stored_bytes=" << local_ovov_bytes << '\n';
    }

    if(options.occupied_tile_size >
       std::numeric_limits<std::size_t>::max() /
           options.virtual_tile_size) {
      throw std::overflow_error(
          "distributed selected OVOV panel size overflows size_t");
    }
    const std::size_t panel_size =
        options.occupied_tile_size * options.virtual_tile_size;
    const std::vector<LocalizedRightPanel> all_panels = localized_right_panels(
        occupied, virtuals, replicated ? 1U : options.mpi_ranks,
        panel_size);
    const std::size_t maximum_owned_right_pairs =
        checked_add(pair_count,
                    (replicated ? 1U : options.mpi_ranks) - 1U,
                    "distributed selected OVOV balanced ownership") /
        (replicated ? 1U : options.mpi_ranks);
    const bool full_localized_cached =
        localized_cached_backend &&
        options.execution_mode == NmExecutionMode::full;
    std::uint64_t selected_entry_count_u64 = 0;
    if(!full_localized_cached) {
      for(const LocalizedTask& task : tasks) {
        selected_entry_count_u64 = checked_u64_add(
            selected_entry_count_u64, task.term_count,
            "selected OVOV retained-entry upper bound");
      }
    }
    if(selected_entry_count_u64 >
       static_cast<std::uint64_t>(
           std::numeric_limits<std::size_t>::max())) {
      throw std::overflow_error(
          "selected OVOV retained-entry count exceeds size_t");
    }
    // No rank can retain more entries than the complete selected excitation
    // space.  Using pair_count times owned pairs grossly overestimates the
    // 1M+2M workspace for many-monomer systems and defeats the explicit
    // truncation.
    const std::size_t maximum_selected_entries =
        static_cast<std::size_t>(selected_entry_count_u64);
    constexpr std::size_t conservative_bytes_per_selected_entry =
        2U * sizeof(SelectedIntegral) + 2U * sizeof(std::uint64_t) +
        2U * sizeof(std::size_t) + 3U * sizeof(double);
    const std::size_t selected_workspace_upper_bound = checked_multiply(
        maximum_selected_entries,
        conservative_bytes_per_selected_entry,
        "distributed selected OVOV retained-workspace upper bound");
    const std::size_t maximum_local_canonical_pairs =
        maximum_owned_right_pairs;
    const std::size_t panel_partial_elements = checked_multiply(
        pair_count, panel_size,
        "distributed selected OVOV panel upper bound");
    const std::size_t panel_transform_elements = checked_multiply(
        maximum_local_canonical_pairs, panel_size,
        "distributed selected OVOV panel-transform upper bound");
    const std::size_t panel_workspace_upper_bound = checked_bytes(
        checked_add(checked_multiply(
                        panel_partial_elements, 2U,
                        "distributed selected OVOV reduced-panel upper bound"),
                    panel_transform_elements,
                    "distributed selected OVOV panel upper bound"),
        "distributed selected OVOV panel upper bound");
    const std::size_t selected_base_bytes = checked_add(
        persistent_bytes, local_ovov_bytes,
        "distributed selected OVOV persistent workspace");
    std::size_t localized_screening_workspace_bytes = 0;
    if(options.localized_energy_schwarz_threshold_hartree > 0.0 ||
       options.localized_point_energy_threshold_hartree > 0.0) {
      const std::size_t task_bound_bytes = checked_multiply(
          tasks.size(), sizeof(long double),
          "localized screening task-bound storage");
      const std::size_t pair_norm_bytes = checked_multiply(
          checked_multiply(pair_count, 2U,
                           "localized screening pair-norm storage"),
          sizeof(double), "localized screening pair-norm storage");
      const std::size_t ordered_block_count = checked_multiply(
          checked_multiply(occupied.blocks.size(), occupied.blocks.size(),
                           "localized screening occupied-block mask"),
          checked_multiply(virtuals.blocks.size(), virtuals.blocks.size(),
                           "localized screening virtual-block mask"),
          "localized screening ordered-block mask");
      localized_screening_workspace_bytes = checked_add(
          checked_add(task_bound_bytes, pair_norm_bytes,
                      "localized screening workspace"),
          checked_add(task_is_active.size(), ordered_block_count,
                      "localized screening masks"),
          "localized screening workspace");
    }
    const std::size_t full_panel_workspace_upper_bound = checked_bytes(
        checked_add(
            checked_multiply(panel_partial_elements, 4U,
                             "full localized OVOV panel temporaries"),
            panel_transform_elements,
            "full localized OVOV panel workspace"),
        "full localized OVOV panel workspace");
    const std::size_t full_exchange_slice_bytes = checked_bytes(
        checked_multiply(pair_count, maximum_owned_right_pairs,
                         "full localized OVOV exchange slice"),
        "full localized OVOV exchange slice");
    const std::size_t selected_peak_upper_bound =
        full_localized_cached
            ? checked_add(
                  checked_add(selected_base_bytes, local_ovov_bytes,
                              "full localized OVOV scaled storage"),
                  std::max(full_panel_workspace_upper_bound,
                           full_exchange_slice_bytes),
                  "full localized OVOV working-memory upper bound")
            : checked_add(
                  selected_base_bytes,
                  std::max(
                      selected_workspace_upper_bound,
                      checked_add(panel_workspace_upper_bound,
                                  localized_screening_workspace_bytes,
                                  "localized screening panel workspace")),
                  "distributed selected OVOV working-memory upper bound");
    if(selected_peak_upper_bound >
       options.maximum_additional_memory_bytes) {
      throw std::runtime_error(
          "distributed selected OVOV conservative working-memory upper "
          "bound of " + std::to_string(selected_peak_upper_bound) +
          " bytes exceeds the explicit limit of " +
          std::to_string(options.maximum_additional_memory_bytes));
    }
    const int pair_dimension = blas_dimension(
        pair_count, "distributed selected OVOV pair count");
    const int local_canonical_pairs = blas_dimension(
        canonical.canonical_right_pairs.count,
        "distributed selected OVOV local canonical pairs");
    constexpr char no_transpose = 'N';
    constexpr double alpha = 1.0;
    constexpr double beta = 0.0;

    const auto project_right_panel = [&](
        const LocalizedRightPanel& panel,
        const linalg::Matrix& occupied_transform,
        const linalg::Matrix& virtual_transform,
        bool screening_pass) {
      const std::size_t panel_columns = panel.pair_indices.size();
      if(canonical.canonical_right_pairs.count != 0 &&
         panel_columns >
             std::numeric_limits<std::size_t>::max() /
                 canonical.canonical_right_pairs.count) {
        throw std::overflow_error(
            "distributed selected OVOV right transform overflows size_t");
      }
      linalg::Matrix right_transform{
          canonical.canonical_right_pairs.count, panel_columns};
      for(std::size_t panel_column = 0;
          panel_column < panel_columns; ++panel_column) {
        const std::size_t localized_pair =
            panel.pair_indices[panel_column];
        const std::size_t J = localized_pair / virtual_count;
        const std::size_t B = localized_pair % virtual_count;
        for(std::size_t local_pair = 0;
            local_pair < canonical.canonical_right_pairs.count;
            ++local_pair) {
          const std::size_t canonical_pair =
              canonical.canonical_right_pairs.first + local_pair;
          const std::size_t j = canonical_pair / virtual_count;
          const std::size_t b = canonical_pair % virtual_count;
          right_transform(local_pair, panel_column) =
              occupied_transform(j, J) * virtual_transform(b, B);
        }
      }
      if(pair_count >
         std::numeric_limits<std::size_t>::max() / panel_columns) {
        throw std::overflow_error(
            "distributed selected OVOV panel storage overflows size_t");
      }
      std::vector<double> partial(pair_count * panel_columns, 0.0);
      const int panel_dimension = blas_dimension(
          panel_columns, "distributed selected OVOV panel width");
      if(local_canonical_pairs != 0) {
        const auto rotation_start = std::chrono::steady_clock::now();
#ifdef LMP2_1M2M_HAS_OPENMP
#pragma omp parallel num_threads(static_cast<int>(options.threads))
        {
          const std::size_t worker =
              static_cast<std::size_t>(omp_get_thread_num());
          const std::size_t workers =
              static_cast<std::size_t>(omp_get_num_threads());
          const std::size_t base_rows = pair_count / workers;
          const std::size_t remainder_rows = pair_count % workers;
          const std::size_t first_row =
              worker * base_rows + std::min(worker, remainder_rows);
          const std::size_t row_count =
              base_rows + (worker < remainder_rows ? 1U : 0U);
          if(row_count != 0) {
            const int row_dimension = blas_dimension(
                row_count, "selected OVOV OpenMP row block");
            dgemm_(&no_transpose, &no_transpose, &row_dimension,
                   &panel_dimension, &local_canonical_pairs, &alpha,
                   canonical.values.data() + first_row,
                   &pair_dimension, right_transform.data(),
                   &local_canonical_pairs, &beta,
                   partial.data() + first_row, &pair_dimension);
          }
        }
#else
        dgemm_(&no_transpose, &no_transpose, &pair_dimension,
               &panel_dimension, &local_canonical_pairs, &alpha,
               canonical.values.data(), &pair_dimension,
               right_transform.data(), &local_canonical_pairs, &beta,
               partial.data(), &pair_dimension);
#endif
        if(!screening_pass) {
          timings[2] += elapsed_seconds(rotation_start);
        }
      }
      std::vector<double> owned_panel;
      if(replicated) {
        owned_panel = std::move(partial);
      } else if(options.mpi_ranks > 1) {
        const auto communication_start =
            std::chrono::steady_clock::now();
        reduce_panel_to_owner(partial, owned_panel, panel.owner, options);
        timings[5] += elapsed_seconds(communication_start);
      } else {
        reduce_panel_to_owner(partial, owned_panel, panel.owner, options);
      }
      const std::size_t panel_elements = checked_add(
          checked_add(partial.size(), owned_panel.size(),
                      "distributed selected OVOV panel storage"),
          right_transform.size(),
          "distributed selected OVOV panel storage");
      const std::size_t panel_bytes = checked_bytes(
          panel_elements, "distributed selected OVOV panel storage");
      peak_bytes = std::max(
          peak_bytes,
          checked_add(checked_add(persistent_bytes,
                                  local_ovov_bytes,
                                  "distributed selected OVOV peak"),
                      panel_bytes, "distributed selected OVOV peak"));
      return owned_panel;
    };

    if(full_localized_cached) {
      distributed_panel_count = all_panels.size();
      for(std::size_t point = 0; point < fit.nodes.size(); ++point) {
        const auto point_start = std::chrono::steady_clock::now();
        const double node = fit.nodes[point];
        const double weight = fit.weights[point];
        if(!std::isfinite(node) || node <= 0.0 ||
           !std::isfinite(weight)) {
          throw std::invalid_argument(
              "full localized cached quadrature is invalid");
        }
        const linalg::Matrix occupied_transform = localized_propagator(
            occupied_energies, energy_shift, node, true,
            occupied.rotation);
        const linalg::Matrix virtual_transform = localized_propagator(
            virtual_energies, energy_shift, node, false,
            virtuals.rotation);
        std::vector<double> scaled_local(canonical.values.size(), 0.0);
        for(const LocalizedRightPanel& panel : all_panels) {
          std::vector<double> owned_panel = project_right_panel(
              panel, occupied_transform, virtual_transform, false);
          if(options.mpi_rank != panel.owner) {
            continue;
          }
          const auto left_start = std::chrono::steady_clock::now();
          std::vector<double> fully_rotated = rotate_full_left_pairs(
              owned_panel, occupied_count, virtual_count,
              occupied_transform, virtual_transform,
              panel.pair_indices.size(), options.threads);
          timings[2] += elapsed_seconds(left_start);
          for(std::size_t column = 0;
              column < panel.pair_indices.size(); ++column) {
            const std::size_t global_pair = panel.pair_indices[column];
            if(global_pair < canonical.canonical_right_pairs.first ||
               global_pair >= checked_add(
                   canonical.canonical_right_pairs.first,
                   canonical.canonical_right_pairs.count,
                   "full localized owned-pair end")) {
              throw std::logic_error(
                  "full localized panel owner is inconsistent");
            }
            std::copy_n(
                fully_rotated.data() + column * pair_count,
                pair_count,
                scaled_local.data() +
                    (global_pair -
                     canonical.canonical_right_pairs.first) * pair_count);
          }
          peak_bytes = std::max(
              peak_bytes,
              checked_add(
                  checked_add(persistent_bytes,
                              checked_multiply(local_ovov_bytes, 2U,
                                               "full localized source and "
                                               "scaled storage"),
                              "full localized persistent storage"),
                  checked_bytes(
                      checked_multiply(
                          checked_multiply(
                              pair_count, panel.pair_indices.size(),
                              "full localized rotated panel elements"),
                          2U,
                          "full localized rotated panel copies"),
                      "full localized rotated panel storage"),
                  "full localized panel peak"));
        }

        std::array<long double,
                   maximum_mp2_body_order + 1> point_energy{};
        auto accumulation_start = std::chrono::steady_clock::now();
        accumulate_full_localized_exchange_slice(
            scaled_local, canonical.canonical_right_pairs,
            scaled_local, canonical.canonical_right_pairs,
            occupied_count, virtual_count, occupied.assignments,
            virtuals.assignments, weight, options.threads,
            point_energy);
        timings[4] += elapsed_seconds(accumulation_start);

#ifdef LMP2_1M2M_HAS_MPI
        if(options.mpi_ranks > 1) {
          constexpr std::size_t message_bytes =
              64U * 1024U * 1024U;
          constexpr std::size_t message_doubles =
              message_bytes / sizeof(double);
          for(std::size_t shift = 1; shift < options.mpi_ranks;
              ++shift) {
            const std::size_t destination =
                (options.mpi_rank + shift) % options.mpi_ranks;
            const std::size_t source =
                (options.mpi_rank + options.mpi_ranks - shift) %
                options.mpi_ranks;
            const IndexRange source_range = balanced_range(
                pair_count, source, options.mpi_ranks);
            const std::size_t receive_elements = checked_multiply(
                pair_count, source_range.count,
                "full localized exchange receive slice");
            std::vector<double> receive(receive_elements, 0.0);
            const std::size_t send_messages =
                (scaled_local.size() + message_doubles - 1U) /
                message_doubles;
            const std::size_t receive_messages =
                (receive.size() + message_doubles - 1U) /
                message_doubles;
            const std::size_t messages =
                std::max(send_messages, receive_messages);
            const auto communication_start =
                std::chrono::steady_clock::now();
            for(std::size_t message = 0; message < messages;
                ++message) {
              const std::size_t send_first =
                  message * message_doubles;
              const std::size_t receive_first =
                  message * message_doubles;
              const std::size_t send_count =
                  send_first < scaled_local.size()
                      ? std::min(message_doubles,
                                 scaled_local.size() - send_first)
                      : 0U;
              const std::size_t receive_count =
                  receive_first < receive.size()
                      ? std::min(message_doubles,
                                 receive.size() - receive_first)
                      : 0U;
              if(MPI_Sendrecv(
                     send_count == 0
                         ? nullptr
                         : scaled_local.data() + send_first,
                     static_cast<int>(send_count), MPI_DOUBLE,
                     static_cast<int>(destination), 11,
                     receive_count == 0
                         ? nullptr
                         : receive.data() + receive_first,
                     static_cast<int>(receive_count), MPI_DOUBLE,
                     static_cast<int>(source), 11, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE) != MPI_SUCCESS) {
                throw std::runtime_error(
                    "full localized exchange-slice communication failed");
              }
            }
            timings[5] += elapsed_seconds(communication_start);
            accumulation_start = std::chrono::steady_clock::now();
            accumulate_full_localized_exchange_slice(
                scaled_local, canonical.canonical_right_pairs,
                receive, source_range, occupied_count, virtual_count,
                occupied.assignments, virtuals.assignments, weight,
                options.threads, point_energy);
            timings[4] += elapsed_seconds(accumulation_start);
            peak_bytes = std::max(
                peak_bytes,
                checked_add(
                    checked_add(
                        persistent_bytes,
                        checked_multiply(
                            local_ovov_bytes, 2U,
                            "full localized source and scaled storage"),
                        "full localized persistent storage"),
                    checked_bytes(receive.size(),
                                  "full localized exchange receive"),
                    "full localized exchange peak"));
          }
        }
#endif
        for(std::size_t body = 1; body <= maximum_mp2_body_order;
            ++body) {
          local_energy[body] += point_energy[body];
        }
        if(options.report_laplace_point_progress) {
          std::array<long double,
                     maximum_mp2_body_order + 1> reported = point_energy;
#ifdef LMP2_1M2M_HAS_MPI
          if(options.mpi_ranks > 1 &&
             MPI_Allreduce(MPI_IN_PLACE, reported.data(),
                           static_cast<int>(reported.size()),
                           MPI_LONG_DOUBLE, MPI_SUM,
                           MPI_COMM_WORLD) != MPI_SUCCESS) {
            throw std::runtime_error(
                "full localized point-energy reduction failed");
          }
#endif
          if(options.mpi_rank == 0) {
            std::cerr << std::setprecision(17)
                      << "progress full localized Laplace point "
                      << (point + 1U) << '/' << fit.nodes.size()
                      << " complete; node=" << node
                      << " weight=" << weight
                      << " energy_1m_hartree="
                      << static_cast<double>(reported[1])
                      << " energy_2m_hartree="
                      << static_cast<double>(reported[2])
                      << " energy_3m_hartree="
                      << static_cast<double>(reported[3])
                      << " energy_4m_hartree="
                      << static_cast<double>(reported[4])
                      << " seconds=" << elapsed_seconds(point_start)
                      << '\n';
          }
        }
      }
    }

    if(!full_localized_cached) {

    const auto one_sided_screening_norms = [&]
        (const linalg::Matrix& occupied_transform,
         const linalg::Matrix& virtual_transform) {
      std::vector<double> norms(pair_count, 0.0);
      for(const LocalizedRightPanel& panel : all_panels) {
        const std::vector<double> owned_panel = project_right_panel(
            panel, occupied_transform, virtual_transform, true);
        if(replicated || options.mpi_rank == panel.owner) {
          if(owned_panel.size() !=
             pair_count * panel.pair_indices.size()) {
            throw std::logic_error(
                "localized screening panel size is inconsistent");
          }
          for(std::size_t panel_column = 0;
              panel_column < panel.pair_indices.size(); ++panel_column) {
            long double squared_norm = 0.0L;
            const double* column =
                owned_panel.data() + panel_column * pair_count;
            for(std::size_t row = 0; row < pair_count; ++row) {
              const long double value = column[row];
              squared_norm += value * value;
            }
            const double norm =
                std::sqrt(static_cast<double>(squared_norm));
            if(!std::isfinite(norm)) {
              throw std::runtime_error(
                  "localized screening produced a nonfinite norm");
            }
            norms[panel.pair_indices[panel_column]] = norm;
          }
        }
      }
#ifdef LMP2_1M2M_HAS_MPI
      if(!replicated && options.mpi_ranks > 1) {
        std::size_t offset = 0;
        while(offset < norms.size()) {
          const int count = static_cast<int>(std::min(
              norms.size() - offset,
              static_cast<std::size_t>(
                  std::numeric_limits<int>::max())));
          if(MPI_Allreduce(MPI_IN_PLACE, norms.data() + offset,
                           count, MPI_DOUBLE, MPI_MAX,
                           MPI_COMM_WORLD) != MPI_SUCCESS) {
            throw std::runtime_error(
                "localized screening norm reduction failed");
          }
          offset += static_cast<std::size_t>(count);
        }
      }
#endif
      return norms;
    };

    if(options.localized_energy_schwarz_threshold_hartree > 0.0) {
      const auto screening_start = std::chrono::steady_clock::now();
      std::vector<long double> task_bounds(tasks.size(), 0.0L);
      const std::size_t screening_first_point =
          replicated ? options.mpi_rank : 0U;
      const std::size_t screening_point_stride =
          replicated ? options.mpi_ranks : 1U;
      for(std::size_t point = screening_first_point;
          point < fit.nodes.size(); point += screening_point_stride) {
        const double node = fit.nodes[point];
        const double weight = fit.weights[point];
        if(!std::isfinite(node) || node <= 0.0 ||
           !std::isfinite(weight)) {
          throw std::invalid_argument(
              "localized LMP2-1M2M screening quadrature is invalid");
        }
        const linalg::Matrix occupied_transform = localized_cached_backend
            ? localized_propagator(occupied_energies, energy_shift, node,
                                   true, occupied.rotation)
            : scaled_rotation(occupied_energies, energy_shift, node, true,
                              occupied.rotation);
        const linalg::Matrix virtual_transform = localized_cached_backend
            ? localized_propagator(virtual_energies, energy_shift, node,
                                   false, virtuals.rotation)
            : scaled_rotation(virtual_energies, energy_shift, node, false,
                              virtuals.rotation);
        const std::vector<double> right_vector_norms =
            scaled_pair_vector_norms(occupied_transform,
                                     virtual_transform);
        const std::vector<double> one_sided_integral_norms =
            one_sided_screening_norms(occupied_transform,
                                      virtual_transform);
        for(std::size_t task_index = 0; task_index < tasks.size();
            ++task_index) {
          task_bounds[task_index] += localized_task_point_energy_bound(
              tasks[task_index], occupied, virtuals,
              right_vector_norms, one_sided_integral_norms, weight);
        }
      }
#ifdef LMP2_1M2M_HAS_MPI
      if(replicated && options.mpi_ranks > 1) {
        std::size_t offset = 0;
        while(offset < task_bounds.size()) {
          const int count = static_cast<int>(std::min(
              task_bounds.size() - offset,
              static_cast<std::size_t>(
                  std::numeric_limits<int>::max())));
          if(MPI_Allreduce(MPI_IN_PLACE, task_bounds.data() + offset,
                           count, MPI_LONG_DOUBLE, MPI_SUM,
                           MPI_COMM_WORLD) != MPI_SUCCESS) {
            throw std::runtime_error(
                "localized screening task-bound reduction failed");
          }
          offset += static_cast<std::size_t>(count);
        }
      }
#endif
      for(std::size_t task_index = 0; task_index < tasks.size();
          ++task_index) {
        const long double bound = task_bounds[task_index];
        if(bound < static_cast<long double>(
                       options.localized_energy_schwarz_threshold_hartree)) {
          task_is_active[task_index] = 0U;
          localized_screened_terms = checked_u64_add(
              localized_screened_terms, tasks[task_index].term_count,
              "localized screened excitation count");
          localized_screened_tasks = checked_u64_add(
              localized_screened_tasks, 1U,
              "localized screened block-task count");
          omitted_terms = checked_u64_add(
              omitted_terms, tasks[task_index].term_count,
              "localized omitted excitation count");
          omitted_tasks_per_point = checked_u64_add(
              omitted_tasks_per_point, 1U,
              "localized omitted block-task count");
          localized_screening_omitted_bound +=
              bound * static_cast<long double>(tasks[task_index].term_count);
        }
      }
      localized_screening_seconds = elapsed_seconds(screening_start);
    }

    const OrderedBlockMask active_blocks = build_ordered_block_mask(
        occupied.blocks.size(), virtuals.blocks.size(), tasks,
        task_is_active);
    std::vector<LocalizedRightPanel> panels;
    panels.reserve(all_panels.size());
    for(const LocalizedRightPanel& panel : all_panels) {
      if(panel_has_active_left_blocks(panel, active_blocks)) {
        panels.push_back(panel);
      }
    }
    distributed_panel_count = panels.size();

    const std::size_t first_point = replicated ? options.mpi_rank : 0U;
    const std::size_t point_stride =
        replicated ? options.mpi_ranks : 1U;
    for(std::size_t point = first_point; point < fit.nodes.size();
        point += point_stride) {
      const auto point_start = std::chrono::steady_clock::now();
      const double node = fit.nodes[point];
      const double weight = fit.weights[point];
      if(!std::isfinite(node) || node <= 0.0 || !std::isfinite(weight)) {
        throw std::invalid_argument("localized LMP2-1M2M quadrature is invalid");
      }
      const linalg::Matrix occupied_transform = localized_cached_backend
          ? localized_propagator(occupied_energies, energy_shift, node,
                                 true, occupied.rotation)
          : scaled_rotation(occupied_energies, energy_shift, node, true,
                            occupied.rotation);
      const linalg::Matrix virtual_transform = localized_cached_backend
          ? localized_propagator(virtual_energies, energy_shift, node,
                                 false, virtuals.rotation)
          : scaled_rotation(virtual_energies, energy_shift, node, false,
                            virtuals.rotation);
      std::vector<double> point_pair_vector_norms;
      std::vector<double> point_one_sided_integral_norms;
      PointScreeningData point_screening{};
      const PointScreeningData* point_screening_pointer = nullptr;
      if(options.localized_point_energy_threshold_hartree > 0.0) {
        const auto screening_start = std::chrono::steady_clock::now();
        point_pair_vector_norms = scaled_pair_vector_norms(
            occupied_transform, virtual_transform);
        point_one_sided_integral_norms = one_sided_screening_norms(
            occupied_transform, virtual_transform);
        localized_screening_seconds += elapsed_seconds(screening_start);
        point_screening = PointScreeningData{
            .threshold_hartree =
                options.localized_point_energy_threshold_hartree,
            .quadrature_weight = weight,
            .pair_vector_norms = &point_pair_vector_norms,
            .one_sided_integral_norms =
                &point_one_sided_integral_norms,
        };
        point_screening_pointer = &point_screening;
      }
      std::vector<SelectedIntegral> local_entries;
      std::uint64_t point_retained_contributions = 0;
      std::uint64_t point_screened_contributions = 0;
      long double point_omitted_bound = 0.0L;

      for(const LocalizedRightPanel& panel : panels) {
        std::vector<double> owned_panel = project_right_panel(
            panel, occupied_transform, virtual_transform, false);
        if(replicated || options.mpi_rank == panel.owner) {
          const auto rotation_start = std::chrono::steady_clock::now();
          SelectedRotationResult panel_result =
              rotate_selected_left_pairs(
                  owned_panel, panel, occupied, virtuals,
                  occupied_transform, virtual_transform, active_blocks,
                  options.threads, point_screening_pointer);
          timings[2] += elapsed_seconds(rotation_start);
          point_retained_contributions = checked_u64_add(
              point_retained_contributions,
              panel_result.retained_point_contributions,
              "selected retained point-contribution count");
          point_screened_contributions = checked_u64_add(
              point_screened_contributions,
              panel_result.screened_point_contributions,
              "selected point-screening contribution count");
          point_omitted_bound +=
              panel_result.omitted_energy_bound_hartree;
          local_entries.insert(
              local_entries.end(),
              std::make_move_iterator(panel_result.entries.begin()),
              std::make_move_iterator(panel_result.entries.end()));
        }
      }
      std::ranges::sort(local_entries, {}, &SelectedIntegral::key);
      if(std::adjacent_find(
             local_entries.begin(), local_entries.end(),
             [](const SelectedIntegral& left,
                const SelectedIntegral& right) {
               return left.key == right.key;
             }) != local_entries.end()) {
        throw std::runtime_error(
            "distributed selected OVOV produced duplicate entries");
      }
      std::uint64_t global_entry_count =
          static_cast<std::uint64_t>(local_entries.size());
      std::uint64_t global_retained_contributions =
          point_retained_contributions;
      std::uint64_t global_screened_contributions =
          point_screened_contributions;
#ifdef LMP2_1M2M_HAS_MPI
      if(!replicated && options.mpi_ranks > 1 &&
         (MPI_Allreduce(MPI_IN_PLACE, &global_entry_count, 1,
                        MPI_UINT64_T, MPI_SUM,
                        MPI_COMM_WORLD) != MPI_SUCCESS ||
          MPI_Allreduce(MPI_IN_PLACE, &global_screened_contributions, 1,
                        MPI_UINT64_T, MPI_SUM,
                        MPI_COMM_WORLD) != MPI_SUCCESS ||
          MPI_Allreduce(MPI_IN_PLACE, &global_retained_contributions, 1,
                        MPI_UINT64_T, MPI_SUM,
                        MPI_COMM_WORLD) != MPI_SUCCESS)) {
        throw std::runtime_error(
            "distributed selected OVOV entry-count reduction failed");
      }
#endif
      std::uint64_t expected_entries = 0;
      for(std::size_t task_index = 0; task_index < tasks.size();
          ++task_index) {
        if(task_is_active[task_index] == 0U) {
          continue;
        }
        expected_entries = checked_u64_add(
            expected_entries, tasks[task_index].term_count,
            "distributed selected OVOV expected entry count");
      }
      if(checked_u64_add(global_retained_contributions,
                         global_screened_contributions,
                         "selected point-screening entry count") !=
         expected_entries) {
        throw std::runtime_error(
            "distributed selected OVOV retained plus screened entry count "
            "does not reproduce the 1M+2M excitation space");
      }
      if(global_entry_count > expected_entries) {
        throw std::runtime_error(
            "distributed selected OVOV materialized entry count exceeds "
            "the 1M+2M excitation space");
      }
      std::vector<double> exchange;
      if(replicated) {
        LocalizedTiledLaplaceMp2Options local_lookup = options;
        local_lookup.mpi_rank = 0;
        local_lookup.mpi_ranks = 1;
        exchange = exchange_values(local_entries, occupied_count,
                                   virtual_count, local_lookup, timings[5]);
      } else {
        exchange = exchange_values(local_entries, occupied_count,
                                   virtual_count, options, timings[5]);
      }
      if(exchange.size() != local_entries.size()) {
        throw std::logic_error(
            "distributed selected OVOV exchange size is inconsistent");
      }
      const auto accumulation_start = std::chrono::steady_clock::now();
      const long double energy_one_before = local_energy[1];
      const long double energy_two_before = local_energy[2];
      for(std::size_t index = 0; index < local_entries.size(); ++index) {
        const SelectedIntegral& entry = local_entries[index];
        if(!entry.accumulate_energy) {
          continue;
        }
        local_energy[entry.body_order] -=
            static_cast<long double>(weight) *
            static_cast<long double>(entry.value) *
            static_cast<long double>(2.0 * entry.value - exchange[index]);
      }
      timings[4] += elapsed_seconds(accumulation_start);
      localized_screened_point_contributions = checked_u64_add(
          localized_screened_point_contributions,
          point_screened_contributions,
          "localized screened point-contribution count");
      localized_point_screening_omitted_bound += point_omitted_bound;
      if(options.report_laplace_point_progress) {
        std::uint64_t reported_retained = point_retained_contributions;
        std::uint64_t reported_screened = point_screened_contributions;
        std::uint64_t reported_materialized =
            static_cast<std::uint64_t>(local_entries.size());
        long double reported_values[3]{
            local_energy[1] - energy_one_before,
            local_energy[2] - energy_two_before,
            point_omitted_bound,
        };
#ifdef LMP2_1M2M_HAS_MPI
        if(!replicated && options.mpi_ranks > 1) {
          reported_retained = global_retained_contributions;
          reported_screened = global_screened_contributions;
          reported_materialized = global_entry_count;
          if(MPI_Allreduce(MPI_IN_PLACE, reported_values, 3,
                           MPI_LONG_DOUBLE, MPI_SUM,
                           MPI_COMM_WORLD) != MPI_SUCCESS) {
            throw std::runtime_error(
                "localized Laplace-point progress reduction failed");
          }
        }
#endif
        if(replicated || options.mpi_rank == 0) {
          std::ostringstream message;
          message << std::setprecision(17)
                  << "progress Laplace point " << (point + 1U) << '/'
                  << fit.nodes.size() << " complete; rank="
                  << options.mpi_rank << " node=" << node
                  << " weight=" << weight
                  << " retained=" << reported_retained
                  << " materialized=" << reported_materialized
                  << " screened=" << reported_screened
                  << " omitted_bound_hartree="
                  << static_cast<double>(reported_values[2])
                  << " energy_1m_hartree="
                  << static_cast<double>(reported_values[0])
                  << " energy_2m_hartree="
                  << static_cast<double>(reported_values[1])
                  << " seconds=" << elapsed_seconds(point_start) << '\n';
          std::cerr << message.str();
        }
      }
      if(local_entries.size() >
         std::numeric_limits<std::size_t>::max() /
             sizeof(SelectedIntegral)) {
        throw std::overflow_error(
            "distributed selected OVOV entry storage overflows size_t");
      }
      const std::size_t selected_bytes = checked_add(
          local_entries.size() * sizeof(SelectedIntegral),
          checked_bytes(exchange.size(),
                        "distributed selected OVOV exchange storage"),
          "distributed selected OVOV retained storage");
      peak_bytes = std::max(
          peak_bytes,
          checked_add(
              checked_add(persistent_bytes,
                          canonical_ovov_max_local_bytes,
                          "distributed selected OVOV retained peak"),
              selected_bytes,
              "distributed selected OVOV retained peak"));
    }
    }
  } else if(localized_direct_selected_backend) {
    const IndexRange owned_left_occupied = balanced_range(
        occupied.assignments.size(), options.mpi_rank,
        options.mpi_ranks);
    std::vector<BatchedOwnedSelectedTask> owned_tasks;
    std::vector<std::vector<BatchedSelectedOperation>>
        operations_by_virtual_block(virtuals.blocks.size());
    for(const LocalizedTask& task : tasks) {
      const OrbitalBlock& ib =
          occupied.blocks[task.occupied_i_block];
      const std::size_t left_first =
          std::max(ib.first, owned_left_occupied.first);
      const std::size_t left_last = std::min(
          checked_add(ib.first, ib.count,
                      "batched localized occupied block end"),
          checked_add(owned_left_occupied.first,
                      owned_left_occupied.count,
                      "batched localized owned occupied end"));
      if(left_first >= left_last) {
        continue;
      }
      const std::size_t owned_task = owned_tasks.size();
      owned_tasks.push_back(BatchedOwnedSelectedTask{
          .task = task,
          .left_occupied_first =
              left_first - owned_left_occupied.first,
          .left_occupied_count = left_last - left_first,
          .direct_by_point = {},
          .exchange_by_point = {},
      });
      operations_by_virtual_block[task.virtual_a_block].push_back(
          BatchedSelectedOperation{
              .owned_task = owned_task,
              .exchange = false,
          });
      if(task.distinct_virtual_blocks) {
        operations_by_virtual_block[task.virtual_b_block].push_back(
            BatchedSelectedOperation{
                .owned_task = owned_task,
                .exchange = true,
            });
      }
    }

    for(std::size_t point_first = 0; point_first < fit.nodes.size();
        point_first += options.quadrature_batch_size) {
      const auto batch_start = std::chrono::steady_clock::now();
      const std::size_t point_count = std::min(
          options.quadrature_batch_size,
          fit.nodes.size() - point_first);
      std::vector<linalg::Matrix> localized_occupied_by_point;
      std::vector<linalg::Matrix> localized_virtual_by_point;
      localized_occupied_by_point.reserve(point_count);
      localized_virtual_by_point.reserve(point_count);
      const auto rotation_start = std::chrono::steady_clock::now();
      for(std::size_t local_point = 0; local_point < point_count;
          ++local_point) {
        const std::size_t point = point_first + local_point;
        const double node = fit.nodes[point];
        if(!std::isfinite(node) || node <= 0.0 ||
           !std::isfinite(fit.weights[point])) {
          throw std::invalid_argument(
              "localized LMP2-1M2M quadrature is invalid");
        }
        localized_occupied_by_point.push_back(
            scaled_localized_coefficients(
                canonical_occupied, occupied_energies, energy_shift,
                node, true, occupied.rotation));
        localized_virtual_by_point.push_back(
            scaled_localized_coefficients(
                canonical_virtual, virtual_energies, energy_shift,
                node, false, virtuals.rotation));
      }
      timings[2] += elapsed_seconds(rotation_start);

      std::size_t preflight_selected_elements = 0;
      for(const BatchedOwnedSelectedTask& owned_task : owned_tasks) {
        const LocalizedTask& task = owned_task.task;
        const OrbitalBlock& jb = occupied.blocks[task.occupied_j_block];
        const OrbitalBlock& ab = virtuals.blocks[task.virtual_a_block];
        const OrbitalBlock& bb = virtuals.blocks[task.virtual_b_block];
        const std::size_t direct_elements = checked_multiply(
            checked_multiply(owned_task.left_occupied_count, ab.count,
                             "batched selected preflight direct left"),
            checked_multiply(jb.count, bb.count,
                             "batched selected preflight direct right"),
            "batched selected preflight direct tile");
        preflight_selected_elements = checked_add(
            preflight_selected_elements,
            checked_multiply(point_count, direct_elements,
                             "batched selected preflight direct points"),
            "batched selected preflight storage");
        if(task.distinct_virtual_blocks) {
          const std::size_t exchange_elements = checked_multiply(
              checked_multiply(owned_task.left_occupied_count, bb.count,
                               "batched selected preflight exchange left"),
              checked_multiply(jb.count, ab.count,
                               "batched selected preflight exchange right"),
              "batched selected preflight exchange tile");
          preflight_selected_elements = checked_add(
              preflight_selected_elements,
              checked_multiply(
                  point_count, exchange_elements,
                  "batched selected preflight exchange points"),
              "batched selected preflight storage");
        }
      }
      const std::size_t preflight_selected_bytes = checked_bytes(
          preflight_selected_elements,
          "batched selected preflight retained tiles");
      const std::size_t preflight_coefficient_bytes = checked_bytes(
          checked_multiply(
              point_count,
              checked_add(canonical_occupied.size(),
                          canonical_virtual.size(),
                          "batched selected preflight coefficients"),
              "batched selected preflight all coefficients"),
          "batched selected preflight coefficient matrices");
      if(checked_add(
             checked_add(persistent_bytes, preflight_selected_bytes,
                         "batched selected preflight persistent storage"),
             preflight_coefficient_bytes,
             "batched selected preflight coefficient storage") >=
         options.maximum_additional_memory_bytes) {
        throw std::runtime_error(
            "batched selected retained tiles exceed the explicit memory "
            "limit before allocation");
      }

      std::size_t selected_elements = 0;
      std::uint64_t exchange_completions = 0;
      std::uint64_t reused_exchange = 0;
      for(BatchedOwnedSelectedTask& owned_task : owned_tasks) {
        const LocalizedTask& task = owned_task.task;
        const OrbitalBlock& jb =
            occupied.blocks[task.occupied_j_block];
        const OrbitalBlock& ab =
            virtuals.blocks[task.virtual_a_block];
        const OrbitalBlock& bb =
            virtuals.blocks[task.virtual_b_block];
        const std::size_t direct_elements = checked_multiply(
            checked_multiply(owned_task.left_occupied_count, ab.count,
                             "batched selected direct left pairs"),
            checked_multiply(jb.count, bb.count,
                             "batched selected direct right pairs"),
            "batched selected direct tile");
        owned_task.direct_by_point.assign(
            point_count, std::vector<double>(direct_elements, 0.0));
        selected_elements = checked_add(
            selected_elements,
            checked_multiply(point_count, direct_elements,
                             "batched selected direct point storage"),
            "batched selected storage");
        if(task.distinct_virtual_blocks) {
          const std::size_t exchange_elements = checked_multiply(
              checked_multiply(owned_task.left_occupied_count, bb.count,
                               "batched selected exchange left pairs"),
              checked_multiply(jb.count, ab.count,
                               "batched selected exchange right pairs"),
              "batched selected exchange tile");
          owned_task.exchange_by_point.assign(
              point_count,
              std::vector<double>(exchange_elements, 0.0));
          selected_elements = checked_add(
              selected_elements,
              checked_multiply(point_count, exchange_elements,
                               "batched selected exchange point storage"),
              "batched selected storage");
          exchange_completions = checked_u64_add(
              exchange_completions,
              static_cast<std::uint64_t>(point_count),
              "batched selected exchange completion count");
        } else {
          owned_task.exchange_by_point.clear();
          reused_exchange = checked_u64_add(
              reused_exchange,
              static_cast<std::uint64_t>(point_count),
              "batched selected reused exchange count");
        }
      }
      const std::size_t selected_bytes = checked_bytes(
          selected_elements, "batched selected retained tiles");
      const std::size_t coefficient_bytes = checked_bytes(
          checked_multiply(
              point_count,
              checked_add(canonical_occupied.size(),
                          canonical_virtual.size(),
                          "batched selected coefficient elements"),
              "batched selected all coefficient elements"),
          "batched selected coefficient matrices");
      const std::size_t maximum_shell_functions =
          std::ranges::max(
              provider.basis_metadata().shells, {},
              &basis::Shell::function_count)
              .function_count;
      const std::size_t maximum_lambda_count = std::max(
          options.lambda_ao_block_size, maximum_shell_functions);
      const std::size_t maximum_t_elements = checked_multiply(
          checked_multiply(
              checked_multiply(maximum_lambda_count, ao_count,
                               "batched selected maximum lambda-sigma"),
              owned_left_occupied.count,
              "batched selected maximum occupied T"),
          options.virtual_tile_size,
          "batched selected maximum virtual T");
      const std::size_t maximum_t_bytes = checked_bytes(
          maximum_t_elements, "batched selected maximum T block");
      const std::size_t batch_base_bytes = checked_add(
          checked_add(persistent_bytes, selected_bytes,
                      "batched selected persistent tiles"),
          coefficient_bytes,
          "batched selected coefficient storage");
      const std::size_t reserved_bytes = checked_add(
          batch_base_bytes, maximum_t_bytes,
          "batched selected reserved storage");
      if(reserved_bytes >= options.maximum_additional_memory_bytes) {
        throw std::runtime_error(
            "batched selected localized intermediates exceed the "
            "explicit memory limit");
      }

      std::uint64_t batch_evaluated = 0;
      std::uint64_t batch_screened = 0;
      std::size_t maximum_provider_peak = 0;
      const auto transform_start = std::chrono::steady_clock::now();
      if(owned_left_occupied.count != 0) {
        provider.for_each_batched_first_index_ao_block(
            localized_occupied_by_point,
            integrals::BatchedFirstIndexAoOptions{
                .occupied_first = owned_left_occupied.first,
                .occupied_count = owned_left_occupied.count,
                .target_lambda_ao_block_size =
                    options.lambda_ao_block_size,
                .threads = options.threads,
                .schwarz_threshold = options.eri_schwarz_threshold,
                .report_progress =
                    options.report_laplace_point_progress &&
                    options.mpi_rank == 0,
                .maximum_additional_memory_bytes =
                    options.maximum_additional_memory_bytes -
                    reserved_bytes,
            },
            [&](const integrals::BatchedFirstIndexAoBlock& block) {
              batch_evaluated = checked_u64_add(
                  batch_evaluated, block.evaluated_shell_quartets,
                  "batched selected evaluated shell quartets");
              batch_screened = checked_u64_add(
                  batch_screened,
                  block.schwarz_screened_shell_quartets,
                  "batched selected screened shell quartets");
              maximum_provider_peak = std::max(
                  maximum_provider_peak,
                  block.estimated_peak_additional_bytes);
              for(std::size_t local_point = 0;
                  local_point < point_count; ++local_point) {
                for(std::size_t virtual_block = 0;
                    virtual_block < virtuals.blocks.size();
                    ++virtual_block) {
                  const auto& operations =
                      operations_by_virtual_block[virtual_block];
                  if(operations.empty()) {
                    continue;
                  }
                  const OrbitalBlock& left_virtual =
                      virtuals.blocks[virtual_block];
                  std::vector<double> transformed_virtual =
                      transform_batched_left_virtual_block(
                          block, local_point,
                          localized_virtual_by_point[local_point],
                          left_virtual.first, left_virtual.count,
                          options.threads);
                  std::vector<std::exception_ptr> operation_errors(
                      options.threads);
                  const auto run_operations =
                      [&](std::size_t thread,
                          std::size_t actual_threads) {
                        for(std::size_t operation_index = thread;
                            operation_index < operations.size();
                            operation_index += actual_threads) {
                          const BatchedSelectedOperation& operation =
                              operations[operation_index];
                          BatchedOwnedSelectedTask& owned_task =
                              owned_tasks[operation.owned_task];
                          const LocalizedTask& task = owned_task.task;
                          const OrbitalBlock& right_occupied =
                              occupied.blocks[task.occupied_j_block];
                          const std::size_t right_virtual_block =
                              operation.exchange
                                  ? task.virtual_a_block
                                  : task.virtual_b_block;
                          const OrbitalBlock& right_virtual =
                              virtuals.blocks[right_virtual_block];
                          std::vector<double>& output =
                              operation.exchange
                                  ? owned_task.exchange_by_point
                                        [local_point]
                                  : owned_task.direct_by_point[local_point];
                          accumulate_batched_selected_operation(
                              block, transformed_virtual,
                              left_virtual.count,
                              localized_occupied_by_point[local_point],
                              localized_virtual_by_point[local_point],
                              right_occupied, right_virtual, owned_task,
                              output);
                        }
                      };
#ifdef LMP2_1M2M_HAS_OPENMP
#pragma omp parallel num_threads(static_cast<int>(options.threads))
                  {
                    const std::size_t thread =
                        static_cast<std::size_t>(omp_get_thread_num());
                    const std::size_t actual_threads =
                        static_cast<std::size_t>(omp_get_num_threads());
                    try {
                      run_operations(thread, actual_threads);
                    } catch(...) {
                      operation_errors[thread] =
                          std::current_exception();
                    }
                  }
#else
                  try {
                    run_operations(0U, 1U);
                  } catch(...) {
                    operation_errors[0] = std::current_exception();
                  }
#endif
                  for(const auto& error : operation_errors) {
                    if(error) {
                      std::rethrow_exception(error);
                    }
                  }
                }
              }
            });
        counters[0] = checked_u64_add(
            counters[0], 1U,
            "batched selected AO-pass count");
        counters[3] = checked_u64_add(
            counters[3], batch_evaluated,
            "batched selected evaluated-shell count");
        counters[4] = checked_u64_add(
            counters[4], batch_screened,
            "batched selected screened-shell count");
      }
      direct_ovov_transform_seconds += elapsed_seconds(transform_start);
      counters[1] = checked_u64_add(
          counters[1], exchange_completions,
          "batched selected exchange-completion count");
      counters[2] = checked_u64_add(
          counters[2], reused_exchange,
          "batched selected reused-exchange count");
      peak_bytes = std::max(
          peak_bytes,
          checked_add(
              checked_add(batch_base_bytes, maximum_provider_peak,
                          "batched selected provider peak"),
              maximum_t_bytes, "batched selected total peak"));

      std::uint64_t reported_ao_passes =
          owned_left_occupied.count == 0 ? 0U : 1U;
      std::uint64_t reported_evaluated = batch_evaluated;
      std::uint64_t reported_screened = batch_screened;
#ifdef LMP2_1M2M_HAS_MPI
      if(options.mpi_ranks > 1 &&
         (MPI_Allreduce(MPI_IN_PLACE, &reported_ao_passes, 1,
                        MPI_UINT64_T, MPI_SUM,
                        MPI_COMM_WORLD) != MPI_SUCCESS ||
          MPI_Allreduce(MPI_IN_PLACE, &reported_evaluated, 1,
                        MPI_UINT64_T, MPI_SUM,
                        MPI_COMM_WORLD) != MPI_SUCCESS ||
          MPI_Allreduce(MPI_IN_PLACE, &reported_screened, 1,
                        MPI_UINT64_T, MPI_SUM,
                        MPI_COMM_WORLD) != MPI_SUCCESS)) {
        throw std::runtime_error(
            "batched selected progress reduction failed");
      }
#endif
      if(options.report_laplace_point_progress &&
         options.mpi_rank == 0) {
        std::cerr
            << "progress localized-direct quadrature batch complete; "
            << "point_first=" << (point_first + 1U)
            << " point_count=" << point_count
            << " ao_passes=" << reported_ao_passes
            << " evaluated_shell_quartets=" << reported_evaluated
            << " screened_shell_quartets=" << reported_screened
            << " seconds=" << elapsed_seconds(batch_start) << '\n';
      }

      const auto accumulation_start = std::chrono::steady_clock::now();
      for(std::size_t local_point = 0; local_point < point_count;
          ++local_point) {
        const std::size_t point = point_first + local_point;
        const long double energy_one_before = local_energy[1];
        const long double energy_two_before = local_energy[2];
        for(const BatchedOwnedSelectedTask& owned_task : owned_tasks) {
          const LocalizedTask& task = owned_task.task;
          const OrbitalBlock& jb =
              occupied.blocks[task.occupied_j_block];
          const OrbitalBlock& ab =
              virtuals.blocks[task.virtual_a_block];
          const OrbitalBlock& bb =
              virtuals.blocks[task.virtual_b_block];
          const std::size_t direct_left_pairs = checked_multiply(
              owned_task.left_occupied_count, ab.count,
              "batched selected direct energy left pairs");
          const std::size_t exchange_left_pairs = checked_multiply(
              owned_task.left_occupied_count, bb.count,
              "batched selected exchange energy left pairs");
          const std::vector<double>& direct =
              owned_task.direct_by_point[local_point];
          const std::vector<double>* exchange =
              task.distinct_virtual_blocks
                  ? &owned_task.exchange_by_point[local_point]
                  : nullptr;
          long double task_energy = 0.0L;
          for(std::size_t i = 0;
              i < owned_task.left_occupied_count; ++i) {
            for(std::size_t a = 0; a < ab.count; ++a) {
              for(std::size_t j = 0; j < jb.count; ++j) {
                for(std::size_t b = 0; b < bb.count; ++b) {
                  const std::size_t direct_right =
                      j * bb.count + b;
                  const double direct_value =
                      direct[direct_right * direct_left_pairs +
                             i * ab.count + a];
                  double exchange_value = 0.0;
                  if(exchange == nullptr) {
                    const std::size_t exchange_right =
                        j * ab.count + a;
                    exchange_value =
                        direct[exchange_right * direct_left_pairs +
                               i * ab.count + b];
                  } else {
                    const std::size_t exchange_right =
                        j * ab.count + a;
                    exchange_value =
                        (*exchange)[
                            exchange_right * exchange_left_pairs +
                            i * bb.count + b];
                  }
                  long double ordered_pair_energy =
                      static_cast<long double>(direct_value) *
                      static_cast<long double>(
                          2.0 * direct_value - exchange_value);
                  if(task.distinct_virtual_blocks) {
                    ordered_pair_energy +=
                        static_cast<long double>(exchange_value) *
                        static_cast<long double>(
                            2.0 * exchange_value - direct_value);
                  }
                  task_energy -=
                      static_cast<long double>(fit.weights[point]) *
                      static_cast<long double>(
                          task.distinct_occupied_blocks ? 2.0 : 1.0) *
                      ordered_pair_energy;
                }
              }
            }
          }
          local_energy[task.body_order] += task_energy;
        }
        if(options.report_laplace_point_progress) {
          std::array<long double, 2> reported_energy{
              local_energy[1] - energy_one_before,
              local_energy[2] - energy_two_before,
          };
#ifdef LMP2_1M2M_HAS_MPI
          if(options.mpi_ranks > 1 &&
             MPI_Allreduce(MPI_IN_PLACE, reported_energy.data(),
                           static_cast<int>(reported_energy.size()),
                           MPI_LONG_DOUBLE, MPI_SUM,
                           MPI_COMM_WORLD) != MPI_SUCCESS) {
            throw std::runtime_error(
                "batched selected point-energy reduction failed");
          }
#endif
          if(options.mpi_rank == 0) {
            std::cerr
                << std::setprecision(17)
                << "progress localized-direct Laplace point "
                << (point + 1U) << '/' << fit.nodes.size()
                << " complete; node=" << fit.nodes[point]
                << " weight=" << fit.weights[point]
                << " shared_ao_batch=" << (point_first + 1U)
                << " energy_1m_hartree="
                << static_cast<double>(reported_energy[0])
                << " energy_2m_hartree="
                << static_cast<double>(reported_energy[1]) << '\n';
          }
        }
      }
      timings[4] += elapsed_seconds(accumulation_start);
    }
  } else {
    for(std::size_t point = 0; point < fit.nodes.size(); ++point) {
      const auto point_start = std::chrono::steady_clock::now();
      const long double energy_one_before = local_energy[1];
      const long double energy_two_before = local_energy[2];
      const std::array<std::uint64_t, 5> counters_before = counters;
      const double node = fit.nodes[point];
      const double weight = fit.weights[point];
      if(!std::isfinite(node) || node <= 0.0 || !std::isfinite(weight)) {
        throw std::invalid_argument("localized LMP2-1M2M quadrature is invalid");
      }
      const auto rotation_start = std::chrono::steady_clock::now();
      const linalg::Matrix localized_occupied =
          scaled_localized_coefficients(
              canonical_occupied, occupied_energies, energy_shift, node, true,
              occupied.rotation);
      const linalg::Matrix localized_virtual =
          scaled_localized_coefficients(
              canonical_virtual, virtual_energies, energy_shift, node, false,
              virtuals.rotation);
      timings[2] += elapsed_seconds(rotation_start);

      for(std::size_t task_index = 0; task_index < tasks.size(); ++task_index) {
        const std::uint64_t ordinal =
            static_cast<std::uint64_t>(point) *
                static_cast<std::uint64_t>(tasks.size()) +
            static_cast<std::uint64_t>(task_index);
        if(ordinal % options.mpi_ranks != options.mpi_rank) {
          continue;
        }
        const LocalizedTask& task = tasks[task_index];
        const OrbitalBlock& ib = occupied.blocks[task.occupied_i_block];
        const OrbitalBlock& jb = occupied.blocks[task.occupied_j_block];
        const OrbitalBlock& ab = virtuals.blocks[task.virtual_a_block];
        const OrbitalBlock& bb = virtuals.blocks[task.virtual_b_block];
        const std::size_t transform_budget =
            options.maximum_additional_memory_bytes - persistent_bytes;
        const auto direct_transform_start =
            std::chrono::steady_clock::now();
        const integrals::OvovTileResult direct = provider.compute_ovov_tile(
            localized_occupied, localized_virtual,
            integrals::OvovTileOptions{
                .left_occupied_first = ib.first,
                .left_occupied_count = ib.count,
                .left_virtual_first = ab.first,
                .left_virtual_count = ab.count,
                .right_occupied_first = jb.first,
                .right_occupied_count = jb.count,
                .right_virtual_first = bb.first,
                .right_virtual_count = bb.count,
                .threads = options.threads,
                .schwarz_threshold = options.eri_schwarz_threshold,
                .maximum_additional_memory_bytes = transform_budget,
            });
        direct_ovov_transform_seconds +=
            elapsed_seconds(direct_transform_start);
        counters[0] = checked_u64_add(
            counters[0], 1, "localized direct-transform count");
        counters[3] = checked_u64_add(
            counters[3], direct.evaluated_shell_quartets,
            "localized evaluated-shell-quartet count");
        counters[4] = checked_u64_add(
            counters[4], direct.schwarz_screened_shell_quartets,
            "localized screened-shell-quartet count");
        peak_bytes = std::max(
            peak_bytes,
            checked_add(persistent_bytes,
                        direct.estimated_peak_additional_bytes,
                        "localized LMP2-1M2M direct-transform peak"));
        const bool reuse_exchange = !task.distinct_virtual_blocks;
        integrals::OvovTileResult exchange;
        if(reuse_exchange) {
          counters[2] = checked_u64_add(
              counters[2], 1, "localized reused-exchange count");
        } else {
          const std::size_t retained = checked_bytes(
              direct.values.size(), "localized LMP2-1M2M retained direct tile");
          if(checked_add(persistent_bytes, retained,
                         "localized LMP2-1M2M exchange storage") >=
             options.maximum_additional_memory_bytes) {
            throw std::runtime_error(
                "localized LMP2-1M2M memory is insufficient for exchange tile");
          }
          const auto exchange_transform_start =
              std::chrono::steady_clock::now();
          exchange = provider.compute_ovov_tile(
              localized_occupied, localized_virtual,
              integrals::OvovTileOptions{
                  .left_occupied_first = ib.first,
                  .left_occupied_count = ib.count,
                  .left_virtual_first = bb.first,
                  .left_virtual_count = bb.count,
                  .right_occupied_first = jb.first,
                  .right_occupied_count = jb.count,
                  .right_virtual_first = ab.first,
                  .right_virtual_count = ab.count,
                  .threads = options.threads,
                  .schwarz_threshold = options.eri_schwarz_threshold,
                  .maximum_additional_memory_bytes =
                      options.maximum_additional_memory_bytes -
                      persistent_bytes - retained,
              });
          direct_ovov_transform_seconds +=
              elapsed_seconds(exchange_transform_start);
          counters[1] = checked_u64_add(
              counters[1], 1, "localized exchange-transform count");
          counters[3] = checked_u64_add(
              counters[3], exchange.evaluated_shell_quartets,
              "localized evaluated-shell-quartet count");
          counters[4] = checked_u64_add(
              counters[4], exchange.schwarz_screened_shell_quartets,
              "localized screened-shell-quartet count");
          peak_bytes = std::max(
              peak_bytes,
              checked_add(
                  checked_add(persistent_bytes, retained,
                              "localized LMP2-1M2M retained direct peak"),
                  exchange.estimated_peak_additional_bytes,
                  "localized LMP2-1M2M exchange-transform peak"));
        }

        const auto accumulation_start = std::chrono::steady_clock::now();
        long double task_energy = 0.0L;
        for(std::size_t i = 0; i < ib.count; ++i) {
          for(std::size_t a = 0; a < ab.count; ++a) {
            for(std::size_t j = 0; j < jb.count; ++j) {
              for(std::size_t b = 0; b < bb.count; ++b) {
                const double direct_value = direct(i, a, j, b);
                const double exchange_value =
                    reuse_exchange ? direct(i, b, j, a)
                                   : exchange(i, b, j, a);
                long double ordered_pair_energy =
                    static_cast<long double>(direct_value) *
                    static_cast<long double>(
                        2.0 * direct_value - exchange_value);
                if(task.distinct_virtual_blocks) {
                  ordered_pair_energy +=
                      static_cast<long double>(exchange_value) *
                      static_cast<long double>(
                          2.0 * exchange_value - direct_value);
                }
                task_energy -= static_cast<long double>(weight) *
                    static_cast<long double>(
                        task.distinct_occupied_blocks ? 2.0 : 1.0) *
                    ordered_pair_energy;
              }
            }
          }
        }
        timings[4] += elapsed_seconds(accumulation_start);
        local_energy[task.body_order] += task_energy;
      }

      if(options.report_laplace_point_progress) {
        std::array<long double, 2> reported_energy{
            local_energy[1] - energy_one_before,
            local_energy[2] - energy_two_before,
        };
        std::array<std::uint64_t, 4> reported_counts{
            counters[0] - counters_before[0],
            counters[1] - counters_before[1],
            counters[3] - counters_before[3],
            counters[4] - counters_before[4],
        };
#ifdef LMP2_1M2M_HAS_MPI
        if(options.mpi_ranks > 1 &&
           (MPI_Allreduce(MPI_IN_PLACE, reported_energy.data(),
                          static_cast<int>(reported_energy.size()),
                          MPI_LONG_DOUBLE, MPI_SUM,
                          MPI_COMM_WORLD) != MPI_SUCCESS ||
            MPI_Allreduce(MPI_IN_PLACE, reported_counts.data(),
                          static_cast<int>(reported_counts.size()),
                          MPI_UINT64_T, MPI_SUM,
                          MPI_COMM_WORLD) != MPI_SUCCESS)) {
          throw std::runtime_error(
              "localized direct Laplace-point progress reduction failed");
        }
#endif
        if(options.mpi_rank == 0) {
          std::ostringstream message;
          message << std::setprecision(17)
                  << "progress localized-direct Laplace point "
                  << (point + 1U) << '/' << fit.nodes.size()
                  << " complete; node=" << node << " weight=" << weight
                  << " direct_transforms=" << reported_counts[0]
                  << " exchange_transforms=" << reported_counts[1]
                  << " evaluated_shell_quartets=" << reported_counts[2]
                  << " screened_shell_quartets=" << reported_counts[3]
                  << " energy_1m_hartree="
                  << static_cast<double>(reported_energy[0])
                  << " energy_2m_hartree="
                  << static_cast<double>(reported_energy[1])
                  << " seconds=" << elapsed_seconds(point_start) << '\n';
          std::cerr << message.str();
        }
      }
    }
  }

  std::array<double, maximum_mp2_body_order + 1> energy{};
  for(std::size_t body = 1; body <= maximum_mp2_body_order; ++body) {
    energy[body] = static_cast<double>(local_energy[body]);
  }
#ifdef LMP2_1M2M_HAS_MPI
  if(options.mpi_ranks > 1) {
    reduce_results(energy, counters, peak_bytes);
    if(MPI_Allreduce(MPI_IN_PLACE,
                     &localized_screened_point_contributions, 1,
                     MPI_UINT64_T, MPI_SUM,
                     MPI_COMM_WORLD) != MPI_SUCCESS ||
       MPI_Allreduce(MPI_IN_PLACE,
                     &localized_point_screening_omitted_bound, 1,
                     MPI_LONG_DOUBLE, MPI_SUM,
                     MPI_COMM_WORLD) != MPI_SUCCESS) {
      throw std::runtime_error(
          "localized point-screening reduction failed");
    }
  }
#endif
  reduce_maximum_timings(timings, options);
  reduce_maximum_double(
      direct_ovov_transform_seconds, options,
      "localized LMP2-1M2M direct OVOV transform time");
  canonical_ovov_stored_bytes_global = canonical_ovov_max_local_bytes;
  reduce_sum_size(canonical_ovov_stored_bytes_global, options,
                  "localized LMP2-1M2M canonical OVOV global stored-byte sum");
  reduce_maximum_size(canonical_ovov_max_local_bytes, options,
                      "localized LMP2-1M2M canonical OVOV local-byte maximum");
  localized_ovov_stored_bytes_global = localized_ovov_max_local_bytes;
  reduce_sum_size(localized_ovov_stored_bytes_global, options,
                  "localized LMP2-1M2M localized OVOV global stored-byte sum");
  reduce_maximum_size(localized_ovov_max_local_bytes, options,
                      "localized LMP2-1M2M localized OVOV local-byte maximum");
  reduce_maximum_double(localized_screening_seconds, options,
                        "localized LMP2-1M2M screening time");
  if(localized_direct_selected_backend &&
     (canonical_ovov_bytes != 0U ||
      canonical_ovov_max_local_bytes != 0U ||
      canonical_ovov_stored_bytes_global != 0U)) {
    throw std::runtime_error(
        "localized direct selected unexpectedly stored canonical OVOV "
        "elements");
  }
  if(localized_cached_backend &&
     (canonical_ovov_bytes != 0U ||
      canonical_ovov_max_local_bytes != 0U ||
      canonical_ovov_stored_bytes_global != 0U)) {
    throw std::runtime_error(
        "localized cached OVOV unexpectedly stored canonical OVOV "
        "elements");
  }
  if(localized_cached_backend &&
     localized_ovov_stored_bytes_global != localized_ovov_bytes) {
    throw std::runtime_error(
        "localized cached OVOV ownership does not form exactly one "
        "global localized tensor");
  }
  if(backend ==
         LocalizedLaplaceMp2Backend::distributed_selected_ovov &&
     canonical_ovov_stored_bytes_global != canonical_ovov_bytes) {
    throw std::runtime_error(
        "distributed selected OVOV ownership does not form exactly one "
        "global canonical tensor");
  }
  if(backend == LocalizedLaplaceMp2Backend::replicated_selected_ovov) {
    const std::size_t expected_replicated_bytes = checked_multiply(
        canonical_ovov_bytes, options.mpi_ranks,
        "replicated canonical OVOV global storage overflows");
    if(canonical_ovov_stored_bytes_global != expected_replicated_bytes) {
      throw std::runtime_error(
          "replicated selected OVOV storage audit is inconsistent");
    }
  }
  if(backend == LocalizedLaplaceMp2Backend::shared_selected_ovov &&
     canonical_ovov_stored_bytes_global != canonical_ovov_bytes) {
    throw std::runtime_error(
        "shared selected OVOV storage audit is inconsistent");
  }

  NmEnergyDecomposition decomposition;
  decomposition.energy_hartree = energy;
  decomposition.confidence_threshold =
      options.assignment_confidence_threshold;
  std::uint64_t computed_terms = 0;
  for(std::size_t task_index = 0; task_index < tasks.size(); ++task_index) {
    if(task_is_active[task_index] == 0U) {
      continue;
    }
    const LocalizedTask& task = tasks[task_index];
    if(decomposition.term_count[task.body_order] >
       std::numeric_limits<std::uint64_t>::max() - task.term_count) {
      throw std::overflow_error("localized LMP2-1M2M term count overflows uint64");
    }
    decomposition.term_count[task.body_order] += task.term_count;
    computed_terms = checked_u64_add(
        computed_terms, task.term_count,
        "computed localized excitation count");
  }
  const std::size_t low_occ = static_cast<std::size_t>(std::ranges::count_if(
      occupied.assignments, [&](const MonomerAssignment& assignment) {
        return assignment.confidence < options.assignment_confidence_threshold;
      }));
  const std::size_t low_vir = static_cast<std::size_t>(std::ranges::count_if(
      virtuals.assignments, [&](const MonomerAssignment& assignment) {
        return assignment.confidence < options.assignment_confidence_threshold;
      }));
  const double total = decomposition.total_energy_hartree();
  if(!std::isfinite(total) || total > 1.0e-12) {
    throw std::runtime_error(
        "localized LMP2-1M2M produced a nonfinite or positive correlation energy");
  }
  const std::uint64_t computed_point_tasks = checked_u64_multiply(
      static_cast<std::uint64_t>(fit.nodes.size()),
      backend == LocalizedLaplaceMp2Backend::cached_canonical_ovov
          ? std::uint64_t{1}
          : selected_ovov_backend
                ? static_cast<std::uint64_t>(distributed_panel_count)
          : static_cast<std::uint64_t>(tasks.size()),
      "computed localized point-task count");
  const std::uint64_t omitted_point_tasks = checked_u64_multiply(
      static_cast<std::uint64_t>(fit.nodes.size()),
      backend == LocalizedLaplaceMp2Backend::cached_canonical_ovov
          ? std::uint64_t{0}
          : omitted_tasks_per_point,
      "omitted localized point-task count");
  return LocalizedTiledLaplaceMp2Result{
      .decomposition = decomposition,
      .execution_mode = options.execution_mode,
      .backend = backend,
      .correlation_energy_hartree = total,
      .computed_excitation_terms = computed_terms,
      .omitted_excitation_terms = omitted_terms,
      .computed_point_tasks = computed_point_tasks,
      .omitted_point_tasks = omitted_point_tasks,
      .direct_transform_count = counters[0],
      .exchange_transform_count = counters[1],
      .reused_exchange_tile_count = counters[2],
      .evaluated_shell_quartets = counters[3],
      .schwarz_screened_shell_quartets = counters[4],
      .localized_screened_excitation_terms = localized_screened_terms,
      .localized_screened_block_tasks = localized_screened_tasks,
      .localized_screening_omitted_energy_bound_hartree =
          static_cast<double>(localized_screening_omitted_bound),
      .localized_screened_point_contributions =
          localized_screened_point_contributions,
      .localized_point_screening_omitted_energy_bound_hartree =
          static_cast<double>(localized_point_screening_omitted_bound),
      .low_confidence_occupied_orbitals = low_occ,
      .low_confidence_virtual_orbitals = low_vir,
      .estimated_peak_additional_bytes = peak_bytes,
      .canonical_ovov_bytes = canonical_ovov_bytes,
      .canonical_ovov_max_local_bytes = canonical_ovov_max_local_bytes,
      .canonical_ovov_stored_bytes_global =
          canonical_ovov_stored_bytes_global,
      .localized_ovov_bytes = localized_ovov_bytes,
      .localized_ovov_max_local_bytes = localized_ovov_max_local_bytes,
      .localized_ovov_stored_bytes_global =
          localized_ovov_stored_bytes_global,
      .canonical_ovov_build_seconds =
          localized_cached_backend ? 0.0 : timings[0],
      .canonical_ovov_broadcast_seconds = timings[1],
      .localized_ovov_build_seconds =
          localized_cached_backend ? timings[0] : 0.0,
      .direct_ovov_transform_seconds = direct_ovov_transform_seconds,
      .localized_rotation_seconds = timings[2],
      .pair_transpose_seconds = timings[3],
      .energy_accumulation_seconds = timings[4],
      .distributed_communication_seconds = timings[5],
      .localized_screening_seconds = localized_screening_seconds,
  };
}

}  // namespace lmp2_1m2m::mp2
