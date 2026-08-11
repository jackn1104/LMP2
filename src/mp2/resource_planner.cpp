#include "modernqc/mp2/resource_planner.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace modernqc::mp2 {
namespace {

[[nodiscard]] std::size_t checked_multiply(std::size_t left,
                                           std::size_t right,
                                           const char* description) {
  if(left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::overflow_error(description);
  }
  return left * right;
}

[[nodiscard]] std::size_t checked_add(std::size_t left, std::size_t right,
                                      const char* description) {
  if(right > std::numeric_limits<std::size_t>::max() - left) {
    throw std::overflow_error(description);
  }
  return left + right;
}

[[nodiscard]] std::size_t ceiling_divide(std::size_t numerator,
                                         std::size_t denominator) {
  if(denominator == 0) {
    throw std::invalid_argument("resource-plan divisor is zero");
  }
  return numerator / denominator +
         static_cast<std::size_t>(numerator % denominator != 0);
}

[[nodiscard]] std::size_t bytes_for_doubles(std::size_t elements,
                                            const char* description) {
  return checked_multiply(elements, sizeof(double), description);
}

[[nodiscard]] std::size_t canonical_ovov_bytes(
    const LocalizedMp2Workload& workload) {
  const std::size_t pairs = checked_multiply(
      workload.active_occupied_count, workload.virtual_count,
      "resource-plan occupied-virtual pair count overflows");
  return bytes_for_doubles(
      checked_multiply(pairs, pairs,
                       "resource-plan canonical OVOV size overflows"),
      "resource-plan canonical OVOV bytes overflow");
}

[[nodiscard]] std::size_t separable_transform_peak(
    const LocalizedMp2Workload& workload, std::size_t occupied_tile) {
  const std::size_t ao_squared = checked_multiply(
      workload.ao_count, workload.ao_count,
      "resource-plan AO squared size overflows");
  const std::size_t ao_cubed = checked_multiply(
      ao_squared, workload.ao_count,
      "resource-plan AO cubed size overflows");
  const std::size_t first_elements = checked_multiply(
      ao_cubed, occupied_tile,
      "resource-plan first transform size overflows");
  const std::size_t second_elements = checked_multiply(
      checked_multiply(ao_squared, occupied_tile,
                       "resource-plan second transform size overflows"),
      workload.virtual_count,
      "resource-plan second transform size overflows");
  const std::size_t result_elements = checked_multiply(
      checked_multiply(occupied_tile, workload.virtual_count,
                       "resource-plan transform result size overflows"),
      checked_multiply(workload.active_occupied_count,
                       workload.virtual_count,
                       "resource-plan transform result size overflows"),
      "resource-plan transform result size overflows");
  const std::size_t exact_bytes = bytes_for_doubles(
      std::max(checked_add(first_elements, second_elements,
                           "resource-plan first transform peak overflows"),
               checked_add(second_elements, result_elements,
                           "resource-plan final transform peak overflows")),
      "resource-plan transform peak bytes overflow");
  // Libint shell workspaces, allocator alignment, and implementation metadata
  // are backend-dependent. Reserve five percent plus 64 MiB rather than
  // treating the analytical two-intermediate estimate as an allocation cap.
  return checked_add(
      checked_add(exact_bytes, exact_bytes / 20U,
                  "resource-plan transform safety margin overflows"),
      64U * 1024U * 1024U,
      "resource-plan transform safety margin overflows");
}

[[nodiscard]] std::size_t panel_workspace_bytes(
    const LocalizedMp2Workload& workload, std::size_t occupied_tile,
    std::size_t virtual_tile) {
  const std::size_t pairs = checked_multiply(
      workload.active_occupied_count, workload.virtual_count,
      "resource-plan panel pair count overflows");
  const std::size_t panel_columns = checked_multiply(
      occupied_tile, virtual_tile,
      "resource-plan panel column count overflows");
  // A projected panel and its owner-side copy can coexist.
  return bytes_for_doubles(
      checked_multiply(
          checked_multiply(pairs, panel_columns,
                           "resource-plan panel workspace overflows"),
          2U, "resource-plan panel workspace overflows"),
      "resource-plan panel workspace bytes overflow");
}

[[nodiscard]] std::size_t coefficient_workspace_bytes(
    const LocalizedMp2Workload& workload) {
  const std::size_t columns = checked_add(
      workload.active_occupied_count, workload.virtual_count,
      "resource-plan MO column count overflows");
  return bytes_for_doubles(
      checked_multiply(
          checked_multiply(workload.ao_count, columns,
                           "resource-plan coefficient workspace overflows"),
          3U, "resource-plan coefficient workspace overflows"),
      "resource-plan coefficient workspace bytes overflow");
}

[[nodiscard]] std::size_t batched_direct_selected_transform_peak(
    const LocalizedMp2Workload& workload,
    std::size_t owned_occupied_count, std::size_t virtual_tile,
    std::size_t lambda_ao_block, std::size_t quadrature_batch) {
  const std::size_t lambda_sigma = checked_multiply(
      lambda_ao_block, workload.ao_count,
      "resource-plan batched lambda-sigma size overflows");
  const std::size_t first_elements = checked_multiply(
      checked_multiply(lambda_sigma, owned_occupied_count,
                       "resource-plan batched first-index size overflows"),
      checked_multiply(workload.ao_count, quadrature_batch,
                       "resource-plan batched point-AO size overflows"),
      "resource-plan batched first-index size overflows");
  const std::size_t virtual_elements = checked_multiply(
      checked_multiply(lambda_sigma, owned_occupied_count,
                       "resource-plan batched virtual size overflows"),
      virtual_tile, "resource-plan batched virtual size overflows");
  const std::size_t exact_bytes = bytes_for_doubles(
      checked_add(first_elements, virtual_elements,
                  "resource-plan batched transform peak overflows"),
      "resource-plan batched transform bytes overflow");
  return checked_add(
      checked_add(exact_bytes, exact_bytes / 20U,
                  "resource-plan batched transform margin overflows"),
      64U * 1024U * 1024U,
      "resource-plan batched transform margin overflows");
}

[[nodiscard]] std::size_t batched_direct_selected_peak(
    const LocalizedMp2Workload& workload,
    std::size_t owned_occupied_count, std::size_t virtual_tile,
    std::size_t lambda_ao_block, std::size_t quadrature_batch) {
  // The exact selected-domain footprint depends on monomer assignments,
  // which are intentionally unavailable to the machine planner. Bound it by
  // two rank-owned dense OVOV slices (direct and exchange) per batched point.
  const std::size_t selected_elements = checked_multiply(
      checked_multiply(
          checked_multiply(owned_occupied_count, workload.virtual_count,
                           "resource-plan selected left pairs overflow"),
          checked_multiply(workload.active_occupied_count,
                           workload.virtual_count,
                           "resource-plan selected right pairs overflow"),
          "resource-plan selected tensor overflow"),
      checked_multiply(2U, quadrature_batch,
                       "resource-plan selected point factor overflow"),
      "resource-plan selected point storage overflow");
  return checked_add(
      checked_add(
          batched_direct_selected_transform_peak(
              workload, owned_occupied_count, virtual_tile,
              lambda_ao_block, quadrature_batch),
          bytes_for_doubles(selected_elements,
                            "resource-plan selected storage overflow"),
          "resource-plan batched selected peak overflows"),
      coefficient_workspace_bytes(workload),
      "resource-plan batched selected peak overflows");
}

[[nodiscard]] std::size_t replicated_peak(
    const LocalizedMp2Workload& workload, std::size_t occupied_tile,
    std::size_t virtual_tile, std::size_t ovov_bytes) {
  return checked_add(
      checked_add(
          checked_add(ovov_bytes,
                      separable_transform_peak(workload, occupied_tile),
                      "resource-plan replicated peak overflows"),
          panel_workspace_bytes(workload, occupied_tile, virtual_tile),
          "resource-plan replicated peak overflows"),
      coefficient_workspace_bytes(workload),
      "resource-plan replicated peak overflows");
}

[[nodiscard]] std::size_t direct_tiled_peak(
    const LocalizedMp2Workload& workload, std::size_t occupied_tile,
    std::size_t virtual_tile) {
  return checked_add(
      checked_add(separable_transform_peak(workload, occupied_tile),
                  panel_workspace_bytes(workload, occupied_tile,
                                        virtual_tile),
                  "resource-plan direct peak overflows"),
      coefficient_workspace_bytes(workload),
      "resource-plan direct peak overflows");
}

[[nodiscard]] std::size_t maximum_distributed_ovov_bytes(
    const LocalizedMp2MachineResources& machine,
    const LocalizedMp2Workload& workload) {
  const std::size_t pairs = checked_multiply(
      workload.active_occupied_count, workload.virtual_count,
      "resource-plan distributed occupied-virtual pair count overflows");
  const std::size_t owned_pairs = ceiling_divide(pairs, machine.mpi_ranks);
  return bytes_for_doubles(
      checked_multiply(
          pairs, owned_pairs,
          "resource-plan distributed canonical OVOV slice overflows"),
      "resource-plan distributed canonical OVOV bytes overflow");
}

[[nodiscard]] std::size_t distributed_transform_occupied_count(
    const LocalizedMp2MachineResources& machine,
    const LocalizedMp2Workload& workload) {
  return ceiling_divide(workload.active_occupied_count,
                        machine.mpi_ranks);
}

[[nodiscard]] std::size_t distributed_peak(
    const LocalizedMp2Workload& workload,
    std::size_t transform_occupied_count,
    std::size_t occupied_tile, std::size_t virtual_tile,
    std::size_t local_ovov_bytes) {
  // A rank retains one disjoint canonical right-pair slice. During its build,
  // that slice coexists with one rank-balanced separable transformation.
  // During the energy step, the transform is gone and is replaced by the
  // bounded localized-panel workspace.
  const std::size_t transform_workspace = separable_transform_peak(
      workload, transform_occupied_count);
  const std::size_t panel_workspace = panel_workspace_bytes(
      workload, occupied_tile, virtual_tile);
  return checked_add(
      checked_add(local_ovov_bytes,
                  std::max(transform_workspace, panel_workspace),
                  "resource-plan distributed peak overflows"),
      coefficient_workspace_bytes(workload),
      "resource-plan distributed peak overflows");
}

}  // namespace

LocalizedMp2ResourcePlan plan_localized_mp2_resources(
    const LocalizedMp2MachineResources& machine,
    const LocalizedMp2Workload& workload,
    const LocalizedMp2ResourceOverrides& overrides) {
  if(machine.mpi_ranks == 0 || machine.ranks_per_node == 0 ||
     machine.threads_per_rank == 0 ||
     machine.available_memory_bytes_per_node == 0 ||
     machine.ranks_per_node > machine.mpi_ranks ||
     workload.atom_count == 0 || workload.ao_count == 0 ||
     workload.active_occupied_count == 0 || workload.virtual_count == 0 ||
     workload.quadrature_points == 0) {
    throw std::invalid_argument("localized MP2 resource inputs are invalid");
  }

  const std::size_t nodes = ceiling_divide(machine.mpi_ranks,
                                           machine.ranks_per_node);
  const std::size_t physical_share =
      machine.available_memory_bytes_per_node / machine.ranks_per_node;
  // Leave 20 percent for the operating system, MPI, Libint engines, RHF,
  // localization, and memory-estimation uncertainty.
  const std::size_t safe_share = physical_share - physical_share / 5U;
  if(safe_share == 0) {
    throw std::runtime_error("localized MP2 safe memory share is zero");
  }
  std::size_t usable_memory = safe_share;
  bool clamped = false;
  if(overrides.memory_limit_bytes_per_rank) {
    usable_memory = std::min(*overrides.memory_limit_bytes_per_rank,
                             safe_share);
    clamped = *overrides.memory_limit_bytes_per_rank > safe_share;
  }
  if(usable_memory == 0) {
    throw std::runtime_error("localized MP2 memory limit is zero");
  }

  const std::size_t ovov_bytes = canonical_ovov_bytes(workload);
  const bool direct_selected_policy =
      overrides.backend ==
      LocalizedLaplaceMp2Backend::localized_direct_selected;
  const bool localized_cached_policy =
      overrides.backend == LocalizedLaplaceMp2Backend::automatic ||
      overrides.backend ==
          LocalizedLaplaceMp2Backend::localized_cached_ovov;
  bool automatic_direct_fallback = false;
  if(direct_selected_policy &&
     workload.execution_mode != NmExecutionMode::one_two_monomer) {
    throw std::invalid_argument(
        "localized direct selected resource plan requires 1M+2M mode");
  }
  const bool distributed_selected_policy =
      localized_cached_policy ||
      overrides.backend ==
          LocalizedLaplaceMp2Backend::distributed_selected_ovov;
  const std::size_t local_ovov_bytes =
      distributed_selected_policy
          ? maximum_distributed_ovov_bytes(machine, workload)
          : direct_selected_policy
                ? 0U
                : ovov_bytes;
  const std::size_t transform_occupied_count =
      distributed_selected_policy
          ? distributed_transform_occupied_count(machine, workload)
          : 0U;
  const std::size_t direct_owned_occupied_count =
      direct_selected_policy
          ? ceiling_divide(workload.active_occupied_count,
                           machine.mpi_ranks)
          : 0U;
  std::size_t virtual_tile = overrides.virtual_tile_size.value_or(0U);
  if(virtual_tile == 0) {
    const std::size_t maximum_candidate =
        std::min<std::size_t>(36U, workload.virtual_count);
    for(std::size_t candidate = maximum_candidate; candidate != 0;
        --candidate) {
      if(workload.virtual_count % candidate == 0) {
        virtual_tile = candidate;
        break;
      }
    }
  }
  if(virtual_tile == 0 || virtual_tile > workload.virtual_count) {
    throw std::invalid_argument("localized MP2 virtual tile is invalid");
  }
  const std::size_t lambda_ao_block = direct_selected_policy
                                          ? virtual_tile
                                          : 1U;
  std::size_t quadrature_batch = 1U;
  if(direct_selected_policy) {
    for(std::size_t candidate = workload.quadrature_points;
        candidate != 0; --candidate) {
      if(batched_direct_selected_peak(
             workload, direct_owned_occupied_count, virtual_tile,
             lambda_ao_block, candidate) <= usable_memory) {
        quadrature_batch = candidate;
        break;
      }
    }
  }

  std::size_t occupied_tile = overrides.occupied_tile_size.value_or(0U);
  if(occupied_tile == 0) {
    // Prefer the smallest number of equal-sized batches. A distributed W20
    // plan can therefore select the complete 80-orbital left space while its
    // canonical right-pair ownership remains disjoint across ranks.
    for(std::size_t batches = 1;
        batches <= workload.active_occupied_count; ++batches) {
      const std::size_t candidate = ceiling_divide(
          workload.active_occupied_count, batches);
      const std::size_t candidate_peak =
          direct_selected_policy
              ? batched_direct_selected_peak(
                    workload, direct_owned_occupied_count, virtual_tile,
                    lambda_ao_block, quadrature_batch)
              : localized_cached_policy
                    ? checked_add(
                          distributed_peak(
                              workload, transform_occupied_count, candidate,
                              virtual_tile, local_ovov_bytes),
                          workload.execution_mode == NmExecutionMode::full
                              ? local_ovov_bytes
                              : 0U,
                          "localized cached resource peak overflows")
              : distributed_selected_policy
                    ? distributed_peak(workload, transform_occupied_count,
                                       candidate, virtual_tile,
                                       local_ovov_bytes)
                    : replicated_peak(workload, candidate, virtual_tile,
                                      ovov_bytes);
      if(candidate_peak <= usable_memory) {
        occupied_tile = candidate;
        break;
      }
    }
  }
  if(occupied_tile == 0 &&
     overrides.backend == LocalizedLaplaceMp2Backend::automatic) {
    automatic_direct_fallback = true;
    for(std::size_t batches = 1;
        batches <= workload.active_occupied_count; ++batches) {
      const std::size_t candidate = ceiling_divide(
          workload.active_occupied_count, batches);
      if(direct_tiled_peak(workload, candidate, virtual_tile) <=
         usable_memory) {
        occupied_tile = candidate;
        break;
      }
    }
  }
  if(occupied_tile == 0 ||
     occupied_tile > workload.active_occupied_count) {
    throw std::runtime_error(
        "no occupied tile fits the localized MP2 memory budget");
  }

  const std::size_t transform_peak =
      automatic_direct_fallback
          ? separable_transform_peak(workload, occupied_tile)
          : direct_selected_policy
                ? batched_direct_selected_transform_peak(
                      workload, direct_owned_occupied_count, virtual_tile,
                      lambda_ao_block, quadrature_batch)
                : distributed_selected_policy
                      ? separable_transform_peak(
                            workload, transform_occupied_count)
                      : separable_transform_peak(workload, occupied_tile);
  std::size_t total_peak =
      automatic_direct_fallback
          ? direct_tiled_peak(workload, occupied_tile, virtual_tile)
          : direct_selected_policy
                ? batched_direct_selected_peak(
                      workload, direct_owned_occupied_count, virtual_tile,
                      lambda_ao_block, quadrature_batch)
                : distributed_selected_policy
                      ? distributed_peak(
                            workload, transform_occupied_count,
                            occupied_tile, virtual_tile, local_ovov_bytes)
                      : replicated_peak(workload, occupied_tile,
                                        virtual_tile, ovov_bytes);
  if(localized_cached_policy && !automatic_direct_fallback &&
     workload.execution_mode == NmExecutionMode::full) {
    total_peak = checked_add(
        total_peak, local_ovov_bytes,
        "full localized cached scaled OVOV storage");
  }
  LocalizedLaplaceMp2Backend backend = overrides.backend;
  if(backend == LocalizedLaplaceMp2Backend::automatic) {
    backend = automatic_direct_fallback
                  ? LocalizedLaplaceMp2Backend::direct_tiled
                  : LocalizedLaplaceMp2Backend::localized_cached_ovov;
  }
  if((backend == LocalizedLaplaceMp2Backend::shared_selected_ovov ||
      backend == LocalizedLaplaceMp2Backend::replicated_selected_ovov ||
      backend == LocalizedLaplaceMp2Backend::distributed_selected_ovov ||
      backend == LocalizedLaplaceMp2Backend::localized_cached_ovov ||
      backend == LocalizedLaplaceMp2Backend::cached_canonical_ovov ||
      backend ==
          LocalizedLaplaceMp2Backend::localized_direct_selected) &&
     total_peak > usable_memory) {
    throw std::runtime_error(
        "requested localized MP2 plan exceeds the safe node-local memory "
        "share");
  }
  if(backend == LocalizedLaplaceMp2Backend::shared_selected_ovov &&
     machine.mpi_ranks != 1) {
    throw std::invalid_argument(
        "shared selected resource plan requires one MPI rank");
  }
  if(backend == LocalizedLaplaceMp2Backend::replicated_selected_ovov &&
     machine.mpi_ranks < 2) {
    throw std::invalid_argument(
        "replicated selected resource plan requires multiple MPI ranks");
  }

  return LocalizedMp2ResourcePlan{
      .nodes = nodes,
      .usable_memory_bytes_per_rank = usable_memory,
      .occupied_tile_size = occupied_tile,
      .virtual_tile_size = virtual_tile,
      .quadrature_batch_size = quadrature_batch,
      .lambda_ao_block_size = lambda_ao_block,
      .distributed_transform_occupied_count =
          direct_selected_policy ? direct_owned_occupied_count
          : (backend == LocalizedLaplaceMp2Backend::localized_cached_ovov ||
             backend ==
                 LocalizedLaplaceMp2Backend::distributed_selected_ovov)
                ? transform_occupied_count
                : 0U,
      .occupied_batches = ceiling_divide(
          workload.active_occupied_count, occupied_tile),
      .virtual_batches = ceiling_divide(workload.virtual_count,
                                        virtual_tile),
      .canonical_ovov_bytes =
          direct_selected_policy ||
                  backend ==
                      LocalizedLaplaceMp2Backend::localized_cached_ovov ||
                  backend == LocalizedLaplaceMp2Backend::direct_tiled
              ? 0U
              : ovov_bytes,
      .localized_ovov_bytes =
          backend == LocalizedLaplaceMp2Backend::localized_cached_ovov
              ? ovov_bytes
              : 0U,
      .estimated_transform_peak_bytes_per_rank = transform_peak,
      .estimated_total_peak_bytes_per_rank = total_peak,
      .backend = backend,
      .memory_was_clamped_to_machine = clamped,
  };
}

}  // namespace modernqc::mp2
