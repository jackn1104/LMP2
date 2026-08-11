#include "modernqc/mp2/resource_planner.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("resource planner distributes W20 without replicated storage") {
  using namespace modernqc::mp2;
  constexpr std::size_t gib = 1024U * 1024U * 1024U;
  const LocalizedMp2ResourcePlan plan = plan_localized_mp2_resources(
      LocalizedMp2MachineResources{
          .mpi_ranks = 4,
          .ranks_per_node = 1,
          .threads_per_rank = 128,
          .available_memory_bytes_per_node = 512U * gib,
      },
      LocalizedMp2Workload{
          .atom_count = 60,
          .ao_count = 820,
          .active_occupied_count = 80,
          .virtual_count = 720,
          .quadrature_points = 50,
          .execution_mode = NmExecutionMode::one_two_monomer,
      },
      LocalizedMp2ResourceOverrides{
          .occupied_tile_size = 80U,
          .virtual_tile_size = 160U,
          .memory_limit_bytes_per_rank = 400U * gib,
      });

  CHECK(plan.nodes == 4);
  CHECK(plan.occupied_tile_size == 80);
  CHECK(plan.distributed_transform_occupied_count == 20);
  CHECK(plan.occupied_batches == 1);
  CHECK(plan.virtual_tile_size == 160);
  CHECK(plan.virtual_batches == 5);
  CHECK(plan.backend ==
        LocalizedLaplaceMp2Backend::localized_cached_ovov);
  CHECK(plan.canonical_ovov_bytes == 0);
  CHECK(plan.localized_ovov_bytes == 26'542'080'000U);
  CHECK(plan.estimated_total_peak_bytes_per_rank <
        plan.usable_memory_bytes_per_rank);
  CHECK(plan.estimated_transform_peak_bytes_per_rank > 150U * gib);
  CHECK(plan.estimated_transform_peak_bytes_per_rank < 200U * gib);
}

TEST_CASE("resource planner respects node-local rank memory sharing") {
  using namespace modernqc::mp2;
  constexpr std::size_t gib = 1024U * 1024U * 1024U;
  const LocalizedMp2ResourcePlan plan = plan_localized_mp2_resources(
      LocalizedMp2MachineResources{
          .mpi_ranks = 32,
          .ranks_per_node = 8,
          .threads_per_rank = 16,
          .available_memory_bytes_per_node = 512U * gib,
      },
      LocalizedMp2Workload{
          .atom_count = 60,
          .ao_count = 820,
          .active_occupied_count = 80,
          .virtual_count = 720,
          .quadrature_points = 50,
          .execution_mode = NmExecutionMode::one_two_monomer,
      });

  CHECK(plan.nodes == 4);
  CHECK(plan.usable_memory_bytes_per_rank < 52U * gib);
  CHECK(plan.backend ==
        LocalizedLaplaceMp2Backend::localized_cached_ovov);
  CHECK(plan.canonical_ovov_bytes == 0);
  CHECK(plan.localized_ovov_bytes == 26'542'080'000U);
  CHECK(plan.occupied_tile_size == 80);
  CHECK(plan.distributed_transform_occupied_count == 3);
  CHECK(plan.estimated_transform_peak_bytes_per_rank > 20U * gib);
}

TEST_CASE("resource planner keeps replication explicit only") {
  using namespace modernqc::mp2;
  constexpr std::size_t gib = 1024U * 1024U * 1024U;
  const LocalizedMp2ResourcePlan plan = plan_localized_mp2_resources(
      LocalizedMp2MachineResources{
          .mpi_ranks = 4,
          .ranks_per_node = 1,
          .threads_per_rank = 128,
          .available_memory_bytes_per_node = 512U * gib,
      },
      LocalizedMp2Workload{
          .atom_count = 60,
          .ao_count = 820,
          .active_occupied_count = 80,
          .virtual_count = 720,
          .quadrature_points = 64,
          .execution_mode = NmExecutionMode::one_two_monomer,
      },
      LocalizedMp2ResourceOverrides{
          .occupied_tile_size = 40,
          .virtual_tile_size = 36,
          .memory_limit_bytes_per_rank = 400U * gib,
          .backend =
              LocalizedLaplaceMp2Backend::replicated_selected_ovov,
      });

  CHECK(plan.backend ==
        LocalizedLaplaceMp2Backend::replicated_selected_ovov);
  CHECK(plan.occupied_tile_size == 40);
}

TEST_CASE("resource planner clamps an unsafe explicit memory request") {
  using namespace modernqc::mp2;
  constexpr std::size_t gib = 1024U * 1024U * 1024U;
  const LocalizedMp2ResourcePlan plan = plan_localized_mp2_resources(
      LocalizedMp2MachineResources{
          .mpi_ranks = 1,
          .ranks_per_node = 1,
          .threads_per_rank = 8,
          .available_memory_bytes_per_node = 32U * gib,
      },
      LocalizedMp2Workload{
          .atom_count = 6,
          .ao_count = 82,
          .active_occupied_count = 8,
          .virtual_count = 72,
          .quadrature_points = 40,
          .execution_mode = NmExecutionMode::one_two_monomer,
      },
      LocalizedMp2ResourceOverrides{
          .memory_limit_bytes_per_rank = 128U * gib,
      });

  CHECK(plan.memory_was_clamped_to_machine);
  CHECK(plan.usable_memory_bytes_per_rank < 32U * gib);
  CHECK(plan.backend ==
        LocalizedLaplaceMp2Backend::localized_cached_ovov);
  CHECK(plan.canonical_ovov_bytes == 0);
  CHECK(plan.localized_ovov_bytes > 0);
}

TEST_CASE("resource planner exposes direct selected without canonical OVOV") {
  using namespace modernqc::mp2;
  constexpr std::size_t gib = 1024U * 1024U * 1024U;
  const LocalizedMp2ResourcePlan plan = plan_localized_mp2_resources(
      LocalizedMp2MachineResources{
          .mpi_ranks = 4,
          .ranks_per_node = 1,
          .threads_per_rank = 128,
          .available_memory_bytes_per_node = 512U * gib,
      },
      LocalizedMp2Workload{
          .atom_count = 18,
          .ao_count = 246,
          .active_occupied_count = 24,
          .virtual_count = 216,
          .quadrature_points = 7,
          .execution_mode = NmExecutionMode::one_two_monomer,
      },
      LocalizedMp2ResourceOverrides{
          .occupied_tile_size = 4,
          .virtual_tile_size = 36,
          .memory_limit_bytes_per_rank = 400U * gib,
          .backend =
              LocalizedLaplaceMp2Backend::localized_direct_selected,
      });

  CHECK(plan.backend ==
        LocalizedLaplaceMp2Backend::localized_direct_selected);
  CHECK(plan.canonical_ovov_bytes == 0);
  CHECK(plan.distributed_transform_occupied_count == 6);
  CHECK(plan.quadrature_batch_size == 7);
  CHECK(plan.lambda_ao_block_size == 36);
  CHECK(plan.estimated_transform_peak_bytes_per_rank > 0);
  CHECK(plan.estimated_total_peak_bytes_per_rank >=
        plan.estimated_transform_peak_bytes_per_rank);
  CHECK(plan.estimated_total_peak_bytes_per_rank <
        plan.usable_memory_bytes_per_rank);

  CHECK_THROWS_AS(
      plan_localized_mp2_resources(
          LocalizedMp2MachineResources{
              .mpi_ranks = 1,
              .ranks_per_node = 1,
              .threads_per_rank = 8,
              .available_memory_bytes_per_node = 32U * gib,
          },
          LocalizedMp2Workload{
              .atom_count = 6,
              .ao_count = 82,
              .active_occupied_count = 8,
              .virtual_count = 72,
              .quadrature_points = 7,
              .execution_mode = NmExecutionMode::full,
          },
          LocalizedMp2ResourceOverrides{
              .backend =
                  LocalizedLaplaceMp2Backend::localized_direct_selected,
          }),
      std::invalid_argument);
}

TEST_CASE("resource planner distributes one localized cached OVOV tensor") {
  using namespace modernqc::mp2;
  constexpr std::size_t gib = 1024U * 1024U * 1024U;
  const LocalizedMp2ResourcePlan plan = plan_localized_mp2_resources(
      LocalizedMp2MachineResources{
          .mpi_ranks = 4,
          .ranks_per_node = 1,
          .threads_per_rank = 128,
          .available_memory_bytes_per_node = 512U * gib,
      },
      LocalizedMp2Workload{
          .atom_count = 60,
          .ao_count = 820,
          .active_occupied_count = 80,
          .virtual_count = 720,
          .quadrature_points = 7,
          .execution_mode = NmExecutionMode::one_two_monomer,
      },
      LocalizedMp2ResourceOverrides{
          .occupied_tile_size = 4,
          .virtual_tile_size = 36,
          .memory_limit_bytes_per_rank = 400U * gib,
          .backend = LocalizedLaplaceMp2Backend::localized_cached_ovov,
      });

  CHECK(plan.backend ==
        LocalizedLaplaceMp2Backend::localized_cached_ovov);
  CHECK(plan.canonical_ovov_bytes == 0);
  CHECK(plan.localized_ovov_bytes == 26'542'080'000U);
  CHECK(plan.distributed_transform_occupied_count == 20);
  CHECK(plan.estimated_total_peak_bytes_per_rank <
        plan.usable_memory_bytes_per_rank);

  const LocalizedMp2ResourcePlan full_plan =
      plan_localized_mp2_resources(
          LocalizedMp2MachineResources{
              .mpi_ranks = 4,
              .ranks_per_node = 1,
              .threads_per_rank = 128,
              .available_memory_bytes_per_node = 512U * gib,
          },
          LocalizedMp2Workload{
              .atom_count = 60,
              .ao_count = 820,
              .active_occupied_count = 80,
              .virtual_count = 720,
              .quadrature_points = 7,
              .execution_mode = NmExecutionMode::full,
          },
          LocalizedMp2ResourceOverrides{
              .occupied_tile_size = 4,
              .virtual_tile_size = 36,
              .memory_limit_bytes_per_rank = 400U * gib,
          });

  CHECK(full_plan.backend ==
        LocalizedLaplaceMp2Backend::localized_cached_ovov);
  CHECK(full_plan.canonical_ovov_bytes == 0);
  CHECK(full_plan.localized_ovov_bytes == 26'542'080'000U);
  CHECK(full_plan.distributed_transform_occupied_count == 20);
  CHECK(full_plan.estimated_total_peak_bytes_per_rank >
        plan.estimated_total_peak_bytes_per_rank);
  CHECK(full_plan.estimated_total_peak_bytes_per_rank <
        full_plan.usable_memory_bytes_per_rank);
}

TEST_CASE("resource planner falls back when localized cache cannot fit") {
  using namespace modernqc::mp2;
  constexpr std::size_t gib = 1024U * 1024U * 1024U;
  const LocalizedMp2ResourcePlan plan = plan_localized_mp2_resources(
      LocalizedMp2MachineResources{
          .mpi_ranks = 4,
          .ranks_per_node = 1,
          .threads_per_rank = 128,
          .available_memory_bytes_per_node = 64U * gib,
      },
      LocalizedMp2Workload{
          .atom_count = 60,
          .ao_count = 820,
          .active_occupied_count = 80,
          .virtual_count = 720,
          .quadrature_points = 7,
          .execution_mode = NmExecutionMode::full,
      });

  CHECK(plan.backend == LocalizedLaplaceMp2Backend::direct_tiled);
  CHECK(plan.canonical_ovov_bytes == 0);
  CHECK(plan.localized_ovov_bytes == 0);
  CHECK(plan.distributed_transform_occupied_count == 0);
  CHECK(plan.estimated_total_peak_bytes_per_rank <
        plan.usable_memory_bytes_per_rank);
}

TEST_CASE("resource planner rejects invalid topology") {
  using namespace modernqc::mp2;
  CHECK_THROWS_AS(
      plan_localized_mp2_resources(
          LocalizedMp2MachineResources{}, LocalizedMp2Workload{}),
      std::invalid_argument);
}
