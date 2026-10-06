#include "lmp2_1m2m/core/build_info.hpp"

#include "lmp2_1m2m/core/version.hpp"

#include <sstream>

namespace lmp2_1m2m::core {
namespace {

constexpr bool mpi_enabled() noexcept {
#ifdef LMP2_1M2M_HAS_MPI
  return true;
#else
  return false;
#endif
}

constexpr bool openmp_enabled() noexcept {
#ifdef LMP2_1M2M_HAS_OPENMP
  return true;
#else
  return false;
#endif
}

constexpr bool hdf5_enabled() noexcept {
#ifdef LMP2_1M2M_HAS_HDF5
  return true;
#else
  return false;
#endif
}

constexpr bool libint2_enabled() noexcept {
#ifdef LMP2_1M2M_HAS_LIBINT2
  return true;
#else
  return false;
#endif
}

}  // namespace

BuildInfo build_info() {
  return BuildInfo{
      .version = LMP2_1M2M_VERSION,
      .compiler = std::string{LMP2_1M2M_CXX_COMPILER_ID} + " " + LMP2_1M2M_CXX_COMPILER_VERSION,
      .build_type = LMP2_1M2M_BUILD_TYPE,
      .mpi_enabled = mpi_enabled(),
      .openmp_enabled = openmp_enabled(),
      .hdf5_enabled = hdf5_enabled(),
      .libint2_enabled = libint2_enabled(),
  };
}

std::string format_build_info(const BuildInfo& info) {
  std::ostringstream output;
  output << "LMP2-1M2M " << info.version << '\n'
         << "  compiler: " << info.compiler << '\n'
         << "  build type: " << info.build_type << '\n'
         << "  MPI: " << (info.mpi_enabled ? "enabled" : "disabled") << '\n'
         << "  OpenMP: " << (info.openmp_enabled ? "enabled" : "disabled") << '\n'
         << "  HDF5: " << (info.hdf5_enabled ? "enabled" : "disabled") << '\n'
         << "  Libint2: " << (info.libint2_enabled ? "enabled" : "disabled");
  return output.str();
}

}  // namespace lmp2_1m2m::core
