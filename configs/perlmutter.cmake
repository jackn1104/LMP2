# Perlmutter CPU initial cache for PrgEnv-gnu with NERSC's Cray wrappers.
# The cc/CC wrappers target compute nodes and supply Cray MPICH. Select a
# current, mutually compatible module/dependency environment before CMake.

set(CMAKE_C_COMPILER "cc" CACHE FILEPATH "NERSC Cray C wrapper")
set(CMAKE_CXX_COMPILER "CC" CACHE FILEPATH "NERSC Cray C++ wrapper")

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type")
set(BUILD_TESTING ON CACHE BOOL "Build tests")
set(LMP2_1M2M_ENABLE_MPI ON CACHE BOOL "Enable MPI")
set(LMP2_1M2M_ENABLE_OPENMP ON CACHE BOOL "Enable OpenMP")
set(LMP2_1M2M_USE_CRAY_WRAPPER_LIBSCI ON CACHE BOOL
    "Use one serial Cray LibSci family with the GNU OpenMP runtime")
set(LMP2_1M2M_ENABLE_LIBINT2 ON CACHE BOOL "Require Libint2")
set(LMP2_1M2M_ENABLE_HDF5 ON CACHE BOOL "Enable HDF5")
set(LMP2_1M2M_ENABLE_PYTHON OFF CACHE BOOL "Build pybind11 bindings")
set(LMP2_1M2M_FETCH_TEST_DEPENDENCIES OFF CACHE BOOL "Compute nodes must not fetch dependencies")
