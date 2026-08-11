# Perlmutter CPU initial cache for PrgEnv-gnu with NERSC's Cray wrappers.
# The cc/CC wrappers target compute nodes and supply Cray MPICH. Select a
# current, mutually compatible module/dependency environment before CMake.

set(CMAKE_C_COMPILER "cc" CACHE FILEPATH "NERSC Cray C wrapper")
set(CMAKE_CXX_COMPILER "CC" CACHE FILEPATH "NERSC Cray C++ wrapper")

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type")
set(BUILD_TESTING ON CACHE BOOL "Build tests")
set(MODERNQC_ENABLE_MPI ON CACHE BOOL "Enable MPI")
set(MODERNQC_ENABLE_OPENMP ON CACHE BOOL "Enable OpenMP")
set(MODERNQC_USE_CRAY_WRAPPER_LIBSCI ON CACHE BOOL
    "Use one serial Cray LibSci family with the GNU OpenMP runtime")
set(MODERNQC_ENABLE_LIBINT2 ON CACHE BOOL "Require Libint2")
set(MODERNQC_ENABLE_HDF5 ON CACHE BOOL "Enable HDF5")
set(MODERNQC_ENABLE_PYTHON OFF CACHE BOOL "Build pybind11 bindings")
set(MODERNQC_FETCH_TEST_DEPENDENCIES OFF CACHE BOOL "Compute nodes must not fetch dependencies")
