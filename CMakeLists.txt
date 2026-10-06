cmake_minimum_required(VERSION 3.25)

project(
  LMP2-1M2M
  VERSION 0.1.0
  DESCRIPTION "Localized Laplace-transform MP2 for water clusters"
  LANGUAGES C CXX
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(CTest)
include(GNUInstallDirs)
include(LMP2_1M2MCompilerWarnings)
include(LMP2_1M2MDependencies)
include(LMP2_1M2MSanitizers)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(LMP2_1M2M_ENABLE_MPI "Enable MPI support" ON)
option(LMP2_1M2M_ENABLE_OPENMP "Enable OpenMP support" ON)
option(LMP2_1M2M_ENABLE_LIBINT2 "Require the Libint2 integral backend" ON)
option(LMP2_1M2M_ENABLE_HDF5 "Enable HDF5 restart files" ON)
option(LMP2_1M2M_ENABLE_PYTHON "Build optional Python bindings" OFF)
option(
  LMP2_1M2M_USE_CRAY_WRAPPER_LIBSCI
  "Use BLAS/LAPACK supplied by the Cray compiler wrapper"
  OFF
)
option(
  LMP2_1M2M_USE_CRAY_THREADED_LIBSCI
  "Let the Cray wrapper select OpenMP-threaded LibSci"
  OFF
)
option(
  LMP2_1M2M_ENABLE_SANITIZERS
  "Enable AddressSanitizer and UndefinedBehaviorSanitizer"
  OFF
)
option(
  LMP2_1M2M_FETCH_TEST_DEPENDENCIES
  "Fetch the pinned test-only Catch2 dependency when absent"
  OFF
)
option(LMP2_1M2M_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
set(
  LMP2_1M2M_SANITIZER_SET
  "address,undefined"
  CACHE STRING
  "Comma-separated sanitizer set"
)

if(APPLE AND LMP2_1M2M_ENABLE_OPENMP AND NOT OpenMP_ROOT)
  find_program(LMP2_1M2M_HOMEBREW_EXECUTABLE brew)
  if(LMP2_1M2M_HOMEBREW_EXECUTABLE)
    execute_process(
      COMMAND "${LMP2_1M2M_HOMEBREW_EXECUTABLE}" --prefix libomp
      OUTPUT_VARIABLE LMP2_1M2M_HOMEBREW_OPENMP_ROOT
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE LMP2_1M2M_HOMEBREW_OPENMP_STATUS
    )
    if(LMP2_1M2M_HOMEBREW_OPENMP_STATUS EQUAL 0 AND
       EXISTS "${LMP2_1M2M_HOMEBREW_OPENMP_ROOT}")
      set(OpenMP_ROOT "${LMP2_1M2M_HOMEBREW_OPENMP_ROOT}")
    endif()
  endif()
endif()

lmp2_1m2m_configure_dependencies()

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/include/lmp2_1m2m/core/version.hpp.in"
  "${CMAKE_CURRENT_BINARY_DIR}/generated/lmp2_1m2m/core/version.hpp"
  @ONLY
)

add_library(lmp2_1m2m_core STATIC
  src/basis/basis_name.cpp
  src/basis/libint2_basis_builder.cpp
  src/core/build_info.cpp
  src/integrals/libint2_provider.cpp
  src/linalg/eigensolver.cpp
  src/linalg/matrix.cpp
  src/linalg/orthogonalization.cpp
  src/localization/orbital_localization.cpp
  src/localization/pipek_mezey.cpp
  src/localization/pipek_mezey_ciah.cpp
  src/molecule/molecule.cpp
  src/molecule/xyz_reader.cpp
  src/mp2/canonical_mp2.cpp
  src/mp2/laplace_fit.cpp
  src/mp2/localized_tiled_laplace_mp2.cpp
  src/mp2/nm_decomposition.cpp
  src/mp2/resource_planner.cpp
  src/mp2/tiled_laplace_mp2.cpp
  src/scf/rhf.cpp
  src/scf/rhf_guess.cpp
)
if(LMP2_1M2M_ENABLE_HDF5)
  target_sources(lmp2_1m2m_core PRIVATE src/io/rhf_checkpoint.cpp)
endif()
add_library(LMP2_1M2M::core ALIAS lmp2_1m2m_core)

target_compile_features(lmp2_1m2m_core PUBLIC cxx_std_20)
target_include_directories(
  lmp2_1m2m_core
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>"
)
target_link_libraries(
  lmp2_1m2m_core
  PUBLIC
    LMP2_1M2M::MPI
    LMP2_1M2M::OpenMP
    LMP2_1M2M::BLAS
    LMP2_1M2M::LAPACK
  PRIVATE
    LMP2_1M2M::Eigen
    LMP2_1M2M::HDF5
    LMP2_1M2M::Libint2
)
lmp2_1m2m_set_project_warnings(lmp2_1m2m_core)
lmp2_1m2m_enable_sanitizers(lmp2_1m2m_core)

add_executable(lmp2_1m2m_engine apps/lmp2_1m2m_engine.cpp)
target_link_libraries(lmp2_1m2m_engine PRIVATE LMP2_1M2M::core)
lmp2_1m2m_set_project_warnings(lmp2_1m2m_engine)
lmp2_1m2m_enable_sanitizers(lmp2_1m2m_engine)

if(LMP2_1M2M_ENABLE_MPI)
  target_compile_definitions(lmp2_1m2m_core PUBLIC LMP2_1M2M_HAS_MPI=1)
endif()
if(LMP2_1M2M_ENABLE_OPENMP)
  target_compile_definitions(lmp2_1m2m_core PUBLIC LMP2_1M2M_HAS_OPENMP=1)
endif()
if(LMP2_1M2M_ENABLE_HDF5)
  target_compile_definitions(lmp2_1m2m_core PUBLIC LMP2_1M2M_HAS_HDF5=1)
endif()
if(LMP2_1M2M_ENABLE_LIBINT2)
  target_compile_definitions(lmp2_1m2m_core PUBLIC LMP2_1M2M_HAS_LIBINT2=1)
endif()

install(
  TARGETS lmp2_1m2m_engine
  RUNTIME DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/LMP2-1M2M"
)
install(PROGRAMS LMP2-1M2M DESTINATION "${CMAKE_INSTALL_BINDIR}")

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

message(STATUS "")
message(STATUS "LMP2-1M2M configuration")
message(STATUS "  Version:       ${PROJECT_VERSION}")
message(STATUS "  Build type:    ${CMAKE_BUILD_TYPE}")
message(STATUS "  MPI:           ${LMP2_1M2M_ENABLE_MPI}")
message(STATUS "  OpenMP:        ${LMP2_1M2M_ENABLE_OPENMP}")
message(STATUS "  Libint2:       ${LMP2_1M2M_ENABLE_LIBINT2}")
message(STATUS "  HDF5 restart:  ${LMP2_1M2M_ENABLE_HDF5}")
message(STATUS "  Tests:         ${BUILD_TESTING}")
