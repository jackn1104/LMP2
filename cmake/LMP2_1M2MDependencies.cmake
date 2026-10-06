include(FetchContent)

function(lmp2_1m2m_add_interface_alias alias target)
  if(NOT TARGET "${alias}")
    add_library("${alias}" INTERFACE IMPORTED GLOBAL)
    target_link_libraries("${alias}" INTERFACE "${target}")
  endif()
endfunction()

function(lmp2_1m2m_configure_dependencies)
  if(LMP2_1M2M_USE_CRAY_THREADED_LIBSCI AND
     (NOT LMP2_1M2M_USE_CRAY_WRAPPER_LIBSCI OR
      NOT LMP2_1M2M_ENABLE_OPENMP))
    message(
      FATAL_ERROR
      "Threaded Cray LibSci requires the Cray-wrapper LibSci and OpenMP options"
    )
  endif()
  if(LMP2_1M2M_ENABLE_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
    lmp2_1m2m_add_interface_alias(LMP2_1M2M::MPI MPI::MPI_CXX)
  else()
    add_library(LMP2_1M2M_MPI INTERFACE)
    add_library(LMP2_1M2M::MPI ALIAS LMP2_1M2M_MPI)
  endif()

  if(LMP2_1M2M_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
    if(LMP2_1M2M_USE_CRAY_WRAPPER_LIBSCI)
      if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        message(
          FATAL_ERROR
          "LMP2_1M2M_USE_CRAY_WRAPPER_LIBSCI currently requires GNU C++ OpenMP"
        )
      endif()
      if(LMP2_1M2M_USE_CRAY_THREADED_LIBSCI)
        separate_arguments(
          LMP2_1M2M_OPENMP_CXX_COMPILE_OPTIONS
          NATIVE_COMMAND
          "${OpenMP_CXX_FLAGS}"
        )
        if(NOT LMP2_1M2M_OPENMP_CXX_COMPILE_OPTIONS)
          message(FATAL_ERROR "GNU OpenMP compile options are empty")
        endif()
        add_library(LMP2_1M2M_OpenMP INTERFACE)
        target_link_libraries(
          LMP2_1M2M_OpenMP INTERFACE OpenMP::OpenMP_CXX
        )
        target_link_options(
          LMP2_1M2M_OpenMP
          INTERFACE ${LMP2_1M2M_OPENMP_CXX_COMPILE_OPTIONS}
        )
        add_library(LMP2_1M2M::OpenMP ALIAS LMP2_1M2M_OpenMP)
        message(
          STATUS
          "Using GNU OpenMP and OpenMP-threaded Cray LibSci selected by the wrapper"
        )
      else()
        separate_arguments(
          LMP2_1M2M_OPENMP_CXX_COMPILE_OPTIONS
          NATIVE_COMMAND
          "${OpenMP_CXX_FLAGS}"
        )
        if(NOT LMP2_1M2M_OPENMP_CXX_COMPILE_OPTIONS)
          message(FATAL_ERROR "GNU OpenMP compile options are empty")
        endif()
        set(lmp2_1m2m_gomp_library "${OpenMP_gomp_LIBRARY}")
        if(NOT lmp2_1m2m_gomp_library)
          foreach(openmp_library IN LISTS OpenMP_CXX_LIBRARIES)
            get_filename_component(openmp_library_name "${openmp_library}" NAME)
            if(openmp_library STREQUAL "gomp" OR
               openmp_library_name MATCHES "^libgomp\\.")
              set(lmp2_1m2m_gomp_library "${openmp_library}")
              break()
            endif()
          endforeach()
        endif()
        if(NOT lmp2_1m2m_gomp_library)
          find_library(lmp2_1m2m_gomp_library NAMES gomp REQUIRED)
        endif()
        add_library(LMP2_1M2M_OpenMP INTERFACE)
        target_compile_options(
          LMP2_1M2M_OpenMP
          INTERFACE ${LMP2_1M2M_OPENMP_CXX_COMPILE_OPTIONS}
        )
        target_link_libraries(
          LMP2_1M2M_OpenMP INTERFACE "${lmp2_1m2m_gomp_library}"
        )
        add_library(LMP2_1M2M::OpenMP ALIAS LMP2_1M2M_OpenMP)
        message(
          STATUS "Using GNU OpenMP runtime only: ${lmp2_1m2m_gomp_library}"
        )
      endif()
    else()
      if(LMP2_1M2M_USE_CRAY_THREADED_LIBSCI)
        message(
          FATAL_ERROR
          "LMP2_1M2M_USE_CRAY_THREADED_LIBSCI requires LMP2_1M2M_USE_CRAY_WRAPPER_LIBSCI"
        )
      endif()
      lmp2_1m2m_add_interface_alias(LMP2_1M2M::OpenMP OpenMP::OpenMP_CXX)
    endif()
  else()
    add_library(LMP2_1M2M_OpenMP INTERFACE)
    add_library(LMP2_1M2M::OpenMP ALIAS LMP2_1M2M_OpenMP)
  endif()

  if(LMP2_1M2M_USE_CRAY_WRAPPER_LIBSCI)
    add_library(LMP2_1M2M_BLAS INTERFACE)
    add_library(LMP2_1M2M::BLAS ALIAS LMP2_1M2M_BLAS)
    add_library(LMP2_1M2M_LAPACK INTERFACE)
    add_library(LMP2_1M2M::LAPACK ALIAS LMP2_1M2M_LAPACK)
    if(LMP2_1M2M_USE_CRAY_THREADED_LIBSCI)
      message(
        STATUS "Using threaded BLAS/LAPACK supplied by the Cray compiler wrapper"
      )
    else()
      message(
        STATUS "Using serial BLAS/LAPACK supplied by the Cray compiler wrapper"
      )
    endif()
  else()
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    lmp2_1m2m_add_interface_alias(LMP2_1M2M::BLAS BLAS::BLAS)
    lmp2_1m2m_add_interface_alias(LMP2_1M2M::LAPACK LAPACK::LAPACK)
  endif()

  if(LMP2_1M2M_ENABLE_LIBINT2)
    # Libint2's public C++ headers include Eigen. Some packages (including
    # Homebrew's) do not propagate that include directory from their exported
    # target, so make the transitive header requirement explicit here.
    find_package(Eigen3 CONFIG REQUIRED)
    lmp2_1m2m_add_interface_alias(LMP2_1M2M::Eigen Eigen3::Eigen)
    find_package(Libint2 CONFIG REQUIRED)
    set(libint2_target "")
    foreach(candidate IN ITEMS Libint2::int2 Libint2::libint2 libint2::int2 libint2::libint2)
      if(TARGET "${candidate}")
        set(libint2_target "${candidate}")
        break()
      endif()
    endforeach()
    if(libint2_target STREQUAL "")
      message(FATAL_ERROR "Libint2 was found, but its CMake package exported no recognized target")
    endif()
    lmp2_1m2m_add_interface_alias(LMP2_1M2M::Libint2 "${libint2_target}")
  else()
    add_library(LMP2_1M2M_Eigen INTERFACE)
    add_library(LMP2_1M2M::Eigen ALIAS LMP2_1M2M_Eigen)
    add_library(LMP2_1M2M_Libint2 INTERFACE)
    add_library(LMP2_1M2M::Libint2 ALIAS LMP2_1M2M_Libint2)
  endif()

  if(LMP2_1M2M_ENABLE_HDF5)
    find_package(HDF5 REQUIRED COMPONENTS C)
    if(TARGET hdf5::hdf5)
      lmp2_1m2m_add_interface_alias(LMP2_1M2M::HDF5 hdf5::hdf5)
    elseif(TARGET HDF5::HDF5)
      lmp2_1m2m_add_interface_alias(LMP2_1M2M::HDF5 HDF5::HDF5)
    else()
      add_library(LMP2_1M2M_HDF5 INTERFACE)
      target_include_directories(LMP2_1M2M_HDF5 INTERFACE "${HDF5_INCLUDE_DIRS}")
      target_link_libraries(LMP2_1M2M_HDF5 INTERFACE "${HDF5_LIBRARIES}")
      add_library(LMP2_1M2M::HDF5 ALIAS LMP2_1M2M_HDF5)
    endif()
  else()
    add_library(LMP2_1M2M_HDF5 INTERFACE)
    add_library(LMP2_1M2M::HDF5 ALIAS LMP2_1M2M_HDF5)
  endif()

  if(LMP2_1M2M_ENABLE_PYTHON)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(pybind11 CONFIG REQUIRED)
  else()
    find_package(Python3 QUIET COMPONENTS Interpreter)
  endif()
  set(Python3_Interpreter_FOUND "${Python3_Interpreter_FOUND}" PARENT_SCOPE)
  set(Python3_EXECUTABLE "${Python3_EXECUTABLE}" PARENT_SCOPE)

  if(BUILD_TESTING)
    find_package(Catch2 3 CONFIG QUIET)
    if(NOT Catch2_FOUND AND LMP2_1M2M_FETCH_TEST_DEPENDENCIES)
      FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.8.1
        GIT_SHALLOW TRUE
      )
      FetchContent_MakeAvailable(Catch2)
    endif()
    if(NOT TARGET Catch2::Catch2WithMain)
      message(
        FATAL_ERROR
        "Catch2 3 is required for C++ tests. Install it, set Catch2_ROOT, "
        "disable BUILD_TESTING, or explicitly enable LMP2_1M2M_FETCH_TEST_DEPENDENCIES."
      )
    endif()
  endif()
endfunction()
