add_executable(
  lmp2_unit_tests
  unit/test_localized_tiled_laplace_mp2.cpp
  unit/test_resource_planner.cpp
)
target_link_libraries(
  lmp2_unit_tests
  PRIVATE
    LMP2::core
    ModernQC::Eigen
    ModernQC::Libint2
    Catch2::Catch2WithMain
)
modernqc_set_project_warnings(lmp2_unit_tests)
modernqc_enable_sanitizers(lmp2_unit_tests)

add_test(
  NAME lmp2_resource_tests
  COMMAND lmp2_unit_tests "resource planner*"
)
add_test(
  NAME lmp2_block_symmetry_tests
  COMMAND lmp2_unit_tests "localized block-pair*"
)
add_test(
  NAME lmp2_fullspace_energy_tests
  COMMAND lmp2_unit_tests "localized tiled Laplace MP2*"
)
set(
  LMP2_TEST_ENVIRONMENT
  "OMP_NUM_THREADS=2;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1"
)
set_tests_properties(
  lmp2_resource_tests
  lmp2_block_symmetry_tests
  lmp2_fullspace_energy_tests
  PROPERTIES
    ENVIRONMENT "${LMP2_TEST_ENVIRONMENT}"
    TIMEOUT 600
)
set_tests_properties(
  lmp2_resource_tests
  PROPERTIES LABELS "fast"
)
set_tests_properties(
  lmp2_block_symmetry_tests
  lmp2_fullspace_energy_tests
  PROPERTIES LABELS "numerical"
)

if(Python3_Interpreter_FOUND)
  add_test(
    NAME lmp2_public_cli
    COMMAND
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/test_public_cli.py"
      "${PROJECT_SOURCE_DIR}/LMP2"
      "$<TARGET_FILE:lmp2_engine>"
      "${PROJECT_SOURCE_DIR}"
  )
  set_tests_properties(
    lmp2_public_cli
    PROPERTIES
      ENVIRONMENT "${LMP2_TEST_ENVIRONMENT}"
      LABELS "numerical"
      TIMEOUT 120
  )
endif()
