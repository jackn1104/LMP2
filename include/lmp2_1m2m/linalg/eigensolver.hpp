#pragma once

#include "lmp2_1m2m/linalg/matrix.hpp"

#include <cstddef>
#include <vector>

namespace lmp2_1m2m::linalg {

struct SymmetricEigendecomposition {
  std::vector<double> eigenvalues;
  Matrix eigenvectors;
};

[[nodiscard]] SymmetricEigendecomposition diagonalize_symmetric(
    const Matrix& matrix, double symmetry_tolerance = 1.0e-12);

[[nodiscard]] Matrix solve_linear_system(Matrix coefficients,
                                         Matrix right_hand_sides);

[[nodiscard]] Matrix solve_symmetric_positive_definite(
    Matrix coefficients, Matrix right_hand_sides);

struct LeastSquaresResult {
  Matrix solution;
  std::vector<double> singular_values;
  std::size_t numerical_rank;
};

[[nodiscard]] LeastSquaresResult solve_least_squares_svd(
    Matrix design, const Matrix& right_hand_sides,
    double relative_singular_value_cutoff = 1.0e-12);

}  // namespace lmp2_1m2m::linalg
