#include "modernqc/scf/rhf_guess.hpp"

#include "modernqc/linalg/eigensolver.hpp"
#include "modernqc/linalg/matrix.hpp"
#include "modernqc/linalg/orthogonalization.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace modernqc::scf {
namespace {

[[nodiscard]] linalg::Matrix first_columns(
    const linalg::Matrix& matrix, std::size_t column_count) {
  if(column_count == 0 || column_count > matrix.columns()) {
    throw std::invalid_argument(
        "projected RHF guess has an invalid occupied-orbital count");
  }
  linalg::Matrix result{matrix.rows(), column_count};
  for(std::size_t column = 0; column < column_count; ++column) {
    for(std::size_t row = 0; row < matrix.rows(); ++row) {
      result(row, column) = matrix(row, column);
    }
  }
  return result;
}

[[nodiscard]] double trace_product(const linalg::Matrix& left,
                                   const linalg::Matrix& right) {
  if(left.rows() != left.columns() || right.rows() != right.columns() ||
     left.rows() != right.rows()) {
    throw std::invalid_argument(
        "trace product requires equal square matrices");
  }
  double result = 0.0;
  for(std::size_t column = 0; column < left.columns(); ++column) {
    for(std::size_t row = 0; row < left.rows(); ++row) {
      result += left(row, column) * right(column, row);
    }
  }
  if(!std::isfinite(result)) {
    throw std::runtime_error("projected RHF electron count is nonfinite");
  }
  return result;
}

}  // namespace

ProjectedRhfGuess project_occupied_rhf_guess(
    const linalg::Matrix& source_coefficients,
    std::size_t occupied_orbitals,
    const linalg::Matrix& target_overlap,
    const linalg::Matrix& target_source_overlap,
    double target_linear_dependency_tolerance,
    double occupied_rank_tolerance) {
  if(!std::isfinite(target_linear_dependency_tolerance) ||
     target_linear_dependency_tolerance <= 0.0 ||
     !std::isfinite(occupied_rank_tolerance) ||
     occupied_rank_tolerance <= 0.0) {
    throw std::invalid_argument(
        "projected RHF tolerances must be finite and positive");
  }
  if(target_overlap.rows() == 0 ||
     target_overlap.rows() != target_overlap.columns() ||
     target_source_overlap.rows() != target_overlap.rows() ||
     target_source_overlap.columns() != source_coefficients.rows()) {
    throw std::invalid_argument(
        "projected RHF matrix dimensions are inconsistent");
  }
  linalg::require_finite(source_coefficients,
                         "source RHF coefficients");
  linalg::require_finite(target_overlap, "target RHF overlap");
  linalg::require_finite(target_source_overlap,
                         "target-source RHF overlap");

  const linalg::Orthogonalization target_orthogonalization =
      linalg::build_orthogonalizer(
          target_overlap, linalg::OrthogonalizationMethod::canonical,
          target_linear_dependency_tolerance);
  if(occupied_orbitals >
     target_orthogonalization.transformation.columns()) {
    throw std::runtime_error(
        "target overlap metric has fewer retained vectors than occupied "
        "orbitals");
  }
  const linalg::Matrix source_occupied =
      first_columns(source_coefficients, occupied_orbitals);
  const linalg::Matrix projected_orthonormal = linalg::multiply(
      linalg::transpose(target_orthogonalization.transformation),
      linalg::multiply(target_source_overlap, source_occupied));
  linalg::Matrix occupied_gram = linalg::multiply(
      linalg::transpose(projected_orthonormal), projected_orthonormal);
  linalg::symmetrize_in_place(occupied_gram);
  const linalg::SymmetricEigendecomposition gram_eigensystem =
      linalg::diagonalize_symmetric(occupied_gram, 2.0e-12);
  const double minimum_gram_eigenvalue =
      gram_eigensystem.eigenvalues.front();
  if(!std::isfinite(minimum_gram_eigenvalue) ||
     minimum_gram_eigenvalue <= occupied_rank_tolerance) {
    throw std::runtime_error(
        "cross-basis RHF projection loses occupied-space rank");
  }

  linalg::Matrix scaled_eigenvectors = gram_eigensystem.eigenvectors;
  for(std::size_t column = 0;
      column < scaled_eigenvectors.columns(); ++column) {
    const double inverse_sqrt =
        1.0 / std::sqrt(gram_eigensystem.eigenvalues[column]);
    for(std::size_t row = 0; row < scaled_eigenvectors.rows(); ++row) {
      scaled_eigenvectors(row, column) *= inverse_sqrt;
    }
  }
  const linalg::Matrix inverse_sqrt_gram = linalg::multiply(
      scaled_eigenvectors,
      linalg::transpose(gram_eigensystem.eigenvectors));
  const linalg::Matrix occupied_coefficients = linalg::multiply(
      target_orthogonalization.transformation,
      linalg::multiply(projected_orthonormal, inverse_sqrt_gram));
  const linalg::Matrix occupied_metric = linalg::multiply(
      linalg::transpose(occupied_coefficients),
      linalg::multiply(target_overlap, occupied_coefficients));
  const double orthonormality_error = linalg::maximum_absolute_value(
      linalg::subtract(
          occupied_metric,
          linalg::Matrix::identity(occupied_metric.rows())));
  linalg::Matrix density = linalg::scaled(
      linalg::multiply(occupied_coefficients,
                       linalg::transpose(occupied_coefficients)),
      2.0);
  linalg::symmetrize_in_place(density);
  const double electron_count = trace_product(density, target_overlap);
  linalg::require_finite(density, "projected RHF density");
  if(orthonormality_error > 1.0e-10 ||
     std::abs(electron_count - 2.0 * static_cast<double>(occupied_orbitals)) >
         1.0e-8) {
    throw std::runtime_error(
        "cross-basis RHF projection violates metric invariants");
  }
  return ProjectedRhfGuess{
      .density = std::move(density),
      .occupied_coefficients = std::move(occupied_coefficients),
      .target_metric_rank =
          target_orthogonalization.transformation.columns(),
      .minimum_occupied_gram_eigenvalue = minimum_gram_eigenvalue,
      .occupied_orthonormality_maximum_error = orthonormality_error,
      .electron_count = electron_count,
  };
}

}  // namespace modernqc::scf
