#include "modernqc/io/rhf_checkpoint.hpp"

#include "modernqc/core/build_info.hpp"
#include "modernqc/linalg/matrix.hpp"
#include "modernqc/mp2/laplace_fit.hpp"
#include "modernqc/mp2/laplace_mp2.hpp"
#include "modernqc/mp2/tiled_laplace_mp2.hpp"

#include <hdf5.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace modernqc::io {
namespace {

class Hdf5Handle {
 public:
  using Closer = herr_t (*)(hid_t);

  Hdf5Handle() = default;
  Hdf5Handle(hid_t identifier, Closer closer)
      : identifier_{identifier}, closer_{closer} {
    if(identifier_ < 0) {
      throw std::runtime_error("HDF5 object creation/open failed");
    }
  }
  ~Hdf5Handle() {
    if(identifier_ >= 0 && closer_ != nullptr) {
      static_cast<void>(closer_(identifier_));
    }
  }
  Hdf5Handle(const Hdf5Handle&) = delete;
  Hdf5Handle& operator=(const Hdf5Handle&) = delete;
  Hdf5Handle(Hdf5Handle&& other) noexcept
      : identifier_{std::exchange(other.identifier_, -1)},
        closer_{std::exchange(other.closer_, nullptr)} {}
  Hdf5Handle& operator=(Hdf5Handle&& other) noexcept {
    if(this != &other) {
      if(identifier_ >= 0 && closer_ != nullptr) {
        static_cast<void>(closer_(identifier_));
      }
      identifier_ = std::exchange(other.identifier_, -1);
      closer_ = std::exchange(other.closer_, nullptr);
    }
    return *this;
  }
  [[nodiscard]] hid_t get() const noexcept { return identifier_; }

 private:
  hid_t identifier_{-1};
  Closer closer_{nullptr};
};

void check_hdf5(herr_t status, const char* operation) {
  if(status < 0) {
    throw std::runtime_error(std::string{"HDF5 failure while "} + operation);
  }
}

[[nodiscard]] Hdf5Handle scalar_space() {
  return Hdf5Handle{H5Screate(H5S_SCALAR), H5Sclose};
}

[[nodiscard]] Hdf5Handle vector_space(std::size_t size) {
  if(size == 0) {
    throw std::invalid_argument("HDF5 vector datasets may not be empty");
  }
  const hsize_t dimension = static_cast<hsize_t>(size);
  return Hdf5Handle{H5Screate_simple(1, &dimension, nullptr), H5Sclose};
}

[[nodiscard]] Hdf5Handle matrix_space(std::size_t rows,
                                      std::size_t columns) {
  if(rows == 0 || columns == 0) {
    throw std::invalid_argument("HDF5 matrix datasets may not be empty");
  }
  const std::array<hsize_t, 2> dimensions{
      static_cast<hsize_t>(rows), static_cast<hsize_t>(columns)};
  return Hdf5Handle{
      H5Screate_simple(2, dimensions.data(), nullptr), H5Sclose};
}

void create_group(hid_t file, const char* path) {
  Hdf5Handle group{
      H5Gcreate2(file, path, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
      H5Gclose};
}

template <typename T>
[[nodiscard]] hid_t native_hdf5_type();

template <>
hid_t native_hdf5_type<int>() {
  return H5T_NATIVE_INT;
}

template <>
hid_t native_hdf5_type<std::uint64_t>() {
  return H5T_NATIVE_UINT64;
}

template <>
hid_t native_hdf5_type<std::uint8_t>() {
  return H5T_NATIVE_UINT8;
}

template <>
hid_t native_hdf5_type<double>() {
  return H5T_NATIVE_DOUBLE;
}

[[nodiscard]] std::size_t checked_size_from_u64(
    std::uint64_t value, std::string_view field) {
  if(value >
     static_cast<std::uint64_t>(
         std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error(
        "checkpoint value exceeds size_t for " + std::string{field});
  }
  return static_cast<std::size_t>(value);
}

template <typename T>
void write_scalar(hid_t file, const char* path, hid_t type, const T& value) {
  Hdf5Handle space = scalar_space();
  Hdf5Handle dataset{
      H5Dcreate2(file, path, type, space.get(), H5P_DEFAULT, H5P_DEFAULT,
                 H5P_DEFAULT),
      H5Dclose};
  check_hdf5(
      H5Dwrite(dataset.get(), native_hdf5_type<T>(), H5S_ALL, H5S_ALL,
               H5P_DEFAULT, &value),
      "writing scalar dataset");
}

template <typename T>
void write_vector(hid_t file, const char* path, hid_t type,
                  const std::vector<T>& values) {
  Hdf5Handle space = vector_space(values.size());
  Hdf5Handle dataset{
      H5Dcreate2(file, path, type, space.get(), H5P_DEFAULT, H5P_DEFAULT,
                 H5P_DEFAULT),
      H5Dclose};
  check_hdf5(H5Dwrite(dataset.get(), native_hdf5_type<T>(), H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, values.data()),
             "writing vector dataset");
}

[[nodiscard]] Hdf5Handle fixed_string_type(std::size_t size) {
  Hdf5Handle type{H5Tcopy(H5T_C_S1), H5Tclose};
  check_hdf5(H5Tset_size(type.get(), size), "setting string size");
  check_hdf5(H5Tset_strpad(type.get(), H5T_STR_NULLTERM),
             "setting string padding");
  check_hdf5(H5Tset_cset(type.get(), H5T_CSET_UTF8),
             "setting string encoding");
  return type;
}

void write_string(hid_t file, const char* path, const std::string& value) {
  const std::size_t stored_size = value.size() + 1;
  Hdf5Handle type = fixed_string_type(stored_size);
  Hdf5Handle space = scalar_space();
  Hdf5Handle dataset{
      H5Dcreate2(file, path, type.get(), space.get(), H5P_DEFAULT,
                 H5P_DEFAULT, H5P_DEFAULT),
      H5Dclose};
  check_hdf5(H5Dwrite(dataset.get(), type.get(), H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, value.c_str()),
             "writing string dataset");
}

void write_string_attribute(hid_t object, const char* name,
                            const std::string& value) {
  Hdf5Handle type = fixed_string_type(value.size() + 1);
  Hdf5Handle space = scalar_space();
  Hdf5Handle attribute{
      H5Acreate2(object, name, type.get(), space.get(), H5P_DEFAULT,
                 H5P_DEFAULT),
      H5Aclose};
  check_hdf5(H5Awrite(attribute.get(), type.get(), value.c_str()),
             "writing string attribute");
}

void write_matrix(hid_t file, const char* path, const linalg::Matrix& matrix,
                  const char* row_index, const char* column_index) {
  linalg::require_finite(matrix, path);
  Hdf5Handle space = matrix_space(matrix.rows(), matrix.columns());
  Hdf5Handle dataset{
      H5Dcreate2(file, path, H5T_IEEE_F64LE, space.get(), H5P_DEFAULT,
                 H5P_DEFAULT, H5P_DEFAULT),
      H5Dclose};
  check_hdf5(H5Dwrite(dataset.get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, matrix.data()),
             "writing matrix dataset");
  write_string_attribute(dataset.get(), "storage_order", "column-major");
  write_string_attribute(dataset.get(), "scalar_type", "IEEE-754 binary64");
  write_string_attribute(
      dataset.get(), "dimensions",
      std::to_string(matrix.rows()) + "x" +
          std::to_string(matrix.columns()));
  write_string_attribute(dataset.get(), "row_index", row_index);
  write_string_attribute(dataset.get(), "column_index", column_index);
  write_string_attribute(dataset.get(), "index_base", "zero");
}

[[nodiscard]] std::string basis_hash(const basis::BasisSet& basis) {
  std::uint64_t hash = UINT64_C(1469598103934665603);
  const auto mix_byte = [&hash](unsigned char byte) {
    hash ^= static_cast<std::uint64_t>(byte);
    hash *= UINT64_C(1099511628211);
  };
  for(const char byte : basis.canonical_name) {
    mix_byte(static_cast<unsigned char>(byte));
  }
  for(const basis::Shell& shell : basis.shells) {
    for(const std::uint64_t value :
        {static_cast<std::uint64_t>(shell.atom_index),
         static_cast<std::uint64_t>(shell.first_ao),
         static_cast<std::uint64_t>(shell.function_count),
         static_cast<std::uint64_t>(shell.angular_momentum)}) {
      for(unsigned int shift = 0; shift < 64; shift += 8) {
        mix_byte(static_cast<unsigned char>((value >> shift) & UINT64_C(0xff)));
      }
    }
    for(const double value : shell.exponents) {
      const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
      for(unsigned int shift = 0; shift < 64; shift += 8) {
        mix_byte(
            static_cast<unsigned char>((bits >> shift) & UINT64_C(0xff)));
      }
    }
    for(const double value : shell.contraction_coefficients) {
      const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
      for(unsigned int shift = 0; shift < 64; shift += 8) {
        mix_byte(
            static_cast<unsigned char>((bits >> shift) & UINT64_C(0xff)));
      }
    }
  }
  std::ostringstream output;
  output << std::hex << std::setfill('0') << std::setw(16) << hash;
  return output.str();
}

void write_symbols(hid_t file, const molecule::Molecule& molecule) {
  constexpr std::size_t width = 3;
  std::vector<char> buffer(molecule.atoms().size() * width, '\0');
  for(std::size_t atom = 0; atom < molecule.atoms().size(); ++atom) {
    const std::string& symbol = molecule.atoms()[atom].symbol;
    if(symbol.size() >= width) {
      throw std::runtime_error("element symbol exceeds checkpoint width");
    }
    for(std::size_t index = 0; index < symbol.size(); ++index) {
      buffer[atom * width + index] = symbol[index];
    }
  }
  Hdf5Handle type = fixed_string_type(width);
  Hdf5Handle space = vector_space(molecule.atoms().size());
  Hdf5Handle dataset{
      H5Dcreate2(file, "/molecule/symbols", type.get(), space.get(),
                 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
      H5Dclose};
  check_hdf5(H5Dwrite(dataset.get(), type.get(), H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, buffer.data()),
             "writing element symbols");
}

void write_localization_space(
    hid_t file, const char* group_path,
    const localization::PipekMezeyResult& localized) {
  create_group(file, group_path);
  const std::string prefix{group_path};
  write_scalar(file, (prefix + "/completed").c_str(), H5T_STD_I32LE, 1);
  write_matrix(file, (prefix + "/rotation").c_str(), localized.rotation,
               "canonical orbital in space", "localized orbital");
  write_matrix(file, (prefix + "/coefficients").c_str(),
               localized.coefficients, "AO", "localized orbital");
  write_matrix(file, (prefix + "/lowdin_populations").c_str(),
               localized.lowdin_populations, "localized orbital", "atom");
  std::vector<std::uint64_t> atom_assignment;
  std::vector<int> monomer_assignment;
  std::vector<double> atom_population;
  std::vector<double> second_atom_population;
  std::vector<double> atom_population_margin;
  std::vector<int> atom_ambiguous;
  std::vector<double> monomer_population;
  std::vector<double> second_monomer_population;
  std::vector<double> monomer_population_margin;
  std::vector<int> monomer_ambiguous;
  atom_assignment.reserve(localized.assignments.size());
  monomer_assignment.reserve(localized.assignments.size());
  atom_population.reserve(localized.assignments.size());
  second_atom_population.reserve(localized.assignments.size());
  atom_population_margin.reserve(localized.assignments.size());
  atom_ambiguous.reserve(localized.assignments.size());
  monomer_population.reserve(localized.assignments.size());
  second_monomer_population.reserve(localized.assignments.size());
  monomer_population_margin.reserve(localized.assignments.size());
  monomer_ambiguous.reserve(localized.assignments.size());
  for(const localization::OrbitalAssignment& assignment :
      localized.assignments) {
    atom_assignment.push_back(
        static_cast<std::uint64_t>(assignment.atom_index));
    monomer_assignment.push_back(assignment.monomer_id);
    atom_population.push_back(assignment.atom_population);
    second_atom_population.push_back(assignment.second_atom_population);
    atom_population_margin.push_back(assignment.atom_population_margin);
    atom_ambiguous.push_back(assignment.atom_ambiguous ? 1 : 0);
    monomer_population.push_back(assignment.monomer_population);
    second_monomer_population.push_back(
        assignment.second_monomer_population);
    monomer_population_margin.push_back(
        assignment.monomer_population_margin);
    monomer_ambiguous.push_back(assignment.monomer_ambiguous ? 1 : 0);
  }
  write_vector(file, (prefix + "/atom_assignment").c_str(),
               H5T_STD_U64LE, atom_assignment);
  write_vector(file, (prefix + "/monomer_assignment").c_str(),
               H5T_STD_I32LE, monomer_assignment);
  write_vector(file, (prefix + "/atom_population").c_str(),
               H5T_IEEE_F64LE, atom_population);
  write_vector(file, (prefix + "/second_atom_population").c_str(),
               H5T_IEEE_F64LE, second_atom_population);
  write_vector(file, (prefix + "/atom_population_margin").c_str(),
               H5T_IEEE_F64LE, atom_population_margin);
  write_vector(file, (prefix + "/atom_ambiguous").c_str(),
               H5T_STD_I32LE, atom_ambiguous);
  write_vector(file, (prefix + "/monomer_population").c_str(),
               H5T_IEEE_F64LE, monomer_population);
  write_vector(file, (prefix + "/second_monomer_population").c_str(),
               H5T_IEEE_F64LE, second_monomer_population);
  write_vector(file, (prefix + "/monomer_population_margin").c_str(),
               H5T_IEEE_F64LE, monomer_population_margin);
  write_vector(file, (prefix + "/monomer_ambiguous").c_str(),
               H5T_STD_I32LE, monomer_ambiguous);
  write_vector(file, (prefix + "/objective_history").c_str(),
               H5T_IEEE_F64LE, localized.objective_history);
  write_scalar(file, (prefix + "/completed_sweeps").c_str(),
               H5T_STD_I32LE, localized.completed_sweeps);
  write_scalar(file, (prefix + "/final_sweep_gain").c_str(),
               H5T_IEEE_F64LE, localized.final_sweep_gain);
  write_scalar(file, (prefix + "/population_method").c_str(),
               H5T_STD_I32LE,
               static_cast<int>(localized.population_method));
  write_scalar(file, (prefix + "/optimizer").c_str(), H5T_STD_I32LE,
               static_cast<int>(localized.optimizer));
  write_scalar(file, (prefix + "/final_gradient_norm").c_str(),
               H5T_IEEE_F64LE, localized.final_gradient_norm);
  write_scalar(file, (prefix + "/total_keyframes").c_str(),
               H5T_STD_I32LE, localized.total_keyframes);
  write_scalar(file, (prefix + "/total_hessian_actions").c_str(),
               H5T_STD_I32LE, localized.total_hessian_actions);
  write_scalar(
      file, (prefix + "/rotation_orthogonality_maximum_error").c_str(),
      H5T_IEEE_F64LE, localized.rotation_orthogonality_maximum_error);
  write_scalar(
      file, (prefix + "/orbital_orthonormality_maximum_error").c_str(),
      H5T_IEEE_F64LE, localized.orbital_orthonormality_maximum_error);
  write_scalar(file, (prefix + "/projector_maximum_error").c_str(),
               H5T_IEEE_F64LE, localized.projector_maximum_error);
}

void write_laplace_fit(hid_t file, const mp2::LaplaceFitResult& fit) {
  create_group(file, "/laplace");
  create_group(file, "/mp2");
  write_scalar(file, "/laplace/denominator_min", H5T_IEEE_F64LE,
               fit.histogram.denominator_minimum);
  write_scalar(file, "/laplace/denominator_max", H5T_IEEE_F64LE,
               fit.histogram.denominator_maximum);
  write_scalar(
      file, "/laplace/total_ordered_denominators", H5T_STD_U64LE,
      fit.histogram.total_ordered_denominators);
  write_vector(file, "/laplace/histogram_edges", H5T_IEEE_F64LE,
               fit.histogram.edges);
  write_vector(file, "/laplace/histogram_midpoints", H5T_IEEE_F64LE,
               fit.histogram.midpoints);
  write_vector(file, "/laplace/histogram_counts", H5T_STD_U64LE,
               fit.histogram.counts);
  write_vector(file, "/laplace/histogram_weights", H5T_IEEE_F64LE,
               fit.histogram.frequencies);
  write_vector(file, "/laplace/nodes", H5T_IEEE_F64LE, fit.nodes);
  write_vector(file, "/laplace/weights", H5T_IEEE_F64LE, fit.weights);
  write_vector(file, "/laplace/condition_numbers", H5T_IEEE_F64LE,
               fit.condition_number_history);
  write_vector(file, "/laplace/optimization_history", H5T_IEEE_F64LE,
               fit.optimization_history);
  write_scalar(
      file, "/laplace/optimizer_iterations", H5T_STD_U64LE,
      static_cast<std::uint64_t>(fit.optimizer_iterations));
  write_scalar(
      file, "/laplace/weight_numerical_rank", H5T_STD_U64LE,
      static_cast<std::uint64_t>(fit.weight_numerical_rank));
  write_scalar(file, "/laplace/fit_objective", H5T_IEEE_F64LE,
               fit.weighted_objective);
  write_scalar(file, "/laplace/weighted_rmse", H5T_IEEE_F64LE,
               fit.weighted_rmse);
  write_scalar(file, "/laplace/histogram_maximum_absolute_error",
               H5T_IEEE_F64LE,
               fit.histogram_maximum_absolute_error);
  write_scalar(file, "/laplace/histogram_maximum_relative_error",
               H5T_IEEE_F64LE,
               fit.histogram_maximum_relative_error);
  write_scalar(file, "/laplace/validation_maximum_absolute_error",
               H5T_IEEE_F64LE,
               fit.validation_maximum_absolute_error);
  write_scalar(file, "/laplace/validation_maximum_relative_error",
               H5T_IEEE_F64LE,
               fit.validation_maximum_relative_error);
}

void write_laplace_mp2(
    hid_t file, const mp2::LaplaceFitResult& fit,
    const mp2::IncoreLaplaceMp2Result& energy) {
  write_laplace_fit(file, fit);
  write_string(file, "/mp2/algorithm", "phase10-in-core-reference");
  write_scalar(file, "/mp2/canonical_reference_energy", H5T_IEEE_F64LE,
               energy.canonical_reference_energy);
  write_scalar(file, "/mp2/laplace_energy", H5T_IEEE_F64LE,
               energy.canonical_laplace_energy);
  write_scalar(file, "/mp2/localized_laplace_energy", H5T_IEEE_F64LE,
               energy.localized_laplace_energy);
  write_scalar(file, "/mp2/localized_energy_computed", H5T_STD_I32LE,
               energy.localized_energy_computed ? 1 : 0);
  write_scalar(file, "/mp2/quadrature_error", H5T_IEEE_F64LE,
               energy.quadrature_error);
  write_scalar(file, "/mp2/localization_error", H5T_IEEE_F64LE,
               energy.localization_error);
  // This guarded reference evaluates exact unscreened AO ERIs.
  write_scalar(file, "/mp2/screening_error", H5T_IEEE_F64LE, 0.0);
  write_scalar(file, "/mp2/minimum_denominator", H5T_IEEE_F64LE,
               energy.minimum_denominator);
  write_scalar(file, "/mp2/maximum_denominator", H5T_IEEE_F64LE,
               energy.maximum_denominator);
  write_scalar(file, "/mp2/minimum_half_scaling_exponent",
               H5T_IEEE_F64LE,
               energy.minimum_half_scaling_exponent);
  write_scalar(file, "/mp2/maximum_half_scaling_exponent",
               H5T_IEEE_F64LE,
               energy.maximum_half_scaling_exponent);
  write_scalar(file, "/mp2/maximum_half_scaling_factor_error",
               H5T_IEEE_F64LE,
               energy.maximum_half_scaling_factor_error);
  write_scalar(
      file, "/mp2/estimated_peak_additional_bytes", H5T_STD_U64LE,
      static_cast<std::uint64_t>(
          energy.estimated_peak_additional_bytes));
  write_scalar(
      file, "/mp2/quadrature_points", H5T_STD_U64LE,
      static_cast<std::uint64_t>(energy.quadrature_points));
}

void write_tiled_laplace_mp2(
    hid_t file, const mp2::LaplaceFitResult& fit,
    const mp2::TiledLaplaceMp2Result& energy) {
  write_laplace_fit(file, fit);
  write_string(file, "/mp2/algorithm",
               "phase11-direct-tiled-laplace-mp2");
  write_scalar(file, "/mp2/laplace_energy", H5T_IEEE_F64LE,
               energy.correlation_energy);
  write_scalar(file, "/mp2/minimum_denominator", H5T_IEEE_F64LE,
               energy.minimum_denominator);
  write_scalar(file, "/mp2/maximum_denominator", H5T_IEEE_F64LE,
               energy.maximum_denominator);
  write_scalar(file, "/mp2/minimum_scaling_exponent",
               H5T_IEEE_F64LE, energy.minimum_scaling_exponent);
  write_scalar(file, "/mp2/maximum_scaling_exponent",
               H5T_IEEE_F64LE, energy.maximum_scaling_exponent);
  write_scalar(
      file, "/mp2/quadrature_points", H5T_STD_U64LE,
      static_cast<std::uint64_t>(energy.quadrature_points));
  write_scalar(
      file, "/mp2/total_tiles", H5T_STD_U64LE,
      static_cast<std::uint64_t>(energy.total_tiles));
  write_scalar(
      file, "/mp2/initially_completed_tiles", H5T_STD_U64LE,
      static_cast<std::uint64_t>(energy.initially_completed_tiles));
  write_scalar(file, "/mp2/minimum_tiles_per_rank", H5T_STD_U64LE,
               energy.minimum_tiles_per_rank);
  write_scalar(file, "/mp2/maximum_tiles_per_rank", H5T_STD_U64LE,
               energy.maximum_tiles_per_rank);
  write_scalar(file, "/mp2/direct_transform_count", H5T_STD_U64LE,
               energy.direct_transform_count);
  write_scalar(file, "/mp2/exchange_transform_count", H5T_STD_U64LE,
               energy.exchange_transform_count);
  write_scalar(file, "/mp2/reused_exchange_tile_count",
               H5T_STD_U64LE, energy.reused_exchange_tile_count);
  write_scalar(file, "/mp2/ordered_shell_quartets_per_transform",
               H5T_STD_U64LE,
               energy.ordered_shell_quartets_per_transform);
  write_scalar(file, "/mp2/evaluated_shell_quartets",
               H5T_STD_U64LE, energy.evaluated_shell_quartets);
  write_scalar(file, "/mp2/schwarz_screened_shell_quartets",
               H5T_STD_U64LE,
               energy.schwarz_screened_shell_quartets);
  write_scalar(file, "/mp2/eri_schwarz_threshold",
               H5T_IEEE_F64LE, energy.eri_schwarz_threshold);
  write_scalar(
      file, "/mp2/estimated_peak_additional_bytes", H5T_STD_U64LE,
      static_cast<std::uint64_t>(
          energy.estimated_peak_additional_bytes));
  write_scalar(file, "/mp2/calculation_fingerprint", H5T_STD_U64LE,
               energy.fingerprint);
  write_vector(file, "/mp2/completed_tiles", H5T_STD_U8LE,
               energy.progress.completed);
  write_vector(file, "/mp2/tile_energy_contributions_hartree",
               H5T_IEEE_F64LE,
               energy.progress.tile_energy_contributions);
}

void write_checkpoint_content(hid_t file, const molecule::Molecule& molecule,
                              const basis::BasisSet& basis,
                              const scf::RhfResult& result, bool frozen_core,
                              CheckpointRuntime runtime,
                              const localization::PipekMezeyResult*
                                  occupied_localization,
                              const localization::PipekMezeyResult*
                                  virtual_localization,
                              const mp2::LaplaceFitResult* laplace_fit,
                              const mp2::IncoreLaplaceMp2Result*
                                  laplace_mp2,
                              const mp2::TiledLaplaceMp2Result*
                                  tiled_laplace_mp2) {
  create_group(file, "/metadata");
  create_group(file, "/molecule");
  create_group(file, "/basis");
  create_group(file, "/rhf");
  create_group(file, "/orbitals");
  create_group(file, "/orbitals/canonical");

  const core::BuildInfo build = core::build_info();
  write_scalar(file, "/metadata/format_version", H5T_STD_I32LE, 2);
  write_scalar(file, "/metadata/write_complete", H5T_STD_I32LE, 0);
  write_string(file, "/metadata/code_version", build.version);
  write_string(file, "/metadata/git_commit", "unrecorded");
  write_string(file, "/metadata/build_type", build.build_type);
  write_string(file, "/metadata/compiler", build.compiler);
  write_scalar(file, "/metadata/mpi_size", H5T_STD_I32LE, runtime.mpi_size);
  write_scalar(file, "/metadata/openmp_threads", H5T_STD_I32LE,
               runtime.openmp_threads);
  write_string(file, "/metadata/floating_point_type", "IEEE-754 binary64");

  write_symbols(file, molecule);
  std::vector<int> atomic_numbers;
  std::vector<double> coordinates;
  atomic_numbers.reserve(molecule.atoms().size());
  coordinates.reserve(molecule.atoms().size() * 3);
  for(const molecule::Atom& atom : molecule.atoms()) {
    atomic_numbers.push_back(atom.atomic_number);
    coordinates.insert(coordinates.end(), atom.position_bohr.begin(),
                       atom.position_bohr.end());
  }
  write_vector(file, "/molecule/atomic_numbers", H5T_STD_I32LE,
               atomic_numbers);
  {
    Hdf5Handle space = matrix_space(molecule.atoms().size(), 3);
    Hdf5Handle dataset{
        H5Dcreate2(file, "/molecule/coordinates_bohr", H5T_IEEE_F64LE,
                   space.get(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
        H5Dclose};
    check_hdf5(H5Dwrite(dataset.get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                        H5P_DEFAULT, coordinates.data()),
               "writing molecular coordinates");
    write_string_attribute(dataset.get(), "storage_order", "row-major");
    write_string_attribute(dataset.get(), "scalar_type",
                           "IEEE-754 binary64");
    write_string_attribute(
        dataset.get(), "dimensions",
        std::to_string(molecule.atoms().size()) + "x3");
    write_string_attribute(dataset.get(), "row_index", "atom");
    write_string_attribute(dataset.get(), "column_index", "Cartesian axis");
    write_string_attribute(dataset.get(), "index_base", "zero");
  }
  write_scalar(file, "/molecule/charge", H5T_STD_I32LE, molecule.charge());
  write_scalar(file, "/molecule/multiplicity", H5T_STD_I32LE,
               molecule.multiplicity());
  write_scalar(file, "/molecule/nuclear_repulsion_energy", H5T_IEEE_F64LE,
               result.nuclear_repulsion_energy);
  write_vector(file, "/molecule/monomer_ids", H5T_STD_I32LE,
               molecule.monomer_ids());

  write_string(file, "/basis/canonical_name", basis.canonical_name);
  write_scalar(file, "/basis/spherical", H5T_STD_I32LE,
               basis.spherical ? 1 : 0);
  write_scalar(file, "/basis/number_of_shells", H5T_STD_U64LE,
               static_cast<std::uint64_t>(basis.shells.size()));
  write_scalar(file, "/basis/number_of_aos", H5T_STD_U64LE,
               static_cast<std::uint64_t>(basis.number_of_aos));
  std::vector<std::uint64_t> shell_to_atom;
  std::vector<int> angular_momenta;
  std::vector<double> exponents;
  std::vector<double> coefficients;
  std::vector<std::uint64_t> exponent_offsets{0};
  std::vector<std::uint64_t> coefficient_offsets{0};
  for(const basis::Shell& shell : basis.shells) {
    shell_to_atom.push_back(static_cast<std::uint64_t>(shell.atom_index));
    angular_momenta.push_back(shell.angular_momentum);
    exponents.insert(exponents.end(), shell.exponents.begin(),
                     shell.exponents.end());
    coefficients.insert(coefficients.end(),
                        shell.contraction_coefficients.begin(),
                        shell.contraction_coefficients.end());
    exponent_offsets.push_back(static_cast<std::uint64_t>(exponents.size()));
    coefficient_offsets.push_back(
        static_cast<std::uint64_t>(coefficients.size()));
  }
  std::vector<std::uint64_t> ao_to_shell;
  ao_to_shell.reserve(basis.ao_to_shell.size());
  for(const std::size_t value : basis.ao_to_shell) {
    ao_to_shell.push_back(static_cast<std::uint64_t>(value));
  }
  write_vector(file, "/basis/shell_to_atom", H5T_STD_U64LE, shell_to_atom);
  write_vector(file, "/basis/ao_to_shell", H5T_STD_U64LE, ao_to_shell);
  write_vector(file, "/basis/angular_momenta", H5T_STD_I32LE,
               angular_momenta);
  write_vector(file, "/basis/exponents", H5T_IEEE_F64LE, exponents);
  write_vector(file, "/basis/exponent_offsets", H5T_STD_U64LE,
               exponent_offsets);
  write_vector(file, "/basis/contraction_coefficients", H5T_IEEE_F64LE,
               coefficients);
  write_vector(file, "/basis/contraction_offsets", H5T_STD_U64LE,
               coefficient_offsets);
  write_string(file, "/basis/ao_ordering",
               "Libint2 standard spherical shell order");
  write_string(file, "/basis/basis_hash", basis_hash(basis));

  write_scalar(file, "/rhf/converged", H5T_STD_I32LE,
               result.converged ? 1 : 0);
  write_scalar(file, "/rhf/total_energy", H5T_IEEE_F64LE,
               result.total_energy);
  write_scalar(file, "/rhf/electronic_energy", H5T_IEEE_F64LE,
               result.electronic_energy);
  write_scalar(file, "/rhf/iteration_count", H5T_STD_I32LE,
               result.iteration_count);
  write_scalar(file, "/rhf/final_energy_change", H5T_IEEE_F64LE,
               result.final_energy_change);
  write_scalar(file, "/rhf/final_density_rms", H5T_IEEE_F64LE,
               result.final_density_rms);
  write_scalar(file, "/rhf/final_commutator_rms", H5T_IEEE_F64LE,
               result.final_commutator_rms);
  write_scalar(file, "/rhf/unique_shell_quartets_per_fock", H5T_STD_U64LE,
               result.unique_shell_quartets_per_fock);
  write_scalar(file, "/rhf/evaluated_shell_quartets_total", H5T_STD_U64LE,
               result.evaluated_shell_quartets_total);
  write_scalar(file, "/rhf/screened_shell_quartets_total", H5T_STD_U64LE,
               result.screened_shell_quartets_total);
  write_scalar(file, "/rhf/schwarz_screened_shell_quartets_total",
               H5T_STD_U64LE,
               result.schwarz_screened_shell_quartets_total);
  write_scalar(file, "/rhf/density_screened_shell_quartets_total",
               H5T_STD_U64LE,
               result.density_screened_shell_quartets_total);
  write_scalar(file, "/rhf/eri_schwarz_threshold", H5T_IEEE_F64LE,
               result.eri_schwarz_threshold);
  write_scalar(file, "/rhf/density_screen", H5T_STD_I32LE,
               result.density_screen ? 1 : 0);
  write_scalar(file, "/rhf/fock_contribution_threshold", H5T_IEEE_F64LE,
               result.fock_contribution_threshold);
  write_scalar(file, "/rhf/final_screened_fock_maximum_error_bound",
               H5T_IEEE_F64LE,
               result.final_screened_fock_maximum_error_bound);
  write_scalar(file, "/rhf/fock_threads", H5T_STD_U64LE,
               static_cast<std::uint64_t>(result.fock_threads));
  write_scalar(file, "/rhf/minimum_owned_shell_quartets_per_fock",
               H5T_STD_U64LE,
               result.minimum_owned_shell_quartets_per_fock);
  write_scalar(file, "/rhf/maximum_owned_shell_quartets_per_fock",
               H5T_STD_U64LE,
               result.maximum_owned_shell_quartets_per_fock);
  write_scalar(file, "/rhf/direct_fock_wall_seconds_total",
               H5T_IEEE_F64LE, result.direct_fock_wall_seconds_total);
  write_scalar(file, "/rhf/minimum_direct_fock_wall_seconds_last",
               H5T_IEEE_F64LE,
               result.minimum_direct_fock_wall_seconds_last);
  write_scalar(file, "/rhf/maximum_direct_fock_wall_seconds_last",
               H5T_IEEE_F64LE,
               result.maximum_direct_fock_wall_seconds_last);
  write_matrix(file, "/rhf/density", result.density, "AO", "AO");
  write_matrix(file, "/rhf/fock", result.fock, "AO", "AO");
  write_matrix(file, "/rhf/core_hamiltonian", result.core_hamiltonian, "AO",
               "AO");
  write_matrix(file, "/rhf/overlap", result.overlap, "AO", "AO");
  write_matrix(file, "/rhf/orthogonalizer", result.orthogonalizer, "AO",
               "orthonormal AO");

  write_matrix(file, "/orbitals/canonical/coefficients",
               result.coefficients, "AO", "canonical MO");
  write_vector(file, "/orbitals/canonical/energies", H5T_IEEE_F64LE,
               result.orbital_energies);
  write_vector(file, "/orbitals/canonical/occupations", H5T_IEEE_F64LE,
               result.occupations);
  const std::size_t occupied = molecule.occupied_orbitals_rhf();
  std::size_t frozen_count = 0;
  if(frozen_core) {
    for(const molecule::Atom& atom : molecule.atoms()) {
      frozen_count += atom.atomic_number == 8 ? 1U : 0U;
    }
    frozen_count = std::min(frozen_count, occupied);
  }
  std::vector<int> frozen_mask(result.orbital_energies.size(), 0);
  for(std::size_t index = 0; index < frozen_count; ++index) {
    frozen_mask[index] = 1;
  }
  std::vector<std::uint64_t> active_occupied;
  for(std::size_t index = frozen_count; index < occupied; ++index) {
    active_occupied.push_back(static_cast<std::uint64_t>(index));
  }
  std::vector<std::uint64_t> virtual_indices;
  for(std::size_t index = occupied; index < result.orbital_energies.size();
      ++index) {
    virtual_indices.push_back(static_cast<std::uint64_t>(index));
  }
  write_vector(file, "/orbitals/canonical/frozen_core_mask",
               H5T_STD_I32LE, frozen_mask);
  if(active_occupied.empty() || virtual_indices.empty()) {
    throw std::runtime_error(
        "checkpoint orbital partition unexpectedly contains an empty space");
  }
  write_vector(file, "/orbitals/canonical/active_occupied_indices",
               H5T_STD_U64LE, active_occupied);
  write_vector(file, "/orbitals/canonical/virtual_indices", H5T_STD_U64LE,
               virtual_indices);
  if((occupied_localization == nullptr) !=
     (virtual_localization == nullptr)) {
    throw std::invalid_argument(
        "checkpoint requires both occupied and virtual localization records");
  }
  if(occupied_localization != nullptr) {
    create_group(file, "/localization");
    write_localization_space(file, "/localization/occupied",
                             *occupied_localization);
    write_localization_space(file, "/localization/virtual",
                             *virtual_localization);
  }
  const bool has_mp2 =
      laplace_mp2 != nullptr || tiled_laplace_mp2 != nullptr;
  if((laplace_fit == nullptr) != !has_mp2 ||
     (laplace_mp2 != nullptr && tiled_laplace_mp2 != nullptr)) {
    throw std::invalid_argument(
        "checkpoint requires one Laplace energy record with its fit");
  }
  if(laplace_mp2 != nullptr) {
    if(laplace_mp2->localized_energy_computed &&
       occupied_localization == nullptr) {
      throw std::invalid_argument(
          "localized Laplace MP2 checkpoint requires localization records");
    }
    write_laplace_mp2(file, *laplace_fit, *laplace_mp2);
  } else if(tiled_laplace_mp2 != nullptr) {
    write_tiled_laplace_mp2(
        file, *laplace_fit, *tiled_laplace_mp2);
  }
}

template <typename T>
[[nodiscard]] T read_scalar(hid_t file, const char* path, hid_t type) {
  Hdf5Handle dataset{H5Dopen2(file, path, H5P_DEFAULT), H5Dclose};
  Hdf5Handle space{H5Dget_space(dataset.get()), H5Sclose};
  if(H5Sget_simple_extent_ndims(space.get()) != 0) {
    throw std::runtime_error(std::string{path} + " is not scalar");
  }
  T result{};
  check_hdf5(H5Dread(dataset.get(), type, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     &result),
             "reading scalar dataset");
  return result;
}

[[nodiscard]] std::string read_string(hid_t file, const char* path) {
  Hdf5Handle dataset{H5Dopen2(file, path, H5P_DEFAULT), H5Dclose};
  Hdf5Handle type{H5Dget_type(dataset.get()), H5Tclose};
  const std::size_t size = H5Tget_size(type.get());
  if(size == 0) {
    throw std::runtime_error(std::string{path} + " has an invalid string type");
  }
  std::vector<char> buffer(size, '\0');
  check_hdf5(H5Dread(dataset.get(), type.get(), H5S_ALL, H5S_ALL,
                     H5P_DEFAULT, buffer.data()),
             "reading string dataset");
  return std::string{buffer.data()};
}

[[nodiscard]] std::string read_string_attribute(hid_t object,
                                                const char* name) {
  Hdf5Handle attribute{H5Aopen(object, name, H5P_DEFAULT), H5Aclose};
  Hdf5Handle type{H5Aget_type(attribute.get()), H5Tclose};
  const std::size_t size = H5Tget_size(type.get());
  if(size == 0) {
    throw std::runtime_error(std::string{"attribute "} + name +
                             " has an invalid string type");
  }
  std::vector<char> buffer(size, '\0');
  check_hdf5(H5Aread(attribute.get(), type.get(), buffer.data()),
             "reading string attribute");
  return std::string{buffer.data()};
}

template <typename T>
[[nodiscard]] std::vector<T> read_vector(hid_t file, const char* path,
                                         hid_t type) {
  Hdf5Handle dataset{H5Dopen2(file, path, H5P_DEFAULT), H5Dclose};
  Hdf5Handle space{H5Dget_space(dataset.get()), H5Sclose};
  if(H5Sget_simple_extent_ndims(space.get()) != 1) {
    throw std::runtime_error(std::string{path} + " is not a vector");
  }
  hsize_t dimension = 0;
  check_hdf5(H5Sget_simple_extent_dims(space.get(), &dimension, nullptr),
             "reading vector dimensions");
  if(dimension >
     static_cast<hsize_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::overflow_error("HDF5 vector dimension exceeds size_t");
  }
  std::vector<T> result(static_cast<std::size_t>(dimension));
  check_hdf5(H5Dread(dataset.get(), type, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     result.data()),
             "reading vector dataset");
  return result;
}

[[nodiscard]] linalg::Matrix read_matrix(hid_t file, const char* path) {
  Hdf5Handle dataset{H5Dopen2(file, path, H5P_DEFAULT), H5Dclose};
  Hdf5Handle type{H5Dget_type(dataset.get()), H5Tclose};
  if(H5Tget_class(type.get()) != H5T_FLOAT ||
     H5Tget_size(type.get()) != sizeof(double)) {
    throw std::runtime_error(std::string{path} +
                             " is not stored as binary64 floating point");
  }
  Hdf5Handle space{H5Dget_space(dataset.get()), H5Sclose};
  if(H5Sget_simple_extent_ndims(space.get()) != 2) {
    throw std::runtime_error(std::string{path} + " is not a matrix");
  }
  std::array<hsize_t, 2> dimensions{};
  check_hdf5(
      H5Sget_simple_extent_dims(space.get(), dimensions.data(), nullptr),
      "reading matrix dimensions");
  const bool coordinates =
      std::string_view{path} == "/molecule/coordinates_bohr";
  const std::string expected_storage =
      coordinates ? "row-major" : "column-major";
  if(read_string_attribute(dataset.get(), "storage_order") !=
         expected_storage ||
     read_string_attribute(dataset.get(), "scalar_type") !=
         "IEEE-754 binary64") {
    throw std::runtime_error(std::string{path} +
                             " matrix metadata is incompatible");
  }
  const std::string expected_dimensions =
      std::to_string(dimensions[0]) + "x" + std::to_string(dimensions[1]);
  if(read_string_attribute(dataset.get(), "dimensions") !=
     expected_dimensions) {
    throw std::runtime_error(std::string{path} +
                             " dimensions attribute is inconsistent");
  }
  linalg::Matrix result{static_cast<std::size_t>(dimensions[0]),
                        static_cast<std::size_t>(dimensions[1])};
  check_hdf5(H5Dread(dataset.get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                     H5P_DEFAULT, result.data()),
             "reading matrix dataset");
  linalg::require_finite(result, path);
  return result;
}

void validate_checkpoint(hid_t file, bool require_complete) {
  if(read_scalar<int>(file, "/metadata/format_version",
                      H5T_NATIVE_INT) != 2) {
    throw std::runtime_error("unsupported checkpoint format version");
  }
  const int complete =
      read_scalar<int>(file, "/metadata/write_complete", H5T_NATIVE_INT);
  if(require_complete && complete != 1) {
    throw std::runtime_error("checkpoint is incomplete");
  }
  if(read_scalar<int>(file, "/rhf/converged", H5T_NATIVE_INT) != 1) {
    throw std::runtime_error("checkpoint RHF state is not converged");
  }
  static_cast<void>(
      read_scalar<double>(file, "/rhf/total_energy", H5T_NATIVE_DOUBLE));
  static_cast<void>(read_string(file, "/basis/canonical_name"));
  static_cast<void>(read_matrix(file, "/rhf/density"));
  static_cast<void>(read_matrix(file, "/orbitals/canonical/coefficients"));
}

void validate_compatible_system(hid_t file,
                                const molecule::Molecule& expected_molecule,
                                const basis::BasisSet& expected_basis) {
  if(read_string(file, "/basis/canonical_name") !=
     expected_basis.canonical_name) {
    throw std::runtime_error("checkpoint basis name does not match");
  }
  if(read_scalar<int>(file, "/basis/spherical", H5T_NATIVE_INT) !=
     (expected_basis.spherical ? 1 : 0)) {
    throw std::runtime_error(
        "checkpoint spherical basis convention does not match");
  }
  if(read_scalar<std::uint64_t>(file, "/basis/number_of_shells",
                                H5T_NATIVE_UINT64) !=
         static_cast<std::uint64_t>(expected_basis.shells.size()) ||
     read_scalar<std::uint64_t>(file, "/basis/number_of_aos",
                                H5T_NATIVE_UINT64) !=
         static_cast<std::uint64_t>(expected_basis.number_of_aos)) {
    throw std::runtime_error(
        "checkpoint shell or AO count does not match");
  }
  if(read_string(file, "/basis/basis_hash") != basis_hash(expected_basis)) {
    throw std::runtime_error("checkpoint basis hash does not match");
  }
  if(read_string(file, "/basis/ao_ordering") !=
     "Libint2 standard spherical shell order") {
    throw std::runtime_error("checkpoint AO ordering is unsupported");
  }
  if(read_scalar<int>(file, "/molecule/charge", H5T_NATIVE_INT) !=
         expected_molecule.charge() ||
     read_scalar<int>(file, "/molecule/multiplicity", H5T_NATIVE_INT) !=
         expected_molecule.multiplicity()) {
    throw std::runtime_error(
        "checkpoint charge or multiplicity does not match");
  }
  const std::vector<int> atomic_numbers =
      read_vector<int>(file, "/molecule/atomic_numbers", H5T_NATIVE_INT);
  if(atomic_numbers.size() != expected_molecule.atoms().size()) {
    throw std::runtime_error("checkpoint atom count does not match");
  }
  for(std::size_t atom = 0; atom < atomic_numbers.size(); ++atom) {
    if(atomic_numbers[atom] != expected_molecule.atoms()[atom].atomic_number) {
      throw std::runtime_error("checkpoint atomic numbers do not match");
    }
  }
  const linalg::Matrix coordinates =
      read_matrix(file, "/molecule/coordinates_bohr");
  if(coordinates.rows() != expected_molecule.atoms().size() ||
     coordinates.columns() != 3) {
    throw std::runtime_error(
        "checkpoint coordinate dimensions do not match");
  }
  // coordinates_bohr is explicitly row-major, unlike matrix datasets.
  for(std::size_t atom = 0; atom < expected_molecule.atoms().size(); ++atom) {
    for(std::size_t axis = 0; axis < 3; ++axis) {
      const double stored = coordinates.values()[atom * 3 + axis];
      if(stored != expected_molecule.atoms()[atom].position_bohr[axis]) {
        throw std::runtime_error(
            "checkpoint coordinates do not match exactly");
      }
    }
  }
}

void require_square_symmetric(const linalg::Matrix& matrix,
                              std::size_t dimension, const char* name) {
  if(matrix.rows() != dimension || matrix.columns() != dimension) {
    throw std::runtime_error(std::string{"checkpoint "} + name +
                             " dimensions are incompatible");
  }
  if(linalg::maximum_asymmetry(matrix) > 1.0e-12) {
    throw std::runtime_error(std::string{"checkpoint "} + name +
                             " exceeds the symmetry tolerance");
  }
}

[[nodiscard]] std::vector<std::size_t> checked_indices(
    const std::vector<std::uint64_t>& stored, std::size_t orbital_count,
    const char* name) {
  std::vector<std::size_t> result;
  result.reserve(stored.size());
  std::set<std::size_t> unique;
  for(const std::uint64_t value : stored) {
    if(value >= static_cast<std::uint64_t>(orbital_count) ||
       value > static_cast<std::uint64_t>(
                   std::numeric_limits<std::size_t>::max())) {
      throw std::runtime_error(std::string{"checkpoint "} + name +
                               " contains an out-of-range index");
    }
    const std::size_t index = static_cast<std::size_t>(value);
    if(!unique.insert(index).second) {
      throw std::runtime_error(std::string{"checkpoint "} + name +
                               " contains a duplicate index");
    }
    result.push_back(index);
  }
  return result;
}

[[nodiscard]] double canonical_orthonormality_error(
    const linalg::Matrix& coefficients, const linalg::Matrix& overlap) {
  const linalg::Matrix metric =
      linalg::multiply(linalg::transpose(coefficients),
                       linalg::multiply(overlap, coefficients));
  return linalg::maximum_absolute_value(
      linalg::subtract(metric, linalg::Matrix::identity(metric.rows())));
}

[[nodiscard]] std::filesystem::path temporary_sibling(
    const std::filesystem::path& destination) {
  static std::atomic<unsigned long long> sequence{0};
  const auto ticks =
      std::chrono::steady_clock::now().time_since_epoch().count();
  return destination.string() + ".tmp." + std::to_string(ticks) + "." +
         std::to_string(sequence.fetch_add(1));
}

}  // namespace

void write_rhf_checkpoint(const std::filesystem::path& path,
                          const molecule::Molecule& molecule,
                          const basis::BasisSet& basis,
                          const scf::RhfResult& result, bool frozen_core,
                          CheckpointRuntime runtime, bool overwrite,
                          const localization::PipekMezeyResult*
                              occupied_localization,
                          const localization::PipekMezeyResult*
                              virtual_localization,
                          const mp2::LaplaceFitResult* laplace_fit,
                          const mp2::IncoreLaplaceMp2Result*
                              laplace_mp2,
                          const mp2::TiledLaplaceMp2Result*
                              tiled_laplace_mp2) {
  if(!result.converged) {
    throw std::invalid_argument(
        "refusing to checkpoint an unconverged RHF result");
  }
  if(runtime.mpi_size <= 0 || runtime.openmp_threads <= 0) {
    throw std::invalid_argument("checkpoint runtime topology is invalid");
  }
  const std::filesystem::path destination =
      std::filesystem::absolute(path).lexically_normal();
  if(std::filesystem::exists(destination) && !overwrite) {
    throw std::runtime_error("checkpoint already exists and overwrite is false: " +
                             destination.string());
  }
  if(!destination.parent_path().empty() &&
     !std::filesystem::exists(destination.parent_path())) {
    throw std::runtime_error("checkpoint parent directory does not exist: " +
                             destination.parent_path().string());
  }
  const std::filesystem::path temporary = temporary_sibling(destination);
  try {
    {
      Hdf5Handle file{
          H5Fcreate(temporary.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT),
          H5Fclose};
      write_checkpoint_content(file.get(), molecule, basis, result, frozen_core,
                               runtime, occupied_localization,
                               virtual_localization, laplace_fit,
                               laplace_mp2, tiled_laplace_mp2);
      check_hdf5(H5Fflush(file.get(), H5F_SCOPE_GLOBAL),
                 "flushing incomplete checkpoint");
    }
    {
      Hdf5Handle file{
          H5Fopen(temporary.c_str(), H5F_ACC_RDWR, H5P_DEFAULT), H5Fclose};
      validate_checkpoint(file.get(), false);
      Hdf5Handle complete{
          H5Dopen2(file.get(), "/metadata/write_complete", H5P_DEFAULT),
          H5Dclose};
      const int value = 1;
      check_hdf5(H5Dwrite(complete.get(), H5T_NATIVE_INT, H5S_ALL, H5S_ALL,
                          H5P_DEFAULT, &value),
                 "marking checkpoint complete");
      check_hdf5(H5Fflush(file.get(), H5F_SCOPE_GLOBAL),
                 "flushing completed checkpoint");
    }
    if(occupied_localization == nullptr) {
      static_cast<void>(
          read_canonical_rhf_checkpoint(temporary, molecule, basis));
    } else {
      static_cast<void>(
          read_localized_rhf_checkpoint(temporary, molecule, basis));
    }
    if(laplace_fit != nullptr) {
      if(laplace_mp2 != nullptr) {
        static_cast<void>(read_laplace_mp2_checkpoint(temporary));
      } else {
        Hdf5Handle file{
            H5Fopen(temporary.c_str(), H5F_ACC_RDONLY,
                    H5P_DEFAULT),
            H5Fclose};
        const double stored_energy = read_scalar<double>(
            file.get(), "/mp2/laplace_energy", H5T_NATIVE_DOUBLE);
        const std::uint64_t stored_fingerprint =
            read_scalar<std::uint64_t>(
                file.get(), "/mp2/calculation_fingerprint",
                H5T_NATIVE_UINT64);
        const std::vector<std::uint8_t> completed =
            read_vector<std::uint8_t>(
                file.get(), "/mp2/completed_tiles",
                H5T_NATIVE_UINT8);
        if(tiled_laplace_mp2 == nullptr ||
           stored_energy !=
               tiled_laplace_mp2->correlation_energy ||
           stored_fingerprint != tiled_laplace_mp2->fingerprint ||
           completed != tiled_laplace_mp2->progress.completed) {
          throw std::runtime_error(
              "tiled Laplace MP2 checkpoint read-back validation failed");
        }
      }
    }
    if(std::filesystem::exists(destination) && !overwrite) {
      throw std::runtime_error(
          "checkpoint appeared during write and overwrite is false: " +
          destination.string());
    }
    std::filesystem::rename(temporary, destination);
  } catch(...) {
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    throw;
  }
}

LaplaceMp2Checkpoint read_laplace_mp2_checkpoint(
    const std::filesystem::path& path) {
  const std::filesystem::path source =
      std::filesystem::absolute(path).lexically_normal();
  Hdf5Handle file{
      H5Fopen(source.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose};
  validate_checkpoint(file.get(), true);

  const std::uint64_t optimizer_iterations_u64 =
      read_scalar<std::uint64_t>(
          file.get(), "/laplace/optimizer_iterations",
          H5T_NATIVE_UINT64);
  const std::uint64_t weight_numerical_rank_u64 =
      read_scalar<std::uint64_t>(
          file.get(), "/laplace/weight_numerical_rank",
          H5T_NATIVE_UINT64);
  const std::uint64_t estimated_peak_additional_bytes_u64 =
      read_scalar<std::uint64_t>(
          file.get(), "/mp2/estimated_peak_additional_bytes",
          H5T_NATIVE_UINT64);
  const std::uint64_t quadrature_points_u64 =
      read_scalar<std::uint64_t>(
          file.get(), "/mp2/quadrature_points", H5T_NATIVE_UINT64);
  const int localized_energy_flag =
      read_scalar<int>(
          file.get(), "/mp2/localized_energy_computed",
          H5T_NATIVE_INT);
  if(localized_energy_flag != 0 && localized_energy_flag != 1) {
    throw std::runtime_error(
        "checkpoint localized-energy flag is not zero or one");
  }

  mp2::DenominatorHistogram histogram{
      .denominator_minimum =
          read_scalar<double>(file.get(), "/laplace/denominator_min",
                              H5T_NATIVE_DOUBLE),
      .denominator_maximum =
          read_scalar<double>(file.get(), "/laplace/denominator_max",
                              H5T_NATIVE_DOUBLE),
      .total_ordered_denominators =
          read_scalar<std::uint64_t>(
              file.get(), "/laplace/total_ordered_denominators",
              H5T_NATIVE_UINT64),
      .edges = read_vector<double>(
          file.get(), "/laplace/histogram_edges", H5T_NATIVE_DOUBLE),
      .midpoints = read_vector<double>(
          file.get(), "/laplace/histogram_midpoints", H5T_NATIVE_DOUBLE),
      .counts = read_vector<std::uint64_t>(
          file.get(), "/laplace/histogram_counts", H5T_NATIVE_UINT64),
      .frequencies = read_vector<double>(
          file.get(), "/laplace/histogram_weights", H5T_NATIVE_DOUBLE),
  };
  mp2::LaplaceFitResult fit{
      .histogram = std::move(histogram),
      .nodes = read_vector<double>(
          file.get(), "/laplace/nodes", H5T_NATIVE_DOUBLE),
      .weights = read_vector<double>(
          file.get(), "/laplace/weights", H5T_NATIVE_DOUBLE),
      .optimization_history = read_vector<double>(
          file.get(), "/laplace/optimization_history", H5T_NATIVE_DOUBLE),
      .condition_number_history = read_vector<double>(
          file.get(), "/laplace/condition_numbers", H5T_NATIVE_DOUBLE),
      .optimizer_iterations = checked_size_from_u64(
          optimizer_iterations_u64, "Laplace optimizer iterations"),
      .weight_numerical_rank = checked_size_from_u64(
          weight_numerical_rank_u64, "Laplace numerical rank"),
      .weighted_objective =
          read_scalar<double>(file.get(), "/laplace/fit_objective",
                              H5T_NATIVE_DOUBLE),
      .weighted_rmse =
          read_scalar<double>(file.get(), "/laplace/weighted_rmse",
                              H5T_NATIVE_DOUBLE),
      .histogram_maximum_absolute_error =
          read_scalar<double>(
              file.get(),
              "/laplace/histogram_maximum_absolute_error",
              H5T_NATIVE_DOUBLE),
      .histogram_maximum_relative_error =
          read_scalar<double>(
              file.get(),
              "/laplace/histogram_maximum_relative_error",
              H5T_NATIVE_DOUBLE),
      .validation_maximum_absolute_error =
          read_scalar<double>(
              file.get(),
              "/laplace/validation_maximum_absolute_error",
              H5T_NATIVE_DOUBLE),
      .validation_maximum_relative_error =
          read_scalar<double>(
              file.get(),
              "/laplace/validation_maximum_relative_error",
              H5T_NATIVE_DOUBLE),
  };
  mp2::IncoreLaplaceMp2Result energy{
      .canonical_reference_energy =
          read_scalar<double>(
              file.get(), "/mp2/canonical_reference_energy",
              H5T_NATIVE_DOUBLE),
      .canonical_laplace_energy =
          read_scalar<double>(file.get(), "/mp2/laplace_energy",
                              H5T_NATIVE_DOUBLE),
      .localized_laplace_energy =
          read_scalar<double>(
              file.get(), "/mp2/localized_laplace_energy",
              H5T_NATIVE_DOUBLE),
      .localized_energy_computed = localized_energy_flag == 1,
      .quadrature_error =
          read_scalar<double>(file.get(), "/mp2/quadrature_error",
                              H5T_NATIVE_DOUBLE),
      .localization_error =
          read_scalar<double>(file.get(), "/mp2/localization_error",
                              H5T_NATIVE_DOUBLE),
      .minimum_denominator =
          read_scalar<double>(file.get(), "/mp2/minimum_denominator",
                              H5T_NATIVE_DOUBLE),
      .maximum_denominator =
          read_scalar<double>(file.get(), "/mp2/maximum_denominator",
                              H5T_NATIVE_DOUBLE),
      .minimum_half_scaling_exponent =
          read_scalar<double>(
              file.get(), "/mp2/minimum_half_scaling_exponent",
              H5T_NATIVE_DOUBLE),
      .maximum_half_scaling_exponent =
          read_scalar<double>(
              file.get(), "/mp2/maximum_half_scaling_exponent",
              H5T_NATIVE_DOUBLE),
      .maximum_half_scaling_factor_error =
          read_scalar<double>(
              file.get(), "/mp2/maximum_half_scaling_factor_error",
              H5T_NATIVE_DOUBLE),
      .estimated_peak_additional_bytes = checked_size_from_u64(
          estimated_peak_additional_bytes_u64,
          "MP2 estimated peak additional bytes"),
      .quadrature_points = checked_size_from_u64(
          quadrature_points_u64, "MP2 quadrature points"),
  };

  const std::size_t bins = fit.histogram.midpoints.size();
  if(bins == 0 || fit.histogram.edges.size() != bins + 1 ||
     fit.histogram.counts.size() != bins ||
     fit.histogram.frequencies.size() != bins ||
     fit.nodes.empty() || fit.nodes.size() != fit.weights.size() ||
     fit.optimization_history.empty() ||
     fit.optimization_history.size() !=
         fit.condition_number_history.size() ||
     fit.weight_numerical_rank == 0 ||
     fit.weight_numerical_rank > fit.nodes.size() ||
     energy.quadrature_points != fit.nodes.size()) {
    throw std::runtime_error(
        "checkpoint Laplace dimensions or ranks are inconsistent");
  }
  std::uint64_t count_sum = 0;
  double frequency_sum = 0.0;
  if(fit.histogram.total_ordered_denominators == 0) {
    throw std::runtime_error(
        "checkpoint Laplace histogram is empty");
  }
  for(std::size_t bin = 0; bin < bins; ++bin) {
    const double lower = fit.histogram.edges[bin];
    const double upper = fit.histogram.edges[bin + 1];
    const double midpoint = fit.histogram.midpoints[bin];
    const double frequency = fit.histogram.frequencies[bin];
    if(!std::isfinite(lower) || !std::isfinite(upper) ||
       !std::isfinite(midpoint) || !std::isfinite(frequency) ||
       lower <= 0.0 || upper <= lower || midpoint <= lower ||
       midpoint >= upper || frequency < 0.0) {
      throw std::runtime_error(
          "checkpoint Laplace histogram geometry is invalid");
    }
    if(count_sum >
       std::numeric_limits<std::uint64_t>::max() -
           fit.histogram.counts[bin]) {
      throw std::runtime_error(
          "checkpoint Laplace histogram count overflows");
    }
    count_sum += fit.histogram.counts[bin];
    frequency_sum += frequency;
    const double expected_frequency =
        static_cast<double>(fit.histogram.counts[bin]) /
        static_cast<double>(
            fit.histogram.total_ordered_denominators);
    if(std::abs(frequency - expected_frequency) >
       16.0 * std::numeric_limits<double>::epsilon()) {
      throw std::runtime_error(
          "checkpoint Laplace histogram frequency is inconsistent "
          "with its count");
    }
  }
  if(count_sum != fit.histogram.total_ordered_denominators ||
     std::abs(frequency_sum - 1.0) >
         64.0 * std::numeric_limits<double>::epsilon() *
             static_cast<double>(bins) ||
     !std::is_sorted(fit.nodes.begin(), fit.nodes.end())) {
    throw std::runtime_error(
        "checkpoint Laplace histogram or node ordering is invalid");
  }
  for(const double node : fit.nodes) {
    if(!std::isfinite(node) || node <= 0.0) {
      throw std::runtime_error(
          "checkpoint Laplace node is invalid");
    }
  }
  for(const double weight : fit.weights) {
    if(!std::isfinite(weight)) {
      throw std::runtime_error(
          "checkpoint Laplace weight is nonfinite");
    }
  }
  for(std::size_t step = 0; step < fit.optimization_history.size();
      ++step) {
    if(!std::isfinite(fit.optimization_history[step]) ||
       fit.optimization_history[step] < 0.0 ||
       (step != 0 &&
        fit.optimization_history[step] >
            fit.optimization_history[step - 1] +
                32.0 * std::numeric_limits<double>::epsilon())) {
      throw std::runtime_error(
          "checkpoint Laplace optimization history is invalid");
    }
    const double condition = fit.condition_number_history[step];
    if(std::isnan(condition) || condition <= 0.0) {
      throw std::runtime_error(
          "checkpoint Laplace condition number is invalid");
    }
  }
  for(const double value :
      {fit.histogram.denominator_minimum,
       fit.histogram.denominator_maximum, fit.weighted_objective,
       fit.weighted_rmse, fit.histogram_maximum_absolute_error,
       fit.histogram_maximum_relative_error,
       fit.validation_maximum_absolute_error,
       fit.validation_maximum_relative_error,
       energy.canonical_reference_energy,
       energy.canonical_laplace_energy,
       energy.localized_laplace_energy, energy.quadrature_error,
       energy.localization_error, energy.minimum_denominator,
       energy.maximum_denominator,
       energy.minimum_half_scaling_exponent,
       energy.maximum_half_scaling_exponent,
       energy.maximum_half_scaling_factor_error}) {
    if(!std::isfinite(value)) {
      throw std::runtime_error(
          "checkpoint Laplace/MP2 scalar is nonfinite");
    }
  }
  const double energy_tolerance =
      64.0 * std::numeric_limits<double>::epsilon();
  const double fit_tolerance =
      128.0 * std::numeric_limits<double>::epsilon();
  if(std::abs(
         energy.canonical_laplace_energy -
         energy.canonical_reference_energy -
         energy.quadrature_error) > energy_tolerance ||
     std::abs(
         energy.localized_laplace_energy -
         energy.canonical_laplace_energy -
         energy.localization_error) > energy_tolerance ||
     energy.minimum_denominator <= 0.0 ||
     energy.maximum_denominator < energy.minimum_denominator ||
     std::abs(energy.minimum_denominator -
              fit.histogram.denominator_minimum) > fit_tolerance ||
     std::abs(energy.maximum_denominator -
              fit.histogram.denominator_maximum) > fit_tolerance ||
     fit.histogram.denominator_minimum !=
         fit.histogram.edges.front() ||
     fit.histogram.denominator_maximum !=
         fit.histogram.edges.back() ||
     fit.weighted_objective < 0.0 || fit.weighted_rmse < 0.0 ||
     std::abs(
         fit.weighted_rmse * fit.weighted_rmse -
         fit.weighted_objective) >
         fit_tolerance *
             std::max(1.0, fit.weighted_objective) ||
     std::abs(fit.optimization_history.back() -
              fit.weighted_objective) >
         fit_tolerance *
             std::max(1.0, fit.weighted_objective) ||
     (!energy.localized_energy_computed &&
      (energy.localized_laplace_energy !=
           energy.canonical_laplace_energy ||
       energy.localization_error != 0.0)) ||
     energy.maximum_half_scaling_exponent > 0.0 ||
     energy.minimum_half_scaling_exponent >
         energy.maximum_half_scaling_exponent ||
     energy.maximum_half_scaling_factor_error < 0.0 ||
     read_scalar<double>(file.get(), "/mp2/screening_error",
                         H5T_NATIVE_DOUBLE) != 0.0) {
    throw std::runtime_error(
        "checkpoint Laplace/MP2 physical invariants are invalid");
  }
  return LaplaceMp2Checkpoint{
      .fit = std::move(fit),
      .energy = energy,
  };
}

linalg::Matrix read_rhf_restart_density(
    const std::filesystem::path& path,
    const molecule::Molecule& expected_molecule,
    const basis::BasisSet& expected_basis) {
  const std::filesystem::path source =
      std::filesystem::absolute(path).lexically_normal();
  Hdf5Handle file{
      H5Fopen(source.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose};
  validate_checkpoint(file.get(), true);
  validate_compatible_system(file.get(), expected_molecule, expected_basis);
  linalg::Matrix density = read_matrix(file.get(), "/rhf/density");
  require_square_symmetric(density, expected_basis.number_of_aos, "density");
  return density;
}

CanonicalRhfCheckpoint read_canonical_rhf_checkpoint(
    const std::filesystem::path& path,
    const molecule::Molecule& expected_molecule,
    const basis::BasisSet& expected_basis) {
  const std::filesystem::path source =
      std::filesystem::absolute(path).lexically_normal();
  Hdf5Handle file{
      H5Fopen(source.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose};
  validate_checkpoint(file.get(), true);
  validate_compatible_system(file.get(), expected_molecule, expected_basis);

  const std::size_t ao_count = expected_basis.number_of_aos;
  CanonicalRhfCheckpoint result{
      .total_energy =
          read_scalar<double>(file.get(), "/rhf/total_energy",
                              H5T_NATIVE_DOUBLE),
      .electronic_energy =
          read_scalar<double>(file.get(), "/rhf/electronic_energy",
                              H5T_NATIVE_DOUBLE),
      .density = read_matrix(file.get(), "/rhf/density"),
      .fock = read_matrix(file.get(), "/rhf/fock"),
      .core_hamiltonian =
          read_matrix(file.get(), "/rhf/core_hamiltonian"),
      .overlap = read_matrix(file.get(), "/rhf/overlap"),
      .orthogonalizer = read_matrix(file.get(), "/rhf/orthogonalizer"),
      .coefficients =
          read_matrix(file.get(), "/orbitals/canonical/coefficients"),
      .orbital_energies =
          read_vector<double>(file.get(), "/orbitals/canonical/energies",
                              H5T_NATIVE_DOUBLE),
      .occupations =
          read_vector<double>(file.get(), "/orbitals/canonical/occupations",
                              H5T_NATIVE_DOUBLE),
      .frozen_core_mask =
          read_vector<int>(file.get(), "/orbitals/canonical/frozen_core_mask",
                           H5T_NATIVE_INT),
      .active_occupied_indices = {},
      .virtual_indices = {},
      .orbital_orthonormality_maximum_error = 0.0,
  };
  if(!std::isfinite(result.total_energy) ||
     !std::isfinite(result.electronic_energy)) {
    throw std::runtime_error("checkpoint RHF energies are nonfinite");
  }
  require_square_symmetric(result.density, ao_count, "density");
  require_square_symmetric(result.fock, ao_count, "Fock matrix");
  require_square_symmetric(result.core_hamiltonian, ao_count,
                           "core Hamiltonian");
  require_square_symmetric(result.overlap, ao_count, "overlap");
  if(result.orthogonalizer.rows() != ao_count ||
     result.orthogonalizer.columns() == 0) {
    throw std::runtime_error(
        "checkpoint orthogonalizer dimensions are incompatible");
  }
  const std::size_t orbital_count = result.coefficients.columns();
  if(result.coefficients.rows() != ao_count || orbital_count == 0 ||
     result.orbital_energies.size() != orbital_count ||
     result.occupations.size() != orbital_count ||
     result.frozen_core_mask.size() != orbital_count) {
    throw std::runtime_error(
        "checkpoint canonical orbital dimensions are incompatible");
  }
  for(std::size_t orbital = 0; orbital < orbital_count; ++orbital) {
    if(!std::isfinite(result.orbital_energies[orbital]) ||
       !std::isfinite(result.occupations[orbital]) ||
       (result.occupations[orbital] != 0.0 &&
        result.occupations[orbital] != 2.0) ||
       (result.frozen_core_mask[orbital] != 0 &&
        result.frozen_core_mask[orbital] != 1)) {
      throw std::runtime_error(
          "checkpoint orbital energies, occupations, or frozen mask are invalid");
    }
  }
  const std::size_t expected_occupied =
      expected_molecule.occupied_orbitals_rhf();
  if(expected_occupied > orbital_count) {
    throw std::runtime_error(
        "checkpoint has fewer orbitals than occupied RHF orbitals");
  }
  result.active_occupied_indices = checked_indices(
      read_vector<std::uint64_t>(
          file.get(), "/orbitals/canonical/active_occupied_indices",
          H5T_NATIVE_UINT64),
      orbital_count, "active occupied partition");
  result.virtual_indices = checked_indices(
      read_vector<std::uint64_t>(file.get(),
                                 "/orbitals/canonical/virtual_indices",
                                 H5T_NATIVE_UINT64),
      orbital_count, "virtual partition");

  std::vector<int> partition_membership(orbital_count, 0);
  std::size_t frozen_count = 0;
  for(std::size_t orbital = 0; orbital < orbital_count; ++orbital) {
    const bool occupied = orbital < expected_occupied;
    if(result.occupations[orbital] != (occupied ? 2.0 : 0.0)) {
      throw std::runtime_error(
          "checkpoint occupations do not match the RHF electron count");
    }
    if(result.frozen_core_mask[orbital] == 1) {
      if(!occupied) {
        throw std::runtime_error(
            "checkpoint marks a virtual orbital as frozen core");
      }
      ++frozen_count;
      ++partition_membership[orbital];
    }
  }
  for(const std::size_t orbital : result.active_occupied_indices) {
    if(orbital >= expected_occupied) {
      throw std::runtime_error(
          "checkpoint active occupied partition contains a virtual orbital");
    }
    ++partition_membership[orbital];
  }
  for(const std::size_t orbital : result.virtual_indices) {
    if(orbital < expected_occupied) {
      throw std::runtime_error(
          "checkpoint virtual partition contains an occupied orbital");
    }
    ++partition_membership[orbital];
  }
  for(const int membership : partition_membership) {
    if(membership != 1) {
      throw std::runtime_error(
          "checkpoint orbital partitions are not disjoint and complete");
    }
  }
  if(frozen_count + result.active_occupied_indices.size() !=
         expected_occupied ||
     result.virtual_indices.size() != orbital_count - expected_occupied) {
    throw std::runtime_error(
        "checkpoint orbital partition sizes are inconsistent");
  }

  result.orbital_orthonormality_maximum_error =
      canonical_orthonormality_error(result.coefficients, result.overlap);
  constexpr double orthonormality_tolerance = 2.0e-10;
  if(!std::isfinite(result.orbital_orthonormality_maximum_error) ||
     result.orbital_orthonormality_maximum_error >
         orthonormality_tolerance) {
    throw std::runtime_error(
        "checkpoint canonical orbitals exceed the C^T S C orthonormality tolerance");
  }
  return result;
}

scf::RhfResult read_rhf_result_checkpoint(
    const std::filesystem::path& path,
    const molecule::Molecule& expected_molecule,
    const basis::BasisSet& expected_basis) {
  CanonicalRhfCheckpoint canonical = read_canonical_rhf_checkpoint(
      path, expected_molecule, expected_basis);
  const std::filesystem::path source =
      std::filesystem::absolute(path).lexically_normal();
  Hdf5Handle file{
      H5Fopen(source.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose};

  const int converged = read_scalar<int>(
      file.get(), "/rhf/converged", H5T_NATIVE_INT);
  const int density_screen = read_scalar<int>(
      file.get(), "/rhf/density_screen", H5T_NATIVE_INT);
  const int mpi_size = read_scalar<int>(
      file.get(), "/metadata/mpi_size", H5T_NATIVE_INT);
  const int iteration_count = read_scalar<int>(
      file.get(), "/rhf/iteration_count", H5T_NATIVE_INT);
  if(converged != 1 || (density_screen != 0 && density_screen != 1) ||
     mpi_size <= 0 || iteration_count < 0) {
    throw std::runtime_error(
        "checkpoint RHF convergence or runtime metadata is invalid");
  }
  const double nuclear_repulsion = read_scalar<double>(
      file.get(), "/molecule/nuclear_repulsion_energy",
      H5T_NATIVE_DOUBLE);
  const double final_energy_change = read_scalar<double>(
      file.get(), "/rhf/final_energy_change", H5T_NATIVE_DOUBLE);
  const double final_density_rms = read_scalar<double>(
      file.get(), "/rhf/final_density_rms", H5T_NATIVE_DOUBLE);
  const double final_commutator_rms = read_scalar<double>(
      file.get(), "/rhf/final_commutator_rms", H5T_NATIVE_DOUBLE);
  const double eri_schwarz_threshold = read_scalar<double>(
      file.get(), "/rhf/eri_schwarz_threshold", H5T_NATIVE_DOUBLE);
  const double fock_contribution_threshold = read_scalar<double>(
      file.get(), "/rhf/fock_contribution_threshold",
      H5T_NATIVE_DOUBLE);
  const double final_screened_bound = read_scalar<double>(
      file.get(), "/rhf/final_screened_fock_maximum_error_bound",
      H5T_NATIVE_DOUBLE);
  const double direct_fock_seconds = read_scalar<double>(
      file.get(), "/rhf/direct_fock_wall_seconds_total",
      H5T_NATIVE_DOUBLE);
  const double minimum_direct_fock_seconds = read_scalar<double>(
      file.get(), "/rhf/minimum_direct_fock_wall_seconds_last",
      H5T_NATIVE_DOUBLE);
  const double maximum_direct_fock_seconds = read_scalar<double>(
      file.get(), "/rhf/maximum_direct_fock_wall_seconds_last",
      H5T_NATIVE_DOUBLE);
  for(const double value :
      {nuclear_repulsion, final_energy_change, final_density_rms,
       final_commutator_rms, eri_schwarz_threshold,
       fock_contribution_threshold, final_screened_bound,
       direct_fock_seconds, minimum_direct_fock_seconds,
       maximum_direct_fock_seconds}) {
    if(!std::isfinite(value)) {
      throw std::runtime_error(
          "checkpoint RHF diagnostic scalar is nonfinite");
    }
  }
  if(final_density_rms < 0.0 || final_commutator_rms < 0.0 ||
     eri_schwarz_threshold < 0.0 || fock_contribution_threshold < 0.0 ||
     final_screened_bound < 0.0 || direct_fock_seconds < 0.0 ||
     minimum_direct_fock_seconds < 0.0 ||
     maximum_direct_fock_seconds < minimum_direct_fock_seconds ||
     std::abs(canonical.total_energy - canonical.electronic_energy -
              nuclear_repulsion) > 2.0e-10) {
    throw std::runtime_error(
        "checkpoint RHF energy or diagnostic invariants are invalid");
  }

  const std::size_t ao_count = expected_basis.number_of_aos;
  if(canonical.orthogonalizer.columns() > ao_count) {
    throw std::runtime_error(
        "checkpoint RHF orthogonalizer has too many columns");
  }
  const std::size_t removed_overlap_vectors =
      ao_count - canonical.orthogonalizer.columns();
  return scf::RhfResult{
      .converged = true,
      .iteration_count = iteration_count,
      .nuclear_repulsion_energy = nuclear_repulsion,
      .electronic_energy = canonical.electronic_energy,
      .total_energy = canonical.total_energy,
      .final_energy_change = final_energy_change,
      .final_density_rms = final_density_rms,
      .final_commutator_rms = final_commutator_rms,
      .overlap = std::move(canonical.overlap),
      .core_hamiltonian = std::move(canonical.core_hamiltonian),
      .orthogonalizer = std::move(canonical.orthogonalizer),
      .density = std::move(canonical.density),
      .fock = std::move(canonical.fock),
      .coefficients = std::move(canonical.coefficients),
      .orbital_energies = std::move(canonical.orbital_energies),
      .occupations = std::move(canonical.occupations),
      .removed_overlap_vectors = removed_overlap_vectors,
      .unique_shell_quartets_per_fock = read_scalar<std::uint64_t>(
          file.get(), "/rhf/unique_shell_quartets_per_fock",
          H5T_NATIVE_UINT64),
      .evaluated_shell_quartets_total = read_scalar<std::uint64_t>(
          file.get(), "/rhf/evaluated_shell_quartets_total",
          H5T_NATIVE_UINT64),
      .screened_shell_quartets_total = read_scalar<std::uint64_t>(
          file.get(), "/rhf/screened_shell_quartets_total",
          H5T_NATIVE_UINT64),
      .schwarz_screened_shell_quartets_total =
          read_scalar<std::uint64_t>(
              file.get(), "/rhf/schwarz_screened_shell_quartets_total",
              H5T_NATIVE_UINT64),
      .density_screened_shell_quartets_total =
          read_scalar<std::uint64_t>(
              file.get(), "/rhf/density_screened_shell_quartets_total",
              H5T_NATIVE_UINT64),
      .eri_schwarz_threshold = eri_schwarz_threshold,
      .density_screen = density_screen == 1,
      .fock_contribution_threshold = fock_contribution_threshold,
      .final_screened_fock_maximum_error_bound = final_screened_bound,
      .mpi_ranks = static_cast<std::size_t>(mpi_size),
      .fock_threads = checked_size_from_u64(
          read_scalar<std::uint64_t>(file.get(), "/rhf/fock_threads",
                                     H5T_NATIVE_UINT64),
          "RHF Fock threads"),
      .minimum_owned_shell_quartets_per_fock =
          read_scalar<std::uint64_t>(
              file.get(), "/rhf/minimum_owned_shell_quartets_per_fock",
              H5T_NATIVE_UINT64),
      .maximum_owned_shell_quartets_per_fock =
          read_scalar<std::uint64_t>(
              file.get(), "/rhf/maximum_owned_shell_quartets_per_fock",
              H5T_NATIVE_UINT64),
      .direct_fock_wall_seconds_total = direct_fock_seconds,
      .minimum_direct_fock_wall_seconds_last = minimum_direct_fock_seconds,
      .maximum_direct_fock_wall_seconds_last = maximum_direct_fock_seconds,
      .history = {},
  };
}

LocalizedRhfCheckpoint read_localized_rhf_checkpoint(
    const std::filesystem::path& path,
    const molecule::Molecule& expected_molecule,
    const basis::BasisSet& expected_basis) {
  CanonicalRhfCheckpoint canonical = read_canonical_rhf_checkpoint(
      path, expected_molecule, expected_basis);
  const std::filesystem::path source =
      std::filesystem::absolute(path).lexically_normal();
  Hdf5Handle file{
      H5Fopen(source.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose};

  const auto read_space =
      [&](const char* group_path,
          const std::vector<std::size_t>& canonical_indices) {
    const std::string prefix{group_path};
    if(read_scalar<int>(file.get(), (prefix + "/completed").c_str(),
                        H5T_NATIVE_INT) != 1) {
      throw std::runtime_error(
          "checkpoint localization space is not completed");
    }
    linalg::Matrix rotation =
        read_matrix(file.get(), (prefix + "/rotation").c_str());
    linalg::Matrix coefficients =
        read_matrix(file.get(), (prefix + "/coefficients").c_str());
    linalg::Matrix populations =
        read_matrix(file.get(), (prefix + "/lowdin_populations").c_str());
    const std::size_t space_size = canonical_indices.size();
    if(space_size == 0 || rotation.rows() != space_size ||
       rotation.columns() != space_size ||
       coefficients.rows() != expected_basis.number_of_aos ||
       coefficients.columns() != space_size ||
       populations.rows() != space_size ||
       populations.columns() != expected_molecule.atoms().size()) {
      throw std::runtime_error(
          "checkpoint localization dimensions are incompatible");
    }

    const std::vector<std::uint64_t> atom_indices =
        read_vector<std::uint64_t>(
            file.get(), (prefix + "/atom_assignment").c_str(),
            H5T_NATIVE_UINT64);
    const std::vector<int> monomer_indices =
        read_vector<int>(file.get(),
                         (prefix + "/monomer_assignment").c_str(),
                         H5T_NATIVE_INT);
    const std::vector<double> atom_population =
        read_vector<double>(file.get(),
                            (prefix + "/atom_population").c_str(),
                            H5T_NATIVE_DOUBLE);
    const std::vector<double> second_atom_population =
        read_vector<double>(
            file.get(), (prefix + "/second_atom_population").c_str(),
            H5T_NATIVE_DOUBLE);
    const std::vector<double> atom_margin =
        read_vector<double>(
            file.get(), (prefix + "/atom_population_margin").c_str(),
            H5T_NATIVE_DOUBLE);
    const std::vector<int> atom_ambiguous =
        read_vector<int>(file.get(),
                         (prefix + "/atom_ambiguous").c_str(),
                         H5T_NATIVE_INT);
    const std::vector<double> monomer_population =
        read_vector<double>(file.get(),
                            (prefix + "/monomer_population").c_str(),
                            H5T_NATIVE_DOUBLE);
    const std::vector<double> second_monomer_population =
        read_vector<double>(
            file.get(), (prefix + "/second_monomer_population").c_str(),
            H5T_NATIVE_DOUBLE);
    const std::vector<double> monomer_margin =
        read_vector<double>(
            file.get(), (prefix + "/monomer_population_margin").c_str(),
            H5T_NATIVE_DOUBLE);
    const std::vector<int> monomer_ambiguous =
        read_vector<int>(file.get(),
                         (prefix + "/monomer_ambiguous").c_str(),
                         H5T_NATIVE_INT);
    for(const std::size_t size :
        {atom_indices.size(), monomer_indices.size(),
         atom_population.size(), second_atom_population.size(),
         atom_margin.size(), atom_ambiguous.size(),
         monomer_population.size(), second_monomer_population.size(),
         monomer_margin.size(), monomer_ambiguous.size()}) {
      if(size != space_size) {
        throw std::runtime_error(
            "checkpoint localization assignment dimensions are incompatible");
      }
    }

    std::vector<localization::OrbitalAssignment> assignments;
    assignments.reserve(space_size);
    for(std::size_t orbital = 0; orbital < space_size; ++orbital) {
      if(atom_indices[orbital] >= expected_molecule.atoms().size() ||
         (atom_ambiguous[orbital] != 0 &&
          atom_ambiguous[orbital] != 1) ||
         (monomer_ambiguous[orbital] != 0 &&
          monomer_ambiguous[orbital] != 1)) {
        throw std::runtime_error(
            "checkpoint localization assignment metadata is invalid");
      }
      for(const double value :
          {atom_population[orbital], second_atom_population[orbital],
           atom_margin[orbital], monomer_population[orbital],
           second_monomer_population[orbital],
           monomer_margin[orbital]}) {
        if(!std::isfinite(value)) {
          throw std::runtime_error(
              "checkpoint localization assignment population is nonfinite");
        }
      }
      if(std::abs(atom_population[orbital] -
                  populations(
                      orbital,
                      static_cast<std::size_t>(atom_indices[orbital]))) >
             2.0e-10 ||
         std::abs(atom_population[orbital] -
                      second_atom_population[orbital] -
                      atom_margin[orbital]) >
             2.0e-10 ||
         std::abs(monomer_population[orbital] -
                      second_monomer_population[orbital] -
                      monomer_margin[orbital]) >
             2.0e-10) {
        throw std::runtime_error(
            "checkpoint localization assignment populations are inconsistent");
      }
      assignments.push_back(localization::OrbitalAssignment{
          .atom_index = static_cast<std::size_t>(atom_indices[orbital]),
          .atom_population = atom_population[orbital],
          .second_atom_population = second_atom_population[orbital],
          .atom_population_margin = atom_margin[orbital],
          .atom_ambiguous = atom_ambiguous[orbital] == 1,
          .monomer_id = monomer_indices[orbital],
          .monomer_population = monomer_population[orbital],
          .second_monomer_population =
              second_monomer_population[orbital],
          .monomer_population_margin = monomer_margin[orbital],
          .monomer_ambiguous = monomer_ambiguous[orbital] == 1,
      });
    }

    std::vector<double> history =
        read_vector<double>(file.get(),
                            (prefix + "/objective_history").c_str(),
                            H5T_NATIVE_DOUBLE);
    if(history.empty()) {
      throw std::runtime_error(
          "checkpoint localization objective history is empty");
    }
    for(std::size_t step = 0; step < history.size(); ++step) {
      // CIAH keyframe acceptance can produce a small nonmonotonic macro
      // history; finiteness and the final optimizer diagnostics are the
      // checkpoint invariants, not strict Jacobi-style monotonicity.
      if(!std::isfinite(history[step]) || history[step] < 0.0) {
        throw std::runtime_error(
            "checkpoint localization objective history is invalid");
      }
    }
    const int completed_sweeps =
        read_scalar<int>(file.get(), (prefix + "/completed_sweeps").c_str(),
                         H5T_NATIVE_INT);
    const double final_sweep_gain =
        read_scalar<double>(file.get(), (prefix + "/final_sweep_gain").c_str(),
                            H5T_NATIVE_DOUBLE);
    const int population_method = read_scalar<int>(
        file.get(), (prefix + "/population_method").c_str(),
        H5T_NATIVE_INT);
    const int optimizer = read_scalar<int>(
        file.get(), (prefix + "/optimizer").c_str(), H5T_NATIVE_INT);
    const double final_gradient_norm = read_scalar<double>(
        file.get(), (prefix + "/final_gradient_norm").c_str(),
        H5T_NATIVE_DOUBLE);
    const int total_keyframes = read_scalar<int>(
        file.get(), (prefix + "/total_keyframes").c_str(),
        H5T_NATIVE_INT);
    const int total_hessian_actions = read_scalar<int>(
        file.get(), (prefix + "/total_hessian_actions").c_str(),
        H5T_NATIVE_INT);
    if(completed_sweeps < 0 ||
       history.size() != static_cast<std::size_t>(completed_sweeps) + 1 ||
       !std::isfinite(final_sweep_gain) ||
       !std::isfinite(final_gradient_norm) || final_gradient_norm < 0.0 ||
       total_keyframes < 0 || total_hessian_actions < 0 ||
       (population_method != static_cast<int>(
                                  localization::PipekMezeyPopulationMethod::
                                      lowdin) &&
        population_method != static_cast<int>(
                                  localization::PipekMezeyPopulationMethod::
                                      meta_lowdin)) ||
       (optimizer != static_cast<int>(
                         localization::PipekMezeyOptimizer::jacobi) &&
        optimizer != static_cast<int>(
                         localization::PipekMezeyOptimizer::pyscf_ciah))) {
      throw std::runtime_error(
          "checkpoint localization convergence metadata is invalid");
    }

    linalg::Matrix canonical_space{canonical.coefficients.rows(), space_size};
    for(std::size_t column = 0; column < space_size; ++column) {
      for(std::size_t row = 0; row < canonical_space.rows(); ++row) {
        canonical_space(row, column) =
            canonical.coefficients(row, canonical_indices[column]);
      }
    }
    const double rotation_error = canonical_orthonormality_error(
        rotation, linalg::Matrix::identity(space_size));
    const double orbital_error =
        canonical_orthonormality_error(coefficients, canonical.overlap);
    const linalg::Matrix canonical_projector =
        linalg::multiply(canonical_space, linalg::transpose(canonical_space));
    const linalg::Matrix localized_projector =
        linalg::multiply(coefficients, linalg::transpose(coefficients));
    const double projector_error = linalg::maximum_absolute_value(
        linalg::subtract(canonical_projector, localized_projector));
    const double coefficient_error = linalg::maximum_absolute_value(
        linalg::subtract(coefficients,
                         linalg::multiply(canonical_space, rotation)));
    constexpr double invariant_tolerance = 2.0e-10;
    if(rotation_error > invariant_tolerance ||
       orbital_error > invariant_tolerance ||
       projector_error > invariant_tolerance ||
       coefficient_error > invariant_tolerance) {
      throw std::runtime_error(
          "checkpoint localization violates rotation, orbital, or projector invariants");
    }
    for(std::size_t orbital = 0; orbital < populations.rows(); ++orbital) {
      double sum = 0.0;
      for(std::size_t atom = 0; atom < populations.columns(); ++atom) {
        sum += populations(orbital, atom);
      }
      if(std::abs(sum - 1.0) > invariant_tolerance) {
        throw std::runtime_error(
            "checkpoint Löwdin populations do not sum to one");
      }
    }

    const double stored_rotation_error = read_scalar<double>(
        file.get(),
        (prefix + "/rotation_orthogonality_maximum_error").c_str(),
        H5T_NATIVE_DOUBLE);
    const double stored_orbital_error = read_scalar<double>(
        file.get(),
        (prefix + "/orbital_orthonormality_maximum_error").c_str(),
        H5T_NATIVE_DOUBLE);
    const double stored_projector_error = read_scalar<double>(
        file.get(), (prefix + "/projector_maximum_error").c_str(),
        H5T_NATIVE_DOUBLE);
    if(!std::isfinite(stored_rotation_error) ||
       !std::isfinite(stored_orbital_error) ||
       !std::isfinite(stored_projector_error) ||
       std::abs(stored_rotation_error - rotation_error) > 2.0e-10 ||
       std::abs(stored_orbital_error - orbital_error) > 2.0e-10 ||
       std::abs(stored_projector_error - projector_error) > 2.0e-10) {
      throw std::runtime_error(
          "checkpoint localization invariant metadata is inconsistent");
    }

    return localization::PipekMezeyResult{
        .coefficients = std::move(coefficients),
        .rotation = std::move(rotation),
        .lowdin_populations = std::move(populations),
        .assignments = std::move(assignments),
        .objective_history = std::move(history),
        .completed_sweeps = completed_sweeps,
        .final_sweep_gain = final_sweep_gain,
        .rotation_orthogonality_maximum_error = rotation_error,
        .orbital_orthonormality_maximum_error = orbital_error,
        .projector_maximum_error = projector_error,
        .population_method = static_cast<
            localization::PipekMezeyPopulationMethod>(population_method),
        .optimizer =
            static_cast<localization::PipekMezeyOptimizer>(optimizer),
        .final_gradient_norm = final_gradient_norm,
        .total_keyframes = total_keyframes,
        .total_hessian_actions = total_hessian_actions,
    };
  };

  localization::PipekMezeyResult occupied =
      read_space("/localization/occupied",
                 canonical.active_occupied_indices);
  localization::PipekMezeyResult virtual_space =
      read_space("/localization/virtual", canonical.virtual_indices);
  return LocalizedRhfCheckpoint{
      .canonical = std::move(canonical),
      .occupied = std::move(occupied),
      .virtual_space = std::move(virtual_space),
  };
}

mp2::TiledLaplaceMp2Progress read_tiled_laplace_mp2_restart(
    const std::filesystem::path& path) {
  const std::filesystem::path source =
      std::filesystem::absolute(path).lexically_normal();
  Hdf5Handle file{
      H5Fopen(source.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose};
  const int format_version = read_scalar<int>(
      file.get(), "/metadata/format_version", H5T_NATIVE_INT);
  const int write_complete = read_scalar<int>(
      file.get(), "/metadata/write_complete", H5T_NATIVE_INT);
  if(format_version != 1 || write_complete != 1) {
    throw std::runtime_error(
        "tiled Laplace MP2 restart format is unsupported or incomplete");
  }
  const std::uint64_t fingerprint =
      read_scalar<std::uint64_t>(
          file.get(), "/calculation/fingerprint", H5T_NATIVE_UINT64);
  std::vector<std::uint8_t> completed =
      read_vector<std::uint8_t>(
          file.get(), "/tiles/completed", H5T_NATIVE_UINT8);
  std::vector<double> contributions =
      read_vector<double>(
          file.get(), "/tiles/energy_contributions_hartree",
          H5T_NATIVE_DOUBLE);
  if(completed.empty() || completed.size() != contributions.size()) {
    throw std::runtime_error(
        "tiled Laplace MP2 restart tile arrays are empty or inconsistent");
  }
  for(std::size_t tile = 0; tile < completed.size(); ++tile) {
    if(completed[tile] > 1 || !std::isfinite(contributions[tile]) ||
       (completed[tile] == 0 && contributions[tile] != 0.0)) {
      throw std::runtime_error(
          "tiled Laplace MP2 restart contains invalid tile state");
    }
  }
  return mp2::TiledLaplaceMp2Progress{
      .fingerprint = fingerprint,
      .completed = std::move(completed),
      .tile_energy_contributions = std::move(contributions),
  };
}

void write_tiled_laplace_mp2_restart(
    const std::filesystem::path& path,
    const mp2::TiledLaplaceMp2Progress& progress, bool overwrite) {
  if(progress.fingerprint == 0 || progress.completed.empty() ||
     progress.completed.size() !=
         progress.tile_energy_contributions.size()) {
    throw std::invalid_argument(
        "tiled Laplace MP2 restart state is empty or inconsistent");
  }
  for(std::size_t tile = 0; tile < progress.completed.size(); ++tile) {
    if(progress.completed[tile] > 1 ||
       !std::isfinite(progress.tile_energy_contributions[tile]) ||
       (progress.completed[tile] == 0 &&
        progress.tile_energy_contributions[tile] != 0.0)) {
      throw std::invalid_argument(
          "tiled Laplace MP2 restart state contains an invalid tile");
    }
  }
  const std::filesystem::path destination =
      std::filesystem::absolute(path).lexically_normal();
  if(std::filesystem::exists(destination) && !overwrite) {
    throw std::runtime_error(
        "tiled Laplace MP2 restart already exists and overwrite is false: " +
        destination.string());
  }
  if(!destination.parent_path().empty() &&
     !std::filesystem::exists(destination.parent_path())) {
    throw std::runtime_error(
        "tiled Laplace MP2 restart parent directory does not exist: " +
        destination.parent_path().string());
  }
  const std::filesystem::path temporary = temporary_sibling(destination);
  try {
    {
      Hdf5Handle file{
          H5Fcreate(temporary.c_str(), H5F_ACC_EXCL, H5P_DEFAULT,
                    H5P_DEFAULT),
          H5Fclose};
      create_group(file.get(), "/metadata");
      create_group(file.get(), "/calculation");
      create_group(file.get(), "/tiles");
      write_scalar(file.get(), "/metadata/format_version",
                   H5T_STD_I32LE, 1);
      write_scalar(file.get(), "/metadata/write_complete",
                   H5T_STD_I32LE, 0);
      write_scalar(file.get(), "/calculation/fingerprint",
                   H5T_STD_U64LE, progress.fingerprint);
      write_vector(file.get(), "/tiles/completed", H5T_STD_U8LE,
                   progress.completed);
      write_vector(file.get(), "/tiles/energy_contributions_hartree",
                   H5T_IEEE_F64LE,
                   progress.tile_energy_contributions);
      check_hdf5(H5Fflush(file.get(), H5F_SCOPE_GLOBAL),
                 "flushing incomplete tiled Laplace MP2 restart");
    }
    {
      Hdf5Handle file{
          H5Fopen(temporary.c_str(), H5F_ACC_RDWR, H5P_DEFAULT),
          H5Fclose};
      Hdf5Handle complete{
          H5Dopen2(file.get(), "/metadata/write_complete",
                   H5P_DEFAULT),
          H5Dclose};
      const int value = 1;
      check_hdf5(
          H5Dwrite(complete.get(), H5T_NATIVE_INT, H5S_ALL, H5S_ALL,
                   H5P_DEFAULT, &value),
          "marking tiled Laplace MP2 restart complete");
      check_hdf5(H5Fflush(file.get(), H5F_SCOPE_GLOBAL),
                 "flushing completed tiled Laplace MP2 restart");
    }
    const mp2::TiledLaplaceMp2Progress verified =
        read_tiled_laplace_mp2_restart(temporary);
    if(verified.fingerprint != progress.fingerprint ||
       verified.completed != progress.completed ||
       verified.tile_energy_contributions !=
           progress.tile_energy_contributions) {
      throw std::runtime_error(
          "tiled Laplace MP2 restart read-back validation failed");
    }
    if(std::filesystem::exists(destination) && !overwrite) {
      throw std::runtime_error(
          "tiled Laplace MP2 restart appeared during write and overwrite "
          "is false: " +
          destination.string());
    }
    std::filesystem::rename(temporary, destination);
  } catch(...) {
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    throw;
  }
}

}  // namespace modernqc::io
