#include "modernqc/molecule/xyz_reader.hpp"

#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace modernqc::molecule {
namespace {

[[nodiscard]] std::size_t parse_atom_count(const std::string& line) {
  std::istringstream parser{line};
  long long count = 0;
  std::string trailing;
  if(!(parser >> count) || count <= 0 || (parser >> trailing)) {
    throw std::invalid_argument(
        "XYZ line 1 must contain exactly one positive atom count");
  }
  if(static_cast<unsigned long long>(count) >
     static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
    throw std::invalid_argument("XYZ atom count is too large");
  }
  return static_cast<std::size_t>(count);
}

}  // namespace

std::vector<Atom> read_xyz(std::istream& input, core::CoordinateUnit input_unit) {
  std::string line;
  if(!std::getline(input, line)) {
    throw std::invalid_argument("XYZ input is empty");
  }
  const std::size_t atom_count = parse_atom_count(line);

  if(!std::getline(input, line)) {
    throw std::invalid_argument("XYZ input is missing the comment line");
  }

  std::vector<Atom> atoms;
  atoms.reserve(atom_count);
  for(std::size_t atom_index = 0; atom_index < atom_count; ++atom_index) {
    if(!std::getline(input, line)) {
      throw std::invalid_argument("XYZ input ended before all atoms were read");
    }
    std::istringstream parser{line};
    std::string symbol_token;
    std::array<double, 3> coordinates{};
    std::string trailing;
    if(!(parser >> symbol_token >> coordinates[0] >> coordinates[1] >> coordinates[2]) ||
       (parser >> trailing)) {
      throw std::invalid_argument("malformed XYZ atom record at atom " +
                                  std::to_string(atom_index + 1));
    }
    const std::string symbol = canonical_element_symbol(symbol_token);
    for(double& coordinate : coordinates) {
      if(!std::isfinite(coordinate)) {
        throw std::invalid_argument("XYZ coordinates must be finite");
      }
      coordinate = core::to_bohr(coordinate, input_unit);
    }
    atoms.push_back(
        Atom{.symbol = symbol,
             .atomic_number = atomic_number(symbol),
             .position_bohr = coordinates});
  }

  while(std::getline(input, line)) {
    if(line.find_first_not_of(" \t\r") != std::string::npos) {
      throw std::invalid_argument("XYZ input contains extra nonempty records");
    }
  }
  return atoms;
}

std::vector<Atom> read_xyz_file(const std::filesystem::path& path,
                                core::CoordinateUnit input_unit) {
  std::ifstream input{path};
  if(!input) {
    throw std::runtime_error("unable to open XYZ geometry '" + path.string() + "'");
  }
  try {
    return read_xyz(input, input_unit);
  } catch(const std::exception& error) {
    throw std::invalid_argument("invalid XYZ geometry '" + path.string() +
                                "': " + error.what());
  }
}

}  // namespace modernqc::molecule
