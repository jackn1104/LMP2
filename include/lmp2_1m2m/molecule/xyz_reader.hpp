#pragma once

#include "lmp2_1m2m/core/units.hpp"
#include "lmp2_1m2m/molecule/atom.hpp"

#include <filesystem>
#include <istream>
#include <vector>

namespace lmp2_1m2m::molecule {

[[nodiscard]] std::vector<Atom> read_xyz(
    std::istream& input, core::CoordinateUnit input_unit);

[[nodiscard]] std::vector<Atom> read_xyz_file(
    const std::filesystem::path& path, core::CoordinateUnit input_unit);

}  // namespace lmp2_1m2m::molecule
