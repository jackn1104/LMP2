#pragma once

#include "lmp2_1m2m/basis/basis_set.hpp"
#include "lmp2_1m2m/molecule/molecule.hpp"

#include <string_view>

namespace lmp2_1m2m::basis {

// Constructs a backend-neutral basis description. Libint2 types remain confined
// to the implementation file.
[[nodiscard]] BasisSet build_libint2_basis(
    const molecule::Molecule& molecule, std::string_view basis_name,
    bool spherical);

}  // namespace lmp2_1m2m::basis
