#include "modernqc/basis/basis_name.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

namespace modernqc::basis {

std::string canonical_basis_name(std::string_view name) {
  std::string normalized{name};
  normalized.erase(
      normalized.begin(),
      std::find_if(normalized.begin(), normalized.end(),
                   [](unsigned char value) { return !std::isspace(value); }));
  normalized.erase(
      std::find_if(normalized.rbegin(), normalized.rend(),
                   [](unsigned char value) { return !std::isspace(value); })
          .base(),
      normalized.end());
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](unsigned char value) {
                   return static_cast<char>(std::tolower(value));
                 });

  if(normalized == "avdz" || normalized == "aug-cc-pvdz") {
    return "aug-cc-pVDZ";
  }
  if(normalized == "avtz" || normalized == "aug-cc-pvtz") {
    return "aug-cc-pVTZ";
  }
  if(normalized == "avqz" || normalized == "aug-cc-pvqz") {
    return "aug-cc-pVQZ";
  }
  throw std::invalid_argument(
      "unsupported basis '" + std::string{name} +
      "'; supported names are aug-cc-pVDZ/avdz, aug-cc-pVTZ/avtz, "
      "and aug-cc-pVQZ/avqz");
}

}  // namespace modernqc::basis
