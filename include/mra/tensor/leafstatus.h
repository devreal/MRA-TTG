#ifndef HAVE_LEAF_STATUS_H
#define HAVE_LEAF_STATUS_H

#include <cstdint>

namespace mra {

  /**
   * Status of a node in the function tree.
   */
  enum class LeafStatus : uint8_t {
      Inner,    // Inner node, has children but is empty
      Leaf,     // Is leaf, has no children, not empty
      Invalid   // Below leaf level, should be considered zero
  };

  inline std::ostream& operator<<(std::ostream& os, LeafStatus status) {
      switch (status) {
          case LeafStatus::Inner: return os << "IN";
          case LeafStatus::Leaf: return os << "LF";
          case LeafStatus::Invalid: return os << "IV";
          default: return os << "UNKNOWN";
      }
  }

} // namespace mra

#endif // HAVE_LEAF_STATUS_H