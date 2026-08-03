#ifndef MRA_JIT_TYPE_NAME_H
#define MRA_JIT_TYPE_NAME_H

#include "mra/misc/types.h"
#include "mra/tensor/tensorview.h"

#include <array>
#include <string>

namespace mra::jit {

  /**
   * Produces the fully-qualified, NVRTC/hiprtc-parseable source spelling of
   * T -- exactly what has to appear in a name-expression string (see
   * Compiler::compile()) to explicitly specify an invented (abbreviated-
   * auto) template parameter, since explicit specification is the only way
   * to instantiate a specific concrete kernel via nvrtcAddNameExpression
   * (there's no call-site argument for the compiler to deduce from). See
   * spike/nvrtc/gaxpy_spike.cc for how this was worked out empirically.
   *
   * Deliberately NOT implemented via a generic reflection trick (e.g. a
   * __PRETTY_FUNCTION__-based type_name<T>()): that produces compiler-
   * internal spellings (alias types resolved to their underlying type, or
   * unspecified formatting differences between GCC/Clang) which may happen
   * to work but aren't a reliable, portable contract. Instead, only the
   * finite, closed set of concrete types actually used as kernel view-
   * parameter arguments in this codebase is specialized here -- extend
   * this list as new concrete types get used with the JIT path.
   */
  template <typename T>
  struct type_name;

  template <>
  struct type_name<double> {
    static std::string value() { return "double"; }
  };

  template <>
  struct type_name<float> {
    static std::string value() { return "float"; }
  };

  template <typename T, Dimension NDIM>
  struct type_name<DenseTensorView<T, NDIM>> {
    static std::string value() {
      return "mra::DenseTensorView<" + type_name<T>::value() + "," + std::to_string(NDIM) + ">";
    }
  };

  template <typename T, Dimension NDIM>
  struct type_name<SparseTensorView<T, NDIM>> {
    static std::string value() {
      return "mra::SparseTensorView<" + type_name<T>::value() + "," + std::to_string(NDIM) + ">";
    }
  };

  /// Generic: std::array<T,N> for any T that itself has a type_name
  /// specialization (e.g. compress_kernel's in_views parameter, a
  /// std::array<SparseTensorView<T,NDIM+1>, num_children>).
  template <typename T, std::size_t N>
  struct type_name<std::array<T, N>> {
    static std::string value() {
      return "std::array<" + type_name<T>::value() + "," + std::to_string(N) + ">";
    }
  };

  /// Convenience wrapper: type_name_v<decltype(some_expr)>() rather than
  /// spelling out type_name<std::decay_t<decltype(some_expr)>>::value().
  template <typename T>
  std::string type_name_v() {
    return type_name<std::decay_t<T>>::value();
  }

} // namespace mra::jit

#endif // MRA_JIT_TYPE_NAME_H
