#ifndef MRA_JIT_STD_COMPAT_H
#define MRA_JIT_STD_COMPAT_H

// Only meaningful when compiling for NVRTC/hiprtc JIT (MRA_JIT_COMPILE).
//
// NVRTC has no standard library access of its own (verified empirically,
// see spike/nvrtc/gaxpy_spike.cc: it can't even find <cstdint>), and real
// system libstdc++/libc++ headers are not a viable fallback via -I either --
// GCC-specific builtin macros like __SIZE_TYPE__ aren't predefined by
// NVRTC's frontend, and NVRTC rejects any unannotated inline function as a
// "host function ... not allowed in JIT mode".
//
// NVIDIA's libcu++ (cuda::std::*, shipped under <cuda/std/...> in every CUDA
// toolkit) is purpose-built to be NVRTC-safe. Rather than rewrite every
// std:: call site in the kernel headers to cuda::std:: (far more invasive
// than the "guard existing headers in place" approach used elsewhere for
// JIT support), this redirects the small set of std:: facilities those
// headers actually use to their cuda::std:: equivalents. Injecting into
// namespace std is technically UB per the standard, but is a well-precedented
// pattern for exactly this problem (e.g. NVIDIA's own Jitify does the same)
// and is confined entirely to the JIT compile pass -- it never affects the
// AOT host/CUDA/HIP build, where MRA_JIT_COMPILE is never defined and this
// header's contents are skipped entirely.
#if defined(MRA_JIT_COMPILE)

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

namespace std {
  using ::cuda::std::array;

  using ::cuda::std::int16_t;
  using ::cuda::std::int32_t;
  using ::cuda::std::uint8_t;
  using ::cuda::std::uint32_t;
  using ::cuda::std::uint64_t;
  using ::cuda::std::size_t;

  using ::cuda::std::tuple;
  using ::cuda::std::get;
  // NOTE: tuple_size/tuple_element are deliberately NOT redeclared here --
  // <cuda/std/array>/<cuda/std/tuple> already inject their own
  // specializations directly into (this) namespace std for structured-
  // binding support, and a using-declaration for the same name conflicts
  // with that ("already declared in the current scope").
  using ::cuda::std::tuple_size_v;

  using ::cuda::std::index_sequence;
  using ::cuda::std::make_index_sequence;

  using ::cuda::std::decay_t;
  using ::cuda::std::enable_if;
  using ::cuda::std::enable_if_t;
  using ::cuda::std::conditional_t;
  using ::cuda::std::make_signed_t;
  using ::cuda::std::is_const_v;
  using ::cuda::std::is_integral_v;
  using ::cuda::std::is_same_v;
  using ::cuda::std::remove_reference_t;
  using ::cuda::std::remove_cv_t;
  using ::cuda::std::add_const_t;
  using ::cuda::std::void_t;
  using ::cuda::std::true_type;
  using ::cuda::std::false_type;
  using ::cuda::std::integral_constant;
  using ::cuda::std::numeric_limits;

  // CUDA's device runtime declares pow/sqrt/tgamma globally (no <cmath>
  // needed), same as ::printf, but only for a handful of fixed overloads
  // (e.g. pow(double,double)/(float,float)/(float,int)/(double,int)) --
  // cuda::std doesn't expose a plain (non-complex) equivalent at a stable
  // public path in this CCCL release either. A `using ::pow;` alone is
  // ambiguous for e.g. pow(uint32_t, uint32_t) (equally-good conversion to
  // (double,double) or (float,float)), which real <cmath>'s much larger
  // overload set doesn't hit -- so cast unconditionally to double instead
  // of importing the raw overload set.
  template <typename A, typename B>
  __host__ __device__ double pow(A a, B b) { return ::pow(static_cast<double>(a), static_cast<double>(b)); }

  template <typename A>
  __host__ __device__ double sqrt(A a) { return ::sqrt(static_cast<double>(a)); }

  template <typename A>
  __host__ __device__ double tgamma(A a) { return ::tgamma(static_cast<double>(a)); }

  // This CCCL/libcu++ release exposes min/max/swap/fill/tie only through
  // internal <cuda/std/__algorithm/*.h>/<cuda/std/__utility/*.h> paths (no
  // public <cuda/std/algorithm> umbrella yet), which are not a stable
  // surface to depend on. These are trivial enough to hand-roll instead.
  template <typename T>
  __host__ __device__ constexpr const T& min(const T& a, const T& b) { return (b < a) ? b : a; }

  template <typename T>
  __host__ __device__ constexpr const T& max(const T& a, const T& b) { return (a < b) ? b : a; }

  template <typename T>
  __host__ __device__ constexpr void swap(T& a, T& b) {
    T tmp = static_cast<T&&>(a);
    a = static_cast<T&&>(b);
    b = static_cast<T&&>(tmp);
  }

  template <typename Iter, typename T>
  __host__ __device__ constexpr void fill(Iter first, Iter last, const T& value) {
    for (; first != last; ++first) *first = value;
  }

  template <typename T>
  struct multiplies {
    __host__ __device__ constexpr T operator()(const T& a, const T& b) const { return a * b; }
  };

  template <typename Iter, typename T, typename BinOp>
  __host__ __device__ constexpr T reduce(Iter first, Iter last, T init, BinOp op) {
    for (; first != last; ++first) init = op(init, *first);
    return init;
  }

  template <typename Iter, typename T>
  __host__ __device__ constexpr void iota(Iter first, Iter last, T value) {
    for (; first != last; ++first, ++value) *first = value;
  }

  template <typename... Ts>
  __host__ __device__ constexpr tuple<Ts&...> tie(Ts&... args) { return tuple<Ts&...>(args...); }

  // This CCCL/libcu++ release has no cuda::std::pair at all; a minimal
  // stand-in (first/second access only, no structured-binding support --
  // not needed by the kernel headers) is simpler than depending on an
  // internal path that may not exist either.
  template <typename T1, typename T2>
  struct pair {
    T1 first;
    T2 second;
    constexpr pair() = default;
    __host__ __device__ constexpr pair(const T1& a, const T2& b) : first(a), second(b) {}
  };

  template <typename T1, typename T2>
  __host__ __device__ constexpr pair<T1, T2> make_pair(T1 a, T2 b) { return pair<T1, T2>(a, b); }

  template <typename T>
  __host__ __device__ constexpr T abs(T x) { return x < T(0) ? -x : x; }

  // Primary template forward declaration only -- mra::Key<NDIM> (misc/key.h)
  // specializes this for its own hashing, and doesn't need a working
  // default std::hash<T> for any other type.
  template <typename T> struct hash;
}  // namespace std

#endif  // MRA_JIT_COMPILE
#endif  // MRA_JIT_STD_COMPAT_H
