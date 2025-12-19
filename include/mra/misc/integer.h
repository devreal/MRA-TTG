#ifndef MRA_MISC_INTEGER_H
#define MRA_MISC_INTEGER_H
#include <cstdint>
#include <type_traits>

namespace mra {

// A simple compile-time Int type.
// Use Int<N> to represent the compile-time value N.
// All operations produce another Int<...> and are constexpr-evaluable.
template<std::intmax_t N>
struct Int {
  using value_type = std::intmax_t;
  static constexpr value_type value = N;
  using type = Int;

  // value object for convenience
  static inline constexpr Int instance{}; // can be used as `Int<3>::instance`

  // Unary
  friend constexpr Int<-N> operator-(Int) noexcept { return {}; }
  friend constexpr Int<~N> operator~(Int) noexcept { return {}; }

  // Arithmetic
  template<value_type M>
  friend constexpr Int<N + M> operator+(Int, Int<M>) noexcept { return {}; }

  template<value_type M>
  friend constexpr Int<N - M> operator-(Int, Int<M>) noexcept { return {}; }

  template<value_type M>
  friend constexpr Int<N * M> operator*(Int, Int<M>) noexcept { return {}; }

  template<value_type M>
  friend constexpr Int<N / M> operator/(Int, Int<M>) noexcept {
    static_assert(M != 0, "division by zero in mra::misc::Int");
    return {};
  }

  template<value_type M>
  friend constexpr Int<N % M> operator%(Int, Int<M>) noexcept {
    static_assert(M != 0, "modulo by zero in mra::misc::Int");
    return {};
  }

  // Bitwise
  template<value_type M>
  friend constexpr Int<(N & M)> operator&(Int, Int<M>) noexcept { return {}; }

  template<value_type M>
  friend constexpr Int<(N | M)> operator|(Int, Int<M>) noexcept { return {}; }

  template<value_type M>
  friend constexpr Int<(N ^ M)> operator^(Int, Int<M>) noexcept { return {}; }

  // Shifts (check shift amount non-negative)
  template<value_type M>
  friend constexpr Int<(N << M)> operator<<(Int, Int<M>) noexcept {
    static_assert(M >= 0, "left shift by negative in mra::misc::Int");
    return {};
  }

  template<value_type M>
  friend constexpr Int<(N >> M)> operator>>(Int, Int<M>) noexcept {
    static_assert(M >= 0, "right shift by negative in mra::misc::Int");
    return {};
  }

  // Comparisons (constexpr bool)
  template<value_type M>
  friend constexpr bool operator==(Int, Int<M>) noexcept { return N == M; }

  template<value_type M>
  friend constexpr bool operator!=(Int, Int<M>) noexcept { return N != M; }

  template<value_type M>
  friend constexpr bool operator<(Int, Int<M>) noexcept { return N < M; }

  template<value_type M>
  friend constexpr bool operator<=(Int, Int<M>) noexcept { return N <= M; }

  template<value_type M>
  friend constexpr bool operator>(Int, Int<M>) noexcept { return N > M; }

  template<value_type M>
  friend constexpr bool operator>=(Int, Int<M>) noexcept { return N >= M; }

  // Convert to runtime value (constexpr)
  static constexpr value_type to_value() noexcept { return N; }
};

// Helper alias and value object
template<std::intmax_t N>
using int_c = Int<N>;

template<std::intmax_t N>
inline constexpr Int<N> Int_v{};

// Support operations between Int<N> and runtime integral values.
// Results cannot be represented as a compile-time Int anymore, so
// these overloads return the runtime value_type (std::intmax_t) or bool for comparisons.

template<std::intmax_t N, typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
constexpr typename Int<N>::value_type operator+(Int<N>, T rhs) noexcept {
  return N + static_cast<typename Int<N>::value_type>(rhs);
}
template<typename T, std::intmax_t N, typename = std::enable_if_t<std::is_integral_v<T>>>
constexpr typename Int<N>::value_type operator+(T lhs, Int<N>) noexcept {
  return static_cast<typename Int<N>::value_type>(lhs) + N;
}

template<std::intmax_t N, typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
constexpr typename Int<N>::value_type operator-(Int<N>, T rhs) noexcept {
    return N - static_cast<typename Int<N>::value_type>(rhs);
}
template<typename T, std::intmax_t N, typename = std::enable_if_t<std::is_integral_v<T>>>
constexpr typename Int<N>::value_type operator-(T lhs, Int<N>) noexcept {
    return static_cast<typename Int<N>::value_type>(lhs) - N;
}

template<std::intmax_t N, typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
constexpr typename Int<N>::value_type operator*(Int<N>, T rhs) noexcept {
  return N * static_cast<typename Int<N>::value_type>(rhs);
}
template<typename T, std::intmax_t N, typename = std::enable_if_t<std::is_integral_v<T>>>
constexpr typename Int<N>::value_type operator*(T lhs, Int<N>) noexcept {
  return static_cast<typename Int<N>::value_type>(lhs) * N;
}

template<std::intmax_t N, typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
constexpr typename Int<N>::value_type operator/(Int<N>, T rhs) noexcept {
  return N / static_cast<typename Int<N>::value_type>(rhs);
}
template<typename T, std::intmax_t N, typename = std::enable_if_t<std::is_integral_v<T>>>
constexpr typename Int<N>::value_type operator/(T lhs, Int<N>) noexcept {
  return static_cast<typename Int<N>::value_type>(lhs) / N;
}

template<std::intmax_t N, typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
constexpr typename Int<N>::value_type operator%(Int<N>, T rhs) noexcept {
  return N % static_cast<typename Int<N>::value_type>(rhs);
}
template<typename T, std::intmax_t N, typename = std::enable_if_t<std::is_integral_v<T>>>
constexpr typename Int<N>::value_type operator%(T lhs, Int<N>) noexcept {
  return static_cast<typename Int<N>::value_type>(lhs) % N;
}

// Bitwise with runtime integral
template<std::intmax_t N, typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
constexpr typename Int<N>::value_type operator&(Int<N>, T rhs) noexcept {
  return N & static_cast<typename Int<N>::value_type>(rhs);
}
template<typename T, std::intmax_t N, typename = std::enable_if_t<std::is_integral_v<T>>>
constexpr typename Int<N>::value_type operator&(T lhs, Int<N>) noexcept {
  return static_cast<typename Int<N>::value_type>(lhs) & N;
}

template<std::intmax_t N, typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
constexpr typename Int<N>::value_type operator|(Int<N>, T rhs) noexcept {
  return N | static_cast<typename Int<N>::value_type>(rhs);
}
template<typename T, std::intmax_t N, typename = std::enable_if_t<std::is_integral_v<T>>>
constexpr typename Int<N>::value_type operator|(T lhs, Int<N>) noexcept {
  return static_cast<typename Int<N>::value_type>(lhs) | N;
}

template<std::intmax_t N, typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
constexpr typename Int<N>::value_type operator^(Int<N>, T rhs) noexcept {
  return N ^ static_cast<typename Int<N>::value_type>(rhs);
}
template<typename T, std::intmax_t N, typename = std::enable_if_t<std::is_integral_v<T>>>
constexpr typename Int<N>::value_type operator^(T lhs, Int<N>) noexcept {
  return static_cast<typename Int<N>::value_type>(lhs) ^ N;
}

// Shifts: runtime shift amount -> runtime result
template<std::intmax_t N, typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
constexpr typename Int<N>::value_type operator<<(Int<N>, T rhs) noexcept {
    return N << static_cast<typename Int<N>::value_type>(rhs);
}
template<typename T, std::intmax_t N, typename = std::enable_if_t<std::is_integral_v<T>>>
constexpr typename Int<N>::value_type operator<<(T lhs, Int<N>) noexcept {
  return static_cast<typename Int<N>::value_type>(lhs) << N;
}

template<std::intmax_t N, typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
constexpr typename Int<N>::value_type operator>>(Int<N>, T rhs) noexcept {
    return N >> static_cast<typename Int<N>::value_type>(rhs);
}
template<typename T, std::intmax_t N, typename = std::enable_if_t<std::is_integral_v<T>>>
constexpr typename Int<N>::value_type operator>>(T lhs, Int<N>) noexcept {
  return static_cast<typename Int<N>::value_type>(lhs) >> N;
}

// Comparisons with runtime integral values
template<std::intmax_t N, typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
constexpr bool operator==(Int<N>, T rhs) noexcept {
  return N == static_cast<typename Int<N>::value_type>(rhs);
}
template<typename T, std::intmax_t N, typename = std::enable_if_t<std::is_integral_v<T>>>
constexpr bool operator==(T lhs, Int<N>) noexcept {
  return static_cast<typename Int<N>::value_type>(lhs) == N;
}

template<std::intmax_t N, typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
constexpr bool operator!=(Int<N>, T rhs) noexcept {
  return N != static_cast<typename Int<N>::value_type>(rhs);
}
template<typename T, std::intmax_t N, typename = std::enable_if_t<std::is_integral_v<T>>>
constexpr bool operator!=(T lhs, Int<N>) noexcept {
  return static_cast<typename Int<N>::value_type>(lhs) != N;
}

template<std::intmax_t N, typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
constexpr bool operator<(Int<N>, T rhs) noexcept {
  return N < static_cast<typename Int<N>::value_type>(rhs);
}
template<typename T, std::intmax_t N, typename = std::enable_if_t<std::is_integral_v<T>>>
constexpr bool operator<(T lhs, Int<N>) noexcept {
  return static_cast<typename Int<N>::value_type>(lhs) < N;
}

template<std::intmax_t N, typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
constexpr bool operator<=(Int<N>, T rhs) noexcept {
  return N <= static_cast<typename Int<N>::value_type>(rhs);
}
template<typename T, std::intmax_t N, typename = std::enable_if_t<std::is_integral_v<T>>>
constexpr bool operator<=(T lhs, Int<N>) noexcept {
  return static_cast<typename Int<N>::value_type>(lhs) <= N;
}

template<std::intmax_t N, typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
constexpr bool operator>(Int<N>, T rhs) noexcept {
  return N > static_cast<typename Int<N>::value_type>(rhs);
}
template<typename T, std::intmax_t N, typename = std::enable_if_t<std::is_integral_v<T>>>
constexpr bool operator>(T lhs, Int<N>) noexcept {
  return static_cast<typename Int<N>::value_type>(lhs) > N;
}

template<std::intmax_t N, typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
constexpr bool operator>=(Int<N>, T rhs) noexcept {
  return N >= static_cast<typename Int<N>::value_type>(rhs);
}
template<typename T, std::intmax_t N, typename = std::enable_if_t<std::is_integral_v<T>>>
constexpr bool operator>=(T lhs, Int<N>) noexcept {
  return static_cast<typename Int<N>::value_type>(lhs) >= N;
}

template<typename T>
struct is_integer : std::is_integral<T> {};

template<std::intmax_t N>
struct is_integer<Int<N>> : std::true_type {};

template<typename T>
constexpr bool is_integer_v = is_integer<std::remove_cv_t<T>>::value;

template<typename T>
concept Integer = is_integer_v<T>;

} // namespace mra

#endif // MRA_MISC_INTEGER_H