#ifndef MRA_MISC_INTEGER_H
#define MRA_MISC_INTEGER_H
#include <cstdint>
#include <type_traits>
#include <mra/misc/types.h>

namespace mra {

  // A simple compile-time Int type.
  // Use Int<N> to represent the compile-time value N.
  // All operations produce another Int<...> and are constexpr-evaluable.
  template<size_type N>
  struct Int {
    using value_type = size_type;
    static constexpr value_type value = N;
    using type = Int;

    constexpr Int() noexcept = default;

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

    constexpr operator value_type() const noexcept { return N; }
  };

  // Helper alias and value object
  template<size_type N>
  using int_c = Int<N>;

  template<size_type N>
  inline constexpr Int<N> Int_v{};

  // Support operations between Int<N> and runtime integral values.
  // Results cannot be represented as a compile-time Int anymore, so
  // these overloads return the runtime value_type (size_type) or bool for comparisons.

  template<size_type N, typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr typename Int<N>::value_type operator+(Int<N>, T rhs) noexcept {
    return N + static_cast<typename Int<N>::value_type>(rhs);
  }
  template<typename T, size_type N, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr typename Int<N>::value_type operator+(T lhs, Int<N>) noexcept {
    return static_cast<typename Int<N>::value_type>(lhs) + N;
  }

  template<size_type N, typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr typename Int<N>::value_type operator-(Int<N>, T rhs) noexcept {
      return N - static_cast<typename Int<N>::value_type>(rhs);
  }
  template<typename T, size_type N, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr typename Int<N>::value_type operator-(T lhs, Int<N>) noexcept {
      return static_cast<typename Int<N>::value_type>(lhs) - N;
  }

  template<size_type N, typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr typename Int<N>::value_type operator*(Int<N>, T rhs) noexcept {
    return N * static_cast<typename Int<N>::value_type>(rhs);
  }
  template<typename T, size_type N, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr typename Int<N>::value_type operator*(T lhs, Int<N>) noexcept {
    return static_cast<typename Int<N>::value_type>(lhs) * N;
  }

  template<size_type N, typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr typename Int<N>::value_type operator/(Int<N>, T rhs) noexcept {
    return N / static_cast<typename Int<N>::value_type>(rhs);
  }
  template<typename T, size_type N, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr typename Int<N>::value_type operator/(T lhs, Int<N>) noexcept {
    return static_cast<typename Int<N>::value_type>(lhs) / N;
  }

  template<size_type N, typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr typename Int<N>::value_type operator%(Int<N>, T rhs) noexcept {
    return N % static_cast<typename Int<N>::value_type>(rhs);
  }
  template<typename T, size_type N, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr typename Int<N>::value_type operator%(T lhs, Int<N>) noexcept {
    return static_cast<typename Int<N>::value_type>(lhs) % N;
  }

  // Bitwise with runtime integral
  template<size_type N, typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr typename Int<N>::value_type operator&(Int<N>, T rhs) noexcept {
    return N & static_cast<typename Int<N>::value_type>(rhs);
  }
  template<typename T, size_type N, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr typename Int<N>::value_type operator&(T lhs, Int<N>) noexcept {
    return static_cast<typename Int<N>::value_type>(lhs) & N;
  }

  template<size_type N, typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr typename Int<N>::value_type operator|(Int<N>, T rhs) noexcept {
    return N | static_cast<typename Int<N>::value_type>(rhs);
  }
  template<typename T, size_type N, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr typename Int<N>::value_type operator|(T lhs, Int<N>) noexcept {
    return static_cast<typename Int<N>::value_type>(lhs) | N;
  }

  template<size_type N, typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr typename Int<N>::value_type operator^(Int<N>, T rhs) noexcept {
    return N ^ static_cast<typename Int<N>::value_type>(rhs);
  }
  template<typename T, size_type N, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr typename Int<N>::value_type operator^(T lhs, Int<N>) noexcept {
    return static_cast<typename Int<N>::value_type>(lhs) ^ N;
  }

  // Shifts: runtime shift amount -> runtime result
  template<size_type N, typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr typename Int<N>::value_type operator<<(Int<N>, T rhs) noexcept {
      return N << static_cast<typename Int<N>::value_type>(rhs);
  }
  template<typename T, size_type N, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr typename Int<N>::value_type operator<<(T lhs, Int<N>) noexcept {
    return static_cast<typename Int<N>::value_type>(lhs) << N;
  }

  template<size_type N, typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr typename Int<N>::value_type operator>>(Int<N>, T rhs) noexcept {
      return N >> static_cast<typename Int<N>::value_type>(rhs);
  }
  template<typename T, size_type N, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr typename Int<N>::value_type operator>>(T lhs, Int<N>) noexcept {
    return static_cast<typename Int<N>::value_type>(lhs) >> N;
  }

  // Comparisons with runtime integral values
  template<size_type N, typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr bool operator==(Int<N>, T rhs) noexcept {
    return N == static_cast<typename Int<N>::value_type>(rhs);
  }
  template<typename T, size_type N, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr bool operator==(T lhs, Int<N>) noexcept {
    return static_cast<typename Int<N>::value_type>(lhs) == N;
  }

  template<size_type N, typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr bool operator!=(Int<N>, T rhs) noexcept {
    return N != static_cast<typename Int<N>::value_type>(rhs);
  }
  template<typename T, size_type N, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr bool operator!=(T lhs, Int<N>) noexcept {
    return static_cast<typename Int<N>::value_type>(lhs) != N;
  }

  template<size_type N, typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr bool operator<(Int<N>, T rhs) noexcept {
    return N < static_cast<typename Int<N>::value_type>(rhs);
  }
  template<typename T, size_type N, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr bool operator<(T lhs, Int<N>) noexcept {
    return static_cast<typename Int<N>::value_type>(lhs) < N;
  }

  template<size_type N, typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr bool operator<=(Int<N>, T rhs) noexcept {
    return N <= static_cast<typename Int<N>::value_type>(rhs);
  }
  template<typename T, size_type N, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr bool operator<=(T lhs, Int<N>) noexcept {
    return static_cast<typename Int<N>::value_type>(lhs) <= N;
  }

  template<size_type N, typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr bool operator>(Int<N>, T rhs) noexcept {
    return N > static_cast<typename Int<N>::value_type>(rhs);
  }
  template<typename T, size_type N, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr bool operator>(T lhs, Int<N>) noexcept {
    return static_cast<typename Int<N>::value_type>(lhs) > N;
  }

  template<size_type N, typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr bool operator>=(Int<N>, T rhs) noexcept {
    return N >= static_cast<typename Int<N>::value_type>(rhs);
  }
  template<typename T, size_type N, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr bool operator>=(T lhs, Int<N>) noexcept {
    return static_cast<typename Int<N>::value_type>(lhs) >= N;
  }

  template<size_type Base, size_type Exponent>
  constexpr auto pow(Int<Base>, Int<Exponent>) noexcept {
    if constexpr (Exponent == 0) {
      return Int<1>{};
    } else if constexpr (Exponent == 1) {
      return Int<Base>{};
    } else {
      return Int<Base>{} * pow(Int<Base>{}, Int<(Exponent - 1)>{});
    }
  }

  constexpr auto pow(auto base, size_type exponent) noexcept {
    if (exponent == 0) {
      if constexpr (std::is_arithmetic_v<decltype(base)>) {
        return (decltype(base))1;
      } else {
        return Int<1>{};
      }
    } else if (exponent == 1) {
      return base;
    } else {
      return base * pow(base, exponent - 1);
    }
  }


  template<typename T>
  struct is_integer : public std::is_integral<T> {};

  template<size_type N>
  struct is_integer<Int<N>> : public std::true_type {};

  /**
   * A trait to check if a type is an integral type (including Int<N>).
   */
  template<typename T>
  constexpr bool is_integer_v = is_integer<std::remove_cv_t<T>>::value;

  namespace concepts {
    template<typename T>
    concept Integer = is_integer_v<T>;
  } // namespace concepts


  template<typename T>
  struct is_ct_integral : std::false_type {};

  template<size_type N>
  struct is_ct_integral<Int<N>> : public std::true_type {};

  /**
   * A trait to check if a type is a compile-time integral constant (Int<N>).
   */
  template<typename T>
  constexpr bool is_ct_integral_v = is_ct_integral<std::decay_t<T>>::value;

  namespace concepts {
    template<typename T>
    concept CtIntegral = is_ct_integral_v<T>;
  } // namespace concepts


} // namespace mra

#endif // MRA_MISC_INTEGER_H