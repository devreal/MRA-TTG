#ifndef MRA_TYPES_H
#define MRA_TYPES_H

#include <complex>
#include <iostream>
#include <cstdint>
#include <cassert>
#include <array>
#include <ttg.h>

#include "mra/misc/platform.h"

#ifdef MRA_ENABLE_HOST
#ifndef TASKTYPE
#define TASKTYPE void
constexpr const ttg::ExecutionSpace Space = ttg::ExecutionSpace::Host;
#endif
#elif defined(MRA_ENABLE_CUDA)
#ifndef TASKTYPE
#define TASKTYPE ttg::device::Task
constexpr const ttg::ExecutionSpace Space = ttg::ExecutionSpace::CUDA;
#endif
#elif defined(MRA_ENABLE_HIP)
#ifndef TASKTYPE
#define TASKTYPE ttg::device::Task
constexpr const ttg::ExecutionSpace Space = ttg::ExecutionSpace::HIP;
#endif
#else
#error One of MRA_ENABLE_HOST, MRA_ENABLE_CUDA, or MRA_ENABLE_HIP must be defined.
#endif

namespace mra {

    // Presently making all of these different to try to identify misuses
    using Level = int16_t; //< Type of level in adaptive tree (>=0) .. 8 bit OK
    using Translation = uint32_t; //< Type of translation in adaptive tree (>=0)
    using Dimension = uint32_t; //size_t; //< Number of dimension in function or tensor (>=1) .. 8 bit OK
    using HashValue = uint64_t; //< Type of hash value used by madness (unsigned)

    using Process = uint32_t; //< MPI process rank (>=0)

    using size_type = uint32_t; // should be enough for a while
    using ssize_type = std::make_signed_t<size_type>;

    static constexpr Level MAX_LEVEL = sizeof(Translation)*8 - 1; //< Maximum level or depth of adaptive refinement

    //template <typename T, Dimension NDIM> using Coordinate = std::array<T,size_t(NDIM)>; //< Type of coordinate in NDIM dimensions

    /// Type of spatial coordinate in NDIM dimensions

    /// A separate type facilitates template matching for NDIM (removing the constraint that Dimension=size_t)
    template <typename T, Dimension NDIM>
    class Coordinate {
        std::array<T,NDIM> r;
    public:
        SCOPE Coordinate() = default;
        template <typename arg0T, typename...argsT>
        SCOPE Coordinate(arg0T arg0, argsT...args) : r{arg0,args...} {
            static_assert(sizeof...(argsT)==NDIM-1, "wrong number of initializers for coordinate");
        }
        SCOPE T operator[](size_t i) const {return r[i];}
        SCOPE T& operator[](size_t i) {return r[i];}
        SCOPE T operator()(size_t i) const {return r[i];}
        SCOPE T& operator()(size_t i) {return r[i];}
        SCOPE auto begin() const {return r.begin();}
        SCOPE auto end() const {return r.end();}
        SCOPE auto begin() {return r.begin();}
        SCOPE auto end() {return r.end();}
        SCOPE const auto& data() const {return r;} // mostly for easy printing
    };

    template <typename T, Dimension NDIM>
    std::ostream& operator<<(std::ostream& s, const Coordinate<T,NDIM>& r) {
        s << "Coordinate" << r.data();
        return s;
    }

    namespace detail {
        template <typename T> struct norm_type {using type = T;};
        template <typename T> struct norm_type<std::complex<T>> {using type = T;};

        template <typename T> struct coord_type {using type = T;};
        template <typename T> struct coord_type<std::complex<T>> {using type = T;};
    }

    /// Type of norms (i.e., complext<T> --> T)
    template <typename T> using norm_type = typename detail::norm_type<T>::type;

    namespace detail {
        // make_array not till c++20 ??
        template <typename T, std::size_t N, typename... Ts>
        SCOPE static inline auto make_array_crude(const Ts&... t) {
            return std::array<T, N>{{t...}};
        }

        // will fail for zero length subtuple
        template <std::size_t Begin, std::size_t End, typename... Ts, std::size_t... I>
        SCOPE static inline auto subtuple_to_array_of_ptrs_(std::tuple<Ts...>& t, std::index_sequence<I...>) {
            using arrayT = typename std::tuple_element<Begin, std::tuple<Ts*...>>::type;
            return make_array_crude<arrayT, End - Begin>(&std::get<I + Begin>(t)...);
        }

        template <std::size_t Begin, std::size_t End, typename... Ts, std::size_t... I>
        SCOPE static inline auto subtuple_to_array_of_ptrs_const(const std::tuple<Ts...>& t, std::index_sequence<I...>) {
            using arrayT = typename std::tuple_element<Begin, std::tuple<const Ts*...>>::type;
            return make_array_crude<arrayT, End - Begin>(&std::get<I + Begin>(t)...);
        }
    }

    // BS _const stuff needs cleaning up somehow likely by moving const from outside tuple to inside

    /// Makes an array of pointers to elements (of same type) in tuple in the open range \c [Begin,End).
    template <std::size_t Begin, std::size_t End, typename... T>
    SCOPE static inline auto subtuple_to_array_of_ptrs(std::tuple<T...>& t) {
        return detail::subtuple_to_array_of_ptrs_<Begin, End>(t, std::make_index_sequence<End - Begin>());
    }

    /// Makes an array of pointers to elements (of same type) in tuple in the open range \c [Begin,End).
    template <std::size_t Begin, std::size_t End, typename... T>
    SCOPE static inline auto subtuple_to_array_of_ptrs_const(const std::tuple<T...>& t) {
        return detail::subtuple_to_array_of_ptrs_const<Begin, End>(t, std::make_index_sequence<End - Begin>());
    }

    /// Makes an array of pointers to elements (of same type) in tuple
    template <typename... T>
    SCOPE static inline auto tuple_to_array_of_ptrs(std::tuple<T...>& t) {
        return detail::subtuple_to_array_of_ptrs_<0, std::tuple_size<std::tuple<T...>>::value>
            (t, std::make_index_sequence<std::tuple_size<std::tuple<T...>>::value>());
    }

    /// Makes an array of pointers to elements (of same type) in tuple
    template <typename... T>
    SCOPE static inline auto tuple_to_array_of_ptrs_const(const std::tuple<T...>& t) {
        return detail::subtuple_to_array_of_ptrs_const<0, std::tuple_size<std::tuple<T...>>::value>
            (t, std::make_index_sequence<std::tuple_size<std::tuple<T...>>::value>());
    }

    namespace detail {
        template <template <class...> class Trait, class Enabler, class... Args>
        struct is_detected : std::false_type{};

        template <template <class...> class Trait, class... Args>
        struct is_detected<Trait, std::void_t<Trait<Args...>>, Args...> : std::true_type{};
    }

    /// Support for is-detected idiom

    /// Use as follows --- this example detects classes with member function with signature `n(double) const`.
    /// \code
    /// template <class T> using n_t = decltype(std::declval<const T>().n(std::declval<double>()));
    /// template <class T> using supports_n = detail::is_detected<n_t,T>;
    /// \endcode
    /// then inside a templated function (or similary using \c std::enable_if ).
    /// \code
    /// if constexpr (supports_n<T>::value) ...
    /// \endcode
    template <template <class...> class Trait, class... Args>
    using is_detected = typename detail::is_detected<Trait, void, Args...>::type;
} // namespace mra

#endif // MRA_TYPES_H
