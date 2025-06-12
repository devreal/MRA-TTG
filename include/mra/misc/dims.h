#ifndef HAVE_DIMS_H
#define HAVE_DIMS_H
#include "mra/misc/types.h"


namespace mra {

  namespace detail {

    template<Dimension NDIM, typename... Sizes, std::size_t... Is>
    constexpr std::array<size_type, NDIM> make_dims_helper(std::index_sequence<Is...>, Sizes&&... sizes) {
      auto tmp = std::array<size_type, sizeof...(Sizes)>{static_cast<size_type>(sizes)...};
      size_type K = tmp[sizeof...(Sizes) - 1];
      return std::array<size_type, NDIM>{static_cast<size_type>(sizes)..., ((void)Is, K)...};
    }

  } // namespace detail

  /* Create a dims array with the provided first sizes and pad to NDIM with the last size */
  template<Dimension NDIM, typename... Sizes>
  constexpr std::array<size_type, NDIM> make_dims(Sizes&&... sizes) {
    static_assert(sizeof...(Sizes) <= NDIM, "Too many sizes provided for the number of dimensions");
    return detail::make_dims_helper<NDIM>(std::make_index_sequence<NDIM-sizeof...(Sizes)>{}, std::forward<Sizes>(sizes)...);
  }

} // namespace mra

#endif // HAVE_DIMS_H