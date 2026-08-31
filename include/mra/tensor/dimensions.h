#ifndef MRA_TENSORDIMS_H
#define MRA_TENSORDIMS_H

#include "mra/misc/integer.h"
#include "mra/misc/types.h"

namespace mra {

  /**
   * Mixed dynamic/compile-time representation of tensor dimensions.
   * This is used to represent the dimensions of a TensorView, which may be a mix of compile-time and runtime dimensions.
   */
  template<typename... Dims>
  struct Dimensions {
  private:
    using tuple_type = std::tuple<Dims...>;
    tuple_type m_dims;
  public:

    Dimensions() = default;

    SCOPE Dimensions(Dims... dims) : m_dims(dims...) {}

    SCOPE Dimensions(std::tuple<Dims...> dims) : m_dims(dims) {}

    SCOPE std::array<size_type, sizeof...(Dims)> array() const {
      std::array<size_type, sizeof...(Dims)> dims_array;
      std::apply([&dims_array](auto&&... args) {
        size_type i = 0;
        ((dims_array[i++] = args), ...);
      }, m_dims);
      return dims_array;
    }

    SCOPE constexpr size_type size() const {
      return std::tuple_size_v<decltype(m_dims)>;
    }

    SCOPE constexpr size_type ndim() const {
      return std::tuple_size_v<decltype(m_dims)>;
    }

    SCOPE constexpr size_type product() const {
      size_type prod = 1;
      std::apply([&prod](auto&&... args) {
        ((prod *= args), ...);
      }, m_dims);
      return prod;
    }

    SCOPE size_type operator[](size_type d) const {
      return array()[d];
    }

    SCOPE size_type dim(size_type d) const {
      return array()[d];
    }

    template<size_type Start>
    SCOPE auto subdims(Int<Start>) const {
      return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                  using result_type = Dimensions<std::tuple_element_t<Is + Start, tuple_type>...>;
                  return result_type(std::get<Is + Start>(m_dims)...);
                }(std::make_index_sequence<sizeof...(Dims) - Start>{});
    }

    template<size_type I>
    SCOPE std::tuple_element_t<I, std::tuple<Dims...>> dim() const {
      return std::get<I>(m_dims);
    }
  };

  /**
   * Overload for dynamic-only dimensions (i.e., all dimensions are runtime values).
   */
  template<size_type NDIM>
  struct DynamicDimensions {

  private:
    std::array<size_type, NDIM> m_dims;
  public:

    DynamicDimensions() = default;

    SCOPE DynamicDimensions(std::initializer_list<size_type> dims) {
      assert(dims.size() == NDIM);
      std::copy(dims.begin(), dims.end(), m_dims.begin());
    }

    template<typename... Dims>
    requires(sizeof...(Dims) <= NDIM && (std::is_integral_v<std::decay_t<Dims>>&&...))
    SCOPE DynamicDimensions(Dims... dims) : m_dims({dims...}) {}

    SCOPE DynamicDimensions(const std::array<size_type, NDIM>& dims) : m_dims(dims) {}

    SCOPE const std::array<size_type, NDIM>& array() const {
      return m_dims;
    }

    SCOPE constexpr size_type ndim() const {
      return NDIM;
    }

    SCOPE constexpr size_type size() const {
      return NDIM;
    }

    SCOPE size_type product() const {
      return std::accumulate(m_dims.begin(), m_dims.end(), size_type(1), std::multiplies<size_type>{});
    }

    SCOPE size_type dim(size_type d) const {
      return m_dims[d];
    }

    SCOPE size_type operator[](size_type d) const {
      return m_dims[d];
    }

    auto begin() { return m_dims.begin(); }
    auto end() { return m_dims.end(); }

    template<size_type Start>
    SCOPE auto subdims(Int<Start>) const {
      static_assert(Start <= NDIM, "Start index must not be larger than NDIM");
      std::array<size_type, NDIM - Start> subdims;
      if constexpr (Start < NDIM) {
        for (size_type i = 0; i < NDIM - Start; ++i) {
          subdims[i] = m_dims[i + Start];
        }
      }
      return DynamicDimensions<NDIM - Start>(subdims);
    }

    template<size_type I>
    SCOPE size_type dim() const {
      return std::get<I>(m_dims);
    }

    template <typename Archive>
    void serialize(Archive &ar) {
      ar & m_dims;
    }

    template <typename Archive>
    void serialize(Archive &ar, const unsigned int) {
      serialize(ar);
    }

  };

  template<typename T>
  struct is_tensor_dims : std::false_type { };
  template<typename... Dims>
  struct is_tensor_dims<Dimensions<Dims...>> : std::true_type { };
  template<std::size_t NDIM>
  struct is_tensor_dims<DynamicDimensions<NDIM>> : std::true_type { };

  template<typename T>
  constexpr bool is_tensor_dims_v = is_tensor_dims<std::decay_t<T>>::value;

  template<typename T>
  struct is_ct_tensor_dims : std::false_type { };
  template<typename... Dims>
  struct is_ct_tensor_dims<Dimensions<Dims...>> : std::conjunction<is_ct_integral<Dims>...> { };
  template<typename T>
  constexpr bool is_ct_tensor_dims_v = is_ct_tensor_dims<std::decay_t<T>>::value;



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
  SCOPE constexpr auto make_dims(Sizes&&... sizes) {
    static_assert(sizeof...(Sizes) <= NDIM, "Too many sizes provided for the number of dimensions");
    if constexpr ((is_ct_integral_v<Sizes> || ...)) {
      // at least one compile-time size provided, use the last size as the padding value
      return Dimensions([&]<std::size_t... Is>(auto first, std::index_sequence<Is...>) {
                return std::tuple_cat(first, std::make_tuple(((void)Is, std::get<sizeof...(Sizes)-1>(first))...));
              }(std::make_tuple(sizes...), std::make_index_sequence<NDIM-sizeof...(Sizes)>{}));
    } else {
      // all runtime sizes provided, return an array with the last size as the padding value
      return DynamicDimensions<NDIM>(detail::make_dims_helper<NDIM>(std::make_index_sequence<NDIM-sizeof...(Sizes)>{}, std::forward<Sizes>(sizes)...));
    }
  }

}
#endif // MRA_TENSORDIMS_H
