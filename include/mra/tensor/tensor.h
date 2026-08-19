#ifndef MRA_TENSOR_H
#define MRA_TENSOR_H

#include <algorithm>
#include <numeric>
#include <array>

#include <ttg.h>
#include <ttg/serialization.h>
#include <ttg/serialization/std/array.h>

#include "mra/misc/allocator.h"
#include "mra/tensor/dimensions.h"
#include "mra/tensor/tensorview.h"
#include "mra/tensor/sparsity.h"
#include "mra/tensor/sparsityinfo.h"

namespace mra {

  namespace detail {

    template<template<typename, typename> typename Sparsity>
    struct is_tensor_compatible_sparsity;

    template<>
    struct is_tensor_compatible_sparsity<RangeSparsityBase> : std::true_type { };

    template<>
    struct is_tensor_compatible_sparsity<DenseViewBase> : std::true_type { };

    template<template<typename, typename> typename Sparsity>
    concept TensorCompatibleSparsity = is_tensor_compatible_sparsity<Sparsity>::value;


    template<typename ValueType, Dimension NDIM, template<typename, typename> typename Sparsity>
    struct make_tensorview {
      using type = TensorView<ValueType, NDIM, Sparsity>;
    };

    /**
     * Change range-based sparsity to sparsity array for tensorview.
     */
    template<typename ValueType, Dimension NDIM>
    struct make_tensorview<ValueType, NDIM, RangeSparsityBase> {
      using type = TensorView<ValueType, NDIM, SparseArrayBase>;
    };

    template<typename ValueType, Dimension NDIM, template<typename, typename> typename Sparsity>
    using make_tensorview_t = typename make_tensorview<ValueType, NDIM, Sparsity>::type;

  } // namespace detail

  template<typename T, Dimension NDIM, template<typename, typename> typename Sparsity, class Allocator = DeviceAllocator<T>>
  requires detail::TensorCompatibleSparsity<Sparsity>
  class Tensor : public ttg::TTValue<Tensor<T, NDIM, Sparsity, Allocator>>,
                 protected Sparsity<Tensor<T, NDIM, Sparsity, Allocator>, T> {


    /**
     * The tensor is not publicly derived from the sparsity class, but it needs to be a friend
     * to access the protected functions for managing sparsity information.
     */
    friend Sparsity<Tensor<T, NDIM, Sparsity, Allocator>, T>;
  public:
    using value_type = std::decay_t<T>;
    using allocator_type = Allocator;
    using sparsity_type = Sparsity<Tensor<T, NDIM, Sparsity, Allocator>, T>;
    using view_type = detail::make_tensorview_t<value_type, NDIM, Sparsity>;
    using view_sparsity_type = typename view_type::sparsity_type;
    using const_view_type = std::add_const_t<view_type>;
    using buffer_type = ttg::Buffer<value_type, allocator_type>;

    static constexpr Dimension ndim() { return NDIM; }

    using dims_type = DynamicDimensions<NDIM>;

    //template<typename Archive>
    //friend madness::archive::ArchiveSerializeImpl<Archive, Tensor>;

  private:
    using ttvalue_type = ttg::TTValue<Tensor>;
    dims_type m_dims;
    buffer_type  m_buffer;

    std::size_t buffer_size() const {
      if constexpr (sparsity_traits<view_sparsity_type>::inline_storage()) {
        return size() + sparsity_traits<view_sparsity_type>::required_space(m_dims);
      } else {
        return size();
      }
    }

    // (Re)allocate the tensor memory
    void realloc() {
      m_buffer.reset(buffer_size());
    }

    template<std::size_t... Is>
    static auto create_dims_array(size_type dim, std::index_sequence<Is...>) {
      return std::array{((void)Is, dim)...};
    }

    template<std::size_t... Is>
    static auto create_dims_array(size_type dim0, size_type dim, std::index_sequence<Is...>) {
      return std::array{dim0, ((void)Is, dim)...};
    }

    template<typename S>
    static void populate_host_sparsity(S&& tensor) {
      if constexpr (sparsity_traits<view_sparsity_type>::is_sparse()) {

        // The TTG Buffer claims that empty buffers are current on all devices. Not sure that is correct...
        if (!tensor.buffer().empty() && tensor.buffer().is_current_on(ttg::device::Device::host())) {
          // cast away const so we can populate the sparsity information (which is logically mutable even in const tensors)
          // Write directly into the tensor's own host buffer -- no need to
          // stage through a pinned MockTensor buffer when the destination is
          // already host memory.
          auto& mutable_tensor = const_cast<std::decay_t<S>&>(tensor);
          detail::sparsity_to_bytes(mutable_tensor.sparsity(),
                                     reinterpret_cast<detail::SparsityState*>(mutable_tensor.buffer().host_ptr()),
                                     mutable_tensor.dim(0));
        }
      }
    }


  public:
    Tensor() = default;

    /* generic */
    explicit Tensor(size_type dim, ttg::scope scope = ttg::scope::SyncIn)
    : ttvalue_type()
    , sparsity_type()
    , m_dims(create_dims_array(dim, std::make_index_sequence<NDIM>{}))
    , m_buffer(buffer_size())
    { }

    template<typename... Dims, typename = std::enable_if_t<(sizeof...(Dims) > 1)>>
    Tensor(Dims... dims)
    : ttvalue_type()
    , sparsity_type()
    , m_dims({static_cast<size_type>(dims)...})
    , m_buffer(buffer_size())
    {

      static_assert(sizeof...(Dims) == NDIM,
                    "Number of arguments does not match number of Dimensions.");
    }

    Tensor(const std::array<size_type, NDIM>& dims, ttg::scope scope = ttg::scope::SyncIn)
    : ttvalue_type()
    , sparsity_type()
    , m_dims(dims)
    , m_buffer(buffer_size(), scope)
    {
      // TODO: make this static_assert (clang 14 doesn't get it)
      assert(dims.size() == NDIM);
    }

    Tensor(const SparsityInfo& sparsity_info,
           size_type K,
           ttg::scope scope = ttg::scope::SyncIn)
    : ttvalue_type()
    , sparsity_type()
    , m_dims(create_dims_array(sparsity_info.dim(0), K, std::make_index_sequence<NDIM-1>{}))
    , m_buffer(buffer_size(), scope)
    {
      this->apply_sparsity(sparsity_info);
    }

    Tensor(const SparsityInfo& sparsity_info,
           const std::array<size_type, NDIM>& dims,
           ttg::scope scope = ttg::scope::SyncIn)
    : ttvalue_type()
    , sparsity_type()
    , m_dims(dims)
    , m_buffer(buffer_size(), scope)
    {
      assert(sparsity_info.dim(0) == dims[0]);
      this->apply_sparsity(sparsity_info);
    }


    Tensor(Tensor&& other) = default;

    Tensor& operator=(Tensor&& other) = default;

    /* Disable copy construction.
     * There is no way we can copy data from anywhere else but the host memory space
     * so let's not even try. */
    Tensor(const Tensor& other) = delete;

    Tensor& operator=(const Tensor& other) = delete;

    size_type size() const {
      return m_dims.product();
    }

    size_type dim(Dimension dim) const {
      return m_dims[dim];
    }

    dims_type dims() const {
      return m_dims;
    }

    auto& buffer() {
      return m_buffer;
    }

    const auto& buffer() const {
      return m_buffer;
    }

    /* returns a view for the current memory space */
    view_type current_view() {
      populate_host_sparsity(*this);
      return view_type(m_buffer.current_device_ptr(), m_dims);
    }

    /* returns a view for the current memory space */
    const view_type current_view() const {
      populate_host_sparsity(*this);
      return view_type(m_buffer.current_device_ptr(), m_dims);
    }

    /* returns a view for the current memory space */
    view_type view_on(const ttg::device::Device& device) {
      if (!m_buffer.is_current_on(device)) {
        throw std::runtime_error("Tensor: view_on() called for device that does not have current data");
      }
      populate_host_sparsity(*this);
      return view_type(m_buffer.device_ptr_on(device), m_dims);
    }

    /* returns a view for the current memory space */
    const view_type view_on(const ttg::device::Device& device) const {
      if (!m_buffer.is_current_on(device)) {
        throw std::runtime_error("Tensor: view_on() called for device that does not have current data");
      }
      populate_host_sparsity(*this);
      return view_type(m_buffer.device_ptr_on(device), m_dims);
    }

    auto host_view() const {
      return view_on(ttg::device::Device::host());
    }

    bool empty() const {
      return size() == 0;
    }

    void clear() {
      m_buffer.clear();
      std::fill(m_dims.begin(), m_dims.end(), 0);
      /* clear out the sparsity */
      sparsity().reset();
    }

    template <typename Archive>
    void serialize(Archive &ar) {
      ar &m_dims &m_buffer & static_cast<sparsity_type&>(*this);
    }

    template <typename Archive>
    void serialize(Archive &ar, const unsigned int) {
      serialize(ar);
    }

    /**
     * Update sparsity information from another sparsity object.
     */
    template<typename S>
    void update_sparsity_info(S&& sparsity_info) {
      sparsity_type::apply_sparsity(std::forward<S>(sparsity_info));
    }

    sparsity_type& sparsity() {
      return *this;
    }

    const sparsity_type& sparsity() const {
      return *this;
    }
  };

  namespace concepts {
    template<typename T>
    struct is_tensor : std::false_type { };

    template<typename T, Dimension NDIM, template<typename, typename> typename Sparsity, class Allocator>
    struct is_tensor<Tensor<T, NDIM, Sparsity, Allocator>> : std::true_type { };

    template<typename T>
    constexpr bool is_tensor_v = is_tensor<std::decay_t<T>>::value;

    template<typename T>
    concept Tensor = is_tensor_v<T>;
  } // namespace concepts

  std::ostream&
  operator<<(std::ostream& s, const concepts::TensorView auto& t) {
    if (t.size() == 0) {
      s << "[empty tensor]\n";
      return s;
    }

    using view_type = std::decay_t<decltype(t)>;
    using T = typename view_type::value_type;
    constexpr const Dimension NDIM = view_type::ndim();

    const Dimension ndim = t.ndim();

    auto dims = t.dims();
    size_type maxdim = *std::max_element(dims.begin(), dims.end());
    size_type index_width = std::max(std::log10(maxdim), 6.0);
    std::ios::fmtflags oldflags = s.setf(std::ios::scientific);
    long oldprec = s.precision();
    long oldwidth = s.width();

    const Dimension lastdim = ndim-1;
    const size_type lastdimsize = t.dim(lastdim);
    for (auto it=t.begin(); it!=t.end(); ) {
      const auto& index = it.index();
      s.unsetf(std::ios::scientific);
      s << '[';
      for (Dimension d=0; d<(ndim-1); d++) {
        s.width(index_width);
        s << index[d];
        s << ",";
      }
      s << " *]";
      s.setf(std::ios::scientific);
      //s.setf(std::ios::fixed);
      if (t.empty()) {
        s << " [empty tensor]\n";
        break;
      }
      for (size_type i=0; i<lastdimsize; ++i,++it) { //<<< it incremented here!
        s.precision(4);
        s << " ";
        //s.precision(8);
        s.width(12);
        //s.precision(16);
        //s.width(20);
        s << *it;
      }
      s.unsetf(std::ios::scientific);
      if (it != t.end()) s << std::endl;
    }

    s.setf(oldflags,std::ios::floatfield);
    s.precision(oldprec);
    s.width(oldwidth);

    return s;
  }

  std::ostream&
  operator<<(std::ostream& s, const concepts::Tensor auto& t) {
    assert(t.buffer().is_current_on(ttg::device::Device::host()));
    s << t.current_view();
    return s;
  }

  template<typename T, mra::Dimension NDIM, class Allocator = DeviceAllocator<T>>
  using DenseTensor = Tensor<T, NDIM, DenseViewBase, Allocator>;

  template<typename T, mra::Dimension NDIM, class Allocator = DeviceAllocator<T>>
  using SparseTensor = Tensor<T, NDIM, RangeSparsityBase, Allocator>;

} // namespace mra

#endif // MRA_TENSOR_H
