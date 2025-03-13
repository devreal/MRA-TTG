#ifndef MRA_TENSOR_H
#define MRA_TENSOR_H

#include <algorithm>
#include <numeric>
#include <array>

#include <ttg.h>
#include <ttg/serialization.h>
#include <ttg/serialization/std/array.h>

#include "mra/misc/allocator.h"
#include "mra/tensor/tensorview.h"

namespace mra {

  template<typename T, Dimension NDIM, class Allocator = DeviceAllocator<T>>
  class Tensor : public ttg::TTValue<Tensor<T, NDIM, Allocator>> {
  public:
    using value_type = std::decay_t<T>;
    using allocator_type = Allocator;
    using view_type = TensorView<value_type, NDIM>;
    using const_view_type = std::add_const_t<TensorView<value_type, NDIM>>;
    using buffer_type = ttg::Buffer<value_type, allocator_type>;

    static constexpr bool is_tensor() { return true; };

    static constexpr Dimension ndim() { return NDIM; }

    using dims_array_t = std::array<size_type, ndim()>;

    //template<typename Archive>
    //friend madness::archive::ArchiveSerializeImpl<Archive, Tensor>;

  private:
    using ttvalue_type = ttg::TTValue<Tensor<T, NDIM, Allocator>>;
    dims_array_t m_dims = {0};
    buffer_type  m_buffer;

    // (Re)allocate the tensor memory
    void realloc() {
      m_buffer.reset(size());
    }

    template<std::size_t... Is>
    static auto create_dims_array(size_type dim, std::index_sequence<Is...>) {
      return std::array{((void)Is, dim)...};
    }

  public:
    Tensor() = default;

    /* generic */
    explicit Tensor(size_type dim, ttg::scope scope = ttg::scope::SyncIn)
    : ttvalue_type()
    , m_dims(create_dims_array(dim, std::make_index_sequence<NDIM>{}))
    , m_buffer(size())
    { }

    template<typename... Dims, typename = std::enable_if_t<(sizeof...(Dims) > 1)>>
    Tensor(Dims... dims)
    : ttvalue_type()
    , m_dims({static_cast<size_type>(dims)...})
    , m_buffer(size())
    {
      static_assert(sizeof...(Dims) == NDIM,
                    "Number of arguments does not match number of Dimensions.");
    }

    Tensor(const std::array<size_type, NDIM>& dims, ttg::scope scope = ttg::scope::SyncIn)
    : ttvalue_type()
    , m_dims(dims)
    , m_buffer(size(), scope)
    {
      // TODO: make this static_assert (clang 14 doesn't get it)
      assert(dims.size() == NDIM);
                    //"Number of arguments does not match number of Dimensions.");
    }


    Tensor(Tensor<T, NDIM, Allocator>&& other) = default;

    Tensor& operator=(Tensor<T, NDIM, Allocator>&& other) = default;

    /* Disable copy construction.
     * There is no way we can copy data from anywhere else but the host memory space
     * so let's not even try. */
    Tensor(const Tensor<T, NDIM, Allocator>& other) = delete;

    Tensor& operator=(const Tensor<T, NDIM, Allocator>& other) = delete;

    size_type size() const {
      return std::reduce(&m_dims[0], &m_dims[ndim()], 1, std::multiplies<size_type>{});
    }

    size_type dim(Dimension dim) const {
      return m_dims[dim];
    }

    dims_array_t dims() const {
      return m_dims;
    }

    auto& buffer() {
      return m_buffer;
    }

    const auto& buffer() const {
      return m_buffer;
    }

    value_type* data() {
      return m_buffer.host_ptr();
    }

    const value_type* data() const {
      return m_buffer.host_ptr();
    }

    /* returns a view for the current memory space
     * TODO: handle const correctness (const Tensor should return a const TensorView)*/
    view_type current_view() {
      return view_type(m_buffer.current_device_ptr(), m_dims);
    }

    /* returns a view for the current memory space
     * TODO: handle const correctness (const Tensor should return a const TensorView)*/
    const view_type current_view() const {
      return view_type(m_buffer.current_device_ptr(), m_dims);
    }

    bool empty() const {
      return m_buffer.empty();
    }

    template <typename Archive>
    void serialize(Archive &ar) {
      ar &m_dims &m_buffer;
    }

    template <typename Archive>
    void serialize(Archive &ar, const unsigned int) {
      serialize(ar);
    }
  };

  template <typename T, Dimension NDIM>
  std::ostream&
  operator<<(std::ostream& s, const TensorView<T, NDIM>& t) {
    if (t.size() == 0) {
      s << "[empty tensor]\n";
      return s;
    }

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
      for (size_type i=0; i<lastdimsize; ++i,++it) { //<<< it incremented here!
        // s.precision(4);
        s << " ";
        //s.precision(8);
        //s.width(12);
        s.precision(16);
        s.width(20);
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

  template <typename tensorT>
  requires(tensorT::is_tensor())
  std::ostream&
  operator<<(std::ostream& s, const tensorT& t) {
    s << t.current_view();
    return s;
  }
} // namespace mra

#endif // MRA_TENSOR_H
