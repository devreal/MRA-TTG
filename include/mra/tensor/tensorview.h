#ifndef MRA_TENSORVIEW_H
#define MRA_TENSORVIEW_H

#include <algorithm>
#include <numeric>
#include <array>
#include <tuple>

#include "mra/misc/types.h"
#include "mra/misc/platform.h"
#include "mra/tensor/tensoriter.h"
#include "mra/tensor/sparsity.h"

namespace mra {


  // fwd-decl
  template<typename T, Dimension NDIM, template<typename, typename> typename Sparsity>
  class TensorView;

  template<typename T, Dimension NDIM>
  using DenseTensorView = TensorView<T, NDIM, DenseViewBase>;

  template<typename T, Dimension NDIM>
  using SparseTensorView = TensorView<T, NDIM, SparseArrayBase>;

  namespace detail {

    /**
     * Type trait to check for TensorView types
     */
    template<typename T>
    struct is_tensorview  : std::false_type { };

    template<typename T, Dimension NDIM, template<typename, typename> typename Sparsity>
    struct is_tensorview<TensorView<T, NDIM, Sparsity>> : std::true_type { };

    template<typename T>
    constexpr bool is_tensorview_v = is_tensorview<std::decay_t<T>>::value;

    /**
     * Type trait to check whether a type is a std::array.
     */
    template<typename T>
    struct is_std_array : std::false_type { };

    template<typename T, std::size_t N>
    struct is_std_array<std::array<T, N>> : std::true_type { };

    template<typename T>
    constexpr bool is_std_array_v = is_std_array<std::decay_t<T>>::value;

    template<typename T>
    struct tensor_view_ndim;

    template<typename T, Dimension NDIM, template<typename, typename> typename Sparsity>
    struct tensor_view_ndim<TensorView<T, NDIM, Sparsity>> : std::integral_constant<Dimension, NDIM> { };

    template<typename T>
    constexpr Dimension tensor_view_ndim_v = tensor_view_ndim<std::decay_t<T>>::value;

    template<typename T, typename Enabler = void>
    struct is_sparse_tensorview : std::false_type { };

    template<typename T, Dimension NDIM, template<typename, typename> typename Sparsity>
    struct is_sparse_tensorview<TensorView<T, NDIM, Sparsity>,
                                std::enable_if_t<is_sparsity_view_v<typename TensorView<T, NDIM, Sparsity>::sparsity_type>>>
    : std::true_type { };

  } // namespace detail

  namespace concepts {
    /**
     * Concept for a TensorView with NDIM dimensions.
     * The NDIM argument is optional to enforce a specific number of dimensions.
     */
    template<typename T, Dimension NDIM = T::ndim()>
    concept TensorView = mra::detail::is_tensorview_v<T> && (detail::tensor_view_ndim_v<T> == NDIM);

    /**
     * Concept for a dense TensorView with NDIM dimensions.
     */
    template<typename T, Dimension NDIM = T::ndim()>
    concept DenseTensorView = mra::detail::is_tensorview_v<T> && (detail::tensor_view_ndim_v<T> == NDIM) && !detail::is_sparse_tensorview<T>::value;

    /**
     * Concept for an array of TensorViews with NDIM dimensions and size N.
     */
    template<typename T, Dimension NDIM = T::value_type::ndim(), std::size_t N = std::tuple_size_v<T>>
    concept TensorViewArray = mra::detail::is_std_array_v<T> && TensorView<typename T::value_type, NDIM> && (T::value_type::ndim() == NDIM);


    /**
     * Concept for an array of TensorViews with NDIM dimensions and size N.
     */
    template<typename T, Dimension NDIM = T::value_type::ndim(), std::size_t N = std::tuple_size_v<T>>
    concept DenseTensorViewArray = mra::detail::is_std_array_v<T>
                                  && TensorView<typename T::value_type, NDIM>
                                  && (T::value_type::ndim() == NDIM)
                                  && !detail::is_sparse_tensorview<typename T::value_type>::value;

  } // namespace concepts


  namespace detail {
    template<Dimension I, typename Fn, typename... Args>
    SCOPE void foreach_idxs_impl(const concepts::TensorView auto& t, Fn&& fn, Args... args)
    {
      constexpr Dimension NDIM = std::decay_t<decltype(t)>::ndim();
#ifdef HAVE_DEVICE_ARCH
      /* distribute the last three dimensions across the z, y, x dimension of the block */
      if constexpr (I == NDIM-3) {
        for (size_type i = threadIdx.z; i < t.dim(I); i += blockDim.z) {
          foreach_idxs_impl<I+1>(t, std::forward<Fn>(fn), args..., i);
        }
      } else if constexpr (I == NDIM-2) {
        for (size_type i = threadIdx.y; i < t.dim(I); i += blockDim.y) {
          foreach_idxs_impl<I+1>(t, std::forward<Fn>(fn), args..., i);
        }
      } else if constexpr (I == NDIM-1) {
        for (size_type i = threadIdx.x; i < t.dim(I); i += blockDim.x) {
          fn(args..., i);
        }
      } else {
        /* general index (NDIM > 3)*/
        for (size_type i = 0; i < t.dim(I); ++i) {
          foreach_idxs_impl<I+1>(t, std::forward<Fn>(fn), args..., i);
        }
      }
#else  // HAVE_DEVICE_ARCH
      if constexpr (I < NDIM-1) {
        for (size_type i = 0; i < t.dim(I); ++i) {
          foreach_idxs_impl<I+1>(t, std::forward<Fn>(fn), args..., i);
        }
      } else {
        for (size_type i = 0; i < t.dim(I); ++i) {
          fn(args..., i);
        }
      }
#endif // HAVE_DEVICE_ARCH
      SYNCTHREADS();
    }
  } // namespace detail

  /* invoke fn for each NDIM index set */
  template<typename Fn>
  SCOPE void foreach_idxs(const concepts::TensorView auto& t, Fn&& fn) {
    detail::foreach_idxs_impl<0>(t, std::forward<Fn>(fn));
  }

  /* invoke fn for each flat index */
  template<typename Fn>
  SCOPE void foreach_idx(const concepts::TensorView auto& t, Fn&& fn) {
    size_type tid = thread_id();
    for (size_type i = tid; i < t.size(); i += block_size()) {
      fn(i);
    }
    SYNCTHREADS();
  }

  /**
   * Scans indices [0,N) of the given view(s) and returns the real index of
   * the `ordinal`-th one (0-based) where at least one view is non-zero.
   * This lets a kernel map a compacted launch position (e.g. blockIdx.x,
   * ranging over only the non-zero count) back to the real function index
   * by reading the per-function sparsity bytes each view already carries --
   * no separate host-computed index array is needed.
   */
  template<typename... Views>
  SCOPE size_type find_nth_nonzero(size_type N, size_type ordinal, const Views&... views) {
    size_type seen = 0;
    for (size_type i = 0; i < N; ++i) {
      if ((!views.is_zero(i) || ...)) {
        if (seen == ordinal) return i;
        ++seen;
      }
    }
    return N; // unreachable: caller guarantees ordinal < n_nonzero
  }

  namespace detail {

    template <typename TensorT, Dimension NDIM>
    struct base_tensor_iterator {
      size_type count;
      const TensorT& t;
      std::array<size_type, std::max(Dimension(1), NDIM)> indx = {};

      constexpr base_tensor_iterator (size_type count, const TensorT& t)
      : count(count)
      , t(t)
      {}

      void inc() {
        assert(count < t.size());
        count++;
        for (int d=int(NDIM)-1; d>=0; --d) { // must be signed loop variable!
          indx[d]++;
          if (indx[d]<t.dim(d)) {
            break;
          } else {
            indx[d] = 0;
          }
        }
      }

      const auto& index() const {return indx;}
    };
  } // namespace detail


  class Slice {
  public:
    using size_type = int;
    static constexpr size_type END = std::numeric_limits<size_type>::max();
    size_type start;  //< Start of slice (must be signed type)
    size_type finish; //< Exclusive end of slice (must be signed type)
    size_type stride;   //< Stride for slice (must be signed type)
    size_type count;  //< Number of elements in slice (not known until dimension is applied; negative indicates not computed)

    SCOPE Slice() : start(0), finish(END), stride(1), count(-1) {}; // indicates entire range
    SCOPE Slice(size_type start) : start(start), finish(start+1), stride(1) {} // a single element
    SCOPE Slice(size_type start, size_type end, size_type stride=1) : start(start), finish(end), stride(stride) {};

    /// Once we know the dimension we adjust the start/end/count/finish to match, and do sanity checks
    SCOPE void apply_dim(size_type dim) {
        if (start == END) {start = dim-1;}
        else if (start < 0) {start += dim;}

        if (finish == END && stride > 0) {finish = dim;}
        else if (finish == END && stride < 0) {finish = -1;}
        else if (finish < 0) {finish += dim;}

        count = std::max(size_type(0),((finish-start-stride/std::abs(stride))/stride+1));
        assert((count==0) || ((count<=dim) && (start>=0 && start<=dim)));
        finish = start + count*stride; // finish is one past the last element
    }

    struct iterator {
      size_type value;
      const size_type stride;
      iterator (size_type value, size_type stride) : value(value), stride(stride) {}
      operator size_type() const {return value;}
      size_type operator*() const {return value;}
      iterator& operator++ () {value+=stride; return *this;}
      bool operator!=(const iterator&other) {return value != other.value;}
    };

    iterator begin() const {assert(count>=0); return iterator(start,stride); }
    iterator end() const {assert(count>=0); return iterator(finish,stride); }

    SCOPE Slice& operator=(const Slice& other) {
      if (this != &other) {
        start = other.start;
        finish = other.finish;
        stride = other.stride;
        count = other.count;
      }
      return *this;
    }
  }; // Slice



  template<concepts::TensorView TV>
  class TensorSlice {

  public:
    using view_type = TV;
    using value_type = typename view_type::value_type;
    using const_value_type = typename view_type::const_value_type;
    using sparsity_type = typename view_type::sparsity_type;

    SCOPE static constexpr Dimension ndim() { return TV::ndim(); }

    SCOPE static constexpr bool is_tensor() { return true; }

  private:
    value_type* m_ptr;
    std::array<Slice, ndim()> m_slices;

    // Computes index in dimension d for underlying tensor using slice info

    template<std::size_t I, std::size_t... Is, typename Arg, typename... Args>
    SCOPE size_type offset_helper(std::index_sequence<I, Is...>, Arg arg, Args... args) const {
      size_type idx = (m_slices[I].start + arg)*m_slices[I].stride;
      if constexpr (sizeof...(Args) > 0) {
        idx += offset_helper(std::index_sequence<Is...>{}, std::forward<Args>(args)...);
      }
      return idx;
    }

    template<typename Fn, typename... Args, std::size_t I, std::size_t... Is>
    SCOPE void last_level_op_helper(Fn&& fn, std::index_sequence<I, Is...>, Args... args) {
      if constexpr (sizeof...(Is) == 0) {
        fn(args...);
      } else {
        /* iterate over this dimension and recurse down one */
        for (std::size_t i = 0; i < m_slices[I].count; ++i) {
          last_level_op_helper(std::forward<Fn>(fn), std::index_sequence<Is...>{}, args..., i);
        }
      }
    }

    SCOPE size_type offset(size_type i) const {
      size_type offset = 0;
      size_type idx    = i;
      for (int d = ndim()-1; d >= 0; --d) {
        offset += ((idx%m_slices[d].count)+m_slices[d].start)*m_slices[d].stride;
        idx    /= m_slices[d].count;
      }
      return offset;
    }

  public:
    TensorSlice() = delete; // slice is useless without a view

    SCOPE TensorSlice(view_type& view, const std::array<Slice,ndim()>& slices)
    : m_ptr(view.data())
    , m_slices(slices)
    {
      /* adjust the slice dimensions to the tensor */
      auto view_slices = view.slices();
      size_type stride = 1;
      for (ssize_type d = ndim()-1; d >= 0; --d) {
        m_slices[d].apply_dim(view.dim(d));
        /* stride stores the stride in the original TensorView */
        m_slices[d].stride *= stride;
        stride *= view.dim(d);
        /* account for the stride of the underlying view */
        m_slices[d].stride *= view_slices[d].stride;
        /* adjust the start relative to the underlying view */
        m_slices[d].start += view_slices[d].start * view_slices[d].stride;
      }
    }

    TensorSlice(TensorSlice&& other) = default;
    TensorSlice(const TensorSlice& other) = default;

    /// Returns the base pointer
    SCOPE value_type* data() {
      return m_ptr;
    }

    /// Returns the const base pointer
    SCOPE const value_type* data() const {
      return m_ptr;
    }

    /// Returns number of elements in the tensor at runtime
    SCOPE size_type size() const {
      size_type nelem = 1;
      for (size_type d = 0; d < ndim(); ++d) {
          nelem *= m_slices[d].count;
      }
      return nelem;
    }

    /// Returns size of dimension d at runtime
    SCOPE size_type dim(size_type d) const { return m_slices[d].count; }

    /// Returns array containing size of each dimension at runtime
    SCOPE std::array<size_type, ndim()> dims() const {
      std::array<size_type, ndim()> dimensions;
      for (size_type d = 0; d < ndim(); ++d) {
        dimensions[d] = m_slices[d].count;
      }
      return dimensions;
    }

    SCOPE std::array<Slice, ndim()> slices() const {
      return m_slices;
    }

    SCOPE value_type& operator[](size_type i) {
      return m_ptr[offset(i)];
    }

    SCOPE const_value_type& operator[](size_type i) const {
      return m_ptr[offset(i)];
    }

    template <typename...Args>
    SCOPE auto& operator()(Args...args) {
      static_assert(ndim() == sizeof...(Args), "TensorSlice number of indices must match dimension");
      return m_ptr[offset_helper(std::index_sequence_for<Args...>{}, std::forward<Args>(args)...)];
    }

    template <typename...Args>
    SCOPE const auto& operator()(Args...args) const {
      static_assert(ndim() == sizeof...(Args), "TensorSlice number of indices must match dimension");
      return m_ptr[offset_helper(std::index_sequence_for<Args...>{}, std::forward<Args>(args)...)];
    }

    /// Fill with scalar
    /// Device: assumes this operation is called by all threads in a block
    /// Host: assumes this operation is called by a single CPU thread
    template <typename X=TensorSlice<TV>>
    typename std::enable_if<!std::is_const_v<TensorSlice>,X&>::type
    SCOPE operator=(const value_type& value) {
      foreach_idx(*this, [&](size_type i){ this->operator[](i) = value; });
      return *this;
    }

    /// Scale by scalar
    /// Device: assumes this operation is called by all threads in a block
    /// Host: assumes this operation is called by a single CPU thread
    template <typename X=TensorSlice<TV>>
    typename std::enable_if<!std::is_const_v<TensorSlice>,X&>::type
    SCOPE operator*=(const value_type& value) {
      foreach_idx(*this, [&](size_type i){ this->operator[](i) *= value; });
      return *this;
    }


    /// Copy into patch
    /// Device: assumes this operation is called by all threads in a block
    /// Host: assumes this operation is called by a single CPU thread
    typename std::enable_if<!std::is_const_v<TV>,TensorSlice&>::type
    SCOPE operator=(const TensorSlice& other) {
      foreach_idx(*this, [&](size_type i){ this->operator[](i) = other[i]; });
      return *this;
    }

    /// Accumulate into patch
    /// Device: assumes this operation is called by all threads in a block
    /// Host: assumes this operation is called by a single CPU thread
    typename std::enable_if<!std::is_const_v<TV>,TensorSlice&>::type
    SCOPE operator+=(const concepts::TensorView<TV::ndim()> auto& other) {
      foreach_idx(*this, [&](size_type i){ this->operator[](i) += other[i]; });
      return *this;
    }


    /// Copy into patch
    /// Defined below once we know TensorView
    /// Device: assumes this operation is called by all threads in a block
    /// Host: assumes this operation is called by a single CPU thread
    SCOPE TensorSlice& operator=(const concepts::DenseTensorView<TV::ndim()> auto& view);
  };


  namespace detail {

    template<typename TensorViewT>
    struct tensor_view_ndim<TensorSlice<TensorViewT>> : std::integral_constant<Dimension, TensorViewT::ndim()> { };
    template<typename TensorViewT>
    struct is_tensorview<TensorSlice<TensorViewT>> : std::true_type { };

  } // namespace detail



  template<typename T, Dimension NDIM, template<typename, typename> typename Sparsity>
  class TensorView : public Sparsity<TensorView<T, NDIM, Sparsity>, T> {
  public:
    using value_type = T;
    using const_value_type = std::add_const_t<value_type>;
    using sparsity_type = Sparsity<TensorView<T, NDIM, Sparsity>, T>;
    template<typename U, Dimension M>
    using subview_type = TensorView<U, M, DenseViewBase>;
    template<typename U, Dimension M>
    using const_subview_type = TensorView<const U, M, DenseViewBase>;
    SCOPE static constexpr Dimension ndim() { return NDIM; }
    using dims_array_t = std::array<size_type, ndim()>;
    SCOPE static constexpr bool is_tensor() { return true; }

    SCOPE static constexpr bool is_sparse() { return is_sparsity_view_v<sparsity_type>; }

  protected:

    template<size_type I, typename... Dims>
    SCOPE size_type offset_impl(size_type idx, Dims... idxs) const {
      /**
       * Special handling for first dimension to account for sparsity.
       */
      size_type offset = 0;
      if constexpr (I == 0) {
        if (this->is_zero(idx)) {
          /* no entry, return zero */
          return 0;
        } else {
          /* adjust the index based on which entries are allocated */
          offset = this->offset_of(idx);
        }
      } else {
        offset = idx*std::reduce(&m_dims[I+1], &m_dims[ndim()], 1, std::multiplies<size_type>{});
      }
      if constexpr (sizeof...(idxs) == 0) {
        return offset;
      } else {
        return offset + offset_impl<I+1>(std::forward<Dims>(idxs)...);
      }
    }

    template<typename... Dims>
    SCOPE auto subview_info(Dims... idxs) const {
      size_type offset = 0;
      if (this->data() != nullptr) {
        std::array<size_type, sizeof...(Dims)> indices = {static_cast<size_type>(idxs)...};
        for (size_type i = 0; i < indices.size(); ++i) {
          assert(indices[i] < dim(i));
        }
        offset = offset_impl<0>(std::forward<Dims>(idxs)...);
      }
      constexpr const Dimension noffs = sizeof...(Dims);
      constexpr const Dimension ndim = NDIM-noffs;
      std::array<size_type, ndim> dims;
      for (Dimension i = 0; i < ndim; ++i) {
        dims[i] = m_dims[noffs+i];
      }
      return std::make_pair(offset, dims);
    }

  public:
    TensorView() = default; // needed for __shared__ construction

    template<typename... Dims>
    SCOPE explicit TensorView(T *ptr, Dims... dims)
    : m_dims({dims...})
    , m_ptr(ptr)
    {
      static_assert(sizeof...(Dims) == NDIM || sizeof...(Dims) == 1,
                    "Number of arguments does not match number of Dimensions. "
                    "A single argument for all dimensions may be provided.");
      if constexpr (sizeof...(Dims) != NDIM) {
        std::fill(m_dims.begin(), m_dims.end(), dims...);
      }
    }

    SCOPE explicit TensorView(T *ptr, const dims_array_t& dims)
    : m_dims(dims)
    , m_ptr(ptr)
    { }

    template<typename S, typename... Dims>
    requires(!std::is_const_v<T> && std::is_same_v<S, T>)
    SCOPE explicit TensorView(const S *ptr, Dims... dims)
    : TensorView(const_cast<T*>(ptr), std::forward<Dims>(dims)...) // remove const, we store a non-const pointer internally
    { }

    template<typename S>
    requires(!std::is_const_v<T> && std::is_same_v<S, T>)
    SCOPE explicit TensorView(const S *ptr, const dims_array_t& dims)
    : TensorView(const_cast<T*>(ptr), dims) // remove const, we store a non-const pointer internally
    { }

    TensorView(TensorView&& other) = default;
    TensorView(const TensorView& other) = default;

    ~TensorView() = default;

    TensorView& operator=(TensorView&& other) = default;

#if 0
    /**
     * Overload to capture assignment of rvalue tensors with const T.
     */
    SCOPE TensorView& operator=(TensorView<const T, NDIM, Sparsity>&& other) requires(std::is_const_v<T>) {
      m_dims = other.m_dims;
      m_ptr = other.m_ptr;
      other.m_ptr = nullptr;
      return *this;
    }
#endif // 0

    /**
     * Move constructor for non-const TensorView from non-const TensorView.
     * A static assert protects from cases where the source is const and the T is not.
     * This overload catches assignments like:
     *    SHARED DenseTensorView<float, 3> b;
     *    a = const_input_tensor(0); // returns a subview of const_input_tensor which is of type TensorView<const float, 3>
     *
     * This overload is important to prevent accidental use of the copy-assignment operator `operator=(const TensorView auto& other)`,
     * which would try to assign values. We assume that move semantics are only used to construct tensors.
     */
    template<typename U = T>
    SCOPE TensorView& operator=(TensorView<U, NDIM, Sparsity>&& other) {
      static_assert(std::is_same_v<U, T>, "Can only move from TensorView of same type. Make sure source and destination have the same T.");
      m_dims = other.m_dims;
      m_ptr = other.m_ptr;
      other.m_ptr = nullptr;
      return *this;
    }

    /**
     * Returns the number of allocated elements in the tensor.
     */
    SCOPE size_type size() const {
      return this->count_allocated() * std::reduce(&m_dims[1], &m_dims[ndim()], 1, std::multiplies<size_type>{});
    }

    SCOPE size_type dim(Dimension d) const {
      return m_dims[d];
    }

    SCOPE const dims_array_t& dims() const {
      return m_dims;
    }

    /**
     * Returns true if the view has no allocated storage or zero size.
     */
    SCOPE bool empty() const {
      return this->data() == nullptr || this->size() == 0;
    }

    SCOPE size_type stride(size_type d) const {
      size_type s = 1;
      for (size_type i = d+1; i < ndim(); ++i) {
        s *= m_dims[i];
      }
      return s;
    }

    /**
     * Array-style flattened access.
     * Returns a reference to the i-th allocated element.
     **/
    SCOPE value_type& operator[](size_type i) {
      if (this->data() == nullptr) THROW("TensorView: non-const call with nullptr");
      return this->data()[i];
    }

    /**
     * Array-style flattened access.
     * Returns a reference to the i-th allocated element.
     **/
    SCOPE const_value_type operator[](size_type i) const {
      if (this->data() == nullptr) return const_value_type{};
      return this->data()[i];
    }

    /* return the offset for the provided indexes */
    template<typename... Dims>
    requires(sizeof...(Dims) == NDIM && (std::is_integral_v<Dims>&&...))
    SCOPE size_type offset(Dims... idxs) const {
      return offset_impl<0>(std::forward<Dims>(idxs)...);
    }

    /* access host-side elements */
    template<typename... Dims>
    requires(!std::is_const_v<std::remove_reference_t<T>> && sizeof...(Dims) == NDIM && (std::is_integral_v<Dims>&&...))
    SCOPE value_type& operator()(Dims... idxs) {
      std::array<size_type, sizeof...(Dims)> indices = {static_cast<size_type>(idxs)...};
      for (size_type i = 0; i < indices.size(); ++i) {
        assert(indices[i] < dim(i));
      }
      if (this->data() == nullptr) THROW("TensorView: non-const call with nullptr");
      if (is_sparse() && this->is_zero(indices[0])) {
        THROW("TensorView: non-const attempt to access unallocated sparse subview or element");
      }
      return this->data()[offset(std::forward<Dims>(idxs)...)];
    }

    /* access host-side elements */
    template<typename... Dims>
    requires(sizeof...(Dims) == NDIM && (std::is_integral_v<Dims>&&...))
    SCOPE const_value_type operator()(Dims... idxs) const {
      if (this->data() == nullptr) {
        return T{};
      } else {
        // let's hope the compiler will hoist this out of loops
        std::array<size_type, sizeof...(Dims)> indices = {static_cast<size_type>(idxs)...};
        if (is_sparse() && this->is_zero(indices[0])) {
          return T{};
        }
        for (size_type i = 0; i < indices.size(); ++i) {
          assert(indices[i] < dim(i));
        }
        return this->data()[offset(std::forward<Dims>(idxs)...)];
      }
    }

    /**
     * Return a TensorView<T, (NDIM-M)> to a subview using the provided first M indices.
     */
    template<typename... Dims>
    requires(sizeof...(Dims) < NDIM-1 && (std::is_integral_v<Dims>&&...) && !std::is_const_v<T>)
    SCOPE subview_type<T, NDIM-sizeof...(Dims)-1> operator()(size_type idx0, Dims... idxs) {
      constexpr Dimension ndim = NDIM - sizeof...(Dims) - 1;
      auto [offset, dims] = subview_info(idx0, std::forward<Dims>(idxs)...);
      if (this->is_zero(idx0)) {
        if constexpr (!std::is_const_v<T>) {
          THROW("TensorView: non-const attempt to access unallocated sparse subview or element");
        }
        return subview_type<T, ndim>(nullptr, dims); // return a view with nullptr data
      }
      return subview_type<T, ndim>(&(this->data()[offset]), dims);
    }

    template<typename... Dims>
    requires(sizeof...(Dims) < NDIM-1 && (std::is_integral_v<Dims>&&...))
    SCOPE const_subview_type<T, NDIM-sizeof...(Dims)-1> operator()(size_type idx0, Dims... idxs) const {
      auto [offset, dims] = subview_info(idx0, std::forward<Dims>(idxs)...);
      constexpr Dimension ndim = NDIM - sizeof...(Dims) - 1;
      if (this->m_ptr == nullptr || this->is_zero(idx0)) {
        return const_subview_type<T, ndim>(nullptr, dims); // return a view with nullptr data
      }
      return const_subview_type<T, ndim>(&(this->data()[offset]), dims);
    }

    SCOPE std::array<Slice, ndim()> slices() const {
      std::array<Slice, ndim()> res;
      for (int d = 0; d < ndim(); ++d) {
        res[d] = Slice(0, m_dims[d]);
      }
      return res;
    }

    /// Fill with scalar
    /// Device: assumes this operation is called by all threads in a block, synchronizes
    /// Host: assumes this operation is called by a single CPU thread
    SCOPE TensorView& operator=(const value_type& value) requires(!std::is_const_v<T>) {
      if (this->data() == nullptr) THROW("TensorView: non-const call with nullptr");
      foreach_idx(*this, [&](size_type i){ this->operator[](i) = value; });
      return *this;
    }

    /// Scale by scalar
    /// Device: assumes this operation is called by all threads in a block, synchronizes
    /// Host: assumes this operation is called by a single CPU thread
    SCOPE TensorView& operator*=(const value_type& value) requires(!std::is_const_v<T>) {
      if (this->data() == nullptr) THROW("TensorView: non-const call with nullptr");
      foreach_idx(*this, [&](size_type i){ this->operator[](i) *= value; });
      return *this;
    }

    /// Add another tensor
    /// Device: assumes this operation is called by all threads in a block, synchronizes
    /// Host: assumes this operation is called by a single CPU thread
    SCOPE TensorView& operator+=(const concepts::DenseTensorView<NDIM> auto& value) requires(!std::is_const_v<T>) {
      if (this->data() == nullptr) THROW("TensorView: non-const call with nullptr");
      foreach_idx(*this, [&](size_type i){ this->operator[](i) += value[i]; });
      return *this;
    }


    /// Copy into patch
    /// Device: assumes this operation is called by all threads in a block, synchronizes
    /// Host: assumes this operation is called by a single CPU thread
    SCOPE TensorView& operator=(const TensorView& other) requires(!std::is_const_v<T>) {
      if (this->data() == nullptr) THROW("TensorView: non-const call with nullptr");
      if (other.data() == nullptr) {
        foreach_idx(*this, [&](size_type i){ this->operator[](i) = value_type{}; });
      } else {
        foreach_idx(*this, [&](size_type i){ this->operator[](i) = other[i]; });
      }
      return *this;
    }


    /// Copy into patch
    /// Device: assumes this operation is called by all threads in a block, synchronizes
    /// Host: assumes this operation is called by a single CPU thread
    SCOPE TensorView& operator=(const concepts::DenseTensorView<NDIM> auto& other) requires(!std::is_const_v<T>) {
      if (this->data() == nullptr) THROW("TensorView: non-const call with nullptr");
      if (other.data() == nullptr) {
        foreach_idx(*this, [&](size_type i){ this->operator[](i) = value_type{}; });
      } else {
        foreach_idx(*this, [&](size_type i){ this->operator[](i) = other[i]; });
      }
      return *this;
    }

    /**
     * Returns the underlying storage pointer. This pointer should not
     * be used directly except for low-level operations as the storage may contain
     * sparsity information. Use the sparsity base class data() methods instead.
     */
    SCOPE value_type* storage() {
      return m_ptr;
    }

    SCOPE const value_type* storage() const {
      return m_ptr;
    }


    /// Copy into patch
    /// Device: assumes this operation is called by all threads in a block, synchronizes
    /// Host: assumes this operation is called by a single CPU thread
    template<typename TensorViewT>
    SCOPE TensorView& operator=(const TensorSlice<TensorViewT>& other) {
      if (this->data() == nullptr) THROW("TensorView: non-const call with nullptr");
      foreach_idx(*this, [&](size_type i){ this->operator[](i) = other[i]; });
      return *this;
    }

    SCOPE void reduce_rank(const T& eps) {return;}

    SCOPE TensorSlice<TensorView> operator()(const std::array<Slice, NDIM>& slices) {
      if (this->data() == nullptr) THROW("TensorView: non-const call with nullptr");
      return TensorSlice<TensorView>(*this, slices);
    }

    SCOPE TensorSlice<TensorView> get_slice(const std::array<Slice, NDIM>& slices) {
      if (this->data() == nullptr) THROW("TensorView: non-const call with nullptr");
      return TensorSlice<TensorView>(*this, slices);
    }


    template<Dimension ndimactive>
    struct iterator : public detail::base_tensor_iterator<TensorView,ndimactive> {
      iterator (size_type count, TensorView& t)
      : detail::base_tensor_iterator<TensorView,ndimactive>(count, t)
      { }
      auto& operator*() { return this->t.data()[this->count]; }
      iterator& operator++() {this->inc(); return *this;}
      bool operator!=(const iterator& other) {return this->count != other.count;}
      bool operator==(const iterator& other) {return this->count == other.count;}
    };

    template<Dimension ndimactive>
    struct const_iterator : public detail::base_tensor_iterator<TensorView,ndimactive> {
      const_iterator (size_type count, const TensorView& t)
      : detail::base_tensor_iterator<TensorView,ndimactive>(count, t)
      { }
      value_type operator*() const { return this->t.data()[this->count]; }
      const_iterator& operator++() {this->inc(); return *this;}
      bool operator!=(const const_iterator& other) {return this->count != other.count;}
      bool operator==(const const_iterator& other) {return this->count == other.count;}
    };


    /// Start for forward iteration through elements in row-major order --- this is convenient but not efficient
    iterator<ndim()> begin() {return iterator<ndim()>(0, *this);}

    /// End for forward iteration through elements in row-major order --- this is convenient but not efficient
    const iterator<ndim()> end() { return iterator<ndim()>(size(), *this); }

    /// Start for forward iteration through elements in row-major order --- this is convenient but not efficient
    const_iterator<ndim()> begin() const { return const_iterator<ndim()>(0, *this); }

    /// End for forward iteration through elements in row-major order --- this is convenient but not efficient
    const const_iterator<ndim()> end() const { return const_iterator<ndim()>(size(), *this); }

    SCOPE TensorIterator<TensorView> unary_iterator(IterLevel iterlevel, ssize_type jdim = TensorIterator<TensorView>::default_jdim) {
      return {this, iterlevel, jdim};
    }

    SCOPE TensorIterator<const TensorView> unary_iterator(IterLevel iterlevel, ssize_type jdim = TensorIterator<TensorView>::default_jdim) const {
      return {this, iterlevel, jdim};
    }

    SCOPE TensorIterator<TensorView> unary_iterator(IterLevel iterlevel, bool fusedim, ssize_type jdim = TensorIterator<TensorView>::default_jdim) {
      return {this, iterlevel, jdim};
    }

    SCOPE TensorIterator<const TensorView> unary_iterator(IterLevel iterlevel, bool fusedim, ssize_type jdim = TensorIterator<TensorView>::default_jdim) const {
      return {this, iterlevel, jdim};
    }

  private:
    dims_array_t m_dims;
    T *m_ptr; // may be const or non-const
  };

#if 0
    template<concepts::TensorView U>
    requires(U::ndim() == ndim())
    SCOPE TensorSlice& operator=(const U& view);
#endif // 0

  template<concepts::TensorView TV>
  SCOPE TensorSlice<TV>& TensorSlice<TV>::operator=(
    const concepts::DenseTensorView<TV::ndim()> auto& view)
  {
    foreach_idx(*this, [&](size_type i){ this->operator[](i) = view[i]; });
    return *this;
  }

} // namespace mra

#endif // MRA_TENSORVIEW_H
