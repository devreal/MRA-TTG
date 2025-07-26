#ifndef MRA_TENSORVIEW_H
#define MRA_TENSORVIEW_H

#include <algorithm>
#include <numeric>
#include <array>

#include "mra/misc/types.h"
#include "mra/misc/platform.h"
#include "mra/tensor/tensoriter.h"

namespace mra {

  namespace detail {
    template<Dimension NDIM, Dimension I, typename TensorViewT, typename Fn, typename... Args>
    SCOPE void foreach_idxs_impl(const TensorViewT& t, Fn&& fn, Args... args)
    {
#ifdef HAVE_DEVICE_ARCH
      /* distribute the last three dimensions across the z, y, x dimension of the block */
      if constexpr (I == NDIM-3) {
        for (size_type i = threadIdx.z; i < t.dim(I); i += blockDim.z) {
          foreach_idxs_impl<NDIM, I+1>(t, std::forward<Fn>(fn), args..., i);
        }
      } else if constexpr (I == NDIM-2) {
        for (size_type i = threadIdx.y; i < t.dim(I); i += blockDim.y) {
          foreach_idxs_impl<NDIM, I+1>(t, std::forward<Fn>(fn), args..., i);
        }
      } else if constexpr (I == NDIM-1) {
        for (size_type i = threadIdx.x; i < t.dim(I); i += blockDim.x) {
          fn(args..., i);
        }
      } else {
        /* general index (NDIM > 3)*/
        for (size_type i = 0; i < t.dim(I); ++i) {
          foreach_idxs_impl<NDIM, I+1>(t, std::forward<Fn>(fn), args..., i);
        }
      }
#else  // HAVE_DEVICE_ARCH
      if constexpr (I < NDIM-1) {
        for (size_type i = 0; i < t.dim(I); ++i) {
          foreach_idxs_impl<NDIM, I+1>(t, std::forward<Fn>(fn), args..., i);
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
  template<class TensorViewT, typename Fn>
  requires(TensorViewT::is_tensor())
  SCOPE void foreach_idxs(const TensorViewT& t, Fn&& fn) {
    constexpr mra::Dimension NDIM = TensorViewT::ndim();
    detail::foreach_idxs_impl<NDIM, 0>(t, std::forward<Fn>(fn));
  }

  /* invoke fn for each flat index */
  template<class TensorViewT, typename Fn>
  requires(TensorViewT::is_tensor())
  SCOPE void foreach_idx(const TensorViewT& t, Fn&& fn) {
    size_type tid = thread_id();
    for (size_type i = tid; i < t.size(); i += block_size()) {
      fn(i);
    }
    SYNCTHREADS();
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


  // fwd-decl
  template<typename T, Dimension NDIM>
  class TensorView;

  template<typename TensorViewT>
  class TensorSlice {

  public:
    using view_type = TensorViewT;
    using value_type = typename view_type::value_type;
    using const_value_type = typename view_type::const_value_type;

    SCOPE static constexpr Dimension ndim() { return TensorViewT::ndim(); }

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
    SCOPE TensorSlice() = delete; // slice is useless without a view

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

    SCOPE TensorSlice(TensorSlice&& other) = default;
    SCOPE TensorSlice(const TensorSlice& other) = default;

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
    template <typename X=TensorSlice<TensorViewT>>
    typename std::enable_if<!std::is_const_v<TensorSlice>,X&>::type
    SCOPE operator=(const value_type& value) {
      foreach_idx(*this, [&](size_type i){ this->operator[](i) = value; });
      return *this;
    }

    /// Scale by scalar
    /// Device: assumes this operation is called by all threads in a block
    /// Host: assumes this operation is called by a single CPU thread
    template <typename X=TensorSlice<TensorViewT>>
    typename std::enable_if<!std::is_const_v<TensorSlice>,X&>::type
    SCOPE operator*=(const value_type& value) {
      foreach_idx(*this, [&](size_type i){ this->operator[](i) *= value; });
      return *this;
    }

    /// Copy into patch
    /// Device: assumes this operation is called by all threads in a block
    /// Host: assumes this operation is called by a single CPU thread
    typename std::enable_if<!std::is_const_v<TensorViewT>,TensorSlice&>::type
    SCOPE operator=(const TensorSlice& other) {
      foreach_idx(*this, [&](size_type i){ this->operator[](i) = other[i]; });
      return *this;
    }

    /// Copy into patch
    /// Defined below once we know TensorView
    /// Device: assumes this operation is called by all threads in a block
    /// Host: assumes this operation is called by a single CPU thread
    std::enable_if_t<!std::is_const_v<TensorViewT>, TensorSlice&>
    SCOPE operator=(const TensorViewT& view);
  };



  template<typename T, Dimension NDIM>
  class TensorView {
  public:
    using value_type = T;
    using const_value_type = std::add_const_t<value_type>;
    SCOPE static constexpr Dimension ndim() { return NDIM; }
    using dims_array_t = std::array<size_type, ndim()>;
    SCOPE static constexpr bool is_tensor() { return true; }

  protected:

    template<size_type I, typename... Dims>
    SCOPE size_type offset_impl(size_type idx, Dims... idxs) const {
      size_type offset = idx*std::reduce(&m_dims[I+1], &m_dims[ndim()], 1, std::multiplies<size_type>{});
      if constexpr (sizeof...(Dims) > 0) {
        return offset + offset_impl<I+1>(std::forward<Dims>(idxs)...);
      }
      return offset;
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

    SCOPE TensorView(TensorView<T, NDIM>&& other) = default;
    SCOPE TensorView(const TensorView<T, NDIM>& other) = default;

    SCOPE ~TensorView() = default;

    SCOPE TensorView& operator=(TensorView<T, NDIM>&& other) = default;

    SCOPE size_type size() const {
      return std::reduce(&m_dims[0], &m_dims[ndim()], 1, std::multiplies<size_type>{});
    }

    SCOPE size_type dim(Dimension d) const {
      return m_dims[d];
    }

    SCOPE const dims_array_t& dims() const {
      return m_dims;
    }

    SCOPE size_type stride(size_type d) const {
      size_type s = 1;
      for (size_type i = d+1; i < ndim(); ++i) {
        s *= m_dims[i];
      }
      return s;
    }

    /* array-style flattened access */
    SCOPE value_type& operator[](size_type i) {
      if (m_ptr == nullptr) THROW("TensorView: non-const call with nullptr");
      return m_ptr[i];
    }

    /* array-style flattened access */
    SCOPE const_value_type operator[](size_type i) const {
      if (m_ptr == nullptr) return const_value_type{};
      return m_ptr[i];
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
        assert(indices[i] >= dim(i));
      }
      if (m_ptr == nullptr) THROW("TensorView: non-const call with nullptr");
      return m_ptr[offset(std::forward<Dims>(idxs)...)];
    }

    /* access host-side elements */
    template<typename... Dims>
    requires(sizeof...(Dims) == NDIM && (std::is_integral_v<Dims>&&...))
    SCOPE const_value_type operator()(Dims... idxs) const {
      // let's hope the compiler will hoist this out of loops
      std::array<size_type, sizeof...(Dims)> indices = {static_cast<size_type>(idxs)...};
      for (size_type i = 0; i < indices.size(); ++i) {
        assert(indices[i] >= dim(i));
      }
      if (m_ptr == nullptr) {
        return T(0);
      } else {
        return m_ptr[offset(std::forward<Dims>(idxs)...)];
      }
    }

    /**
     * Return a TensorView<T, (NDIM-M)> to a subview using the provided first M indices.
     */
    template<typename... Dims>
    requires(sizeof...(Dims) < NDIM && (std::is_integral_v<Dims>&&...))
    SCOPE TensorView<T, NDIM-sizeof...(Dims)> operator()(Dims... idxs) const {
      std::array<size_type, sizeof...(Dims)> indices = {static_cast<size_type>(idxs)...};
      for (size_type i = 0; i < indices.size(); ++i) {
        assert(indices[i] >= dim(i));
      }
      constexpr const Dimension noffs = sizeof...(Dims);
      constexpr const Dimension ndim = NDIM-noffs;
      size_type offset = offset_impl<0>(std::forward<Dims>(idxs)...);
      std::array<size_type, ndim> dims;
      for (Dimension i = 0; i < ndim; ++i) {
        dims[i] = m_dims[noffs+i];
      }
      return TensorView<T, ndim>(m_ptr+offset, dims);
    }


    SCOPE value_type* data() {
      return m_ptr;
    }

    SCOPE const_value_type* data() const {
      return m_ptr;
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
    SCOPE TensorView& operator=(const value_type& value) {
      if (m_ptr == nullptr) THROW("TensorView: non-const call with nullptr");
      foreach_idx(*this, [&](size_type i){ this->operator[](i) = value; });
      return *this;
    }

    /// Scale by scalar
    /// Device: assumes this operation is called by all threads in a block, synchronizes
    /// Host: assumes this operation is called by a single CPU thread
    SCOPE TensorView& operator*=(const value_type& value) {
      if (m_ptr == nullptr) THROW("TensorView: non-const call with nullptr");
      foreach_idx(*this, [&](size_type i){ this->operator[](i) *= value; });
      return *this;
    }

    /// Add another tensor
    /// Device: assumes this operation is called by all threads in a block, synchronizes
    /// Host: assumes this operation is called by a single CPU thread
    SCOPE TensorView& operator+=(const TensorView<T, NDIM>& value) {
      if (m_ptr == nullptr) THROW("TensorView: non-const call with nullptr");
      foreach_idx(*this, [&](size_type i){ this->operator[](i) += value[i]; });
      return *this;
    }

    /// Copy into patch
    /// Device: assumes this operation is called by all threads in a block, synchronizes
    /// Host: assumes this operation is called by a single CPU thread
    SCOPE TensorView& operator=(const TensorView<T, NDIM>& other) {
      if (m_ptr == nullptr) THROW("TensorView: non-const call with nullptr");
      if (other.m_ptr == nullptr) {
        foreach_idx(*this, [&](size_type i){ this->operator[](i) = 0; });
      } else {
        foreach_idx(*this, [&](size_type i){ this->operator[](i) = other[i]; });
      }
      return *this;
    }

    /// Copy into patch
    /// Device: assumes this operation is called by all threads in a block, synchronizes
    /// Host: assumes this operation is called by a single CPU thread
    template<typename TensorViewT>
    SCOPE TensorView& operator=(const TensorSlice<TensorViewT>& other) {
      if (m_ptr == nullptr) THROW("TensorView: non-const call with nullptr");
      foreach_idx(*this, [&](size_type i){ this->operator[](i) = other[i]; });
      return *this;
    }

    SCOPE void reduce_rank(const T& eps) {return;}

    SCOPE TensorSlice<TensorView> operator()(const std::array<Slice, NDIM>& slices) {
      if (m_ptr == nullptr) THROW("TensorView: non-const call with nullptr");
      return TensorSlice<TensorView>(*this, slices);
    }

    SCOPE TensorSlice<TensorView> get_slice(const std::array<Slice, NDIM>& slices) {
      if (m_ptr == nullptr) THROW("TensorView: non-const call with nullptr");
      return TensorSlice<TensorView>(*this, slices);
    }


    template<Dimension ndimactive>
    struct iterator : public detail::base_tensor_iterator<TensorView,ndimactive> {
      iterator (size_type count, TensorView& t)
      : detail::base_tensor_iterator<TensorView,ndimactive>(count, t)
      { }
      value_type& operator*() { return this->t.m_ptr[this->count]; }
      iterator& operator++() {this->inc(); return *this;}
      bool operator!=(const iterator& other) {return this->count != other.count;}
      bool operator==(const iterator& other) {return this->count == other.count;}
    };

    template<Dimension ndimactive>
    struct const_iterator : public detail::base_tensor_iterator<TensorView,ndimactive> {
      const_iterator (size_type count, const TensorView& t)
      : detail::base_tensor_iterator<TensorView,ndimactive>(count, t)
      { }
      value_type operator*() const { return this->t.m_ptr[this->count]; }
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

  template<typename TensorViewT>
  std::enable_if_t<!std::is_const_v<TensorViewT>, TensorSlice<TensorViewT>&>
  SCOPE TensorSlice<TensorViewT>::operator=(
    const TensorViewT& view)
  {
    foreach_idx(*this, [&](size_type i){ this->operator[](i) = view[i]; });
    return *this;
  }

} // namespace mra

#endif // MRA_TENSORVIEW_H
