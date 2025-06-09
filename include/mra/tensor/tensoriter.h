#ifndef MRA_TENSORITER_H
#define MRA_TENSORITER_H

#include "mra/misc/types.h"
#include "mra/misc/platform.h"

namespace mra {

  enum class IterLevel { Element, Vector };

  template<typename TensorView>
  struct TensorIterator {
    using tensor_type = TensorView;
    using value_type  = std::conditional_t<std::is_const_v<TensorView>,
                                           typename TensorView::const_value_type,
                                           typename TensorView::value_type>;
  private:
    std::array<ssize_type, TensorView::ndim()> m_idx;
    Dimension m_ndim = TensorView::ndim();
    tensor_type* m_t0 = nullptr;
    value_type* m_p0 = nullptr;
    ssize_type m_jdim;

    SCOPE size_type stride(size_type i) const {
      assert(i < m_ndim);
      if (i < m_jdim) {
        return m_t0->stride(i);
      } else {
        return m_t0->stride(i+1);
      }
    }

    SCOPE size_type dim(size_type i) const {
      assert(i < m_ndim);
      if (i < m_jdim) {
        return m_t0->dim(i);
      } else {
        return m_t0->dim(i+1);
      }
    }

    template<typename View>
    friend ::std::ostream& ::std::operator<<(std::ostream& s, const mra::TensorIterator<View>& iter);

  public:
    constexpr static ssize_type default_jdim = std::numeric_limits<ssize_type>::max();
    SCOPE TensorIterator(TensorView* t0,
                         IterLevel iterlevel,
                         ssize_type jdim = default_jdim)
    : m_t0(t0)
    , m_jdim(jdim)
    {
      if (!t0) {
        // Used to indicate end of iteration.
        return;
      }

      m_p0 = m_t0->data();

      if (iterlevel == IterLevel::Element) {
        // Iteration will include all dimensions
        m_jdim = TensorView::ndim();
      } else {
        // Iterations will exclude dimension jdim, default is last one
        if (jdim == default_jdim) {
          m_jdim = TensorView::ndim()-1;
        }

        m_ndim--;
      }
      std::fill(m_idx.begin(), m_idx.end(), 0);
    }

    SCOPE size_type s0() const {
      return m_t0->stride(m_jdim);
    }

    SCOPE size_type ptr() const {
      return m_p0;
    }

    SCOPE TensorIterator& operator++() {
      ssize_type d = m_ndim-1;
      if (m_p0==0) {
        return *this;
      }
      while (m_idx[d] >= (dim(d) - 1)) {
        m_p0 -= m_idx[d] * stride(d);
        m_idx[d] = 0;
        d--;
        if (d < 0) {
          m_p0 = 0;
          return *this;
        }
      }
      m_p0 += stride(d);
      ++(m_idx[d]);
      return *this;
    }

    SCOPE TensorIterator& operator+=(size_type c) {
      ssize_type d = m_ndim-1;
      if (m_p0==0) {
        return *this;
      }
      while (c > 0) {
        ssize_type inc = std::min(c, size_type(dim(d)-m_idx[d]));
        m_p0 += inc*stride(d);
        m_idx[d] += inc;
        c -= inc;
        if (c > 0) {
          m_p0 -= m_idx[d]*stride(d);
          m_idx[d] = 0;
          d--;
          if (d < 0) {
            m_p0 = 0;
            return *this;
          }
        }
      }
      // TODO: why do we need this?
      //m_p0 += stride(d);
      //++(m_idx[d]);
      return *this;
    }

    SCOPE value_type& operator*() {
      return *m_p0;
    }

    SCOPE const value_type& operator*() const {
      return *m_p0;
    }

    SCOPE void reset() {
      std::fill(m_idx.begin(), m_idx.end(), 0);
      m_p0 = m_t0->data();
    }

    SCOPE value_type* data() {
      return m_p0;
    }

    SCOPE const value_type* data() const {
      return m_p0;
    }

    /**
     * The number of elements this iterator spans.
     */
    SCOPE size_type size() const {
      size_type s = 1;
      for (ssize_type d = 0; d < m_ndim; ++d) {
        s *= dim(d);
      }
      return s;
    }
  };

} // namespace mra

namespace std {

  template<typename View>
  inline std::ostream& operator<<(std::ostream& s, const mra::TensorIterator<View>& iter) {
    s << "TensorIterator(idx=[";
    for (mra::size_type i = 0; i < iter.m_ndim; ++i) {
      s << iter.m_idx[i];
      if (i < iter.m_ndim-1) {
        s << ", ";
      }
    }
    std::cout << ", m_dims=[";
    for (mra::size_type i = 0; i < iter.m_ndim; ++i) {
      s << iter.dim(i);
      if (i < iter.m_ndim-1) {
        s << ", ";
      }
    }
    std::cout << ", strides=[";
    for (mra::size_type i = 0; i < iter.m_ndim; ++i) {
      s << iter.stride(i);
      if (i < iter.m_ndim-1) {
        s << ", ";
      }
    }
    s << "])";
    return s;
  }
} // namespace std

#endif // MRA_TENSORITER_H
