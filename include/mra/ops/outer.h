#ifndef MRA_OUTER_H
#define MRA_OUTER_H

#include "mra/misc/types.h"
#include "mra/misc/platform.h"
#include "mra/tensor/tensor.h"
#include "mra/tensor/tensorview.h"

namespace mra{

  namespace detail{

    template <typename T, Dimension NDIM>
    SCOPE void outer(
      const concepts::TensorView<NDIM> auto& left,
      const concepts::TensorView<NDIM> auto& right,
      concepts::TensorView<NDIM> auto& result)
    {
      static_assert(result.ndim() <= NDIM, "too many dimensions in the result");
      static_assert(left.ndim() + right.ndim() == result.ndim(),
                     "inconsistent dimension in outer result");

      if (threadIdx.z == 0) {
        T* ptr = result.ptr();
        auto iter = right.unary_iterator(IterLevel::Vector, false, 0);
        for (auto iter0=left.unary_iterator(IterLevel::Vector, false, 0);
            iter0.ptr() != nullptr;
            ++iter0) {
          T val = *iter0;
          for (iter += thread_id();
               iter.ptr() != nullptr;
               iter += blockDim.x*blockDim.y) {
            T dimj = right.dim(0); // dimj = iter.dimj ??
            T* _p0  = iter.ptr();
            T Tstride = iter.s0();
            for (size_type i = 0; i<dimj; ++i, _p0 += Tstride) {
              *ptr++ = val * (*_p0);
            }
          }
        }
      }
    }
  }
}

#endif // MRA_OUTER_H_INCL
