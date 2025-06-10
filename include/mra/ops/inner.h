#ifndef MRA_INNER_H
#define MRA_INNER_H

#include <cassert>
#include "mra/ops/mxm.h"
#include "mra/misc/types.h"
#include "mra/misc/platform.h"
#include "mra/tensor/tensor.h"
#include "mra/tensor/tensorview.h"

namespace mra{

  namespace detail{

    template <typename T, Dimension NDIM1, Dimension NDIM2>
    SCOPE void inner(
      const TensorView<T, NDIM1>& left,
      const TensorView<T, NDIM2>& right,
      TensorView<T, NDIM1+NDIM2-2>& result,
      long k0 = -1,
      long k1 = 0)
    {

      if (k0 < 0) k0 += left.ndim();
      if (k1 < 0) k1 += right.ndim();

      assert(left.dim(k0) == right.dim(k1));
      if (k0==0 && k1==0) {
        // c[i,j] = a[k,i]*b[k,j] ... collapsing extra indices to i & j
        long dimk = left.dim(k0);
        long dimj = right.stride(0);
        long dimi = left.stride(0);
        mTxm(dimi,dimj,dimk,result.data(),left.data(),right.data());
        return;
      }
      else if (k0==(left.ndim()-1) && k1==(right.ndim()-1)) {
        // c[i,j] = a[i,k]*b[j,k] ... collapsing extra indices to i & j
        long dimk = left.dim(k0);
        long dimi = left.size()/dimk;
        long dimj = right.size()/dimk;
        mxmT(dimi,dimj,dimk,result.data(),left.data(),right.data());
        return;
      }
      else if (k0==0 && k1==(right.ndim()-1)) {
        // c[i,j] = a[k,i]*b[j,k] ... collapsing extra indices to i & j
        long dimk = left.dim(k0);
        long dimi = left.stride(0);
        long dimj = right.size()/dimk;
        mTxmT(dimi,dimj,dimk,result.data(),left.data(),right.data());
        return;
      }
      else if (k0==(left.ndim()-1) && k1==0) {
        // c[i,j] = a[i,k]*b[k,j] ... collapsing extra indices to i & j
        long dimk = left.dim(k0);
        long dimi = left.size()/dimk;
        long dimj = right.stride(0);
        mxm(dimi,dimj,dimk,result.data(),left.data(),right.data());
        return;
      }

      // TODO: use more than the first slice of threads in z here
      if (threadIdx.z == 0) {
        size_type dimj = left.dim(k0);
        auto iter1 = right.unary_iterator(IterLevel::Vector, false, k1);
        T* ptr = result.data();
        for (auto iter0 = left.unary_iterator(IterLevel::Vector, false, k0);
             iter0.data() != nullptr;
             ++iter0, ptr += iter1.size()) {
          const T* __restrict__ xp0 = iter0.data();
          ssize_type s0 = iter0.s0();
          T *__restrict__ res = ptr;
          iter1.reset();
          for (iter1 += thread_id();
               iter1.data() != nullptr;
               iter1 += blockDim.x*blockDim.y, res += blockDim.x*blockDim.y) {
            const T* __restrict__ p0 = xp0;
            const T* __restrict__ p1 = iter1.data();
            ssize_type s1 = iter1.s0();
            T sum = 0;
            for (size_type j=0; j<dimj; ++j, p0+=s0, p1+=s1) {
              sum += (*p0) * (*p1);
            }
            res[thread_id()] += sum;
          }
        }
      }
    }
  }
}

#endif // MRA_INNER_H
