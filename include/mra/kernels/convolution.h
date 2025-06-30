#ifndef MRA_KERNELS_CONVOLUTION_H
#define MRA_KERNELS_CONVOLUTION_H

#include "mra/kernels.h"
#include "mra/kernels/transform.h"
#include "mra/misc/key.h"
#include "mra/misc/maxk.h"
#include "mra/misc/types.h"
#include "mra/misc/platform.h"
#include "mra/tensor/tensorview.h"
#include "mra/tensor/child_slice.h"

namespace mra{

  template<mra::Dimension NDIM>
  SCOPE size_type convolution_tmp_size(size_type K) {
    return 0;
  }

  namespace detail {

    template<typename T, Dimension NDIM>
    DEVSCOPE void convolution_kernel_impl()
    {
      // Implement the convolution kernel logic here
      // This is a placeholder for the actual implementation
    }

    template<typename T, Dimension NDIM>
    LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK)
    GLOBALSCOPE void convolution_kernel()
    {
      // Call the implementation function
      convolution_kernel_impl<T, NDIM>();
    }
  } // namespace detail

  template<typename T, Dimension NDIM>
  void submit_compress_kernel()
  {

  }

  /* explicit instantiation */
  extern template
  void submit_compress_kernel<double, 3>();

}

#endif // MRA_KERNELS_CONVOLUTION_H