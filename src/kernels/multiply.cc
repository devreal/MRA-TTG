
#include "mra/kernels/multiply.h"

namespace mra {

  /* explicit instanatiation */
  template
  void submit_multiply_kernel<double, 3>(
    const Domain<3>& D,
    const Key<3>& keyA,
    const Key<3>& keyB,
    const TensorView<double, 3+1>& funcA,
    const TensorView<double, 3+1>& funcB,
    TensorView<double, 3+1>& funcR,
    const TensorView<double, 2>& hgT,
    const TensorView<double, 2>& phi,
    const TensorView<double, 2>& phiT,
    const TensorView<double, 2>& phibar,
    const TensorView<double, 1>& quad_x,
    size_type N,
    size_type K,
    double* tmp,
    ttg::device::Stream stream);
} // namespace mra