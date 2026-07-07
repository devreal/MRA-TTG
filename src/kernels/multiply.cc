
#include "mra/kernels/multiply.h"

namespace mra {

#if defined(MRA_ENABLE_EXPLICIT_INSTANTIATION)
  /* explicit instanatiation */
  template
  void submit_multiply_kernel<double, 3>(
    const Domain<3>& D,
    const Key<3>& keyA,
    const Key<3>& keyB,
    const SparseTensorView<double, 3+1>& funcA,
    const SparseTensorView<double, 3+1>& funcB,
    SparseTensorView<double, 3+1>& funcR,
    const SparseTensorView<double, 2>& hgT,
    const SparseTensorView<double, 2>& phi,
    const SparseTensorView<double, 2>& phiT,
    const SparseTensorView<double, 2>& phibar,
    const SparseTensorView<double, 1>& quad_x,
    size_type N,
    size_type K,
    double* tmp,
    ttg::device::Stream stream);
#endif // MRA_ENABLE_EXPLICIT_INSTANTIATION
} // namespace mra