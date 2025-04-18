
#include "mra/kernels/multiply.h"

namespace mra {

  /* explicit instanatiation */
  template
  void submit_multiply_kernel<double, 3>(
    const Domain<3>& D,
    const TensorView<double, 3+1>& funcA,
    const TensorView<double, 3+1>& funcB,
    TensorView<double, 3+1>& funcR,
    const TensorView<double, 2>& phiT,
    const TensorView<double, 2>& phibar,
    size_type N,
    size_type K,
    const Key<3>& key,
    double* tmp,
    ttg::device::Stream stream);

} // namespace mra