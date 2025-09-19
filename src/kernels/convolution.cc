
#include "mra/kernels/convolution.h"


namespace mra {

  template
  void submit_convolution_kernel<double, 3>(
    Key<3> key,
    size_type K,
    size_type N,
    const double opnorm,
    const double normr,
    const double norms,
    const double fac,
    const TensorView<double, 3+1>& f,
    TensorView<double, 3+1>& result,
    const std::array<TensorView<double, 2>, 3>& transr,
    const std::array<TensorView<double, 2>, 3>& transs,
    const double tol,
    double* tmp,
    ttg::device::Stream stream);

} // namespace mra
