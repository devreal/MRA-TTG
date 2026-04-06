
#include "mra/kernels/convolution.h"


namespace mra {

  template
  void submit_convolution_kernel<double, 3>(
    Key<3> key,
    Key<3> displacement,
    size_type K,
    size_type N,
    const double opnorm,
    const double fac,
    const double tol,
    const TensorView<double, 3+1>& in,
    const TensorView<double, 3+1>& f,
    TensorView<double, 3+1>& result,
    TensorView<double, 1>& resnorms,
    const std::array<TensorView<double, 3>, 3>& transr,
    const std::array<TensorView<double, 3>, 3>& transs,
    const TensorView<double, 3>& opnorms,
    const std::array<bool, 2>& at,
    double* tmp,
    ttg::device::Stream stream);

} // namespace mra
