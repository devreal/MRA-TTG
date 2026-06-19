
#include "mra/kernels/convolution.h"


namespace mra {

  template
  void submit_convolution_kernel<double, 3>(
    Key<3> key,
    Key<3> displacement,
    size_type K,
    size_type N,
    const double fac,
    const double tol,
    const SparseTensorView<double, 3+1>& in,
    const SparseTensorView<double, 3+1>& f,
    SparseTensorView<double, 3+1>& result,
    SparseTensorView<double, 1>& resnorms,
    const std::array<SparseTensorView<double, 4>, 3>& transr,
    const std::array<SparseTensorView<double, 4>, 3>& transs,
    const DenseTensorView<double, 4>& opnorms,
    const std::array<bool, 2>& at,
    double* tmp,
    ttg::device::Stream stream);

} // namespace mra
