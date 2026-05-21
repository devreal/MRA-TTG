
#include "mra/kernels/derivative.h"


namespace mra {

  template
  void submit_derivative_kernel<double, 3>(
    const Domain<3>& D,
    const Key<3>& key,
    const Key<3>& left,
    const Key<3>& center,
    const Key<3>& right,
    const SparseTensorView<double, 3+1>& node_left,
    const SparseTensorView<double, 3+1>& node_center,
    const SparseTensorView<double, 3+1>& node_right,
    const DenseTensorView<double, 3>& operators,
    SparseTensorView<double, 3+1>& deriv,
    const DenseTensorView<double, 2>& phi,
    const DenseTensorView<double, 2>& phibar,
    const DenseTensorView<double, 1>& quad_x,
    double* tmp,
    size_type N,
    size_type K,
    const double g1,
    const double g2,
    size_type axis,
    const int bc_left,
    const int bc_right,
    ttg::device::Stream stream);

} // namespace mra
