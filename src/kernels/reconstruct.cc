
#include "mra/kernels/reconstruct.h"


namespace mra {
  template
  void submit_reconstruct_kernel<double, 3>(
    const Key<3>& key,
    size_type N,
    size_type K,
    TensorView<double, 3+1>& node,
    const TensorView<double, 2>& hg,
    const TensorView<double, 3+1>& from_parent,
    const std::array<TensorView<double, 3+1>, mra::Key<3>::num_children()>& r_arr,
    double* tmp,
    ttg::device::Stream stream);

} // namespace mra
