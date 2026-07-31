
#include "mra/kernels/reconstruct.h"


namespace mra {
#if defined(MRA_ENABLE_EXPLICIT_INSTANTIATION)
  template
  void submit_reconstruct_kernel<double, 3>(
    const Key<3>& key,
    size_type N,
    size_type K,
    bool accumulate_NS,
    const SparseTensorView<double, 3+1>& node,
    const SparseTensorView<double, 2>& hg,
    const SparseTensorView<double, 3+1>& from_parent,
    const std::array<SparseTensorView<double, 3+1>, mra::Key<3>::num_children()>& r_arr,
    SparseTensorView<double, 3+1>& result,
    double* tmp,
    ttg::device::Stream stream);
#endif // MRA_ENABLE_EXPLICIT_INSTANTIATION

} // namespace mra
