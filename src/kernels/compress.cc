
#include "mra/kernels/compress.h"


namespace mra {

#if defined(MRA_ENABLE_EXPLICIT_INSTANTIATION)
  template
  void submit_compress_kernel<double, 3>(
    const Key<3>& key,
    size_type N,
    size_type K,
    bool is_ns,
    const SparseTensorView<const double, 3+1>& in_view,
    SparseTensorView<double, 3+1>& p_view,
    SparseTensorView<double, 3+1>& result_view,
    const SparseTensorView<double, 2>& hgT_view,
    double* tmp,
    double* d_sumsq,
    const std::array<SparseTensorView<double, 3+1>, Key<3>::num_children()>& in_views,
    ttg::device::Stream stream);
#endif // MRA_ENABLE_EXPLICIT_INSTANTIATION

} // namespace mra
