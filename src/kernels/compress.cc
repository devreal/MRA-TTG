
#include "mra/kernels/compress.h"


namespace mra {

  template
  void submit_compress_kernel<double, 3>(
    const Key<3>& key,
    size_type N,
    size_type K,
    bool is_ns,
    TensorView<double, 3+1>& p_view,
    TensorView<double, 3+1>& result_view,
    const TensorView<double, 2>& hgT_view,
    double* tmp,
    double* d_sumsq,
    const std::array<TensorView<double, 3+1>, Key<3>::num_children()>& in_views,
    ttg::device::Stream stream);

} // namespace mra
