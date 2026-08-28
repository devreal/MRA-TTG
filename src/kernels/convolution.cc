
#include "mra/kernels/convolution.h"


namespace mra {

#if defined(MRA_ENABLE_EXPLICIT_INSTANTIATION)
  template
  void submit_convolution_kernel<double, 3>(
    Key<3> key,
    Key<3> displacement,
    size_type K,
    size_type N,
    size_type n_nonzero,
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


  template
  void submit_convolution_kernel_batched<double, 3>(
    detail::BatchPool<detail::ConvolutionBatchArg<double, 3>>& pool,
    typename detail::BatchPool<detail::ConvolutionBatchArg<double, 3>>::slot_t& slot,
    detail::BatchPool<detail::SparsityState>& sparsity_pool,
    typename detail::BatchPool<detail::SparsityState>::slot_t& sparsity_slot,
    detail::BatchPool<size_type>& offset_pool,
    typename detail::BatchPool<size_type>::slot_t& offset_slot,
    size_type total_nonzero,
    size_type K,
    const double fac,
    ttg::device::Stream stream);

#endif // MRA_ENABLE_EXPLICIT_INSTANTIATION

} // namespace mra
