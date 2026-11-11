
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
    const std::array<SparseTensorView<double, 3>, 3>& transr,
    const std::array<SparseTensorView<double, 3>, 3>& transs,
    const DenseTensorView<double, 4>& opnorms,
    const std::array<bool, 2>& at,
    double* tmp,
    ttg::device::Stream stream);

#ifndef MRA_ENABLE_HOST
  template
  void submit_convolution_kernel_batched<double, 3>(
    detail::GroupedBatchPool<detail::ConvolutionBatchArg<double, 3>>& pool,
    typename detail::GroupedBatchPool<detail::ConvolutionBatchArg<double, 3>>::slot_t& slot,
    size_type total_nonzero,
    size_type K,
    const double fac,
    ttg::device::Stream stream);
#endif // !MRA_ENABLE_HOST

#endif // MRA_ENABLE_EXPLICIT_INSTANTIATION

#if defined(MRA_ENABLE_EXPLICIT_INSTANTIATION)
  /* The instantiations the tasks (mra/tasks/) use, which translation units compiled
   * without nvcc/hipcc link against: see MRA_HAVE_KERNEL_DEFINITIONS in mra/misc/platform.h. */
  template void submit_convolution_kernel<
      double,
      3,
      SparseTensorView<double, 4>,
      SparseTensorView<double, 4>,
      SparseTensorView<double, 4>,
      DenseTensorView<double, 1>,
      std::array<DenseTensorView<const double, 3>, 3>,
      std::array<DenseTensorView<const double, 3>, 3>,
      DenseTensorView<double, 4>>(
    Key<3>,
    Key<3>,
    size_type,
    size_type,
    size_type,
    double,
    double,
    SparseTensorView<double, 4> const&,
    SparseTensorView<double, 4> const&,
    SparseTensorView<double, 4>&,
    DenseTensorView<double, 1>&,
    std::array<DenseTensorView<const double, 3>, 3> const&,
    std::array<DenseTensorView<const double, 3>, 3> const&,
    DenseTensorView<double, 4> const&,
    std::array<bool, 2> const&,
    double*,
    ttg::device::Stream);
  // submit_convolution_kernel_batched<double, 3> is instantiated above
#endif // MRA_ENABLE_EXPLICIT_INSTANTIATION

} // namespace mra
