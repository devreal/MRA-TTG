#include "mra/kernels/norm.h"

namespace mra {

#if defined(MRA_ENABLE_EXPLICIT_INSTANTIATION)
  /* The instantiations the tasks (mra/tasks/) use, which translation units compiled
   * without nvcc/hipcc link against: see MRA_HAVE_KERNEL_DEFINITIONS in mra/misc/platform.h. */
  template void submit_norm_kernel<
      double,
      3,
      SparseTensorView<double, 4>,
      DenseTensorView<double, 1>>(
    Key<3> const&,
    size_type,
    size_type,
    SparseTensorView<double, 4> const&,
    DenseTensorView<double, 1>&,
    ttg::device::Stream);
#ifndef MRA_ENABLE_HOST
  template void detail::launch_norm_kernel_batched<
      double,
      3>(
    detail::NormBatchArg<double, 3>*,
    size_type const*,
    size_type,
    size_type,
    size_type,
    ttg::device::Stream);
#endif // !MRA_ENABLE_HOST
#endif // MRA_ENABLE_EXPLICIT_INSTANTIATION

} // namespace mra
