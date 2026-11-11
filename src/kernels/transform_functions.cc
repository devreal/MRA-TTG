#include "mra/kernels/transform_functions.h"

namespace mra {

#if defined(MRA_ENABLE_EXPLICIT_INSTANTIATION)
  /* The instantiations the tasks (mra/tasks/) use, which translation units compiled
   * without nvcc/hipcc link against: see MRA_HAVE_KERNEL_DEFINITIONS in mra/misc/platform.h. */
  template void submit_transform_functions_kernel<
      double,
      3,
      SparseTensorView<double, 4>,
      SparseTensorView<double, 4>>(
    Key<3> const&,
    SparseTensorView<double, 4> const&,
    SparseTensorView<double, 4>&,
    double const*,
    size_type,
    size_type,
    size_type,
    size_type,
    ttg::device::Stream);
#ifndef MRA_ENABLE_HOST
  template void detail::launch_transform_functions_kernel_batched<
      double,
      3>(
    detail::TransformFunctionsBatchArg<double, 3>*,
    size_type const*,
    size_type,
    size_type,
    double const*,
    size_type,
    ttg::device::Stream);
#endif // !MRA_ENABLE_HOST
#endif // MRA_ENABLE_EXPLICIT_INSTANTIATION

} // namespace mra
