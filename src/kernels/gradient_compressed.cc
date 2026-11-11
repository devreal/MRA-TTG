#include "mra/kernels/gradient_compressed.h"

namespace mra {

#if defined(MRA_ENABLE_EXPLICIT_INSTANTIATION)
  /* The instantiations the tasks (mra/tasks/) use, which translation units compiled
   * without nvcc/hipcc link against: see MRA_HAVE_KERNEL_DEFINITIONS in mra/misc/platform.h. */
  template void submit_gradient_derivative_kernel<
      double,
      3,
      SparseTensorView<double, 4>,
      DenseTensorView<double, 3>,
      DenseTensorView<double, 2>,
      DenseTensorView<double, 2>,
      DenseTensorView<double, 1>>(
    Domain<3> const&,
    Key<3> const&,
    Key<3> const*,
    std::array<SparseTensorView<double, 4>, 2*3> const&,
    SparseTensorView<double, 4> const&,
    std::array<SparseTensorView<double, 4>, 3>&,
    DenseTensorView<double, 3> const&,
    DenseTensorView<double, 2> const&,
    DenseTensorView<double, 2> const&,
    DenseTensorView<double, 1> const&,
    double*,
    size_type,
    std::array<size_type, 3> const&,
    size_type,
    int,
    int,
    ttg::device::Stream);

  template void submit_gradient_spawn_kernel<
      double,
      3,
      SparseTensorView<double, 4>,
      SparseTensorView<double, 4>,
      DenseTensorView<double, 2>,
      std::array<SparseTensorView<double, 4>, 8>>(
    Key<3> const&,
    size_type,
    size_type,
    SparseTensorView<double, 4> const&,
    SparseTensorView<double, 4> const&,
    DenseTensorView<double, 2> const&,
    std::array<SparseTensorView<double, 4>, 8> const&,
    unsigned char const*,
    double*,
    ttg::device::Stream);
#ifndef MRA_ENABLE_HOST
  template void detail::launch_gradient_derivative_kernel_batched<
      double,
      3,
      DenseTensorView<double, 3>,
      DenseTensorView<double, 2>,
      DenseTensorView<double, 2>,
      DenseTensorView<double, 1>>(
    detail::GroupedBatchPool<detail::GradientDerivativeBatchArg<double, 3>>&,
    detail::GroupedBatchPool<detail::GradientDerivativeBatchArg<double, 3>>::slot_t&,
    size_type,
    size_type,
    Domain<3> const&,
    DenseTensorView<double, 3> const&,
    DenseTensorView<double, 2> const&,
    DenseTensorView<double, 2> const&,
    DenseTensorView<double, 1> const&,
    size_type,
    int,
    int,
    ttg::device::Stream);

  template void detail::launch_gradient_spawn_kernel_batched<
      double,
      3,
      DenseTensorView<double, 2>>(
    detail::GroupedBatchPool<detail::GradientSpawnBatchArg<double, 3>>&,
    detail::GroupedBatchPool<detail::GradientSpawnBatchArg<double, 3>>::slot_t&,
    size_type,
    size_type,
    size_type,
    DenseTensorView<double, 2> const&,
    ttg::device::Stream);
#endif // !MRA_ENABLE_HOST
#endif // MRA_ENABLE_EXPLICIT_INSTANTIATION

} // namespace mra
