#include "mra/kernels/truncate.h"

namespace mra {

#if defined(MRA_ENABLE_EXPLICIT_INSTANTIATION)
  /* The instantiations the tasks (mra/tasks/) use, which translation units compiled
   * without nvcc/hipcc link against: see MRA_HAVE_KERNEL_DEFINITIONS in mra/misc/platform.h. */
  template void submit_truncate_kernel<
      double,
      3,
      SparseTensorView<double, 4>&,
      DenseTensorView<double, 1>&>(
    Key<3> const&,
    SparseTensorView<double, 4>&,
    DenseTensorView<double, 1>&,
    std::array<double const*, Key<3>::num_children()> const&,
    size_type,
    double,
    ttg::device::Stream);
#ifndef MRA_ENABLE_HOST
  template void submit_truncate_kernel_batched<
      double,
      3>(
    detail::GroupedBatchPool<detail::TruncateBatchArg<double, 3>>&,
    detail::GroupedBatchPool<detail::TruncateBatchArg<double, 3>>::slot_t&,
    size_type,
    ttg::device::Stream);
#endif // !MRA_ENABLE_HOST
#endif // MRA_ENABLE_EXPLICIT_INSTANTIATION

} // namespace mra
