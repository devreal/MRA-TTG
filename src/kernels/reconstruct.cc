
#include "mra/kernels/reconstruct.h"


namespace mra {
#if defined(MRA_ENABLE_EXPLICIT_INSTANTIATION)
  template
  void submit_reconstruct_kernel<double, 3>(
    const Key<3>& key,
    size_type N,
    size_type n_nonzero,
    size_type K,
    bool accumulate_NS,
    const SparseTensorView<double, 3+1>& node,
    const SparseTensorView<double, 2>& hg,
    const SparseTensorView<double, 3+1>& from_parent,
    const std::array<SparseTensorView<double, 3+1>, mra::Key<3>::num_children()>& r_arr,
    SparseTensorView<double, 3+1>& result,
    SparseTensorView<double, 3+1>& ns,
    double* tmp,
    ttg::device::Stream stream);
#endif // MRA_ENABLE_EXPLICIT_INSTANTIATION

#if defined(MRA_ENABLE_EXPLICIT_INSTANTIATION)
  /* The instantiations the tasks (mra/tasks/) use, which translation units compiled
   * without nvcc/hipcc link against: see MRA_HAVE_KERNEL_DEFINITIONS in mra/misc/platform.h. */
  template void submit_reconstruct_kernel<
      double,
      3,
      SparseTensorView<double, 4>,
      DenseTensorView<double, 2>,
      SparseTensorView<double, 4>,
      std::array<SparseTensorView<double, 4>, 8>,
      SparseTensorView<double, 4>,
      SparseTensorView<double, 4>>(
    Key<3> const&,
    size_type,
    size_type,
    size_type,
    bool,
    SparseTensorView<double, 4> const&,
    DenseTensorView<double, 2> const&,
    SparseTensorView<double, 4> const&,
    std::array<SparseTensorView<double, 4>, 8> const&,
    SparseTensorView<double, 4>&,
    SparseTensorView<double, 4>&,
    double*,
    ttg::device::Stream);
#ifndef MRA_ENABLE_HOST
  template void submit_reconstruct_kernel_batched<
      double,
      3,
      DenseTensorView<double, 2>>(
    detail::GroupedBatchPool<detail::ReconstructBatchArg<double, 3>>&,
    detail::GroupedBatchPool<detail::ReconstructBatchArg<double, 3>>::slot_t&,
    size_type,
    size_type,
    bool,
    DenseTensorView<double, 2> const&,
    ttg::device::Stream);
#endif // !MRA_ENABLE_HOST
#endif // MRA_ENABLE_EXPLICIT_INSTANTIATION

} // namespace mra
