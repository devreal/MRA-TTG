
#include "mra/kernels/compress.h"


namespace mra {

#if defined(MRA_ENABLE_EXPLICIT_INSTANTIATION)
  template
  void submit_compress_kernel<double, 3>(
    const Key<3>& key,
    size_type N,
    size_type n_nonzero,
    size_type K,
    bool is_ns,
    const SparseTensorView<const double, 3+1>& in_view,
    SparseTensorView<double, 3+1>& p_view,
    SparseTensorView<double, 3+1>& result_view,
    const SparseTensorView<double, 2>& hgT_view,
    double* tmp,
    double* d_sumsq,
    const std::array<SparseTensorView<double, 3+1>, Key<3>::num_children()>& in_views,
    const SparseTensorView<const double, 3+1>& ns_view,
    const unsigned char* child_kept,
    double trunc_tol,
    ttg::device::Stream stream);
#endif // MRA_ENABLE_EXPLICIT_INSTANTIATION

#if defined(MRA_ENABLE_EXPLICIT_INSTANTIATION)
  /* The instantiations the tasks (mra/tasks/) use, which translation units compiled
   * without nvcc/hipcc link against: see MRA_HAVE_KERNEL_DEFINITIONS in mra/misc/platform.h. */
  template void submit_compress_kernel<
      double,
      3,
      SparseTensorView<double, 4>,
      SparseTensorView<double, 4>,
      SparseTensorView<double, 4>,
      DenseTensorView<double, 2>,
      std::array<SparseTensorView<double, 4>, 8>>(
    Key<3> const&,
    size_type,
    size_type,
    size_type,
    bool,
    SparseTensorView<double, 4> const&,
    SparseTensorView<double, 4>&,
    SparseTensorView<double, 4>&,
    DenseTensorView<double, 2> const&,
    double*,
    double*,
    std::array<SparseTensorView<double, 4>, 8> const&,
    SparseTensorView<double, 4> const&,
    unsigned char const*,
    double,
    ttg::device::Stream);
#ifndef MRA_ENABLE_HOST
  template void submit_compress_kernel_batched<
      double,
      3,
      DenseTensorView<double, 2>>(
    detail::GroupedBatchPool<detail::CompressBatchArg<double, 3>>&,
    detail::GroupedBatchPool<detail::CompressBatchArg<double, 3>>::slot_t&,
    size_type,
    size_type,
    bool,
    DenseTensorView<double, 2> const&,
    ttg::device::Stream);
#endif // !MRA_ENABLE_HOST
#endif // MRA_ENABLE_EXPLICIT_INSTANTIATION

} // namespace mra
