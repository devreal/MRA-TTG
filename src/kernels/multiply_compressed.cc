#include "mra/kernels/multiply_compressed.h"

namespace mra {

#if defined(MRA_ENABLE_EXPLICIT_INSTANTIATION)
  /* The instantiations the tasks (mra/tasks/) use, which translation units compiled
   * without nvcc/hipcc link against: see MRA_HAVE_KERNEL_DEFINITIONS in mra/misc/platform.h. */
  template void submit_multiply_compressed_kernel<
      double,
      3,
      SparseTensorView<double, 4>,
      SparseTensorView<double, 4>,
      SparseTensorView<double, 4>,
      SparseTensorView<double, 4>,
      std::array<SparseTensorView<double, 4>, 8>,
      DenseTensorView<double, 2>,
      DenseTensorView<double, 2>,
      DenseTensorView<double, 2>,
      DenseTensorView<double, 1>>(
    Domain<3> const&,
    Key<3> const&,
    SparseTensorView<double, 4> const&,
    SparseTensorView<double, 4> const&,
    SparseTensorView<double, 4> const&,
    SparseTensorView<double, 4>&,
    std::array<SparseTensorView<double, 4>, 8> const&,
    DenseTensorView<double, 2> const&,
    DenseTensorView<double, 2> const&,
    DenseTensorView<double, 2> const&,
    DenseTensorView<double, 1> const&,
    detail::MulCompressedWork<3> const*,
    size_type,
    bool,
    size_type,
    size_type,
    double*,
    ttg::device::Stream);
#ifndef MRA_ENABLE_HOST
  template void detail::launch_multiply_compressed_kernel_batched<
      double,
      3,
      DenseTensorView<double, 2>,
      DenseTensorView<double, 2>,
      DenseTensorView<double, 2>,
      DenseTensorView<double, 1>>(
    detail::MulCompressedBatchArg<double, 3>*,
    size_type const*,
    size_type,
    size_type,
    Domain<3> const&,
    DenseTensorView<double, 2> const&,
    DenseTensorView<double, 2> const&,
    DenseTensorView<double, 2> const&,
    DenseTensorView<double, 1> const&,
    bool,
    size_type,
    unsigned char const*,
    ttg::device::Stream);
#endif // !MRA_ENABLE_HOST
#endif // MRA_ENABLE_EXPLICIT_INSTANTIATION

} // namespace mra
