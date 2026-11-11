#include "mra/kernels/generate_op1d.h"

namespace mra {

#if defined(MRA_ENABLE_EXPLICIT_INSTANTIATION)
  /* The instantiations the tasks (mra/tasks/) use, which translation units compiled
   * without nvcc/hipcc link against: see MRA_HAVE_KERNEL_DEFINITIONS in mra/misc/platform.h. */
  template void submit_assemble_op1d_norms_kernel<
      double,
      3,
      DenseTensorView<double, 3>,
      DenseTensorView<double, 2>,
      DenseTensorView<double, 4>,
      DenseTensorView<double, 1>,
      DenseTensorView<double, 3>>(
    size_type,
    size_type,
    Level,
    std::array<Translation, 3>,
    Translation,
    DenseTensorView<double, 3> const&,
    DenseTensorView<double, 2> const&,
    DenseTensorView<double, 4> const&,
    DenseTensorView<double, 1> const&,
    DenseTensorView<double, 3> const&,
    ttg::device::Stream);

  template void submit_assemble_op1d_opnorm_kernel<
      double,
      DenseTensorView<double, 4>,
      DenseTensorView<double, 1>>(
    size_type,
    size_type,
    DenseTensorView<double, 4> const&,
    DenseTensorView<double, 1> const&,
    ttg::device::Stream);

  template void submit_generate_op1d_kernel<
      double,
      DenseTensorView<double, 3>,
      DenseTensorView<double, 2>,
      DenseTensorView<double, 2>,
      DenseTensorView<double, 4>,
      DenseTensorView<double, 4>,
      DenseTensorView<double, 3>>(
    GenerateOp1DItem<double> const*,
    size_type,
    size_type,
    size_type,
    DenseTensorView<double, 3> const&,
    DenseTensorView<double, 2> const&,
    DenseTensorView<double, 2> const&,
    double const*,
    double const*,
    size_type,
    Translation,
    DenseTensorView<double, 4> const&,
    DenseTensorView<double, 4> const&,
    DenseTensorView<double, 3> const&,
    ttg::device::Stream);
#endif // MRA_ENABLE_EXPLICIT_INSTANTIATION

} // namespace mra
