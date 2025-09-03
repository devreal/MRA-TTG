#ifndef MRA_KERNELS_KERNEL_STATE_H
#define MRA_KERNELS_KERNEL_STATE_H

namespace mra::detail {

  /**
   * The state a kernel can be in.
   *
   * Typical state transitions are
   * Initialized -> Select -> Submit -> Wait -> Epilogue
   */
  enum class KernelState {
    Initialized, ///< The kernel was initialized
    Select,      ///< select() was called on the kernel
    Submit,      ///< The kernel was submitted
    Wait,        ///< The operations to wait for were queried
    Epilogue,    ///< The epilogue was called
  };

} // namespace mra::detail


#endif // MRA_KERNELS_KERNEL_STATE_H