#include "mra/misc/device_batch_pool.h"

namespace mra::detail {

  /* One warp per descriptor, grid-stride over descriptors. */
  __global__ void grouped_copy_scatter_kernel(const GroupedCopyDesc* __restrict__ descs, std::uint32_t num_descs,
                                              const unsigned char* __restrict__ src)
  {
    const unsigned lane = threadIdx.x % warpSize;
    const unsigned warps_per_block = blockDim.x / warpSize;
    const unsigned num_warps = gridDim.x * warps_per_block;
    for (unsigned d = blockIdx.x * warps_per_block + threadIdx.x / warpSize; d < num_descs; d += num_warps) {
      const GroupedCopyDesc desc = descs[d];
      unsigned char* dst = static_cast<unsigned char*>(desc.dst);
      if (desc.src_offset == GroupedCopyDesc::zero_fill) {
        for (std::uint32_t i = lane; i < desc.size; i += warpSize) dst[i] = 0;
      } else {
        const unsigned char* s = src + desc.src_offset;
        for (std::uint32_t i = lane; i < desc.size; i += warpSize) dst[i] = s[i];
      }
    }
  }

  void launch_grouped_copy_scatter(const GroupedCopyDesc* descs, std::uint32_t num_descs,
                                   const unsigned char* src, ttg::device::Stream stream)
  {
    // The kernel uses the device's actual warpSize; 32 only sizes the grid.
    constexpr unsigned threads = 128;
    constexpr unsigned warps_per_block = threads / 32;
    const unsigned blocks = std::min<unsigned>((num_descs + warps_per_block - 1) / warps_per_block, 256);
    CALL_KERNEL(grouped_copy_scatter_kernel, blocks, threads, 0, stream, (descs, num_descs, src));
  }

} // namespace mra::detail
