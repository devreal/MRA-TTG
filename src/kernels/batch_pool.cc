#include <algorithm>
#include <cstdint>
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

  /* Phase 1: every thread copies 16-byte words (bytes if either side is not aligned) of
   * [0, total) from host to dev. Phase 2: one warp per descriptor, as
   * grouped_copy_scatter_kernel, with the descriptors and sources read from host. The
   * phases are independent: no source of phase 2 is read from dev. */
  __global__ void grouped_copy_from_host_kernel(const unsigned char* __restrict__ host, unsigned char* __restrict__ dev,
                                                std::size_t total, const GroupedCopyDesc* __restrict__ descs,
                                                std::uint32_t num_descs, const unsigned char* __restrict__ src)
  {
    const std::size_t nthreads = std::size_t(gridDim.x) * blockDim.x;
    const std::size_t tid = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    std::size_t done = 0;
    if (((reinterpret_cast<std::uintptr_t>(host) | reinterpret_cast<std::uintptr_t>(dev)) & 15) == 0) {
      const std::size_t nvec = total / 16;
      const uint4* hv = reinterpret_cast<const uint4*>(host);
      uint4* dv = reinterpret_cast<uint4*>(dev);
      for (std::size_t i = tid; i < nvec; i += nthreads) dv[i] = hv[i];
      done = nvec * 16;
    }
    for (std::size_t i = done + tid; i < total; i += nthreads) dev[i] = host[i];

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

  void launch_grouped_copy_from_host(const unsigned char* host, unsigned char* dev, std::size_t total,
                                     std::size_t descs_off, std::uint32_t num_descs, std::size_t src_off,
                                     ttg::device::Stream stream)
  {
    constexpr unsigned threads = 128;
    constexpr unsigned warps_per_block = threads / 32;
    const std::size_t copy_blocks = (total / 16 + threads - 1) / threads;
    const std::size_t desc_blocks = (num_descs + warps_per_block - 1) / warps_per_block;
    const unsigned blocks = static_cast<unsigned>(std::clamp<std::size_t>(std::max(copy_blocks, desc_blocks), 1, 256));
    CALL_KERNEL(grouped_copy_from_host_kernel, blocks, threads, 0, stream,
                (host, dev, total, reinterpret_cast<const GroupedCopyDesc*>(host + descs_off), num_descs,
                 host + src_off));
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
