#ifndef MRA_OPS_MXM_CUBLASDX_H
#define MRA_OPS_MXM_CUBLASDX_H

#include "mra/misc/types.h"

#if __has_include(<cublasdx.hpp>)
#include <cublasdx.hpp>

namespace mra {

  namespace detail {

    constexpr size_type CUBLAS_MAX_MN = 32;

    template<typename GEMM>
    __device__ void mTxmq_cublasdx_core(auto& a_global_tensor, auto& b_global_tensor, auto& c_global_tensor,
                                        auto& a_shared_tensor, auto& b_shared_tensor,
                                        bool load_a = true, bool load_b = true) {

      /* copy to shared memory */
      using alignment = cublasdx::alignment_of<GEMM>;
      if (load_a) {
        cublasdx::copy<GEMM, alignment::a>(a_global_tensor, a_shared_tensor);
      }
      if (load_b) {
        cublasdx::copy<GEMM, alignment::b>(b_global_tensor, b_shared_tensor);
      }
      cublasdx::copy_wait();

      // Execute using register API
      auto [c_register_fragment, partitioner] = GEMM().execute(a_shared_tensor, b_shared_tensor);

      // Store back to global memory using cublasdx::copy_fragment API
      cublasdx::copy_fragment<alignment::c>(c_register_fragment, c_global_tensor, partitioner);

      /* no synchronization needed, the caller will synchronize eventually */

    #if 0
      auto c_shared_tensor = cublasdx::make_tensor(smem_c, GEMM::suggest_layout_smem_c());
      // Execute GEMM
      GEMM().execute(1.0, a_shared_tensor, b_shared_tensor, 0.0, c_shared_tensor);
      __syncthreads();

      // Store data from shared memory tensor to global memory tensor
      cublasdx::copy<GEMM, alignment::c>(c_shared_tensor, c_global_tensor);
      cublasdx::copy_wait(); // Needed to ensure c_global_tensor has a defined state and data in it can be used for any following operations in the kernel. If there are no further instruction a kernel's finalization will be the final synchronization point.
    #endif // 0
    }


    template<typename T, size_type K>
    constexpr size_type cublasdx_shmem_size_k() {
      constexpr auto blockdims = mra::max_thread_dims(K);
      using BaseGEMM = decltype(cublasdx::Precision<T>()
                              + cublasdx::Type<cublasdx::type::real>()
                              + cublasdx::Function<cublasdx::function::MM>()
                              + cublasdx::SM<MRA_CUDA_ARCH>() // TODO
                              + cublasdx::Block()
                              + cublasdx::BlockDim<blockdims.x, blockdims.y, blockdims.z>()
                              + cublasdx::MaxAlignment());
      using GEMMBlockA = decltype(BaseGEMM() + cublasdx::Size<std::min(CUBLAS_MAX_MN, K*K), K, K>()
                                            + cublasdx::Arrangement<cublasdx::col_major, cublasdx::row_major, cublasdx::row_major>()
                                            + cublasdx::LeadingDimension<K*K, K, K>());
      using GEMMBlockB = decltype(BaseGEMM() + cublasdx::Size<K, std::min(CUBLAS_MAX_MN, K*K), K>()
                                            + cublasdx::Arrangement<cublasdx::col_major, cublasdx::row_major, cublasdx::row_major>()
                                            + cublasdx::LeadingDimension<K, K*K, K*K>());
      auto size = std::max(cublasdx::get_shared_storage_size_ab<GEMMBlockA>(GEMMBlockA::suggest_layout_smem_a(), GEMMBlockA::suggest_layout_smem_b()),
              cublasdx::get_shared_storage_size_ab<GEMMBlockB>(GEMMBlockB::suggest_layout_smem_a(), GEMMBlockB::suggest_layout_smem_b()));
      return size;
    }

    template<size_type M, size_type N, size_type K, typename aT, typename bT, typename cT>
    __device__ void mTxmq_cublasdx_block(cT* c, aT* a, bT* b) {
      constexpr auto blockdims = mra::max_thread_dims(K);
      extern __shared__ __align__(16) char smem[];

      using GEMM = decltype(cublasdx::Size<std::min(CUBLAS_MAX_MN, M), std::min(CUBLAS_MAX_MN, N), K>()
                          + cublasdx::Precision<cT>()
                          + cublasdx::Type<cublasdx::type::real>()
                          + cublasdx::Function<cublasdx::function::MM>()
                          + cublasdx::Arrangement<cublasdx::col_major, cublasdx::row_major, cublasdx::row_major>()
                          + cublasdx::SM<MRA_CUDA_ARCH>() // TODO
                          + cublasdx::Block()
                          + cublasdx::BlockDim<blockdims.x, blockdims.y, blockdims.z>()
                          + cublasdx::MaxAlignment()
                          + cublasdx::LeadingDimension<M, N, N>());

      //if (is_team_lead()) printf("mTxmq_cublasdx_block: shared_memory %u, smem %p, M = %d, N = %d, K = %d\n", cublasdx::get_shared_storage_size_ab<GEMM>(), smem, M, N, K);
      //__syncthreads();
      using alignment = cublasdx::alignment_of<GEMM>;

      auto [smem_a, smem_b, _] = cublasdx::slice_shared_memory<GEMM>(smem, GEMM::suggest_layout_smem_a(), GEMM::suggest_layout_smem_b(), GEMM::suggest_layout_smem_c());

      if constexpr (M == K*K) {
        if constexpr (M >= CUBLAS_MAX_MN) {
          /* copy b tensor into shared memory and leave there */
          auto b_global_tensor = cublasdx::make_tensor(b, GEMM::get_layout_gmem_b());
          auto b_shared_tensor = cublasdx::make_tensor(smem_b, GEMM::suggest_layout_smem_b());
          cublasdx::copy<GEMM, alignment::b>(b_global_tensor, b_shared_tensor);

          /* pass the a shared tensor into mTxmq_cublasdx_core */
          auto a_shared_tensor = cublasdx::make_tensor(smem_a, GEMM::suggest_layout_smem_a());

          for (int i = 0; i < M/CUBLAS_MAX_MN; i++) {
            // Make global memory tensors
            auto a_global_tensor = cublasdx::make_tensor(a+i*CUBLAS_MAX_MN,     GEMM::get_layout_gmem_a());
            auto c_global_tensor = cublasdx::make_tensor(c+i*(CUBLAS_MAX_MN*N), GEMM::get_layout_gmem_c());
            mTxmq_cublasdx_core<GEMM>(a_global_tensor, b_global_tensor, c_global_tensor,
                                      a_shared_tensor, b_shared_tensor, true, false);
          }
        }
        /* handle remainder */
        if constexpr (0 < (M%CUBLAS_MAX_MN)) {
          // Make global memory tensors
          constexpr const auto R = M%CUBLAS_MAX_MN;
          using GEMM = decltype(cublasdx::Size<R, N, K>()
                              + cublasdx::Precision<cT>()
                              + cublasdx::Type<cublasdx::type::real>()
                              + cublasdx::Function<cublasdx::function::MM>()
                              + cublasdx::Arrangement<cublasdx::col_major, cublasdx::row_major, cublasdx::row_major>()
                              + cublasdx::SM<MRA_CUDA_ARCH>() // TODO
                              + cublasdx::Block()
                              + cublasdx::BlockDim<blockdims.x, blockdims.y, blockdims.z>()
                              + cublasdx::MaxAlignment()
                              + cublasdx::LeadingDimension<M, N, N>());
          auto [smem_a, smem_b, _] = cublasdx::slice_shared_memory<GEMM>(smem, GEMM::suggest_layout_smem_a(), GEMM::suggest_layout_smem_b(), GEMM::suggest_layout_smem_c());
          auto b_global_tensor = cublasdx::make_tensor(b, GEMM::get_layout_gmem_b());
          auto b_shared_tensor = cublasdx::make_tensor(smem_b, GEMM::suggest_layout_smem_b());
          auto a_shared_tensor = cublasdx::make_tensor(smem_a, GEMM::suggest_layout_smem_a());
          auto a_global_tensor = cublasdx::make_tensor(a+((M/CUBLAS_MAX_MN)*CUBLAS_MAX_MN),   GEMM::get_layout_gmem_a());
          auto c_global_tensor = cublasdx::make_tensor(c+((M/CUBLAS_MAX_MN)*CUBLAS_MAX_MN*N), GEMM::get_layout_gmem_c());
          mTxmq_cublasdx_core<GEMM>(a_global_tensor, b_global_tensor, c_global_tensor,
                                    a_shared_tensor, b_shared_tensor, true, true);
        }
      } else {
        // TODO: implement!
        static_assert(M == K*K, "N equal to K*K currently not supported");
      }
    }

  } // namespace detail

  template <typename aT, typename bT, typename cT>
  __device__ void mTxmq(long dimi, long dimj, long dimk,
                        cT* c, const aT* a, const bT* b) {
    int M = dimi;
    int N = dimj;
    int K = dimk;
    if (M == K*K) {
      // A is tall and skinny, B is square
      if (K == 6) {
        detail::mTxmq_cublasdx_block<36, 6, 6>(c, a, b);
      } else if (K == 8) {
        detail::mTxmq_cublasdx_block<64, 8, 8>(c, a, b);
      } else if (K == 10) {
        detail::mTxmq_cublasdx_block<100, 10, 10>(c, a, b);
      } else if (K == 12) {
        detail::mTxmq_cublasdx_block<12*12, 12, 12>(c, a, b);
      } else if (K == 16) {
        detail::mTxmq_cublasdx_block<16*16, 16, 16>(c, a, b);
      } else if (K == 20) {
        detail::mTxmq_cublasdx_block<400, 20, 20>(c, a, b);
      } else {
        if (is_team_lead()) printf("mTxmq: Unsupport K = %d\n", K);
      }
    } else if (N == K*K) {
      // B is wide and narrow, A is square
      if (K == 6) {
        //mTxmq_cublasdx_block<6, 36, 6>(c, a, b);
      } else if (K == 8) {
        //mTxmq_cublasdx_block<8, 64, 8>(c, a, b);
      } else if (K == 10) {
        //mTxmq_cublasdx_block<10, 100, 10>(c, a, b);
      } else if (K == 12) {
        //mTxmq_cublasdx_block<12, 12*12, 12>(c, a, b);
      } else if (K == 16) {
        //mTxmq_cublasdx_block<16, 16*16, 16>(c, a, b);
      } else if (K == 20) {
        //mTxmq_cublasdx_block<20, 400, 20>(c, a, b);
      } else {
        if (is_team_lead()) printf("mTxmq: Unsupport K = %d\n", K);
      }
    } else {
        printf("mTxmq: Unknown configuration with M = %d, N = %d, K = %d\n", M, N, K);
    }
  }

  template<typename T>
  constexpr size_type mTxmq_shmem_size(size_type K) {
    switch (K) {
      /* TODO: assume GEMM for compressed form, so 2K */
      case 6: return detail::cublasdx_shmem_size_k<T, 6>();
      case 8: return detail::cublasdx_shmem_size_k<T, 8>();
      case 10: return detail::cublasdx_shmem_size_k<T, 10>();
      case 12: return detail::cublasdx_shmem_size_k<T, 12>();
      case 16: return detail::cublasdx_shmem_size_k<T, 16>();
      case 20: return detail::cublasdx_shmem_size_k<T, 20>();
      default: THROW("CUBLASdx: Unsupported K");
    }
  }

} // namespace mra


#define MRA_HAVE_MTXMQ 1

#endif // __has_include(<cublasdx.hpp>)

#endif // MRA_OPS_MXM_CUBLASDX_H