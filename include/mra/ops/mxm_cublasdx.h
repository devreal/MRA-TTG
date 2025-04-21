#ifndef MRA_OPS_MXM_CUBLASDX_H
#define MRA_OPS_MXM_CUBLASDX_H

#include "mra/misc/types.h"
#include "mra/misc/platform.h"

#if __has_include(<cublasdx.hpp>)

#if !defined(MRA_CUDA_ARCH) || MRA_CUDA_ARCH < 70
#error "MRA_CUDA_ARCH must be defined and >= 70 to use cublasdx"
#endif

#include <cublasdx.hpp>

#if MRA_CUDA_ARCH == 70
#define MRA_CUBLASDX_SM 700
#elif MRA_CUDA_ARCH == 80
#define MRA_CUBLASDX_SM 800
#elif MRA_CUDA_ARCH == 90
#define MRA_CUBLASDX_SM 900
#else
#warning "Unknown MRA_CUDA_ARCH for cublasdx, using 80"
#define MRA_CUBLASDX_SM 800
#endif

namespace mra {

  namespace detail {

    constexpr size_type CUBLAS_MAX_MN = 64;

    template<typename GEMM>
    __device__ void mTxmq_cublasdx_core(auto& a_shared_tensor, auto& b_shared_tensor,
                                        auto& c_shared_tensor, auto& c_global_tensor,
                                        auto&& load = [](){}, auto&& prefetch = [](){}) {

      using alignment = cublasdx::alignment_of<GEMM>;

      /* load data to shared memory */
      load();
      /* wait for load to complete */
      cublasdx::copy_wait();

      /* prefetch data for next iteration */
      prefetch();

      // Execute using register API
      auto [c_register_fragment, partitioner] = GEMM().execute(a_shared_tensor, b_shared_tensor);

      // Store back to global memory using cublasdx::copy_fragment API
      // TODO: copying directly to global memory leads zero result tensors. WTF?!
      //cublasdx::copy_fragment<alignment::c>(c_register_fragment, c_global_tensor, partitioner);

      cublasdx::copy_fragment<alignment::c>(c_register_fragment, c_shared_tensor, partitioner);

      __syncthreads();

      // Store data from shared memory tensor to global memory tensor
      cublasdx::copy<GEMM, alignment::c>(c_shared_tensor, c_global_tensor);
    }


    /**
     * Compute the shared memory requirements for a given GEMM.
     * Takes into account double buffering of A (block_a) and B (block_b) as well as
     * staging of results through shared memory (block_c).
     */
    template<typename GEMM>
    constexpr size_type cublasdx_shmem_size_for(bool block_a, bool block_b, bool block_c) {
      auto calc = cublasdx::make_shared_storage_calc()
                  .add(cublasdx::alignment_of_v_a<GEMM>, sizeof(typename GEMM::a_value_type), GEMM::suggest_layout_smem_a())
                  .add(cublasdx::alignment_of_v_b<GEMM>, sizeof(typename GEMM::b_value_type), GEMM::suggest_layout_smem_b());
      if (block_a) {
        calc.add(cublasdx::alignment_of_v_a<GEMM>, sizeof(typename GEMM::a_value_type), GEMM::suggest_layout_smem_a());
      }
      if (block_b) {
        calc.add(cublasdx::alignment_of_v_b<GEMM>, sizeof(typename GEMM::b_value_type), GEMM::suggest_layout_smem_b());
      }
      if (block_c) {
        calc.add(cublasdx::alignment_of_v_c<GEMM>, sizeof(typename GEMM::c_value_type), GEMM::suggest_layout_smem_c());
      }

      size_type shared_memory_size = calc.get();
      return shared_memory_size;
    }

    template<typename T, size_type K>
    constexpr size_type cublasdx_shmem_size_k() {
      constexpr auto blockdims = mra::max_thread_dims(K);
      using BaseGEMM = decltype(cublasdx::Precision<T>()
                              + cublasdx::Type<cublasdx::type::real>()
                              + cublasdx::Function<cublasdx::function::MM>()
                              + cublasdx::SM<MRA_CUBLASDX_SM>() // TODO
                              + cublasdx::Block()
                              + cublasdx::BlockDim<blockdims.x, blockdims.y, blockdims.z>()
                              + cublasdx::MaxAlignment());
      using GEMMBlockA = decltype(BaseGEMM() + cublasdx::Size<std::min(CUBLAS_MAX_MN, K*K), K, K>()
                                             + cublasdx::Arrangement<cublasdx::col_major, cublasdx::row_major, cublasdx::row_major>()
                                             + cublasdx::LeadingDimension<K*K, K, K>());
      using GEMMBlockB = decltype(BaseGEMM() + cublasdx::Size<K, std::min(CUBLAS_MAX_MN, K*K), K>()
                                             + cublasdx::Arrangement<cublasdx::col_major, cublasdx::row_major, cublasdx::row_major>()
                                             + cublasdx::LeadingDimension<K, K*K, K*K>());
      auto size = std::max(cublasdx_shmem_size_for<GEMMBlockA>(true, false, true),
                           cublasdx_shmem_size_for<GEMMBlockB>(false, true, true));
      return size;
    }

    template<size_type M, size_type N, size_type K, typename aT, typename bT, typename cT>
    __device__ void mTxmq_cublasdx_block(cT* c, aT* a, bT* b) {
      constexpr auto blockdims = mra::max_thread_dims(K);
      extern SHARED __align__(16) char smem[];

      using GEMM = decltype(cublasdx::Size<std::min(CUBLAS_MAX_MN, M), std::min(CUBLAS_MAX_MN, N), K>()
                          + cublasdx::Precision<cT>()
                          + cublasdx::Type<cublasdx::type::real>()
                          + cublasdx::Function<cublasdx::function::MM>()
                          + cublasdx::Arrangement<cublasdx::col_major, cublasdx::row_major, cublasdx::row_major>()
                          + cublasdx::SM<MRA_CUBLASDX_SM>()
                          + cublasdx::Block()
                          + cublasdx::BlockDim<blockdims.x, blockdims.y, blockdims.z>()
                          + cublasdx::MaxAlignment()
                          + cublasdx::LeadingDimension<M, N, N>());

      //if (is_team_lead()) printf("mTxmq_cublasdx_block: shared_memory %u, smem %p, M = %d, N = %d, K = %d\n", cublasdx::get_shared_storage_size_ab<GEMM>(), smem, M, N, K);
      //__syncthreads();
      using alignment = cublasdx::alignment_of<GEMM>;

      if constexpr (M == K*K) {

        if (M > CUBLAS_MAX_MN) {
          auto [smem_a, smem_b, smem_a_n, smem_c] =
            cublasdx::slice_shared_memory_generic<GEMM::a_value_type, GEMM::b_value_type, GEMM::a_value_type, GEMM::c_value_type>(
                smem,
                cute::make_tuple(cublasdx::cosize(GEMM::suggest_layout_smem_a()), cublasdx::alignment_of_v_a<GEMM>),
                cute::make_tuple(cublasdx::cosize(GEMM::suggest_layout_smem_b()), cublasdx::alignment_of_v_b<GEMM>),
                cute::make_tuple(cublasdx::cosize(GEMM::suggest_layout_smem_a()), cublasdx::alignment_of_v_a<GEMM>),
                cute::make_tuple(cublasdx::cosize(GEMM::suggest_layout_smem_c()), cublasdx::alignment_of_v_c<GEMM>));

          /* copy b tensor into shared memory and leave there */
          auto b_global_tensor = cublasdx::make_tensor(b, GEMM::get_layout_gmem_b());
          auto b_shared_tensor = cublasdx::make_tensor(smem_b, GEMM::suggest_layout_smem_b());
          cublasdx::copy<GEMM, alignment::b>(b_global_tensor, b_shared_tensor);

          auto b_shared_tensor = cublasdx::make_tensor(smem_b, GEMM::suggest_layout_smem_b());

          auto a_shared_tensor   = cublasdx::make_tensor(smem_a,   GEMM::suggest_layout_smem_a());
          auto a_shared_tensor_n = cublasdx::make_tensor(smem_a_n, GEMM::suggest_layout_smem_a());
          auto c_shared_tensor   = cublasdx::make_tensor(smem_c,   GEMM::suggest_layout_smem_c());

          constexpr auto num_iter = M/CUBLAS_MAX_MN;
          for (int i = 0; i < num_iter; i++) {
            // Make global memory tensors
            auto a_global_tensor = cublasdx::make_tensor(a+(i*CUBLAS_MAX_MN),     GEMM::get_layout_gmem_a(cute::Int<M>{}));
            auto c_global_tensor = cublasdx::make_tensor(c+((i*CUBLAS_MAX_MN)*N), GEMM::get_layout_gmem_c());
            mTxmq_cublasdx_core<GEMM>(a_shared_tensor, b_shared_tensor, c_shared_tensor, c_global_tensor,
                                      [&](){
                                        /* load only on first iteration, all others are prefetched */
                                        if (i == 0) {
                                          cublasdx::copy<GEMM, alignment::a>(a_global_tensor, a_shared_tensor);
                                        }
                                      },
                                      [&](){
                                        /* prefetch into shared memory */
                                        if ((i+1) < num_iter) {
                                          auto a_global_tensor = cublasdx::make_tensor(a+((i+1)*CUBLAS_MAX_MN), GEMM::get_layout_gmem_a(cute::Int<M>{}));
                                          cublasdx::copy<GEMM, alignment::a>(a_global_tensor, a_shared_tensor_n);
                                        }
                                      });
            std::swap(a_shared_tensor, a_shared_tensor_n);
          }
        }

        /* handle remainder */
        constexpr const auto R = M%CUBLAS_MAX_MN;
        if constexpr (0 < R) {
          // Make global memory tensors
          using GEMM = decltype(cublasdx::Size<R, N, K>()
                              + cublasdx::Precision<cT>()
                              + cublasdx::Type<cublasdx::type::real>()
                              + cublasdx::Function<cublasdx::function::MM>()
                              + cublasdx::Arrangement<cublasdx::col_major, cublasdx::row_major, cublasdx::row_major>()
                              + cublasdx::SM<MRA_CUBLASDX_SM>()
                              + cublasdx::Block()
                              + cublasdx::BlockDim<blockdims.x, blockdims.y, blockdims.z>()
                              + cublasdx::MaxAlignment()
                              + cublasdx::LeadingDimension<M, N, N>());
          auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<GEMM>(smem, GEMM::suggest_layout_smem_a(),
                                                                                    GEMM::suggest_layout_smem_b(),
                                                                                    GEMM::suggest_layout_smem_c());
          auto a_shared_tensor = cublasdx::make_tensor(smem_a, GEMM::suggest_layout_smem_a());
          auto a_global_tensor = cublasdx::make_tensor(a+((M/CUBLAS_MAX_MN)*CUBLAS_MAX_MN),   GEMM::get_layout_gmem_a(cute::Int<M>{}));
          auto b_global_tensor = cublasdx::make_tensor(b, GEMM::get_layout_gmem_b());
          auto b_shared_tensor = cublasdx::make_tensor(smem_b, GEMM::suggest_layout_smem_b());
          auto c_global_tensor = cublasdx::make_tensor(c+((M/CUBLAS_MAX_MN)*CUBLAS_MAX_MN*N), GEMM::get_layout_gmem_c());
          auto c_shared_tensor = cublasdx::make_tensor(smem_c, GEMM::suggest_layout_smem_c());
          mTxmq_cublasdx_core<GEMM>(a_shared_tensor, b_shared_tensor, c_shared_tensor, c_global_tensor,
                                    [&](){
                                      cublasdx::copy<GEMM, alignment::a>(a_global_tensor, a_shared_tensor);
                                      cublasdx::copy<GEMM, alignment::b>(b_global_tensor, b_shared_tensor);
                                    },
                                    [](){});
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
    /* make sure all is done */
    SYNCTHREADS();
  }

  template<typename T>
  constexpr size_type mTxmq_shmem_size(size_type K) {
    switch (K) {
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