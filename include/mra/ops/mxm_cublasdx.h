#ifndef MRA_OPS_MXM_CUBLASDX_H
#define MRA_OPS_MXM_CUBLASDX_H

#include "mra/misc/types.h"
#include "mra/misc/platform.h"

#define MRA_CUBLASDX_BLOCK_C 1

#if __has_include(<cublasdx.hpp>)

#define MRA_HAVE_CUBLASDX 1

#if !defined(MRA_CUDA_ARCH) || MRA_CUDA_ARCH < 70
#error "MRA_CUDA_ARCH must be defined and >= 70 to use cublasdx"
#endif

#include <cublasdx.hpp>

#if MRA_CUDA_ARCH == 70
#define MRA_CUBLASDX_SM 700
#define MRA_CUBLASDX_MAX_SHM (30*1024)
#elif MRA_CUDA_ARCH == 80
#define MRA_CUBLASDX_SM 800
#define MRA_CUBLASDX_MAX_SHM (40*1024)
#elif MRA_CUDA_ARCH == 90
#define MRA_CUBLASDX_SM 900
#define MRA_CUBLASDX_MAX_SHM (110*1024)
#else
#warning "Unknown MRA_CUDA_ARCH for cublasdx, using 80"
#define MRA_CUBLASDX_SM 800
#endif

namespace mra {

  namespace detail {

    constexpr size_type CUBLAS_MIN_MN = 16;

    template<typename T, size_type K>
    constexpr size_type cublasdx_max_mn() {
      // K^2 for square B/A, double buffering for A/B and C
#ifdef MRA_CUBLASDX_BLOCK_C
      auto max_nm = ((MRA_CUBLASDX_MAX_SHM / sizeof(T)) - K*K) / (4*K);
#else
      auto max_nm = ((MRA_CUBLASDX_MAX_SHM / sizeof(T)) - K*K) / (2*K);
#endif // MRA_CUBLASDX_BLOCK_C
      // round down to the nearest power of 2
      // TODO: std::log2 is constexpr only since C++26
      //size_type p = std::pow(2, (int)std::log2(max_nm));
      size_type l = 1;
      while ((l<<1) <= max_nm) l <<= 1;
      return std::min(l, K*K);
#if 0
      size_type max_mn = CUBLAS_MIN_MN;
      auto size = [](size_type mn){
        return (mn*K*4 // double buffering for A/B and C
                + K*K    // buffering for B/A
               )*sizeof(T);
      };
      while (size(max_mn*2) <= MRA_CUBLASDX_MAX_SHM) {
        max_mn *= 2;
      }

      return (max_mn > 64) ? 64 : max_mn;
#endif // 0
    }

    template<typename GEMM>
    __device__ void mTxmq_cublasdx_core(auto& a_shared_tensor, auto& b_shared_tensor,
                                        auto& c_tensor, /* auto& c_global_tensor, */
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

      cublasdx::copy_fragment<alignment::c>(c_register_fragment, c_tensor, partitioner);

      //GEMM().execute(1.0, a_shared_tensor, b_shared_tensor, 0.0, c_shared_tensor);
#ifdef MRA_CUBLASDX_BLOCK_C
      __syncthreads();
#endif // MRA_CUBLASDX_BLOCK_C

      // Store data from shared memory tensor to global memory tensor
      //cublasdx::copy<GEMM, alignment::c>(c_shared_tensor, c_global_tensor);
    }


    /**
     * Compute the shared memory requirements for a given GEMM.
     * Takes into account double buffering of A (block_a) and B (block_b) as well as
     * staging of results through shared memory (block_c).
     */
    template<typename GEMM>
    constexpr size_type cublasdx_shmem_size_for(bool block_a, bool block_b, bool block_c) {
      auto calc = cublasdx::make_shared_storage_calc()
                  .add(cublasdx::alignment_of_v_a<GEMM>, sizeof(typename GEMM::a_value_type), GEMM::get_layout_smem_a())
                  .add(cublasdx::alignment_of_v_b<GEMM>, sizeof(typename GEMM::b_value_type), GEMM::get_layout_smem_b());
      if (block_a) {
        calc.add(cublasdx::alignment_of_v_a<GEMM>, sizeof(typename GEMM::a_value_type), GEMM::get_layout_smem_a());
      }
      if (block_b) {
        calc.add(cublasdx::alignment_of_v_b<GEMM>, sizeof(typename GEMM::b_value_type), GEMM::get_layout_smem_b());
      }
#ifdef MRA_CUBLASDX_BLOCK_C
      if (block_c) {
        // double buffering of C
        calc.add(cublasdx::alignment_of_v_c<GEMM>, sizeof(typename GEMM::c_value_type), GEMM::get_layout_smem_c());
        calc.add(cublasdx::alignment_of_v_c<GEMM>, sizeof(typename GEMM::c_value_type), GEMM::get_layout_smem_c());
      }
#endif // MRA_CUBLASDX_BLOCK_C

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
      constexpr auto max_mn = cublasdx_max_mn<T, K>();
      using GEMMBlockA_ = decltype(BaseGEMM() + cublasdx::Size<std::min(max_mn, K*K), K, K>()
                                             + cublasdx::Arrangement<cublasdx::col_major, cublasdx::row_major, cublasdx::row_major>());
      using GEMMBLockA_suggested_ld = cublasdx::suggested_leading_dimension_of_t<GEMMBlockA_, MRA_CUBLASDX_SM>;
      using GEMMBlockA = decltype(GEMMBlockA_() + GEMMBLockA_suggested_ld());
      using GEMMBlockB_ = decltype(BaseGEMM() + cublasdx::Size<K, std::min(max_mn, K*K), K>()
                                             + cublasdx::Arrangement<cublasdx::col_major, cublasdx::row_major, cublasdx::row_major>());
      using GEMMBLockB_suggested_ld = cublasdx::suggested_leading_dimension_of_t<GEMMBlockB_, MRA_CUBLASDX_SM>;
      using GEMMBlockB = decltype(GEMMBlockB_() + GEMMBLockB_suggested_ld());
      auto size = std::max(cublasdx_shmem_size_for<GEMMBlockA>(true, false, true),
                           cublasdx_shmem_size_for<GEMMBlockB>(false, true, true));
      return size;
    }

    template<size_type M, size_type N, size_type K, typename aT, typename bT, typename cT>
    __device__ void mTxmq_cublasdx_block(cT* c, aT* a, bT* b, bool all_shared = false) {
      constexpr auto blockdims = mra::max_thread_dims(K);
      if (all_shared) {
        using BaseGEMM = decltype(cublasdx::Size<M, N, K>()
                            + cublasdx::Precision<cT>()
                            + cublasdx::Type<cublasdx::type::real>()
                            + cublasdx::Function<cublasdx::function::MM>()
                            + cublasdx::Arrangement<cublasdx::col_major, cublasdx::row_major, cublasdx::row_major>()
                            + cublasdx::SM<MRA_CUBLASDX_SM>()
                            + cublasdx::Block()
                            + cublasdx::BlockDim<blockdims.x, blockdims.y, blockdims.z>()
                            + cublasdx::MaxAlignment());
        using suggested_ld = cublasdx::suggested_leading_dimension_of_t<BaseGEMM, MRA_CUBLASDX_SM>;
        using GEMM = decltype(BaseGEMM() + suggested_ld());
        auto a_shared_tensor   = cublasdx::make_tensor(a,   GEMM::get_layout_smem_a());
        auto b_shared_tensor   = cublasdx::make_tensor(b,   GEMM::get_layout_smem_b());
        auto c_shared_tensor   = cublasdx::make_tensor(c,   GEMM::get_layout_smem_c());
        mTxmq_cublasdx_core<GEMM>(a_shared_tensor, b_shared_tensor, c_shared_tensor,
                                  [](){}, [](){});
        return;
      }
      extern SHARED __align__(16) char smem[];
      constexpr auto max_mn = cublasdx_max_mn<cT, K>();
      using BaseGEMM = decltype(cublasdx::Size<std::min(max_mn, M), std::min(max_mn, N), K>()
                          + cublasdx::Precision<cT>()
                          + cublasdx::Type<cublasdx::type::real>()
                          + cublasdx::Function<cublasdx::function::MM>()
                          + cublasdx::Arrangement<cublasdx::col_major, cublasdx::row_major, cublasdx::row_major>()
                          + cublasdx::SM<MRA_CUBLASDX_SM>()
                          + cublasdx::Block()
                          + cublasdx::BlockDim<blockdims.x, blockdims.y, blockdims.z>()
                          + cublasdx::MaxAlignment());
      using suggested_ld = cublasdx::suggested_leading_dimension_of_t<BaseGEMM, MRA_CUBLASDX_SM>;
      using GEMM = decltype(BaseGEMM() + suggested_ld());

      using alignment = cublasdx::alignment_of<GEMM>;


      if constexpr (M == K*K) {
        constexpr auto num_iter = M/max_mn;
        //if (is_team_lead()) printf("mTxmq_cublasdx_block: max_mn %d, shared_memory %u, smem %p, M = %d, N = %d, K = %d iter %d\n", max_mn, cublasdx_shmem_size_for<GEMM>(true, false, true), smem, M, N, K, num_iter);
        //__syncthreads();

        if constexpr (num_iter > 0) {
          auto [smem_a, smem_b, smem_a_n, smem_c, smem_c_n] =
            cublasdx::slice_shared_memory_generic<GEMM::a_value_type, GEMM::b_value_type, GEMM::a_value_type,
                                                  GEMM::c_value_type, GEMM::c_value_type>(
                smem,
                cute::make_tuple(cublasdx::cosize(GEMM::get_layout_smem_a()), cublasdx::alignment_of_v_a<GEMM>),
                cute::make_tuple(cublasdx::cosize(GEMM::get_layout_smem_b()), cublasdx::alignment_of_v_b<GEMM>),
                cute::make_tuple(cublasdx::cosize(GEMM::get_layout_smem_a()), cublasdx::alignment_of_v_a<GEMM>),
                cute::make_tuple(cublasdx::cosize(GEMM::get_layout_smem_c()), cublasdx::alignment_of_v_c<GEMM>),
                cute::make_tuple(cublasdx::cosize(GEMM::get_layout_smem_c()), cublasdx::alignment_of_v_c<GEMM>));

          /* copy b tensor into shared memory and leave there */
          auto b_global_tensor = cublasdx::make_tensor(b, GEMM::get_layout_gmem_b());
          auto b_shared_tensor = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b());
          cublasdx::copy<GEMM, alignment::b>(b_global_tensor, b_shared_tensor);

          auto a_shared_tensor   = cublasdx::make_tensor(smem_a,   GEMM::get_layout_smem_a());
          auto a_shared_tensor_n = cublasdx::make_tensor(smem_a_n, GEMM::get_layout_smem_a());
          auto c_shared_tensor   = cublasdx::make_tensor(smem_c,   GEMM::get_layout_smem_c());
          auto c_shared_tensor_n = cublasdx::make_tensor(smem_c_n, GEMM::get_layout_smem_c());

          int i;

          auto store_c = [&](auto& c_shared_tensor) {
            auto c_global_tensor = cublasdx::make_tensor(c+(((i-1)*max_mn)*N), GEMM::get_layout_gmem_c());
            cublasdx::copy<GEMM, alignment::c>(c_shared_tensor, c_global_tensor);
          };
          for (i = 0; i < num_iter; i++) {
            // Make global memory tensors
            auto a_global_tensor = cublasdx::make_tensor(a+(i*max_mn),     GEMM::get_layout_gmem_a(cute::Int<M>{}));
            //auto c_global_tensor = cublasdx::make_tensor(c+((i*max_mn)*N), GEMM::get_layout_gmem_c());
            mTxmq_cublasdx_core<GEMM>(a_shared_tensor, b_shared_tensor,
#ifdef MRA_CUBLASDX_BLOCK_C
			              c_shared_tensor,
#else  // MRA_CUBLASDX_BLOCK_C
				      c_global_tensor,
#endif // MRA_CUBLASDX_BLOCK_C
                                      [&](){
                                        /* load only on first iteration, all others are prefetched */
                                        if (i == 0) {
                                          //if (is_team_lead()) printf("Loading initial block %d\n", i);
                                          cublasdx::copy<GEMM, alignment::a>(a_global_tensor, a_shared_tensor);
                                        }
                                      },
                                      [&](){
                                        /* prefetch into shared memory */
                                        if ((i+1) < num_iter) {
                                          //if (is_team_lead()) printf("Prefetching block %d\n", i);
                                          auto a_global_tensor = cublasdx::make_tensor(a+((i+1)*max_mn), GEMM::get_layout_gmem_a(cute::Int<M>{}));
                                          cublasdx::copy<GEMM, alignment::a>(a_global_tensor, a_shared_tensor_n);
                                        }
                                        /* store prior iteration's result */
#ifdef MRA_CUBLASDX_BLOCK_C
                                        if (i > 0) {
                                          //if (is_team_lead()) printf("Storing block %d\n", i-1);
                                          store_c(c_shared_tensor_n);
                                        }
#endif // MRA_CUBLASDX_BLOCK_C
                                      });
            std::swap(a_shared_tensor, a_shared_tensor_n);
            std::swap(c_shared_tensor, c_shared_tensor_n);
          }
          /* store the last block of C */
          store_c(c_shared_tensor_n);
        }

        /* handle remainder */
        constexpr const auto R = M%max_mn;
        if constexpr (0 < R) {
          // Make global memory tensors
          using BaseGEMM = decltype(cublasdx::Size<R, N, K>()
                              + cublasdx::Precision<cT>()
                              + cublasdx::Type<cublasdx::type::real>()
                              + cublasdx::Function<cublasdx::function::MM>()
                              + cublasdx::Arrangement<cublasdx::col_major, cublasdx::row_major, cublasdx::row_major>()
                              + cublasdx::SM<MRA_CUBLASDX_SM>()
                              + cublasdx::Block()
                              + cublasdx::BlockDim<blockdims.x, blockdims.y, blockdims.z>()
                              + cublasdx::MaxAlignment());
	  using suggested_ld = cublasdx::suggested_leading_dimension_of_t<BaseGEMM, MRA_CUBLASDX_SM>;
	  using GEMM = decltype(BaseGEMM() + suggested_ld());
          auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<GEMM>(smem, GEMM::get_layout_smem_a(),
                                                                                    GEMM::get_layout_smem_b(),
                                                                                    GEMM::get_layout_smem_c());
          auto a_shared_tensor = cublasdx::make_tensor(smem_a, GEMM::get_layout_smem_a());
          auto a_global_tensor = cublasdx::make_tensor(a+((M/max_mn)*max_mn),   GEMM::get_layout_gmem_a(cute::Int<M>{}));
          auto b_global_tensor = cublasdx::make_tensor(b, GEMM::get_layout_gmem_b());
          auto b_shared_tensor = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b());
          auto c_global_tensor = cublasdx::make_tensor(c+((M/max_mn)*max_mn*N), GEMM::get_layout_gmem_c());
          auto c_shared_tensor = cublasdx::make_tensor(smem_c, GEMM::get_layout_smem_c());
          mTxmq_cublasdx_core<GEMM>(a_shared_tensor, b_shared_tensor,
#ifdef MRA_CUBLASDX_BLOCK_C
			            c_shared_tensor,
#else  // MRA_CUBLASDX_BLOCK_C
				    c_global_tensor,
#endif // MRA_CUBLASDX_BLOCK_C
                                    [&](){
                                      cublasdx::copy<GEMM, alignment::a>(a_global_tensor, a_shared_tensor);
                                      cublasdx::copy<GEMM, alignment::b>(b_global_tensor, b_shared_tensor);
                                    },
                                    [](){});
#ifdef MRA_CUBLASDX_BLOCK_C
          /* move the C block back to global memory */
          cublasdx::copy<GEMM, alignment::c>(c_shared_tensor, c_global_tensor);
#endif // MRA_CUBLASDX_BLOCK_C
        }
      } else {
        // TODO: implement!
        static_assert(M == K*K, "N equal to K*K currently not supported");
      }
      /* final sync */
      cublasdx::copy_wait();
    }

  } // namespace detail

  template <typename aT, typename bT, typename cT>
  __device__ void mTxmq(long dimi, long dimj, long dimk,
                        cT* c, const aT* a, const bT* b, bool all_shared = false) {
    int M = dimi;
    int N = dimj;
    int K = dimk;
    if (M == K*K) {
      // A is tall and skinny, B is square
      if (K == 6) {
        detail::mTxmq_cublasdx_block<36, 6, 6>(c, a, b, all_shared);
      } else if (K == 8) {
        detail::mTxmq_cublasdx_block<64, 8, 8>(c, a, b, all_shared);
      } else if (K == 10) {
        detail::mTxmq_cublasdx_block<100, 10, 10>(c, a, b, all_shared);
      } else if (K == 12) {
        detail::mTxmq_cublasdx_block<12*12, 12, 12>(c, a, b, all_shared);
      } else if (K == 16) {
        detail::mTxmq_cublasdx_block<16*16, 16, 16>(c, a, b, all_shared);
      } else if (K == 20) {
        detail::mTxmq_cublasdx_block<400, 20, 20>(c, a, b, all_shared);
      } else if (K == 32) {
        detail::mTxmq_cublasdx_block<32*32, 32, 32>(c, a, b, all_shared);
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
      case 32: return detail::cublasdx_shmem_size_k<T, 32>();
      default: THROW("CUBLASdx: Unsupported K");
    }
  }

} // namespace mra


#define MRA_HAVE_MTXMQ 1

#endif // __has_include(<cublasdx.hpp>)

#endif // MRA_OPS_MXM_CUBLASDX_H
