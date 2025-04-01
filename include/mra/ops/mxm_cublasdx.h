#ifndef MRA_OPS_MXM_CUBLASDX_H
#define MRA_OPS_MXM_CUBLASDX_H

#include <cublasdx.hpp>

#include "mra/misc/types.h"

namespace mra::detail {

  constexpr const size_type MRA_CUBLASDX_MIN_K = 3;
  constexpr const size_type MRA_CUBLASDX_MAX_K = 30;

  template<typename aT, typename bT, typename cT, bool Q>
  struct mTxm_a_cublasdx {

    template<size_type K>
    using gemm_type = decltype(cublasdx::Size<K*K, K, K>()
                        + cublasdx::Precision<T>()
                        + cublasdx::Type<cublasdx::type::real>()
                        + cublasdx::Arrangement<cublasdx::col_major, cublasdx::row_major>()
                        + cublasdx::Function<cublasdx::function::MM>()
                        + Block());
    /* explicitly instantiate all GEMM for K=[5, 30) */
    auto gemm_tuple = std::tuple<gemm_type<5>, gemm_type<6>, gemm_type<7>, gemm_type<8>, gemm_type<9>,
                                 gemm_type<10>, gemm_type<11>, gemm_type<12>, gemm_type<13>, gemm_type<14>, gemm_type<14>,
                                 /* GEMM for 2K (compressed form) */
                                 gemm_type<16>, gemm_type<18>, gemm_type<20>, gemm_type<22>, gemm_type<24>,
                                 gemm_type<26>, gemm_type<28>, gemm_type<30>>;

    template <size_type K = MRA_CUBLASDX_MIN_K>
    bool run_gemm(size_type k, cT* __restrict__ c, const aT* a, const bT* b) {

      if constexpr (K < MRA_CUBLASDX_MIN_K) return false;
      else if constexpr (K > MRA_CUBLASDX_MAX_K) return false;
      } else if (k > K) {
        return run_gemm<K+1>(k, c, a, b);
      } else {

        using GEMM = decltype(cublasdx::Size<K*K, K, K>()
                      + cublasdx::Precision<T>()
                      + cublasdx::Type<cublasdx::type::real>()
                      + cublasdx::Arrangement<cublasdx::col_major, cublasdx::row_major>()
                      + cublasdx::Function<cublasdx::function::MM>()
                      + Block());
        auto a_global_tensor = cublasdx::make_tensor(a, BLAS::get_layout_gmem_a());
        auto b_global_tensor = cublasdx::make_tensor(b, BLAS::get_layout_gmem_b());
        auto c_global_tensor = cublasdx::make_tensor(c, BLAS::get_layout_gmem_c());

        GEMM().execute(1.0, a_global_tensor, b_global_tensor, (Q ? 1.0 : 0.0), c_global_tensor);
        SYNCTHREADS();
        return true;
      }
    }

    bool operator()(size_type k, T* __restrict__ C, const T* A, const T* B) {
      if constexpr (k < MRA_CUBLASDX_MIN_K) return false;
      else if constexpr (k > MRA_CUBLASDX_MAX_K) return false;
      // start from the bottom and search upwards
      return run_gemm(k, C, A, B);
    }
  }

  template <typename aT, typename bT, typename cT, bool Q = false>
  SCOPE bool mTxm_block_a(size_type dimi, size_type dimj, size_type dimk,
                          cT* __restrict__ c, const aT* a, const bT* b) {
    if (dimi != dimj*dimj || dimj != dimk) {
      /* TODO: support more variations */
      return false;
    }


  }

} // mra::detail

#endif // MRA_OPS_MXM_CUBLASDX_H