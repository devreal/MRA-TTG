#ifndef MRA_KERNELS_TRANSFORM_CUBLASDX_H
#define MRA_KERNELS_TRANSFORM_CUBLASDX_H

#include "mra/tensor/tensorview.h"

namespace mra {

#if __has_include(<cublasdx.hpp>)
#include <cublasdx.hpp>

template <typename T, int K>
__forceinline__ __device__
void transform_cublasdx_k(
    const T* t, // input tensor
    const T* c, // input matrix
    T* result)
{
  constexpr const int ndim = 3; // fixed for benchmark
  using GEMM = typename mra::detail::GEMMBuilder<T, K*K, K, K,
                                                cublasdx::col_major,
                                                cublasdx::row_major,
                                                cublasdx::row_major>::GEMM;

  using alignment = cublasdx::alignment_of<GEMM>;

  extern __shared__ __align__(16) char smem[];

  auto [smem_a, smem_b] =
    cublasdx::shared_memory::slice_into_pointers<GEMM::a_value_type, GEMM::b_value_type>(
        smem,
        cublasdx::alignment_of_v_a<GEMM>, cublasdx::cosize(GET_SHARED_LAYOUT(GEMM, a)),
        cublasdx::alignment_of_v_b<GEMM>, cublasdx::cosize(GET_SHARED_LAYOUT(GEMM, b)));


  /* global memory tensors */
  auto a_global_tensor = cublasdx::make_tensor(t, GEMM::get_layout_gmem_a());
  auto b_global_tensor = cublasdx::make_tensor(c, GEMM::get_layout_gmem_b());
  auto c_global_tensor = cublasdx::make_tensor(result, GEMM::get_layout_gmem_c());

  /* shared memory tensors */
  auto a_shared_tensor   = cublasdx::make_tensor(smem_a,   GET_SHARED_LAYOUT(GEMM, a));
  auto b_shared_tensor   = cublasdx::make_tensor(smem_b,   GET_SHARED_LAYOUT(GEMM, b));

  cublasdx::copy<GEMM, alignment::a>(a_global_tensor, a_shared_tensor);
  cublasdx::copy<GEMM, alignment::b>(b_global_tensor, b_shared_tensor);

  /* wait for loads to complete */
  cublasdx::copy_wait();

  for (int n=0; n<ndim; ++n) {
    /* execute the GEMM operation */
    auto [c_register_fragment, partitioner] = GEMM().execute(a_shared_tensor, b_shared_tensor);

    /* wait for all threads to complete so we can write the result back to shared memory */
    __syncthreads();

    /* copy the result over to the the A shared tensor */
    cublasdx::copy_fragment<alignment::a>(c_register_fragment, a_shared_tensor, partitioner);

    /* wait for stores to complete */
    cublasdx::copy_wait();
  }
  /* copy the result from shared memory to global memory */
  cublasdx::copy<GEMM, alignment::c>(a_shared_tensor, c_global_tensor);

  /* wait for the copy to complete */
  cublasdx::copy_wait();
}


template <typename T, Dimension NDIM>
__forceinline__ __device__
bool transform_cublasdx(
  const TensorView<T, NDIM>& t,
  const TensorView<T, 2>& c,
  TensorView<T, NDIM>& result)
{
  int K = t.dim(0);

  switch (K) {
    case 8:
      transform_cublasdx_k<8>(t.data(), c.data(), result.data());
      return true;
    //case 10:
    //  transform_cublasdx_k<10>(t.data(), c.data(), result.data());
    //  return true;
    case 16:
      transform_cublasdx_k<16>(t.data(), c.data(), result.data());
      return true;
    //case 20:
    //  transform_cublasdx_k<20>(t.data(), c.data(), result.data());
    //  return true;
    default:
      break;
  }
  return false;
}

#else // __has_include(<cublasdx.hpp>)

template <typename T, Dimension NDIM>
bool transform_cublasdx(
    const TensorView<T, NDIM>& t,
    const TensorView<T, 2>& c,
    TensorView<T, NDIM>& result)
{
  return false;
}

#endif

} // namespace mra


#endif // __has_include(<cublasdx.hpp>)
