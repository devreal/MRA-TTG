
if (HAVE_CUDA)
  mra_generate_gpu_wrappers(
    SOURCES transformbench/transformbench.cc
    CUDA_OUTPUT_VAR _tb_cu
  )

  if (TARGET Kokkos::kokkos)
    add_executable(transformbench-cuda-kokkos-parsec ${_tb_cu})
    target_link_libraries(transformbench-cuda-kokkos-parsec PUBLIC mra::cuda-base Kokkos::kokkos)
    target_compile_definitions(transformbench-cuda-kokkos-parsec PUBLIC MRA_HAVE_KOKKOS=1)
  endif()

  add_executable(transformbench-cuda-parsec ${_tb_cu})
  target_link_libraries(transformbench-cuda-parsec PUBLIC mra::cuda)

  find_package(CUDAToolkit)
  if (TARGET CUDA::cublas AND TARGET mathdx::cublasdx)
    add_executable(mxm-test-cuda ../tests/mxm_test.cu)
    target_link_libraries(mxm-test-cuda PUBLIC CUDA::cublas mra::cuda-base)
  endif()
endif()
