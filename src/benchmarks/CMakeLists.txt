
# ── Convolution kernel benchmark (MADNESS separated operators) ───────────────
add_executable(convbench-host-parsec ../autocorr.cc convbench/convbench.cc)
target_link_libraries(convbench-host-parsec PUBLIC mra::host)

if (HAVE_CUDA OR HAVE_HIP)
  mra_generate_gpu_wrappers(
    SOURCES convbench/convbench.cc
    CUDA_OUTPUT_VAR _cb_cu
    HIP_OUTPUT_VAR  _cb_hip
  )
endif()

if (HAVE_CUDA)
  add_executable(convbench-cuda-parsec ../autocorr.cc ${_cb_cu})
  target_link_libraries(convbench-cuda-parsec PUBLIC mra::cuda)
endif()

if (HAVE_HIP)
  add_executable(convbench-hip-parsec ../autocorr.cc ${_cb_hip})
  target_link_libraries(convbench-hip-parsec PUBLIC mra::hip)
endif()

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
