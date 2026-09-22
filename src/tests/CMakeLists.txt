
# ── Data-structure unit tests ─────────────────────────────────────────────────
add_executable(mra-cycledimtest cycledim_test.cc)
target_link_libraries(mra-cycledimtest PUBLIC mra::host)

add_executable(mra-gaxpytest gaxpy_test.cc)
target_link_libraries(mra-gaxpytest PUBLIC mra::host)

add_executable(mra-tensortest tensor_test.cc)
target_link_libraries(mra-tensortest PUBLIC mra::host)

add_executable(mra-matrixtest matrix_test.cc)
target_link_libraries(mra-matrixtest PUBLIC mra::host)

# ── MADNESS integration tests ─────────────────────────────────────────────────
add_executable(mra-mad-deriv-host-parsec mad_derivative_test.cc)
target_link_libraries(mra-mad-deriv-host-parsec PUBLIC MADworld MADtensor mra::host)

add_executable(mra-mad-conv-host-parsec mad_convolution_test.cc ../autocorr.cc)
target_link_libraries(mra-mad-conv-host-parsec PUBLIC MADworld MADtensor mra::host)

# Host-only (no GPU needed): dumps real MADNESS GaussianConvolution1D
# reference data + nonstandard()-computed R/S/norms to a C++ header, used to
# numerically validate generate_op1d_pure_cuda.h (the dependency-free CUDA
# port of include/mra/kernels/generate_op1d.h) against MADNESS as ground
# truth -- see generate_op1d_compare_madness.cu at the repo root. When built
# for CUDA/HIP (below), the same source also runs that comparison directly.
add_executable(mra-generate-op1d-reference-dump generate_op1d_reference_dump.cc)
target_link_libraries(mra-generate-op1d-reference-dump PUBLIC MADworld MADtensor mra::host)

add_executable(mra-mad-vmra-host-parsec mad_vmra_test.cc ../autocorr.cc)
target_link_libraries(mra-mad-vmra-host-parsec PUBLIC MADworld MADtensor mra::host)

if (HAVE_CUDA OR HAVE_HIP)
  mra_generate_gpu_wrappers(
    SOURCES mad_derivative_test.cc
    CUDA_OUTPUT_VAR _deriv_cu
    HIP_OUTPUT_VAR  _deriv_hip
  )
  mra_generate_gpu_wrappers(
    SOURCES mad_convolution_test.cc
    CUDA_OUTPUT_VAR _conv_cu
    HIP_OUTPUT_VAR  _conv_hip
  )

  mra_generate_gpu_wrappers(
    SOURCES mad_vmra_test.cc
    CUDA_OUTPUT_VAR _vmra_cu
    HIP_OUTPUT_VAR  _vmra_hip
  )

  # CUDA only for now: generate_op1d_pure_cuda.h (repo root) is written
  # directly against <cuda_runtime.h>, not a portable SCOPE/DEVSCOPE kernel,
  # so there's no HIP counterpart to wrap yet.
  mra_generate_gpu_wrappers(
    SOURCES generate_op1d_reference_dump.cc
    CUDA_OUTPUT_VAR _op1d_ref_dump_cu
  )

endif()

if (HAVE_CUDA)
  add_executable(mra-mad-deriv-device-cuda-parsec ${_deriv_cu})
  target_link_libraries(mra-mad-deriv-device-cuda-parsec PUBLIC MADworld MADtensor MADmra mra::cuda)

  add_executable(mra-mad-conv-device-cuda-parsec ${_conv_cu})
  target_link_libraries(mra-mad-conv-device-cuda-parsec PUBLIC MADworld MADtensor MADmra mra::cuda)

  add_executable(mra-mad-vmra-device-cuda-parsec ${_vmra_cu})
  target_link_libraries(mra-mad-vmra-device-cuda-parsec PUBLIC MADworld MADtensor MADmra mra::cuda)

  # Deliberately does NOT need mra::cuda's TTG task-graph headers -- see this
  # source's header comment for why staying dependency-light here matters.
  # Still links mra::cuda (not mra::host) for an ABI-consistent
  # mra::initialize/finalize in a CUDA-compiled translation unit, matching
  # every other -device-cuda-parsec target above.
  add_executable(mra-generate-op1d-reference-dump-device-cuda-parsec ${_op1d_ref_dump_cu})
  target_include_directories(mra-generate-op1d-reference-dump-device-cuda-parsec PRIVATE ${CMAKE_SOURCE_DIR})
  target_link_libraries(mra-generate-op1d-reference-dump-device-cuda-parsec PUBLIC MADworld MADtensor MADmra mra::cuda)
endif()

if (HAVE_HIP)
  add_executable(mra-mad-deriv-device-hip-parsec ${_deriv_hip})
  target_link_libraries(mra-mad-deriv-device-hip-parsec PUBLIC mra::hip)

  add_executable(mra-mad-conv-device-hip-parsec ${_conv_hip})
  target_link_libraries(mra-mad-conv-device-hip-parsec PUBLIC MADworld MADtensor MADmra mra::hip)

  add_executable(mra-mad-vmra-device-hip-parsec ${_vmra_hip})
  target_link_libraries(mra-mad-vmra-device-hip-parsec PUBLIC MADworld MADtensor MADmra mra::hip)
endif()
