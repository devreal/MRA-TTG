
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

endif()

if (HAVE_CUDA)
  add_executable(mra-mad-deriv-device-cuda-parsec ${_deriv_cu})
  target_link_libraries(mra-mad-deriv-device-cuda-parsec PUBLIC MADworld MADtensor MADmra mra::cuda)

  add_executable(mra-mad-conv-device-cuda-parsec ${_conv_cu} ../autocorr.cc)
  target_link_libraries(mra-mad-conv-device-cuda-parsec PUBLIC MADworld MADtensor MADmra mra::cuda)
endif()

if (HAVE_HIP)
  add_executable(mra-mad-deriv-device-hip-parsec ${_deriv_hip})
  target_link_libraries(mra-mad-deriv-device-hip-parsec PUBLIC mra::hip)

  add_executable(mra-mad-conv-device-hip-parsec ${_conv_hip} ../autocorr.cc)
  target_link_libraries(mra-mad-conv-device-hip-parsec PUBLIC MADworld MADtensor MADmra mra::hip)
endif()
