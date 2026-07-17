
# ── Host examples ─────────────────────────────────────────────────────────────
add_executable(mra-pcr-host-parsec pcr/test_pcr.cc)
target_link_libraries(mra-pcr-host-parsec PUBLIC mra::host)

add_executable(mra-deriv-host-parsec derivative/bench_derivative.cc)
target_link_libraries(mra-deriv-host-parsec PUBLIC mra::host)

add_executable(mra-convolution-host-parsec ../autocorr.cc convolution/test_convolution.cc)
target_link_libraries(mra-convolution-host-parsec PUBLIC mra::host)

add_executable(mra-unit-host-parsec unit/test_unit.cc)
target_link_libraries(mra-unit-host-parsec PUBLIC mra::host)

# MADNESS reference executables
add_executable(mad-pcr pcr/test_pcr_madness.cc)
target_link_libraries(mad-pcr PUBLIC MADchem)
target_include_directories(mad-pcr PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_executable(mad-derivative derivative/bench_mad_derivative.cc)
target_link_libraries(mad-derivative PUBLIC MADchem)
target_include_directories(mad-derivative PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_executable(mad-convolution convolution/test_convolution_madness.cc)
target_link_libraries(mad-convolution PUBLIC mra::host)

# ── CUDA / HIP examples ───────────────────────────────────────────────────────
if (HAVE_CUDA OR HAVE_HIP)
  mra_generate_gpu_wrappers(
    SOURCES pcr/test_pcr.cc derivative/bench_derivative.cc
            convolution/test_convolution.cc unit/test_unit.cc
    CUDA_OUTPUT_VAR _example_cu
    HIP_OUTPUT_VAR  _example_hip
  )
  list(GET _example_cu 0 _pcr_cu)
  list(GET _example_cu 1 _deriv_cu)
  list(GET _example_cu 2 _conv_cu)
  list(GET _example_cu 3 _unit_cu)
  list(GET _example_hip 0 _pcr_hip)
  list(GET _example_hip 1 _deriv_hip)
  list(GET _example_hip 2 _conv_hip)
  list(GET _example_hip 3 _unit_hip)
endif()

if (HAVE_CUDA)
  add_executable(mra-pcr-device-cuda-parsec ${_pcr_cu})
  target_link_libraries(mra-pcr-device-cuda-parsec PUBLIC mra::cuda)

  add_executable(mra-deriv-device-cuda-parsec ${_deriv_cu})
  target_link_libraries(mra-deriv-device-cuda-parsec PUBLIC mra::cuda)

  add_executable(mra-convolution-device-cuda-parsec ../autocorr.cc ${_conv_cu})
  target_link_libraries(mra-convolution-device-cuda-parsec PUBLIC mra::cuda)

  add_executable(mra-unit-device-cuda-parsec ${_unit_cu})
  target_link_libraries(mra-unit-device-cuda-parsec PUBLIC mra::cuda)
endif()

if (HAVE_HIP)
  add_executable(mra-pcr-device-hip-parsec ${_pcr_hip})
  target_link_libraries(mra-pcr-device-hip-parsec PUBLIC mra::hip)

  add_executable(mra-deriv-device-hip-parsec ${_deriv_hip})
  target_link_libraries(mra-deriv-device-hip-parsec PUBLIC mra::hip)

  add_executable(mra-convolution-device-hip-parsec ../autocorr.cc ${_conv_hip})
  target_link_libraries(mra-convolution-device-hip-parsec PUBLIC mra::hip)

  add_executable(mra-unit-device-hip-parsec ${_unit_hip})
  target_link_libraries(mra-unit-device-hip-parsec PUBLIC mra::hip)
endif()
