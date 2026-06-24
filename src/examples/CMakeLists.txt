
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

# ── CUDA examples ─────────────────────────────────────────────────────────────
if (HAVE_CUDA)
  add_executable(mra-pcr-device-cuda-parsec pcr/test_pcr.cu)
  target_link_libraries(mra-pcr-device-cuda-parsec PUBLIC mra::cuda)

  add_executable(mra-deriv-device-cuda-parsec derivative/bench_derivative.cu)
  target_link_libraries(mra-deriv-device-cuda-parsec PUBLIC mra::cuda)

  add_executable(mra-convolution-device-cuda-parsec ../autocorr.cc convolution/test_convolution.cu)
  target_link_libraries(mra-convolution-device-cuda-parsec PUBLIC mra::cuda)

  add_executable(mra-unit-device-cuda-parsec unit/test_unit.cu)
  target_link_libraries(mra-unit-device-cuda-parsec PUBLIC mra::cuda)
endif()

# ── HIP examples ──────────────────────────────────────────────────────────────
if (HAVE_HIP)
  add_executable(mra-pcr-device-hip-parsec pcr/test_pcr.hip)
  target_link_libraries(mra-pcr-device-hip-parsec PUBLIC mra::hip)

  add_executable(mra-deriv-device-hip-parsec derivative/bench_derivative.hip)
  target_link_libraries(mra-deriv-device-hip-parsec PUBLIC mra::hip)

  add_executable(mra-convolution-device-hip-parsec ../autocorr.cc convolution/test_convolution.hip)
  target_link_libraries(mra-convolution-device-hip-parsec PUBLIC mra::hip)

  add_executable(mra-unit-device-hip-parsec unit/test_unit.hip)
  target_link_libraries(mra-unit-device-hip-parsec PUBLIC mra::hip)
endif()
