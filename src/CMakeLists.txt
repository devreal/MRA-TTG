include_directories(../include/)
include_directories(../include/mra)

add_executable(mra-pcr-host-parsec
               examples/pcr/test_pcr.cc
               gl.cc
               twoscale.cc)
target_compile_definitions(mra-pcr-host-parsec PUBLIC MRA_ENABLE_HOST=1 TTG_USE_PARSEC=1)
target_link_libraries(mra-pcr-host-parsec PUBLIC MADworld ttg-parsec)

add_executable(mad-pcr
               examples/pcr/test_pcr_madness.cc)
target_link_libraries(mad-pcr PUBLIC MADchem)


add_executable(mra-derivative-host-parsec
               examples/derivative/test_derivative.cc
               gl.cc
               twoscale.cc)
target_compile_definitions(mra-derivative-host-parsec PUBLIC MRA_ENABLE_HOST=1 TTG_USE_PARSEC=1)
target_link_libraries(mra-derivative-host-parsec PUBLIC MADworld ttg-parsec)

add_executable(mra-unit-host-parsec
               examples/unit/test_unit.cc
               gl.cc
               twoscale.cc)
target_compile_definitions(mra-unit-host-parsec PUBLIC MRA_ENABLE_HOST=1 TTG_USE_PARSEC=1)
target_link_libraries(mra-unit-host-parsec PUBLIC MADworld ttg-parsec)

# Holding this off until BLAS in MADNESS works for our use case
add_executable(mra-mad-host-parsec
               tests/mad_derivative_test.cc
               gl.cc
               twoscale.cc)
target_compile_definitions(mra-mad-host-parsec PUBLIC MRA_ENABLE_HOST=1 TTG_USE_PARSEC=1)
target_link_libraries(mra-mad-host-parsec PUBLIC MADworld MADtensor MADmra ttg-parsec)

add_executable(mra-cycledimtest tests/cycledim_test.cc)
target_compile_definitions(mra-cycledimtest PUBLIC MRA_ENABLE_HOST=1 TTG_USE_PARSEC=1)
target_link_libraries(mra-cycledimtest PUBLIC MADworld ttg-parsec)

add_executable(mra-gaxpytest tests/gaxpy_test.cc)
target_compile_definitions(mra-gaxpytest PUBLIC MRA_ENABLE_HOST=1 TTG_USE_PARSEC=1)
target_link_libraries(mra-gaxpytest PUBLIC MADworld ttg-parsec)

# test to verify that tensor slices work
add_executable(mra-tensortest tests/tensor_test.cc)
target_compile_definitions(mra-tensortest PUBLIC MRA_ENABLE_HOST=1 TTG_USE_PARSEC=1)
target_link_libraries(mra-tensortest PUBLIC MADworld ttg-parsec)

# test to verify that matrix multiplication works
add_executable(mra-matrixtest tests/matrix_test.cc)
target_compile_definitions(mra-matrixtest PUBLIC MRA_ENABLE_HOST=1 TTG_USE_PARSEC=1)
target_link_libraries(mra-matrixtest PUBLIC MADworld ttg-parsec)

# test to verify that inner() works
# FIXME: this test breaks because of BLAS in MADNESS, the baine of our existence
#add_executable(mra-innertest madness/tests/inner_test.cc)
#target_compile_definitions(mra-innertest PUBLIC MRA_ENABLE_HOST=1 TTG_USE_PARSEC=1)
#target_link_libraries(mra-innertest PUBLIC MADworld ttg-parsec)


if (HAVE_CUDA)

  add_executable(mra-pcr-device-cuda-parsec
                     examples/pcr/test_pcr.cu
                     gl.cc
                     twoscale.cc)
  target_compile_definitions(mra-pcr-device-cuda-parsec PUBLIC MRA_ENABLE_CUDA=1 TTG_USE_PARSEC=1 $<$<COMPILE_LANGUAGE:CUDA>:MRA_CUDA_ARCH=${CMAKE_CUDA_ARCHITECTURES}>)
  target_compile_options(mra-pcr-device-cuda-parsec PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  target_link_libraries(mra-pcr-device-cuda-parsec PUBLIC MADworld ttg-parsec
                                                          $<$<TARGET_EXISTS:tiledarray>:tiledarray>
                                                          $<$<TARGET_EXISTS:mathdx::cublasdx>:mathdx::cublasdx>)

  add_executable(mra-derivative-device-cuda-parsec
                     examples/derivative/test_derivative.cu
                     gl.cc
                     twoscale.cc)
  target_compile_definitions(mra-derivative-device-cuda-parsec PUBLIC MRA_ENABLE_CUDA=1 TTG_USE_PARSEC=1)
  target_compile_options(mra-derivative-device-cuda-parsec PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  target_link_libraries(mra-derivative-device-cuda-parsec PUBLIC MADworld ttg-parsec $<$<TARGET_EXISTS:tiledarray>:tiledarray>)

  add_executable(mra-unit-device-cuda-parsec
                     examples/unit/test_unit.cu
                     gl.cc
                     twoscale.cc)
  target_compile_definitions(mra-unit-device-cuda-parsec PUBLIC MRA_ENABLE_CUDA=1 TTG_USE_PARSEC=1)
  target_compile_options(mra-unit-device-cuda-parsec PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  target_link_libraries(mra-unit-device-cuda-parsec PUBLIC MADworld ttg-parsec $<$<TARGET_EXISTS:tiledarray>:tiledarray>)
endif (HAVE_CUDA)

if (HAVE_HIP)

  add_executable(mra-pcr-device-hip-parsec
                     examples/pcr/test_pcr.hip
                     gl.cc
                     twoscale.cc)
  target_compile_definitions(mra-pcr-device-hip-parsec PUBLIC MRA_ENABLE_CUDA=1 TTG_USE_PARSEC=1)
  target_compile_options(mra-pcr-device-hip-parsec PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  target_link_libraries(mra-pcr-device-hip-parsec PUBLIC MADworld ttg-parsec $<$<TARGET_EXISTS:tiledarray>:tiledarray>)

  add_executable(mra-derivative-device-hip-parsec
                     examples/derivative/test_derivative.hip
                     gl.cc
                     twoscale.cc)
  target_compile_definitions(mra-derivative-device-hip-parsec PUBLIC MRA_ENABLE_CUDA=1 TTG_USE_PARSEC=1)
  target_compile_options(mra-derivative-device-hip-parsec PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  target_link_libraries(mra-derivative-device-hip-parsec PUBLIC MADworld ttg-parsec $<$<TARGET_EXISTS:tiledarray>:tiledarray>)

  add_executable(mra-unit-device-hip-parsec
                     examples/unit/test_unit.hip
                     gl.cc
                     twoscale.cc)
  target_compile_definitions(mra-unit-device-hip-parsec PUBLIC MRA_ENABLE_CUDA=1 TTG_USE_PARSEC=1)
  target_compile_options(mra-unit-device-hip-parsec PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  target_link_libraries(mra-unit-device-hip-parsec PUBLIC MADworld ttg-parsec $<$<TARGET_EXISTS:tiledarray>:tiledarray>)
endif (HAVE_HIP)
