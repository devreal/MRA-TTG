include_directories(../include/)
include_directories(../include/mra)

add_executable(mra-pcr-host-parsec
               examples/pcr/test_pcr.cc
               gl.cc
               twoscale.cc)
target_compile_definitions(mra-pcr-host-parsec PUBLIC MRA_ENABLE_HOST=1 TTG_USE_PARSEC=1)
target_link_libraries(mra-pcr-host-parsec PUBLIC MADworld ttg-parsec)

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

add_executable(transformbench-host-parsec benchmarks/transformbench/transformbench.cc)
target_compile_definitions(transformbench-host-parsec PUBLIC MRA_ENABLE_HOST=1 TTG_USE_PARSEC=1)
target_link_libraries(transformbench-host-parsec PUBLIC MADworld ttg-parsec)

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
  target_compile_definitions(mra-pcr-device-cuda-parsec PUBLIC MRA_ENABLE_CUDA=1 TTG_USE_PARSEC=1)
  target_compile_options(mra-pcr-device-cuda-parsec PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  target_link_libraries(mra-pcr-device-cuda-parsec PUBLIC MADworld ttg-parsec $<$<TARGET_EXISTS:tiledarray>:tiledarray>)

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



  add_executable(transformbench-cuda-parsec benchmarks/transformbench/transformbench.cu)
  target_compile_definitions(transformbench-cuda-parsec PUBLIC MRA_ENABLE_CUDA=1 TTG_USE_PARSEC=1)
  target_compile_options(transformbench-cuda-parsec PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  target_link_libraries(transformbench-cuda-parsec PUBLIC MADworld ttg-parsec)
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

  add_executable(transformbench-hip-parsec benchmarks/transformbench/transformbench.hip)
  target_compile_definitions(transformbench-hip-parsec PUBLIC MRA_ENABLE_HIP=1 TTG_USE_PARSEC=1)
  target_link_libraries(transformbench-hip-parsec PUBLIC MADworld ttg-parsec)
endif (HAVE_HIP)


if (TARGET Kokkos::kokkos)
  add_library(transformbench-kokkos-parsec STATIC benchmarks/transformbench-kokkos/transformbench.cc)
  target_link_libraries(transformbench-kokkos-parsec PRIVATE Kokkos::kokkos)
  target_compile_options(transformbench-kokkos-parsec PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>)
  # we still need to tell MRA whether we're using CUDA or HIP
  target_compile_definitions(transformbench-kokkos-parsec PRIVATE $<$<HAVE_CUDA>:MRA_ENABLE_CUDA=1> $<$<HAVE_HIP>:MRA_ENABLE_HIP=1> TTG_USE_PARSEC=1)
  # TODO: target_include_directories does not seem to work with CUDA :/
  #include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../ttg/ ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/potrf)
  #add_library(plgsy-host STATIC potrf/core_plgsy.cc)
  #target_link_libraries(plgsy-host PRIVATE Kokkos::kokkos)
endif(TARGET Kokkos::kokkos)
