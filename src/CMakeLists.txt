
# define a base target for the MRA
add_library(libmra-base-parsec INTERFACE)
target_include_directories(libmra-base-parsec INTERFACE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(libmra-base-parsec INTERFACE MADworld ttg-parsec $<$<TARGET_EXISTS:blaspp>:blaspp>)
target_compile_definitions(libmra-base-parsec INTERFACE TTG_USE_PARSEC=1)

#define a target for the host
add_library(libmra-host-parsec OBJECT gl.cc twoscale.cc kernels/compress.cc kernels/reconstruct.cc kernels/multiply.cc kernels/derivative.cc)
target_link_libraries(libmra-host-parsec PUBLIC libmra-base-parsec)
target_compile_definitions(libmra-host-parsec PUBLIC MRA_ENABLE_HOST=1)

add_executable(mad-pcr
               examples/pcr/test_pcr_madness.cc)
target_link_libraries(mad-pcr PUBLIC MADchem)
target_include_directories(mad-pcr PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_executable(mra-pcr-host-parsec
               examples/pcr/test_pcr.cc)
target_link_libraries(mra-pcr-host-parsec PUBLIC libmra-host-parsec)

add_executable(mra-deriv-host-parsec
               examples/derivative/bench_derivative.cc)
target_link_libraries(mra-deriv-host-parsec PUBLIC libmra-host-parsec)

add_executable(mad-bench-derivative
               examples/derivative/bench_mad_derivative.cc)
target_link_libraries(mad-bench-derivative PUBLIC MADchem)
target_include_directories(mad-bench-derivative PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_executable(mra-unit-host-parsec
               examples/unit/test_unit.cc)
target_link_libraries(mra-unit-host-parsec PUBLIC libmra-host-parsec)

# Holding this off until BLAS in MADNESS works for our use case
add_executable(mra-mad-deriv-host-parsec
               tests/mad_derivative_test.cc)
target_link_libraries(mra-mad-deriv-host-parsec PUBLIC MADworld MADtensor MADmra libmra-host-parsec)

add_executable(mra-cycledimtest tests/cycledim_test.cc)
target_link_libraries(mra-cycledimtest PUBLIC libmra-host-parsec)

add_executable(mra-gaxpytest tests/gaxpy_test.cc)
target_link_libraries(mra-gaxpytest PUBLIC libmra-host-parsec)

# test to verify that tensor slices work
add_executable(mra-tensortest tests/tensor_test.cc)
target_link_libraries(mra-tensortest PUBLIC libmra-host-parsec)

# test to verify that matrix multiplication works
add_executable(mra-matrixtest tests/matrix_test.cc)
target_link_libraries(mra-matrixtest PUBLIC libmra-host-parsec)

#test to verify transition matrices
add_executable(mra-trmatrixtest autocorr.cc tests/trmat_test.cc)
target_link_libraries(mra-trmatrixtest PUBLIC MADworld MADmra MADtensor libmra-host-parsec)

# test to verify that inner() works
# FIXME: this test breaks because of BLAS in MADNESS, the baine of our existence
#add_executable(mra-innertest madness/tests/inner_test.cc)
#target_compile_definitions(mra-innertest PUBLIC MRA_ENABLE_HOST=1 TTG_USE_PARSEC=1)
#target_link_libraries(mra-innertest PUBLIC MADworld ttg-parsec)


if (HAVE_CUDA)

  #define a target for CUDA
  add_library(libmra-cuda-base-parsec INTERFACE)
  #if (TARGET Kokkos::kokkos)
  #  target_link_libraries(libmra-cuda-base-parsec INTERFACE Kokkos::kokkos)
  #  target_compile_definitions(libmra-cuda-base-parsec INTERFACE MRA_HAVE_KOKKOS=1)
  #else (TARGET Kokkos::kokkos)
    target_compile_definitions(libmra-cuda-base-parsec INTERFACE MRA_CUDA_ARCH=${CMAKE_CUDA_ARCHITECTURES})
    target_compile_options(libmra-cuda-base-parsec INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>)
  #endif (TARGET Kokkos::kokkos)

  target_compile_definitions(libmra-cuda-base-parsec INTERFACE MRA_ENABLE_CUDA=1)
  target_link_libraries(libmra-cuda-base-parsec INTERFACE libmra-base-parsec
                                               $<$<TARGET_EXISTS:tiledarray>:tiledarray>
                                               $<$<TARGET_EXISTS:mathdx::cublasdx>:mathdx::cublasdx>)

  add_library(libmra-cuda-parsec OBJECT gl.cc twoscale.cc kernels/compress.cu kernels/reconstruct.cu kernels/multiply.cu kernels/derivative.cu)
  target_link_libraries(libmra-cuda-parsec PUBLIC libmra-cuda-base-parsec)

  add_executable(mra-pcr-device-cuda-parsec examples/pcr/test_pcr.cu)
  target_link_libraries(mra-pcr-device-cuda-parsec PUBLIC libmra-cuda-parsec)

  add_executable(mra-deriv-device-cuda-parsec examples/derivative/bench_derivative.cu)
  target_link_libraries(mra-deriv-device-cuda-parsec PUBLIC libmra-cuda-parsec)

  add_executable(mra-unit-device-cuda-parsec examples/unit/test_unit.cu)
  target_link_libraries(mra-unit-device-cuda-parsec PUBLIC libmra-cuda-parsec)

  add_executable(mra-mad-deriv-device-cuda-parsec
                 tests/mad_derivative_test.cu)
  target_link_libraries(mra-mad-deriv-device-cuda-parsec PUBLIC MADworld MADtensor MADmra libmra-cuda-parsec)

  if (TARGET Kokkos::kokkos)
    add_executable(transformbench-cuda-kokkos-parsec benchmarks/transformbench/transformbench.cu)
    target_link_libraries(transformbench-cuda-kokkos-parsec PUBLIC libmra-cuda-base-parsec Kokkos::kokkos)
    target_compile_definitions(transformbench-cuda-kokkos-parsec PUBLIC MRA_HAVE_KOKKOS=1)
  endif (TARGET Kokkos::kokkos)

  add_executable(transformbench-cuda-parsec benchmarks/transformbench/transformbench.cu)
  target_link_libraries(transformbench-cuda-parsec PUBLIC libmra-cuda-parsec)

  find_package(CUDAToolkit)
  if (TARGET CUDA::cublas AND TARGET mathdx::cublasdx)
    add_executable(mxm-test-cuda tests/mxm_test.cu)
    target_link_libraries(mxm-test-cuda PUBLIC CUDA::cublas libmra-cuda-base-parsec)
  endif (TARGET CUDA::cublas AND TARGET mathdx::cublasdx)

endif (HAVE_CUDA)

if (HAVE_HIP)

  # define a target for HIP
  add_library(libmra-hip-parsec OBJECT gl.cc twoscale.cc kernels/compress.hip kernels/reconstruct.hip kernels/multiply.hip kernels/derivative.hip)
  target_compile_definitions(libmra-hip-parsec PUBLIC MRA_ENABLE_HIP=1)
  target_link_libraries(libmra-hip-parsec PUBLIC libmra-base-parsec
                                              $<$<TARGET_EXISTS:tiledarray>:tiledarray>
                                              $<$<TARGET_EXISTS:mathdx::cublasdx>:mathdx::cublasdx>)

  add_executable(mra-pcr-device-hip-parsec examples/pcr/test_pcr.hip)
  target_link_libraries(mra-pcr-device-hip-parsec PUBLIC libmra-hip-parsec)

  add_executable(mra-deriv-device-hip-parsec examples/derivative/bench_derivative.hip)
  target_link_libraries(mra-deriv-device-hip-parsec PUBLIC libmra-hip-parsec)

  add_executable(mra-unit-device-hip-parsec examples/unit/test_unit.hip)
  target_link_libraries(mra-unit-device-hip-parsec PUBLIC libmra-hip-parsec)

  add_executable(mra-mad-deriv-device-hip-parsec
                 tests/mad_derivative_test.cu)
  target_link_libraries(mra-mad-deriv-device-hip-parsec PUBLIC MADworld MADtensor MADmra libmra-hip-parsec)

endif (HAVE_HIP)
