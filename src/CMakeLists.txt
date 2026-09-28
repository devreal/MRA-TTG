
# ── Base interface library ────────────────────────────────────────────────────
add_library(mra-base INTERFACE)
add_library(mra::base ALIAS mra-base)
target_include_directories(mra-base INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(mra-base INTERFACE
  MADworld MADtensor MADmra ttg-parsec
  $<$<TARGET_EXISTS:blaspp>:blaspp>
)
target_compile_definitions(mra-base INTERFACE TTG_USE_PARSEC=1)
if (MRA_ENABLE_EXPLICIT_INSTANTIATION)
  target_compile_definitions(mra-base INTERFACE MRA_ENABLE_EXPLICIT_INSTANTIATION=1)
endif()

include(MRAGPUWrappers)

# ── Host (CPU) implementation ─────────────────────────────────────────────────
set(MRA_SOURCES gl.cc init.cc twoscale.cc)
set(MRA_KERNEL_SOURCES "")
if (MRA_ENABLE_EXPLICIT_INSTANTIATION)
  set(MRA_KERNEL_SOURCES
    kernels/compress.cc kernels/reconstruct.cc kernels/multiply.cc
    kernels/derivative.cc kernels/convolution.cc)
endif()

add_library(mra-host STATIC ${MRA_SOURCES} ${MRA_KERNEL_SOURCES})
add_library(mra::host ALIAS mra-host)
target_link_libraries(mra-host PUBLIC mra-base)
target_compile_definitions(mra-host PUBLIC MRA_ENABLE_HOST=1)

# ── CUDA implementation ───────────────────────────────────────────────────────
if (HAVE_CUDA)

  add_library(mra-cuda-base INTERFACE)
  add_library(mra::cuda-base ALIAS mra-cuda-base)
  target_compile_definitions(mra-cuda-base INTERFACE
    MRA_CUDA_ARCH=${CMAKE_CUDA_ARCHITECTURES}
    MRA_ENABLE_CUDA=1
  )
  set(_mra_cuda_compile_options --expt-relaxed-constexpr --extended-lambda)

  if (MRA_CUDA_SPLIT_COMPILE)
    list(APPEND _mra_cuda_compile_options --split-compile=${MRA_CUDA_SPLIT_COMPILE_THREADS})

    if (MRA_CUDA_SPLIT_COMPILE_JOBSERVER)
      # nvcc's --jobserver flag was added in CUDA 13.0, requires GNU Make >= 4.3,
      # and is Linux-only; see
      # https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html
      set(_mra_jobserver_supported FALSE)
      if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "13.0"
          AND CMAKE_GENERATOR MATCHES "Makefiles"
          AND CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
        execute_process(
          COMMAND ${CMAKE_MAKE_PROGRAM} --version
          OUTPUT_VARIABLE _mra_make_version_output
          ERROR_QUIET
        )
        if (_mra_make_version_output MATCHES "GNU Make ([0-9]+)\\.([0-9]+)")
          if (CMAKE_MATCH_1 GREATER 4 OR (CMAKE_MATCH_1 EQUAL 4 AND CMAKE_MATCH_2 GREATER_EQUAL 3))
            set(_mra_jobserver_supported TRUE)
          endif()
        endif()
        unset(_mra_make_version_output)
      endif()

      if (_mra_jobserver_supported)
        list(APPEND _mra_cuda_compile_options --jobserver)
        message(STATUS "nvcc --split-compile will coordinate with GNU Make's jobserver (build with e.g. 'cmake --build ${CMAKE_BINARY_DIR} -- -j$(nproc)')")
      else()
        message(STATUS "MRA_CUDA_SPLIT_COMPILE_JOBSERVER requested but unsupported here "
          "(needs nvcc >= 13.0 and GNU Make >= 4.3 on Linux with the Makefiles generator; "
          "nvcc=${CMAKE_CUDA_COMPILER_VERSION}, generator=${CMAKE_GENERATOR}, host=${CMAKE_HOST_SYSTEM_NAME}); "
          "nvcc will use --split-compile=${MRA_CUDA_SPLIT_COMPILE_THREADS} without jobserver coordination")
      endif()
      unset(_mra_jobserver_supported)
    endif()
  endif()

  target_compile_options(mra-cuda-base INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:${_mra_cuda_compile_options}>
  )
  unset(_mra_cuda_compile_options)
  target_link_libraries(mra-cuda-base INTERFACE
    mra-base
    $<$<TARGET_EXISTS:mathdx::cublasdx>:mathdx::cublasdx>
  )
  if (MRA_ENABLE_JIT)
    target_compile_definitions(mra-cuda-base INTERFACE MRA_ENABLE_JIT=1)
    target_link_libraries(mra-cuda-base INTERFACE mra::jit-headers mra::jit-runtime)
  endif (MRA_ENABLE_JIT)

  if (MRA_KERNEL_SOURCES)
    mra_generate_gpu_wrappers(
      SOURCES ${MRA_KERNEL_SOURCES}
      CUDA_OUTPUT_VAR MRA_CUDA_KERNEL_SOURCES
    )
  endif()

  add_library(mra-cuda STATIC ${MRA_SOURCES} ${MRA_CUDA_KERNEL_SOURCES})
  add_library(mra::cuda ALIAS mra-cuda)
  # Embed resolved device code so downstream links are self-contained
  set_target_properties(mra-cuda PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_link_libraries(mra-cuda PUBLIC mra-cuda-base)

endif (HAVE_CUDA)

# ── HIP implementation ────────────────────────────────────────────────────────
if (HAVE_HIP)

  if (MRA_KERNEL_SOURCES)
    mra_generate_gpu_wrappers(
      SOURCES ${MRA_KERNEL_SOURCES}
      HIP_OUTPUT_VAR MRA_HIP_KERNEL_SOURCES
    )
  endif()

  add_library(mra-hip STATIC ${MRA_SOURCES} ${MRA_HIP_KERNEL_SOURCES})
  add_library(mra::hip ALIAS mra-hip)
  target_compile_definitions(mra-hip PUBLIC MRA_ENABLE_HIP=1)
  target_link_libraries(mra-hip PUBLIC
    mra-base
    $<$<TARGET_EXISTS:mathdx::cublasdx>:mathdx::cublasdx>
  )

endif (HAVE_HIP)

# ── Optional subdirectories ───────────────────────────────────────────────────
if (MRA_BUILD_TESTS)
  add_subdirectory(tests)
endif()

if (MRA_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if (MRA_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
