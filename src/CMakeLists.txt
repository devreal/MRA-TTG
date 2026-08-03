
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
if (NOT MRA_ENABLE_EXPLICIT_INSTANTIATION)
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
  target_compile_options(mra-cuda-base INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>
  )
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
