cmake_minimum_required(VERSION 3.19)
project(MRA-TTG VERSION 0.1.0 LANGUAGES CXX)

option(MRA_ENABLE_DEBUG "Whether to enable debug mode" OFF)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules/")

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_HIP_STANDARD 20)

# needed for atomicAdd_block
set(CMAKE_CUDA_ARCHITECTURES 80)

if (MRA_ENABLE_DEBUG)
    message(STATUS "Enabling debug mode")
    add_compile_definitions(MRA_CHECK_NORMS=1)
endif()

# whoever thought enabling OpenMP by default should be tarred and feathered!
set(use_openmp OFF)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(TTG_ENABLE_CUDA ON)
endif(CMAKE_CUDA_COMPILER)
set(HAVE_CUDA ${CMAKE_CUDA_COMPILER} CACHE BOOL "True if we can compile .cu files")

check_language(HIP)
if(CMAKE_HIP_COMPILER)
  enable_language(HIP)
  find_package(hip REQUIRED)
  set(TTG_ENABLE_HIP ON)
endif(CMAKE_HIP_COMPILER)
set(HAVE_HIP ${CMAKE_HIP_COMPILER} CACHE BOOL "True if we can compile .hip files")

# don't bother with Boost
set(TTG_FETCH_BOOST OFF)
set(TTG_PARSEC_USE_BOOST_SERIALIZATION OFF)
set(TTG_EXAMPLES OFF)

# whoever thought enabling OpenMP by default should be tarred and feathered!
set(use_openmp OFF)
# disable PaRSEC's Fortran support
set(SUPPORT_FORTRAN OFF)
# disable testing in sub projects
set(BUILD_TESTING OFF)

# needed for tests
set(MADNESS_BUILD_MADWORLD_ONLY OFF CACHE BOOL "Whether to build MADNESS runtime only")

option(MRA_ENABLE_ASAN "Whether to enable address sanitizer" OFF)
if (MRA_ENABLE_ASAN)
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
endif (MRA_ENABLE_ASAN)

# Build examples/tests/benchmarks only when MRA-TTG is the top-level project
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  set(_MRA_DEFAULT ON)
else()
  set(_MRA_DEFAULT OFF)
endif()
option(MRA_BUILD_TESTS      "Build MRA tests"      ${_MRA_DEFAULT})
option(MRA_BUILD_EXAMPLES   "Build MRA examples"   ${_MRA_DEFAULT})
option(MRA_BUILD_BENCHMARKS "Build MRA benchmarks" ${_MRA_DEFAULT})

include(FindOrFetchTTG)

##
# BLASpp is not needed for CUDA/HIP builds, since we don't use BLAS on the host in that case.
# TTG provides the mechanism to find blaspp.
##
set(MRA_BLASPP_ENABLED OFF)
if (NOT (HAVE_HIP OR HAVE_CUDA))
  include(FindOrFetchBLASpp)
  if (TARGET blaspp)
    set(MRA_BLASPP_ENABLED ON)
  endif()
endif (NOT (HAVE_HIP OR HAVE_CUDA))

# look for Kokkos
find_package(Kokkos COMPONENTS separable_compilation)

# look for cublasDx
find_package(mathdx COMPONENTS cublasdx)
if (TARGET mathdx::cublasdx)
  message(STATUS "Found cublasDx at ${mathdx_CUBLASDX_DIR}")
else()
  message(STATUS "cublasDx not found")
endif()


add_subdirectory(src)

# ── Install / export ──────────────────────────────────────────────────────────

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT mra-dev
)

install(TARGETS mra-base mra-host
  EXPORT mra-targets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if(HAVE_CUDA)
  install(TARGETS mra-cuda-base mra-cuda
    EXPORT mra-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )
endif()

if(HAVE_HIP)
  install(TARGETS mra-hip
    EXPORT mra-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )
endif()

# Export for the build tree (FetchContent / add_subdirectory consumers)
export(EXPORT mra-targets
  NAMESPACE mra::
  FILE "${CMAKE_CURRENT_BINARY_DIR}/mra-targets.cmake"
)

# Export for the install tree
install(EXPORT mra-targets
  NAMESPACE mra::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mra
)

# Record MADNESS version for the config template
if(MADNESS_VERSION)
  set(MRA_MADNESS_VERSION ${MADNESS_VERSION})
else()
  set(MRA_MADNESS_VERSION "")
endif()

configure_package_config_file(cmake/mra-config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/mra-config.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mra
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/mra-config-version.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/mra-config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/mra-config-version.cmake"
  cmake/modules/MRAGPUWrappers.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mra
)
