cmake_minimum_required(VERSION 3.19)
project(MRA-TTG LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules/")

set(BUILD_TESTING OFF CACHE BOOL "Whether to build tests in this project and its subprojects")

# needed for atomicAdd_block
set(CMAKE_CUDA_ARCHITECTURES 70)

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
set(TTG_EXAMPLES ON) # to pull in TA

option(MRA_ENABLE_ASAN "Whether to enable address sanitizer" OFF)
if (MRA_ENABLE_ASAN)
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
endif (MRA_ENABLE_ASAN)

# disable tests, may want to re-enable later
set(BUILD_TESTING OFF)

include(FindOrFetchTTG)

add_subdirectory(src)
