if (NOT TARGET mathdx::cublasdx)
  find_package(mathdx CONFIG)
endif(NOT TARGET mathdx::cublasdx)

if (TARGET mathdx::cublasdx)
  message(STATUS "Found mathdx::cublasdx CONFIG at ${mathdx_CONFIG}")
else (TARGET mathdx::cublasdx)

  set(MRA_CUBLASDX_VERSION "25.06" CACHE STRING "Version of cublasDx to use")

  # fetch cublasDx
  include(FetchContent)
  FetchContent_Declare(
  mathdx
  URL https://developer.download.nvidia.com/compute/cublasdx/redist/cublasdx/nvidia-mathdx-${MRA_CUBLASDX_VERSION}.0.tar.gz
  )
  FetchContent_MakeAvailable(mathdx)
  FetchContent_GetProperties(mathdx
      SOURCE_DIR MATHDX_SOURCE_DIR
      BINARY_DIR MATHDX_BINARY_DIR
  )

  set(MATHDX_DIR ${MATHDX_SOURCE_DIR}/nvidia/mathdx/25.06/)

  # look for cublasDx
  find_package(mathdx REQUIRED COMPONENTS cublasdx HINTS ${MATHDX_DIR})
  if (TARGET mathdx::cublasdx)
  message(STATUS "Found cublasDx at ${MATHDX_DIR}")
  else()
  message(FATAL_ERROR "cublasDx not found")
  endif()

endif(TARGET mathdx::cublasdx)

# postcond check
if (NOT TARGET mathdx::cublasdx)
  message(FATAL_ERROR "FindOrFetchCUBLASDX could not make mathdx::cublasdx target available")
endif(NOT TARGET mathdx::cublasdx)
