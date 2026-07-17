if (NOT TARGET blaspp)
  find_package(blaspp CONFIG QUIET)
endif()

if (TARGET blaspp)
  message(STATUS "Found blaspp CONFIG at ${blaspp_CONFIG}")
else()
  message(STATUS "blaspp not found via find_package, fetching from GitHub")

  include(FetchContent)
  FetchContent_Declare(
    blaspp
    GIT_REPOSITORY https://github.com/icl-utk-edu/blaspp.git
    GIT_TAG        master
  )

  set(build_tests OFF CACHE BOOL "" FORCE)

  FetchContent_MakeAvailable(blaspp)
endif()

# postcond check
if (NOT TARGET blaspp)
  message(FATAL_ERROR "FindOrFetchBLASpp could not make blaspp target available")
endif()
