# MRA-TTG's device tasks select up to 28 buffers and hold up to 23 data copies per task
# (make_gradient_compressed), more than PaRSEC's default of 20 parameters per task.
# PaRSEC tracks flows in 32-bit masks, so 32 is the most it supports.
set(MRA_MIN_PARSEC_PARAM_COUNT 28)

# PaRSEC not configured yet (it is fetched through TTG below): raise the default,
# keeping a larger value set by the user. A project that configures PaRSEC before
# MRA-TTG (e.g. MADNESS, for its task backend) has to do the same; the check at the
# end of this file catches it if not.
if (NOT TARGET PaRSEC::parsec AND (NOT DEFINED PARSEC_MAX_PARAM_COUNT
                                   OR PARSEC_MAX_PARAM_COUNT LESS MRA_MIN_PARSEC_PARAM_COUNT))
  message(STATUS "MRA-TTG: setting PARSEC_MAX_PARAM_COUNT=32")
  set(PARSEC_MAX_PARAM_COUNT 32 CACHE STRING "Number of parameters for tasks" FORCE)
endif ()

if (NOT TARGET ttg)
  find_package(ttg CONFIG)
endif(NOT TARGET ttg)

if (TARGET ttg)
    message(STATUS "Found ttg CONFIG at ${ttg_CONFIG}")
else (TARGET ttg)

include(FetchContent)
FetchContent_Declare(
      ttg
      GIT_REPOSITORY      https://github.com/devreal/ttg.git
	  GIT_TAG             ttg-kernel-batching
	  #GIT_TAG             924a8bacacabaa0be5279c19616a01e8c211197c
  )

#FetchContent_Declare(
#      ttg
#      GIT_REPOSITORY      https://github.com/TESSEorg/ttg.git
#      GIT_TAG             9f20ec02af70cd414d7454596b06270f734eb134
#  )

  FetchContent_MakeAvailable(ttg)
  FetchContent_GetProperties(ttg
      SOURCE_DIR TTG_SOURCE_DIR
      BINARY_DIR TTG_BINARY_DIR
      )

endif(TARGET ttg)

# postcond check
if (NOT TARGET ttg)
message(FATAL_ERROR "FindOrFetchTTG could not make ttg target available")
endif(NOT TARGET ttg)


include(CheckPaRSECParamCount)
