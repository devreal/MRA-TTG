# Fails unless the PaRSEC in use (PaRSEC::parsec) supports MRA_MIN_PARSEC_PARAM_COUNT
# parameters per task, see FindOrFetchTTG.cmake.
if (NOT DEFINED MRA_MIN_PARSEC_PARAM_COUNT)
  set(MRA_MIN_PARSEC_PARAM_COUNT 28)
endif ()

# check the parameter count of the PaRSEC actually used: configured in this tree (its cache
# variable), or installed (its parsec_options.h)
set(_mra_parsec_param_count "")
if (DEFINED PARSEC_MAX_PARAM_COUNT AND TARGET parsec)
  set(_mra_parsec_param_count ${PARSEC_MAX_PARAM_COUNT})
elseif (TARGET PaRSEC::parsec)
  get_target_property(_mra_parsec_include_dirs PaRSEC::parsec INTERFACE_INCLUDE_DIRECTORIES)
  foreach (_dir IN LISTS _mra_parsec_include_dirs)
    string(REGEX REPLACE "^\\$<BUILD_INTERFACE:(.*)>$" "\\1" _dir "${_dir}")
    if (NOT _dir MATCHES "\\$<" AND EXISTS "${_dir}/parsec/parsec_options.h")
      file(STRINGS "${_dir}/parsec/parsec_options.h" _line REGEX "^#define[ \t]+MAX_PARAM_COUNT[ \t]+[0-9]+")
      string(REGEX REPLACE "^#define[ \t]+MAX_PARAM_COUNT[ \t]+([0-9]+).*" "\\1" _mra_parsec_param_count "${_line}")
      break()
    endif ()
  endforeach ()
endif ()
if (_mra_parsec_param_count STREQUAL "")
  message(WARNING "MRA-TTG: could not determine PaRSEC's MAX_PARAM_COUNT; MRA-TTG needs at least "
                  "${MRA_MIN_PARSEC_PARAM_COUNT} (configure PaRSEC with -DPARSEC_MAX_PARAM_COUNT=32)")
elseif (_mra_parsec_param_count LESS MRA_MIN_PARSEC_PARAM_COUNT)
  message(FATAL_ERROR "MRA-TTG: PaRSEC has MAX_PARAM_COUNT=${_mra_parsec_param_count}, MRA-TTG needs at least "
                      "${MRA_MIN_PARSEC_PARAM_COUNT}: configure PaRSEC with -DPARSEC_MAX_PARAM_COUNT=32 "
                      "(a project that configures PaRSEC before MRA-TTG has to set it)")
else ()
  message(STATUS "MRA-TTG: PaRSEC supports ${_mra_parsec_param_count} parameters per task "
                 "(at least ${MRA_MIN_PARSEC_PARAM_COUNT} needed)")
endif ()
