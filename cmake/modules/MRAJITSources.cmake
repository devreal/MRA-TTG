# mra_generate_jit_sources
#
# For each given kernel header, computes -- via a compiler-driven dependency
# scan with MRA_JIT_COMPILE=1 defined (the same macro NVRTC/hiprtc compiles
# will define; see include/mra/misc/platform.h et al. for the guards it
# gates) -- the transitive closure of *project* headers it needs, and embeds
# their verbatim content as C++ raw string literals in one generated
# translation unit (via cmake/scripts/mra_embed_headers.cmake). This lets
# the JIT compiler hand NVRTC/hiprtc the embedded map as the
# headers/includeNames arrays, so a kernel header's own
# #include "mra/misc/key.h"-style directives resolve without any runtime
# filesystem access -- required for hiprtc (no filesystem fallback at all)
# and desirable for NVRTC (portable to installed/packaged binaries; see
# spike/nvrtc/ for the empirical background on why real system headers and
# bare -I aren't viable here).
#
# Only headers that resolve to a path under INCLUDE_DIR are embedded --
# system headers, the CUDA toolkit's libcu++ (cuda/std/*, supplied via -I at
# actual JIT-compile time instead), and anything else guarded out under
# MRA_JIT_COMPILE (ttg.h, madness/mra/key.h, ...) are discarded.
#
# Usage:
#   mra_generate_jit_sources(
#     KERNEL_HEADERS <include/mra/kernels/gaxpy.h> [...]
#     INCLUDE_DIR <dir>              # project include root, e.g. ${PROJECT_SOURCE_DIR}/include
#     OUTPUT_VAR <var>               # receives the path to the generated .cc
#   )
#
# Limitation: the dependency scan runs at CMake *configure* time, so the
# add_custom_command's DEPENDS list (and therefore what triggers a rebuild)
# is fixed for the lifetime of one configure. Editing an existing embedded
# header's *content* still triggers a rebuild (its mtime changes); adding or
# removing a #include that changes the transitive *set* of headers needed
# requires re-running cmake before the new/removed header is picked up.
function(mra_generate_jit_sources)
  cmake_parse_arguments(ARGS "" "INCLUDE_DIR;OUTPUT_VAR" "KERNEL_HEADERS" ${ARGN})

  if(NOT ARGS_KERNEL_HEADERS)
    message(FATAL_ERROR "mra_generate_jit_sources: KERNEL_HEADERS must not be empty")
  endif()
  if(NOT ARGS_INCLUDE_DIR)
    message(FATAL_ERROR "mra_generate_jit_sources: INCLUDE_DIR is required")
  endif()
  if(NOT ARGS_OUTPUT_VAR)
    message(FATAL_ERROR "mra_generate_jit_sources: OUTPUT_VAR is required")
  endif()

  get_filename_component(_include_dir_abs "${ARGS_INCLUDE_DIR}" ABSOLUTE)

  # Union, across all requested kernel headers, of every project header
  # (including the kernel headers themselves) needed under MRA_JIT_COMPILE=1,
  # deduplicated by relative include path.
  set(_embed_rel_paths "")
  set(_embed_abs_paths "")

  foreach(_kernel_header IN LISTS ARGS_KERNEL_HEADERS)
    get_filename_component(_kernel_header_abs "${_kernel_header}" ABSOLUTE)

    execute_process(
      COMMAND ${CMAKE_CXX_COMPILER} -std=c++20 -MM -MG
              -DMRA_JIT_COMPILE=1
              -I${_include_dir_abs}
              "${_kernel_header_abs}"
      OUTPUT_VARIABLE _dep_output
      RESULT_VARIABLE _dep_result
      ERROR_VARIABLE _dep_error
    )
    if(NOT _dep_result EQUAL 0)
      message(FATAL_ERROR "mra_generate_jit_sources: dependency scan failed for ${_kernel_header_abs}:\n${_dep_error}")
    endif()

    # -MM output is a Makefile rule: "<obj>: <hdr1> \\\n  <hdr2> \\\n  ...".
    # Strip the "target:" prefix and line-continuation backslash+newlines,
    # then split on whitespace.
    string(REGEX REPLACE "^[^:]*:" "" _dep_output "${_dep_output}")
    string(REPLACE "\\\n" " " _dep_output "${_dep_output}")
    string(REPLACE "\n" " " _dep_output "${_dep_output}")
    separate_arguments(_dep_list UNIX_COMMAND "${_dep_output}")

    foreach(_dep IN LISTS _dep_list)
      get_filename_component(_dep_abs "${_dep}" ABSOLUTE)
      if(NOT EXISTS "${_dep_abs}")
        continue() # e.g. cuda/std/* under -MG: unresolved, not ours to embed
      endif()
      string(FIND "${_dep_abs}" "${_include_dir_abs}/" _pos)
      if(NOT _pos EQUAL 0)
        continue() # not a project header (system/fetched-dependency header)
      endif()
      file(RELATIVE_PATH _dep_rel "${_include_dir_abs}" "${_dep_abs}")
      list(FIND _embed_rel_paths "${_dep_rel}" _existing_idx)
      if(_existing_idx EQUAL -1)
        list(APPEND _embed_rel_paths "${_dep_rel}")
        list(APPEND _embed_abs_paths "${_dep_abs}")
      endif()
    endforeach()
  endforeach()

  list(LENGTH _embed_rel_paths _n)
  if(_n EQUAL 0)
    message(FATAL_ERROR "mra_generate_jit_sources: no embeddable headers found for ${ARGS_KERNEL_HEADERS} -- dependency scan likely misconfigured")
  endif()

  # Manifest passed to the generator script: one "rel|abs" pair per line
  # (simpler and more robust than a long command-line argument list once
  # dozens of headers are involved).
  set(_lines "")
  math(EXPR _last "${_n} - 1")
  foreach(_i RANGE 0 ${_last})
    list(GET _embed_rel_paths ${_i} _rel)
    list(GET _embed_abs_paths ${_i} _abs)
    list(APPEND _lines "${_rel}|${_abs}")
  endforeach()
  set(_manifest "${CMAKE_CURRENT_BINARY_DIR}/mra_jit_embed_manifest.txt")
  string(REPLACE ";" "\n" _manifest_content "${_lines}")
  file(WRITE "${_manifest}" "${_manifest_content}\n")

  set(_generated "${CMAKE_CURRENT_BINARY_DIR}/mra_jit_embedded_headers.cc")
  set(_script "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../scripts/mra_embed_headers.cmake")

  add_custom_command(
    OUTPUT "${_generated}"
    COMMAND ${CMAKE_COMMAND} "-DMANIFEST=${_manifest}" "-DOUTPUT=${_generated}" -P "${_script}"
    DEPENDS ${_embed_abs_paths} "${_manifest}" "${_script}"
    COMMENT "Embedding ${_n} JIT-compile-safe headers into ${_generated}"
    VERBATIM
  )

  set(${ARGS_OUTPUT_VAR} "${_generated}" PARENT_SCOPE)
endfunction()
