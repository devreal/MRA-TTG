
# mra_generate_gpu_wrappers
#
# Generates thin .cu/.hip wrapper files that include the original .cc source,
# allowing the same code to be compiled by the CUDA or HIP compiler.
#
# Usage:
#   mra_generate_gpu_wrappers(
#     SOURCES file1.cc [file2.cc ...]
#     [SOURCE_BASE_DIR <dir>]   # base dir for resolving relative source paths
#                               # (defaults to CMAKE_CURRENT_SOURCE_DIR)
#     [CUDA_OUTPUT_VAR <var>]   # receives list of generated .cu paths
#     [HIP_OUTPUT_VAR  <var>]   # receives list of generated .hip paths
#   )
#
# - Paths in SOURCES may be relative (resolved against SOURCE_BASE_DIR, which
#   defaults to CMAKE_CURRENT_SOURCE_DIR) or absolute.
# - SOURCE_BASE_DIR is useful when wrapping sources that live outside the
#   calling CMakeLists.txt directory (e.g. files from a FetchContent subproject).
# - Generated files are placed in CMAKE_CURRENT_BINARY_DIR at the same relative
#   sub-path as the source within SOURCE_BASE_DIR, with .cc replaced by .cu/.hip.
#   If the source is outside SOURCE_BASE_DIR, only the filename is used as stem.
# - Wrappers use an absolute #include path so they work regardless of the build
#   tree location.
# - Only the output variables you name are populated; omit CUDA_OUTPUT_VAR to
#   skip CUDA wrapper generation and vice versa.
#
function(mra_generate_gpu_wrappers)
  cmake_parse_arguments(ARGS "" "CUDA_OUTPUT_VAR;HIP_OUTPUT_VAR;SOURCE_BASE_DIR" "SOURCES" ${ARGN})

  if(NOT ARGS_SOURCES)
    message(FATAL_ERROR "mra_generate_gpu_wrappers: SOURCES must not be empty")
  endif()

  if(NOT DEFINED ARGS_SOURCE_BASE_DIR)
    set(ARGS_SOURCE_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
  endif()

  set(_cuda_srcs "")
  set(_hip_srcs  "")

  foreach(src IN LISTS ARGS_SOURCES)
    if(NOT IS_ABSOLUTE "${src}")
      set(_abs "${ARGS_SOURCE_BASE_DIR}/${src}")
    else()
      set(_abs "${src}")
    endif()

    # Compute the output stem relative to SOURCE_BASE_DIR.
    # Fall back to just the filename when the source lives outside that dir.
    file(RELATIVE_PATH _rel "${ARGS_SOURCE_BASE_DIR}" "${_abs}")
    if(_rel MATCHES "^\\.\\.")
      get_filename_component(_rel "${_abs}" NAME)
    endif()
    string(REGEX REPLACE "\\.cc$" "" _stem "${_rel}")

    if(DEFINED ARGS_CUDA_OUTPUT_VAR)
      set(_out "${CMAKE_CURRENT_BINARY_DIR}/${_stem}.cu")
      get_filename_component(_out_dir "${_out}" DIRECTORY)
      file(MAKE_DIRECTORY "${_out_dir}")
      file(WRITE "${_out}" "#include \"${_abs}\"\n")
      list(APPEND _cuda_srcs "${_out}")
    endif()

    if(DEFINED ARGS_HIP_OUTPUT_VAR)
      set(_out "${CMAKE_CURRENT_BINARY_DIR}/${_stem}.hip")
      get_filename_component(_out_dir "${_out}" DIRECTORY)
      file(MAKE_DIRECTORY "${_out_dir}")
      file(WRITE "${_out}" "#include \"${_abs}\"\n")
      list(APPEND _hip_srcs "${_out}")
    endif()
  endforeach()

  if(DEFINED ARGS_CUDA_OUTPUT_VAR)
    set(${ARGS_CUDA_OUTPUT_VAR} "${_cuda_srcs}" PARENT_SCOPE)
  endif()
  if(DEFINED ARGS_HIP_OUTPUT_VAR)
    set(${ARGS_HIP_OUTPUT_VAR} "${_hip_srcs}" PARENT_SCOPE)
  endif()
endfunction()
