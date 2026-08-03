#ifndef MRA_JIT_EMBEDDED_HEADERS_H
#define MRA_JIT_EMBEDDED_HEADERS_H

#include <string>
#include <unordered_map>

namespace mra::jit {

/**
 * Verbatim content of every project header a JIT-compiled kernel needs,
 * keyed by project-relative include path (e.g. "mra/misc/key.h"). Defined
 * in the CMake-generated translation unit produced by
 * mra_generate_jit_sources() (see cmake/modules/MRAJITSources.cmake) via
 * cmake/scripts/mra_embed_headers.cmake -- never hand-written.
 *
 * Suitable for nvrtcCreateProgram's/hiprtcCreateProgram's headers/
 * includeNames arrays, so a kernel header's own #include "mra/..."
 * directives resolve without any runtime filesystem access.
 */
const std::unordered_map<std::string, std::string>& embedded_headers();

} // namespace mra::jit

#endif // MRA_JIT_EMBEDDED_HEADERS_H
