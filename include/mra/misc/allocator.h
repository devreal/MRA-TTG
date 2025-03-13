#ifndef MRA_ALLOCATOR_H
#define MRA_ALLOCATOR_H

#ifdef MRA_HAVE_TILEDARRAY
#include <TiledArray/external/device.h>
#if defined(TILEDARRAY_HAS_DEVICE)

#define HAVE_SCRATCH_ALLOCATOR 1
template<typename T>
using DeviceAllocator = TiledArray::device_pinned_allocator<T>;

inline void allocator_init(int argc, char **argv) {
  // initialize MADNESS so that TA allocators can be created
#if defined(TTG_PARSEC_IMPORTED)
  madness::ParsecRuntime::initialize_with_existing_context(ttg::default_execution_context().impl().context());
#endif // TTG_PARSEC_IMPORTED
  madness::initialize(argc, argv, /* nthread = */ 1, /* quiet = */ true);
}

inline void allocator_fini() {
  madness::finalize();
}
#endif // TILEDARRAY_HAS_DEVICE
#endif // MRA_HAVE_TILEDARRAY

#ifndef HAVE_SCRATCH_ALLOCATOR

/* fallback to std::allocator */

template<typename T>
using DeviceAllocator = std::allocator<T>;

inline void allocator_init(int argc, char **argv) { }

inline void allocator_fini() { }

#endif // HAVE_SCRATCH_ALLOCATOR

#endif // MRA_ALLOCATOR_H
