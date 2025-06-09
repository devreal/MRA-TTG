#ifndef MRA_ALLOCATOR_H
#define MRA_ALLOCATOR_H

#if !defined(MRA_ENABLE_HOST)
#if __has_include(<TiledArray/external/device.h>)
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
  TiledArray::device::Env::initialize(TiledArray::get_default_world(), 1UL<<32, 1UL<<40);
}

inline void allocator_fini() {
  madness::finalize();
}
#endif // TILEDARRAY_HAS_DEVICE
#endif // MRA_HAVE_TILEDARRAY
#endif // MRA_ENABLE_HOST

#ifndef HAVE_SCRATCH_ALLOCATOR

/* fallback to std::allocator */

template<typename T>
using DeviceAllocator = std::allocator<T>;

inline void allocator_init(int argc, char **argv) { }

inline void allocator_fini() { }

#endif // HAVE_SCRATCH_ALLOCATOR

#endif // MRA_ALLOCATOR_H
