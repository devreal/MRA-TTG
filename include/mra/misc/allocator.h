#ifndef MRA_ALLOCATOR_H
#define MRA_ALLOCATOR_H

#if !defined(MRA_ENABLE_HOST)
#if __has_include(<TiledArray/external/device.h>)
#include <TiledArray/external/device.h>
#if defined(TILEDARRAY_HAS_DEVICE)

#define HAVE_SCRATCH_ALLOCATOR 1
template<typename T>
using DeviceAllocator = TiledArray::device_pinned_allocator<T>;

#endif // TILEDARRAY_HAS_DEVICE
#endif // MRA_HAVE_TILEDARRAY
#endif // MRA_ENABLE_HOST

#ifndef HAVE_SCRATCH_ALLOCATOR

/* fallback to std::allocator */

template<typename T>
using DeviceAllocator = std::allocator<T>;

#endif // HAVE_SCRATCH_ALLOCATOR

#endif // MRA_ALLOCATOR_H
