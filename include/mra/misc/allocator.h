#ifndef MRA_ALLOCATOR_H
#define MRA_ALLOCATOR_H

#include <ttg/env.h>

#if !defined(MRA_ENABLE_HOST)

#define MRA_HAVE_SCRATCH_ALLOCATOR 1
template<typename T>
using DeviceAllocator = ttg::pinned_allocator_t<T>;

#endif // MRA_ENABLE_HOST

#ifndef MRA_HAVE_SCRATCH_ALLOCATOR

/* fallback to std::allocator */

template<typename T>
using DeviceAllocator = std::allocator<T>;

#endif // MRA_HAVE_SCRATCH_ALLOCATOR

#endif // MRA_ALLOCATOR_H
