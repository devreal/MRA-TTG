#ifndef MRA_STACKED_ALLOCATOR_H
#define MRA_STACKED_ALLOCATOR_H

#include <assert.h>
#include <cstddef>

#include "mra/misc/platform.h"
#include "mra/misc/types.h"

namespace mra {

#ifndef MRA_ENABLE_HOST

  namespace detail {
    extern __shared__ char arena[];
  }

#endif // MRA_ENABLE_HOST

class BlockStackAllocator {

  private:

    // ---- Alignment helpers -----
    SCOPE constexpr size_t align_up(size_t offset, size_t align) {
        return (offset + align - 1u) & ~(align - 1u);
    }

#ifndef MRA_ENABLE_HOST
    __device__ __forceinline__
    char* get_arena() {
      return detail::arena;
    }
#else
    char* get_arena() {
      if (!m_arena) {
        m_arena = new char[m_capacity];
      }
      return m_arena;
    }
#endif // MRA_ENABLE_HOST

  public:
    static constexpr size_t DefaultAlign = 16;

    BlockStackAllocator() = default;
    BlockStackAllocator(const BlockStackAllocator&)            = delete;
    BlockStackAllocator& operator=(const BlockStackAllocator&) = delete;
    BlockStackAllocator(BlockStackAllocator&&)                 = default;
    BlockStackAllocator& operator=(BlockStackAllocator&&)      = default;


    // Every thread in the block executes these collective calls identically,
    // so m_offset/m_capacity are plain per-thread state (registers/local) kept
    // in lockstep by redundant computation, not by broadcasting a single
    // thread's result through __shared__. __syncthreads() here is purely a
    // barrier guarding reuse of the shared `arena` bytes, not a data handoff.
    SCOPE BlockStackAllocator(size_type capacity)
    : m_capacity(capacity)
    , m_offset(0)
    {
      SYNCTHREADS();
    }

    ~BlockStackAllocator() {
#ifdef MRA_ENABLE_HOST
      delete[] m_arena;
      m_arena = nullptr;
#endif // MRA_ENABLE_HOST
    }

    template<typename T>
    class BlockScopedAlloc {
      public:
        BlockScopedAlloc(BlockStackAllocator& bsa, T* ptr, size_t cp)
        : bsa_(bsa), ptr_(ptr), cp_(cp)
        { }

        BlockScopedAlloc(const BlockScopedAlloc&)            = delete;
        BlockScopedAlloc& operator=(const BlockScopedAlloc&) = delete;

        BlockScopedAlloc(BlockScopedAlloc&& o)
            : bsa_(o.bsa_), ptr_(o.ptr_), cp_(o.cp_) {
            o.ptr_ = nullptr;
        }

        ~BlockScopedAlloc() {
            if (ptr_) bsa_.restore(cp_);
        }

        SCOPE operator T*() const { return ptr_; }
        SCOPE T* get()      const { return ptr_; }

      private:
        BlockStackAllocator& bsa_;
        T*     ptr_;
        size_t cp_;
    };

    SCOPE void* alloc_raw(size_t bytes, size_t align = DefaultAlign) {
      size_t aligned = align_up(m_offset, align);
      void* ptr = nullptr;
      if (aligned + bytes <= m_capacity) {
        ptr     = static_cast<void*>(get_arena() + aligned);
        m_offset = aligned + bytes;
      }
      SYNCTHREADS();
      return ptr;
    }

    template<typename T>
    SCOPE BlockScopedAlloc<T> alloc(size_t count = 1, bool zero_init = false) {
      size_t cp  = m_offset;
      T*     ptr = static_cast<T*>(alloc_raw(sizeof(T) * count, alignof(T)));

      return BlockScopedAlloc<T>(*this, ptr, cp);
    }

    // ---- Collective checkpoint / restore / reset -----------------------
    SCOPE size_t checkpoint() {
      SYNCTHREADS();
      return m_offset;
    }

    SCOPE void restore(size_t cp) {
      m_offset = cp;
      SYNCTHREADS();
    }

    SCOPE void reset() {
      m_offset = 0;
      SYNCTHREADS();
    }

    // ---- Queries -------------------------------------------------------
    SCOPE size_t used()      const { return m_offset; }
    SCOPE size_t remaining() const { return m_capacity - m_offset; }
    SCOPE size_t capacity()  const { return m_capacity; }

  private:
    size_t m_capacity = 0;
    size_t m_offset = 0;
#ifndef MRA_ENABLE_HOST
    char* m_arena = nullptr;
#endif // MRA_ENABLE_HOST
};

} // namespace mra


#endif // MRA_STACKED_ALLOCATOR_H