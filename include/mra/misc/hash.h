#ifndef MRA_HASH_H
#define MRA_HASH_H

#include "mra/misc/types.h"
#include "mra/misc/platform.h"

namespace mra {

    namespace detail {
        SCOPE uint64_t rot(uint64_t x)
        {
            constexpr const uint64_t n = 27;
            return (x<<n) | (x>>(64-n));
        }

        SCOPE uint32_t rot(uint32_t x)
        {
            constexpr const uint32_t n = 13;
            return (x<<n) | (x>>(32-n));
        }

        // /// Simplest more powerful hashfn I could find that mixes bits (std::hash does not)

        // /// Presently not used
        // class FNVhasher {
        //     // https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
        //     static const HashValue offset_basis = 14695981039346656037ul;
        //     static const HashValue prime = 1099511628211ul;
        //     HashValue value = offset_basis;
        // public:
        //     /// Initializes the hash
        //     FNVhasher() {}

        //     /// Updates the hash with one hash
        //     void update(uint8_t byte) {value = (value^byte)*prime;}

        //     /// Updates the hash with an additional n bytes
        //     void update(size_t n, const void* bytes) {
        //         for (size_t i=0; i<n; i++) update(static_cast<const uint8_t*>(bytes)[i]);
        //     }

        //     /// Returns the value of the hash of the stream
        //     HashValue get() const noexcept {return value;}
        // };
    }

    /// Multiplicative hash with rotation to mix bits (std::hash does not) - empirically good enuf for keys
    SCOPE uint64_t mulhash(uint64_t hash, Translation data) {
        constexpr const uint64_t m = 11400714819323198393ul;
        return detail::rot(hash)^(data*m);
    }

    /// Multiplicative hash with rotation to mix bits (std::hash does not) - empirically good enuf for keys
    SCOPE uint32_t mulhash(uint32_t hash, Translation data) {
        constexpr const uint32_t m = 2654435761u;
        return detail::rot(hash)^(data*m);
    }

} // namespace mra

#endif // MRA_HASH_H
