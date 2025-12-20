#ifndef MRA_KEY_H
#define MRA_KEY_H

#include "mra/misc/types.h"
#include "mra/misc/hash.h"
#include "mra/misc/misc.h"
#include "mra/misc/platform.h"

namespace mra {

    /// Extracts the n'th bit as 0 or 1
    SCOPE Translation get_bit(int bits, Dimension n) {return ((bits>>n) & 0x1ul);}

    /// Extracts the low bit as 0 or 1
    SCOPE Translation low_bit(Translation l) {return l & Translation(1);}

    template <Dimension NDIM>
    class Key {
    private:
        Batch b;  // = 0; used for batching functions in the same union node
        Level n;  // = 0; cannot default initialize if want to be POD
        std::array<Translation,NDIM> l; // = {}; ditto

        /// Refreshes the hash value.  Note that the default std::hash does not mix enough
        SCOPE HashValue rehash() const {
            HashValue hashvalue = n ^ (static_cast<HashValue>(b)<<48);
            //for (Dimension d=0; d<NDIM; d++) mulhash(hashvalue,l[d]);
            for (Dimension d=0; d<NDIM; d++) hashvalue = (hashvalue<<7) | l[d];
            return hashvalue;
        }

    public:
        static constexpr int num_children() { return (1ul<<NDIM); }

        /// Default constructor is deliberately default so that is POD
        constexpr SCOPE Key() = default;

        /// Copy constructor default is OK
        constexpr SCOPE Key(const Key<NDIM>& key) = default;

        /// Move constructor default is OK
        constexpr SCOPE Key(Key<NDIM>&& key) = default;

        /// Construct from batch, level and translation
        constexpr SCOPE Key(Batch b, Level n, const std::array<Translation,NDIM>& l)
        : b(b), n(n), l(l)
        { }

        /// Construct from batch and level with translation=0
        constexpr SCOPE Key(Batch b, Level n)
        : b(b), n(n), l({0})
        { }

        /// Assignment default is OK
        SCOPE Key& operator=(const Key<NDIM>& other) = default;

        /// Move assignment default is OK
        SCOPE Key& operator=(Key<NDIM>&& key) = default;

        /* The HIP compiler seems to stumble over the spaceship operator (rocm/6.4.1)
         * and complains about missing references to memcmp when linking.
         * Maybe one day we can actually have nice things... */
        //SCOPE auto operator<=>(const Key<NDIM>&) const = default;

        /// Less-than comparison
        SCOPE bool operator<(const Key<NDIM>& other) const {
          auto compare = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            return std::tie(b, n, l[Is]...) < std::tie(other.b, other.n, other.l[Is]...);
          };
          return compare(std::make_index_sequence<NDIM>{});
        }

        /// Equality comparison
        SCOPE bool operator==(const Key<NDIM>& other) const {
          auto compare = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            return b == other.b && n == other.n && ((l[Is] == other.l[Is]) && ...);
          };
          return compare(std::make_index_sequence<NDIM>{});
        }

        /// Inequality comparison
        SCOPE bool operator!=(const Key<NDIM>& other) const {
          return !(*this == other);
        }


        /// Hash to unsigned value
        SCOPE HashValue hash() const {return rehash();}

        /// Level (n = 0, 1, 2, ...)
        SCOPE Level level() const {return n;}

        /// Translation (each element 0, 1, ..., 2**level-1)
        SCOPE const std::array<Translation,NDIM>& translation() const {return l;}

        /// Batch number (used for batching functions in the same union node)
        SCOPE Batch batch() const {return b;}

        /// Parent key

        /// Default is the immediate parent (generation=1).  To get
        /// the grandparent use generation=2, and similarly for
        /// great-grandparents.
        ///
        /// !! If there is no such parent it quietly returns the
        /// closest match (which may be self if this is the top of the
        /// tree).
        SCOPE Key<NDIM> parent(Level generation = 1) const {
            generation = std::min(generation,n);
            std::array<Translation,NDIM> pl;
            for (Dimension i=0; i<NDIM; i++) pl[i] = (l[i] >> generation);
            return Key<NDIM>(b, n-generation, pl);
        }

        /// First child in lexical ordering of KeyChildren iteration
        SCOPE Key<NDIM> first_child() const {
            assert(n<MAX_LEVEL);
            std::array<Translation,NDIM> l = this->l;
            for (auto& x : l) x = 2*x;
            return Key<NDIM>(b, n+1, l);
        }

        /// Last child in lexical ordering of KeyChildren iteration
        SCOPE Key<NDIM> last_child() const {
            assert(n<MAX_LEVEL);
            std::array<Translation,NDIM> l = this->l;
            for (auto& x : l) x = 2*x + 1;
            return Key<NDIM>(b, n+1, l);
        }

        /// Used by iterator to increment child translation
        SCOPE void next_child(int& bits) {
            int oldbits = bits++;
            for (Dimension d = 0; d < NDIM; ++d) {
                l[d] += get_bit(bits, d) - get_bit(oldbits,d);
            }
        }

        /// Map translation to child index in parent which is formed from binary code (bits)
        SCOPE int childindex() const {
            int b = low_bit(l[NDIM-1]);
            for (Dimension d=NDIM-1; d>0; d--) b = (b<<1) | low_bit(l[d-1]);
            return b;
        }

        /// Return the Key of the child at position idx \in [0, 1<<NDIM)
        SCOPE Key<NDIM> child_at(int idx) const {
            assert(n<MAX_LEVEL);
            assert(idx<num_children());
            std::array<Translation,NDIM> l = this->l;
            for (Dimension d = 0; d < NDIM; ++d) l[d] = 2*l[d] + ((idx & (1<<d)) ? 1 : 0);
            return Key<NDIM>(b, n+1, l);
        }

        SCOPE Key<NDIM> child_left(Dimension axis) const {
            assert(n<MAX_LEVEL);
            assert(axis < NDIM);
            std::array<Translation,NDIM> l = this->l;
            for (auto& x : l) x = 2*x;
            return Key<NDIM>(b, n+1, l);
        }

        SCOPE Key<NDIM> child_right(Dimension axis) const {
            assert(n<MAX_LEVEL);
            assert(axis < NDIM);
            std::array<Translation,NDIM> l = this->l;
            for (auto& x : l) x = 2*x;
            l[axis]++;
            return Key<NDIM>(b, n+1, l);
        }

        SCOPE bool is_left_child(Dimension axis) const {
            assert(n<MAX_LEVEL);
            return (this->l[axis] % 2) == 0;
        }

        SCOPE bool is_right_child(Dimension axis) const {
            assert(n<MAX_LEVEL);
            return (this->l[axis] % 2) == 1;
        }

        SCOPE Key<NDIM> neighbor(const Key<NDIM>& disp) const {
            std::array<Translation,NDIM> l = this->l + disp.l;
            return Key<NDIM>(b, n, l);
        }

        SCOPE Key<NDIM> neighbor(Dimension axis, int disp) const {
            if ((is_right_boundary(axis) && disp > 0) ||
                (is_left_boundary(axis) && disp < 0)) return invalid();
            std::array<Translation,NDIM> l = this->l;
            l[axis] += disp;
            return Key<NDIM>(b, n, l);
        }


        SCOPE bool is_left_boundary(Dimension axis) const {
            return (l[axis] == 0);
        }

        SCOPE bool is_right_boundary(Dimension axis) const {
            return (l[axis] == (1ul<<n)-1);
        }

        SCOPE bool is_boundary(Dimension axis) const {
            return is_left_boundary(axis) || is_right_boundary(axis);
        }

        SCOPE constexpr Key<NDIM> invalid() const {
            return Key<NDIM>(b, -1);
        }

        SCOPE constexpr bool is_invalid() const {
            return n == -1;
        }

        SCOPE constexpr bool is_valid() const {
            return n != -1;
        }

        SCOPE constexpr Key<NDIM> step(Dimension axis, int width) const {
            std::array<Translation, NDIM> l = translation();
            l[axis] += width;
            return Key<NDIM>(batch(), level(), l);
        }
    };

    /// Range object used to iterate over children of a key
    template <Dimension NDIM>
    class KeyChildren {
        struct iterator {
            Key<NDIM> value;
            int bits;
            SCOPE iterator (const Key<NDIM>& value, int bits) : value(value), bits(bits) {}
            SCOPE operator const Key<NDIM>&() const {return value;}
            SCOPE const Key<NDIM>& operator*() const {return value;}
            SCOPE iterator& operator++() {
                value.next_child(bits);
                return *this;
            }
            SCOPE bool operator!=(const iterator& other) {return bits != other.bits;}

            /// Provides the index of the child (0, 1, ..., Key<NDIM>::num_children-1) while iterating
            SCOPE int index() const {return bits;}
        };
        iterator start, finish;

    public:
        SCOPE KeyChildren(const Key<NDIM>& key) : start(key.first_child(),0ul), finish(key.last_child(),(1ul<<NDIM)) {}
        SCOPE iterator begin() const {return start;}
        SCOPE iterator end() const {return finish;}
    };

    /// Returns range object for iterating over children of a key
    template <Dimension NDIM>
    SCOPE KeyChildren<NDIM> children(const Key<NDIM>& key) {return KeyChildren<NDIM>(key);}

    template <Dimension NDIM>
    std::ostream& operator<<(std::ostream& s, const Key<NDIM>& key) {
        s << "Key<" << int(NDIM) << ">[" << key.batch() << "](" << int(key.level()) << "," << key.translation() << ")";
        return s;
    }
}

namespace std {
    /// Ensures key satifies std::hash protocol
    template <mra::Dimension NDIM>
    struct hash<mra::Key<NDIM>> {
        SCOPE int operator()(const mra::Key<NDIM>& s) const noexcept { return s.hash(); }
    };
} // namespace std

#endif // MRA_KEY_H
