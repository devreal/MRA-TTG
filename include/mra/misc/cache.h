#ifndef MRA_CACHE_H
#define MRA_CACHE_H

#include "mra/misc/types.h"

namespace mra{

  template<typename ValueT, Dimension NDIM>
  class Cache {
    private:
      typedef ConcurrentHashMap<Key<NDIM>> mapT;
      typedef std::pair<Key<NDIM>, ValueT> pairT;
      mapT cache;

    public:
      Cache() : cache() {};

      Cache(const Cache& c) : cache(c.cache) {};

      Cache& operator=(const Cache& c) {
        if(this != c) {
          cache.clear();
          cache = c.cache;
        }
        return *this;
      }

      inline const ValueT* getptr(const Key<NDIM>& key) const {
        auto it = cache.find(key);
        if(it != cache.end()) {
          return &(it->second);
        } else {
          return nullptr;
        }
      }

      inline const ValueT* getptr(Level n, Translation l) const {
        Key<NDIM> key(n, std::array<Translation, NDIM>(l));
        return getptr(key);
      }

      inline const ValueT* getptr(Level n, const Key<NDIM>& disp) const {
        Key<NDIM> key(n, disp.translation);
        return getptr(key);
      }

      inline void set(const Key<NDIM>& key, const ValueT& val) {
        auto && [it, inserted] = cache.insert(pairT(key, val));
      }

      inline void set(Level n, Translation l, const ValueT& val) {
        Key<NDIM> key(n, std::array<Translation, NDIM>(l));
        set(key, val);
      }

      inline void set(Level n, const Key<NDIM>& disp, const ValueT& val) {
        Key<NDIM> key(n, disp.translation());
        set(key, val);
      }
  };
}

#endif // MRA_CACHE_H
