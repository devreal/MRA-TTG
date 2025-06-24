#ifndef MRA_CACHE_H
#define MRA_CACHE_H

#include "mra/misc/types.h"

namespace mra{

  template<typename valueT, Dimension NDIM>
  class Cache {
    private:
      using keyT = Key<NDIM>;
      using mapT = std::map<keyT, valueT>;
      using pairT = std::pair<keyT, valueT>;
      mapT cache;
      std::mutex cachemutex;

      auto insert(const pairT& p) {
        cachemutex.lock();
        auto result = cache.insert(p);
        cachemutex.unlock();
        return result;
      }

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

      inline const valueT* getptr(const Key<NDIM>& key) const {
        auto it = cache.find(key);
        if(it != cache.end()) {
          return &(it->second);
        } else {
          return nullptr;
        }
      }

      inline const valueT* getptr(Level n, Translation l) const {
        Key<NDIM> key(n, std::array<Translation, NDIM>({l}));
        return getptr(key);
      }

      inline const valueT* getptr(Level n, const Key<NDIM>& disp) const {
        Key<NDIM> key(n, disp.translation);
        return getptr(key);
      }

      inline void set(const Key<NDIM>& key, const valueT& val) {
        auto && [it, inserted] = cache.insert(pairT(key, val));
      }

      inline void set(Level n, Translation l, const valueT& val) {
        Key<NDIM> key(n, std::array<Translation, NDIM>({l}));
        set(key, val);
      }

      inline void set(Level n, const Key<NDIM>& disp, const valueT& val) {
        Key<NDIM> key(n, disp.translation());
        set(key, val);
      }
  };
}

#endif // MRA_CACHE_H
