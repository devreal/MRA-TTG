#ifndef MRA_KEYMAPS_H
#define MRA_KEYMAPS_H

#include <key.h>

#include<cmath>

namespace mra {

#if 0
  /// A pmap that spatially decomposes the domain and by default slightly overdcomposes to attempt to load balance
  template <Dimension NDIM>
  class PartitionPmap {
  public:
    enum class Type { CYCLIC, BLOCK };
  private:
    int nproc;
    Type type;
    Level target_level;
  public:
    // Default is to try to distribute single chunks
    // but the user can specify an oversubscribe factor
    PartitionPmap(int factor = 1, Type type = Type::BLOCK, size_t nproc = ttg::default_execution_context()::size())
    : nproc(nproc)
    , type(type)
    {
      /* the minimum number of levels needed in 3D: (nproc*factor)**(1/8) */
      target_level = std::ceil(std::pow(float(nproc * factor), 1.0/(1<<NDIM)));
    }

    /// Find the owner of a given key
    HashValue operator()(const Key<NDIM>& key) const {
      HashValue hash = 0;
      auto compute_hash = [&](const Key<NDIM>& k) {
        for (Dimension d = 0; d < NDIM; ++d) {
          hash = (hash << NDIM) + k.translation(d);
        }
      };
      if (key.level() <= target_level) {
        return compute_hash(key) % nproc;
      }
      else {
        /* find the parent at the target level and use the parent's process */
        auto parent = key.parent(key.level() - target_level);
        auto hash = compute_hash(parent);
        if (type == Type::BLOCK) {
          return hash / nproc; // cluster as many neighboring nodes as possible
        } else {
          return hash % nproc; // cycle through neighboring nodes
        }
      }
    }
  };
#endif // 0

  /// A pmap that spatially decomposes the domain and by default slightly overdcomposes to attempt to load balance
  template <Dimension NDIM>
  class PartitionKeymap {
  private:
    const int m_num_pe = 0;
    Level m_target_level = 0;
  public:

    // Default is to try to optimize the target_level, but you can specify any value > 0
    PartitionKeymap(int np = ttg::default_execution_context().size(), const Level target_level=0)
    : m_num_pe(np)
    {
      if (target_level > 0) {
        this->m_target_level = target_level;
      } else if (m_num_pe > 0) {
        this->m_target_level = 1;
        int p = m_num_pe-1;
        while (p) {
          p >>= NDIM;
          this->m_target_level++;
        }
      }
    }

    /// Find the owner of a given key
    HashValue operator()(const Key<NDIM>& key) const {
      HashValue hash;
      if (key.level() <= m_target_level) {
        hash = key.hash();
      }
      else {
        hash = key.parent(key.level() - m_target_level).hash();
      }
      return (m_num_pe > 0) ? hash%m_num_pe : 0;
    }

    Level target_level() const {
      return m_target_level;
    }

    int num_pe() const {
      return m_num_pe;
    }
  };


} // namespace mra

#endif // MRA_KEYMAPS_H
