#ifndef MRA_KEYMAPS_H
#define MRA_KEYMAPS_H

#include "mra/misc/key.h"

#include<cmath>

namespace mra {

  /**
   * A pmap that spatially decomposes the domain and by default slightly overdcomposes to attempt to load balance
   *
   * This is the legacy keymap. For batching, use BatchedPartitionKeymap and make_procmap()/make_devicemap().
   */
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
