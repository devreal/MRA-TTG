#ifndef MRA_MISC_TWO_LEVEL_H
#define MRA_MISC_TWO_LEVEL_H

#include <ttg.h>

#include "mra/misc/key.h"
#include "mra/misc/types.h"

namespace mra {

  /**
   * Which keys of a tree walked top-down get a task that handles two levels (the key and its
   * children), given the process and device maps of the tasks.
   *
   * A task covers two levels only where the key and all its children are mapped to the same
   * process and device: there, the children's nodes are where the task runs anyway, and
   * handling them in the task moves no data that the one-level walk would not. Above the
   * level where the maps distribute the tree, and along every cut of the maps further down,
   * a key gets a task of its own and its children are the roots of the next tasks.
   *
   *  - root: the key has a task (the root of the tree; a child of a one-level task; a
   *    grandchild of a two-level task). It covers two levels if colocated, else one.
   *  - covered: the key is a child of a two-level task, which handles it.
   *
   * The roles follow from the maps alone, so every process computes the same ones. With
   * ttg::Void maps (one process, one device) every root covers two levels, which puts the
   * tasks at even levels.
   */
  template <Dimension NDIM, typename ProcMap, typename DeviceMap>
  struct TwoLevelPlan {
    using key_type = Key<NDIM>;
    enum class Role { One, Two, Covered };

    ProcMap procmap;
    DeviceMap devicemap;

    /** whether key and all its children map to the same process and device */
    bool colocated(const key_type& key) const {
      constexpr bool has_procmap = !std::is_same_v<ProcMap, ttg::Void>;
      constexpr bool has_devicemap = !std::is_same_v<DeviceMap, ttg::Void>;
      if constexpr (has_procmap || has_devicemap) {
        [[maybe_unused]] int p = 0, d = 0;
        if constexpr (has_procmap) p = procmap(key);
        if constexpr (has_devicemap) d = devicemap(key);
        for (size_type c = 0; c < key_type::num_children(); ++c) {
          const key_type child = key.child_at(c);
          if constexpr (has_procmap) if (procmap(child) != p) return false;
          if constexpr (has_devicemap) if (devicemap(child) != d) return false;
        }
      }
      return true;
    }

    Role root_role(const key_type& key) const { return colocated(key) ? Role::Two : Role::One; }

    /** the role of key, from the roles of its ancestors */
    Role role(const key_type& key) const {
      const Level n = key.level();
      Role r = root_role(key.parent(n));
      for (Level l = 1; l <= n; ++l) {
        r = (r == Role::Two) ? Role::Covered : root_role(key.parent(n - l));
      }
      return r;
    }

    /** the key of the task that handles key: its own or, if covered, its parent's */
    key_type task_of(const key_type& key) const {
      return role(key) == Role::Covered ? key.parent() : key;
    }
  };

  template <Dimension NDIM, typename ProcMap, typename DeviceMap>
  TwoLevelPlan<NDIM, ProcMap, DeviceMap> make_two_level_plan(ProcMap procmap, DeviceMap devicemap) {
    return TwoLevelPlan<NDIM, ProcMap, DeviceMap>{std::move(procmap), std::move(devicemap)};
  }

} // namespace mra

#endif // MRA_MISC_TWO_LEVEL_H
