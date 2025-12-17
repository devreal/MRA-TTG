#ifndef MRA_MISC_BATCHING_H
#define MRA_MISC_BATCHING_H

#include <algorithm>

#include "mra/misc/key.h"

namespace mra {

  // Maximum number of functions per batch
  // If we have less functions we batch everything into one batch.
  // If we have more, we split them into batches.
  static constexpr size_type MAX_SINGLE_BATCH_SIZE = 1024;

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
    PartitionPmap(int factor = 1, unsigned nblocks = 1, Type type = Type::BLOCK, int nproc = ttg::default_execution_context()::size())
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


  class ProcessInfo {
    ttg::World m_world;

  public:
    ProcessInfo(ttg::World& world)
    : m_world(world)
    { }

    int num_pes() const { return m_world.size(); }
    int my_rank() const { return m_world.rank(); }
  };

  class DeviceInfo {
  public:
    int num_pes() const { return ttg::device::num_devices(); }
    int my_rank() const { return ttg::device::current_device(); }
  };


  enum class BatchDistribution {
    BLOCK, //< Block distribution: each batch is assigned to a set of PE
    FULL,  //< Full distribution: each PE is involved in all batches
    SINGLE_PE, //< Single distribution: each batch is assigned to a single PE
    INVALID,
  };


  template<typename PEInfo>
  class DefaultBatchManager {

  public:
    using peinfo_type = PEInfo;

  private:
    size_type m_num_funcs;
    Batch m_num_batches;
    BatchDistribution m_distribution;
    PEInfo m_peinfo;

    size_type compute_batches_per_pe() const {
      switch (m_distribution) {
        case BatchDistribution::BLOCK:
        {
          if (m_num_batches <= m_peinfo.num_pes()) {
            return 1; // one batch per PE
          } else {
            return (m_num_batches + m_peinfo.num_pes() - 1) / m_peinfo.num_pes(); // round up
          }
        }
        case BatchDistribution::FULL:
          return m_num_batches; // all processes get all batches
        case BatchDistribution::SINGLE_PE:
          return (m_num_batches + m_peinfo.num_pes() - 1) / m_peinfo.num_pes(); // each batch is assigned to a single PE
        default:
          throw std::runtime_error("Invalid distribution type");
      }
    }

    size_type compute_pes_per_batch() const {
      int batches_per_pe = compute_batches_per_pe();
      switch (m_distribution) {
        case BatchDistribution::BLOCK:
        {
          int pes_per_batches;
          if (m_num_batches >= m_peinfo.num_pes()) {
            pes_per_batches = 1; // one batch per PE
          } else {
            pes_per_batches = m_peinfo.num_pes() / m_num_batches; // round up
          }
          return pes_per_batches;
        }
        case BatchDistribution::FULL:
          return m_peinfo.num_pes(); // all processes get all batches
        case BatchDistribution::SINGLE_PE:
          return 1; // each batch is assigned to a single PE
        default:
          throw std::runtime_error("Invalid distribution type");
      }
    }

    std::pair<Batch, Batch> local_batch_range() const {
      /* compute the batches I am responsible for */
      std::pair<Batch, Batch> local_range;
      size_type batches_per_pe = compute_batches_per_pe();
      size_type pes_per_batch = compute_pes_per_batch();
      switch (m_distribution) {
        case BatchDistribution::BLOCK:
          if (pes_per_batch == 1) {
            assert(batches_per_pe >= 1);
            Batch start_batch = m_peinfo.my_rank() * batches_per_pe;
            Batch end_batch = std::min(static_cast<Batch>(start_batch + batches_per_pe), m_num_batches);
            local_range = {start_batch, end_batch};
          } else {
            /* more than one PE per batch, one batch per PE */
            assert(batches_per_pe == 1);
            Batch start_batch = m_peinfo.my_rank() / pes_per_batch;
            Batch end_batch = std::min(static_cast<Batch>(start_batch + 1), m_num_batches);
            local_range = {start_batch, end_batch};
          }
          break;
        case BatchDistribution::FULL:
          local_range = {0, m_num_batches};
          break;
        case BatchDistribution::SINGLE_PE:
        {
          assert(batches_per_pe >= 1);
          Batch start_batch = m_peinfo.my_rank() * batches_per_pe;
          Batch end_batch = std::min(static_cast<Batch>(start_batch + batches_per_pe), m_num_batches);
          local_range = {start_batch, end_batch};
          break;
        }
        default:
          throw std::runtime_error("Invalid distribution type");
      }
      return local_range;
    }

  public:
    DefaultBatchManager(BatchDistribution dist,
                        size_type num_funcs,
                        Batch num_batches,
                        PEInfo peinfo)
    : m_num_funcs(num_funcs)
    , m_num_batches(num_batches)
    , m_distribution(dist)
    , m_peinfo(peinfo)
    { }

    /**
     * Returns the number of functions.
     */
    size_type num_functions() const {
      return m_num_funcs;
    }

    /**
     * Returns the number of batches.
     */
    Batch num_batches() const {
      return m_num_batches;
    }

    /**
     * Returns the used distribution.
     */
    BatchDistribution distribution() const {
      return m_distribution;
    }

    /**
     * Returns the number of PEs used for computation of distributions.
     */
    int num_pes() const {
      return m_peinfo.num_pes();
    }

    /**
     * Returns the number of batches the current process is involved with.
     */
    Batch num_local_batches() const {
      auto local_batches = local_batch_range();
      return local_batches.second - local_batches.first;
    }

    template<typename Fn>
    void foreach_local_batch(Fn&& fn) {
      auto local_batches = local_batch_range();
      for (int i = local_batches.first; i < local_batches.second; ++i) {
        if (!fn(i)) break;
      }
    }

#if 0
    /* SIGH! Clang17 still does not support std::generator :/ */
    std::generator<int> local_batches() const {
      switch (m_distribution) {
        case BatchDistribution::BLOCK: {
          int batches_per_pe = (m_num_batches + m_num_pe - 1) / m_num_pe;
          int start_batch = ttg::default_execution_context().rank() * batches_per_pe;
          int end_batch = std::min(first_batch + batches_per_pe, m_num_batches);
          for (int i = start_batch; i < end_batch; ++i) {
            co_yield i;
          }
          break;
        }
        case BatchDistribution::FULL: {
          for (int i = 0; i < m_num_batches; ++i) {
            co_yield i;
          }
          break;
        }
        default:
          throw std::runtime_error("Invalid distribution type");
      }
    }
#endif // 0

    /**
     * Returns the size of each batch.
     */
    size_type batch_size(size_type batch) const {
      size_type batch_size = (m_num_funcs + m_num_batches - 1) / m_num_batches; // round up
      if (batch < m_num_batches - 1) {
        return batch_size; // all but the last batch have the same size
      } else {
        return m_num_funcs - (batch_size * (m_num_batches - 1)); // last batch may be smaller
      }
    }

    /**
     * Returns the range of processes [start, end) that a batch is assigned to.
     * For example, with the FULL distribution, the range is [0, world_size).
     * With the BLOCK distribution, if there are 8 batches and 4 processes,
     * the first batch is assigned to processes 0 and 1, the second batch to processes 2 and 3, and so on.
     * If there are more batches than processes, then each process gets multiple batches
     * (e.g., process 0 gets batches 0, 1; process 1 gets 2, 3; ...).
     */
    std::pair<int, int> batch_pes(Batch batch) const {
      assert(batch < m_num_batches);
      switch (m_distribution) {
        case BatchDistribution::BLOCK: {
          if (m_num_batches <= m_peinfo.num_pes()) {
            /* more processes than batches: one batch per process */
            int start_pe = batch;
            int end_pe = start_pe + 1;
            return {start_pe, end_pe};
          }
          int pes_per_batch = compute_pes_per_batch();
          int batches_per_pe = compute_batches_per_pe();
          int start_pe;
          if (pes_per_batch > 1) {
            start_pe = batch * pes_per_batch;
          } else {
            start_pe = batch / batches_per_pe;
          }
          int end_pe = std::min(start_pe + pes_per_batch, m_peinfo.num_pes());
          return {start_pe, end_pe};
        }
        case BatchDistribution::FULL:
          return {0, m_peinfo.num_pes()}; // all processes are involved in all batches
        case BatchDistribution::SINGLE_PE: {
          int start_pe = batch % m_peinfo.num_pes();
          return {start_pe, start_pe + 1};
        }
        default:
          throw std::runtime_error("Invalid distribution type");
      }
    }

    /**
     * Maps the index of a function to its batch.
     */
    size_type batch_index(size_type func_idx) const {
      size_type batch_size = (m_num_funcs + m_num_batches - 1) / m_num_batches; // round up
      return (func_idx) / batch_size;
    }

    /**
     * Returns the default batch distribution.
     * If we compile for execution on the host, we use FULL distribution,
     * i.e., all processes get all batches.
     * If we compile for execution on the device and have more than MAX_SINGLE_BATCH_SIZE,
     * we use BLOCK distribution so that we distributed batches evenly across processes.
     */
    static BatchDistribution suggest_distribution(size_type num_functions) {
#if MRA_ENABLE_HOST
      return BatchDistribution::FULL; // default distribution is FULL on host
#else
      if (MAX_SINGLE_BATCH_SIZE > num_functions) {
        return BatchDistribution::FULL; // if we have less functions than MAX_SINGLE_BATCH_SIZE, use FULL distribution
      } else {
        return BatchDistribution::BLOCK; // otherwise, use BLOCK distribution
      }
#endif // MRA_ENABLE_HOST
    }

    /**
     * Suggests the number of batches based on the number of functions.
     * Up to MAX_SINGLE_BATCH_SIZE, we use one batch. Beyond that, we split into evenly
     * sized batches. For example, if we have 1500 functions we suggest 2 batches of 750 functions each.
     */
    static Batch suggest_num_batches(size_type num_functions) {
#if MRA_ENABLE_HOST
      return num_functions; // on host, we use one batch per function
#else  // MRA_ENABLE_HOST
      if (num_functions <= MAX_SINGLE_BATCH_SIZE) {
        return 1; // if we have less functions than MAX_SINGLE_BATCH_SIZE, use 1 batch
      } else {
        return (num_functions + MAX_SINGLE_BATCH_SIZE - 1) / MAX_SINGLE_BATCH_SIZE; // round up to the nearest batch size
      }
#endif // MRA_ENABLE_HOST
    }
  };

  /**
   * Creates a batch manager with the given number of functions and batches.
   * If number of batches is -1, we use the default suggested by DefaultBatchManager::suggest_num_batches.
   */
  template<typename PEInfo>
  auto make_batch_manager(
    size_type num_functions,
    size_type num_batches,
    PEInfo&& peinfo,
    BatchDistribution distribution = BatchDistribution::INVALID)
  {
    using BatchManager = DefaultBatchManager<PEInfo>;
    if (num_batches == 0) {
      num_batches = BatchManager::suggest_num_batches(num_functions);
    }
    if (distribution == BatchDistribution::INVALID) {
      distribution = BatchManager::suggest_distribution(num_functions);
    }
    return std::make_shared<BatchManager>(distribution, num_functions,
                                          num_batches, std::forward<PEInfo>(peinfo));
  }


  /**
   * A simple batch manager for a single batch distributed across all processes.
   */


  class SimpleBatchManager {

  private:
    size_type m_num_funcs;

  public:
    SimpleBatchManager(size_type num_funcs)
    : m_num_funcs(num_funcs)
    { }

    /**
     * Returns the number of functions.
     */
    size_type num_functions() const {
      return m_num_funcs;
    }

    /**
     * Returns the number of batches.
     */
    Batch num_batches() const {
      return 1;
    }

    /**
     * Returns the used distribution.
     */
    BatchDistribution distribution() const {
      return BatchDistribution::FULL;
    }

    /**
     * Returns the number of PEs used for computation of distributions.
     */
    int num_pes() const {
      return ttg::default_execution_context().size();
    }

    /**
     * Returns the number of batches the current process is involved with.
     */
    Batch num_local_batches() const {
      return 1;
    }

    template<typename Fn>
    void foreach_local_batch(Fn&& fn) {
      fn(1);
    }

    /**
     * Returns the size of each batch.
     */
    size_type batch_size(size_type batch) const {
      return m_num_funcs;
    }

    /**
     * Returns the range of processes [start, end) that a batch is assigned to.
     * For example, with the FULL distribution, the range is [0, world_size).
     * With the BLOCK distribution, if there are 8 batches and 4 processes,
     * the first batch is assigned to processes 0 and 1, the second batch to processes 2 and 3, and so on.
     * If there are more batches than processes, then each process gets multiple batches
     * (e.g., process 0 gets batches 0, 1; process 1 gets 2, 3; ...).
     */
    std::pair<int, int> batch_pes(Batch batch) const {
      return std::pair<int, int>(0, ttg::default_execution_context().size());
    }

    /**
     * Maps the index of a function to its batch.
     */
    size_type batch_index(size_type func_idx) const {
      return 0;
    }

    /**
     * Returns the default batch distribution.
     * If we compile for execution on the host, we use FULL distribution,
     * i.e., all processes get all batches.
     * If we compile for execution on the device and have more than MAX_SINGLE_BATCH_SIZE,
     * we use BLOCK distribution so that we distributed batches evenly across processes.
     */
    static BatchDistribution suggest_distribution(size_type num_functions) {
      return BatchDistribution::FULL;
    }

    /**
     * Suggests the number of batches based on the number of functions.
     * Up to MAX_SINGLE_BATCH_SIZE, we use one batch. Beyond that, we split into evenly
     * sized batches. For example, if we have 1500 functions we suggest 2 batches of 750 functions each.
     */
    static Batch suggest_num_batches(size_type num_functions) {
      return 1;
    }
  };



  template<Dimension NDIM, typename BatchManager>
  class BatchedPartitionKeymap {

  public:
    static constexpr Dimension ndim() { return NDIM; }

  private:
    std::shared_ptr<BatchManager> m_batch_manager;
    Level m_target_level; // Target level for partitioning

  public:
    BatchedPartitionKeymap(std::shared_ptr<BatchManager> batch_manager,
                           const Level target_level=0)
    : m_batch_manager(std::move(batch_manager))
    , m_target_level(target_level)
    {
      int num_pe = m_batch_manager->num_pes();
      if (m_target_level <= 0) {
        m_target_level = 1;
        int p = num_pe-1;
        while (p) {
          p >>= NDIM;
          m_target_level++;
        }
      }
    }

    /**
     * Maps a key to a process.
     * The key is used to determine the batch, and the batch is mapped to processes.
     */
    int operator()(const Key<NDIM>& key) const {
      auto batch_procs = m_batch_manager->batch_pes(key.batch());

      // Compute the hash value at the target level
      HashValue hash;
      if (key.level() <= m_target_level) {
        hash = key.hash();
      }
      else {
        hash = key.parent(key.level() - m_target_level).hash();
      }

      // Map the hash value to a process within the batch range
      int num_pe = batch_procs.second - batch_procs.first;
      return (num_pe > 0) ? (batch_procs.first + hash%num_pe) : 0;
    }

    /// Returns the batch manager
    std::shared_ptr<BatchManager> batch_manager() const {
      return m_batch_manager;
    }

    /// Returns the target level for partitioning
    int initial_level() const {
      return m_target_level;
    }
  };

#if 0
  // Deduction guide for PartitionKeymap
  template<Dimension NDIM, typename BatchManagerT>
  PartitionKeymap(BatchManagerT&& batch_manager, int np,
                  const Level target_level) -> PartitionKeymap<NDIM, BatchManagerT>;
#endif // 0


  /**
   * Creates a partition keymap for the given number of functions and world.
   * If the number of batches is 0, it will be computed based on the number of processes and functions.
   * The target level is used to determine the level at which the tree is partitioned in each batch.
   * If the target level is 0, it will be computed based on the number of processes.
   * The batch manager is created with the suggested number of batches and distribution.
   */
  template<Dimension NDIM>
  auto make_procmap(size_type num_functions, Batch num_batches = 1, const Level target_level=0,
                    ttg::World world = ttg::default_execution_context()) {
    using BatchManager = DefaultBatchManager<ProcessInfo>;
    return BatchedPartitionKeymap<NDIM, BatchManager>(
                make_batch_manager(num_functions,
                                   num_batches > 0 ? std::min(static_cast<size_type>(num_batches), num_functions)
                                                   : BatchManager::suggest_num_batches(num_functions),
                                   ProcessInfo(world),
                                   (num_batches > world.size())
                                                   ? BatchDistribution::BLOCK
                                                   : BatchManager::suggest_distribution(num_functions)),
                target_level);
  }

  /**
   * Returns a device map for the given number of functions and and devics.
   * We use the procmap to determine the initial level, at least one level below the procmap.
   *
   */
  template<Dimension NDIM>
  auto make_devicemap(const BatchedPartitionKeymap<NDIM, DefaultBatchManager<ProcessInfo>>& procmap) {
    using BatchManager = DefaultBatchManager<DeviceInfo>;
    // set the initial level so we have at least one high-level node per device
    int num_devs = ttg::device::num_devices();
    int target_level = procmap.initial_level() + (num_devs + 7) / 8; // Adjust initial level based on number of devices
    return BatchedPartitionKeymap<NDIM, BatchManager>(
              make_batch_manager(procmap.batch_manager()->num_functions(),
                                 procmap.batch_manager()->num_local_batches(),
                                 DeviceInfo(),
                                 BatchDistribution::FULL),
              target_level);
  }

} // namespace mra



#endif // MRA_MISC_BATCHING_H
