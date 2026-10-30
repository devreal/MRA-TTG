#ifndef MRA_MISC_BATCH_GROUP_H
#define MRA_MISC_BATCH_GROUP_H

#include <any>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <ttg.h>

namespace mra {

  /**
   * Lets the device tasks of several instances of the same operation be
   * batched together into one kernel launch, e.g., two make_compress or
   * make_concat instances working on different function sets at the same
   * time. Without a shared group, tasks are only batched with tasks of the
   * same instance. Pass the same BatchGroup (copies refer to the same group)
   * to every instance that should share batches:
   *
   *   mra::BatchGroup compress_group;
   *   auto c1 = make_compress(fns1, K, false, fd, in1, out1, "c1", {}, {}, false, T{}, 0, T{}, compress_group);
   *   auto c2 = make_compress(fns2, K, false, fd, in2, out2, "c2", {}, {}, false, T{}, 0, T{}, compress_group);
   *
   * All instances of a group must be created by the same factory with the same
   * template arguments, and must agree on the parameters the batched kernel
   * shares across a batch (e.g., K); the factories throw otherwise. The
   * instances also share one batch pool.
   */
  class BatchGroup {
  public:
    BatchGroup() : m_state(std::make_shared<state_t>()) { }

    /**
     * Returns the batch pool of this group, created by make_pool() for the first
     * instance. params are the parameters shared by every batch of the group.
     * @throw std::invalid_argument if an instance with different params, or a
     *        different pool type, joined before
     */
    template <typename Pool, typename MakePool, typename... Params>
    std::shared_ptr<Pool> pool(MakePool&& make_pool, const Params&... params) const {
      using params_t = std::tuple<Params...>;
      std::lock_guard<std::mutex> lock(m_state->mtx);
      if (!m_state->params.has_value()) {
        m_state->params = params_t(params...);
        m_state->pool = std::shared_ptr<Pool>(make_pool());
      }
      auto* p = std::any_cast<params_t>(&m_state->params);
      auto* pool = std::any_cast<std::shared_ptr<Pool>>(&m_state->pool);
      if (nullptr == p || nullptr == pool) {
        throw std::invalid_argument("mra::BatchGroup: all instances of a group must be of the same operation");
      }
      if (*p != params_t(params...)) {
        throw std::invalid_argument("mra::BatchGroup: all instances of a group must use the same batch parameters");
      }
      return *pool;
    }

    /**
     * Allows the tasks of tt to be batched with those of every other TT added to
     * this group (see TT::set_batch_group). A no-op on host builds.
     */
    template <typename TTPtr>
    void add(TTPtr& tt) const {
#ifndef MRA_ENABLE_HOST
      tt->set_batch_group(m_state->group);
#endif // MRA_ENABLE_HOST
    }

  private:
    struct state_t {
      std::mutex mtx;
      std::any params;
      std::any pool;
#ifndef MRA_ENABLE_HOST
      ttg::device::batch_group group;
#endif // MRA_ENABLE_HOST
    };
    std::shared_ptr<state_t> m_state;
  };

} // namespace mra

#endif // MRA_MISC_BATCH_GROUP_H
