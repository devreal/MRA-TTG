#ifndef MRA_MISC_INIT_H
#define MRA_MISC_INIT_H

#include <functional>
#include <mutex>
#include <utility>
#include <vector>

namespace mra {

  namespace detail {
    struct finalize_hooks_t {
      std::mutex mtx;
      std::vector<std::function<void()>> hooks;
    };

    inline finalize_hooks_t& finalize_hooks() {
      static finalize_hooks_t h;
      return h;
    }

    /**
     * Registers fn to run in mra::finalize(), after MADNESS is finalized and
     * before TTG (and with it PaRSEC) is. For process-wide state that holds
     * runtime resources, e.g. device memory from PaRSEC's zone, which has to
     * be returned while the runtime still exists rather than at static
     * destruction. Hooks run in reverse order of registration.
     */
    inline void at_finalize(std::function<void()> fn) {
      auto& h = finalize_hooks();
      std::lock_guard<std::mutex> lock(h.mtx);
      h.hooks.push_back(std::move(fn));
    }

    inline void run_finalize_hooks() {
      auto& h = finalize_hooks();
      std::vector<std::function<void()>> hooks;
      {
        std::lock_guard<std::mutex> lock(h.mtx);
        hooks.swap(h.hooks);
      }
      for (auto it = hooks.rbegin(); it != hooks.rend(); ++it) (*it)();
    }
  } // namespace detail

  /**
   * Initializes the runtime systems necessary for MRA.
   * This includes TTG, MADNESS, and some adjustments to TA.
   * TODO: take a MADNESS world as argument.
   */
  void initialize(int& argc, char **& argv, int ncores);

  /**
   * Finalize the environment set up in mra::initialize.
   */
  void finalize();


} // namespace mra


#endif // MRA_MISC_INIT_H