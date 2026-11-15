
#include <cstdlib>
#include <madness/mra/mra.h>
#include <madness/world/world.h>
#include <madness/mra/operator.h>
#include <ttg.h>

#include "mra/misc/init.h"
#include "mra/misc/platform.h"
#include "mra/misc/gl.h"
#include "mra/misc/allocator.h"

namespace mra {

  void initialize(int& argc, char **& argv, int ncores) {
    // PaRSEC's device memory zone allocator hands out multiples of its block size (512 KB by
    // default), which wastes most of a block on the many small allocations of the MRA-TTG
    // graphs (tree nodes, norms, sparsity bytes): 4 KB blocks, unless set in the environment
    setenv("PARSEC_MCA_device_cuda_memory_block_size", "4096", /* overwrite = */ 0);
    ttg::initialize(argc, argv, ncores);

    mra::GLinitialize();

  /* initialize MADNESS PaRSEC backend with the same PaRSEC context */

#if defined(TTG_PARSEC_IMPORTED)
    madness::ParsecRuntime::initialize_with_existing_context(ttg::default_execution_context().impl().context());
#endif // TTG_PARSEC_IMPORTED
    madness::initialize(argc, argv, /* nthread = */ 1, /* quiet = */ true);
    madness::World& world = madness::World::get_default();
    madness::startup(world, argc, argv, false);

#if !defined(MRA_ENABLE_HOST) && defined(MRA_HAVE_SCRATCH_ALLOCATOR)
    // adjust the pinned memory allocator through TTG's environment settings
    ttg::Env::initialize(1UL<<32, 1UL<<40);
#endif // MRA_ENABLE_HOST
  }


  void finalize() {
    madness::finalize();
    detail::run_finalize_hooks();
    ttg::finalize();
  }




} // namespace mra