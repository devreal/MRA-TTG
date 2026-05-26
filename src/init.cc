
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
    ttg::initialize(argc, argv, ncores);

    mra::GLinitialize();

  /* initialize MADNESS PaRSEC backend with the same PaRSEC context */

#if defined(TTG_PARSEC_IMPORTED)
    madness::ParsecRuntime::initialize_with_existing_context(ttg::default_execution_context().impl().context());
#endif // TTG_PARSEC_IMPORTED
    madness::initialize(argc, argv, /* nthread = */ 1, /* quiet = */ true);
    madness::World& world = madness::World::get_default();
    madness::startup(world, 0, nullptr, false);

#if !defined(MRA_ENABLE_HOST) && defined(MRA_HAVE_SCRATCH_ALLOCATOR)
    // adjust the pinned memory allocator through TA
    TiledArray::device::Env::initialize(TiledArray::get_default_world(), 1UL<<32, 1UL<<40);
#endif // MRA_ENABLE_HOST
  }


  void finalize() {
    madness::finalize();
    ttg::finalize();
  }




} // namespace mra