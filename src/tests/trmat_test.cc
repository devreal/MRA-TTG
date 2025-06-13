#include <ttg.h>

#include "mra/misc/convolutiondata.h"
#include <madness/mra/mra.h>
#include <madness/world/world.h>
#include <madness/mra/operator.h>


void test_convmat(int argc, char **argv){

  mra::ConvolutionData<double, 1> conv_data(4, 3, 10, 2, 10);

  madness::World world(SafeMPI::COMM_WORLD);
  startup(world,argc,argv);
}

int main(int argc, char **argv){

  #if defined(TTG_PARSEC_IMPORTED)
  madness::ParsecRuntime::initialize_with_existing_context(ttg::default_execution_context().impl().context());
  #endif // TTG_PARSEC_IMPORTED
  madness::initialize(argc, argv, /* nthread = */ 1, /* quiet = */ true);


  test_convmat(argc, argv);
  madness::finalize();

  return 0;
}
