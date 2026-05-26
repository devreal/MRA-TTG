
/***
 * NOTE: this file is outdated but is kept for the time being in case we go back to generating translation matrices ourselves.
 *       It is currently not compiled or tested.
 */



#include <ttg.h>

#include "mra/mra.h"
#include "mra/misc/misc.h"
#include "mra/misc/types.h"
#include "mra/misc/convolutiondata.h"
#include "mra/misc/conv_mad.h"
#include <madness/mra/mra.h>
#include "mra/tensor/tensor.h"
#include <madness/world/world.h>
#include <madness/mra/twoscale.h>
#include <madness/mra/convolution1d.h>

void test_coeffs(int argc, char** argv) {
  constexpr int K = 6; // wavelet order
  constexpr int npt = 2*K; // number of quadrature points
  constexpr double expnt = 5.0; // exponent for the Gaussian
  static double coeff = std::pow(2.0*expnt/std::numbers::pi, 0.25*3);; // coefficient for the Gaussian
  mra::FunctionData<double, 3> functiondata(K), functiondata2(2*K);

  // mra::Convolution<double, 3> conv(K, npt, coeff, expnt, functiondata, functiondata2);
  // const mra::Tensor<double, 1>& rnlp1 = conv.make_rnlp(2, 1);
  // const mra::Tensor<double, 2>& rnlij1 = conv.make_rnlij(0, -1);
  // const mra::Tensor<double, 1>& rnlp2 = conv.get_rnlp(2, -1);
  // const mra::Tensor<double, 2>& rnlij2 = conv.make_rnlij(2, -1);
  // auto rnlij1_view = rnlij1.current_view();
  // auto rnlij2_view = rnlij2.current_view();
  // auto rnlp1_view = rnlp1.current_view();
  // auto rnlp2_view = rnlp2.current_view();

  madness::World world(SafeMPI::COMM_WORLD);
  startup(world, argc, argv);

  mra::OperatorInfo op_info(K, expnt, coeff);
  mra::GaussianConvolutionOperator<double, 3> op(op_info);
  std::shared_ptr<const mra::GaussianOperatorData<double, 3>> op_data = op.get_op(7, mra::Key<3>(0, 7, {64, 64, 64}));

  madness::GaussianConvolution1D<double> conv1d(K, coeff, expnt, 0, false);
  const madness::Tensor<double>& rnlp1_mad = conv1d.rnlp(2, 1);
  madness::Tensor<double> rnlij1_mad = conv1d.rnlij(0, -1);
  const madness::Tensor<double>& rnlp2_mad = conv1d.rnlp(2, -1);
  madness::Tensor<double> rnlij2_mad = conv1d.rnlij(2, -1);
  const madness::ConvolutionData1D<double>* cd_mad = conv1d.nonstandard(7, 64);


  std::cout << "MRA op_R:\n" << op_data->ops[0]->R.current_view() << std::endl << "with norm " << op_data->ops[0]->Rnorm << std::endl;
  std::cout << "MADNESS op_R:\n" << cd_mad->R << std::endl << "with norm " << cd_mad->Rnorm << std::endl;
  std::cout << "Difference between MRA and MADNESS norm for R is " << std::abs(op_data->ops[0]->Rnorm - cd_mad->Rnorm) << std::endl;
  std::cout << "MRA op_S:\n" << op_data->ops[0]->S.current_view() << std::endl << "with norm " << op_data->ops[0]->Snorm << std::endl;
  std::cout << "MADNESS op_S:\n" << cd_mad->T << std::endl << "with norm " << cd_mad->Tnorm << std::endl;
  std::cout << "Difference between MRA and MADNESS norm for S is " << std::abs(op_data->ops[0]->Snorm - cd_mad->Tnorm) << std::endl;

  // std::cout << "MRA rnlij1:\n" << rnlij1_view << std::endl;
  // std::cout << "MADNESS rnlij1:\n" << rnlij1_mad << std::endl;

  // for (int i = 0; i < K; ++i) {
  //   for (int j = 0; j < K; ++j) {
  //       assert(std::abs(rnlij1_view(i, j) - rnlij1_mad(i, j)) < 1e-06);
  //       assert(std::abs(rnlij2_view(i, j) - rnlij2_mad(i, j)) < 1e-06);
  //   }
  // }

  // for (int i = 0; i < 2*K; ++i) {
  //   assert(std::abs(rnlp1_view(i) - rnlp1_mad(i)) < 1e-06);
  //   assert(std::abs(rnlp2_view(i) - rnlp2_mad(i)) < 1e-06);
  // }

  for (int i=0; i<2*K; ++i){
    for (int j=0; j<2*K; ++j){
      assert(std::abs(op_data->ops[0]->R.current_view()(i,j) - cd_mad->R(i,j)) < 1e-06);
    }
  }

  for (int i=0; i<K; ++i){
    for (int j=0; j<K; ++j){
      assert(std::abs(op_data->ops[0]->S.current_view()(i,j) - cd_mad->T(i,j)) < 1e-06);
    }
  }

  world.gop.fence();
}

int main(int argc, char **argv){

  ttg::initialize(argc, argv, 4);
  mra::GLinitialize();

  #if defined(TTG_PARSEC_IMPORTED)
  madness::ParsecRuntime::initialize_with_existing_context(ttg::default_execution_context().impl().context());
  #endif // TTG_PARSEC_IMPORTED
  madness::initialize(argc, argv, /* nthread = */ 1, /* quiet = */ true);

  test_coeffs(argc, argv);

  madness::finalize();
  ttg::execute();
  ttg::fence();
  ttg::finalize();
  return 0;
}
