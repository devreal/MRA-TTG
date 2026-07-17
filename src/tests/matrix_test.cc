#include <cassert>
#include <ttg.h>
#include "mra/ops/mxm.h"
#include "mra/tensor/tensor.h"

using namespace mra;

static void test_mxm() {


  /**
   * mxm test:
   *            | 1 1 2 |
   *            | 1 1 1 |
   *            | 1 1 1 |
   *
   * | 1 2 1 |  | 4 4 5 |
   * | 1 1 1 |  | 3 3 4 |
   * | 1 1 1 |  | 3 3 4 |
   */

  constexpr const mra::size_type K = 3;
  mra::DenseTensor<double, 2> nodeA(K), nodeB(K), nodeC(K), expected(K);
  mra::DenseTensorView<double, 2> nodeAv = nodeA.current_view();
  mra::DenseTensorView<double, 2> nodeBv = nodeB.current_view();
  mra::DenseTensorView<double, 2> nodeCv = nodeC.current_view();
  mra::DenseTensorView<double, 2> expectedv = expected.current_view();
  for (int i=0; i<nodeA.size(); ++i){
    nodeAv[i] = 1;
    nodeBv[i] = 1;
    nodeCv[i] = 100.0;
    expectedv[i] = 3;
  }

  nodeAv(0, 1) += 1;
  nodeBv(0, 2) += 1;

  expectedv(0, 0) = 4;
  expectedv(0, 1) = 4;
  expectedv(0, 2) = 5;
  expectedv(1, 2) = 4;
  expectedv(2, 2) = 4;

  std::cout << "mxm test" << std::endl;

  auto check = [&](double init) {
    for (int i = 0; i < K; ++i) {
      for (int j = 0; j < K; ++j) {
        if (nodeCv(i, j) != expectedv(i, j)+init) {
          std::cout << "Mismatch in " << i << ", " << j << ":\n" << nodeC << std::endl;
          std::cout << "A:\n" << nodeA << std::endl;
          std::cout << "B:\n" << nodeB << std::endl;
          std::cout << "expected:\n" << expected << std::endl;
          throw std::runtime_error("Validation in mTxmT failed!");
        }
      }
    }
  };


  /**
   * check mxmq: old values are overwritten
   */
  mra::mxmq(K, K, K, nodeCv.data(), nodeAv.data(), nodeBv.data());
  check(0);

  /**
   * check mxm: old values are 100
   */
  nodeCv = 100.0;
  mra::mxm(K, K, K, nodeCv.data(), nodeAv.data(), nodeBv.data());
  check(100);
}


static void test_mTxm() {


  /**
   * mTxm test:
   *            | 1 1 2 |
   *            | 1 1 1 |
   *            | 1 1 1 |
   *
   * | 1 2 1 |  | 3 3 4 |
   * | 1 1 1 |  | 4 4 5 |
   * | 1 1 1 |  | 3 3 4 |
   */

  constexpr const mra::size_type K = 3;
  mra::DenseTensor<double, 2> nodeA(K), nodeB(K), nodeC(K), expected(K);
  mra::DenseTensorView<double, 2> nodeAv = nodeA.current_view();
  mra::DenseTensorView<double, 2> nodeBv = nodeB.current_view();
  mra::DenseTensorView<double, 2> nodeCv = nodeC.current_view();
  mra::DenseTensorView<double, 2> expectedv = expected.current_view();
  for (int i=0; i<nodeA.size(); ++i){
    nodeAv[i] = 1;
    nodeBv[i] = 1;
    nodeCv[i] = 100.0;
    expectedv[i] = 3;
  }

  nodeAv(0, 1) += 1;
  nodeBv(0, 2) += 1;

  expectedv(0, 2) = 4;
  expectedv(1, 0) = 4;
  expectedv(1, 1) = 4;
  expectedv(1, 2) = 6;
  expectedv(2, 2) = 4;

  std::cout << "mTxm test" << std::endl;

  auto check = [&](double init) {
    for (int i = 0; i < K; ++i) {
      for (int j = 0; j < K; ++j) {
        if (!(nodeCv(i, j) == expectedv(i, j)+init)) {
          std::cout << "Mismatch in " << i << ", " << j << ":\n" << nodeC << std::endl;
          std::cout << "A:\n" << nodeA << std::endl;
          std::cout << "B:\n" << nodeB << std::endl;
          std::cout << "expected:\n" << expected << std::endl;
          throw std::runtime_error("Validation in mTxm failed!");
        }
        assert(nodeCv(i, j) == expectedv(i, j)+init);
      }
    }
  };


  /**
   * check mxmq: old values are overwritten
   */
  mra::mTxmq(K, K, K, nodeCv.data(), nodeAv.data(), nodeBv.data());
  check(0);

  /**
   * check mxm: old values are 100
   */
  nodeCv = 100.0;
  mra::mTxm(K, K, K, nodeCv.data(), nodeAv.data(), nodeBv.data());
  check(100);
}

static void test_mxmT() {


  /**
   * mxmT test:
   *            | 1 1 2 |
   *            | 1 1 1 |
   *            | 1 1 1 |
   *
   * | 1 2 1 |  | 5 4 4 |
   * | 1 1 1 |  | 4 3 3 |
   * | 1 1 1 |  | 4 3 3 |
   */

  constexpr const mra::size_type K = 3;
  mra::DenseTensor<double, 2> nodeA(K), nodeB(K), nodeC(K), expected(K);
  mra::DenseTensorView<double, 2> nodeAv = nodeA.current_view();
  mra::DenseTensorView<double, 2> nodeBv = nodeB.current_view();
  mra::DenseTensorView<double, 2> nodeCv = nodeC.current_view();
  mra::DenseTensorView<double, 2> expectedv = expected.current_view();
  for (int i=0; i<nodeA.size(); ++i){
    nodeAv[i] = 1;
    nodeBv[i] = 1;
    nodeCv[i] = 100.0;
    expectedv[i] = 3;
  }

  nodeAv(0, 1) += 1;
  nodeBv(0, 2) += 1;

  expectedv(0, 0) = 5;
  expectedv(0, 1) = 4;
  expectedv(0, 2) = 4;
  expectedv(1, 0) = 4;
  expectedv(2, 0) = 4;

  std::cout << "mxmT test" << std::endl;

  auto check = [&](double init) {
    for (int i = 0; i < K; ++i) {
      for (int j = 0; j < K; ++j) {
        if (!(nodeCv(i, j) == expectedv(i, j)+init)) {
          std::cout << "Mismatch in " << i << ", " << j << ":\n" << nodeC << std::endl;
          std::cout << "A:\n" << nodeA << std::endl;
          std::cout << "B:\n" << nodeB << std::endl;
          std::cout << "expected:\n" << expected << std::endl;
          throw std::runtime_error("Validation in mxmT failed!");
        }
      }
    }
  };


  /**
   * check mxmq: old values are overwritten
   */
  mra::mxmTq(K, K, K, nodeCv.data(), nodeAv.data(), nodeBv.data());
  check(0);

  /**
   * check mxm: old values are 100
   */
  nodeCv = 100.0;
  mra::mxmT(K, K, K, nodeCv.data(), nodeAv.data(), nodeBv.data());
  check(100);
}

static void test_mTxmT() {


  /**
   * mTxmT test:
   *            | 1 1 2 |
   *            | 1 1 1 |
   *            | 1 1 1 |
   *
   * | 1 2 1 |  | 4 3 3 |
   * | 1 1 1 |  | 5 4 4 |
   * | 1 1 1 |  | 4 3 3 |
   */

  constexpr const mra::size_type K = 3;
  mra::DenseTensor<double, 2> nodeA(K), nodeB(K), nodeC(K), expected(K);
  mra::DenseTensorView<double, 2> nodeAv = nodeA.current_view();
  mra::DenseTensorView<double, 2> nodeBv = nodeB.current_view();
  mra::DenseTensorView<double, 2> nodeCv = nodeC.current_view();
  mra::DenseTensorView<double, 2> expectedv = expected.current_view();
  for (int i=0; i<nodeA.size(); ++i){
    nodeAv[i] = 1;
    nodeBv[i] = 1;
    nodeCv[i] = 100.0;
    expectedv[i] = 3;
  }

  nodeAv(0, 1) += 1;
  nodeBv(0, 2) += 1;

  expectedv(0, 0) = 4;
  expectedv(1, 0) = 5;
  expectedv(1, 1) = 4;
  expectedv(1, 2) = 4;
  expectedv(1, 1) = 4;
  expectedv(2, 0) = 4;

  std::cout << "mTxmT test" << std::endl;

  auto check = [&](double init) {
    for (int i = 0; i < K; ++i) {
      for (int j = 0; j < K; ++j) {
        if (!(nodeCv(i, j) == expectedv(i, j)+init)) {
          std::cout << "Mismatch in " << i << ", " << j << ":\n" << nodeC << std::endl;
          std::cout << "A:\n" << nodeA << std::endl;
          std::cout << "B:\n" << nodeB << std::endl;
          std::cout << "expected:\n" << expected << std::endl;
          throw std::runtime_error("Validation in mTxmT failed!");
        }
      }
    }
  };


  /**
   * check mxmq: old values are overwritten
   */
  mra::mTxmTq(K, K, K, nodeCv.data(), nodeAv.data(), nodeBv.data());
  check(0);

  /**
   * check mxm: old values are 100
   */
  nodeCv = 100.0;
  mra::mTxmT(K, K, K, nodeCv.data(), nodeAv.data(), nodeBv.data());
  check(100);
}

int main(int argc, char **argv) {
  ttg::initialize(argc, argv);

  test_mxm();
  test_mTxm();
  test_mxmT();
  test_mTxmT();

  ttg::execute();
  ttg::fence();

  ttg::finalize();
}