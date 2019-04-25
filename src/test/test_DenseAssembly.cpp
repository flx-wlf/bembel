// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universtaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
#include "Data.hpp"
#include "DenseAssembly.hpp"
#include "bemtest.h"

using namespace Bembel;
/**
 *  @brief         This is a test to see if the dense and H2 assembly yield the
 * same matrices
 *
 */
int Test::test_DenseAssembly() {
  Geometry myGeom(Bembel::Test::mkSphere());
  MaxwellSingle myMax(std::complex<double>(1, 0));
  HelmholtzSingle myHel(std::complex<double>(1, 0));
  LaplaceSingle myLap;

  for (auto M : {0, 1})
    for (auto P : {0, 1, 2, 3, 4}) {
      {
        Discretization<LaplaceSingle> myDisc(myGeom, myLap, P, 1, M);
        Eigen::HierarchicalMatrix<LaplaceSingle> myH(myDisc, 16, 0.0);
        Eigen::MatrixXd mat = denseAssembly(myDisc);
        Eigen::VectorXd rand = Eigen::VectorXd::Random(mat.rows());
        if ((mat * rand - myH * rand).norm() > 1e-8) {
          return 1;
        }
        Eigen::VectorXd diag1 = diagAssembly(myDisc);
        Eigen::VectorXd diag2 = mat.diagonal();
        if ((diag1 - diag2).norm() > 1e-12) {
          return 1;
        }
      }
      {
        Discretization<HelmholtzSingle> myDisc(myGeom, myHel, P, 1, M);
        Eigen::HierarchicalMatrix<HelmholtzSingle> myH(myDisc, 16, 0.0);
        Eigen::MatrixXcd mat = denseAssembly(myDisc);
        Eigen::VectorXcd rand = Eigen::VectorXcd::Random(mat.rows());
        if ((mat * rand - myH * rand).norm() > 1e-8) {
          return 1;
        }
        Eigen::VectorXcd diag1 = diagAssembly(myDisc);
        Eigen::VectorXcd diag2 = mat.diagonal();
        if ((diag1 - diag2).norm() > 1e-12) {
          return 1;
        }
      }
      {
        Discretization<MaxwellSingle> myDisc(myGeom, myMax, P + 1, 1, M);
        Eigen::HierarchicalMatrix<MaxwellSingle> myH(myDisc, 16, 0.0);
        Eigen::MatrixXcd mat = denseAssembly(myDisc);
        Eigen::VectorXcd rand = Eigen::VectorXcd::Random(mat.rows());
        if ((mat * rand - myH * rand).norm() > 1e-8) {
          return 1;
        }
        Eigen::VectorXcd diag1 = diagAssembly(myDisc);
        Eigen::VectorXcd diag2 = mat.diagonal();
        if ((diag1 - diag2).norm() > 1e-12) {
          return 1;
        }
      }
    }

  return (0);
}
