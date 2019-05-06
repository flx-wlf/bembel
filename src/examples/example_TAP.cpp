// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universtaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>
#include "Discretization.hpp"
#include "Geometry.hpp"
#include "HierarchicalMatrix.hpp"
#include "PDEproblem.hpp"

#include "Data.hpp"
#include "Error.hpp"
#include "EvalSolution.hpp"
#include "Grids.hpp"
#include "Logger.hpp"
#include "Rhs.hpp"
#include "Stopwatch.hpp"
#include "Visualize.hpp"


int main() {
  using namespace Bembel;

  std::vector<std::string> geo_files = {"sphere", "fichera", "toy_boat"};

  for (std::string g : geo_files) {
    const std::complex<double> wavenumber = (g == "toy_boat")
                                                ? std::complex<double>(5, 0)
                                                : std::complex<double>(1, 0);
    MaxwellSingle myMax(wavenumber);

    Geometry myGeom("../geo/" + g + ".dat");

    Eigen::MatrixXd gridpoints;

    if (g == "toy_boat")
      gridpoints =
          Util::makeTensorProductGrid(Eigen::VectorXd::LinSpaced(3, .99, 1.01),
                                      Eigen::VectorXd::LinSpaced(3, -.01, .01),
                                      Eigen::VectorXd::LinSpaced(3, -.01, .01));
    if (g == "sphere") gridpoints = Util::makeSphereGrid(3, 100);
    if (g == "fichera") gridpoints = Util::makeSphereGrid(4, 100);

    Eigen::Vector3d position;
    if (g == "sphere") position = Eigen::Vector3d(0.1, 0.1, 0);
    if (g == "fichera") position = Eigen::Vector3d(.5, .5, .5);
    if (g == "toy_boat") position = Eigen::Vector3d(7, 2, 0);

    const std::function<Eigen::Vector3cd(Eigen::Vector3d, std::complex<double>)>
        fun = [=](Eigen::Vector3d pt, std::complex<double> kappa) {
          const Eigen::Vector3d length(0, 0.1, 0.1);
          return Data::Dipole(pt, kappa, position, length);
        };

    std::vector<int> Ps = {1, 2, 3};
    if (g != "toy_boat") Ps.push_back(4);

    for (auto P : Ps) {
      Util::Logger<10> log("Maxwell_TAP_" + g + "_" + std::to_string(P) +
                           ".log");
      log.term("- - -", g, "P=" + std::to_string(P), "- - -", "- - -");
      log.both("M", "t_mat_B", "t_mat_R", "t_solve_B", "t_solve_R", "t_tot_B",
               "t_tot_R", "DOFs_B", "DOFs_R", "error_B", "error_R", "iter_B",
               "iter_R");

      Util::Stopwatch sw;

      std::vector<int> Ms = {1, 2, 3};
      if (g == "sphere" && P < 3) Ms.push_back(4);
      if (g == "sphere" && P < 2) Ms.push_back(5);
      if (g == "toy_boat" && P < 3) Ms.push_back(4);

      for (auto M : Ms) {
        double timeMat_B;
        double timeSolve_B;
        double timeTot_B;
        long DOFS_B;
        long iter_B;
        double err_B;
        double timeMat_R;
        double timeSolve_R;
        double timeTot_R;
        long DOFS_R;
        long iter_R;
        double err_R;

        {  // Higher Regular Functions
          Discretization<MaxwellSingle> myDisc(myGeom, myMax, P, 1, M);
          DOFS_B = myDisc.get_num_dofs();
          Eigen::VectorXcd rhs = Rhs::computeRhs(myDisc, fun);

          sw.start();

          Eigen::HierarchicalMatrix<MaxwellSingle> myH(myDisc, 20);

          timeMat_B = sw.lap();

          Eigen::GMRES<Eigen::HierarchicalMatrix<MaxwellSingle>,
                       Eigen::IdentityPreconditioner>
              gmres;
          gmres.setTolerance(1e-10);
          gmres.set_restart(1500);
          gmres.compute(myH);
          Eigen::VectorXcd rho = gmres.solve(rhs);
          timeSolve_B = sw.lap();

          iter_B = gmres.iterations();

          Eigen::MatrixXcd pot = Sol::evalSolution(gridpoints, rho, myDisc);
          timeTot_B = sw.stop();

          err_B =
              maxPointwiseError(pot, gridpoints, fun, myMax.get_wavenumber());
        }

        {  // Equiv. to RT on Quads
          Discretization<MaxwellSingle> myDisc(myGeom, myMax, P, P + 1, M);
          DOFS_R = myDisc.get_num_dofs();

          Eigen::VectorXcd rhs = Rhs::computeRhs(myDisc, fun);

          sw.start();

          Eigen::HierarchicalMatrix<MaxwellSingle> myH(myDisc, 20);

          timeMat_R = sw.lap();

          Eigen::GMRES<Eigen::HierarchicalMatrix<MaxwellSingle>,
                       Eigen::IdentityPreconditioner>
              gmres;
          gmres.setTolerance(1e-10);
          gmres.set_restart(1500);
          gmres.compute(myH);
          Eigen::VectorXcd rho = gmres.solve(rhs);
          timeSolve_R = sw.lap();

          iter_R = gmres.iterations();

          Eigen::MatrixXcd pot = Sol::evalSolution(gridpoints, rho, myDisc);
          timeTot_R = sw.stop();

          err_R =
              maxPointwiseError(pot, gridpoints, fun, myMax.get_wavenumber());
        }

        log.both(M, timeMat_B, timeMat_R, timeSolve_B, timeSolve_R, timeTot_B,
                 timeTot_R, DOFS_B, DOFS_R, err_B, err_R, iter_B, iter_R);
      }
    }
  }

  return 0;
}
