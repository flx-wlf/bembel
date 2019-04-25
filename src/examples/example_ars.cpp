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
#include "DenseAssembly.hpp"
#include "Error.hpp"
#include "EvalSolution.hpp"
#include "Grids.hpp"
#include "Logger.hpp"
#include "Rhs.hpp"
#include "Stopwatch.hpp"
#include "Visualize.hpp"

int main() {
  using namespace Bembel;
  const std::complex<double> wavenumber(1, 0);
  MaxwellSingle myMax(wavenumber);

  {
    Geometry myGeom("../geo/sphere.dat");
    Eigen::MatrixXd gridpoints = Util::makeSphereGrid(2, 100);
    const std::function<Eigen::Vector3cd(Eigen::Vector3d, std::complex<double>)>
        fun = [](Eigen::Vector3d pt, std::complex<double> kappa) {
          const Eigen::Vector3d position(0, 0.1, 0.1);
          const Eigen::Vector3d length(0, 0.1, 0.1);
          return Data::Dipole(pt, kappa, position, length);
        };

    std::cout << "\n+ + + + + + + + + + + + + + + + + + Sphere + + + + + + + + + + + + + + + + + +\n";
    Util::Logger<10> log("ars_sphere.log");
    log.both("P", "t_mat", "t_solve", "t_tot", "DOFs", "error");

    for (auto P : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}) {
      Util::Stopwatch sw;
      for (auto M : {0}) {
        Discretization<MaxwellSingle> myDisc(myGeom, myMax, P, 1,
                                             M);
        Eigen::VectorXcd rhs = Rhs::computeRhs(myDisc, fun);
        sw.start();
        Eigen::MatrixXcd mat = denseAssembly(myDisc);
        const double timeMat = sw.lap();
        Eigen::VectorXcd rho = mat.lu().solve(rhs);
        const double timeSolve = sw.lap();
        Eigen::MatrixXcd pot = Sol::evalSolution(gridpoints, rho, myDisc);
        const double timeTot = sw.stop();
        if (P == 4)
          Bembel::Vis::plotDiscretizationToVTK(myDisc, rho, "ars_sphere.vtk",
                                               7);
        log.both(
            P, timeMat, timeSolve, timeTot, myDisc.get_num_dofs(),
            maxPointwiseError(pot, gridpoints, fun, myMax.get_wavenumber()));
      }
    }
  }

  {
    Geometry myGeom("../geo/toy_boat.dat");
    Eigen::MatrixXd gridpoints = Util::makeSphereGrid(6, 100);
    const std::function<Eigen::Vector3cd(Eigen::Vector3d, std::complex<double>)>
        fun = [](Eigen::Vector3d pt, std::complex<double> kappa) {
          const Eigen::Vector3d position(1, 0.0, 0);
          const Eigen::Vector3d length(0, 0, 0.1);
          return Data::Dipole(pt, kappa, position, length);
        };
    Util::Logger<10> log("ars_ship.log");
    std::cout << "\n+ + + + + + + + + + + + + + + + + + Ship + + + + + + + + + + + + + + + + + +\n";
    log.both("P", "t_mat", "t_solve", "t_tot", "DOFs", "error");
    for (auto P : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) {
      Util::Stopwatch sw;
      for (auto M : {0}) {
        Discretization<MaxwellSingle> myDisc(myGeom, myMax, P, 1,
                                             M);
        Eigen::VectorXcd rhs = Rhs::computeRhs(myDisc, fun);
        sw.start();
        Eigen::MatrixXcd mat = denseAssembly(myDisc);
        const double timeMat = sw.lap();
        Eigen::VectorXcd rho = mat.lu().solve(rhs);
        const double timeSolve = sw.lap();
        Eigen::MatrixXcd pot = Sol::evalSolution(gridpoints, rho, myDisc);
        const double timeTot = sw.stop();
        if (P == 4){
          Bembel::Vis::plotDiscretizationToVTK(myDisc, rho, "ars_ship.vtk", 7);
        }
        log.both(
            P, timeMat, timeSolve, timeTot, myDisc.get_num_dofs(),
            maxPointwiseError(pot, gridpoints, fun, myMax.get_wavenumber()));
      }
    }
  }

  return (0);
}
