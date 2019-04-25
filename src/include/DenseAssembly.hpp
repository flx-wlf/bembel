// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universtaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
#ifndef __BEMBEL_DENSE_ASSEMBL_
#define __BEMBEL_DENSE_ASSEMBL_

#include "Discretization.hpp"
#include "HierarchicalMatrix.hpp"
#include "constants.h"
#include "hmatrixfactory.h"
#include "hmatrixsettings.h"

namespace Bembel {

template <class PDE>
Eigen::SparseMatrix<double> get_projector_as_eigen(
    Bembel::Discretization<PDE>& disc) {
  Bembel::sparsecore<double>* T = disc.get_disc().T;
  const int Tsmall = std::min(T->m, T->n);
  const int Tbig = std::max(T->m, T->n);
  const int num_dofs = disc.get_num_dofs();

  const int large_projsize = [&]() {
    if (num_dofs == Tsmall) return Tbig;          // real valued
    if (num_dofs == Tsmall / 2) return Tbig / 2;  // complex valued
    assert(false && "Something is wrong with the legacy projector");
  }();

  const int inner_loopsize = [&]() {
    if (PDEproblemTraits<PDE>::dimFunctions == 2)
      return large_projsize / 2;  // vector-valued dofs
    return large_projsize;        // scalar dofs
  }();

  Eigen::SparseMatrix<double> out(num_dofs, large_projsize);

  std::vector<Eigen::Triplet<double>> trips;
  trips.reserve(Tsmall);

  for (int i = 0; i < num_dofs; i++) {
    for (int j = 0; j < inner_loopsize; j++) {
      double val = Bembel::get_sparse<double>(T, j, i);
      if (std::abs(val) > 0.000001) {
        trips.emplace_back(i, j, val);
      }
      if (PDEproblemTraits<PDE>::dimFunctions == 2) {
        double val2 = Bembel::get_sparse<double>(T, j + large_projsize, i);
        if (std::abs(val2) > 0.000001)
          trips.emplace_back(i, j + inner_loopsize, val2);
      }
    }
  }
  out.setFromTriplets(trips.begin(), trips.end());
  return out;
}

// These will convert the output of get_stiff, i.e., the entries of one
// block of the large superspace matrix corresponding to one element to
// element interaction to an eigen sparse matrix.
inline Eigen::SparseMatrix<std::complex<double>> ptrmat_to_sparse(
    Bembel::Discretization<Bembel::HelmholtzSingle>& disc, double* in,
    int pos_i, int pos_j) {
  // static_assert(not(std::is_same<T, double>::value));  // Exclude Laplace
  const int a_bs = disc.get_disc().a_bs;
  const int num_els = disc.get_num_elem();
  const int big_dofsize = num_els * a_bs;
  const int a_bs2 = a_bs * a_bs;
  Eigen::SparseMatrix<std::complex<double>> local_matrix(big_dofsize,
                                                         big_dofsize);
  std::vector<Eigen::Triplet<std::complex<double>>> trips;

  trips.reserve(2 * a_bs2);
  for (int i = 0; i < a_bs; i++) {
    for (int j = 0; j < a_bs; j++) {
      std::complex<double> val(in[i * a_bs + j], in[i * a_bs + j + a_bs2]);
      trips.emplace_back(pos_i * a_bs + i, pos_j * a_bs + j, val);
      if (not(pos_i == pos_j)) {
        trips.emplace_back(pos_j * a_bs + j, pos_i * a_bs + i, val);
      }
    }
  }
  local_matrix.setFromTriplets(trips.begin(), trips.end());
  return local_matrix;
};

inline Eigen::SparseMatrix<std::complex<double>> ptrmat_to_sparse(
    Bembel::Discretization<Bembel::MaxwellSingle>& disc, double* in, int pos_i,
    int pos_j) {
  // static_assert(not(std::is_same<T, double>::value));  // Exclude Laplace
  const int a_bs = disc.get_disc().a_bs;
  const int num_els = disc.get_num_elem();
  const int big_dofsize = num_els * a_bs * 2;
  const int a_bs2 = a_bs * a_bs;
  Eigen::SparseMatrix<std::complex<double>> local_matrix(big_dofsize,
                                                         big_dofsize);
  std::vector<Eigen::Triplet<std::complex<double>>> trips;

  trips.reserve(8 * a_bs2);
  const int s = big_dofsize / 2;
  for (int i = 0; i < a_bs; i++) {
    for (int j = 0; j < a_bs; j++) {
      std::complex<double> dxdx(in[i * a_bs + j + 0 * a_bs2],
                                in[i * a_bs + j + 4 * a_bs2]);
      std::complex<double> dxdy(in[i * a_bs + j + 1 * a_bs2],
                                in[i * a_bs + j + 5 * a_bs2]);
      std::complex<double> dydx(in[i * a_bs + j + 2 * a_bs2],
                                in[i * a_bs + j + 6 * a_bs2]);
      std::complex<double> dydy(in[i * a_bs + j + 3 * a_bs2],
                                in[i * a_bs + j + 7 * a_bs2]);

      trips.emplace_back(pos_i * a_bs + i, pos_j * a_bs + j, dxdx);
      trips.emplace_back(pos_i * a_bs + i + s, pos_j * a_bs + j + s, dydy);
      trips.emplace_back(pos_i * a_bs + i, pos_j * a_bs + j + s, dxdy);
      trips.emplace_back(pos_i * a_bs + i + s, pos_j * a_bs + j, dydx);
      if (not(pos_i == pos_j)) {
        trips.emplace_back(pos_j * a_bs + j, pos_i * a_bs + i, dxdx);
        trips.emplace_back(pos_j * a_bs + j + s, pos_i * a_bs + i + s, dydy);
        trips.emplace_back(pos_j * a_bs + j + s, pos_i * a_bs + i, dxdy);
        trips.emplace_back(pos_j * a_bs + j, pos_i * a_bs + i + s, dydx);
      }
    }
  }
  local_matrix.setFromTriplets(trips.begin(), trips.end());
  return local_matrix;
};

inline Eigen::SparseMatrix<double> ptrmat_to_sparse(
    Bembel::Discretization<Bembel::LaplaceSingle>& disc, double* in, int pos_i,
    int pos_j) {
  const int a_bs = disc.get_disc().a_bs;
  const int num_els = disc.get_num_elem();
  const int big_dofsize = num_els * a_bs;
  Eigen::SparseMatrix<double> local_matrix(big_dofsize, big_dofsize);
  std::vector<Eigen::Triplet<double>> trips;
  trips.reserve(a_bs * a_bs);
  for (int i = 0; i < a_bs; i++) {
    for (int j = 0; j < a_bs; j++) {
      trips.emplace_back(pos_i * a_bs + i, pos_j * a_bs + j, in[i * a_bs + j]);
      if (not(pos_i == pos_j)) {
        trips.emplace_back(pos_j * a_bs + j, pos_i * a_bs + i,
                           in[i * a_bs + j]);
      }
    }
  }
  local_matrix.setFromTriplets(trips.begin(), trips.end());
  return local_matrix;
};

template <class PDE>
inline Eigen::Matrix<typename Bembel::PDEproblemTraits<PDE>::Scalar, -1, -1>
denseAssembly(Bembel::Discretization<PDE>& disc) {
  // We set some required sizes
  const int small_dofsize = disc.get_num_dofs();
  const int a_bs = disc.get_disc().a_bs;
  const int num_els = disc.get_num_elem();
  const int big_dofsize = num_els * a_bs;
  const int bufferSize = disc.get_pde().get_pde().quadrature_bufsize;

  // We initialize the projector. This is done slightly differently than in the
  // projector for the compressed matrix, since the compressed version still
  // uses the legacy sparse format.

  // We allocate the output as a dense matrix
  Eigen::Matrix<typename Bembel::PDEproblemTraits<PDE>::Scalar, -1, -1> out =
      Eigen::Matrix<typename Bembel::PDEproblemTraits<PDE>::Scalar, -1,
                    -1>::Zero(small_dofsize, small_dofsize);

  Bembel::hmatrixsettings hmatset;
  Bembel::hmatrixfactory hmatfac;

  // 4 is a placeholder, it only affects min_bsize during matrix compression,
  // which is not the case here
  hmatset = Bembel::get_hmatrixsettings(4, &(disc.get_disc()));
  hmatset.eta = 0;
  hmatfac.disc = &(disc.get_disc());
  hmatfac.hmatset = &hmatset;
  hmatfac.assemfmats = 0;
  hmatfac.assemsfmats = 0;
  hmatfac.assemrkmats = 0;
  Bembel::init_Gauss_Square(&hmatfac.Q, Bembel::g_max + 1);
  Bembel::pdeproblem pde = disc.get_pde().get_pde();

  std::vector<Spl::Patch> geom = disc.get_plain_patchdata();

  pde.init_randwerte(&hmatfac.RW, &hmatfac.Q[disc.get_disc().g_far], geom,
                     1 << disc.get_M());

  Bembel::et_node* pE = disc.get_disc().mesh->E.patch[0];
  while (pE->son[0]) pE = pE->son[0];

  Eigen::SparseMatrix<double> projector = get_projector_as_eigen(disc);

  // Now we assemble the dense matrix from the blocks without assemblying the
  // large dense system

#pragma omp parallel for shared(out) default(shared)
  for (int zi = 0; zi < num_els; zi++) {
    std::vector<double> ar(bufferSize * a_bs * a_bs);
    for (int si = zi; si < num_els; si++) {
      get_stiff(&hmatfac, disc.get_disc().mesh->P, ar.data(), pE, disc.get_M(),
                zi, si);
#pragma omp critical
      {
        out += projector * ptrmat_to_sparse(disc, ar.data(), zi, si) *
               projector.transpose();
      }
    }
  }

  pde.free_randwerte(&hmatfac.RW, 1 << disc.get_disc().mesh->M, geom.size());
  Bembel::free_Gauss_Square(&hmatfac.Q, Bembel::g_max + 1);
  return out;
}

template <class PDE>
inline Eigen::Matrix<typename Bembel::PDEproblemTraits<PDE>::Scalar, -1, 1>
diagAssembly(Bembel::Discretization<PDE>& disc) {
  // We set some required sizes
  const int small_dofsize = disc.get_num_dofs();
  const int a_bs = disc.get_disc().a_bs;
  const int num_els = disc.get_num_elem();
  const int big_dofsize = num_els * a_bs;
  const int bufferSize = disc.get_pde().get_pde().quadrature_bufsize;

  // We initialize the projector. This is done slightly differently than in the
  // projector for the compressed matrix, since the compressed version still
  // uses the legacy sparse format.

  // We allocate the output as a dense matrix
  Eigen::Matrix<typename Bembel::PDEproblemTraits<PDE>::Scalar, -1, -1> out =
      Eigen::Matrix<typename Bembel::PDEproblemTraits<PDE>::Scalar, -1,
                    1>::Zero(small_dofsize);

  Bembel::hmatrixsettings hmatset;
  Bembel::hmatrixfactory hmatfac;

  // 4 is a placeholder, it only affects min_bsize during matrix compression,
  // which is not the case here
  hmatset = Bembel::get_hmatrixsettings(4, &(disc.get_disc()));
  hmatset.eta = 0;
  hmatfac.disc = &(disc.get_disc());
  hmatfac.hmatset = &hmatset;
  hmatfac.assemfmats = 0;
  hmatfac.assemsfmats = 0;
  hmatfac.assemrkmats = 0;
  Bembel::init_Gauss_Square(&hmatfac.Q, Bembel::g_max + 1);
  Bembel::pdeproblem pde = disc.get_pde().get_pde();

  std::vector<Spl::Patch> geom = disc.get_plain_patchdata();

  pde.init_randwerte(&hmatfac.RW, &hmatfac.Q[disc.get_disc().g_far], geom,
                     1 << disc.get_M());

  Bembel::et_node* pE = disc.get_disc().mesh->E.patch[0];
  while (pE->son[0]) pE = pE->son[0];

  Eigen::SparseMatrix<double> projector = get_projector_as_eigen(disc);

  // Now we assemble the dense matrix from the blocks without assemblying the
  // large dense system

#pragma omp parallel for shared(out) default(shared)
  for (int zi = 0; zi < num_els; zi++) {
    std::vector<double> ar(bufferSize * a_bs * a_bs);
    for (int si = zi; si < num_els; si++) {
      get_stiff(&hmatfac, disc.get_disc().mesh->P, ar.data(), pE, disc.get_M(),
                zi, si);
#pragma omp critical
      {
        out += (projector * ptrmat_to_sparse(disc, ar.data(), zi, si) *
                projector.transpose())
                   .eval()
                   .diagonal();
      }
    }
  }

  pde.free_randwerte(&hmatfac.RW, 1 << disc.get_disc().mesh->M, geom.size());
  Bembel::free_Gauss_Square(&hmatfac.Q, Bembel::g_max + 1);
  return out;
}

}  // namespace Bembel

#endif
