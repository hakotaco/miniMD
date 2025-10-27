/* ----------------------------------------------------------------------
   miniMD is a simple, parallel molecular dynamics (MD) code.   miniMD is
   an MD microapplication in the Mantevo project at Sandia National
   Laboratories ( http://www.mantevo.org ). The primary
   authors of miniMD are Steve Plimpton (sjplimp@sandia.gov) , Paul Crozier
   (pscrozi@sandia.gov) and Christian Trott (crtrott@sandia.gov).

   Copyright (2008) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This library is free software; you
   can redistribute it and/or modify it under the terms of the GNU Lesser
   General Public License as published by the Free Software Foundation;
   either version 3 of the License, or (at your option) any later
   version.

   This library is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with this software; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
   USA.  See also: http://www.gnu.org/licenses/lgpl.txt .

   For questions, contact Paul S. Crozier (pscrozi@sandia.gov) or
   Christian Trott (crtrott@sandia.gov).

   Please read the accompanying README and LICENSE files.
---------------------------------------------------------------------- */

#include "stdio.h"
#include "math.h"
#include "types.h"
#include "force_lj.h"
#include <Kokkos_SIMD.hpp>

#ifndef VECTORLENGTH
#define VECTORLENGTH 4
#endif

ForceLJ::ForceLJ(int ntypes_)
{
  cutforce = 0.0;
  use_oldcompute = 0;
  reneigh = 1;
  style = FORCELJ;
  ntypes = ntypes_;

  float_1d_view_type d_cut("ForceLJ::cutforcesq",ntypes*ntypes);
  float_1d_host_view_type h_cut = Kokkos::create_mirror_view(d_cut);
  cutforcesq = d_cut;

  float_1d_view_type d_epsilon("ForceLJ::epsilon",ntypes*ntypes);
  float_1d_host_view_type h_epsilon = Kokkos::create_mirror_view(d_epsilon);
  epsilon = d_epsilon;

  float_1d_view_type d_sigma6("ForceLJ::sigma6",ntypes*ntypes);
  float_1d_host_view_type h_sigma6 = Kokkos::create_mirror_view(d_sigma6);
  sigma6 = d_sigma6;

  float_1d_view_type d_sigma("ForceLJ::sigma",ntypes*ntypes);
  float_1d_host_view_type h_sigma = Kokkos::create_mirror_view(d_sigma);
  sigma = d_sigma;

  for(int i = 0; i<ntypes*ntypes; i++) {
    h_cut[i] = 0.0;
    h_epsilon[i] = 1.0;
    h_sigma6[i] = 1.0;
    h_sigma[i] = 1.0;
    if(i<MAX_STACK_TYPES*MAX_STACK_TYPES) {
      epsilon_s[i] = 1.0;
      sigma6_s[i] = 1.0;
    }
  }

  Kokkos::deep_copy(d_cut,h_cut);
  Kokkos::deep_copy(d_epsilon,h_epsilon);
  Kokkos::deep_copy(d_sigma6,h_sigma6);
  Kokkos::deep_copy(d_sigma,h_sigma);

  nthreads = Kokkos::HostSpace::execution_space::concurrency();
}

ForceLJ::~ForceLJ() {}

void ForceLJ::setup()
{
  float_1d_view_type d_cut("ForceLJ::cutforcesq",ntypes*ntypes);
  float_1d_host_view_type h_cut = Kokkos::create_mirror_view(d_cut);
  cutforcesq = d_cut;

  for(int i = 0; i<ntypes*ntypes; i++) {
    h_cut[i] = cutforce * cutforce;
    if(i<MAX_STACK_TYPES*MAX_STACK_TYPES)
      cutforcesq_s[i] = cutforce * cutforce;
  }

  Kokkos::deep_copy(d_cut,h_cut);
}


void ForceLJ::compute(Atom &atom, Neighbor &neighbor, Comm & /* comm */, int me)
{
  eng_vdwl = 0;
  virial = 0;

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABEL_ROCM) || defined(KOKKOS_ENABLE_SYCL)
  int host_device = 0;
#else
  int host_device = 1;
#endif

  nlocal = atom.nlocal;
  nall = atom.nlocal + atom.nghost;

  x = atom.x;
  f_a = atom.f;
  f = atom.f;
  type = atom.type;

  neighbors = neighbor.neighbors;
  numneigh = neighbor.numneigh;

  // clear force on own and ghost atoms

  Kokkos::deep_copy(f,0.0);

  /* switch to correct compute */

  if(evflag) {
    if(use_oldcompute && host_device)
      return compute_original<1>(atom, neighbor, me);

    if(neighbor.halfneigh) {
      if(neighbor.ghost_newton) {
        if(nthreads > 1 || !host_device)
          return compute_halfneigh_threaded<1, 1>(atom, neighbor, me);
        else
          return compute_halfneigh<1, 1>(atom, neighbor, me);
      } else {
        if(nthreads > 1 || !host_device)
          return compute_halfneigh_threaded<1, 0>(atom, neighbor, me);
        else
          return compute_halfneigh<1, 0>(atom, neighbor, me);
      }
    } else return compute_fullneigh<1>(atom, neighbor, me);
  } else {
    if(use_oldcompute)
      return compute_original<0>(atom, neighbor, me);

    if(neighbor.halfneigh) {
      if(neighbor.ghost_newton) {
        if(nthreads > 1 || !host_device)
          return compute_halfneigh_threaded<0, 1>(atom, neighbor, me);
        else
          return compute_halfneigh<0, 1>(atom, neighbor, me);
      } else {
        if(nthreads > 1 || !host_device)
          return compute_halfneigh_threaded<0, 0>(atom, neighbor, me);
        else
          return compute_halfneigh<0, 0>(atom, neighbor, me);
      }
    } else return compute_fullneigh<0>(atom, neighbor, me);

  }
}

//original version of force compute in miniMD
//  -MPI only
//  -not vectorizable
template<int EVFLAG>
void ForceLJ::compute_original(Atom & /* atom */, Neighbor & /* neighbor */, int /* me */)
{
  eng_vdwl = 0;
  virial = 0;

  // loop over all neighbors of my atoms
  // store force on both atoms i and j

  for(int i = 0; i < nlocal; i++) {
    const int jnum = numneigh[i];
    const MMD_float xtmp = x(i,0);
    const MMD_float ytmp = x(i,1);
    const MMD_float ztmp = x(i,2);
    const int type_i = type[i];

    for(int k = 0; k < jnum; k++) {
      const int j = neighbors(i,k);
      const MMD_float delx = xtmp - x(j,0);
      const MMD_float dely = ytmp - x(j,1);
      const MMD_float delz = ztmp - x(j,2);
      int type_j = type[j];
      const MMD_float rsq = delx * delx + dely * dely + delz * delz;

      const int type_ij = type_i*ntypes+type_j;

      if(rsq < cutforcesq(type_ij)) {
        const MMD_float sr2 = 1.0 / rsq;
        const MMD_float sr6 = sr2 * sr2 * sr2 * sigma6(type_ij);
        const MMD_float force = 48.0 * sr6 * (sr6 - 0.5) * sr2 * epsilon(type_ij);
        f(i,0) += delx * force;
        f(i,1) += dely * force;
        f(i,2) += delz * force;
        f(j,0) -= delx * force;
        f(j,1) -= dely * force;
        f(j,2) -= delz * force;

        if(EVFLAG) {
          eng_vdwl += (4.0 * sr6 * (sr6 - 1.0)) * epsilon(type_ij);
          virial += (delx * delx + dely * dely + delz * delz) * force;
        }
      }
    }
  }
}


//Not Thread-safe variant of force kernel using half-neighborlists
template<int EVFLAG, int GHOST_NEWTON>
void ForceLJ::compute_halfneigh(Atom & /* atom */, Neighbor & /* neighbor */, int /* me */)
{
  // loop over all neighbors of my atoms
  // store force on both atoms i and j
  MMD_float t_energy = 0;
  MMD_float t_virial = 0;

  for(int i = 0; i < nlocal; i++) {
    const int numneighs = numneigh[i];
    const MMD_float xtmp = x(i,0);
    const MMD_float ytmp = x(i,1);
    const MMD_float ztmp = x(i,2);
    const int type_i = type[i];

    MMD_float fix = 0.0;
    MMD_float fiy = 0.0;
    MMD_float fiz = 0.0;

#ifdef USE_SIMD
    #pragma simd reduction (+: fix,fiy,fiz)
#endif
    for(int k = 0; k < numneighs; k++) {
      const int j = neighbors(i,k);

      const MMD_float delx = xtmp - x(j,0);
      const MMD_float dely = ytmp - x(j,1);
      const MMD_float delz = ztmp - x(j,2);
      const int type_j = type[j];
      const MMD_float rsq = delx * delx + dely * dely + delz * delz;
      const int type_ij = type_i*ntypes+type_j;

      if(rsq < cutforcesq(type_ij)) {
        const MMD_float sr2 = 1.0 / rsq;
        const MMD_float sr6 = sr2 * sr2 * sr2 * sigma6(type_ij);
        const MMD_float force = 48.0 * sr6 * (sr6 - 0.5) * sr2 * epsilon(type_ij);

        fix += delx * force;
        fiy += dely * force;
        fiz += delz * force;

        if(GHOST_NEWTON || j < nlocal) {
          f(j,0) -= delx * force;
          f(j,1) -= dely * force;
          f(j,2) -= delz * force;
        }

        if(EVFLAG) {
          const MMD_float scale = (GHOST_NEWTON || j < nlocal) ? 1.0 : 0.5;
          t_energy += scale * (4.0 * sr6 * (sr6 - 1.0)) * epsilon(type_ij);
          t_virial += scale * (delx * delx + dely * dely + delz * delz) * force;
        }

      }
    }

    f(i,0) += fix;
    f(i,1) += fiy;
    f(i,2) += fiz;

  }

  eng_vdwl += t_energy;
  virial += t_virial;

}

//Thread-safe variant of force kernel using half-neighborlists with atomics
template<int EVFLAG, int GHOST_NEWTON>
void ForceLJ::compute_halfneigh_threaded(Atom & /* atom */, Neighbor & /* neighbor */, int /* me */)
{
  eng_virial_type t_eng_virial;

  // loop over all neighbors of my atoms
  // store force on both atoms i and j

  if(ntypes>MAX_STACK_TYPES) {
    if(EVFLAG)
      Kokkos::parallel_reduce(Kokkos::RangePolicy<TagComputeHalfNeighThread<1,GHOST_NEWTON,0> >(0,nlocal), *this , t_eng_virial);
    else
      Kokkos::parallel_for(Kokkos::RangePolicy<TagComputeHalfNeighThread<0,GHOST_NEWTON,0> >(0,nlocal), *this );
  } else {
    if(EVFLAG)
      Kokkos::parallel_reduce(Kokkos::RangePolicy<TagComputeHalfNeighThread<1,GHOST_NEWTON,1> >(0,nlocal), *this , t_eng_virial);
    else
      Kokkos::parallel_for(Kokkos::RangePolicy<TagComputeHalfNeighThread<0,GHOST_NEWTON,1> >(0,nlocal), *this );
  }
  eng_vdwl += t_eng_virial.eng;
  virial += t_eng_virial.virial;
}

//Thread-safe variant of force kernel using full-neighborlists
//   -trades more calculation for no atomics
//   -compared to halgneigh_threads:
//        2x reads, 0x writes (reads+writes the same as with half)
//        2x flops
template<int EVFLAG>
void ForceLJ::compute_fullneigh(Atom & /* atom */, Neighbor & /* neighbor */, int /* me */)
{
  eng_virial_type t_eng_virial;

  // loop over all neighbors of my atoms
  // store force on atom i

  if(ntypes>MAX_STACK_TYPES) {
    if(EVFLAG)
      Kokkos::parallel_reduce(Kokkos::RangePolicy<TagComputeFullNeigh<1,0> >(0,nlocal), *this , t_eng_virial);
    else
      Kokkos::parallel_for(Kokkos::RangePolicy<TagComputeFullNeigh<0,0> >(0,nlocal), *this );
  } else {
    if(EVFLAG)
      Kokkos::parallel_reduce(Kokkos::RangePolicy<TagComputeFullNeigh<1,1> >(0,nlocal), *this , t_eng_virial);
    else
      Kokkos::parallel_for(Kokkos::RangePolicy<TagComputeFullNeigh<0,1> >(0,nlocal), *this );
  }
  t_eng_virial.eng *= 4.0;
  t_eng_virial.virial *= 0.5;

  eng_vdwl += t_eng_virial.eng;
  virial += t_eng_virial.virial;
}

template<int EVFLAG, int GHOST_NEWTON, int STACK_PARAMS>
KOKKOS_INLINE_FUNCTION
void ForceLJ::operator() (TagComputeHalfNeighThread<EVFLAG,GHOST_NEWTON,STACK_PARAMS> , const int& i) const {
  eng_virial_type dummy;
  this->operator()(TagComputeHalfNeighThread<EVFLAG,GHOST_NEWTON,STACK_PARAMS> (), i, dummy);
}

template<int EVFLAG, int GHOST_NEWTON, int STACK_PARAMS>
KOKKOS_INLINE_FUNCTION
void ForceLJ::operator() (TagComputeHalfNeighThread<EVFLAG,GHOST_NEWTON,STACK_PARAMS> , const int& i, eng_virial_type& eng_virial) const {
/* Explicit SIMD version, slower, unfinished (may contain errors)

  using simd_t = Kokkos::Experimental::native_simd<MMD_float>;
  using simd_int_t = Kokkos::Experimental::native_simd<MMD_int>;
  using mask_t = typename simd_t::mask_type;
  const int VL = simd_t::size();

  const int numneighs = numneigh[i];

  const MMD_float xtmp = x(i,0);
  const MMD_float ytmp = x(i,1);
  const MMD_float ztmp = x(i,2);
  const int type_i = type[i];

  simd_t fix = 0.0;
  simd_t fiy = 0.0;
  simd_t fiz = 0.0;

  int k;
  for (k=0; k + VL <= numneighs; k += VL) {
    // gather neighbor indices j into simd lane vector
    simd_int_t j_vec;
    simd_t xj, yj, zj;
    simd_int_t type_j;
    for (int l=0; l < VL; l++) {
      const int j = neighbors(i, k + l);
      j_vec[l] = j;
      xj[l] = x(j,0);
      yj[l] = x(j,1);
      zj[l] = x(j,2);
      type_j[l] = type[j];
    }

    // compute deltas in vector lanes
    const simd_t delx = simd_t(xtmp) - xj;
    const simd_t dely = simd_t(ytmp) - yj;
    const simd_t delz = simd_t(ztmp) - zj;

    // rsq = delx^2 + dely^2 + delz^2
    const simd_t rsq = delx*delx + dely*dely + delz*delz;
    const simd_int_t type_ij = type_i*ntypes+type_j;

    //load the cutoff, sigma6, and epsilon values into vectors
    simd_t cutforce_sq, sigma6_v, eps_v;
    for (int l=0; l < VL; l++) {
      const int idx = type_ij[l];
      if constexpr (STACK_PARAMS) {
        cutforce_sq[l] = cutforcesq_s[idx];
        sigma6_v[l] = sigma6_s[idx];
        eps_v[l] = epsilon_s[idx];
      } else {
        cutforce_sq[l] = cutforcesq(idx);
        sigma6_v[l] = sigma6(idx);
        eps_v[l] = epsilon(idx);
      }
    }

    //if(rsq < (STACK_PARAMS?cutforcesq_s[type_ij]:cutforcesq(type_ij)))
    mask_t mask_cutoff = (rsq < cutforce_sq);
    if (Kokkos::Experimental::any_of(mask_cutoff)) {
      const simd_t sr2 = simd_t(1.0) / rsq;
      const simd_t sr6 = sr2 * sr2 * sr2 * sigma6_v;
      const simd_t force = simd_t(48.0) * sr6 * (sr6 - simd_t(0.5)) * sr2 * eps_v;

      Kokkos::Experimental::where(mask_cutoff, fix) = fix + delx * force;
      Kokkos::Experimental::where(mask_cutoff, fiy) = fiy + dely * force;
      Kokkos::Experimental::where(mask_cutoff, fiz) = fiz + delz * force;

      //if statemtns safely (scalar)
      for (int l = 0; l < VL; l++) {
        if (mask_cutoff[l]) {
          int jj = j_vec[l];

          if (GHOST_NEWTON || jj < nlocal) {
            f_a(jj,0) -= delx[l] * force[l];
            f_a(jj,1) -= dely[l] * force[l];
            f_a(jj,2) -= delz[l] * force[l];
          }

          if (EVFLAG) {
            MMD_float scale = (GHOST_NEWTON || jj < nlocal) ? 1.0 : 0.5;
            eng_virial.eng += scale * 4.0 * sr6[l] * (sr6[l] - 1.0) * epsilon[l];
            eng_virial.virial += scale * (delx[l]*delx[l] + dely[l]*dely[l] + delz[l]*delz[l]) * force[l];
          }
        }
      }
    }
  } //end for k loop

  MMD_float fix_sum = 0.0, fiy_sum = 0.0, fiz_sum = 0.0;
  for (int l=0; l < VL; l++) {
    fix_sum += fix[l];
    fiy_sum += fiy[l];
    fiz_sum += fiz[l];
  }
  f_a(i,0) += fix_sum;
  f_a(i,1) += fiy_sum;
  f_a(i,2) += fiz_sum;


  //scalar loop for tail
  MMD_float fixs=0, fiys=0, fizs=0;
  for (; k < numneighs; k++) {
    int j = neighbors(i,k);
    MMD_float delx = xtmp - x(j,0);
    MMD_float dely = ytmp - x(j,1);
    MMD_float delz = ztmp - x(j,2);
    int type_j = type[j];
    int type_ij = type_i*ntypes + type_j;

    MMD_float rsq = delx*delx + dely*dely + delz*delz;
    if(rsq < cutforcesq_s[type_ij]) {
      MMD_float sr2 = 1.0/rsq;
      MMD_float sr6 = sr2*sr2*sr2*(STACK_PARAMS?sigma6_s[type_ij]:sigma6(type_ij));
      MMD_float force = 48.0*sr6*(sr6-0.5)*sr2*(STACK_PARAMS?epsilon_s[type_ij]:epsilon(type_ij));

      fixs += delx*force;
      fiys += dely*force;
      fizs += delz*force;

      if(GHOST_NEWTON || j<nlocal){
        f_a(j,0) -= delx*force;
        f_a(j,1) -= dely*force;
        f_a(j,2) -= delz*force;
      }

      if(EVFLAG){
        MMD_float scale = (GHOST_NEWTON || j<nlocal) ? 1.0 : 0.5;
        eng_virial.eng += scale*4.0*sr6*(sr6-1.0)*(STACK_PARAMS?epsilon_s[type_ij]:epsilon(type_ij));
        eng_virial.virial += scale*(delx*delx + dely*dely + delz*delz)*force;
      }
    }
  }

  f_a(i,0) += fixs;
  f_a(i,1) += fiys;
  f_a(i,2) += fizs;
*/

  const int numneighs = numneigh[i];

  const MMD_float xtmp = x(i,0);
  const MMD_float ytmp = x(i,1);
  const MMD_float ztmp = x(i,2);
  const int type_i = type[i];

  MMD_float fix = 0.0;
  MMD_float fiy = 0.0;
  MMD_float fiz = 0.0;

  for(int k = 0; k < numneighs; k++) {
    const MMD_int j = neighbors(i,k);
    const MMD_float delx = xtmp - x(j,0);
    const MMD_float dely = ytmp - x(j,1);
    const MMD_float delz = ztmp - x(j,2);
    const int type_j = type[j];
    const MMD_float rsq = delx * delx + dely * dely + delz * delz;
    const int type_ij = type_i*ntypes+type_j;

    if(rsq < (STACK_PARAMS?cutforcesq_s[type_ij]:cutforcesq(type_ij))) {
      const MMD_float sr2 = 1.0 / rsq;
      const MMD_float sr6 = sr2 * sr2 * sr2 * (STACK_PARAMS?sigma6_s[type_ij]:sigma6(type_ij));
      const MMD_float force = 48.0 * sr6 * (sr6 - 0.5) * sr2 * (STACK_PARAMS?epsilon_s[type_ij]:epsilon(type_ij));

      fix += delx * force;
      fiy += dely * force;
      fiz += delz * force;

      if(GHOST_NEWTON || j < nlocal) {
        f_a(j,0) -= delx * force;
        f_a(j,1) -= dely * force;
        f_a(j,2) -= delz * force;
      }

      if(EVFLAG) {
        const MMD_float scale = (GHOST_NEWTON || j < nlocal) ? 1.0 : 0.5;
        eng_virial.eng += scale * 4.0 * sr6 * (sr6 - 1.0) * (STACK_PARAMS?epsilon_s[type_ij]:epsilon(type_ij));
        eng_virial.virial += scale * (delx * delx + dely * dely + delz * delz) * force;
      }
    }
  }

  f_a(i,0) += fix;
  f_a(i,1) += fiy;
  f_a(i,2) += fiz;
}

template<int EVFLAG, int STACK_PARAMS>
KOKKOS_INLINE_FUNCTION
void ForceLJ::operator() (TagComputeFullNeigh<EVFLAG,STACK_PARAMS> , const int& i) const {
  eng_virial_type dummy;
  this->operator()(TagComputeFullNeigh<EVFLAG,STACK_PARAMS> (), i, dummy);
}

template<int EVFLAG, int STACK_PARAMS>
KOKKOS_INLINE_FUNCTION
void ForceLJ::operator() (TagComputeFullNeigh<EVFLAG,STACK_PARAMS> , const int& i, eng_virial_type& eng_virial) const {

  const int numneighs = numneigh[i];

  const MMD_float xtmp = x(i,0);
  const MMD_float ytmp = x(i,1);
  const MMD_float ztmp = x(i,2);
  const int type_i = type[i];

  MMD_float fix = 0;
  MMD_float fiy = 0;
  MMD_float fiz = 0;

  //pragma simd forces vectorization (ignoring the performance objections of the compiler)
  //also give hint to use certain vectorlength for MIC, Sandy Bridge and WESTMERE this should be be 8 here
  //give hint to compiler that fix, fiy and fiz are used for reduction only

#ifdef USE_SIMD
  #pragma simd reduction (+: fix,fiy,fiz,eng_virial)
#endif
  for(int k = 0; k < numneighs; k++) {
    const MMD_int j = neighbors(i,k);
    const MMD_float delx = xtmp - x(j,0);
    const MMD_float dely = ytmp - x(j,1);
    const MMD_float delz = ztmp - x(j,2);
    const int type_j = type[j];
    const MMD_float rsq = delx * delx + dely * dely + delz * delz;
    int type_ij = type_i*ntypes+type_j;

    if(rsq < (STACK_PARAMS?cutforcesq_s[type_ij]:cutforcesq(type_ij))) {
      const MMD_float sr2 = 1.0 / rsq;
      const MMD_float sr6 = sr2 * sr2 * sr2 * (STACK_PARAMS?sigma6_s[type_ij]:sigma6(type_ij));
      const MMD_float force = 48.0 * sr6 * (sr6 - 0.5) * sr2 * (STACK_PARAMS?epsilon_s[type_ij]:epsilon(type_ij));
      fix += delx * force;
      fiy += dely * force;
      fiz += delz * force;

      if(EVFLAG) {
        eng_virial.eng += sr6 * (sr6 - 1.0) * (STACK_PARAMS?epsilon_s[type_ij]:epsilon(type_ij));
        eng_virial.virial += (delx * delx + dely * dely + delz * delz) * force;
      }
    }

  }

  f(i,0) += fix;
  f(i,1) += fiy;
  f(i,2) += fiz;
}

