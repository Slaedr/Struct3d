/** \file
 * \brief Some common utilities required by most solvers
 */

#ifndef STRUCT3D_COMMON_UTILS_H
#define STRUCT3D_COMMON_UTILS_H

#include <petscmat.h>
#include "cartmesh.hpp"

/// Computes L2 norm of a mesh function v
/** Assumes piecewise constant values in a dual cell around each node.
 * Note that the actual norm will only be returned by process 0; 
 * the other processes return only local norms.
 */
PetscReal computeNorm(const MPI_Comm comm, const CartMesh *const m, Vec v, DM da);

/// Computes L2 function norm of the error between u and uexact
PetscReal compute_error(const MPI_Comm comm, const CartMesh& m, const DM da,
                        const Vec u, const Vec uexact);

/// Get the rank of the current process in a communicator
int get_mpi_rank(MPI_Comm comm);

/// Get number of ranks in a communicator
int get_mpi_size(MPI_Comm comm);

#endif
