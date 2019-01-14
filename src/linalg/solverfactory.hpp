/** \file
 * \brief Factory for generating solver contexts from runtime options
 */

#ifndef STRUCT3D_SOLVERFACTORY_H
#define STRUCT3D_SOLVERFACTORY_H

#include "s3d_solverbase.hpp"

/// Creates a solver from options read from the PETSc options database. As such, PETSc is needed.
/** Options are very similar to PETSc ksp and pc options, just prefixed with 's3d_'.
 */
SolverBase *createSolver(const SMat& lhs_matrix);

#endif
