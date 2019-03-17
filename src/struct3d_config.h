/** \file
 * \brief Some constants and types
 */

#ifndef STRUCT3D_CONFIG_H
#define STRUCT3D_CONFIG_H

/// Cache line size for alignment purposes
#define S3D_CACHE_LINE_LEN 64

/// Spatial dimension
#define NDIM 3

/// Number of entries in a row of the matrix for an interior point
#define NSTENCIL 7
/// Index of diagonal entry in the ordered list of stencil entries
#define STENCIL_DIAG 3

#define PI 3.141592653589793238

#include <petscsys.h>

/// Default scalar type
typedef PetscScalar sreal;
/// Default index type
typedef PetscInt sint;

/// Types of boundary conditions (BCs) available
enum BCType { S3D_EXTRAPOLATION, S3D_DIRICHLET };

#endif
