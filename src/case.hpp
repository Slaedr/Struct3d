/** \file
 * \brief Setup data to run a case
 * \author Aditya
 */

#ifndef STRUCT3D_CASE_H
#define STRUCT3D_CASE_H

#include <array>
#include "struct3d_config.h"

/// Type of point distribution in a Cartesian grid
enum GridType { S3D_UNIFORM, S3D_CHEBYSHEV };

/// Data defining a case to be run
/** Faces of the rectangular domain are numbered in the order:
 * i-face at i=0, i-face at i=imax, j-face at j=0, j-face at j=jmax, k-face at k=0, k-face at k=kmax.
 */
struct CaseData {
	sint npdim[NDIM];                ///< Number of grid points in each dimension
	sreal rmin[NDIM];                ///< Starting coordinate of the domain in each direction
	sreal rmax[NDIM];                ///< Ending coordinate of the domain in each direction
	GridType gridtype;               ///< Type of Cartesian grid
	std::string pdetype;             ///< PDE to solve
	int nruns;                       ///< Number of times to repeat the experiment

	std::array<sreal,NDIM> vel;      ///< Advection velocity
	sreal diffcoeff;                 ///< Diffusion coefficient

	std::array<BCType,6> btypes;     ///< Type of BC at each of the 6 boundaries
	std::array<sreal,6> bvals;       ///< Boundary values at each of the 6 boundaries
};

CaseData readCtrl(FILE *const fp);

#endif
