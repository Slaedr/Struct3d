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

struct CaseData {
	sint npdim[NDIM];                ///< Number of grid points in each dimension
	sreal rmin[NDIM];                ///< Starting coordinate of the domain in each direction
	sreal rmax[NDIM];                ///< Ending coordinate of the domain in each direction
	GridType gridtype;               ///< Type of Cartesian grid
	std::string pdetype;             ///< PDE to solve
	int nruns;                       ///< Number of times to repeat the experiment

	std::array<sreal,NDIM> vel;      ///< Advection velocity
	sreal diffcoeff;                 ///< Diffusion coefficient
};

CaseData readCtrl(FILE *const fp);

#endif
