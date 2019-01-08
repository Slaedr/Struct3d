/** \file
 * \brief Implementation of rudimentary control file reader
 */

#include <cstdio>
#include "case.hpp"

CaseData readCtrl(FILE *const conf)
{
	CaseData cas;

	char temp[50], grid_type[50], pde_type[25];
	int fstatus = 1;
	fstatus = fscanf(conf, "%s", temp);
	fstatus = fscanf(conf, "%s", grid_type);
	if(!strcmp(grid_type, "uniform"))
		cas.gridtype = S3D_UNIFORM;
	else if(!strcmp(grid_type, "chebyshev"))
		cas.gridtype = S3D_CHEBYSHEV;
	else {
		printf("Invalid grid type!");
		std::abort();
	}

	fstatus = fscanf(conf, "%s", temp);
	if(!fstatus) {
		std::printf("! Error reading control file!\n");
		std::abort();
	}
	for(int i = 0; i < NDIM; i++)
		fstatus = fscanf(conf, "%d", &cas.npdim[i]);
	fstatus = fscanf(conf, "%s", temp);
	for(int i = 0; i < NDIM; i++)
		fstatus = fscanf(conf, "%lf", &cas.rmin[i]);
	fstatus = fscanf(conf, "%s", temp);
	for(int i = 0; i < NDIM; i++)
		fstatus = fscanf(conf, "%lf", &cas.rmax[i]);
	fstatus = fscanf(conf, "%s", temp);
	fstatus = fscanf(conf, "%d", &cas.nruns);
	fstatus = fscanf(conf, "%s", temp);
	fstatus = fscanf(conf, "%s", pde_type);
	cas.pdetype = pde_type;

	if(!fstatus) {
		std::printf("! Error reading control file!\n");
		std::abort();
	}

	return cas;
}
