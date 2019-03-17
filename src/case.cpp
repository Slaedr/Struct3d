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

	if(cas.pdetype == "convdiff") {
		fstatus = fscanf(conf, "%s", temp);
		for(int i = 0; i < NDIM; i++)
			fstatus = fscanf(conf, "%lf", &cas.vel[i]);
		fstatus = fscanf(conf, "%s", temp);
		fstatus = fscanf(conf, "%lf", &cas.diffcoeff);
	}
	else if(cas.pdetype == "convdiff_circular") {
		fstatus = fscanf(conf, "%s", temp);
		fstatus = fscanf(conf, "%lf", &cas.vel[0]);
		fstatus = fscanf(conf, "%s", temp);
		fstatus = fscanf(conf, "%lf", &cas.diffcoeff);
	}

	if(!fstatus) {
		std::printf("! Error reading control file!\n");
		std::abort();
	}

	// BC types
	fstatus = fscanf(conf, "%s", temp);
	for(int j = 0; j < 6; j++) {
		char bct[3];
		fstatus = fscanf(conf, "%s", bct);
		if(!strcmp(bct,"D"))
			cas.btypes[j] = S3D_DIRICHLET;
		else if(!strcmp(bct,"E"))
			cas.btypes[j] = S3D_EXTRAPOLATION;
		else
			printf("Invalid BC type! Must be D or E");
	}

	// BC values
	fstatus = fscanf(conf, "%s", temp);
	for(int j = 0; j < 6; j++)
		fstatus = fscanf(conf, "%lf", &cas.bvals[j]);

	if(!fstatus) {
		std::printf("! Error reading control file!\n");
		std::abort();
	}

	return cas;
}
