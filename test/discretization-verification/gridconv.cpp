/** \file
 * \brief Verification by measuring order of grid convergence of manufactured solutions
 */
#undef NDEBUG
#include <petscksp.h>
#include <vector>
#include <cassert>

#include "poisson.hpp"
#include "convdiff.hpp"
#include "common_utils.hpp"
#include "case.hpp"

#define PETSCOPTION_STR_LEN 30

int main(int argc, char* argv[])
{
	using namespace std;

	if(argc < 3) {
		printf("Please specify a control file and a Petsc options file.\n");
		return 0;
	}

	char help[] = "Solves 3D Poisson equation by finite differences.\
				   Arguments: (1) Control file (2) Petsc options file\n\n";
	char * confile = argv[1];
	PetscMPIInt size, rank;
	PetscErrorCode ierr = 0;

	ierr = PetscInitialize(&argc, &argv, NULL, help); CHKERRQ(ierr);
	MPI_Comm comm = PETSC_COMM_WORLD;
	MPI_Comm_size(comm,&size);
	MPI_Comm_rank(comm,&rank);
	if(rank == 0)
		printf("Number of MPI ranks = %d.\n", size);

	// Read control file

	FILE* conf = fopen(confile, "r");
	// int fstatus = 1;
	// fstatus = fscanf(conf, "%s", temp);
	// fstatus = fscanf(conf, "%s", gridtype);
	// fstatus = fscanf(conf, "%s", temp);
	// if(!fstatus) {
	// 	std::printf("! Error reading control file!\n");
	// 	std::abort();
	// }
	// for(int i = 0; i < NDIM; i++)
	// 	fstatus = fscanf(conf, "%d", &npdim[i]);
	// fstatus = fscanf(conf, "%s", temp);
	// for(int i = 0; i < NDIM; i++)
	// 	fstatus = fscanf(conf, "%lf", &rmin[i]);
	// fstatus = fscanf(conf, "%s", temp);
	// for(int i = 0; i < NDIM; i++)
	// 	fstatus = fscanf(conf, "%lf", &rmax[i]);
	// fstatus = fscanf(conf, "%s", temp); 
	// fstatus = fscanf(conf, "%d", &nruns);
	// fstatus = fscanf(conf, "%s", temp);
	// fstatus = fscanf(conf, "%s", pdetype);
	const CaseData cdata = readCtrl(conf);
	fclose(conf);

	const int nmesh = cdata.nruns;
	if(nmesh < 2) {
		PetscFinalize();
		throw std::runtime_error("Need at least 2 meshes!");
	}

	PDEBase *pde = nullptr;
	if(cdata.pdetype == "poisson")
		pde = new Poisson();
	else if(cdata.pdetype == "convdiff")
		pde = new ConvDiff(10.0);
	else {
		std::printf("PDE type not recognized!\n");
		std::abort();
	}

	if(rank == 0) {
		printf("Domain boundaries in each dimension:\n");
		for(int i = 0; i < NDIM; i++)
			printf("%f %f \n", cdata.rmin[i], cdata.rmax[i]);
	}

	// set up Petsc variables
	const PetscInt ndofpernode = 1;
	const PetscInt stencil_width = 1;
	const DMBoundaryType bx = DM_BOUNDARY_GHOSTED;
	const DMBoundaryType by = DM_BOUNDARY_GHOSTED;
	const DMBoundaryType bz = DM_BOUNDARY_GHOSTED;
	const DMDAStencilType stencil_type = DMDA_STENCIL_STAR;

	std::vector<sreal> h(nmesh);
	std::vector<sreal> errors(nmesh);

	for(int imesh = 0; imesh < nmesh; imesh++)
	{
		printf("\n");
		DM da;                        //< Distributed array context for the cart grid
		// grid structure - a copy of the mesh is stored by all processes
		CartMesh m;
		sint npoindim[NDIM];
		for(int j = 0; j < NDIM; j++)
			npoindim[j] = cdata.npdim[j]*(imesh+1);
		ierr = m.createMeshAndDMDA(comm, npoindim, ndofpernode, stencil_width, bx, by, bz,
		                           stencil_type, &da);
		CHKERRQ(ierr);

		// generate grid
		if(cdata.gridtype == S3D_CHEBYSHEV)
			m.generateMesh_ChebyshevDistribution(cdata.rmin,cdata.rmax);
		else
			m.generateMesh_UniformDistribution(cdata.rmin,cdata.rmax);

		Vec u, uexact, b, err;
		Mat A;

		// create vectors and matrix according to the DMDA's structure
	
		ierr = DMCreateGlobalVector(da, &u); CHKERRQ(ierr);
		ierr = VecDuplicate(u, &b); CHKERRQ(ierr);
		VecDuplicate(u, &uexact);
		VecDuplicate(u, &err);
		VecSet(u, 0.0);
		ierr = DMCreateMatrix(da, &A); CHKERRQ(ierr);

		// compute values of LHS, RHS and exact soln
	
		ierr = pde->computeVector(&m, da, pde->manufactured_solution()[1], b); CHKERRQ(ierr);
		ierr = pde->computeVector(&m, da, pde->manufactured_solution()[0], uexact); CHKERRQ(ierr);
		ierr = pde->computeLHS(&m, da, A); CHKERRQ(ierr);

		// Assemble LHS

		ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
		ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

		KSP ksp; 
	
		ierr = KSPCreate(comm, &ksp);
		KSPSetType(ksp, KSPRICHARDSON);
		KSPRichardsonSetScale(ksp, 1.0);
		KSPSetFromOptions(ksp);
	
		ierr = KSPSetOperators(ksp, A, A); CHKERRQ(ierr);
	
		ierr = KSPSolve(ksp, b, u); CHKERRQ(ierr);

		KSPConvergedReason ksp_reason;
		ierr = KSPGetConvergedReason(ksp, &ksp_reason); CHKERRQ(ierr);
		assert(ksp_reason > 0);

		PetscInt kspiters;
		ierr = KSPGetIterationNumber(ksp, &kspiters);
		errors[imesh] = compute_error(comm,m,da,u,uexact);
		h[imesh] = 1.0/pow(npoindim[0]*npoindim[1]*npoindim[2], 1.0/3);

		if(rank==0) {
			printf("Ref run: error = %.16f\n", errors[imesh]);
		}

		KSPDestroy(&ksp);

		VecDestroy(&u);
		VecDestroy(&uexact);
		VecDestroy(&b);
		VecDestroy(&err);
		MatDestroy(&A);
		DMDestroy(&da);
	}

	delete pde;

	sreal slope = 0;
	for(int i = 1; i < nmesh; i++) {
		slope = (log10(errors[i])-log10(errors[i-1]))/(log10(h[i])-log10(h[i-1]));
		printf("Slope %d = %f\n", i, slope);
	}

	if(cdata.pdetype == "poisson")
		assert(slope >= 1.9 && slope <= 2.1);
	else
		assert(slope >= 0.9 && slope <= 2.1);

	PetscFinalize();
	return ierr;
}

