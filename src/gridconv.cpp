/** \file
 * \brief Verification by measuring order of grid convergence of manufactured solutions
 */
#include <petscksp.h>

#include <sys/time.h>
#include <ctime>
#include <cfloat>
#include <cassert>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <blasted_petsc.h>

#include "poisson.hpp"
#include "common_utils.hpp"

#define PETSCOPTION_STR_LEN 30

PetscReal compute_error(const MPI_Comm comm, const CartMesh& m, const DM da,
		const Vec u, const Vec uexact) {
	PetscReal errnorm;
	Vec err;
	VecDuplicate(u, &err);
	VecCopy(u,err);
	VecAXPY(err, -1.0, uexact);
	errnorm = computeNorm(comm, &m, err, da);
	VecDestroy(&err);
	return errnorm;
}

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
	int nruns;

	ierr = PetscInitialize(&argc, &argv, NULL, help); CHKERRQ(ierr);
	MPI_Comm comm = PETSC_COMM_WORLD;
	MPI_Comm_size(comm,&size);
	MPI_Comm_rank(comm,&rank);
	if(rank == 0)
		printf("Number of MPI ranks = %d.\n", size);

	// Read control file
	
	PetscInt npdim[NDIM];
	PetscReal rmax[NDIM], rmin[NDIM];
	char temp[50], gridtype[50], pdetype[25];
	FILE* conf = fopen(confile, "r");
	int fstatus = 1;
	fstatus = fscanf(conf, "%s", temp);
	fstatus = fscanf(conf, "%s", gridtype);
	fstatus = fscanf(conf, "%s", temp);
	if(!fstatus) {
		std::printf("! Error reading control file!\n");
		std::abort();
	}
	for(int i = 0; i < NDIM; i++)
		fstatus = fscanf(conf, "%d", &npdim[i]);
	fstatus = fscanf(conf, "%s", temp);
	for(int i = 0; i < NDIM; i++)
		fstatus = fscanf(conf, "%lf", &rmin[i]);
	fstatus = fscanf(conf, "%s", temp);
	for(int i = 0; i < NDIM; i++)
		fstatus = fscanf(conf, "%lf", &rmax[i]);
	fstatus = fscanf(conf, "%s", temp); 
	fstatus = fscanf(conf, "%d", &nruns);
	fstatus = fscanf(conf, "%s", temp);
	fstatus = fscanf(conf, "%s", pdetype);
	fclose(conf);
	
	if(!fstatus) {
		std::printf("! Error reading control file!\n");
		std::abort();
	}

	const std::string pdtype = pdetype;
	PDEBase *pde = nullptr;
	if(pdtype == "poisson")
		pde = new Poisson();
	else if(pdetype == "convdiff")
		pde = new ConvDiff(10.0);
	else {
		std::printf("PDE type not recognized!\n");
		std::abort();
	}

	if(rank == 0) {
		printf("Domain boundaries in each dimension:\n");
		for(int i = 0; i < NDIM; i++)
			printf("%f %f ", rmin[i], rmax[i]);
		printf("\n");
		printf("Number of runs: %d\n", nruns);
	}

	//----------------------------------------------------------------------------------

	// set up Petsc variables
	DM da;                        ///< Distributed array context for the cart grid
	PetscInt ndofpernode = 1;
	PetscInt stencil_width = 1;
	DMBoundaryType bx = DM_BOUNDARY_GHOSTED;
	DMBoundaryType by = DM_BOUNDARY_GHOSTED;
	DMBoundaryType bz = DM_BOUNDARY_GHOSTED;
	DMDAStencilType stencil_type = DMDA_STENCIL_STAR;

	// grid structure - a copy of the mesh is stored by all processes as the mesh structure is very small
	CartMesh m;
	ierr = m.createMeshAndDMDA(comm, npdim, ndofpernode, stencil_width, bx, by, bz, stencil_type, &da);
	CHKERRQ(ierr);

	// generate grid
	if(!strcmp(gridtype, "chebyshev"))
		m.generateMesh_ChebyshevDistribution(rmin,rmax);
	else
		m.generateMesh_UniformDistribution(rmin,rmax);

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

	KSP kspref; 

	// compute reference solution using a preconditioner from PETSc
	
	ierr = KSPCreate(comm, &kspref);
	KSPSetType(kspref, KSPRICHARDSON);
	KSPRichardsonSetScale(kspref, 1.0);
	KSPSetOptionsPrefix(kspref, "ref_");
	KSPSetFromOptions(kspref);
	
	ierr = KSPSetOperators(kspref, A, A); CHKERRQ(ierr);
	
	ierr = KSPSolve(kspref, b, u); CHKERRQ(ierr);

	KSPConvergedReason ref_ksp_reason;
	ierr = KSPGetConvergedReason(kspref, &ref_ksp_reason); CHKERRQ(ierr);
	assert(ref_ksp_reason > 0);

	PetscInt refkspiters;
	ierr = KSPGetIterationNumber(kspref, &refkspiters);
	PetscReal errnormref = compute_error(comm,m,da,u,uexact);

	if(rank==0) {
		printf("Ref run: error = %.16f\n", errnormref);
	}

	KSPDestroy(&kspref);

	// run the solve to be tested as many times as requested
	
	int avgkspiters = 0;
	PetscReal errnorm = 0;
	for(int irun = 0; irun < nruns; irun++)
	{
		if(rank == 0)
			printf("Run %d:\n", irun);
		KSP ksp;

		ierr = KSPCreate(comm, &ksp); CHKERRQ(ierr);
		KSPSetType(ksp, KSPRICHARDSON);
		KSPRichardsonSetScale(ksp, 1.0);
	
		// Options MUST be set before setting shell routines!
		ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);

		// Operators MUST be set before extracting sub KSPs!
		ierr = KSPSetOperators(ksp, A, A); CHKERRQ(ierr);
		
		Blasted_data_list bctx = newBlastedDataList();
		ierr = setup_blasted_stack(ksp, &bctx); CHKERRQ(ierr);
		
		ierr = KSPSolve(ksp, b, u); CHKERRQ(ierr);

		// post-process
		int kspiters; PetscReal rnorm;
		KSPGetIterationNumber(ksp, &kspiters);
		avgkspiters += kspiters;
		
		KSPConvergedReason ksp_reason;
		ierr = KSPGetConvergedReason(ksp, &ksp_reason); CHKERRQ(ierr);
		assert(ksp_reason > 0);

		if(rank == 0) {
			KSPGetResidualNorm(ksp, &rnorm);
			printf(" KSP residual norm = %f\n", rnorm);
		}
		
		errnorm = compute_error(comm,m,da,u,uexact);
		if(rank == 0) {
			printf("Test run:\n");
			printf(" h and error: %f  %.16f\n", m.gh(), errnorm);
			printf(" log h and log error: %f  %f\n", log10(m.gh()), log10(errnorm));
		}

		ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
		destroyBlastedDataList(&bctx);
	}

	if(rank == 0)
		printf("KSP Iters: Reference %d vs BLASTed %d.\n", refkspiters, avgkspiters/nruns);

	// the test
	assert(avgkspiters/nruns == refkspiters);
	assert(std::fabs(errnorm-errnormref) < 2.0*DBL_EPSILON);

	VecDestroy(&u);
	VecDestroy(&uexact);
	VecDestroy(&b);
	VecDestroy(&err);
	MatDestroy(&A);
	DMDestroy(&da);
	delete pde;
	PetscFinalize();

	return ierr;
}
		
// some unused snippets that might be useful at some point

/* For viewing the ILU factors computed by PETSc PCILU
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <../src/ksp/pc/impls/factor/factor.h>
#include <../src/ksp/pc/impls/factor/ilu/ilu.h>*/

	/*if(precch == 'i') {	
		// view factors
		PC_ILU* ilu = (PC_ILU*)pc->data;
		//PC_Factor* pcfact = (PC_Factor*)pc->data;
		//Mat fact = pcfact->fact;
		Mat fact = ((PC_Factor*)ilu)->fact;
		printf("ILU0 factored matrix:\n");

		Mat_SeqAIJ* fseq = (Mat_SeqAIJ*)fact->data;
		for(int i = 0; i < fact->rmap->n; i++) {
			printf("Row %d: ", i);
			for(int j = fseq->i[i]; j < fseq->i[i+1]; j++)
				printf("(%d: %f) ", fseq->j[j], fseq->a[j]);
			printf("\n");
		}
	}*/

