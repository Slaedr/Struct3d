#undef NDEBUG

#include <petscksp.h>

#include <sys/time.h>
#include <ctime>
#include <cfloat>
#include <cassert>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <blasted_petsc.h>

#include "pde/pdefactory.hpp"
#include "common_utils.hpp"
#include "case.hpp"

#define PETSCOPTION_STR_LEN 30

int main(int argc, char* argv[])
{
	if(argc < 3) {
		printf("Please specify a control file and a Petsc options file.\n");
		return 0;
	}

	char help[] = "Solves 3D Poisson equation by finite differences.\
				   Arguments: (1) Control file (2) Petsc options file\n\n";
	const char *confile = argv[1];
	PetscErrorCode ierr = 0;

	ierr = PetscInitialize(&argc, &argv, NULL, help); CHKERRQ(ierr);
	MPI_Comm comm = PETSC_COMM_WORLD;
	const int mpisize = get_mpi_size(comm);
	const int mpirank = get_mpi_rank(comm);
	if(mpirank == 0)
		printf("Number of MPI ranks = %d.\n", mpisize);

#ifdef _OPENMP
	const int nthreads = omp_get_max_threads();
	if(mpirank == 0)
		printf("Max OMP threads = %d\n", nthreads);
#endif

	// Read control file
	
	FILE* conf = fopen(confile, "r");
	const CaseData cdata = readCtrl(conf);
	fclose(conf);
	
	const PDEBase *const pde = construct_pde(cdata);

	if(mpirank == 0) {
		printf("Domain boundaries in each dimension:\n");
		for(int i = 0; i < NDIM; i++)
			printf("%f %f ", cdata.rmin[i], cdata.rmax[i]);
		printf("\n");
		printf("Number of runs: %d\n", cdata.nruns);
	}

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
	ierr = m.createMeshAndDMDA(comm, cdata.npdim, ndofpernode, stencil_width, bx, by, bz,
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
	
	ierr = pde->computeVectorPetsc(&m, da, pde->manufactured_solution()[1], b); CHKERRQ(ierr);
	ierr = pde->computeVectorPetsc(&m, da, pde->manufactured_solution()[0], uexact); CHKERRQ(ierr);
	ierr = pde->computeLHSPetsc(&m, da, A); CHKERRQ(ierr);

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
	PetscReal errnormref = compute_error(m,da,u,uexact);

	if(mpirank==0) {
		printf("Ref run: error = %.16f\n", errnormref);
	}

	KSPDestroy(&kspref);

	// run the solve to be tested as many times as requested
	
	int avgkspiters = 0;
	PetscReal errnorm = 0;
	for(int irun = 0; irun < cdata.nruns; irun++)
	{
		if(mpirank == 0)
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

		if(mpirank == 0) {
			KSPGetResidualNorm(ksp, &rnorm);
			printf(" KSP residual norm = %f\n", rnorm);
		}
		
		errnorm = compute_error(m,da,u,uexact);
		if(mpirank == 0) {
			printf("Test run:\n");
			printf(" h and error: %f  %.16f\n", m.gh(), errnorm);
			printf(" log h and log error: %f  %f\n", log10(m.gh()), log10(errnorm));
		}

		ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
		destroyBlastedDataList(&bctx);
	}

	if(mpirank == 0)
		printf("KSP Iters: Reference %d vs BLASTed %d.\n", refkspiters, avgkspiters/cdata.nruns);

	// the test
	assert(avgkspiters/cdata.nruns == refkspiters);
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

