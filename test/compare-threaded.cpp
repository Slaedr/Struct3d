/** Checks correctness of the solution computed by threaded async preconditioners.
 */

#undef NDEBUG
#define DEBUG 1

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
#include "case.hpp"

#define PETSCOPTION_STR_LEN 30

int main(int argc, char* argv[])
{
	if(argc < 3) {
		printf("Please specify a control file and a Petsc options file.\n");
		return 0;
	}

	char help[] = "Solves 3D Poisson equation by finite differences.\
				   Arguments: (1) Control file (2) Petsc options file\n \
           Command line options: -test_type <'compare_error','compare_its'>, -num_runs <int>";
	
	char * confile = argv[1];
	PetscErrorCode ierr = 0;

	ierr = PetscInitialize(&argc, &argv, NULL, help); CHKERRQ(ierr);
	MPI_Comm comm = PETSC_COMM_WORLD;
	const int size = get_mpi_size(comm);
	const int rank = get_mpi_rank(comm);
	if(rank == 0)
		printf("Number of MPI ranks = %d.\n", size);

#ifdef _OPENMP
	const int nthreads = omp_get_max_threads();
	if(rank == 0)
		printf("Max OMP threads = %d\n", nthreads);
#endif

	// Read control file
	
	FILE* conf = fopen(confile, "r");
	const CaseData cdata = readCtrl(conf);
	fclose(conf);

	PDEBase *pde = nullptr;
	if(cdata.pdetype == "poisson")
		pde = new Poisson();
	else {
		std::printf("PDE type not recognized!\n");
		std::abort();
	}

	if(rank == 0) {
		printf("Domain boundaries in each dimension:\n");
		for(int i = 0; i < NDIM; i++)
			printf("%f %f ", cdata.rmin[i], cdata.rmax[i]);
		printf("\n");
		printf("Number of runs: %d\n", cdata.nruns);
	}

	char testtype[PETSCOPTION_STR_LEN];
	PetscBool set = PETSC_FALSE;
	ierr = PetscOptionsGetString(NULL,NULL,"-test_type",testtype, PETSCOPTION_STR_LEN, &set);
	CHKERRQ(ierr);
	if(!set) {
		printf("Test type not set; testing convergence only.\n");
		strcpy(testtype,"convergence");
	}

	// Get error check tolerance
	PetscReal error_tol;
	set = PETSC_FALSE;
	ierr = PetscOptionsGetReal(NULL, NULL, "-error_tolerance_factor", &error_tol, &set);
	if(!set) {
		printf("Error tolerance factor not set; using the default 1e6.\n");
		error_tol = 1e6;
	}

	// PetscInt cmdnumruns;
	// ierr = PetscOptionsGetInt(NULL,NULL,"-num_runs",&cmdnumruns,&set); CHKERRQ(ierr);
	// if(set)
	// 	nruns = cmdnumruns;

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

	PetscInt refkspiters;
	ierr = KSPGetIterationNumber(kspref, &refkspiters);
	const PetscReal errnormref = compute_error(m,da,u,uexact);

	if(rank==0) {
		printf("Ref run: error = %.16f\n", errnormref);
	}
	
	KSPConvergedReason ref_ksp_reason;
	ierr = KSPGetConvergedReason(kspref, &ref_ksp_reason); CHKERRQ(ierr);
	assert(ref_ksp_reason > 0);

	KSPDestroy(&kspref);

	// run the solve to be tested as many times as requested
	
	int avgkspiters = 0;
	PetscReal errnorm = 0;
	
	// test with 4 threads - or not
#ifdef _OPENMP
	//omp_set_num_threads(4);
#endif
	
	for(int irun = 0; irun < cdata.nruns; irun++)
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
		
		// Create BLASTed data structure and setup the PC
		Blasted_data_list bctx = newBlastedDataList();
		ierr = setup_blasted_stack(ksp, &bctx); CHKERRQ(ierr);
		
		ierr = KSPSolve(ksp, b, u); CHKERRQ(ierr);

		// post-process
		int kspiters; PetscReal rnorm;
		KSPGetIterationNumber(ksp, &kspiters);
		avgkspiters += kspiters;

		if(rank == 0) {
			//printf(" Number of KSP iterations = %d\n", kspiters);
			KSPGetResidualNorm(ksp, &rnorm);
			printf(" KSP residual norm = %f, num iters = %d.\n", rnorm, kspiters);
		}
		
		errnorm += compute_error(m,da,u,uexact);
		if(rank == 0) {
			printf("Test run:\n");
			printf(" h and error: %f  %.16f\n", m.gh(), errnorm);
			printf(" log h and log error: %f  %f\n", log10(m.gh()), log10(errnorm));
		}
		
		// test
		KSPConvergedReason ksp_reason;
		ierr = KSPGetConvergedReason(ksp, &ksp_reason); CHKERRQ(ierr);
		assert(ksp_reason > 0);

		ierr = KSPDestroy(&ksp); CHKERRQ(ierr);

		// rudimentary test for time-totaller
		computeTotalTimes(&bctx);
		assert(bctx.factorwalltime > DBL_EPSILON);
		assert(bctx.applywalltime > DBL_EPSILON);
		// looks like the problem is too small for the unix clock() to record it
		assert(bctx.factorcputime >= 0);
		assert(bctx.applycputime >= 0);

		destroyBlastedDataList(&bctx);
	}

	if(rank == 0)
		printf("KSP Iters: Reference %d vs BLASTed %d.\n", refkspiters, avgkspiters/cdata.nruns);

	if(!strcmp(testtype, "compare_its"))
		assert(refkspiters >= avgkspiters/cdata.nruns);

	// the following test is probably not workable..
	printf("Difference in error norm = %.16f.\n", std::fabs(errnorm-errnormref));
	if(!strcmp(testtype,"compare_error"))
		assert(std::fabs(errnorm/cdata.nruns-errnormref) < error_tol*DBL_EPSILON);


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
