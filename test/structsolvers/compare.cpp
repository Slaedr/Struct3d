#undef NDEBUG

#include <petscksp.h>

#include <sys/time.h>
#include <ctime>
#include <cfloat>
#include <cassert>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "pde/poisson.hpp"
#include "pde/convdiff.hpp"
#include "common_utils.hpp"
#include "case.hpp"
#include "linalg/solverfactory.hpp"

#define PETSCOPTION_STR_LEN 30

int main(int argc, char* argv[])
{
	if(argc < 3) {
		printf("Please specify a control file and a Petsc options file.\n");
		return 0;
	}

	char help[] = "Compares PETSc solvers with Struct3d native solvers.\
				   Arguments: (1) Control file (2) Petsc options file \
                   (3) -relative_iters_deviation <float> \n\n";
	const char *confile = argv[1];
	PetscErrorCode ierr = 0;

	ierr = PetscInitialize(&argc, &argv, NULL, help); CHKERRQ(ierr);
	MPI_Comm comm = PETSC_COMM_WORLD;
	const int mpisize = get_mpi_size(comm);
	const int mpirank = get_mpi_rank(comm);
	if(mpirank == 0)
		printf("Number of MPI ranks = %d.\n", mpisize);
	if(mpisize > 1) {
		printf("Multi-process runs are nor yet supported!\n");
		PetscFinalize();
		exit(-1);
	}

#ifdef _OPENMP
	const int nthreads = omp_get_max_threads();
	if(mpirank == 0)
		printf("Max OMP threads = %d\n", nthreads);
#endif

	// Read control file
	
	FILE* conf = fopen(confile, "r");
	const CaseData cdata = readCtrl(conf);
	fclose(conf);

	const sreal relitersdev = petscoptions_get_real("-relative_iters_deviation");

	printf("PDE: %s\n", cdata.pdetype.c_str());
	PDEBase *pde = nullptr;
	if(cdata.pdetype == "poisson")
		pde = new Poisson();
	else if(cdata.pdetype == "convdiff")
		pde = new ConvDiff(cdata.vel, cdata.diffcoeff);
	else {
		std::printf("PDE type not recognized!\n");
		std::abort();
	}

	if(mpirank == 0) {
		printf("Domain boundaries in each dimension:\n");
		for(int i = 0; i < NDIM; i++)
			printf("%f %f ", cdata.rmin[i], cdata.rmax[i]);
		printf("\n");
		printf("Number of runs: %d\n", cdata.nruns);
		fflush(stdout);
	}

	// set up Petsc variables
	DM da;                        ///< Distributed array context for the cart grid
	const PetscInt ndofpernode = 1;
	const PetscInt stencil_width = 1;
	const DMBoundaryType bx = DM_BOUNDARY_GHOSTED;
	const DMBoundaryType by = DM_BOUNDARY_GHOSTED;
	const DMBoundaryType bz = DM_BOUNDARY_GHOSTED;
	const DMDAStencilType stencil_type = DMDA_STENCIL_STAR;

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

	PetscInt refkspiters;
	PetscReal errnormref;
	{
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
		//KSPSetOptionsPrefix(kspref, "ref_");
		KSPSetFromOptions(kspref);
	
		ierr = KSPSetOperators(kspref, A, A); CHKERRQ(ierr);
	
		ierr = KSPSolve(kspref, b, u); CHKERRQ(ierr);

		KSPConvergedReason ref_ksp_reason;
		ierr = KSPGetConvergedReason(kspref, &ref_ksp_reason); CHKERRQ(ierr);
		assert(ref_ksp_reason > 0);

		ierr = KSPGetIterationNumber(kspref, &refkspiters);
		errnormref = compute_error(m,da,u,uexact);

		if(mpirank==0) {
			printf("Ref run: error = %.16f\n", errnormref);
		}

		KSPDestroy(&kspref);
		VecDestroy(&u);
		VecDestroy(&uexact);
		VecDestroy(&b);
		VecDestroy(&err);
		MatDestroy(&A);
		DMDestroy(&da);
	}

	// Prepare native solver
	
	const SVec b = pde->computeVector(&m, pde->manufactured_solution()[1]);
	const SVec uexact = pde->computeVector(&m, pde->manufactured_solution()[0]);
	const SMat A = pde->computeLHS(&m);

	// run the solve to be tested as many times as requested
	
	int avgkspiters = 0;
	PetscReal errnorm = 0;
	sreal wtime = 0;
	for(int irun = 0; irun < cdata.nruns; irun++)
	{
		if(mpirank == 0)
			printf("Test run %d:\n", irun);

		SolverBase *const solver = createSolver(A);
		solver->updateOperator();

		SVec u(&m);

		sreal starttime = MPI_Wtime();
		const SolveInfo slvinfo = solver->apply(b, u);
		sreal endtime = MPI_Wtime() - starttime;
		wtime += endtime;

		assert(slvinfo.converged);
		avgkspiters += slvinfo.iters;
		if(mpirank == 0) {
			printf(" KSP residual norm = %f\n", slvinfo.resnorm);
		}
		
		errnorm = compute_error_L2(u,uexact);
		if(mpirank == 0) {
			printf(" h and error: %f  %.16f\n", m.gh(), errnorm);
			printf(" log h and log error: %f  %f\n", log10(m.gh()), log10(errnorm));
		}

		delete solver;
	}

	wtime /= cdata.nruns;
	printf("Time taken by native solve = %f\n", wtime);

	if(mpirank == 0)
		printf("KSP Iters: Reference %d vs native %d.\n", refkspiters, avgkspiters/cdata.nruns);
	fflush(stdout);

	// the test
	const sreal measuredpcdev = fabs((sreal)avgkspiters/cdata.nruns - refkspiters)/refkspiters;
	printf("Relative deviation in number of iterations is %f.\n", measuredpcdev); fflush(stdout);

	assert(measuredpcdev < relitersdev);
	//assert(std::fabs(errnorm-errnormref) < 100.0*DBL_EPSILON);

	delete pde;
	PetscFinalize();

	return ierr;
}
