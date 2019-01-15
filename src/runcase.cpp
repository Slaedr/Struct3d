#undef NDEBUG

#include <cfloat>
#include <cassert>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <petscsys.h>

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
				   Arguments: (1) Control file (2) Petsc options file\n\n";
	const char *confile = argv[1];
	int ierr = 0;

	ierr = PetscInitialize(&argc, &argv, NULL, help); CHKERRQ(ierr);
	MPI_Comm comm = MPI_COMM_WORLD;
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

	printf("PDE: %s\n", cdata.pdetype.c_str());
	PDEBase *pde = nullptr;
	if(cdata.pdetype == "poisson")
		pde = new Poisson();
	else if(cdata.pdetype == "convdiff")
		pde = new ConvDiff(std::array<sreal,3>{1.0/sqrt(3.0),1.0/sqrt(3.0),1.0/sqrt(3.0)}, 0.01);
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

	// grid structure - a copy of the mesh is stored by all processes as the mesh structure is very small
	CartMesh m;
	ierr = m.createMesh(cdata.npdim);

	// generate grid
	if(cdata.gridtype == S3D_CHEBYSHEV)
		m.generateMesh_ChebyshevDistribution(cdata.rmin,cdata.rmax);
	else
		m.generateMesh_UniformDistribution(cdata.rmin,cdata.rmax);

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
		printf("Solver iters: %d.\n", avgkspiters/cdata.nruns);
	fflush(stdout);

	delete pde;
	PetscFinalize();

	return ierr;
}
