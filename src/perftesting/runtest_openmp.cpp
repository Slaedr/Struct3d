#undef NDEBUG

#include <cfloat>
#include <cassert>

#include <omp.h>

#include <petscsys.h>

#include "pde/poisson.hpp"
#include "pde/convdiff.hpp"
#include "pde/convdiff_circular.hpp"
#include "common_utils.hpp"
#include "case.hpp"
#include "scaling_openmp.hpp"

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

	const int nthreads = omp_get_max_threads();
	if(mpirank == 0)
		printf("Max OMP threads = %d\n", nthreads);

	// Read control file

	FILE* conf = fopen(confile, "r");
	if(!conf)
		printf("Could not open file!\n");
	fflush(stdout);
	const CaseData cdata = readCtrl(conf);
	fclose(conf);

	printf("PDE: %s\n", cdata.pdetype.c_str());
	PDEBase *pde = nullptr;
	if(cdata.pdetype == "poisson")
		pde = new Poisson();
	else if(cdata.pdetype == "convdiff")
		pde = new ConvDiff(cdata.vel, cdata.diffcoeff);
	else if(cdata.pdetype == "convdiff_circular")
		pde = new ConvDiffCirc(cdata.vel[0], cdata.diffcoeff);
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

	runtests_openmp(pde, m);
	
	delete pde;
	PetscFinalize();

	return ierr;
}
