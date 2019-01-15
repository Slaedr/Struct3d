/** \file
 * \brief Verification by measuring order of grid convergence of manufactured solutions
 */
#undef NDEBUG
#include <vector>
#include <cassert>

#include "pde/poisson.hpp"
#include "pde/convdiff.hpp"
#include "common_utils.hpp"
#include "case.hpp"

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
	int size, rank;
	int ierr = 0;

	MPI_Init(&argc, &argv);
	MPI_Comm comm = MPI_COMM_WORLD;
	MPI_Comm_size(comm,&size);
	MPI_Comm_rank(comm,&rank);
	if(rank == 0)
		printf("Number of MPI ranks = %d.\n", size);

	// Read control file

	FILE* conf = fopen(confile, "r");
	const CaseData cdata = readCtrl(conf);
	char tmp[100];
	int nrefinedirs;
	fscanf(conf, "%s", tmp); fscanf(conf, "%d", &nrefinedirs);
	fclose(conf);

	const int nmesh = cdata.nruns;
	assert(nmesh >= 2);
	assert(nrefinedirs == 1 || nrefinedirs == 2 || nrefinedirs == 3);

	PDEBase *pde = nullptr;
	if(cdata.pdetype == "poisson")
		pde = new Poisson();
	else if(cdata.pdetype == "convdiff")
		pde = new ConvDiff({1.0,0.0,0}, 0.1);
	else {
		std::printf("PDE type not recognized!\n");
		std::abort();
	}

	if(rank == 0) {
		printf("Domain boundaries in each dimension:\n");
		for(int i = 0; i < NDIM; i++)
			printf("%f %f \n", cdata.rmin[i], cdata.rmax[i]);
	}

	std::vector<sreal> h(nmesh);
	std::vector<sreal> errors(nmesh);

	for(int imesh = 0; imesh < nmesh; imesh++)
	{
		if(rank == 0)
			printf("\n");
		sint npoindim[NDIM];
		for(int j = 0; j < nrefinedirs; j++)
			npoindim[j] = cdata.npdim[j]*pow(2.0,imesh)/*(imesh+1)*/;
		for(int j = nrefinedirs; j < NDIM; j++)
			npoindim[j] = cdata.npdim[j];

		// grid structure - a copy of the mesh is stored by all processes
		CartMesh m;
		ierr = m.createMesh(npoindim);

		// generate grid
		if(cdata.gridtype == S3D_CHEBYSHEV) {
			m.generateMesh_ChebyshevDistribution(cdata.rmin,cdata.rmax);
			// printf("Chebyshev grid type not supported for discretization verification.\n");
			// delete pde;
			// MPI_Finalize();
			// exit(-1);
		}
		else
			m.generateMesh_UniformDistribution(cdata.rmin,cdata.rmax);

		// create vectors and matrix
		SVec u = m->createGridVec();
		SVec err = m->createGridVec();

		// compute values of LHS, RHS and exact soln
	
		const SVec b = pde->computeVector(&m, pde->manufactured_solution()[1]);
		const SVec uexact = pde->computeVector(&m, pde->manufactured_solution()[0]);
		const SMat A = pde->computeLHS(&m);

		//errors[imesh] = compute_error(m,u,uexact);

		//h[imesh] = 1.0/pow(npoindim[0]*npoindim[1]*npoindim[2], 1.0/3);
		//h[imesh] = m.gh();
		h[imesh] = 1.0/pow(2.0,imesh);
		//h[imesh] = 1.0/(imesh+1);

		if(rank==0) {
			printf("Mesh size = %f\n", h[imesh]);
			printf("Error = %f\n", errors[imesh]);
		}
	}

	delete pde;

	sreal slope = 0;
	for(int i = 1; i < nmesh; i++) {
		slope = (log10(errors[i])-log10(errors[i-1]))/(log10(h[i])-log10(h[i-1]));
		if(rank == 0)
			printf("Slope %d = %f\n", i, slope);
	}

	if(cdata.pdetype == "poisson")
		assert(slope >= 1.9 && slope <= 2.1);
	else
		assert(slope >= 0.9 && slope <= 1.9);

	MPI_finalize();
	return ierr;
}

