/** \file
 * \brief Verification by measuring order of grid convergence of manufactured solutions
 */
#undef NDEBUG
#include <vector>
#include <cassert>
#include <fenv.h>

#include "pde/pdefactory.hpp"
#include "common_utils.hpp"
#include "case.hpp"
#include "linalg/solverfactory.hpp"

int main(int argc, char* argv[])
{
#ifdef DEBUG
	feenableexcept(FE_DIVBYZERO | FE_INVALID);
#endif
	
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

	PetscInitialize(&argc, &argv, NULL, help);
	MPI_Comm comm = PETSC_COMM_WORLD;
	MPI_Comm_size(comm,&size);
	MPI_Comm_rank(comm,&rank);
	if(rank == 0)
		printf("Number of MPI ranks = %d.\n", size);

	// Read control file

	FILE* conf = fopen(confile, "r");
	const CaseData cdata = readCtrl(conf);
	char tmp[100];
	int nrefinedirs;
	int fstatus = fscanf(conf, "%s", tmp);
	fstatus = fscanf(conf, "%d", &nrefinedirs);
	if(fstatus < 1)
		if(rank == 0)
			printf("Could not read number of refinement directions!\n");
	fclose(conf);

	const int nmesh = cdata.nruns;
	assert(nmesh >= 2);
	assert(nrefinedirs == 1 || nrefinedirs == 2 || nrefinedirs == 3);
	printf("Refinement in %d directions.\n", nrefinedirs);

	printf("PDE: %s\n", cdata.pdetype.c_str());
	const PDEBase *const pde = construct_pde(cdata);

	if(rank == 0) {
		printf("Domain boundaries in each dimension:\n");
		for(int i = 0; i < NDIM; i++)
			printf("%f %f \n", cdata.rmin[i], cdata.rmax[i]);
	}

	std::vector<sreal> h(nmesh);
	std::vector<sreal> errors(nmesh);
	std::vector<sreal> ress(nmesh);

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
		}
		else
			m.generateMesh_UniformDistribution(cdata.rmin,cdata.rmax);

		// compute values of LHS, RHS and exact soln
	
		const SVec b = pde->computeVector(&m, pde->manufactured_solution()[1]);
		const SVec uexact = pde->computeVector(&m, pde->manufactured_solution()[0]);
		const SMat A = pde->computeLHS(&m);

		SVec res(&m);
		A.apply_res(b, uexact, res);
		ress[imesh] = norm_vector_l2(res)/sqrt(m.gnPoinTotal());

		SolverBase *const solver = createSolver(A);
		solver->updateOperator();
		
		SVec u(&m);
		const SolveInfo sinfo = solver->apply(b, u);
		assert(sinfo.converged);

		errors[imesh] = compute_error_L2(u,uexact);

		//h[imesh] = 1.0/pow(npoindim[0]*npoindim[1]*npoindim[2], 1.0/3);
		//h[imesh] = m.gh();
		h[imesh] = 1.0/pow(2.0,imesh);
		//h[imesh] = 1.0/(imesh+1);

		if(rank==0) {
			printf("Mesh size = %f\n", h[imesh]);
			printf("  Converged in %d itrs.\n", sinfo.iters);
			printf("  Defect = %f\n", ress[imesh]);
			printf("  Error = %f\n", errors[imesh]);
		}

		delete solver;
	}

	delete pde;

	sreal slope = 0, resslope = 0;
	for(int i = 1; i < nmesh; i++) {
		slope = (log10(errors[i])-log10(errors[i-1]))/(log10(h[i])-log10(h[i-1]));
		if(rank == 0)
			printf("Error slope %d = %f\n", i, slope);
		resslope = (log10(ress[i])-log10(ress[i-1]))/(log10(h[i])-log10(h[i-1]));
		if(rank == 0)
			printf("Defect slope %d = %f\n", i, resslope);
		fflush(stdout);
	}

	if(cdata.pdetype == "poisson")
		assert(resslope >= 1.9 && resslope <= 2.1);
	else {
		assert(resslope >= 0.9);
		if(cdata.gridtype == S3D_UNIFORM)
			assert(resslope <= 1.1);
	}

	if(cdata.pdetype == "poisson")
		assert(slope >= 1.9 && slope <= 2.1);
	else {
		assert(slope >= 0.9);
		if(cdata.gridtype == S3D_UNIFORM)
			assert(slope <= 1.1);
	}

	PetscFinalize();
	return ierr;
}

