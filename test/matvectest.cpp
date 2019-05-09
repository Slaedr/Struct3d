/** \file
 * \brief Test native matrix-vector product using manufactured solutions
 */

#undef NDEBUG
#include <cassert>
#include <algorithm>
#include "common_utils.hpp"
#include "pde/pdefactory.hpp"
#include "linalg/matvec.hpp"
#include "case.hpp"

int main(int argc, char *argv[])
{
	int ierr = 0;
	char * confile = argv[1];
	MPI_Init(&argc, &argv);

	const int mpisize = get_mpi_size(MPI_COMM_WORLD);
	assert(mpisize == 1);


	FILE* conf = fopen(confile, "r");
	const CaseData cdata = readCtrl(conf);
	const int nmesh = cdata.nruns;
	const int nrefinedirs = 3;

	const PDEBase *const pde = construct_pde(cdata);

	printf("Domain boundaries in each dimension:\n");
	for(int i = 0; i < NDIM; i++)
		printf("%f %f \n", cdata.rmin[i], cdata.rmax[i]);

	std::vector<sreal> h(nmesh);
	//std::vector<sreal> errors(nmesh);
	std::vector<sreal> ress(nmesh);

	for(int imesh = 0; imesh < nmesh; imesh++)
	{
		sint npoindim[NDIM];
		for(int j = 0; j < nrefinedirs; j++)
			npoindim[j] = cdata.npdim[j]*pow(2.0,imesh);
		for(int j = nrefinedirs; j < NDIM; j++)
			npoindim[j] = cdata.npdim[j];
		
		CartMesh m;
		ierr = m.createMesh(npoindim);
		if(ierr) throw std::runtime_error("Could not create mesh!");

		if(cdata.gridtype == S3D_CHEBYSHEV) {
			//m.generateMesh_ChebyshevDistribution(cdata.rmin,cdata.rmax);
			throw std::runtime_error("Chebyshev mesh not supported for grid convergence!");
		}
		else
			m.generateMesh_UniformDistribution(cdata.rmin,cdata.rmax);

		const SVec b = pde->computeVector(&m, pde->manufactured_solution()[1]);
		const SVec uexact = pde->computeVector(&m, pde->manufactured_solution()[0]);
		const SMat A = pde->computeLHS(&m);

		SVec res(&m);

		A.apply_res(b, uexact, res);

		SVec tempv(&m);
		A.apply(uexact,tempv);
		vecaxpy(-1.0, b, tempv);
		vecaxpy(1.0,res,tempv);
		const sreal diffnorm = norm_vector_l2(tempv);
		printf("  Difference between apply_res and apply = %g.\n", diffnorm);
		assert(diffnorm < 1e-14);

		const sreal defectnorm = norm_vector_l2(res)/sqrt(m.gnPoinTotal());
		printf("Defect = %f\n", defectnorm);

		h[imesh] = 1.0/pow(2.0,imesh);
		ress[imesh] = defectnorm;
		printf("Mesh size = %f\n", h[imesh]);
		//printf("Error norm = %f\n", errors[imesh]);
		printf("--Mesh %d--\n", imesh+1);
	}

	delete pde;

	sreal resslope = 0;
	for(int i = 1; i < nmesh; i++) {
		resslope = (log10(ress[i])-log10(ress[i-1]))/(log10(h[i])-log10(h[i-1]));
		printf("Slope %d = %f\n", i, resslope);
	}

	if(cdata.pdetype == "poisson")
		assert(resslope >= 1.9 && resslope <= 2.1);
	else
		assert(resslope >= 0.9 && resslope <= 1.1);

	MPI_Finalize();
	return ierr;
}
