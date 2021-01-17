/** \file
 * \brief Time native matrix-vector product using manufactured solutions
 */

#include <cassert>
#include <algorithm>
#include <chrono>
#include "common_utils.hpp"
#include "pde/pdefactory.hpp"
#include "linalg/matvec.hpp"
#include "case.hpp"

int main(int argc, char *argv[])
{
	int ierr = 0;
	char * confile = argv[1];
	MPI_Init(&argc, &argv);

	FILE* conf = fopen(confile, "r");
	const CaseData cdata = readCtrl(conf);
	const int nruns = cdata.nruns;

	const PDEBase *const pde = construct_pde(cdata);

	printf("Domain boundaries in each dimension:\n");
	for(int i = 0; i < NDIM; i++)
		printf("%f %f \n", cdata.rmin[i], cdata.rmax[i]);

	CartMesh m;
	ierr = m.createMesh(cdata.npdim);
	if(ierr)
		throw std::runtime_error("Could not create mesh!");

	if(cdata.gridtype == S3D_CHEBYSHEV)
		m.generateMesh_ChebyshevDistribution(cdata.rmin,cdata.rmax);
	else
		m.generateMesh_UniformDistribution(cdata.rmin,cdata.rmax);

	const SVec uexact = pde->computeVector(&m, pde->manufactured_solution()[0]);
	auto startlhs = std::chrono::steady_clock::now();
	const SMat A = pde->computeLHS(&m);
	auto endlhs = std::chrono::steady_clock::now();
	std::chrono::duration<double> difflhs = endlhs-startlhs;
	printf(" Time taken by LHS assembly = %f.\n", difflhs.count());

	double time_taken{};

	SVec tempv(&m);

	for(int irun = 0; irun < nruns; irun++) {
		auto start = std::chrono::steady_clock::now();

		A.apply(uexact,tempv);

		auto end = std::chrono::steady_clock::now();
		std::chrono::duration<double> diff = end-start;
		if(irun > 0)
			time_taken += diff.count();
	}

	printf(" Time taken by matvec = %f.\n", time_taken);
	fflush(stdout);

	delete pde;

	MPI_Finalize();
	return ierr;
}
