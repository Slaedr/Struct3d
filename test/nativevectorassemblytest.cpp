/** \file
 * \brief Tests native vector assembly
 */

#undef NDEBUG
#include <cassert>
#include <limits>
#include "pde/poisson.hpp"
#include "case.hpp"

int main(int argc, char *argv[])
{
	int ierr = 0;
	char help[] = "Solves 3D Poisson equation by finite differences.\
				   Arguments: (1) Control file (2) Petsc options file\n\n";
	const char *confile = argv[1];

	ierr = PetscInitialize(&argc, &argv, NULL, help); CHKERRQ(ierr);

	FILE *fp = fopen(confile, "r");
	CaseData cdata = readCtrl(fp);
	fclose(fp);

	CartMesh m1;
	ierr = m1.createMesh(cdata.npdim);
	if(ierr) throw std::runtime_error("Could not create mesh!");

	// set up Petsc variables
	const PetscInt ndofpernode = 1;
	const PetscInt stencil_width = 1;
	const DMBoundaryType bx = DM_BOUNDARY_GHOSTED;
	const DMBoundaryType by = DM_BOUNDARY_GHOSTED;
	const DMBoundaryType bz = DM_BOUNDARY_GHOSTED;
	const DMDAStencilType stencil_type = DMDA_STENCIL_STAR;

	CartMesh m2;
	DM da;
	ierr = m2.createMeshAndDMDA(PETSC_COMM_WORLD, cdata.npdim, ndofpernode, stencil_width, bx, by, bz,
	                            stencil_type, &da);
	CHKERRQ(ierr);

	// generate grid
	if(cdata.gridtype == S3D_CHEBYSHEV) {
		m1.generateMesh_ChebyshevDistribution(cdata.rmin,cdata.rmax);
		m2.generateMesh_ChebyshevDistribution(cdata.rmin,cdata.rmax);
	}
	else {
		m1.generateMesh_UniformDistribution(cdata.rmin,cdata.rmax);
		m2.generateMesh_UniformDistribution(cdata.rmin,cdata.rmax);
	}

	Poisson pde;

	const SVec u1 = pde.computeVector(&m1, pde.manufactured_solution()[0]);

	Vec u2;
	ierr = DMCreateGlobalVector(da, &u2); CHKERRQ(ierr);
	ierr = pde.computeVectorPetsc(&m2, da, pde.manufactured_solution()[0], u2); CHKERRQ(ierr);

	// Compare

	// get the starting global indices and sizes (in each direction) of the local mesh partition
	PetscInt start[NDIM], lsize[NDIM];
	ierr = DMDAGetCorners(da, &start[0], &start[1], &start[2], &lsize[0], &lsize[1], &lsize[2]);
	CHKERRQ(ierr);
	printf("Petsc Vec starting indices and sizes: %d %d %d, %d %d %d\n", start[0], start[1], start[2],
	       lsize[0], lsize[1], lsize[2]);
	printf("Native Vec starting indices and sizes: %d, %d %d %d\n", u1.start, u1.sz[0], u1.sz[1],
	       u1.sz[2]);
	assert(start[0] == 0 && start[1] == 0 && start[2] == 0);  // for the test below to work

	// get local data that can be accessed by global indices
	PetscReal ***u2vec;
	ierr = DMDAVecGetArray(da, u2, (void*)&u2vec); CHKERRQ(ierr);

	// iterate over interior nodes
	for(PetscInt k = start[2]; k < start[2]+lsize[2]; k++)
		for(PetscInt j = start[1]; j < start[1]+lsize[1]; j++)
			for(PetscInt i = start[0]; i < start[0]+lsize[0]; i++)
			{
				assert(u2vec[k][j][i] == u1.vals[m1.localFlattenedIndexAll(k+1,j+1,i+1)]);
			}

	// iterate over ghost points, see if they're zero
	for(sint j = 0; j < u1.sz[1]+2; j++) {
		for(sint i = 0; i < u1.sz[0]+2; i++)
			assert(u1.vals[m1.localFlattenedIndexAll(0,j,i)] == 0);
	}
	for(sint j = 0; j < u1.sz[1]+2; j++) {
		for(sint i = 0; i < u1.sz[0]+2; i++)
			assert(u1.vals[m1.localFlattenedIndexAll(u1.start+u1.sz[2],j,i)] == 0);
	}

	for(sint k = 0; k < u1.sz[2]+2; k++) {
		for(sint i = 0; i < u1.sz[0]+2; i++)
			assert(u1.vals[m1.localFlattenedIndexAll(k,0,i)] == 0);
	}
	for(sint k = 0; k < u1.sz[2]+2; k++) {
		for(sint i = 0; i < u1.sz[0]+2; i++)
			assert(u1.vals[m1.localFlattenedIndexAll(k,u1.start+u1.sz[1],i)] == 0);
	}

	for(sint k = 0; k < u1.sz[2]+2; k++) {
		for(sint j = 0; j < u1.sz[1]+2; j++)
			assert(u1.vals[m1.localFlattenedIndexAll(k,j,0)] == 0);
	}
	for(sint k = 0; k < u1.sz[2]+2; k++) {
		for(sint j = 0; j < u1.sz[1]+2; j++)
			assert(u1.vals[m1.localFlattenedIndexAll(k,j,u1.start+u1.sz[0])] == 0);
	}

	// Matrix test

	const sreal dx = m1.gcoords(0,2) - m1.gcoords(0,1);
	printf("Diag val = %f, offdiag val = %f\n", 6.0/(dx*dx), -1.0/(dx*dx));
	SMat A1 = pde.computeLHS(&m1);
	for(sint k = u1.start; k < u1.start+u1.sz[2]; k++)
		for(sint j = u1.start; j < u1.start+u1.sz[1]; j++)
			for(sint i = u1.start; i < u1.start+u1.sz[0]; i++)
			{
				const sint idx = m1.localFlattenedIndexReal(k-1,j-1,i-1);

				assert(std::abs(A1.vals[3][idx] - 6.0/(dx*dx))
				       <= 100*std::numeric_limits<sreal>::epsilon());

				for(int is = 0; is < NSTENCIL; is++)
					if(is != 3)
						assert(std::abs(A1.vals[is][idx] + 1.0/(dx*dx))
						       <= 100*std::numeric_limits<sreal>::epsilon());
			}
	
	DMDAVecRestoreArray(da, u2, (void*)&u2vec);

	VecDestroy(&u2);
	DMDestroy(&da);
	PetscFinalize();
	return ierr;
}
