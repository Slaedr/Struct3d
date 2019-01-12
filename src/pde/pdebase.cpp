/** \file
 * \brief Implementation of functionality that is common to most PDEs
 */

#include "pdebase.hpp"
#include "common_utils.hpp"

int PDEBase::computeVectorPetsc(const CartMesh *const m, DM da,
                                const std::function<sreal(const sreal[NDIM])> func, Vec f) const
{
	PetscErrorCode ierr = 0;
	const int rank = get_mpi_rank(PETSC_COMM_WORLD);
	if(rank == 0)
		printf("ComputeRHS: Starting\n");

	// get the starting global indices and sizes (in each direction) of the local mesh partition
	PetscInt start[NDIM], lsize[NDIM];
	ierr = DMDAGetCorners(da, &start[0], &start[1], &start[2], &lsize[0], &lsize[1], &lsize[2]);
	CHKERRQ(ierr);

	// get local data that can be accessed by global indices
	PetscReal *** rhs;
	ierr = DMDAVecGetArray(da, f, (void*)&rhs); CHKERRQ(ierr);

	// iterate over interior nodes
	for(PetscInt k = start[2]; k < start[2]+lsize[2]; k++)
		for(PetscInt j = start[1]; j < start[1]+lsize[1]; j++)
			for(PetscInt i = start[0]; i < start[0]+lsize[0]; i++)
			{
				const sreal crds[NDIM] = {m->gcoords(0,i+1), m->gcoords(1,j+1), m->gcoords(2,k+1)};
				rhs[k][j][i] = func(crds);
			}
	
	DMDAVecRestoreArray(da, f, (void*)&rhs);
	if(rank == 0)
		printf("ComputeRHS: Done\n");

	return ierr;
}

SVec PDEBase::computeVector(const CartMesh *const m,
                            const std::function<sreal(const sreal[NDIM])> func) const
{
	const int rank = get_mpi_rank(PETSC_COMM_WORLD);

	SVec f(m);

	if(rank == 0)
		printf("ComputeRHS: Starting\n");

	// iterate over interior nodes
	for(PetscInt k = f.start; k < f.start + f.sz[2]; k++)
		for(PetscInt j = f.start; j < f.start + f.sz[1]; j++)
			for(PetscInt i = f.start; i < f.start + f.sz[0]; i++)
			{
				const sreal crds[NDIM] = {m->gcoords(0,i), m->gcoords(1,j), m->gcoords(2,k)};
				f.vals[m->localFlattenedIndexAll(k,j,i)] = func(crds);
			}
	
	if(rank == 0)
		printf("ComputeRHS: Done\n");

	return f;
}
