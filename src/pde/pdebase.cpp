/** \file
 * \brief Implementation of functionality that is common to most PDEs
 */

#include "pdebase.hpp"
#include "common_utils.hpp"

PDEBase::PDEBase(const std::array<BCType,6>& bc_types, const std::array<sreal,6>& bc_vals)
	: bctypes(bc_types), bvals(bc_vals)
{ }

int PDEBase::computeVectorPetsc(const CartMesh *const m, DM da,
                                const std::function<sreal(const sreal[NDIM])> func,
                                Vec f) const
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

	for(PetscInt k = start[2]; k < start[2]+lsize[2]; k++)
		for(PetscInt j = start[1]; j < start[1]+lsize[1]; j++)
			for(PetscInt i = start[0]; i < start[0]+lsize[0]; i++)
			{
				// const sreal crds[NDIM] = {m->gcoords(0,i+1), m->gcoords(1,j+1), m->gcoords(2,k+1)};
				// rhs[k][j][i] = func(crds);
				rhs[k][j][i] = rhs_kernel(m, func, i+1,j+1,k+1);
			}

	DMDAVecRestoreArray(da, f, (void*)&rhs);
	if(rank == 0)
		printf("ComputeRHS: Done\n");

	return ierr;
}

SVec PDEBase::computeVector(const CartMesh *const m,
                            const std::function<sreal(const sreal[NDIM])> func) const
{
	const int rank = get_mpi_rank(MPI_COMM_WORLD);

	SVec f(m);

	if(rank == 0)
		printf("ComputeRHS: Starting\n");

	// iterate over nodes
	for(PetscInt k = f.start; k < f.start + f.sz[2]; k++)
		for(PetscInt j = f.start; j < f.start + f.sz[1]; j++)
			for(PetscInt i = f.start; i < f.start + f.sz[0]; i++)
			{
				f.vals[m->localFlattenedIndexAll(k,j,i)] = rhs_kernel(m,func,i,j,k);
			}
	
	if(rank == 0)
		printf("ComputeRHS: Done\n");

	return f;
}

/// Set stiffness matrix corresponding to (real) mesh points
/** Inserts entries rowwise into the matrix.
 */
int PDEBase::computeLHSPetsc(const CartMesh *const m, DM da, Mat A) const
{
	PetscErrorCode ierr = 0;
	const int rank = get_mpi_rank(PETSC_COMM_WORLD);
	if(rank == 0)	
		printf("PDEBase: ComputeLHSPetsc: Setting values of the LHS matrix...\n");

	// get the starting global indices and sizes (in each direction) of the local mesh partition
	PetscInt start[NDIM], lsize[NDIM];
	ierr = DMDAGetCorners(da, &start[0], &start[1], &start[2], &lsize[0], &lsize[1], &lsize[2]);
	CHKERRQ(ierr);

	for(PetscInt k = start[2]; k < start[2]+lsize[2]; k++)
		for(PetscInt j = start[1]; j < start[1]+lsize[1]; j++)
			for(PetscInt i = start[0]; i < start[0]+lsize[0]; i++)
			{
				PetscReal values[NSTENCIL];
				MatStencil cindices[NSTENCIL];
				MatStencil rindices[1];
				const PetscInt n = NSTENCIL;
				const PetscInt mm = 1;

				rindices[0] = {k,j,i,0};

				cindices[0] = {k,j,i-1,0};
				cindices[1] = {k,j-1,i,0};
				cindices[2] = {k-1,j,i,0};
				cindices[3] = {k,j,i,0};
				cindices[4] = {k,j,i+1,0};
				cindices[5] = {k,j+1,i,0};
				cindices[6] = {k+1,j,i,0};

				lhsmat_kernel(m, i,j,k, 1, values[0], values[1], values[2], values[3], values[4],
				              values[5], values[6]);

				MatSetValuesStencil(A, mm, rindices, n, cindices, values, INSERT_VALUES);
			}

	if(rank == 0)
		printf("PDEBase: ComputeLHSPetsc: Done.\n");
	
	return ierr;
}

SMat PDEBase::computeLHS(const CartMesh *const m) const
{
	const int rank = get_mpi_rank(MPI_COMM_WORLD);

	SMat A(m);

	if(rank == 0)	
		printf("PDEBase: ComputeLHS: Setting values of the LHS matrix...\n");

	for(PetscInt k = A.start; k < A.start+A.sz[2]; k++)
		for(PetscInt j = A.start; j < A.start+A.sz[1]; j++)
			for(PetscInt i = A.start; i < A.start+A.sz[0]; i++)
			{
				const sint idx = m->localFlattenedIndexReal(k,j,i);

				lhsmat_kernel(m, i,j,k, A.nghost, A.vals[0][idx], A.vals[1][idx], A.vals[2][idx],
				              A.vals[3][idx], A.vals[4][idx], A.vals[5][idx], A.vals[6][idx]);
			}

	if(rank == 0)
		printf("PDEBase: ComputeLHS: Done.\n");
	
	return A;
}

