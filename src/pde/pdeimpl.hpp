/** \file
 * \brief Implementation class for PDE-based problems
 */

#ifndef STRUCT3D_PDEIMPL_H
#define STRUCT3D_PDEIMPL_H

#include <cstdio>
#include <array>
#include <functional>
#include "cartmesh.hpp"
#include "pdebase.hpp"
#include "common_utils.hpp"

/// Class which implements common PDE discretization operations
template <typename ConcretePDE>
class PDEImpl : public PDEBase
{
public:
	PDEImpl(const std::array<BCType,6>& bc_types, const std::array<sreal,6>& bc_vals)
		: PDEBase(bc_types, bc_vals)
	{ }

	SMat computeLHS(const CartMesh *const m) const override
	{
		const int rank = get_mpi_rank(MPI_COMM_WORLD);

		SMat A(m);

		if(rank == 0)
			printf("PDEImpl: ComputeLHS: Setting values of the LHS matrix...\n");

		const sint lend[3] = { A.start+A.sz[0], A.start+A.sz[1], A.start+A.sz[2] };

#pragma omp parallel for collapse(2) default(shared)
		for(PetscInt k = A.start; k < lend[2]; k++)
			for(PetscInt j = A.start; j < lend[1]; j++) {
				//#pragma omp simd
				for(PetscInt i = A.start; i < lend[0]; i++)
				{
					const sint idx = m->localFlattenedIndexAll(k,j,i);

					sreal values[NSTENCIL];
					static_cast<const ConcretePDE*>(this)->lhsmat_kernel(m, i,j,k, A.nghost, values);

					for(int j = 0; j < NSTENCIL; j++)
						A.vals[j][idx] = values[j];
				}
			}

		if(rank == 0)
			printf("PDEImpl: ComputeLHS: Done.\n");

		return A;
	}

	/// Set stiffness matrix corresponding to (real) mesh points
	/** Inserts entries rowwise into the matrix.
	 */
	int computeLHSPetsc(const CartMesh *const m, DM da, Mat A) const override
	{
		PetscErrorCode ierr = 0;
		const int rank = get_mpi_rank(PETSC_COMM_WORLD);
		if(rank == 0)
			printf("PDEImpl: ComputeLHSPetsc: Setting values of the LHS matrix...\n");

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

					static_cast<const ConcretePDE*>(this)->lhsmat_kernel(m, i+1,j+1,k+1, 1, values);

#pragma omp critical
					{
						MatSetValuesStencil(A, mm, rindices, n, cindices, values, INSERT_VALUES);
					}
				}

		if(rank == 0)
			printf("PDEImpl: ComputeLHSPetsc: Done.\n");

		return ierr;
	}

	int computeVectorPetsc(const CartMesh *const m, DM da,
	                       const std::function<sreal(const sreal[NDIM])> func, Vec f) const override
	{
		PetscErrorCode ierr = 0;
		const int rank = get_mpi_rank(PETSC_COMM_WORLD);

		// get the starting global indices and sizes (in each direction) of the local mesh partition
		PetscInt start[NDIM], lsize[NDIM];
		ierr = DMDAGetCorners(da, &start[0], &start[1], &start[2], &lsize[0], &lsize[1], &lsize[2]);
		CHKERRQ(ierr);
		const PetscInt end[NDIM] {start[0]+lsize[0], start[1]+lsize[1], start[2]+lsize[2]};

		// get local data that can be accessed by global indices
		PetscReal *** rhs;
		ierr = DMDAVecGetArray(da, f, (void*)&rhs); CHKERRQ(ierr);

#pragma omp parallel for collapse(2)
		for(PetscInt k = start[2]; k < end[2]; k++)
			for(PetscInt j = start[1]; j < end[1]; j++)
				for(PetscInt i = start[0]; i < end[0]; i++)
				{
					rhs[k][j][i] =
						static_cast<const ConcretePDE*>(this)->rhs_kernel(m, func, i+1,j+1,k+1);
				}

		DMDAVecRestoreArray(da, f, (void*)&rhs);
		if(rank == 0)
			printf("ComputeRHS: Done\n");

		return ierr;
	}

	SVec computeVector(const CartMesh *const m,
	                   const std::function<sreal(const sreal[NDIM])> func) const override
	{
		const int rank = get_mpi_rank(MPI_COMM_WORLD);

		SVec f(m);
		const sint idxmax[3] = {f.start + f.sz[0],f.start + f.sz[1], f.start + f.sz[2]};

		// iterate over nodes
#pragma omp parallel for collapse(2)
		for(PetscInt k = f.start; k < idxmax[2]; k++)
			for(PetscInt j = f.start; j < idxmax[1]; j++)
				for(PetscInt i = f.start; i < idxmax[0]; i++)
				{
					f.vals[m->localFlattenedIndexAll(k,j,i)] =
						static_cast<const ConcretePDE*>(this)->rhs_kernel(m,func,i,j,k);
				}

		if(rank == 0)
			printf("ComputeRHS: Done\n");

		return f;
	}

	std::array<std::function<sreal(const sreal[NDIM])>,2> manufactured_solution() const override
	{
		return static_cast<const ConcretePDE*>(this)->manufactured_solution();
	}

	std::function<sreal(const sreal[NDIM])> test_rhs() const override {
		return static_cast<const ConcretePDE*>(this)->test_rhs();
	}
};

#endif
