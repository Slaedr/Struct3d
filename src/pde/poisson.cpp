/** \file
 * \brief PETSc-based finite difference routines for Poisson Dirichlet problem on a Cartesian grid
 * \author Aditya Kashi
 *
 * Note that only zero Dirichlet BCs are currently supported.
 */

#include "poisson.hpp"
#include "common_utils.hpp"

#if 0
/// Gives the index of a point in the point grid collapsed to 1D
static inline PetscInt getFlattenedIndex(const CartMesh *const m, 
		const PetscInt i, const PetscInt j, const PetscInt k)
{
	return i + m->gnpoind(0)*j + m->gnpoind(0)*m->gnpoind(1)*k;
}

/// Gives the index of a point in the point grid collapsed to 1D
/** Assumes boundary points don't exist.
 * Returns -1 when passed a boundary point.
 * Make sure there's at least one interior point, or Bad Things (TM) may happen.
 */
static inline PetscInt getFlattenedInteriorIndex(const CartMesh *const m, 
		const PetscInt i, const PetscInt j, const PetscInt k)
{
	PetscInt retval = i-1 + (m->gnpoind(0)-2)*(j-1) + (m->gnpoind(0)-2)*(m->gnpoind(1)-2)*(k-1);
	if(i == 0 || i == m->gnpoind(0)-1 || j == 0 || j == m->gnpoind(1)-1 
			|| k == 0 || k == m->gnpoind(2)-1) 
	{
		//std::printf("getFlattenedInteriorIndex(): i, j, or k index corresponds to boundary node.
		//Flattened index = %d, returning -1\n", retval);
		return -1;
	}
	return retval;
}
#endif

std::array<std::function<sreal(const sreal[NDIM])>,2> Poisson::manufactured_solution() const
{
	std::array<std::function<sreal(const sreal[NDIM])>,2> soln;
	soln[0] = [](const sreal r[NDIM]) { return sin(2*PI*r[0])*sin(2*PI*r[1])*sin(2*PI*r[2]); };
	soln[1] = [](const sreal r[NDIM]) { return 12*PI*PI*sin(2*PI*r[0])*sin(2*PI*r[1])*sin(2*PI*r[2]); };
	return soln;
}

/// Set stiffness matrix corresponding to interior points
/** Inserts entries rowwise into the matrix.
 */
int Poisson::computeLHS(const CartMesh *const m, DM da, Mat A) const
{
	PetscErrorCode ierr = 0;	
	const int rank = get_mpi_rank(PETSC_COMM_WORLD);
	if(rank == 0)	
		printf("Poisson: ComputeLHS: Setting values of the LHS matrix...\n");

	// get the starting global indices and sizes (in each direction) of the local mesh partition
	PetscInt start[NDIM], lsize[NDIM];
	ierr = DMDAGetCorners(da, &start[0], &start[1], &start[2], &lsize[0], &lsize[1], &lsize[2]);
	CHKERRQ(ierr);

	// const sreal dx = m->gcoords(0,2)-m->gcoords(0,1);

	for(PetscInt k = start[2]; k < start[2]+lsize[2]; k++)
		for(PetscInt j = start[1]; j < start[1]+lsize[1]; j++)
			for(PetscInt i = start[0]; i < start[0]+lsize[0]; i++)
			{
				PetscReal values[NSTENCIL];
				MatStencil cindices[NSTENCIL];
				MatStencil rindices[1];
				PetscInt n = NSTENCIL;
				PetscInt mm = 1;

				rindices[0] = {k,j,i,0};

				cindices[0] = {k,j,i-1,0};
				cindices[1] = {k,j-1,i,0};
				cindices[2] = {k-1,j,i,0};
				cindices[3] = {k,j,i,0};
				cindices[4] = {k,j,i+1,0};
				cindices[5] = {k,j+1,i,0};
				cindices[6] = {k+1,j,i,0};

				const PetscInt I = i+1, J = j+1, K = k+1;  // 1-offset indices for mesh coords access
				
				values[0] = -1.0/( (m->gcoords(0,I)-m->gcoords(0,I-1)) 
						* 0.5*(m->gcoords(0,I+1)-m->gcoords(0,I-1)) );
				values[1] = -1.0/( (m->gcoords(1,J)-m->gcoords(1,J-1)) 
						* 0.5*(m->gcoords(1,J+1)-m->gcoords(1,J-1)) );
				values[2] = -1.0/( (m->gcoords(2,K)-m->gcoords(2,K-1)) 
						* 0.5*(m->gcoords(2,K+1)-m->gcoords(2,K-1)) );

				values[3] =  2.0/(m->gcoords(0,I+1)-m->gcoords(0,I-1))*
				  (1.0/(m->gcoords(0,I+1)-m->gcoords(0,I))+1.0/(m->gcoords(0,I)-m->gcoords(0,I-1)));
				values[3] += 2.0/(m->gcoords(1,J+1)-m->gcoords(1,J-1))*
				  (1.0/(m->gcoords(1,J+1)-m->gcoords(1,J))+1.0/(m->gcoords(1,J)-m->gcoords(1,J-1)));
				values[3] += 2.0/(m->gcoords(2,K+1)-m->gcoords(2,K-1))*
				  (1.0/(m->gcoords(2,K+1)-m->gcoords(2,K))+1.0/(m->gcoords(2,K)-m->gcoords(2,K-1)));

				values[4] = -1.0/( (m->gcoords(0,I+1)-m->gcoords(0,I)) 
						* 0.5*(m->gcoords(0,I+1)-m->gcoords(0,I-1)) );
				values[5] = -1.0/( (m->gcoords(1,J+1)-m->gcoords(1,J)) 
						* 0.5*(m->gcoords(1,J+1)-m->gcoords(1,J-1)) );
				values[6] = -1.0/( (m->gcoords(2,K+1)-m->gcoords(2,K)) 
						* 0.5*(m->gcoords(2,K+1)-m->gcoords(2,K-1)) );

				// // Uniform-grid central difference
				// values[0] = values[1] = values[2] = values[4] = values[5] = values[6] = -1.0/(dx*dx);
				// values[3] = 6.0/(dx*dx);

				MatSetValuesStencil(A, mm, rindices, n, cindices, values, INSERT_VALUES);
			}

	if(rank == 0) {
		printf("Poisson: ComputeLHS: Done.\n");
	}
	
	return ierr;
}
