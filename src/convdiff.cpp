/** \file
 * \brief PETSc-based finite difference routines for convection-diffusion problem on a Cartesian grid
 * \author Aditya Kashi
 */

#include <cmath>
#include "convdiff.hpp"
#include "common_utils.hpp"

ConvDiff::ConvDiff(const sreal pn) : peclet{1.0}, b{1.0/sqrt(3),1.0/sqrt(3), 1.0/sqrt(3)}
{ }

/// Set RHS = 12*pi^2*sin(2pi*x)sin(2pi*y)sin(2pi*z) for u_exact = sin(2pi*x)sin(2pi*y)sin(2pi*z)
/** Note that the values are only set for interior points.
 * \param f is the rhs vector
 * \param uexact is the exact solution
 */
int ConvDiff::computeRHS(const CartMesh *const m, DM da, Vec f, Vec uexact) const
{
	PetscErrorCode ierr = 0;
	const int rank = get_mpi_rank(PETSC_COMM_WORLD);
	if(rank == 0)
		printf("Poisson: ComputeRHS: Starting\n");

	// get the starting global indices and sizes (in each direction) of the local mesh partition
	PetscInt start[NDIM], lsize[NDIM];
	ierr = DMDAGetCorners(da, &start[0], &start[1], &start[2], &lsize[0], &lsize[1], &lsize[2]);
	CHKERRQ(ierr);

	// get local data that can be accessed by global indices
	PetscReal *** rhs, *** uex;
	ierr = DMDAVecGetArray(da, f, (void*)&rhs); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da, uexact, (void*)&uex); CHKERRQ(ierr);

	// iterate over interior nodes
	for(PetscInt k = start[2]; k < start[2]+lsize[2]; k++)
		for(PetscInt j = start[1]; j < start[1]+lsize[1]; j++)
			for(PetscInt i = start[0]; i < start[0]+lsize[0]; i++)
			{
				rhs[k][j][i] = 12.0*PI*PI * std::sin(2*PI*m->gcoords(0,i))
					* std::sin(2*PI*m->gcoords(1,j))*std::sin(2*PI*m->gcoords(2,k));
				uex[k][j][i] = std::sin(2*PI*m->gcoords(0,i))
					* std::sin(2*PI*m->gcoords(1,j))*std::sin(2*PI*m->gcoords(2,k));
			}
	
	DMDAVecRestoreArray(da, f, (void*)&rhs);
	DMDAVecRestoreArray(da, uexact, (void*)&uex);
	if(rank == 0)
		printf("Poisson: ComputeRHS: Done\n");

	return ierr;
}

/// Set stiffness matrix corresponding to interior points
/** Inserts entries rowwise into the matrix.
 */
int ConvDiff::computeLHS(const CartMesh *const m, DM da, Mat A) const
{
	PetscErrorCode ierr = 0;	
	const int rank = get_mpi_rank(PETSC_COMM_WORLD);
	if(rank == 0)	
		printf("Poisson: ComputeLHS: Setting values of the LHS matrix...\n");

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

				PetscInt I = i+1, J = j+1, K = k+1;		// 1-offset indices for mesh coords access
				
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

				MatSetValuesStencil(A, mm, rindices, n, cindices, values, INSERT_VALUES);
				//if(rank == 0)
				//	printf("\tProcessed index %d, diag value = %f\n", rindices[0], values[3]);
			}

	if(rank == 0)
		printf("Poisson: ComputeLHS: Done.\n");
	
	return ierr;
}
