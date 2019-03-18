/** \file
 * \brief PETSc-based finite difference routines for Poisson Dirichlet problem on a Cartesian grid
 * \author Aditya Kashi
 *
 * Note that only zero Dirichlet BCs are currently supported.
 */

#include "poisson.hpp"
#include "common_utils.hpp"

Poisson::Poisson(const std::array<BCType,6>& bc_types, const std::array<sreal,6>& bc_vals)
	: PDEBase(bc_types, bc_vals)
{ }

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
// int Poisson::computeLHSPetsc(const CartMesh *const m, DM da, Mat A) const
// {
// 	PetscErrorCode ierr = 0;	
// 	const int rank = get_mpi_rank(PETSC_COMM_WORLD);
// 	if(rank == 0)	
// 		printf("Poisson: ComputeLHS: Setting values of the LHS matrix...\n");

// 	// get the starting global indices and sizes (in each direction) of the local mesh partition
// 	PetscInt start[NDIM], lsize[NDIM];
// 	ierr = DMDAGetCorners(da, &start[0], &start[1], &start[2], &lsize[0], &lsize[1], &lsize[2]);
// 	CHKERRQ(ierr);

// 	// const sreal dx = m->gcoords(0,2)-m->gcoords(0,1);

// 	for(PetscInt k = start[2]; k < start[2]+lsize[2]; k++)
// 		for(PetscInt j = start[1]; j < start[1]+lsize[1]; j++)
// 			for(PetscInt i = start[0]; i < start[0]+lsize[0]; i++)
// 			{
// 				PetscReal values[NSTENCIL];
// 				MatStencil cindices[NSTENCIL];
// 				MatStencil rindices[1];
// 				PetscInt n = NSTENCIL;
// 				PetscInt mm = 1;

// 				rindices[0] = {k,j,i,0};

// 				cindices[0] = {k,j,i-1,0};
// 				cindices[1] = {k,j-1,i,0};
// 				cindices[2] = {k-1,j,i,0};
// 				cindices[3] = {k,j,i,0};
// 				cindices[4] = {k,j,i+1,0};
// 				cindices[5] = {k,j+1,i,0};
// 				cindices[6] = {k+1,j,i,0};

// 				const PetscInt I = i+1, J = j+1, K = k+1;  // 1-offset indices for mesh coords access
				
// 				values[0] = -1.0/( (m->gcoords(0,I)-m->gcoords(0,I-1)) 
// 						* 0.5*(m->gcoords(0,I+1)-m->gcoords(0,I-1)) );
// 				values[1] = -1.0/( (m->gcoords(1,J)-m->gcoords(1,J-1)) 
// 						* 0.5*(m->gcoords(1,J+1)-m->gcoords(1,J-1)) );
// 				values[2] = -1.0/( (m->gcoords(2,K)-m->gcoords(2,K-1)) 
// 						* 0.5*(m->gcoords(2,K+1)-m->gcoords(2,K-1)) );

// 				values[3] =  2.0/(m->gcoords(0,I+1)-m->gcoords(0,I-1))*
// 				  (1.0/(m->gcoords(0,I+1)-m->gcoords(0,I))+1.0/(m->gcoords(0,I)-m->gcoords(0,I-1)));
// 				values[3] += 2.0/(m->gcoords(1,J+1)-m->gcoords(1,J-1))*
// 				  (1.0/(m->gcoords(1,J+1)-m->gcoords(1,J))+1.0/(m->gcoords(1,J)-m->gcoords(1,J-1)));
// 				values[3] += 2.0/(m->gcoords(2,K+1)-m->gcoords(2,K-1))*
// 				  (1.0/(m->gcoords(2,K+1)-m->gcoords(2,K))+1.0/(m->gcoords(2,K)-m->gcoords(2,K-1)));

// 				values[4] = -1.0/( (m->gcoords(0,I+1)-m->gcoords(0,I)) 
// 						* 0.5*(m->gcoords(0,I+1)-m->gcoords(0,I-1)) );
// 				values[5] = -1.0/( (m->gcoords(1,J+1)-m->gcoords(1,J)) 
// 						* 0.5*(m->gcoords(1,J+1)-m->gcoords(1,J-1)) );
// 				values[6] = -1.0/( (m->gcoords(2,K+1)-m->gcoords(2,K)) 
// 						* 0.5*(m->gcoords(2,K+1)-m->gcoords(2,K-1)) );

// 				// // Uniform-grid central difference
// 				// values[0] = values[1] = values[2] = values[4] = values[5] = values[6] = -1.0/(dx*dx);
// 				// values[3] = 6.0/(dx*dx);

// 				MatSetValuesStencil(A, mm, rindices, n, cindices, values, INSERT_VALUES);
// 			}

// 	if(rank == 0) {
// 		printf("Poisson: ComputeLHS: Done.\n");
// 	}
	
// 	return ierr;
// }

// SMat Poisson::computeLHS(const CartMesh *const m) const
// {
// 	const int rank = get_mpi_rank(MPI_COMM_WORLD);

// 	SMat A(m);

// 	if(rank == 0)	
// 		printf("Poisson: ComputeLHS: Setting values of the LHS matrix...\n");

// 	const sreal dx = m->gcoords(0,2)-m->gcoords(0,1);
// 	printf("Poisson: ComputeLHS: dx = %f\n", dx);

// 	for(PetscInt k = A.start; k < A.start + A.sz[2]; k++)
// 		for(PetscInt j = A.start; j < A.start + A.sz[1]; j++)
// 			for(PetscInt i = A.start; i < A.start + A.sz[0]; i++)
// 			{
// 				const sint idx = m->localFlattenedIndexReal(k,j,i);
				
// 				const sint I = i+1, J = j+1, K = k+1;  // 1-offset indices for mesh coords access

// 				A.vals[0][idx] = -1.0/( (m->gcoords(0,I)-m->gcoords(0,I-1)) 
// 				                        * 0.5*(m->gcoords(0,I+1)-m->gcoords(0,I-1)) );
// 				A.vals[1][idx] = -1.0/( (m->gcoords(1,J)-m->gcoords(1,J-1)) 
// 				                        * 0.5*(m->gcoords(1,J+1)-m->gcoords(1,J-1)) );
// 				A.vals[2][idx] = -1.0/( (m->gcoords(2,K)-m->gcoords(2,K-1)) 
// 				                        * 0.5*(m->gcoords(2,K+1)-m->gcoords(2,K-1)) );

// 				A.vals[3][idx] =  2.0/(m->gcoords(0,I+1)-m->gcoords(0,I-1))*
// 					(1.0/(m->gcoords(0,I+1)-m->gcoords(0,I))+1.0/(m->gcoords(0,I)-m->gcoords(0,I-1)));
// 				A.vals[3][idx] += 2.0/(m->gcoords(1,J+1)-m->gcoords(1,J-1))*
// 					(1.0/(m->gcoords(1,J+1)-m->gcoords(1,J))+1.0/(m->gcoords(1,J)-m->gcoords(1,J-1)));
// 				A.vals[3][idx] += 2.0/(m->gcoords(2,K+1)-m->gcoords(2,K-1))*
// 					(1.0/(m->gcoords(2,K+1)-m->gcoords(2,K))+1.0/(m->gcoords(2,K)-m->gcoords(2,K-1)));

// 				A.vals[4][idx] = -1.0/( (m->gcoords(0,I+1)-m->gcoords(0,I)) 
// 				                        * 0.5*(m->gcoords(0,I+1)-m->gcoords(0,I-1)) );
// 				A.vals[5][idx] = -1.0/( (m->gcoords(1,J+1)-m->gcoords(1,J)) 
// 				                        * 0.5*(m->gcoords(1,J+1)-m->gcoords(1,J-1)) );
// 				A.vals[6][idx] = -1.0/( (m->gcoords(2,K+1)-m->gcoords(2,K)) 
// 				                        * 0.5*(m->gcoords(2,K+1)-m->gcoords(2,K-1)) );

// 			}

// 	if(rank == 0) {
// 		printf("Poisson: ComputeLHS: Done.\n");
// 	}
	
// 	return A;
// }

void Poisson::lhsmat_kernel(const CartMesh *const m, const sint i, const sint j, const sint k,
                            const sint nghost,
                            sreal& v0, sreal& v1, sreal& v2, sreal& v3,
                            sreal& v4, sreal& v5, sreal& v6) const
{
	// 1-offset indices for mesh coords access
	const sint I = i + nghost, J = j + nghost, K = k + nghost;

	sreal drp[NDIM], drm[NDIM], drs[NDIM];
	drm[0] = m->gcoords(0,I)-m->gcoords(0,I-1);
	drm[1] = m->gcoords(1,J)-m->gcoords(1,J-1);
	drm[2] = m->gcoords(2,K)-m->gcoords(2,K-1);
	drp[0] = m->gcoords(0,I+1)-m->gcoords(0,I);
	drp[1] = m->gcoords(1,J+1)-m->gcoords(1,J);
	drp[2] = m->gcoords(2,K+1)-m->gcoords(2,K);
	drs[0] = m->gcoords(0,I+1)-m->gcoords(0,I-1);
	drs[1] = m->gcoords(1,J+1)-m->gcoords(1,J-1);
	drs[2] = m->gcoords(2,K+1)-m->gcoords(2,K-1);

	// diffusion
	v0 = -1.0/( drm[0]*0.5*drs[0] );
	v1 = -1.0/( drm[1]*0.5*drs[1] );
	v2 = -1.0/( drm[2]*0.5*drs[2] );

	v3 =  2.0/drs[0]*(1.0/drp[0]+1.0/drm[0]);
	v3 += 2.0/drs[1]*(1.0/drp[1]+1.0/drm[1]);
	v3 += 2.0/drs[2]*(1.0/drp[2]+1.0/drm[2]);

	v4 = -1.0/( drp[0]*0.5*drs[0] );
	v5 = -1.0/( drp[1]*0.5*drs[1] );
	v6 = -1.0/( drp[2]*0.5*drs[2] );

	// v0 *= mu;
	// v1 *= mu;
	// v2 *= mu;
	// v3 *= mu;
	// v4 *= mu;
	// v5 *= mu;
	// v6 *= mu;

	// // Uniform-grid central difference
	// v0 = v1 = v2 = v4 = v5 = v6 = -1.0/(dx*dx);
	// v3 = 6.0/(dx*dx);
}

sreal Poisson::rhs_kernel(const CartMesh *const m, const std::function<sreal(const sreal[NDIM])>& func,
                          const sint i, const sint j, const sint k) const
{
	const sreal crds[NDIM] = {m->gcoords(0,i), m->gcoords(1,j), m->gcoords(2,k)};
	sreal rhs = func(crds);
	// TODO: Add BC
	return rhs;
}

