/** \file
 * \brief PETSc-based finite difference routines for convection-diffusion problem on a Cartesian grid
 * \author Aditya Kashi
 */

#include <cmath>
#include "convdiff.hpp"
#include "common_utils.hpp"

ConvDiff::ConvDiff(const std::array<sreal,NDIM> advel, const sreal diffc) : mu{diffc}, b(advel)
{
	const int rank = get_mpi_rank(PETSC_COMM_WORLD);
	if(rank == 0) {
		printf("ConvDiff: Using b = (%f,%f,%f), mu = %f.\n", b[0], b[1], b[2], mu);
	}
}

std::array<std::function<sreal(const sreal[NDIM])>,2> ConvDiff::manufactured_solution() const
{
	const sreal munum = mu;
	const std::array<sreal,NDIM> advec = b;
	std::array<std::function<sreal(const sreal[NDIM])>,2> soln;

	soln[0] = [](const sreal r[NDIM]) { return sin(2*PI*r[0])*sin(2*PI*r[1])*sin(2*PI*r[2]); };

	soln[1] = [munum,advec](const sreal r[NDIM]) {
		sreal retval = munum*12*PI*PI*sin(2*PI*r[0])*sin(2*PI*r[1])*sin(2*PI*r[2]);
		for(int i = 0; i < NDIM; i++)
		{
			sreal term = 2*PI*advec[i];
			for(int j = 0; j < NDIM; j++)
				term *= (i==j) ? cos(2*PI*r[j]) : sin(2*PI*r[j]);
			retval += term;
		}
		return retval;
	};

	return soln;
}

/// Set stiffness matrix corresponding to interior points
/** Inserts entries rowwise into the matrix.
 */
int ConvDiff::computeLHSPetsc(const CartMesh *const m, DM da, Mat A) const
{
	PetscErrorCode ierr = 0;
	const int rank = get_mpi_rank(PETSC_COMM_WORLD);
	if(rank == 0)	
		printf("ConvDiff: ComputeLHS: Setting values of the LHS matrix...\n");

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

				assembly_kernel(m, i,j,k, 1, values[0], values[1], values[2], values[3], values[4],
				                values[5], values[6]);

				MatSetValuesStencil(A, mm, rindices, n, cindices, values, INSERT_VALUES);
			}

	if(rank == 0)
		printf("ConvDiff: ComputeLHS: Done.\n");
	
	return ierr;
}

SMat ConvDiff::computeLHS(const CartMesh *const m) const
{
	const int rank = get_mpi_rank(MPI_COMM_WORLD);

	SMat A(m);

	if(rank == 0)	
		printf("ConvDiff: ComputeLHS: Setting values of the LHS matrix...\n");

	for(PetscInt k = A.start; k < A.start+A.sz[2]; k++)
		for(PetscInt j = A.start; j < A.start+A.sz[1]; j++)
			for(PetscInt i = A.start; i < A.start+A.sz[0]; i++)
			{
				const sint idx = m->localFlattenedIndexReal(k,j,i);

				assembly_kernel(m, i,j,k, A.nghost, A.vals[0][idx], A.vals[1][idx], A.vals[2][idx],
				                A.vals[3][idx], A.vals[4][idx], A.vals[5][idx], A.vals[6][idx]);
			}

	if(rank == 0)
		printf("ConvDiff: ComputeLHS: Done.\n");
	
	return A;
}

inline void
ConvDiff::assembly_kernel(const CartMesh *const m, const sint i, const sint j, const sint k,
                          const sint nghost,
                          sreal& v0, sreal& v1, sreal& v2, sreal& v3,
                          sreal& v4, sreal& v5, sreal& v6) const
{
	// 1-offset indices for mesh coords access
	const sint I = i + nghost, J = j + nghost, K = k + nghost;

	sreal drp[NDIM];
	drp[0] = m->gcoords(0,I)-m->gcoords(0,I-1);
	drp[1] = m->gcoords(1,J)-m->gcoords(1,J-1);
	drp[2] = m->gcoords(2,K)-m->gcoords(2,K-1);

	// diffusion
	v0 = -1.0/( (m->gcoords(0,I)-m->gcoords(0,I-1)) 
	            * 0.5*(m->gcoords(0,I+1)-m->gcoords(0,I-1)) );
	v1 = -1.0/( (m->gcoords(1,J)-m->gcoords(1,J-1)) 
	            * 0.5*(m->gcoords(1,J+1)-m->gcoords(1,J-1)) );
	v2 = -1.0/( (m->gcoords(2,K)-m->gcoords(2,K-1)) 
	            * 0.5*(m->gcoords(2,K+1)-m->gcoords(2,K-1)) );

	v3 =  2.0/(m->gcoords(0,I+1)-m->gcoords(0,I-1))*
		(1.0/(m->gcoords(0,I+1)-m->gcoords(0,I))+1.0/(m->gcoords(0,I)-m->gcoords(0,I-1)));
	v3 += 2.0/(m->gcoords(1,J+1)-m->gcoords(1,J-1))*
		(1.0/(m->gcoords(1,J+1)-m->gcoords(1,J))+1.0/(m->gcoords(1,J)-m->gcoords(1,J-1)));
	v3 += 2.0/(m->gcoords(2,K+1)-m->gcoords(2,K-1))*
		(1.0/(m->gcoords(2,K+1)-m->gcoords(2,K))+1.0/(m->gcoords(2,K)-m->gcoords(2,K-1)));

	v4 = -1.0/( (m->gcoords(0,I+1)-m->gcoords(0,I)) 
	            * 0.5*(m->gcoords(0,I+1)-m->gcoords(0,I-1)) );
	v5 = -1.0/( (m->gcoords(1,J+1)-m->gcoords(1,J)) 
	            * 0.5*(m->gcoords(1,J+1)-m->gcoords(1,J-1)) );
	v6 = -1.0/( (m->gcoords(2,K+1)-m->gcoords(2,K)) 
	            * 0.5*(m->gcoords(2,K+1)-m->gcoords(2,K-1)) );

	v0 *= mu;
	v1 *= mu;
	v2 *= mu;
	v3 *= mu;
	v4 *= mu;
	v5 *= mu;
	v6 *= mu;

	// upwind advection
	v0 += -b[0]/drp[0];
	v1 += -b[1]/drp[1];
	v2 += -b[2]/drp[2];
	v3 += b[0]/drp[0] + b[1]/drp[1] + b[2]/drp[2];
}
