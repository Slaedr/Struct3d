/** \file
 * \brief PETSc-based finite difference routines for convection-diffusion problem on a Cartesian grid
 * \author Aditya Kashi
 */

#include <cmath>
#include "convdiff.hpp"
#include "common_utils.hpp"

ConvDiff::ConvDiff(const std::array<BCType,6>& bc_types, const std::array<sreal,6>& bc_vals,
                   const std::array<sreal,NDIM> advel, const sreal diffc)
	: PDEBase(bc_types,bc_vals), mu{diffc}, b(advel)
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

void
ConvDiff::lhsmat_kernel(const CartMesh *const m, const sint i, const sint j, const sint k,
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

sreal ConvDiff::rhs_kernel(const CartMesh *const m, const std::function<sreal(const sreal[NDIM])>& func,
                           const sint i, const sint j, const sint k) const
{
	const sreal crds[NDIM] = {m->gcoords(0,i), m->gcoords(1,j), m->gcoords(2,k)};
	sreal rhs = func(crds);
	// TODO: Add BC
	return rhs;
}
