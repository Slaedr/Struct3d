/** \file
 * \brief PETSc-based finite difference routines for convection-diffusion problem on a Cartesian grid
 * \author Aditya Kashi
 */

#include <cmath>
#include "convdiff_circular.hpp"
#include "common_utils.hpp"

ConvDiffCirc::ConvDiffCirc(const std::array<BCType,6>& bc_types, const std::array<sreal,6>& bc_vals,
                           const sreal advel, const sreal diffc)
	: PDEBase(bc_types,bc_vals), mu{diffc}, bmag(advel)
{
	const int rank = get_mpi_rank(PETSC_COMM_WORLD);
	if(rank == 0) {
		printf("ConvDiff: Using |b| = %f, mu = %f.\n", bmag, mu);
	}
}

inline std::array<sreal,NDIM> ConvDiffCirc::advectionVel(const sreal r[NDIM]) const
{
	std::array<sreal,NDIM> v;
	v[0] = 2*r[1]*(1.0-r[0]*r[0]);
	v[1] = -2*r[0]*(1.0-r[1]*r[1]);
	v[2] = std::sin(PI*r[2]);
	return v;
}

std::array<std::function<sreal(const sreal[NDIM])>,2> ConvDiffCirc::manufactured_solution() const
{
	const sreal munum = mu;
	const sreal bmagnum = bmag;
	std::array<std::function<sreal(const sreal[NDIM])>,2> soln;

	soln[0] = [](const sreal r[NDIM]) { return sin(2*PI*r[0])*sin(2*PI*r[1])*sin(2*PI*r[2]); };

	soln[1] = [this,munum,bmagnum](const sreal r[NDIM]) {
		sreal retval = munum*12*PI*PI*sin(2*PI*r[0])*sin(2*PI*r[1])*sin(2*PI*r[2]);
		const std::array<sreal,NDIM> vel = this->advectionVel(r);
		for(int i = 0; i < NDIM; i++)
		{
			sreal term = 2*PI*bmagnum*vel[i];
			for(int j = 0; j < NDIM; j++)
				term *= (i==j) ? cos(2*PI*r[j]) : sin(2*PI*r[j]);
			retval += term;
		}
		return retval;
	};

	return soln;
}

inline void
ConvDiffCirc::lhsmat_kernel(const CartMesh *const m, const sint i, const sint j, const sint k,
                            const sint nghost,
                            sreal& v0, sreal& v1, sreal& v2, sreal& v3,
                            sreal& v4, sreal& v5, sreal& v6) const
{
	// 1-offset indices for mesh coords access
	const sint I = i + nghost, J = j + nghost, K = k + nghost;

	sreal drm[NDIM], drp[NDIM], drs[NDIM];
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

	v0 *= mu;
	v1 *= mu;
	v2 *= mu;
	v3 *= mu;
	v4 *= mu;
	v5 *= mu;
	v6 *= mu;

	const sreal r[NDIM] = {m->gcoords(0,I), m->gcoords(1,J), m->gcoords(2,K)};
	const std::array<sreal,NDIM> b = advectionVel(r);

	// upwind advection
	v0 += -b[0]/drp[0];
	v1 += -b[1]/drp[1];
	v2 += -b[2]/drp[2];
	v3 += b[0]/drp[0] + b[1]/drp[1] + b[2]/drp[2];
}

sreal ConvDiffCirc::rhs_kernel(const CartMesh *const m,
                               const std::function<sreal(const sreal[NDIM])>& func,
                               const sint i, const sint j, const sint k) const
{
	const sreal crds[NDIM] = {m->gcoords(0,i), m->gcoords(1,j), m->gcoords(2,k)};
	sreal rhs = func(crds);
	// TODO: Add BC
	return rhs;
}
