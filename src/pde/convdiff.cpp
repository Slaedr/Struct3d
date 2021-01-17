/** \file
 * \brief PETSc-based finite difference routines for convection-diffusion problem on a Cartesian grid
 * \author Aditya Kashi
 */

#include <cmath>
#include "convdiff.hpp"
#include "common_utils.hpp"

ConvDiff::ConvDiff(const std::array<BCType,6>& bc_types, const std::array<sreal,6>& bc_vals,
                   const std::array<sreal,NDIM> advel, const sreal diffc)
	: PDEImpl<ConvDiff>(bc_types,bc_vals), mu{diffc}, b(advel)
{
	const int rank = get_mpi_rank(MPI_COMM_WORLD);
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

std::function<sreal(const sreal[NDIM])> ConvDiff::test_rhs() const
{
	return [](const sreal r[NDIM]) { return sin(PI*r[0])*sin(PI*r[1])*sin(PI*r[2]); };
}

//#pragma omp declare simd uniform(this,m,nghost,j,k) linear(i:1) notinbranch
void ConvDiff::lhsmat_kernel(const CartMesh *const m, sint i, const sint j, const sint k,
                             const sint nghost, sreal *const __restrict v) const
{
	// 1-offset indices for mesh coords access
	//const sint I = i + nghost, J = j + nghost, K = k + nghost;
	const sint I = i, J = j, K = k;

	const sreal drm[NDIM] = { m->gcoords(0,I)-m->gcoords(0,I-1),
	                          m->gcoords(1,J)-m->gcoords(1,J-1),
	                          m->gcoords(2,K)-m->gcoords(2,K-1) };
	const sreal drp[NDIM] = { m->gcoords(0,I+1)-m->gcoords(0,I),
	                          m->gcoords(1,J+1)-m->gcoords(1,J),
	                          m->gcoords(2,K+1)-m->gcoords(2,K) };
	const sreal drs[NDIM] = { m->gcoords(0,I+1)-m->gcoords(0,I-1),
	                          m->gcoords(1,J+1)-m->gcoords(1,J-1),
	                          m->gcoords(2,K+1)-m->gcoords(2,K-1) };

	// diffusion
	v[3] = 0;
	for(int j = 0; j < NDIM; j++)
	{
		v[j] = -1.0/( drm[j]*0.5*drs[j] );            // lower
		v[3] += 2.0/drs[j]*(1.0/drp[j]+1.0/drm[j]);   // diagonal
		v[4+j] = -1.0/( drp[j]*0.5*drs[j] );          // upper
	}

	for(int j = 0; j < NSTENCIL; j++)
		v[j] *= mu;

	// upwind advection
	for(int j = 0; j < NDIM; j++) {
		if(b[j] > 0) {
			v[j] -= b[j]/drm[j];
			v[3] += b[j]/drm[j];
		}
		else {
			v[j+4] += b[j]/drp[j];
			v[3] -= b[j]/drp[j];
		}
	}
}

sreal ConvDiff::rhs_kernel(const CartMesh *const m, const std::function<sreal(const sreal[NDIM])>& func,
                           const sint i, const sint j, const sint k) const
{
	const sreal crds[NDIM] = {m->gcoords(0,i), m->gcoords(1,j), m->gcoords(2,k)};
	sreal rhs = func(crds);

	/*const sreal drm[NDIM] = { m->gcoords(0,i)-m->gcoords(0,i-1),
	                          m->gcoords(1,j)-m->gcoords(1,j-1),
	                          m->gcoords(2,k)-m->gcoords(2,k-1) };
	const sreal drp[NDIM] = { m->gcoords(0,i+1)-m->gcoords(0,i),
	                          m->gcoords(1,j+1)-m->gcoords(1,j),
	                          m->gcoords(2,k+1)-m->gcoords(2,k) };
	const sreal drs[NDIM] = { m->gcoords(0,i+1)-m->gcoords(0,i-1),
	                          m->gcoords(1,j+1)-m->gcoords(1,j-1),
	                          m->gcoords(2,k+1)-m->gcoords(2,k-1) };
	// Add BCs
	if(i == m->gnghost()) {
		if(bctypes[0] == S3D_DIRICHLET) {
			rhs += bvals[0]*(b[0]/drm[0] + 1.0/(drm[0]*0.5*drs[0]));
		}
		else {
			printf("! Invalid BC!\n");
		}
	}
	else if(i == m->gnpoind(0)-m->gnghost()-1) {
		if(bctypes[0] == S3D_DIRICHLET) {
			rhs += bvals[1]/(drp[0]*0.5*drs[0]);
		}
		else {
			printf("! Invalid BC!\n");
		}
		}*/
	return rhs;
}

template class PDEImpl<ConvDiff>;
