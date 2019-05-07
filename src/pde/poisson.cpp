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

std::function<sreal(const sreal[NDIM])> Poisson::test_rhs() const
{
	return [](const sreal r[NDIM]) { return sin(PI*r[0])*sin(PI*r[1])*sin(PI*r[2]); };
}

void Poisson::lhsmat_kernel(const CartMesh *const m, const sint i, const sint j, const sint k,
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
}

sreal Poisson::rhs_kernel(const CartMesh *const m, const std::function<sreal(const sreal[NDIM])>& func,
                          const sint i, const sint j, const sint k) const
{
	const sreal crds[NDIM] = {m->gcoords(0,i), m->gcoords(1,j), m->gcoords(2,k)};
	sreal rhs = func(crds);

	// Add BCs
	const sreal drm[NDIM] = { m->gcoords(0,i)-m->gcoords(0,i-1),
	                          m->gcoords(1,j)-m->gcoords(1,j-1),
	                          m->gcoords(2,k)-m->gcoords(2,k-1) };
	const sreal drp[NDIM] = { m->gcoords(0,i+1)-m->gcoords(0,i),
	                          m->gcoords(1,j+1)-m->gcoords(1,j),
	                          m->gcoords(2,k+1)-m->gcoords(2,k) };
	const sreal drs[NDIM] = { m->gcoords(0,i+1)-m->gcoords(0,i-1),
	                          m->gcoords(1,j+1)-m->gcoords(1,j-1),
	                          m->gcoords(2,k+1)-m->gcoords(2,k-1) };
	if(i == m->gnghost()) {
		if(bctypes[0] == S3D_DIRICHLET) {
			rhs += bvals[0]/(drm[0]*0.5*drs[0]);
		}
		else {
			throw std::runtime_error("Invalid BC type at i-!"); 
		}
	}
	if(i == m->gnpoind(0)-1-m->gnghost()) {
		if(bctypes[1] == S3D_DIRICHLET) {
			rhs += bvals[1]/(drp[0]*0.5*drs[0]);
		}
		else {
			throw std::runtime_error("Invalid BC type at i+!"); 
		}
	}

	if(j == m->gnghost()) {
		if(bctypes[2] == S3D_DIRICHLET) {
			rhs += bvals[2]/(drm[1]*0.5*drs[1]);
		}
		else {
			throw std::runtime_error("Invalid BC type at j-!"); 
		}
	}
	if(j == m->gnpoind(1)-1-m->gnghost()) {
		if(bctypes[3] == S3D_DIRICHLET) {
			rhs += bvals[3]/(drp[1]*0.5*drs[1]);
		}
		else {
			throw std::runtime_error("Invalid BC type at j+!"); 
		}
	}

	if(k == m->gnghost()) {
		if(bctypes[4] == S3D_DIRICHLET) {
			rhs += bvals[4]/(drm[2]*0.5*drs[2]);
		}
		else {
			throw std::runtime_error("Invalid BC type at k-!"); 
		}
	}
	if(k == m->gnpoind(2)-1-m->gnghost()) {
		if(bctypes[5] == S3D_DIRICHLET) {
			rhs += bvals[5]/(drp[2]*0.5*drs[2]);
		}
		else {
			throw std::runtime_error("Invalid BC type at k+!"); 
		}
	}

	return rhs;
}

