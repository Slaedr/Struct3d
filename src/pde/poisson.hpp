/** \file
 * \brief PETSc-based finite difference routines for Poisson Dirichlet problem on a Cartesian grid
 * \author Aditya Kashi
 *
 * Note that only zero Dirichlet BCs are currently supported.
 */

#ifndef STRUCT3D_POISSON_H
#define STRUCT3D_POISSON_H

#include <petscmat.h>
#include "cartmesh.hpp"
#include "pdebase.hpp"

/// Assembles LHS matrix and RHS vector for Poisson eqn: - div grad u = f
class Poisson : public PDEBase
{
public:
	Poisson(const std::array<BCType,6>& bc_types, const std::array<sreal,6>& bc_vals);

	/// Returns a pair of functions: the first being the solution and the second the right hand side
	/** Set RHS = 12*pi^2*sin(2pi*x)sin(2pi*y)sin(2pi*z) for u_exact = sin(2pi*x)sin(2pi*y)sin(2pi*z)
	 */
	std::array<std::function<sreal(const sreal[NDIM])>,2> manufactured_solution() const;

protected:
	/// Kernel used for assembling the matrix
	void lhsmat_kernel(const CartMesh *const m, const sint i, const sint j, const sint k,
	                   const sint nghost,
	                   sreal& v0, sreal& v1, sreal& v2, sreal& v3, sreal& v4, sreal& v5, sreal& v6)
		const;

	/// Kernel used for assembling the right hand side vector using source and BCs
	sreal rhs_kernel(const CartMesh *const m, const std::function<sreal(const sreal[NDIM])>& sourcef,
	                 const sint i, const sint j, const sint k) const;
};

#endif
