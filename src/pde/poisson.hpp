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
	Poisson() { }

	/// Set stiffness matrix corresponding to interior points
	/** Inserts entries rowwise into the matrix.
	 */
	PetscErrorCode computeLHS(const CartMesh *const m, DM da, Mat A) const;

	/// Returns a pair of functions: the first being the solution and the second the right hand side
	/** Set RHS = 12*pi^2*sin(2pi*x)sin(2pi*y)sin(2pi*z) for u_exact = sin(2pi*x)sin(2pi*y)sin(2pi*z)
	 */
	std::array<std::function<sreal(const sreal[NDIM])>,2> manufactured_solution() const;
};

#endif
