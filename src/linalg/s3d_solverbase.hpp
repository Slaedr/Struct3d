/** \file
 * \brief Base class for solvers
 */

#ifndef STRUCT3D_SOLVER_BASE_H
#define STRUCT3D_SOLVER_BASE_H

#include "cartmesh.hpp"

/// Abstract class for cartesian-mesh dependent solvers
class SolverBase
{
public:
	/// Set the mesh on which the problem is posed
	SolverBase(const CartMesh& mesh) : m(mesh) { }

	virtual void apply(const sreal ***const b, sreal ***const x) const = 0;

protected:
	const CartMesh& m;
};

#endif
